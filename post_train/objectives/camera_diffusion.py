"""Camera-controlled causal diffusion model (Stage 1 AR).

Inherits from CausalDiffusion and only overrides:
1. Model initialization (WanDiffusionWrapper with use_camera=True)
2. generator_loss to pass viewmats/Ks to the generator
"""

from typing import Optional, Tuple
import torch

from post_train.objectives.diffusion import CausalDiffusion
from models.wan_wrapper import WanDiffusionWrapper


def _frustum_overlap(viewmats, Ks, query, candidates, depth=3.0, grid=5):
    """How much of the query frame's view each candidate frame already saw.

    Poses alone do not answer this: two cameras can sit together and face
    apart, or sit far apart and share a wall. So a grid of rays is pushed
    through the query frustum to a working depth and the resulting points are
    projected into each candidate; the score is the fraction that land inside
    its image and in front of it.

    Image bounds come from the principal point rather than a passed-in size, so
    this is agnostic to whether the intrinsics are in pixels or latents.
    """
    vm = viewmats[0].float()
    K = (Ks[0, query].float() if Ks is not None
         else torch.eye(3, device=vm.device, dtype=vm.dtype))
    w, h = K[0, 2] * 2, K[1, 2] * 2
    ys, xs = torch.meshgrid(
        torch.linspace(0.1, 0.9, grid, device=vm.device) * h,
        torch.linspace(0.1, 0.9, grid, device=vm.device) * w, indexing="ij")
    pix = torch.stack([xs.reshape(-1), ys.reshape(-1),
                       torch.ones(grid * grid, device=vm.device)], 0)
    dirs = torch.linalg.inv(K) @ pix
    dirs = dirs / dirs.norm(dim=0, keepdim=True)

    c2w_q = torch.linalg.inv(vm[query])
    pts_w = c2w_q[:3, :3] @ (dirs * depth) + c2w_q[:3, 3:4]

    out = {}
    for c in candidates:
        pc = vm[c][:3, :3] @ pts_w + vm[c][:3, 3:4]
        uv = (Ks[0, c].float() if Ks is not None else K) @ pc
        z = pc[2]
        uv = uv[:2] / uv[2:3].clamp_min(1e-6)
        inside = ((uv[0] >= 0) & (uv[0] < w) & (uv[1] >= 0) & (uv[1] < h)
                  & (z > 1e-3))
        out[c] = float(inside.float().mean())
    return out


def _detail_energy(x0: torch.Tensor):
    """Fine and coarse gradient energy, in log space.

    An earlier version of this returned only the fine/coarse ratio, on the
    reasoning that a ratio cannot be paid off with grain the way an energy can.
    Measured on the actual rollouts that was blind to the failure it existed to
    catch: across a 95-frame clip the ratio moves 0.119 at worst, while both
    energies fall together -- fine -55%, coarse -60% on the damaged run against
    -19%/-23% on its parent. The tail does not go selectively soft, it loses
    amplitude at every scale at once, and a scale-free measure cannot see that
    by construction.

    So both are returned and the caller constrains both. The three failure
    modes separate cleanly: grain lifts fine alone, blur drops fine alone,
    collapse drops both.
    """
    field = x0.float().mean(dim=2)

    def energy(step: int) -> torch.Tensor:
        gx = field[..., :, step:] - field[..., :, :-step]
        gy = field[..., step:, :] - field[..., :-step, :]
        return (gx.pow(2).mean(dim=(-2, -1))
                + gy.pow(2).mean(dim=(-2, -1))).clamp_min(1.0e-8).log()

    return energy(1), energy(2)


def _detail_collapse_loss(deep_x0: torch.Tensor,
                          clean_latent: torch.Tensor) -> torch.Tensor:
    """Penalise the deep block for losing detail the clip's own frames hold.

    One-sided on amplitude. The camera has travelled somewhere else by the time
    this block is generated -- open ground and large smooth surfaces where the
    clip was a corridor of petals and stonework -- so energy above the clip's
    level is a legitimate content change and only the shortfall is charged.
    Even the healthy parent run gives up 19% by its tail, so a two-sided target
    would spend most of its gradient fighting the scene.

    The amplitude is measured after a 2x box downsample. Measured on the raw
    field it was buyable: a 0.4x collapse costs 6.72, and the same collapse
    with grain at 0.2 costs 0.43, because white noise lifts both energies back
    over the threshold. Downsampling drops white-noise power about fourfold
    while structure survives it, so grain no longer pays for the shortfall.

    The ratio term is two-sided and carries real weight. It is the part that
    reads texture rather than amount -- grain pushes fine energy toward coarse,
    blur pulls it away -- and grain is how every sharpness term in this lineage
    has been paid off so far.
    """
    def smooth(x):
        b, f = x.shape[:2]
        return torch.nn.functional.avg_pool2d(
            x.float().mean(dim=2).flatten(0, 1), 2).unflatten(0, (b, f))[:, :, None]

    gt_lo_f, gt_lo_c = (e.mean(dim=1).detach()
                        for e in _detail_energy(smooth(clean_latent)))
    lo_f, lo_c = (e.mean(dim=1) for e in _detail_energy(smooth(deep_x0)))
    collapse = (torch.relu(gt_lo_f - lo_f).pow(2)
                + torch.relu(gt_lo_c - lo_c).pow(2)).mean()

    gt_f, gt_c = (e.mean(dim=1).detach() for e in _detail_energy(clean_latent))
    f_, c_ = (e.mean(dim=1) for e in _detail_energy(deep_x0))
    texture = ((f_ - c_) - (gt_f - gt_c)).pow(2).mean()
    return collapse + 4.0 * texture


def _global_motion_split(x0: torch.Tensor, ridge: float = 1.0e-1):
    """Split a block's motion into a global camera part and the rest.

    Returns ``(params, residual)``: the fitted ``(pan, tilt, radial)`` vector,
    and the temporal delta with everything that vector explains taken out. The
    first is what a camera move does to the whole frame; the second is content
    that moves on its own -- falling petals, swaying branches -- which no
    global motion can account for.

    Two terms read this. The camera anchor takes the direction of ``params``.
    The scene term takes ``residual`` and asks it to match the real video's,
    which is the part that stops moving when a rollout goes static.

    Below, on the camera half:

    Direction of the global camera motion implied by a block of latents.

    Measured, not assumed: with the global affine fitted out of the optical
    flow and subtracted, the base and both adapters carry the same residual
    scene motion (23-25% of the field -- petals still fall under the adapter).
    What the adapter changes is the global component's *direction*: radial over
    horizontal runs 4.34 on the base and 0.07 at step 575, while the global
    magnitude does not fall (3.40 -> 3.91). The adapter does not spend less
    motion, it spends it sideways.

    So this returns direction only. Under the brightness-constancy relation
    ``dI/dt = -(grad I . v)``, a global flow ``v = (a + s*x, b + s*y)`` -- pan,
    tilt and radial expansion, the dolly/sweep distinction -- makes the
    temporal delta linear in ``(a, b, s)``, and those three fall out of one
    3x3 least squares over the block. Latents are not images and the relation
    is approximate here; it does not need to be exact, because only the
    direction of the resulting vector is ever used.

    Pooled over the block rather than per frame: one trajectory per sample is
    what the anchor constrains, and pooling keeps the normal equations well
    conditioned when a single frame pair is nearly static.
    """
    field = x0.float().mean(dim=2)
    delta_t = field[:, 1:] - field[:, :-1]
    reference = field[:, :-1]
    # Central differences, not forward. A forward difference reports the
    # gradient half a pixel off the point whose temporal delta it is paired
    # with, and that bias leaks into the radial column specifically: on
    # synthetic pure zoom the forward-difference version recovered
    # (-0.33, -0.19, +0.92) instead of (0, 0, 1), which left a pure sweep and a
    # pure dolly only 0.33 apart in cosine rather than orthogonal. The anchor
    # still discriminated, but over a needlessly compressed range.
    grad_x = (reference[..., 1:-1, 2:] - reference[..., 1:-1, :-2]) * 0.5
    grad_y = (reference[..., 2:, 1:-1] - reference[..., :-2, 1:-1]) * 0.5
    delta_t = delta_t[..., 1:-1, 1:-1]

    height, width = grad_x.shape[-2:]
    ys = torch.linspace(-1.0, 1.0, height, device=x0.device,
                        dtype=grad_x.dtype).view(1, 1, height, 1)
    xs = torch.linspace(-1.0, 1.0, width, device=x0.device,
                        dtype=grad_x.dtype).view(1, 1, 1, width)
    radial = xs * grad_x + ys * grad_y
    design = torch.stack([grad_x, grad_y, radial], dim=-1).flatten(1, 3)
    target = (-delta_t).flatten(1).unsqueeze(-1)

    normal = design.transpose(1, 2) @ design
    rhs = design.transpose(1, 2) @ target
    # Ridge relative to the system's own scale: absolute damping would dominate
    # a low-contrast block and vanish on a high-contrast one.
    scale = normal.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1.0e-8)
    eye = torch.eye(3, device=x0.device, dtype=normal.dtype)
    normal = normal + ridge * scale.view(-1, 1, 1) * eye
    # Detached: gradient reaches x0 through the right-hand side only.
    #
    # When the two directions nearly agree -- which is the normal case, the
    # 20-step probe measured 0.0006 at block 1 -- the cosine's own gradient is
    # tiny while the Jacobian of the solve is not, and differentiating through
    # the matrix put grad_norm at mean 5.97 against ctx20's 0.56, with a peak of
    # 55 clipped down to 10. That is noise entering the update in place of
    # signal, from a term whose loss was already near zero. Treating the normal
    # matrix as a constant whitener keeps the descriptor and drops that path.
    params = torch.linalg.solve(normal.detach(), rhs).squeeze(-1)
    # What the global fit accounts for, put back in the field's own shape, and
    # subtracted. Gradient flows: the residual is what the scene term supervises
    # and it must be able to change x0.
    explained = (design @ params.unsqueeze(-1)).squeeze(-1)
    residual = (-delta_t).flatten(1) - explained
    return params, residual.view_as(delta_t)


def _global_motion_direction(x0: torch.Tensor,
                             ridge: float = 1.0e-1) -> torch.Tensor:
    """Just the camera half, for callers that do not need the residual."""
    return _global_motion_split(x0, ridge)[0]


class CameraCausalDiffusion(CausalDiffusion):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.error_buffer = None
        self.noise_error_buffer = None
        # Defaults for the whole er_* family. generator_loss reads
        # er_boundary_weight and er_luma_weight unconditionally, but they were
        # only ever assigned inside the error-recycling branch below, so any
        # camera_diffusion config without an `error_recycling` block died at
        # the first backward with AttributeError -- including
        # wan22_stage1_ar_camera_control2v.yaml, the plain Control2V SFT
        # template. Zero weight means the corresponding term contributes
        # nothing, which is what "no error recycling configured" should mean.
        self.er_context_prob = 0.8
        self.er_latent_prob = 0.0
        self.er_noise_prob = 0.0
        self.er_clean_prob = 0.2
        self.er_skip_first = True
        self.er_boundary_weight = 0.0
        self.er_luma_weight = 0.0
        cfg = getattr(args, "error_recycling", None)
        if cfg is not None and bool(getattr(cfg, "enabled", False)):
            from post_train.objectives.error_recycling import PositionErrorBuffer
            blocks = int(args.image_or_video_shape[1]) // int(
                self.num_frame_per_block)
            self.error_buffer = PositionErrorBuffer(
                blocks,
                num_buckets=int(getattr(cfg, "num_buckets", 50)),
                size_per_bucket=int(getattr(cfg, "buffer_size_per_bucket", 32)),
                num_timesteps=self.scheduler.num_train_timesteps,
                modulation=float(getattr(cfg, "modulate_factor", 0.3)),
            )
            self.noise_error_buffer = PositionErrorBuffer(
                blocks,
                num_buckets=int(getattr(cfg, "num_buckets", 50)),
                size_per_bucket=int(getattr(cfg, "buffer_size_per_bucket", 32)),
                num_timesteps=self.scheduler.num_train_timesteps,
                modulation=float(getattr(cfg, "modulate_factor", 0.3)),
            )
            self.er_context_prob = float(getattr(cfg, "context_inject_prob", 0.8))
            self.er_latent_prob = float(getattr(cfg, "latent_inject_prob", 0.0))
            self.er_noise_prob = float(getattr(cfg, "noise_inject_prob", 0.0))
            self.er_clean_prob = float(getattr(cfg, "clean_prob", 0.2))
            self.er_skip_first = bool(getattr(cfg, "skip_first_block", True))
            self.er_boundary_weight = float(getattr(cfg, "boundary_loss_weight", 0.0))
            self.er_luma_weight = float(getattr(cfg, "luma_loss_weight", 0.0))

    def _initialize_models(self, args, device):
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True
        )
        assert self.generator.use_camera is True
        self.generator.set_trainable()

        from models.wan_wrapper import build_text_encoder, build_vae
        # Variant-aware: 2.1's bespoke 16ch VAE, or 2.2's AutoencoderKLWan.
        self.text_encoder = build_text_encoder(args)
        self.vae = build_vae(args)
        # Event prompt encoder. The umT5 encoder is frozen and identical in
        # both roles, so by default the event text goes through the *same*
        # instance -- a second copy would cost ~5 GB to compute exactly the
        # same embeddings. What makes the channel independent is the separate
        # cross-attention in the DiT and the separate key in the conditional
        # dict, not a separate copy of the encoder weights. Set
        # `separate_event_encoder: true` to instantiate a distinct one anyway,
        # which is what you want if the event encoder is ever unfrozen.
        if not bool(getattr(args, "use_event", False)):
            self.event_encoder = None
        elif bool(getattr(args, "separate_event_encoder", False)):
            self.event_encoder = build_text_encoder(args)
        else:
            self.event_encoder = self.text_encoder

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

        # Optional bidirectional restore teacher for on-policy scheduled
        # sampling. Unlike the old precausal teacher-forcing path, this teacher
        # supervises a target block only after the student has populated its
        # causal cache with its own four-step samples.
        self.rollout_teacher = None
        teacher_path = getattr(args, "scheduled_teacher_ckpt", None)
        if teacher_path:
            self.rollout_teacher = WanDiffusionWrapper(
                **getattr(args, "model_kwargs", {}), is_causal=False)
            self.rollout_teacher.model.requires_grad_(False)
            print(f"Loading scheduled-sampling teacher from {teacher_path}")
            state = torch.load(
                teacher_path, map_location="cpu", mmap=True)
            for key in ("generator", "model", "generator_ema"):
                if key in state:
                    state = state[key]
                    break
            state = {
                key.replace("model._fsdp_wrapped_module.", "model.", 1): value
                for key, value in state.items()
            }
            # Merged restore checkpoints store LoRA bases as X.weight, while
            # this wrapper has X.base.weight modules installed.
            remapped = {}
            for key, value in state.items():
                mapped = key
                if ".base." not in mapped and any(
                        f".{target}." in mapped
                        for target in WanDiffusionWrapper.LORA_TARGETS):
                    if mapped.endswith(".weight"):
                        mapped = mapped[:-7] + ".base.weight"
                    elif mapped.endswith(".bias"):
                        mapped = mapped[:-5] + ".base.bias"
                remapped[mapped] = value
            missing, unexpected = self.rollout_teacher.load_state_dict(
                remapped, strict=False)
            # The restore teacher was trained with ``use_action``; when the
            # student is action-free the teacher's action embedder is the only
            # structural difference and carries nothing the direction/identity
            # objectives need, so drop those keys rather than fail the load.
            unexpected = [k for k in unexpected
                          if "action_embedder" not in k]
            if unexpected:
                raise RuntimeError(
                    f"scheduled teacher has unexpected keys: "
                    f"{unexpected[:5]}")
            print(f"Scheduled teacher: {len(remapped)} tensors loaded, "
                  f"{len(missing)} pretrained tensors unchanged")
            self.rollout_teacher.to(device=device, dtype=torch.bfloat16)
            self.rollout_teacher.eval()

    def depth_spectrum_loss(self, conditional_dict, clean_latent,
                            unconditional_dict=None, viewmats=None, Ks=None):
        """Charge a loss outside the training clip, in its own backward pass.

        Measured per segment, the damage this lineage does is all in the tail:
        over 320 steps the opening held (sharpness 1060 -> 954) while the tail
        halved (867 -> 440). No reweighting of the existing terms can reach
        that, because every one of them lives inside the 20 latent frames the
        clips carry, and inference runs 95.

        This rolls past the end of the data and supervises there -- LongLive's
        streaming long tuning reduced to the part that matters here. It is not
        their DMD, which needs a teacher score model and a critic this trainer
        does not build; the target is :func:`_detail_spectrum`, which the
        ground-truth clip supplies and a natural video holds roughly constant
        over time.

        It runs as a separate forward and backward for a reason. Sharing the
        main pass's caches does not work: the backbone gradient-checkpoints, so
        its blocks are re-run during backward, and a rollout that writes into
        the same kv/PRoPE caches while that graph is still pending leaves the
        recomputation reading a cache that has moved on -- "cache holds 8
        frames but prope_window_viewmats has 12". Running after the main
        backward also means the two peaks do not add, which is what brings this
        under a 44 GiB card that the main pass already fills to 43.1.
        """
        depth_blocks = int(getattr(self.args, "depth_rollout_blocks", 0))
        weight = float(getattr(self.args, "depth_spectrum_weight", 0.0))
        if depth_blocks <= 0 or weight <= 0.0 or viewmats is None:
            return None

        from models.fm_solvers import retrieve_timesteps
        from models.fm_solvers_unipc import FlowUniPCMultistepScheduler

        # The main pass's caches are freed by the backward that just returned,
        # but the allocator holds their blocks: at the OOM it was sitting on
        # 1.04 GiB reserved and unallocated while refusing a 28 MiB request.
        # This pass needs four caches of its own -- CFG doubles them -- so that
        # reservation has to go back before they are built.
        torch.cuda.empty_cache()

        batch_size, num_frames = clean_latent.shape[:2]
        block = int(self.num_frame_per_block)
        steps = int(getattr(self.args, "depth_rollout_steps", 2))

        model = self.generator
        while hasattr(model, "module"):
            model = model.module
        backbone = model.model
        frame_tokens = ((clean_latent.shape[-2] // backbone.patch_size[1])
                        * (clean_latent.shape[-1] // backbone.patch_size[2]))
        num_heads = int(backbone.num_heads)
        head_dim = int(backbone.dim // backbone.num_heads)
        num_layers = len(backbone.blocks)
        # A shorter window than the main path uses, and only here. CFG is what
        # makes this rollout show the failure at all -- with it the term reads
        # 0.28, without it 0.0006 no matter how many denoising steps -- and CFG
        # costs a second pair of caches that a 44 GiB card cannot hold at 12
        # frames. Eight frames buys back the 2.3 GiB.
        #
        # The cost is one point of fidelity: inference attends over 12, so this
        # rollout accumulates error faster than deployment does. That makes the
        # phenomenon stronger here, not absent, and the gradient still points
        # at holding detail deep -- but a checkpoint trained this way has been
        # taught against a slightly harsher condition than it will be run in.
        kv_window = int(getattr(self.args, "depth_kv_window_frames", 0)) or \
            int(getattr(self.args, "kv_window_frames", 0))
        cache_frames = (min(num_frames, kv_window) if kv_window > 0 else
                        max(num_frames,
                            int(backbone.local_attn_size)
                            if int(backbone.local_attn_size) > 0 else 0))
        cache_tokens = cache_frames * frame_tokens

        def cache_entry():
            return {
                "k": torch.zeros(batch_size, cache_tokens, num_heads, head_dim,
                                 device=self.device, dtype=self.dtype),
                "v": torch.zeros(batch_size, cache_tokens, num_heads, head_dim,
                                 device=self.device, dtype=self.dtype),
                "global_end_index": torch.zeros(1, device=self.device,
                                                dtype=torch.long),
                "local_end_index": torch.zeros(1, device=self.device,
                                               dtype=torch.long),
                "history_full_k": None,
                "history_full_v": None,
            }

        # The deployment sampler, not a cheap stand-in. A two-step pass without
        # CFG was tried first and carried no signal: it puts the shallow
        # reference and the deep block on the same quality floor, so the term
        # read 0.026 where the healthiest real rollout reads 0.282. The
        # degradation this loss exists to catch only appears in blocks
        # generated the way inference generates them, which costs the negative
        # caches and a doubled forward count.
        # CFG here costs a second pair of caches, and the pair does not fit:
        # the 5B training state leaves about 1.5 GiB on a 44 GiB card, and four
        # caches OOM'd at the same step twice, before and after reclaiming the
        # allocator's reservation. Whether the guided trajectory is what
        # produced the signal, or just the four denoising steps, is settled by
        # measurement rather than by assumption.
        use_cfg = bool(getattr(self.args, "depth_rollout_cfg", True))
        kv_cache = [cache_entry() for _ in range(num_layers)]
        prope_kv_cache = [cache_entry() for _ in range(num_layers)]
        kv_cache_neg = ([cache_entry() for _ in range(num_layers)]
                        if use_cfg else None)
        prope_kv_cache_neg = ([cache_entry() for _ in range(num_layers)]
                              if use_cfg else None)

        def cross_cache():
            return [{
                "k": torch.zeros(batch_size, backbone.text_len, num_heads,
                                 head_dim, device=self.device, dtype=self.dtype),
                "v": torch.zeros(batch_size, backbone.text_len, num_heads,
                                 head_dim, device=self.device, dtype=self.dtype),
                "is_init": False,
            } for _ in range(num_layers)]

        crossattn_cache = cross_cache()
        crossattn_cache_neg = cross_cache() if use_cfg else None
        cfg = float(getattr(self.args, "guidance_scale", 3.0))

        n_ext = depth_blocks * block
        with torch.no_grad():
            # Continue the clip's own camera motion: one relative step,
            # repeated. The trajectory past the data has to come from
            # somewhere, and the clip's last step is the only evidence
            # available about where this camera was going.
            vm_f = viewmats.float()
            step_tf = vm_f[:, -1] @ torch.linalg.inv(vm_f[:, -2])
            cur, grown = vm_f[:, -1], []
            for _ in range(n_ext):
                cur = step_tf @ cur
                grown.append(cur)
            vm_ext = torch.cat([vm_f, torch.stack(grown, dim=1)],
                               dim=1).to(viewmats.dtype)
        ks_ext = (torch.cat([Ks, Ks[:, -1:].expand(-1, n_ext, -1, -1)], dim=1)
                  if Ks is not None else None)

        def cond_for(s_, e_):
            out = dict(conditional_dict)
            for key in ("actions", "embodiment_id"):
                value = out.get(key)
                if not (torch.is_tensor(value) and value.ndim >= 2
                        and value.shape[1] == num_frames):
                    continue
                if e_ <= num_frames:
                    out[key] = value[:, s_:e_]
                elif key == "actions":
                    # Zero, which is what the input masking already feeds at
                    # dropout 1.0 and what the zero-initialised graft reads as
                    # "no action".
                    out[key] = torch.zeros_like(value[:, :e_ - s_])
                else:
                    out[key] = value[:, -1:].expand(-1, e_ - s_)
            return out

        total_blocks = (num_frames + n_ext) // block
        # Reference block, generated by this same cheap sampler at shallow
        # depth. Charging the deep block against the ground-truth clip instead
        # measured the sampler, not the depth: two steps without CFG produce a
        # visibly worse block than the four-step CFG path the clip was rendered
        # with, and the term read 7.3 against a value of 2.5 on the worst real
        # rollout, driving grad_norm to 72 from 0.62. Comparing like with like
        # cancels the sampler and leaves depth as the only difference.
        reference_block = 1
        shallow_x0 = None
        deep_x0 = None
        for b in range(total_blocks):
            s_, e_ = b * block, b * block + block
            last = (b == total_blocks - 1)
            sched = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.scheduler.num_train_timesteps,
                shift=1, use_dynamic_shifting=False)
            retrieve_timesteps(
                sched, steps, device=self.device,
                shift=float(getattr(self.args, "timestep_shift", 5.0)))
            cur_x = torch.randn_like(clean_latent[:, :block])
            # Plain per-block cameras, no chunk-anchor window. The window has
            # to describe the PRoPE cache exactly or the backbone refuses, and
            # reproducing that bookkeeping past the clip is a second problem on
            # top of the one this rollout exists to study.
            vm_b = vm_ext[:, s_:e_]
            ks_b = ks_ext[:, s_:e_] if ks_ext is not None else None
            cond_b = cond_for(s_, e_)
            uncond_b = cond_for(s_, e_) if unconditional_dict is None else {
                **unconditional_dict,
                **{k: v for k, v in cond_for(s_, e_).items()
                   if k in ("actions", "embodiment_id")},
            }
            for i, t in enumerate(sched.timesteps):
                final = last and i == len(sched.timesteps) - 1
                # Gradient on the final denoising step of the final block only;
                # every earlier pass is history and is detached, so memory is
                # one block however deep the rollout goes.
                with torch.enable_grad() if final else torch.no_grad():
                    ts = t * torch.ones([batch_size, block], device=self.device,
                                        dtype=torch.float32)
                    flow_c, x0_b = self.generator(
                        noisy_image_or_video=cur_x, conditional_dict=cond_b,
                        timestep=ts, kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start=s_ * frame_tokens,
                        cache_start=s_ * frame_tokens,
                        viewmats=vm_b, Ks=ks_b, prope_kv_cache=prope_kv_cache,
                        prope_window_viewmats=None, prope_window_Ks=None)
                    if use_cfg:
                        flow_u, _ = self.generator(
                            noisy_image_or_video=cur_x,
                            conditional_dict=uncond_b,
                            timestep=ts, kv_cache=kv_cache_neg,
                            crossattn_cache=crossattn_cache_neg,
                            current_start=s_ * frame_tokens,
                            cache_start=s_ * frame_tokens,
                            viewmats=vm_b, Ks=ks_b,
                            prope_kv_cache=prope_kv_cache_neg,
                            prope_window_viewmats=None, prope_window_Ks=None)
                        flow_b = flow_u + cfg * (flow_c - flow_u)
                    else:
                        flow_b = flow_c
                if final:
                    # x0_b is the conditional estimate, which is what the
                    # in-clip target block is supervised on; the guided flow
                    # only steers the trajectory that produced this block.
                    deep_x0 = x0_b
                    del flow_b, flow_c
                elif b == reference_block and i == len(sched.timesteps) - 1:
                    shallow_x0 = x0_b.detach()
                else:
                    cur_x = sched.step(
                        flow_b.flatten(0, 1).float(), t,
                        cur_x.flatten(0, 1).float(), return_dict=False
                    )[0].unflatten(0, (batch_size, block)).to(cur_x.dtype)

        if deep_x0 is None or shallow_x0 is None:
            return None
        return weight * _detail_collapse_loss(deep_x0, shallow_x0)

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        from post_train.objectives.flow_matching import flow_matching_loss
        from models.prope import _invert_SE3, anchor_viewmats_to_first

        if bool(getattr(self.args, "scheduled_sampling", False)):
            return self._scheduled_sampling_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                viewmats=viewmats,
                Ks=Ks,
            )

        if bool(getattr(self.args, "causal_prope_anchor", False)):
            viewmats = anchor_viewmats_to_first(viewmats)

        noise = torch.randn_like(clean_latent)
        batch_size, num_frame = image_or_video_shape[:2]

        index = self._get_timestep(
            0,
            self.scheduler.num_train_timesteps,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=False
        )
        timestep = self.scheduler.timesteps[index].to(dtype=self.dtype, device=self.device)
        # LongLive uses a clean master branch, then independently replays
        # E_vid into the clean latent and E_noise into the sampled noise.  Our
        # first port only replayed E_img below, which teaches the network to
        # suppress history errors mostly by smoothing them.
        er_ready = self.error_buffer is not None and self.error_buffer.total_added > 0
        er_use_clean = (er_ready and
                        torch.rand((), device=self.device).item() < self.er_clean_prob)
        clean_latent_for_noise = clean_latent
        noise_for_train = noise
        er_latent_injected = False
        er_noise_injected = False
        block = int(self.num_frame_per_block)
        if er_ready and not er_use_clean:
            if torch.rand((), device=self.device).item() < self.er_latent_prob:
                clean_latent_for_noise = clean_latent.clone()
                for pos, start in enumerate(range(0, num_frame, block)):
                    if pos == 0 and self.er_skip_first:
                        continue
                    end = min(start + block, num_frame)
                    err = self.error_buffer.sample(
                        pos, int(index[0, start].item()), self.device,
                        clean_latent.dtype)
                    if err is not None:
                        clean_latent_for_noise[:, start:end] += err[:, :end-start]
                        er_latent_injected = True
            if (self.noise_error_buffer.total_added > 0 and
                    torch.rand((), device=self.device).item() < self.er_noise_prob):
                noise_for_train = noise.clone()
                for pos, start in enumerate(range(0, num_frame, block)):
                    if pos == 0 and self.er_skip_first:
                        continue
                    end = min(start + block, num_frame)
                    err = self.noise_error_buffer.sample(
                        pos, int(index[0, start].item()), self.device,
                        noise.dtype)
                    if err is not None:
                        noise_for_train[:, start:end] += err[:, :end-start]
                        er_noise_injected = True

        noisy_latents = self.scheduler.add_noise(
            clean_latent_for_noise.flatten(0, 1),
            noise_for_train.flatten(0, 1),
            timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))
        training_target = self.scheduler.training_target(
            clean_latent, noise_for_train, timestep)

        # Reference-image conditioning, trained the way bidirectional inference
        # actually applies it.
        #
        # `initial_latent` reached this function as an argument and was never
        # read -- the signature accepted it, the 137-line body ignored it, and
        # the same held for the bidirectional trainer. Both models were trained
        # purely as t2v, so every i2v result this repo has produced came from a
        # conditioning path the weights had never seen.
        #
        # BidirectionalInferencePipeline gets away with it because of *how* it
        # injects: frame 0 of the noisy sequence is overwritten with the clean
        # latent, its timestep is pinned to 0, and both are re-imposed after
        # every denoising step. The frame is a member of the sequence carrying
        # "already denoised", which is close enough to the clean context a
        # teacher-forced model sees in training. The causal path instead
        # prefills the reference into the KV cache as history, where it decays
        # as the window slides -- which is what `sink_size`, `ref_cond_scale`,
        # `cond_prefill_timestep` and `prope_pin_sink` were all added to fight.
        #
        # So train the causal model on the bidirectional recipe: frame 0 clean,
        # timestep 0, no loss on it. `ref_cond_prob` applies it to a fraction of
        # rows so t2v does not decay -- at 1.0 the model would only ever see a
        # reference and lose the ability to start from noise alone.
        ref_cond_prob = float(getattr(self.args, "ref_cond_prob", 0.0))
        ref_rows = None
        if ref_cond_prob > 0 and initial_latent is not None:
            ref_rows = torch.rand(batch_size, device=noisy_latents.device) < ref_cond_prob
            if ref_rows.any():
                sel = ref_rows.view(batch_size, *([1] * (noisy_latents.dim() - 1)))
                ref = initial_latent[:, :1].to(noisy_latents.dtype)
                noisy_latents[:, :1] = torch.where(
                    sel, ref, noisy_latents[:, :1])
                timestep[:, :1] = torch.where(
                    ref_rows.view(batch_size, 1), torch.zeros_like(timestep[:, :1]),
                    timestep[:, :1])

        if self.noise_augmentation_max_timestep > 0:
            # Legacy context augmentation sampled [configured, 1000). CMD's
            # Diffusion Forcing instead needs the full [0, configured) range.
            history_min = 0 if bool(getattr(
                self.args, "diffusion_forcing_history", False)) else self.noise_augmentation_max_timestep
            history_max = self.noise_augmentation_max_timestep if bool(getattr(
                self.args, "diffusion_forcing_history", False)) else 1000
            index_clean_aug = self._get_timestep(
                history_min,
                history_max,
                image_or_video_shape[0],
                image_or_video_shape[1],
                self.num_frame_per_block,
                uniform_timestep=False
            )
            timestep_clean_aug = self.scheduler.timesteps[index_clean_aug].to(dtype=self.dtype, device=self.device)
            # Diffusion Forcing needs a corruption independent of the target
            # noise. Reusing ``noise`` lets the network cancel the two paths
            # and is not the independently-corrupted history in CMD Eq. (4).
            history_noise = (torch.randn_like(clean_latent)
                             if bool(getattr(self.args, "diffusion_forcing_history", False))
                             else noise)
            if bool(getattr(self.args, "df_clean_first_block", False)):
                first = int(getattr(
                    self.args, "df_condition_frames", self.num_frame_per_block))
                timestep_clean_aug[:, :first] = 0
            clean_latent_aug = self.scheduler.add_noise(
                clean_latent.flatten(0, 1),
                history_noise.flatten(0, 1),
                timestep_clean_aug.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))
            if bool(getattr(self.args, "df_clean_first_block", False)):
                clean_latent_aug[:, :first] = clean_latent[:, :first]
        else:
            clean_latent_aug = clean_latent
            timestep_clean_aug = None

        # LongLive/SVI error recycling: approximate self-generated KV history
        # by adding errors previously observed at the same temporal position
        # and nearby noise level.  Keep a clean fraction as an anchor and never
        # corrupt the first sink block.
        er_injected = False
        if (er_ready
                and not er_use_clean
                and torch.rand((), device=self.device).item() < self.er_context_prob):
            clean_latent_aug = clean_latent_aug.clone()
            for pos, start in enumerate(range(0, num_frame, block)):
                if pos == 0 and self.er_skip_first:
                    continue
                end = min(start + block, num_frame)
                err = self.error_buffer.sample_position(
                    pos, self.device, clean_latent_aug.dtype)
                if err is not None:
                    clean_latent_aug[:, start:end] += err[:, :end-start]
                    er_injected = True

        return_actions = bool(getattr(self.generator, "action_output", False))
        # Training the action head against the very actions it is being fed is
        # a copy task: the branch reads `actions` off the modulation path and
        # scores zero loss without learning anything about the scene. So the
        # input stream is masked for a random subset of rows, and the action
        # loss is charged on exactly those rows -- there the head has no choice
        # but to infer the action from the prompt and the video.
        action_mask = None
        if return_actions:
            conditional_dict, action_mask = self._mask_action_input(
                conditional_dict, batch_size, noisy_latents.device)
        model_out = self.generator(
            noisy_image_or_video=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clean_x=clean_latent_aug if self.teacher_forcing else None,
            aug_t=timestep_clean_aug if self.teacher_forcing else None,
            viewmats=viewmats,
            Ks=Ks,
            return_actions=return_actions,
        )
        if return_actions:
            flow_pred, x0_pred, action_pred = model_out
        else:
            flow_pred, x0_pred = model_out
        weight = self.scheduler.training_weight(timestep).unflatten(0, (batch_size, num_frame))
        weight = weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # In the streaming I2V factorisation the first frame is observed
        # context, not a generation target.  Keeping a loss on it lets the
        # teacher-forced model optimise a computation that never occurs at
        # inference and obscures the [frame 0] -> [frames 1..B] boundary.
        if self.teacher_forcing and self.independent_first_frame:
            weight[:, :1] = 0
        if bool(getattr(self.args, "df_clean_first_block", False)):
            # The first block is I2V conditioning memory, not a generated CMD
            # target. Later targets may attend to it, but it carries no loss.
            weight[:, :int(getattr(
                self.args, "df_condition_frames", 1))] = 0
        if ref_rows is not None and ref_rows.any():
            # The reference frame is given, not predicted. Charging loss on it
            # would train the model to reproduce an input it already has.
            weight[ref_rows, :1] = 0
        loss = flow_matching_loss(flow_pred, training_target, weight=weight)

        # Current-model additions: explicitly preserve the temporal delta at
        # causal chunk boundaries and the low-frequency per-channel exposure.
        # These operate on x0 (not RGB) and use the ground-truth change, so
        # intentional lighting transitions remain learnable.
        boundary_loss = x0_pred.sum() * 0.0
        luma_loss = x0_pred.sum() * 0.0
        if self.er_boundary_weight > 0 and num_frame > block:
            starts = torch.arange(block, num_frame, block, device=x0_pred.device)
            pred_delta = x0_pred[:, starts] - x0_pred[:, starts - 1]
            gt_delta = clean_latent[:, starts] - clean_latent[:, starts - 1]
            boundary_loss = torch.nn.functional.mse_loss(
                pred_delta.float(), gt_delta.float())
            loss = loss + self.er_boundary_weight * boundary_loss
        if self.er_luma_weight > 0 and num_frame > 1:
            # The *change* in exposure, not its level.
            #
            # Matching the level pins every frame's low frequency to the value a
            # 20-latent training clip happens to have, which is very nearly
            # constant. Inference runs 96 latents, the picture legitimately
            # drifts away from that opening value, and the term hauls it back --
            # visible as the similarity trace falling into a pit and recovering,
            # three times in sixteen seconds on a checkpoint trained this way
            # against one on the checkpoint it started from.
            #
            # Constraining the frame-to-frame difference says what the term is
            # actually for: exposure must not jump, and it is free to go
            # wherever the content takes it. boundary_loss above already works
            # on differences for the same reason; this brings luma into line.
            pred_low = x0_pred.float().mean(dim=(-2, -1))
            gt_low = clean_latent.float().mean(dim=(-2, -1))
            luma_loss = torch.nn.functional.mse_loss(
                pred_low[:, 1:] - pred_low[:, :-1],
                gt_low[:, 1:] - gt_low[:, :-1])
            loss = loss + self.er_luma_weight * luma_loss

        if self.error_buffer is not None:
            with torch.no_grad():
                error = x0_pred.detach() - clean_latent.detach()
                sigma = self.scheduler.sigmas.to(flow_pred.device)[index].reshape(
                    batch_size, num_frame, 1, 1, 1).to(flow_pred.dtype)
                noise_error = ((flow_pred.detach() - training_target.detach())
                               * (1.0 - sigma))
                for pos, start in enumerate(range(0, num_frame, block)):
                    end = min(start + block, num_frame)
                    self.error_buffer.add(
                        pos, int(index[0, start].item()), error[:, start:end])
                    self.noise_error_buffer.add(
                        pos, int(index[0, start].item()),
                        noise_error[:, start:end])

        if return_actions:
            action_loss = self._action_output_loss(
                action_pred, self._gt_actions, action_mask)
            if action_loss is not None:
                loss = loss + float(
                    getattr(self.args, "action_loss_weight", 1.0)) * action_loss
                log_dict = {"x0": clean_latent.detach(),
                            "x0_pred": x0_pred.detach(),
                            "action_pred": action_pred.detach(),
                            "action_loss": action_loss.detach()}
                return loss, log_dict

        return loss, {"x0": clean_latent.detach(), "x0_pred": x0_pred.detach(),
                      "error_recycling_injected": torch.tensor(
                          float(er_injected), device=self.device),
                      "er_latent_injected": torch.tensor(
                          float(er_latent_injected), device=self.device),
                      "er_noise_injected": torch.tensor(
                          float(er_noise_injected), device=self.device),
                      "boundary_loss": boundary_loss.detach(),
                      "luma_loss": luma_loss.detach(),
                      # The timestep this step happened to draw. Flow-matching
                      # loss varies severalfold across the schedule and this
                      # trainer resamples the full 0-1000 range every step, per
                      # block -- so most of the per-step swing (21% between
                      # neighbours, p10 0.250 against p90 0.500) is which
                      # timesteps came up, not how the model is doing. Logging
                      # it lets the curve be read per bucket instead of raw,
                      # which is the difference between seeing a trend and
                      # seeing the sampler.
                      "timestep_mean": timestep.detach().float().mean()}

    # ---- action output head ------------------------------------------------

    def _mask_action_input(self, conditional_dict: dict, batch_size: int, device):
        """Blank the input action stream on a random subset of rows.

        Returns a shallow copy of ``conditional_dict`` with the masked stream
        and a bool mask [B] marking the rows the action loss may be charged on.
        The ground truth is stashed on ``self._gt_actions`` first, because the
        dict the model sees no longer carries it.

        ``action_output_input_dropout`` defaults to 1.0: with the head switched
        on, the default is to train it purely as a predictor. Lower it to train
        one model that both consumes and predicts actions.
        """
        gt_actions = conditional_dict.get("actions", None)
        self._gt_actions = gt_actions
        if gt_actions is None:
            return conditional_dict, None

        p = float(getattr(self.args, "action_output_input_dropout", 1.0))
        if p >= 1.0:
            mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        elif p <= 0.0:
            return conditional_dict, torch.zeros(
                batch_size, dtype=torch.bool, device=device)
        else:
            mask = torch.rand(batch_size, device=device) < p

        masked = dict(conditional_dict)
        keep = (~mask).to(gt_actions.dtype).view(-1, *([1] * (gt_actions.dim() - 1)))
        # Zero, not absent: `apply_action_modulation` skips the embedder
        # entirely when `actions` is None, and under FSDP a rank that skips a
        # module while its peers run it hangs on mismatched collectives.
        # The graft is zero-initialised, so a zero action *is* "no action".
        masked["actions"] = gt_actions * keep
        return masked, mask

    def _action_output_loss(self, action_pred, gt_actions, mask):
        """MSE on the rows whose input action stream was masked out."""
        if gt_actions is None:
            return None
        gt = gt_actions.to(action_pred.dtype)
        if mask is None or bool(mask.all()):
            return torch.nn.functional.mse_loss(action_pred, gt)
        if not bool(mask.any()):
            # Nothing to charge this step, but the head still has to appear in
            # the graph or FSDP's reduce-scatter for its parameters never fires
            # and the ranks desynchronise.
            return action_pred.sum() * 0.0
        return torch.nn.functional.mse_loss(action_pred[mask], gt[mask])

    def _scheduled_sampling_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        viewmats: Optional[torch.Tensor],
        Ks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, dict]:
        """Supervise one block behind model-generated causal history.

        Earlier blocks are predicted under ``no_grad`` and written back to the
        normal and PRoPE KV caches at timestep zero, exactly as AR inference
        does. Only the randomly selected target block retains autograd. This is
        scheduled sampling with a real Control2 flow target, not DMD: the model
        cannot read motion from pristine future history, but the optimisation
        target remains the ground-truth video.
        """
        from post_train.objectives.flow_matching import flow_matching_loss
        from models.prope import _invert_SE3, anchor_viewmats_to_first

        batch_size, num_frames = clean_latent.shape[:2]

        # The action objective, which this path did not have.
        #
        # `generator_loss` trains the action head; `_scheduled_sampling_loss`
        # never did, so every rollout run in this lineage has carried
        # `use_action: true` while giving the head no gradient at all. The
        # masking is the same as there and is the reason the term is worth
        # anything: fed the actions it is scored on, the branch reads them off
        # the modulation path and scores zero without learning the scene, so
        # the input is withheld on a random subset of rows and the loss is
        # charged on exactly those.
        action_output = bool(getattr(self.generator, "action_output", False))
        action_mask = None
        if action_output:
            conditional_dict, action_mask = self._mask_action_input(
                conditional_dict, batch_size, self.device)

        block = int(self.num_frame_per_block)
        independent_first = bool(getattr(self, "independent_first_frame", False))
        if independent_first:
            if (num_frames - 1) % block:
                raise ValueError(
                    f"I2V scheduled sampling needs F=1+n*block, got "
                    f"F={num_frames}, block={block}")
            spans = [(0, 1)] + [
                (start, min(start + block, num_frames))
                for start in range(1, num_frames, block)]
        else:
            if num_frames % block:
                raise ValueError(
                    f"scheduled sampling needs whole blocks: F={num_frames}, block={block}")
            spans = [(start, start + block)
                     for start in range(0, num_frames, block)]
        num_blocks = len(spans)
        if num_blocks < 2:
            raise ValueError("scheduled sampling needs at least two temporal blocks")

        # FSDP performs collectives on every model call, so all ranks must pick
        # the same block and execute the same number of rollout forwards.
        configured_target = int(getattr(
            self.args, "scheduled_target_block", -1))
        if configured_target >= 0:
            if not 1 <= configured_target < num_blocks:
                raise ValueError(
                    f"scheduled_target_block must be in [1,{num_blocks - 1}]")
            target_tensor = torch.tensor(
                [configured_target], device=self.device, dtype=torch.long)
        else:
            target_tensor = torch.randint(
                1, num_blocks, (1,), device=self.device, dtype=torch.long)
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(target_tensor, src=0)
        target_block = int(target_tensor.item())

        noise = torch.randn_like(clean_latent)
        index = self._get_timestep(
            0, self.scheduler.num_train_timesteps, batch_size, num_frames,
            block, uniform_timestep=False)

        # Rolling Forcing's denoising window, in training.
        #
        # `_get_timestep` gives every frame of a block the same noise level, so
        # the block is one flat step and whatever error it carries is frozen
        # into the cache when it is done. Rolling Forcing (arXiv 2509.25161)
        # instead holds "consecutive frames with progressively higher noise
        # levels in temporal order" -- its [1000, 800, 600, 400, 200] -- and
        # denoises them jointly, so each frame is refined against a cleaner
        # predecessor inside the same pass rather than against a finished one.
        #
        # Scaled by the block's own sampled level rather than pinned to the
        # paper's constants: the flat draw is what sets this run's noise
        # distribution, and replacing it with five fixed numbers would change
        # the training distribution as well as its shape. Frame j of F gets
        # t * (j+1)/F -- ascending, so the last frame keeps the sampled value
        # and the earlier ones are cleaner.
        #
        # Mixed with the ordinary path at equal probability, which is the
        # paper's own recipe ("alternates between SF training and Rolling
        # Forcing training with equal probability", SF as the regulariser).
        # The draw is broadcast: FSDP collectives require every rank to take
        # the same branch.
        rolling_forcing = bool(getattr(self.args, "rolling_forcing", False))
        rolling_now = False
        if rolling_forcing:
            pick = torch.rand(1, device=self.device)
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(pick, src=0)
            rolling_now = bool(pick.item() < float(getattr(
                self.args, "rolling_forcing_prob", 0.5)))
        if rolling_now:
            t_start, t_end = spans[target_block]
            span_len = t_end - t_start
            if span_len > 1:
                # The window is the full schedule, not a scaled version of
                # the block's own draw. That is the paper's construction --
                # [1000, 800, 600, 400, 200] regardless of anything sampled --
                # and it is what makes the two alternating paths different
                # regimes rather than the same one twice. The random draw stays
                # on the SF steps, where it belongs.
                #
                # Built in timestep space and looked up, not in index space.
                # `set_timesteps` warps sigmas by `shift`, so a linear ramp over
                # indices is not linear over timesteps: from index 600 it spans
                # 770 to 966, a window narrow enough that the mechanism would
                # barely differ from the flat block it replaces.
                levels = (
                    torch.arange(1, span_len + 1, device=self.device,
                                 dtype=torch.float32) / float(span_len)
                    * float(self.scheduler.num_train_timesteps))
                schedule = self.scheduler.timesteps.to(self.device).float()
                nearest = (schedule.unsqueeze(0)
                           - levels.unsqueeze(1)).abs().argmin(dim=1)
                index = index.clone()
                index[:, t_start:t_end] = nearest.unsqueeze(0).expand(
                    batch_size, -1)

        timestep = self.scheduler.timesteps[index].to(
            dtype=self.dtype, device=self.device)
        noisy_latents = self.scheduler.add_noise(
            clean_latent.flatten(0, 1), noise.flatten(0, 1),
            timestep.flatten(0, 1)).unflatten(0, (batch_size, num_frames))
        training_target = self.scheduler.training_target(
            clean_latent, noise, timestep)

        model = self.generator
        while hasattr(model, "module"):
            model = model.module
        backbone = model.model
        frame_tokens = ((clean_latent.shape[-2] // backbone.patch_size[1])
                        * (clean_latent.shape[-1] // backbone.patch_size[2]))
        num_heads = int(backbone.num_heads)
        head_dim = int(backbone.dim // backbone.num_heads)
        num_layers = len(backbone.blocks)
        kv_window = int(getattr(self.args, "kv_window_frames", 0))
        if kv_window > 0:
            # Bounded sliding-window KV cache. Memory stays constant for
            # arbitrarily long training sequences: once the window fills, the
            # backbone evicts the oldest frames (same eviction path inference
            # uses), so minute-level sequences train at the same footprint as
            # the 17-frame run while still learning to predict from bounded
            # history.
            cache_frames = min(num_frames, kv_window)
        else:
            cache_frames = max(
                num_frames,
                int(backbone.local_attn_size) if int(backbone.local_attn_size) > 0 else 0)
        cache_tokens = cache_frames * frame_tokens

        def cache_entry():
            return {
                "k": torch.zeros(
                    batch_size, cache_tokens, num_heads, head_dim,
                    device=self.device, dtype=self.dtype),
                "v": torch.zeros(
                    batch_size, cache_tokens, num_heads, head_dim,
                    device=self.device, dtype=self.dtype),
                "global_end_index": torch.zeros(
                    1, device=self.device, dtype=torch.long),
                "local_end_index": torch.zeros(
                    1, device=self.device, dtype=torch.long),
                # Keep the same non-evicting raw content bank used by
                # 4-step inference.  The attention module grows/overwrites
                # these entries as each rollout block is denoised and the
                # timestep-zero commit becomes the final history writer.
                "history_full_k": None,
                "history_full_v": None,
            }

        kv_cache = [cache_entry() for _ in range(num_layers)]
        kv_cache_neg = [cache_entry() for _ in range(num_layers)]
        # Two of the four caches exist only to serve PRoPE, and each is
        # cache_frames * frame_tokens * dim * 2 bytes * 2 * num_layers -- about
        # 2.9 GB apiece at 20 frames. block_cameras() already returns None for
        # every camera argument when viewmats is None, so with no camera signal
        # these are 5.7 GB reserved to cache a constant. CrossFPS ships
        # [0,0,0, 0,0,0,1] for every frame of every clip, so that is exactly the
        # case here.
        prope_kv_cache = ([cache_entry() for _ in range(num_layers)]
                          if viewmats is not None else None)
        prope_kv_cache_neg = ([cache_entry() for _ in range(num_layers)]
                              if viewmats is not None else None)
        def cross_cache():
            return [{
            "k": torch.zeros(
                batch_size, backbone.text_len, num_heads, head_dim,
                device=self.device, dtype=self.dtype),
            "v": torch.zeros(
                batch_size, backbone.text_len, num_heads, head_dim,
                device=self.device, dtype=self.dtype),
            "is_init": False,
            } for _ in range(num_layers)]
        crossattn_cache = cross_cache()
        crossattn_cache_neg = cross_cache()

        def slice_cond(start, end):
            result = dict(conditional_dict)
            for key in ("actions", "embodiment_id"):
                value = result.get(key)
                if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == num_frames:
                    result[key] = value[:, start:end]
            return result

        def slice_uncond(start, end):
            result = dict(unconditional_dict)
            # Camera/action controls are shared between CFG branches at
            # inference. Preserve per-frame controls when present.
            for key in ("actions", "embodiment_id"):
                value = result.get(key)
                if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == num_frames:
                    result[key] = value[:, start:end]
            return result

        # Chunk-relative PRoPE: every block sees its cache window re-anchored
        # to the last frame before it, matching chunked inference. The window
        # mirrors the backbone's KV eviction so raw cached K/V re-projection
        # lines up 1:1 with the window cameras.
        chunk_anchor = bool(getattr(
            self.generator.model.blocks[0].self_attn, "prope_chunk_anchor", False))
        prope_sink_frames = int(getattr(
            self.generator.model.blocks[0].self_attn, "sink_size", 1))
        local_attn_size = int(getattr(self.generator.model, "local_attn_size", -1))
        cache_cap = int(getattr(self.args, "kv_window_frames", 0))
        if cache_cap <= 0:
            cache_cap = max(
                num_frames, local_attn_size if local_attn_size != -1 else 0)

        retrieval = bool(getattr(self.args, "retrieval_window", False))
        retrieval_recent = int(getattr(self.args, "retrieval_recent_frames", 4))

        previous_retrieved = []

        def prope_window_for(new_frames, mirror):
            nonlocal previous_retrieved
            combined = list(mirror) + list(new_frames)
            if len(combined) <= cache_cap:
                return combined
            evict = len(combined) - cache_cap
            sink = max(0, min(prope_sink_frames, len(combined)))
            if not (retrieval and viewmats is not None):
                return combined[:sink] + combined[sink + evict:]
            # Keep what the incoming block is about to look at. Age-based
            # eviction is why a corridor walked forty frames ago cannot be
            # redrawn consistently on the way back: the frames that saw it are
            # already gone. The newest few are pinned regardless -- Memory
            # Forcing reports that leaning wholly on spatial memory degrades
            # generation of genuinely new scenery, and those frames are the
            # temporal half of that trade.
            body = combined[sink:len(combined) - len(new_frames)]
            keep_n = cache_cap - sink - len(new_frames)
            from utils.cache_selection import select_history_frames
            recent = min(keep_n, max(0, retrieval_recent))
            candidates = body[:-recent] if recent else body
            room = keep_n - recent
            scores = (_frustum_overlap(viewmats, Ks, list(new_frames)[-1], candidates)
                      if room and candidates else {})
            body, previous_retrieved = select_history_frames(
                body, keep_n, recent, scores,
                float(getattr(self.args, "retrieval_min_overlap", 0.3)),
                previous_retrieved,
                int(getattr(self.args, "retrieval_swap_per_block", 2)))
            return combined[:sink] + body + list(new_frames)

        block_windows = {}
        if chunk_anchor and viewmats is not None:
            mirror = []
            for b in range(target_block + 1):
                s, e = spans[b]
                window = prope_window_for(list(range(s, e)), mirror)
                block_windows[b] = (s, e, window)
                mirror = window

        keep_now = []

        def block_cameras(b, start, end):
            """Return (chunk vm, chunk Ks, window vm, window Ks)."""
            if viewmats is None:
                return None, None, None, None
            if chunk_anchor:
                s, e, window_frames = block_windows[b]
                anchor_frame = max(0, start - 1)
                from models.prope import anchor_viewmats_to_frame
                vm_window = anchor_viewmats_to_frame(
                    viewmats[:, window_frames], window_frames.index(anchor_frame))
                ks_window = Ks[:, window_frames] if Ks is not None else None
                n = end - start
                # The backbone must evict to exactly this window or PRoPE and
                # the residents disagree and it refuses; the survivors are the
                # window minus the block being written now.
                keep_now[:] = [f for f in window_frames[prope_sink_frames:] if f < start]
                return (vm_window[:, -n:],
                        ks_window[:, -n:] if ks_window is not None else None,
                        vm_window, ks_window)
            return (viewmats[:, start:end],
                    Ks[:, start:end] if Ks is not None else None,
                    None, None)

        # Roll out all blocks before the target with the exact four-step UniPC
        # trajectory used by inference. The old path made one random-timestep
        # x0 estimate, which was neither self-forcing nor the distribution the
        # distilled student sees at deployment.
        from models.fm_solvers import retrieve_timesteps
        from models.fm_solvers_unipc import FlowUniPCMultistepScheduler
        rollout_steps = int(getattr(self.args, "self_forcing_sampling_steps", 4))
        rollout_cfg = float(getattr(self.args, "guidance_scale", 3.0))
        last_history_x0 = None
        with torch.no_grad():
            for block_idx in range(target_block):
                start, end = spans[block_idx]
                vm, intr, vm_window, ks_window = block_cameras(
                    block_idx, start, end)
                cond = slice_cond(start, end)
                uncond = slice_uncond(start, end)
                # I2V inference writes the exact clean reference frame at
                # timestep zero.  Generating this one-frame span from noise
                # trains against a T2V-like anchor and recreates the very
                # train/serve mismatch the history adapter is meant to fix.
                clean_reference = independent_first and block_idx == 0
                if clean_reference:
                    history_x0 = clean_latent[:, start:end].detach()
                else:
                    history_x0 = noise[:, start:end]
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.scheduler.num_train_timesteps,
                        shift=1, use_dynamic_shifting=False)
                    retrieve_timesteps(
                        sample_scheduler, rollout_steps, device=self.device,
                        shift=float(getattr(self.args, "timestep_shift", 5.0)))
                    for sample_t in sample_scheduler.timesteps:
                        sample_ts = sample_t * torch.ones(
                            [batch_size, end - start], device=self.device,
                            dtype=torch.float32)
                        flow_cond, _ = self.generator(
                            noisy_image_or_video=history_x0,
                            conditional_dict=cond, timestep=sample_ts,
                            kv_cache=kv_cache, crossattn_cache=crossattn_cache,
                            current_start=start * frame_tokens,
                            cache_start=start * frame_tokens,
                            viewmats=vm, Ks=intr,
                            prope_kv_cache=prope_kv_cache,
                            prope_window_viewmats=vm_window,
            cache_keep_frames=(keep_now if retrieval else None),
                            prope_window_Ks=ks_window)
                        flow_uncond, _ = self.generator(
                            noisy_image_or_video=history_x0,
                            conditional_dict=uncond, timestep=sample_ts,
                            kv_cache=kv_cache_neg,
                            crossattn_cache=crossattn_cache_neg,
                            current_start=start * frame_tokens,
                            cache_start=start * frame_tokens,
                            viewmats=vm, Ks=intr,
                            prope_kv_cache=prope_kv_cache_neg,
                            prope_window_viewmats=vm_window,
            cache_keep_frames=(keep_now if retrieval else None),
                            prope_window_Ks=ks_window)
                        flow = flow_uncond + rollout_cfg * (flow_cond - flow_uncond)
                        history_x0 = sample_scheduler.step(
                            flow, sample_t, history_x0, return_dict=False)[0]
                last_history_x0 = history_x0.detach()
                # Replace the noisy cache entry by the model's own prediction.
                #
                # Timestep zero says "this history is exact". At inference it is
                # not: the cache holds what the student itself produced a few
                # blocks ago, carrying its own residual error, and the student
                # has only ever been asked to predict from history it was told
                # was clean. ss_context_noise commits the history at a small
                # nonzero timestep instead -- rolling forcing's context_noise,
                # which is the part of that method that needs no multi-block
                # window and so no rework of the per-block PRoPE anchoring.
                # 0 keeps the previous behaviour exactly.
                ctx_noise = int(getattr(self.args, "ss_context_noise", 0))
                if ctx_noise > 0:
                    zero_ts = torch.full_like(timestep[:, start:end], ctx_noise)
                    history_x0 = self.scheduler.add_noise(
                        history_x0.flatten(0, 1),
                        torch.randn_like(history_x0.flatten(0, 1)),
                        torch.full((history_x0.shape[0] * history_x0.shape[1],),
                                   ctx_noise, device=history_x0.device,
                                   dtype=torch.long),
                    ).unflatten(0, history_x0.shape[:2])
                else:
                    zero_ts = torch.zeros_like(timestep[:, start:end])
                for cache_cond, cache, cross, prope in (
                    (cond, kv_cache, crossattn_cache, prope_kv_cache),
                    (uncond, kv_cache_neg, crossattn_cache_neg,
                     prope_kv_cache_neg),
                ):
                    self.generator(
                        noisy_image_or_video=history_x0.detach(),
                        conditional_dict=cache_cond, timestep=zero_ts,
                        kv_cache=cache, crossattn_cache=cross,
                        current_start=start * frame_tokens,
                        cache_start=start * frame_tokens,
                        viewmats=vm, Ks=intr, prope_kv_cache=prope,
                        prope_window_viewmats=vm_window,
            cache_keep_frames=(keep_now if retrieval else None),
                        prope_window_Ks=ks_window)

        start, end = spans[target_block]
        target_frames = end - start
        vm, intr, vm_window, ks_window = block_cameras(
            target_block, start, end)
        teacher_target_x0 = None
        if self.rollout_teacher is not None:
            # The teacher sees the whole real clip bidirectionally, but its
            # target is applied only to the block that the causal student now
            # predicts from model-generated history. This supplies a stable
            # identity target without teacher-forcing the student's cache.
            teacher_input = noisy_latents.detach().clone()
            teacher_timestep = timestep.detach().clone()
            if independent_first:
                teacher_input[:, 0] = clean_latent[:, 0]
                teacher_timestep[:, 0] = 0
            teacher_cond = {
                key: value for key, value in conditional_dict.items()
                if key not in ("actions", "embodiment_id", "cosmos_actions")
            }
            with torch.no_grad():
                _, teacher_x0 = self.rollout_teacher(
                    teacher_input, teacher_cond, teacher_timestep,
                    viewmats=viewmats, Ks=Ks)
            teacher_target_x0 = teacher_x0[:, start:end].detach()
        target_out = self.generator(
            noisy_image_or_video=noisy_latents[:, start:end],
            conditional_dict=slice_cond(start, end),
            timestep=timestep[:, start:end],
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=start * frame_tokens,
            cache_start=start * frame_tokens,
            viewmats=vm,
            Ks=intr,
            prope_kv_cache=prope_kv_cache,
            prope_window_viewmats=vm_window,
            cache_keep_frames=(keep_now if retrieval else None),
            prope_window_Ks=ks_window,
            return_actions=action_output,
        )
        if action_output:
            pred, x0_pred, action_pred = target_out
        else:
            pred, x0_pred = target_out
            action_pred = None
        # Anchor the global camera direction to the frozen base.
        #
        # Every other term in this loss is an appearance term -- boundary,
        # luma, luma_delta, spatial_gradient, foreground -- and none of them
        # constrains where the camera goes. Measured over the 575 lineage, the
        # base moves forward (radial/horizontal 4.34, parallax 2.08) and the
        # adapter turns that into a sideways sweep (0.07, 1.10) while the
        # global motion magnitude does not fall. Nothing in the objective
        # objects, because a sweep renders sharper than a dolly.
        #
        # The base is already here: the adapter is an overlay on a frozen
        # backbone, so `using_adapter(model, None)` is the base itself -- one
        # extra forward, no second copy of the weights. It runs against the KV
        # cache the student's own rollout filled, holding the history fixed so
        # the comparison isolates what the adapter does to *this* block.
        #
        # Direction only, via cosine on the fitted (pan, tilt, radial) vector.
        # Magnitude is left free, and the residual after the global fit -- the
        # falling petals, which the adapter never harmed -- is not touched at
        # all. The constrained subspace is three numbers wide, which is why
        # this is not expected to fight the sharpness terms the way anchoring
        # the whole delta field would.
        # Scene motion: make what moves on its own move the way it really does.
        #
        # The camera anchor above deliberately leaves the residual free, and
        # this is the term that takes it. Everything the rollout is asked for
        # otherwise -- boundary, luma, spatial_gradient, foreground -- is
        # satisfiable by a still frame rendered well, and a still frame is what
        # a long rollout drifts towards.
        #
        # Cosine against the ground truth's residual, not an energy match. An
        # energy target is satisfiable by grain, and at step 575 the noise floor
        # runs 46 against the base's 25, so an energy term would be paid in
        # exactly the wrong currency. Matching structure cannot be: noise
        # correlates with the real field at zero. It is also magnitude-free, so
        # it asks the petals to fall in the right places rather than asking the
        # picture to become busier.
        # Charged on the target block only, so the ground truth is sliced to the
        # same span the head predicted. `_gt_actions` is set by
        # `_mask_action_input` above and spans the whole clip.
        action_loss = x0_pred.sum() * 0.0
        if action_pred is not None and getattr(self, "_gt_actions", None) is not None:
            charged = self._action_output_loss(
                action_pred, self._gt_actions[:, start:end], action_mask)
            if charged is not None:
                action_loss = charged

        scene_motion_loss = x0_pred.sum() * 0.0
        scene_motion_weight = float(getattr(
            self.args, "scene_motion_weight", 0.0))

        motion_anchor_loss = x0_pred.sum() * 0.0
        motion_anchor_weight = float(getattr(
            self.args, "motion_anchor_weight", 0.0))
        if motion_anchor_weight > 0 and target_frames > 1:
            from models.lora import using_adapter
            with torch.no_grad():
                with using_adapter(self.generator.model, None):
                    _, x0_base = self.generator(
                        noisy_image_or_video=noisy_latents[:, start:end],
                        conditional_dict=slice_cond(start, end),
                        timestep=timestep[:, start:end],
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start=start * frame_tokens,
                        cache_start=start * frame_tokens,
                        viewmats=vm,
                        Ks=intr,
                        prope_kv_cache=prope_kv_cache,
                        prope_window_viewmats=vm_window,
            cache_keep_frames=(keep_now if retrieval else None),
                        prope_window_Ks=ks_window,
                    )
            student_motion = _global_motion_direction(x0_pred)
            base_motion = _global_motion_direction(x0_base.detach())
            motion_anchor_loss = (
                1.0 - torch.nn.functional.cosine_similarity(
                    student_motion, base_motion, dim=-1, eps=1e-6)).mean()

        camera_direction_loss = x0_pred.sum() * 0.0
        camera_direction_weight = float(getattr(
            self.args, "camera_direction_weight", 0.0))
        direction_first_block_only = bool(getattr(
            self.args, "camera_direction_first_block_only", False))
        if (camera_direction_weight > 0 and self.rollout_teacher is not None
                and viewmats is not None and vm is not None
                and (not direction_first_block_only or target_block == 1)):
            # Counterfactual camera on the target block.  The student history
            # (rollout blocks) stays on the real trajectory; only the target
            # block's chunk camera is inverted, so the *difference* between the
            # two forwards isolates the camera-direction response from content.
            # This is the scheduled-sampling analogue of precausal_cd's signed
            # direction objective: keep self-attention LoRA consuming PRoPE's
            # forward/backward signal instead of learning a content prior that
            # overwrites it.
            n = target_frames
            vm_rev = _invert_SE3(vm.float()).to(vm.dtype)
            vm_window_rev = vm_window.clone()
            vm_window_rev[:, -n:] = vm_rev
            # The counterfactual forward only supplies the baseline of the
            # direction delta.  Gradients flow through the real forward, which
            # is what must become direction-sensitive; recomputing gradients
            # for the counterfactual would double peak activation memory for
            # no signal.
            with torch.no_grad():
                _, x0_pred_rev = self.generator(
                    noisy_image_or_video=noisy_latents[:, start:end],
                    conditional_dict=slice_cond(start, end),
                    timestep=timestep[:, start:end],
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=start * frame_tokens,
                    cache_start=start * frame_tokens,
                    viewmats=vm_rev,
                    Ks=intr,
                    prope_kv_cache=prope_kv_cache,
                    prope_window_viewmats=vm_window_rev,
                    prope_window_Ks=ks_window,
                )
            student_delta = x0_pred.float() - x0_pred_rev.float()
            with torch.no_grad():
                _, teacher_x0_rev = self.rollout_teacher(
                    teacher_input, teacher_cond, teacher_timestep,
                    viewmats=_invert_SE3(
                        anchor_viewmats_to_first(viewmats.float())).to(
                            viewmats.dtype),
                    Ks=Ks)
            teacher_delta = (teacher_target_x0.float()
                             - teacher_x0_rev[:, start:end].float())
            # Camera direction is a background/geometry objective.  Letting
            # its asymmetric counterfactual delta act on the persistent
            # foreground made identity depend on the sign: W erased the hand
            # while S retained it.  Exclude the same training-only reference
            # region used by identity distillation before computing direction.
            direction_region = getattr(
                self.args, "camera_direction_exclude_region", None)
            if direction_region is not None:
                if len(direction_region) != 4:
                    raise ValueError(
                        "camera_direction_exclude_region must be [y0,y1,x0,x1]")
                height, width = student_delta.shape[-2:]
                y0, y1, x0, x1 = [float(value) for value in direction_region]
                iy0 = max(0, min(height - 1, int(round(y0 * height))))
                iy1 = max(iy0 + 1, min(height, int(round(y1 * height))))
                ix0 = max(0, min(width - 1, int(round(x0 * width))))
                ix1 = max(ix0 + 1, min(width, int(round(x1 * width))))
                keep_mask = torch.ones_like(student_delta[:, :, :1])
                keep_mask[..., iy0:iy1, ix0:ix1] = 0
                student_delta = student_delta * keep_mask
                teacher_delta = teacher_delta * keep_mask
            student_delta = student_delta.flatten(2)
            teacher_delta = teacher_delta.flatten(2)
            camera_direction_loss = (
                1.0 - torch.nn.functional.cosine_similarity(
                    student_delta, teacher_delta, dim=2, eps=1e-6)).mean()
        weight = self.scheduler.training_weight(
            timestep[:, start:end]).reshape(batch_size, target_frames)
        weight = weight[..., None, None, None]
        loss = flow_matching_loss(
            pred, training_target[:, start:end], weight=weight)
        boundary_loss = x0_pred.sum() * 0.0
        luma_loss = x0_pred.sum() * 0.0
        luma_delta_loss = x0_pred.sum() * 0.0
        luma_velocity_loss = x0_pred.sum() * 0.0
        spatial_gradient_loss = x0_pred.sum() * 0.0
        foreground_loss = x0_pred.sum() * 0.0
        reference_region_loss = x0_pred.sum() * 0.0
        rollout_teacher_identity_loss = x0_pred.sum() * 0.0
        rollout_teacher_gradient_loss = x0_pred.sum() * 0.0
        if last_history_x0 is not None:
            pred_delta = x0_pred[:, :1] - last_history_x0[:, -1:]
            gt_delta = clean_latent[:, start:start + 1] - clean_latent[:, start - 1:start]
            boundary_loss = torch.nn.functional.mse_loss(
                pred_delta.float(), gt_delta.float())
            pred_low = x0_pred.float().mean(dim=(-2, -1))
            gt_low = clean_latent[:, start:end].float().mean(dim=(-2, -1))
            luma_loss = torch.nn.functional.mse_loss(pred_low, gt_low)
            # Absolute per-frame latent means alone do not constrain the
            # exposure jump seen by an autoregressive rollout: the history is
            # generated, not ground truth.  Match the low-frequency change
            # from the actual generated history into the target block.
            history_low = last_history_x0[:, -1:].float().mean(dim=(-2, -1))
            gt_history_low = clean_latent[:, start - 1:start].float().mean(
                dim=(-2, -1))
            luma_delta_loss = torch.nn.functional.mse_loss(
                pred_low[:, :1] - history_low,
                gt_low[:, :1] - gt_history_low)
            # Suppress within-block exposure pulsing while retaining genuine
            # lighting changes present in the teacher sequence.
            if target_frames > 1:
                luma_velocity_loss = torch.nn.functional.mse_loss(
                    pred_low[:, 1:] - pred_low[:, :-1],
                    gt_low[:, 1:] - gt_low[:, :-1])
            # Long autoregressive rollouts otherwise converge toward smooth
            # latent fields even when their global colour remains correct.
            # Match horizontal/vertical detail at the generated-history target
            # rather than rewarding artificial sharpening by magnitude alone.
            pred_f = x0_pred.float()
            gt_f = clean_latent[:, start:end].float()
            if scene_motion_weight > 0 and target_frames > 1:
                _, pred_residual = _global_motion_split(x0_pred)
                with torch.no_grad():
                    _, gt_residual = _global_motion_split(gt_f)
                scene_motion_loss = (
                    1.0 - torch.nn.functional.cosine_similarity(
                        pred_residual.flatten(1), gt_residual.flatten(1),
                        dim=1, eps=1e-6)).mean()
            spatial_gradient_loss = (
                torch.nn.functional.mse_loss(
                    pred_f[..., 1:, :] - pred_f[..., :-1, :],
                    gt_f[..., 1:, :] - gt_f[..., :-1, :])
                + torch.nn.functional.mse_loss(
                    pred_f[..., :, 1:] - pred_f[..., :, :-1],
                    gt_f[..., :, 1:] - gt_f[..., :, :-1]))
            # Whole-frame flow loss is dominated by static background.  In
            # case171 the hand-held lantern occupies only ~4% of the image,
            # so losing it can reduce the average loss.  Build a generic
            # target-only saliency map from spatial structure and motion,
            # then normalize it to mean one.  This does not depend on prompts
            # or case-specific masks and focuses the adapter on small moving,
            # structured subjects without changing the frozen backbone.
            grad_y = torch.nn.functional.pad(
                (gt_f[..., 1:, :] - gt_f[..., :-1, :]).abs().mean(dim=2),
                (0, 0, 0, 1))
            grad_x = torch.nn.functional.pad(
                (gt_f[..., :, 1:] - gt_f[..., :, :-1]).abs().mean(dim=2),
                (0, 1, 0, 0))
            saliency = grad_x + grad_y
            if target_frames > 1:
                temporal = torch.nn.functional.pad(
                    (gt_f[:, 1:] - gt_f[:, :-1]).abs().mean(dim=2),
                    (0, 0, 0, 0, 1, 0))
                saliency = saliency + temporal
            flat = saliency.flatten(1)
            scale = flat.mean(dim=1, keepdim=True).clamp_min(1.0e-6)
            saliency = (flat / scale).clamp(max=8.0).view_as(saliency)
            saliency = saliency / saliency.mean(
                dim=(1, 2, 3), keepdim=True).clamp_min(1.0e-6)
            foreground_loss = (
                (x0_pred.float() - gt_f).square().mean(dim=2)
                * saliency.detach()).mean()
            # Optional focused distillation for a persistent reference
            # foreground. Coordinates are normalized [y0,y1,x0,x1] in latent
            # space. This is a training-only supervision mask: inference still
            # runs the unmodified model and receives no prompt/mask shortcut.
            region = getattr(self.args, "self_forcing_reference_region", None)
            if region is not None:
                if len(region) != 4:
                    raise ValueError(
                        "self_forcing_reference_region must be [y0,y1,x0,x1]")
                height, width = gt_f.shape[-2:]
                y0, y1, x0, x1 = [float(value) for value in region]
                iy0 = max(0, min(height - 1, int(round(y0 * height))))
                iy1 = max(iy0 + 1, min(height, int(round(y1 * height))))
                ix0 = max(0, min(width - 1, int(round(x0 * width))))
                ix1 = max(ix0 + 1, min(width, int(round(x1 * width))))
                reference_region_loss = torch.nn.functional.mse_loss(
                    x0_pred.float()[..., iy0:iy1, ix0:ix1],
                    gt_f[..., iy0:iy1, ix0:ix1])
            if teacher_target_x0 is not None:
                teacher_f = teacher_target_x0.float()
                teacher_saliency = (
                    torch.nn.functional.pad(
                        (teacher_f[..., 1:, :] - teacher_f[..., :-1, :])
                        .abs().mean(2), (0, 0, 0, 1))
                    + torch.nn.functional.pad(
                        (teacher_f[..., :, 1:] - teacher_f[..., :, :-1])
                        .abs().mean(2), (0, 1, 0, 0)))
                teacher_saliency = teacher_saliency / teacher_saliency.mean(
                    dim=(1, 2, 3), keepdim=True).clamp_min(1.0e-6)
                teacher_saliency = teacher_saliency.clamp(max=8.0)
                rollout_teacher_identity_loss = (
                    (x0_pred.float() - teacher_f).square().mean(2)
                    * teacher_saliency.detach()).mean()
                rollout_teacher_gradient_loss = (
                    torch.nn.functional.smooth_l1_loss(
                        x0_pred.float()[..., 1:, :] - x0_pred.float()[..., :-1, :],
                        teacher_f[..., 1:, :] - teacher_f[..., :-1, :])
                    + torch.nn.functional.smooth_l1_loss(
                        x0_pred.float()[..., :, 1:] - x0_pred.float()[..., :, :-1],
                        teacher_f[..., :, 1:] - teacher_f[..., :, :-1]))

        loss = (loss
                    + float(getattr(self.args, "action_loss_weight", 1.0))
                    * action_loss
                    + scene_motion_weight * scene_motion_loss
                    + motion_anchor_weight * motion_anchor_loss
                    + camera_direction_weight * camera_direction_loss
                    + float(getattr(self.args, "self_forcing_boundary_weight", 0.05)) * boundary_loss
                    + float(getattr(self.args, "self_forcing_luma_weight", 0.02)) * luma_loss
                    + float(getattr(
                        self.args, "self_forcing_luma_delta_weight", 0.0))
                    * luma_delta_loss
                    + float(getattr(
                        self.args, "self_forcing_luma_velocity_weight", 0.0))
                    * luma_velocity_loss
                    + float(getattr(
                        self.args, "self_forcing_spatial_gradient_weight", 0.0))
                    * spatial_gradient_loss
                    + float(getattr(
                        self.args, "self_forcing_foreground_weight", 0.0))
                    * foreground_loss
                    + float(getattr(
                        self.args, "self_forcing_reference_region_weight", 0.0))
                    * reference_region_loss
                    + float(getattr(
                        self.args, "rollout_teacher_identity_weight", 0.0))
                    * rollout_teacher_identity_loss
                    + float(getattr(
                        self.args, "rollout_teacher_gradient_weight", 0.0))
                    * rollout_teacher_gradient_loss)
        return loss, {
            "x0": clean_latent[:, start:end].detach(),
            "x0_pred": x0_pred.detach(),
            "scheduled_target_block": torch.tensor(
                target_block, device=self.device),
            "self_forcing_boundary_loss": boundary_loss.detach(),
            "rolling_forcing_step": torch.tensor(
                float(rolling_now), device=self.device),
            "action_loss": action_loss.detach(),
            "scene_motion_loss": scene_motion_loss.detach(),
            "motion_anchor_loss": motion_anchor_loss.detach(),
            "camera_direction_loss": camera_direction_loss.detach(),
            "self_forcing_luma_loss": luma_loss.detach(),
            "self_forcing_luma_delta_loss": luma_delta_loss.detach(),
            "self_forcing_luma_velocity_loss": luma_velocity_loss.detach(),
            "self_forcing_spatial_gradient_loss": spatial_gradient_loss.detach(),
            "self_forcing_foreground_loss": foreground_loss.detach(),
            "self_forcing_reference_region_loss": reference_region_loss.detach(),
            "rollout_teacher_identity_loss": rollout_teacher_identity_loss.detach(),
            "rollout_teacher_gradient_loss": rollout_teacher_gradient_loss.detach(),
        }
