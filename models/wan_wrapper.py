import os
import types
from typing import List, Optional
import torch
import re
from torch import nn

from models.scheduler import SchedulerInterface, FlowMatchScheduler
from models.wan_model import WanModel, RegisterTokens, GanAttentionBlock
from models.causal_model import CausalWanModel


def _require_wan22(args=None):
    """Astronex is built on Wan2.2 TI2V-5B; refuse anything else up front."""
    variant = (getattr(args, "backbone_variant", None) if args is not None else None) or "wan2.2"
    if str(variant) != "wan2.2":
        raise ValueError(f"backbone_variant {variant!r} is not supported: "
                         f"Astronex-World runs on wan2.2 only")


def build_text_encoder(args=None):
    """Frozen umT5-xxl encoder from the Wan2.2 release."""
    from models.wan22_components import Wan22TextEncoder
    _require_wan22(args)
    encoder = Wan22TextEncoder()
    encoder.requires_grad_(False)
    return encoder


def build_vae(args=None):
    """Frozen Wan2.2 VAE (``AutoencoderKLWan``).

    fp32 by default, because training uses the VAE to *encode* targets and
    those feed a loss. `vae_dtype: bfloat16` is for inference, where decoding
    was measured at 52.6% of the total -- more than all the denoising steps
    combined -- and nothing downstream of it is differentiated.
    """
    from models.wan22_components import Wan22VAEWrapper
    _require_wan22(args)
    vae_dtype = getattr(args, "vae_dtype", None) if args is not None else None
    kwargs = ({"dtype": getattr(torch, str(vae_dtype))} if vae_dtype else {})
    vae = Wan22VAEWrapper(**kwargs)
    vae.requires_grad_(False)
    return vae


def wan_model_root() -> str:
    """The released weights directory; its `transformer/config.json` defines
    the module. `model_root` in the config's `model_kwargs` takes precedence."""
    return os.environ.get("ASTRONEX_WEIGHTS", "../Astronex")


def filter_state_dict_prope_layers(state_dict, prope_num_layers):
    """Drop PRoPE output projections for layers at or above ``prope_num_layers``.

    The limited-layer AC3D layout equips only ``blocks[0:prope_num_layers]``
    with ``prope_o``.  Checkpoints trained with the legacy all-layer layout
    therefore carry extra ``prope_o`` keys; ``load_state_dict(strict=False)``
    would otherwise report them as unexpected and abort.  This filter is only
    applied when the caller explicitly requests a limited layer count.
    """
    if prope_num_layers is None:
        return state_dict
    prope_num_layers = int(prope_num_layers)
    pattern = re.compile(r"blocks\.(\d+)\.self_attn\.prope_o\.")
    filtered = {}
    for key, value in state_dict.items():
        match = pattern.search(key)
        if match is not None and int(match.group(1)) >= prope_num_layers:
            continue
        filtered[key] = value
    return filtered


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            use_camera=False,
            model_root=None,
            use_action=False,
            action_cross_norm=False,
            action_dim=64,
            num_embodiments=32,
            text_cross_attn_late_scale=1.0,
            text_cross_attn_decay_start=1000000000,
            text_cross_attn_decay_frames=1,
            history_cross_attention=False,
            history_cross_topk=3,
            history_cross_block_frames=4,
            history_cross_rank=64,
            history_cross_active=True,
            history_cross_content_space=False,
            history_cross_scale=1.0,
            history_cross_use_memory=False,
            history_cross_full_memory=False,
            # LongLive routing corrections for small foreground subjects.
            # Defaults match the pre-fix behaviour exactly (pooled routing,
            # no sink boost), so old checkpoints keep bit-identical output
            # until these flags are deliberately turned on.
            history_cross_per_token_routing=False,
            history_cross_tokenwise_blocks=False,
            history_cross_sink_boost=1.0,
            history_cross_contiguous=False,
            history_cross_route_pool_tokens=64,
            history_cross_route_heads=0,
            history_cross_route_topk_frac=0.125,
            history_cross_reference_only=False,
            sink_attention_repeat=1,
            rollout_attention_repeat=1,
            history_attention_repeat=1,
            moba_bidirectional_ratio=0.0,
            camera_film=False,
            camera_film_rank=64,
            camera_film_only=False,
            camera_prope_only=False,
            prope_scale=1.0,
            prope_chunk_anchor=False,
            prope_exclude_sink=False,
            prope_sink_dual=False,
            prope_sink_dual_repeat=1,
            prope_num_layers=None,
            prope_layer_start=0,
            lora=None,
            base_dtype=None,
            control_only=None,
            use_event=False,
            event_num_layers=None,
            event_layer_start=0,
            event_scale=1.0,
            action_output=False,
            action_head_hidden=None,
    ):
        super().__init__()

        root = (model_root or wan_model_root()).rstrip("/")

        # `from_pretrained` upcasts to fp32 unless told otherwise, even when the
        # checkpoint on disk is bf16 -- 21.4 GB for this 5B backbone instead of
        # 10.7.
        #
        # `base_dtype="bfloat16"` halves that and is the right setting for
        # inference and for the eval scripts. It is deliberately NOT used for
        # FSDP training: FSDP flattens each wrapped unit into a single
        # FlatParameter and refuses mixed dtypes ("Must flatten tensors with
        # uniform dtype"), so a bf16 backbone forces the fp32 LoRA masters to
        # bf16 as well -- and at lr 5e-5 against weights of order 1e-2, bf16
        # masters round small updates away entirely. Under FSDP the memory is
        # recovered by sharding instead: fp32 weights split across ranks, with
        # bf16 compute copies materialised one unit at a time.
        load_kwargs = {}
        if base_dtype is not None:
            load_kwargs["torch_dtype"] = (
                getattr(torch, base_dtype) if isinstance(base_dtype, str) else base_dtype)

        # Structure only: `transformer/config.json` defines the module and the
        # released checkpoint supplies every tensor (all 825 backbone tensors
        # plus the control grafts), so loading the original Wan2.2 shards
        # first would read 10 GB only to overwrite it.
        model_dir = f"{root}/{model_name}/"
        if is_causal:
            self.model = CausalWanModel.from_config(
                CausalWanModel.load_config(model_dir),
                local_attn_size=local_attn_size, sink_size=sink_size)
        else:
            self.model = WanModel.from_config(WanModel.load_config(model_dir))
        if "torch_dtype" in load_kwargs:
            self.model.to(load_kwargs["torch_dtype"])
        self.model.eval()

        if is_causal and history_cross_attention:
            self.model.history_cross_reference_only = bool(
                history_cross_reference_only)
            for block in self.model.blocks:
                block.self_attn.enable_history_cross_attention(
                    topk=history_cross_topk,
                    block_frames=history_cross_block_frames,
                    rank=history_cross_rank)
                block.self_attn.history_cross_enabled = bool(history_cross_active)
                block.self_attn.history_cross_content_space = bool(
                    history_cross_content_space)
                block.self_attn.history_cross_scale = float(history_cross_scale)
                block.self_attn.history_cross_use_memory = bool(
                    history_cross_use_memory)
                block.self_attn.history_cross_full_memory = bool(
                    history_cross_full_memory)
                block.self_attn.history_cross_per_token_routing = bool(
                    history_cross_per_token_routing)
                block.self_attn.history_cross_tokenwise_blocks = bool(
                    history_cross_tokenwise_blocks)
                block.self_attn.history_cross_sink_boost = float(
                    history_cross_sink_boost)
                block.self_attn.history_cross_contiguous = bool(
                    history_cross_contiguous)
                block.self_attn.history_cross_route_pool_tokens = int(
                    history_cross_route_pool_tokens)
                block.self_attn.history_cross_route_heads = int(
                    history_cross_route_heads)
                block.self_attn.history_cross_route_topk_frac = float(
                    history_cross_route_topk_frac)
        if is_causal:
            self.model.moba_bidirectional_ratio = float(moba_bidirectional_ratio)
            for block in self.model.blocks:
                block.self_attn.sink_attention_repeat = int(
                    sink_attention_repeat)
                block.self_attn.rollout_attention_repeat = int(
                    rollout_attention_repeat)
                block.self_attn.history_attention_repeat = int(
                    history_attention_repeat)
                block.self_attn.prope_scale = float(prope_scale)
                block.self_attn.prope_chunk_anchor = bool(prope_chunk_anchor)
                block.self_attn.prope_exclude_sink = bool(prope_exclude_sink)
                block.self_attn.prope_sink_dual = bool(prope_sink_dual)
                block.self_attn.prope_sink_dual_repeat = int(prope_sink_dual_repeat)
                block.text_cross_attn_late_scale = float(
                    text_cross_attn_late_scale)
                block.text_cross_attn_decay_start = int(
                    text_cross_attn_decay_start)
                block.text_cross_attn_decay_frames = int(
                    text_cross_attn_decay_frames)
        if is_causal and camera_film:
            for block in self.model.blocks:
                block.enable_camera_film(rank=camera_film_rank)
            self.model.camera_film_enabled = True

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal
        self.use_camera = use_camera
        self.use_action = use_action
        self.action_cross_norm = bool(action_cross_norm)
        self.action_dim = action_dim
        self.num_embodiments = num_embodiments
        self.lora_cfg = dict(lora) if lora else None
        # `control_only` names the control pathways to train *exclusively*;
        # everything else -- backbone included -- stays frozen and no LoRA is
        # applied. See set_trainable for why this is the only way to keep a
        # capability provably intact.
        self.control_only = control_only
        self.use_event = bool(use_event)
        self.event_num_layers = None if event_num_layers is None else int(event_num_layers)
        self.event_layer_start = int(event_layer_start)
        self.event_scale = float(event_scale)
        self.action_output = bool(action_output)
        self.camera_film_only = bool(camera_film_only)
        self.camera_prope_only = bool(camera_prope_only)
        self.prope_num_layers = None if prope_num_layers is None else int(prope_num_layers)
        self.prope_layer_start = int(prope_layer_start)
        self.lora_stats = None

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        if self.action_output:
            self.model.add_action_head(self.action_dim, action_head_hidden)

        self.seq_len = None  # dynamically computed from input shape
        self.post_init()

    # Which parameter names each named control pathway owns. Shared by
    # `train_control_params` (which grafts train alongside LoRA) and by
    # `control_only` (which grafts train *instead of* everything else).
    CONTROL_MARKERS = {
        "action": ("action_embedder",),
        "camera": ("prope_o", "camera_film"),
        "context": ("history_cross_o",),
        "event": ("event_embedding", "event_cross_attn", "norm_event"),
        "action_out": ("action_head",),
    }

    def _resolve_control_markers(self, spec, field):
        """Control-pathway names -> the parameter-name substrings they own."""
        if spec is True:
            return tuple(m for ms in self.CONTROL_MARKERS.values() for m in ms)
        if not spec:
            return ()
        names = [spec] if isinstance(spec, str) else list(spec)
        unknown = [n for n in names if n not in self.CONTROL_MARKERS]
        if unknown:
            raise ValueError(
                f"unknown {field} entries {unknown}; "
                f"expected any of {sorted(self.CONTROL_MARKERS)}, or a bool")
        return tuple(m for n in names for m in self.CONTROL_MARKERS[n])

    def set_trainable(self) -> None:
        """Mark which parameters this wrapper should optimise.

        Call this instead of ``wrapper.model.requires_grad_(True)``. With LoRA
        configured, ``_apply_lora`` has already frozen the backbone and unfrozen
        the adapters and control grafts; a blanket ``requires_grad_(True)``
        afterwards silently reverts all of it, turning a few-hundred-MB adapter
        run into a 5B full fine-tune. That either OOMs -- or, worse, fits, and
        trains something other than what the config says.
        """
        if self.control_only:
            # Train the named grafts and NOTHING else -- no LoRA, no backbone.
            #
            # This is the only setting that makes a capability *provably*
            # untouched rather than probably untouched. Camera control lives in
            # the backbone's attention geometry and in `prope_o`; if neither
            # moves, a camera-only rollout is bit-identical, not merely similar.
            # LoRA does not give that: adapting 300 backbone layers on DROID --
            # whose external cameras are static, so PRoPE sees identity viewmats
            # on every clip -- halved case 171's camera displacement (total_dx
            # 0.121 -> 0.058) while the action path was provably inert at
            # inference, i.e. the damage came entirely from the adapters.
            #
            # The cost is real: a graft trained this way is a readout on frozen
            # features, so it has less to work with. That is the trade this
            # switch exists to make explicit.
            markers = self._resolve_control_markers(self.control_only, "control_only")
            # _apply_lora has already unfrozen the adapters and the grafts named
            # by train_control_params; this reverses all of it before selecting.
            self.model.requires_grad_(False)
            found = 0
            for name, param in self.model.named_parameters():
                if any(m in name for m in markers):
                    param.requires_grad_(True)
                    found += 1
            if found == 0:
                raise RuntimeError(
                    f"control_only={self.control_only} matched no parameters; "
                    "the graft it names was never installed (check use_camera / "
                    "use_action / action_output / use_event in model_kwargs)")
            self._cast_trainable_to_fp32()
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            lora_live = sum(p.requires_grad for n, p in self.model.named_parameters()
                            if "lora" in n)
            print(f"[control-only] {self.control_only}: {found} tensors, "
                  f"{trainable / 1e6:.1f}M trainable (fp32); backbone frozen "
                  f"({lora_live} LoRA tensors trainable, must be 0)")
            assert lora_live == 0, "control_only left LoRA adapters trainable"
            return
        if self.camera_film_only or self.camera_prope_only:
            self.model.requires_grad_(False)
            found = 0
            for name, param in self.model.named_parameters():
                selected = ("camera_film" in name if self.camera_film_only
                            else "prope_o" in name)
                if selected:
                    param.requires_grad_(True)
                    found += 1
            if found == 0:
                branch = "camera_film" if self.camera_film_only else "prope_o"
                raise RuntimeError(f"requested exclusive training but no {branch} parameters exist")
            self._cast_trainable_to_fp32()
            return
        if self.lora_cfg is not None:
            return
        self.model.requires_grad_(True)

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def adding_cls_branch(self, atten_dim=None, num_class=1, time_embed_dim=0,
                          num_registers=3, hidden=1536, mode="pooled") -> None:
        """Attach the GAN discriminator head used alongside the DMD loss.

        Three register tokens read the backbone at blocks 13, 21 and 29 through
        cross-attention, and the head maps their concatenation to `num_class`
        logits. `model.py` already implements that path under `classify_mode`;
        this builds the modules it asserts on.

        Dimensions come from the instantiated backbone rather than a constant.
        The previous signature hard-coded Wan2.1-1.3B's 1536 and, worse, gave
        the output layer `in_features=atten_dim` when the layer before it emits
        `hidden` -- which only happened to work because 1.3B's dim equals the
        hidden width. On Wan2.2-5B (dim 3072, 24 heads) it would have raised on
        the first forward.
        """
        dim = int(atten_dim if atten_dim is not None else self.model.dim)
        heads = int(getattr(self.model, "num_heads", 12))
        in_features = dim * num_registers + time_embed_dim
        device = next(self.model.parameters()).device

        self._cls_pred_branch = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_class),
        ).to(device)
        self._cls_pred_branch.requires_grad_(True)

        # Two ways to reduce a layer's sequence to one vector.
        #
        # `attn` is the register-token version this file already had: a learned
        # query cross-attends the whole sequence at each tapped layer. It is
        # also 278.5M parameters on a 5B backbone -- 1.1 GiB of weights, 1.1 of
        # gradients and 2.2 of AdamW state -- which measured as 18 GiB on the
        # critic phase's peak and left two gigabytes of headroom on this card.
        # At the learning rate a freshly initialised head of that size needs,
        # it also swung `gan_d_loss` from 1.39 to 2.58 in two steps.
        #
        # `pooled` mean-pools instead. No parameters per tap, ~14M in total,
        # and for telling noisy texture from clean it is the right size of
        # question: the signal is a statistic of the whole frame, not something
        # a learned query has to go find.
        self.gan_head_kind = str(mode)
        if mode == "attn":
            self._register_tokens = RegisterTokens(
                num_registers=num_registers, dim=dim).to(device)
            self._register_tokens.requires_grad_(True)
            self._gan_ca_blocks = nn.ModuleList([
                GanAttentionBlock(dim=dim, num_heads=heads)
                for _ in range(num_registers)
            ]).to(device)
            self._gan_ca_blocks.requires_grad_(True)
        elif mode == "pooled":
            self._register_tokens = None
            self._gan_ca_blocks = None
        else:
            raise ValueError(f"unknown gan head mode {mode!r}")

        # The head trains in fp32 for the same reason the LoRA adapters do: it
        # is what the optimiser updates, and bf16 masters round lr-sized steps
        # away. `_cast_trainable_to_fp32` covers the adapters; these modules are
        # added after it has run.
        mods = [m for m in (self._cls_pred_branch, self._register_tokens,
                            self._gan_ca_blocks) if m is not None]
        for mod in mods:
            for prm in mod.parameters():
                prm.data = prm.data.float()

        n = sum(p.numel() for mod in mods for p in mod.parameters())
        print(f"[GAN] discriminator head: {mode}, dim={dim}, "
              f"{num_class} logit(s), {n / 1e6:.1f}M trainable (fp32)",
              flush=True)

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,

        classify_mode: Optional[bool] = False, # DF
        concat_time_embeddings: Optional[bool] = False, #DF
        clean_x: Optional[torch.Tensor] = None, # TF
        aug_t: Optional[torch.Tensor] = None, # for TF clean GT, if it's also noisy and needs denoising by the model, aug_t is its timestep

        cache_start: Optional[int] = None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        prope_kv_cache: Optional[List[dict]] = None,
        prope_window_viewmats: Optional[torch.Tensor] = None,
        prope_window_Ks: Optional[torch.Tensor] = None,
        replace_sink: bool = False,
        cache_keep_frames=None,
        cache_sink_frames=None,
        return_actions: bool = False,
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]
        event_context = conditional_dict.get("event_embeds", None)
        return_actions = bool(
            return_actions and self.action_output
            and getattr(self.model, "action_head", None) is not None)

        # Actions ride in conditional_dict rather than in the signature, the way
        # viewmats/Ks do for the camera models, so the DMD/ODE/CD code paths
        # forward them for free -- including into the unconditional branch,
        # which is what we want: CFG should vary the *text*, not the control
        # signal. Guidance applied to actions would fight the distillation.
        actions = conditional_dict.get("actions", None)
        embodiment_id = conditional_dict.get("embodiment_id", None)
        if actions is not None and actions.shape[1] != noisy_image_or_video.shape[1]:
            if current_start is None:
                raise ValueError("full-length actions require current_start for causal slicing")
            frame_tokens = ((noisy_image_or_video.shape[-2] // self.model.patch_size[1])
                            * (noisy_image_or_video.shape[-1] // self.model.patch_size[2]))
            start_frame = int(current_start) // frame_tokens
            actions = actions[:, start_frame:start_frame + noisy_image_or_video.shape[1]]

        if self.uniform_timestep:
            if timestep.dim() == 2 and timestep.shape[1] > 1 and (
                    timestep[:, 1:] != timestep[:, :1]).any():
                # TI2V: the first frame is conditioned at t=0 while the rest
                # denoise at t; keep the per-frame signal instead of collapsing.
                input_timestep = timestep
            else:
                input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        # Dynamically compute seq_len from input: [B, F, C, H, W]
        # After patch_embedding (2x2 spatial), tokens = F * (H/2) * (W/2)
        _, F, _, H, W = noisy_image_or_video.shape
        seq_len = F * (H // 2) * (W // 2)

        logits = None

        # X0 prediction
        if kv_cache is not None:
            model_out = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                viewmats=viewmats,
                Ks=Ks,
                prope_kv_cache=prope_kv_cache,
                prope_window_viewmats=prope_window_viewmats,
                prope_window_Ks=prope_window_Ks,
                replace_sink=replace_sink,
                cache_keep_frames=cache_keep_frames,
                cache_sink_frames=cache_sink_frames,
                actions=actions,
                embodiment_id=embodiment_id,
                event_context=event_context,
                return_actions=return_actions,
            )
            if return_actions:
                flow_pred, action_pred = model_out
            else:
                flow_pred = model_out
            flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                # teacher forcing
                model_out = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4), # => [B, C, F, H, W]
                    t=input_timestep, context=prompt_embeds,
                    seq_len=seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4), # => [B, C, F, H, W]
                    aug_t=aug_t,
                    viewmats=viewmats,
                    Ks=Ks,
                    actions=actions,
                    embodiment_id=embodiment_id,
                    event_context=event_context,
                    return_actions=return_actions,
                )
                if return_actions:
                    flow_pred, action_pred = model_out
                else:
                    flow_pred = model_out
                flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
            else:
                # diffusion forcing or bidirectional
                if classify_mode:
                    model_out = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings,
                        viewmats=viewmats,
                        Ks=Ks,
                        actions=actions,
                        embodiment_id=embodiment_id,
                        event_context=event_context,
                        return_actions=return_actions,
                    )
                    if return_actions:
                        flow_pred, logits, action_pred = model_out
                    else:
                        flow_pred, logits = model_out
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    model_out = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=seq_len,
                        viewmats=viewmats,
                        Ks=Ks,
                        actions=actions,
                        embodiment_id=embodiment_id,
                        event_context=event_context,
                        return_actions=return_actions,
                    )
                    if return_actions:
                        flow_pred, action_pred = model_out
                    else:
                        flow_pred = model_out
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits

        if return_actions:
            return flow_pred, pred_x0, action_pred
        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()

        # Add PRoPE parameters if camera control is enabled
        if self.use_camera:
            from models.prope import add_prope_parameters
            add_prope_parameters(
                self.model,
                layer_start=self.prope_layer_start,
                layer_end=self.prope_num_layers,
            )

        # Add the action embedder if action control is enabled. Both grafts
        # happen after from_pretrained so the base checkpoint loads strictly,
        # and both are zero-initialised so the extended model starts out
        # numerically identical to pretrained Wan.
        if self.use_action:
            from models.action import add_action_parameters
            # Set before the graft: add_action_parameters only creates the gate
            # when this is on, and apply_action_modulation only uses it when
            # both exist. Off, the pathway is the raw additive one every
            # checkpoint before this was trained with.
            self.model.action_cross_norm = self.action_cross_norm
            add_action_parameters(
                self.model,
                action_dim=self.action_dim,
                num_embodiments=self.num_embodiments,
            )

        # Independent event-prompt cross-attention. Grafted here for the same
        # reason as the two above: after from_pretrained, zero-initialised at
        # the output projection, so enabling it costs nothing until it is
        # trained.
        if self.use_event:
            self.model.enable_event_conditioning(
                num_layers=self.event_num_layers,
                layer_start=self.event_layer_start)
            self.model.set_event_scale(self.event_scale)

        # LoRA is still *applied* under control_only, and deliberately so: the
        # stage-1 checkpoints in this repo are `--keep-lora` exports carrying
        # `.base.weight` / `.lora_a` keys, so a model without the adapter
        # structure cannot load them at all. What control_only changes is that
        # set_trainable then freezes the adapters along with everything else --
        # they keep their loaded values and never update, so the backbone's
        # effective weights are constant and the frozen capability stays
        # bit-identical. Structure and trainability are separate questions.
        if self.lora_cfg is not None:
            self._apply_lora()

    # Names of the linear layers LoRA adapts in Wan's original module layout.
    # The adapter's generic defaults describe a different DiT and match nothing here,
    # and `apply_lora` raises on an empty match rather than silently adapting
    # zero layers -- which is the behaviour we want, but only once the patterns
    # are right.
    LORA_TARGETS = (
        "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
        "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
        "ffn.0", "ffn.2",
    )
    LORA_EXCLUDE = ("head", "text_embedding", "time_embedding", "time_projection",
                    "modulation", "action_embedder", "prope_o",
                    # "event_cross_attn.q" contains "cross_attn.q", so without
                    # this the event branch would be silently LoRA-adapted --
                    # a low-rank update of a zero matrix, i.e. crippled before
                    # it starts. It trains in full, like the other grafts.
                    "event_cross_attn", "norm_event")

    def _apply_lora(self):
        """Freeze the backbone and train low-rank adapters instead.

        A 5B student under AdamW needs roughly 60 GB of master weights and
        moments before a single activation is stored; two of those (generator
        and critic) do not fit on any pair of 48 GB cards. LoRA turns that into
        a few hundred MB per model, which is the difference between this stage
        running on the available hardware and not running at all.

        The grafted control parameters are deliberately exempted. They are new,
        zero-initialised and tiny, and constraining brand-new camera and action
        pathways to a low-rank update of a zero matrix would cripple exactly the
        capability these stages exist to add.
        """
        from models.lora import LoRAConfig, apply_lora

        cfg = dict(self.lora_cfg)
        cfg.setdefault("target_patterns", self.LORA_TARGETS)
        cfg.setdefault("exclude_patterns", self.LORA_EXCLUDE)
        cfg.setdefault("adapters", ("default",))
        cfg["target_patterns"] = tuple(cfg["target_patterns"])
        cfg["exclude_patterns"] = tuple(cfg["exclude_patterns"])
        cfg["adapters"] = tuple(cfg["adapters"])
        train_control = cfg.pop("train_control_params", True)
        train_adapters = cfg.pop("train_adapter_patterns", None)

        _, stats = apply_lora(self.model, LoRAConfig(**cfg))
        self.lora_stats = stats

        # Keep the complete adapter structure for strict checkpoint loading,
        # while optionally optimising only the pathways relevant to this
        # stage. Rolling long-horizon adaptation primarily changes temporal
        # self-attention; freezing cross-attention and FFN adapters preserves
        # prompt semantics and removes their Adam states from the 48-GB peak.
        if train_adapters:
            patterns = ((train_adapters,) if isinstance(train_adapters, str)
                        else tuple(train_adapters))
            for name, param in self.model.named_parameters():
                if "lora_" in name and not any(p in name for p in patterns):
                    param.requires_grad_(False)

        # `train_control_params` selects which grafted control pathways train:
        # True/False for both, or a list naming them ("action", "camera").
        #
        # The two have to be separable. DROID teaches action but its external
        # cameras are static -- `camera_extrinsics` has std exactly 0 across
        # time -- so PRoPE sees identity viewmats on every clip. Training it
        # there does not teach camera control, it lets weights that already
        # encode camera control drift against a constant input. Stage 0 on
        # this model went `w*19 vs h*19` from 6.15 to 27.27; that is the asset this
        # switch exists to protect.
        markers = self._resolve_control_markers(train_control, "train_control_params")

        if markers:
            for name, param in self.model.named_parameters():
                if any(m in name for m in markers):
                    param.requires_grad_(True)

        self._cast_trainable_to_fp32()

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_dtype = next(
            (p.dtype for p in self.model.parameters() if not p.requires_grad), None)
        print(f"[LoRA] adapted {int(stats['adapted_layers'])} layers, "
              f"{trainable / 1e6:.1f}M trainable (fp32) of "
              f"{stats['base_params_b']:.2f}B frozen ({frozen_dtype})")

    def _cast_trainable_to_fp32(self):
        """Keep the optimised parameters in fp32 whatever the backbone dtype is.

        The backbone is loaded in bf16 to halve its footprint, but the adapters
        and the zero-initialised control grafts are what the optimiser updates,
        and bf16 master weights lose small updates outright: at lr 5e-5 against
        weights of order 1e-2, the update falls below bf16's resolution and
        rounds to nothing. The run then looks healthy and learns nothing.
        """
        for param in self.model.parameters():
            if param.requires_grad and param.dtype != torch.float32:
                param.data = param.data.float()
