import os
import time
from tqdm import tqdm
from typing import List, Optional
import torch
import torch.nn.functional as F

from models.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from models.fm_solvers_unipc import FlowUniPCMultistepScheduler
from models.prope import anchor_viewmats_to_frame
from models.wan_wrapper import (
    WanDiffusionWrapper,
    build_text_encoder,
    build_vae,
)


def _clone_cache(cache):
    """Fresh cache set with the same structure and zeroed tensors."""
    if cache is None:
        return None
    return [{k: (v.clone() if torch.is_tensor(v) else v) for k, v in blk.items()}
            for blk in cache]


class CausalDiffusionInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None,
            need_vae = True
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        # Select the tokenizer/text encoder by `backbone_variant` instead of
        # hardcoding the 2.1 pair. This path was constructing a Wan2.1 text
        # encoder unconditionally, so a 2.2 causal checkpoint could not be
        # sampled at all -- it failed looking for a 2.1 weights file. The
        # bidirectional pipeline has always used the builders; this brings the
        # causal one in line.
        self.text_encoder = (
            build_text_encoder(args) if text_encoder is None else text_encoder)
        if need_vae:
            self.vae = build_vae(args) if vae is None else vae

        # Step 2: Initialize scheduler
        self.num_train_timesteps = args.num_train_timestep
        # 50 is this pipeline's original constant, kept as the default. It is
        # right for a model that was never distilled; our causal student was
        # collapsed to four steps by CF++ and DMD, and running it for 50 takes
        # it far off that distribution -- case 171 comes back with every book
        # spine etched, high-frequency energy 10.2 against the 50-step
        # bidirectional teacher's 4.9. Fewer steps sit closer to what the
        # student was trained to do.
        self.sampling_steps = int(getattr(args, "sampling_steps", 50))
        self.sample_solver = 'unipc'
        self.shift = args.timestep_shift
        # Ported from causal_inference.py, where it has always existed and this
        # pipeline never had it -- so every UniPC run so far has been without it.
        # See the anchoring block below for what it does and why.
        self.prope_pin_sink = bool(getattr(args, "prope_pin_sink", False))
        # Classifier-free guidance on the CAMERA condition, separate from the
        # text CFG above it.
        #
        # prope_scale was the wrong operator for this. It multiplies the whole
        # PRoPE branch output -- geometry and the content that branch writes --
        # back into the trunk, so past ~2.0 the camera branch overwhelms the
        # content branch and the direction difference is buried rather than
        # amplified: on the one checkpoint that actually has direction, case 171
        # W-S went +0.326 at scale 1 to +0.035 at 2 to +0.004 at 3, while the
        # foreground fell to 0.43 of its sharpness.
        #
        # This extrapolates the DIFFERENCE between a camera-conditioned and a
        # camera-neutral prediction, so only what the camera changes is scaled
        # and the content prediction is subtracted out. 0 disables it and costs
        # nothing; above 0 it adds one forward pass per step.
        self.camera_guidance_scale = float(getattr(args, "camera_guidance_scale", 0.0))
        self.kv_cache_cam = None
        self.crossattn_cache_cam = None
        self.prope_kv_cache_cam = None

        # These were hardcoded for Wan2.1-1.3B: 30 blocks, 12 heads of 128, and
        # 1560 tokens per frame (60x104 latent, patch 2). Wan2.2-TI2V-5B has a
        # 16x-downsampling VAE and a wider trunk, so every one of those numbers
        # is wrong for it, and the wrongness surfaces as a shape error deep in
        # the KV-cache write rather than anywhere near the cause. Read them off
        # the model that was actually built.
        model = self.generator.model
        self.num_transformer_blocks = len(model.blocks)
        self.model_num_heads = model.num_heads
        self.model_head_dim = model.dim // model.num_heads
        self.text_len = getattr(model, "text_len", 512)
        shape = getattr(args, "image_or_video_shape", None)
        if shape is not None:
            self.frame_seq_length = ((shape[3] // model.patch_size[1])
                                     * (shape[4] // model.patch_size[2]))
        else:
            self.frame_seq_length = 1560

        self.kv_cache_pos = None
        self.kv_cache_neg = None
        self.crossattn_cache_pos = None
        self.crossattn_cache_neg = None
        self.prope_kv_cache_pos = None
        self.prope_kv_cache_neg = None
        self._prope_mirror = []
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size
        self.retrieval_window = bool(getattr(args, "retrieval_window", False))
        self.retrieval_recent_frames = int(getattr(args, "retrieval_recent_frames", 4))
        self.retrieval_min_overlap = float(getattr(args, "retrieval_min_overlap", 0.3))
        self.retrieval_swap_per_block = int(getattr(args, "retrieval_swap_per_block", 2))
        self._prev_retrieved = []
        self._cache_keep = None
        self._sink_promote = None
        self.sink_refresh = bool(getattr(args, "sink_refresh", False))
        self.sink_refresh_min_overlap = float(getattr(
            args, "sink_refresh_min_overlap", 0.15))
        self.ref_sequence_conditioning = bool(getattr(
            args, "ref_sequence_conditioning", False))

        # Latency of producing the first chunk (set by inference()).
        self.last_chunk0_latency = None

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    _CACHES = ("kv_cache_pos", "kv_cache_neg", "kv_cache_cam",
               "crossattn_cache_pos", "crossattn_cache_neg", "crossattn_cache_cam",
               "prope_kv_cache_pos", "prope_kv_cache_neg", "prope_kv_cache_cam")

    def _release_denoiser_for_decode(self):
        """Make room for the VAE decode on a small card.

        The decode is the memory peak of a rollout: the denoiser and its KV
        caches are still resident when the whole clip goes through the VAE.
        `free_cache_before_decode` drops the caches -- the next rollout
        re-initialises them, so nothing is lost. `offload_denoiser_for_decode`
        also moves the denoiser to the host for the decode; the caller moves
        it back. Returns the device to restore the denoiser to, or None.
        """
        released = False
        if bool(getattr(self.args, "free_cache_before_decode", False)):
            for name in self._CACHES:
                setattr(self, name, None)
            released = True
        device = None
        if bool(getattr(self.args, "offload_denoiser_for_decode", False)):
            device = next(self.generator.parameters()).device
            self.generator.to("cpu")
            released = True
        if released:
            torch.cuda.empty_cache()
        return device

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        start_frame_index: Optional[int] = 0,
        return_video=True,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        event_prompts: Optional[List[str]] = None,
        event_start_frame: int = 0,
        actions: Optional[torch.Tensor] = None,
        embodiment_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
            start_frame_index (int): In long video generation, where does the current window start?
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        self._prope_mirror = []
        self._prev_retrieved = []
        self._cache_keep = None
        sequence_ref = self.ref_sequence_conditioning and initial_latent is not None
        if sequence_ref:
            if self.independent_first_frame:
                raise ValueError(
                    "ref_sequence_conditioning requires independent_first_frame: false")
            if num_input_frames != 1:
                raise ValueError(
                    "ref_sequence_conditioning expects exactly one reference latent; "
                    "do not use --cond_repeat")
            if (num_frames + 1) % self.num_frame_per_block != 0:
                raise ValueError(
                    "reference + generated latent frames must form complete blocks: "
                    f"got 1 + {num_frames} with block size {self.num_frame_per_block}")

        # Start chunk0 latency timer 
        if sequence_ref:
            num_blocks = (num_frames + 1) // self.num_frame_per_block
        elif not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        elif self.independent_first_frame and initial_latent is None:
            # Using a [1, 4, 4, 4, 4, 4] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        from inference.pipelines.event_prompt import add_event_embeds, merge_event_into_captions
        event_in_caption = bool(getattr(self.args, "event_in_caption", False))
        event_start_frame = int(event_start_frame or 0)
        # `event_start_frame` holds the event back until the rollout reaches it,
        # so the caption establishes the scene and the event lands mid-video.
        # Step 3 below denoises one chunk at a time, so the switch is per chunk
        # and the chunks before it are what an event-free run would produce.
        # Both captions are encoded up front: the text encoder is the expensive
        # part and the schedule only ever moves one way.
        conditioned_prompts = (merge_event_into_captions(text_prompts, event_prompts)
                               if event_in_caption else text_prompts)
        scheduled_caption = bool(
            event_in_caption and event_prompts and event_start_frame)
        conditional_dict = self.text_encoder(
            text_prompts=(text_prompts if scheduled_caption else conditioned_prompts)
        )
        event_caption_dict = (self.text_encoder(text_prompts=conditioned_prompts)
                              if scheduled_caption else None)
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * len(text_prompts)
        )
        # The independent branch is optional.  With event_in_caption enabled a
        # plain pretrained/base model can handle open-domain events without an
        # event overlay; when the branch exists both controls are combined.
        model_kwargs = getattr(self.args, "model_kwargs", {})
        use_event_branch = bool(
            model_kwargs.get("use_event", False) if hasattr(model_kwargs, "get")
            else getattr(model_kwargs, "use_event", False))
        if use_event_branch:
            add_event_embeds(conditional_dict, self.text_encoder,
                             event_prompts, len(text_prompts))
        elif event_prompts and not event_in_caption:
            raise ValueError(
                "an event prompt requires model_kwargs.use_event or "
                "event_in_caption: true")
        # An explicit action stream wins over the camera-derived plan: the plan
        # is a description of the trajectory PRoPE already carries, while these
        # are the commands the embedder was trained on. The wrapper slices a
        # full-length stream per chunk off current_start, so it is passed whole.
        if actions is not None:
            conditional_dict["actions"] = actions
            unconditional_dict["actions"] = actions
            if embodiment_id is not None:
                conditional_dict["embodiment_id"] = embodiment_id
                unconditional_dict["embodiment_id"] = embodiment_id
        elif getattr(self.args, "camera_plan_condition", False) and viewmats is not None:
            from models.action import camera_plan_actions
            action_dim = int(getattr(self.args, "model_kwargs", {}).get("action_dim", 64))
            plan = camera_plan_actions(viewmats, action_dim)
            conditional_dict["actions"] = plan
            unconditional_dict["actions"] = plan

        # Held off the dict and re-attached per chunk rather than sliced: the
        # event branch is all-or-nothing per forward, and popping the key is the
        # exact computation an event-free run performs.
        scheduled_embeds = (conditional_dict.pop("event_embeds", None)
                            if event_start_frame else None)
        event_caption_applied = False

        def apply_event(start_frame):
            """Attach the event to the chunk beginning at `start_frame`."""
            nonlocal event_caption_applied
            if not event_start_frame:
                return
            active = start_frame >= event_start_frame
            if scheduled_embeds is not None:
                if active:
                    conditional_dict["event_embeds"] = scheduled_embeds
                else:
                    conditional_dict.pop("event_embeds", None)
            if event_caption_dict is not None and active and not event_caption_applied:
                # The caption's cross-attention K/V is computed on the first
                # forward and reused for every later chunk, so swapping the
                # embedding alone would change nothing -- the positive cache has
                # to be invalidated too. The negative prompt never changes, so
                # its cache still stands.
                conditional_dict.update(event_caption_dict)
                for block_index in range(self.num_transformer_blocks):
                    self.crossattn_cache_pos[block_index]["is_init"] = False
                event_caption_applied = True
                print(f"[event] caption switched at latent frame {start_frame}",
                      flush=True)

        # Start chunk0 latency timer AFTER text encoder, BEFORE VAE decode
        # — matches the reported latency definition (excludes both text encoder and decode).
        torch.cuda.synchronize()
        _chunk0_t0 = time.perf_counter()
        self.last_chunk0_latency = None

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache_pos is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            if viewmats is not None:
                self._initialize_prope_kv_cache(
                    batch_size=batch_size,
                    dtype=noise.dtype,
                    device=noise.device
                )
            if self.camera_guidance_scale > 0:
                # A third rollout needs its own caches: it sees different
                # viewmats from the other two, so sharing any of them would
                # feed each branch the other's history.
                self.kv_cache_cam = _clone_cache(self.kv_cache_neg)
                self.crossattn_cache_cam = _clone_cache(self.crossattn_cache_neg)
                self.prope_kv_cache_cam = (_clone_cache(self.prope_kv_cache_neg)
                                           if self.prope_kv_cache_neg is not None else None)
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False
                if self.crossattn_cache_cam is not None:
                    self.crossattn_cache_cam[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache_pos)):
                self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_neg[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_neg[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                if self.kv_cache_cam is not None:
                    self.kv_cache_cam[block_index]["global_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                    self.kv_cache_cam[block_index]["local_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                for cache in ([self.kv_cache_pos[block_index],
                               self.kv_cache_neg[block_index]] +
                              ([self.kv_cache_cam[block_index]]
                               if self.kv_cache_cam is not None else [])):
                    cache["memory_ref_k"] = None
                    cache["memory_ref_v"] = None
                    cache["memory_summary_k"] = None
                    cache["memory_summary_v"] = None
                    cache["history_full_k"] = None
                    cache["history_full_v"] = None
            # reset prope kv cache
            if viewmats is not None and self.prope_kv_cache_pos is not None:
                for block_index in range(len(self.prope_kv_cache_pos)):
                    self.prope_kv_cache_pos[block_index]["global_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                    self.prope_kv_cache_pos[block_index]["local_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                    self.prope_kv_cache_neg[block_index]["global_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                    self.prope_kv_cache_neg[block_index]["local_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                    if self.prope_kv_cache_cam is not None:
                        self.prope_kv_cache_cam[block_index]["global_end_index"] = torch.tensor(
                            [0], dtype=torch.long, device=noise.device)
                        self.prope_kv_cache_cam[block_index]["local_end_index"] = torch.tensor(
                            [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = start_frame_index
        cache_start_frame = 0
        if initial_latent is not None and not sequence_ref:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                vm_slice = viewmats[:, current_start_frame:current_start_frame + 1] if viewmats is not None else None
                ks_slice = Ks[:, current_start_frame:current_start_frame + 1] if Ks is not None else None
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    viewmats=vm_slice,
                    Ks=ks_slice,
                    prope_kv_cache=self.prope_kv_cache_pos
                )
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    viewmats=vm_slice,
                    Ks=ks_slice,
                    prope_kv_cache=self.prope_kv_cache_neg
                )
                current_start_frame += 1
                cache_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for block_index in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, cache_start_frame:cache_start_frame + self.num_frame_per_block]
                output[:, cache_start_frame:cache_start_frame + self.num_frame_per_block] = current_ref_latents
                vm_chunk = viewmats[:, current_start_frame:current_start_frame + self.num_frame_per_block] if viewmats is not None else None
                ks_chunk = Ks[:, current_start_frame:current_start_frame + self.num_frame_per_block] if Ks is not None else None
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    viewmats=vm_chunk,
                    Ks=ks_chunk,
                    prope_kv_cache=self.prope_kv_cache_pos
                )
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    viewmats=vm_chunk,
                    Ks=ks_chunk,
                    prope_kv_cache=self.prope_kv_cache_neg
                )
                current_start_frame += self.num_frame_per_block
                cache_start_frame += self.num_frame_per_block

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_num_frames in all_num_frames:
            apply_event(current_start_frame)
            is_reference_block = sequence_ref and cache_start_frame == 0
            if is_reference_block:
                # No prefix has been consumed from `noise`: the reference is
                # external and occupies output slot zero, so take the first
                # block-1 generated frames directly.  Using the generic
                # prefix-offset formula here produces the Python slice -1:3,
                # i.e. an empty tensor.
                noisy_input = noise[:, :current_num_frames - 1]
                reference_latent = initial_latent[:, :1].to(noisy_input.dtype)
                noisy_input = torch.cat([reference_latent, noisy_input], dim=1)
            else:
                noisy_input = noise[
                    :, cache_start_frame - num_input_frames:cache_start_frame + current_num_frames - num_input_frames]
            latents = noisy_input

            # Slice viewmats/Ks for the current chunk.  When the backbone was
            # trained with ``prope_chunk_anchor``, the camera branch stores RAW
            # (gauge-free) K/V and re-projects the whole resident window under
            # the previous chunk's last frame at every call.  Replicate that
            # window here so inference reads the same bounded, re-anchored
            # history distribution the adapter was trained on -- the ordinary
            # unbounded ``history_full_k`` path is NOT what training saw.
            prope_window_viewmats = None
            prope_window_Ks = None
            if (viewmats is not None and self.prope_kv_cache_pos is not None
                    and getattr(
                        self.generator.model.blocks[0].self_attn,
                        "prope_chunk_anchor", False)):
                # Bounded sliding window mirroring the training cache: keep the
                # persistent sink frame and the most recent frames up to the
                # local-attention window size.
                cache_cap = max(1, int(self.local_attn_size))
                prope_sink_frames = int(getattr(
                    self.generator.model.blocks[0].self_attn, "sink_size", 1))
                new_frames = list(range(
                    current_start_frame, current_start_frame + current_num_frames))
                combined = list(self._prope_mirror) + new_frames
                self._cache_keep = None
                # Reset per call: a promotion left over from the previous block
                # skipped the normal truncation and let the window grow past
                # the cache.
                self._sink_promote = None
                if len(combined) > cache_cap:
                    evict = len(combined) - cache_cap
                    sink = max(0, min(prope_sink_frames, len(combined)))
                    if getattr(self, "retrieval_window", False):
                        # Keep what this block is about to look at, not what is
                        # merely recent. Age-based eviction is why a corridor
                        # walked forty frames ago cannot be redrawn on the way
                        # back: the frames that saw it are already gone. The
                        # newest few stay pinned regardless -- Memory Forcing
                        # reports that leaning wholly on spatial memory hurts
                        # genuinely new scenery, and those are the temporal
                        # half of that trade.
                        from post_train.objectives.camera_diffusion import _frustum_overlap
                        body = combined[sink:len(combined) - len(new_frames)]
                        evictable = list(body)
                        from utils.cache_selection import select_history_frames
                        keep_n = cache_cap - sink - len(new_frames)
                        recent = min(keep_n, max(0, int(self.retrieval_recent_frames)))
                        candidates = body[:-recent] if recent else body
                        room = keep_n - recent
                        scores = (_frustum_overlap(viewmats, Ks, new_frames[-1], candidates)
                                  if room and candidates else {})

                        body, self._prev_retrieved = select_history_frames(
                            body, keep_n, recent, scores,
                            float(getattr(self, "retrieval_min_overlap", 0.3)),
                            self._prev_retrieved,
                            int(getattr(self, "retrieval_swap_per_block", 2)))
                        if os.environ.get("WAN_CACHE_TRACE") == "1":
                            print(f"[cache-selection] start={new_frames[0]} room={room} "
                                  f"keep={body} new={new_frames}", flush=True)
                        # Refresh the sink once it stops showing anything the
                        # current view can see.
                        #
                        # The sink is pinned to frames 0..3 for the life of the
                        # clip, and window-relative RoPE deliberately fixes the
                        # sink->query distance so the reference never decays
                        # out of attention. That is exactly the problem once
                        # the camera has walked away: measured on case 73 the
                        # sink's content correlation falls under 0.3 by frame
                        # 48, and the visible damage follows about a hundred
                        # frames later, which is error accumulating with no
                        # valid anchor left to correct it.
                        #
                        # Dropping the sink entirely was tried and is far
                        # worse -- it hallucinates gold objects and colour
                        # noise -- and so is giving the sink the current pose,
                        # which washes the tail out. So the content moves
                        # instead: a frame that is still visible takes over the
                        # slots. It is drawn from the frames this step was
                        # about to evict anyway, which keeps every slot count
                        # unchanged and costs nothing.
                        if self.sink_refresh and sink > 0 and viewmats is not None:
                            held = combined[:sink]
                            sc_held = _frustum_overlap(
                                viewmats, Ks, new_frames[-1], held)
                            if max(sc_held.values()) < self.sink_refresh_min_overlap:
                                pool = [f for f in evictable if f not in body]
                                if len(pool) >= sink:
                                    sc_pool = _frustum_overlap(
                                        viewmats, Ks, new_frames[-1], pool)
                                    best = sorted(
                                        pool, key=lambda f: -sc_pool[f])[:sink]
                                    if max(sc_pool[f] for f in best) > \
                                            max(sc_held.values()):
                                        self._sink_promote = sorted(best)
                                        combined = (self._sink_promote + body
                                                    + new_frames)
                                        if os.environ.get("WAN_CACHE_TRACE") == "1":
                                            print(f"[sink-refresh] start={new_frames[0]} "
                                                  f"old={held} new={self._sink_promote}",
                                                  flush=True)
                        if self._sink_promote is None:
                            combined = combined[:sink] + body + new_frames
                        self._cache_keep = list(body)
                    else:
                        combined = combined[:sink] + combined[sink + evict:]
                self._prope_mirror = list(combined)
                anchor_frame = max(0, current_start_frame - 1)
                if anchor_frame not in self._prope_mirror:
                    anchor_frame = self._prope_mirror[0]
                anchor_idx = self._prope_mirror.index(anchor_frame)
                vm_window = anchor_viewmats_to_frame(
                    viewmats[:, self._prope_mirror], anchor_idx)
                if self.prope_pin_sink:
                    # Give the persistent sink frames the anchor's pose instead
                    # of their own.
                    #
                    # PRoPE encodes the RELATIVE camera transform between the
                    # querying chunk and each cached frame. The window is
                    # anchored on the previous frame, but the sink keeps its
                    # true pose from the start of the clip, so as the camera
                    # walks the sink->current transform grows without bound and
                    # the geometry it encodes runs off the end of anything the
                    # model was trained on.
                    #
                    # This is the same failure Rolling Forcing (arXiv 2509.25161)
                    # fixes on the RoPE side by placing global-context frames
                    # immediately before the temporal context, "freezing
                    # relative distance to denoising frames and preventing
                    # positional extrapolation artifacts". We already do that
                    # for the temporal RoPE (window-relative re-roping pins the
                    # sink<->query rotation distance); PRoPE had no equivalent.
                    #
                    # It shows up as direction decaying along the clip rather
                    # than being absent: case 171 background zoom on W reads
                    # +0.151 over f1-8 and -0.016 over f9-24. First-person
                    # shots suffer most, because the sink holds a
                    # camera-attached hand that this growing transform warps.
                    sink_n = max(0, min(prope_sink_frames, vm_window.shape[1]))
                    if sink_n and anchor_idx >= sink_n:
                        vm_window = vm_window.clone()
                        vm_window[:, :sink_n] = vm_window[:, anchor_idx:anchor_idx + 1]
                ks_window = (Ks[:, self._prope_mirror]
                             if Ks is not None else None)
                prope_window_viewmats = vm_window
                prope_window_Ks = ks_window
                n = current_num_frames
                vm_chunk = vm_window[:, -n:]
                ks_chunk = (ks_window[:, -n:] if ks_window is not None else None)
            else:
                vm_chunk = viewmats[:, current_start_frame:current_start_frame + current_num_frames] if viewmats is not None else None
                ks_chunk = Ks[:, current_start_frame:current_start_frame + current_num_frames] if Ks is not None else None

            # Step 3.1: Spatial denoising loop
            sample_scheduler = self._initialize_sample_scheduler(noise)
            for _, t in enumerate(tqdm(sample_scheduler.timesteps)):
                latent_model_input = latents
                timestep = t * torch.ones(
                    [batch_size, current_num_frames], device=noise.device, dtype=torch.float32
                )
                if is_reference_block:
                    timestep[:, :1] = 0

                flow_pred_cond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    viewmats=vm_chunk,
                    Ks=ks_chunk,
                    prope_kv_cache=self.prope_kv_cache_pos,
                    prope_window_viewmats=prope_window_viewmats,
                    cache_keep_frames=self._cache_keep,
                    cache_sink_frames=self._sink_promote,
                    prope_window_Ks=prope_window_Ks
                )
                flow_pred_uncond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=unconditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    viewmats=vm_chunk,
                    Ks=ks_chunk,
                    prope_kv_cache=self.prope_kv_cache_neg,
                    prope_window_viewmats=prope_window_viewmats,
                    cache_keep_frames=self._cache_keep,
                    cache_sink_frames=self._sink_promote,
                    prope_window_Ks=prope_window_Ks
                )

                flow_pred = flow_pred_uncond + self.args.guidance_scale * (
                    flow_pred_cond - flow_pred_uncond)

                if self.camera_guidance_scale > 0 and vm_chunk is not None:
                    # Same text, same noise, camera held still: every frame in
                    # the window gets the anchor's pose, so PRoPE sees no motion
                    # to encode. The difference between this and the
                    # conditioned prediction is what the trajectory is asking
                    # for, and nothing else.
                    vm_chunk_n = torch.eye(
                        4, dtype=vm_chunk.dtype, device=vm_chunk.device
                    ).expand_as(vm_chunk).contiguous()
                    pw_n = (torch.eye(
                        4, dtype=prope_window_viewmats.dtype,
                        device=prope_window_viewmats.device
                    ).expand_as(prope_window_viewmats).contiguous()
                        if prope_window_viewmats is not None else None)
                    flow_pred_camneutral, _ = self.generator(
                        noisy_image_or_video=latent_model_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache_cam,
                        crossattn_cache=self.crossattn_cache_cam,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_start=cache_start_frame * self.frame_seq_length,
                        viewmats=vm_chunk_n,
                        Ks=ks_chunk,
                        prope_kv_cache=self.prope_kv_cache_cam,
                        prope_window_viewmats=pw_n,
                        cache_keep_frames=self._cache_keep,
                        cache_sink_frames=self._sink_promote,
                        prope_window_Ks=prope_window_Ks
                    )
                    flow_pred = flow_pred + self.camera_guidance_scale * (
                        flow_pred_cond - flow_pred_camneutral)

                temp_x0 = sample_scheduler.step(
                    flow_pred,
                    t,
                    latents,
                    return_dict=False)[0]
                latents = temp_x0
                if is_reference_block:
                    # Keep the sequence member identical to the clean
                    # reference after every solver update, matching both
                    # ref_cond_prob training and bidirectional I2V inference.
                    latents[:, :1] = reference_latent
                print(f"kv_cache['local_end_index']: {self.kv_cache_pos[0]['local_end_index']}")
                print(f"kv_cache['global_end_index']: {self.kv_cache_pos[0]['global_end_index']}")

            # Step 3.2: record the model's output
            output[:, cache_start_frame:cache_start_frame + current_num_frames] = latents

            # Capture chunk0 latency: stop timer once the first chunk's denoised output is ready.
            if self.last_chunk0_latency is None:
                torch.cuda.synchronize()
                self.last_chunk0_latency = time.perf_counter() - _chunk0_t0

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length,
                viewmats=vm_chunk,
                Ks=ks_chunk,
                prope_kv_cache=self.prope_kv_cache_pos,
                prope_window_viewmats=prope_window_viewmats,
                cache_keep_frames=self._cache_keep,
                cache_sink_frames=self._sink_promote,
                prope_window_Ks=prope_window_Ks
            )
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=unconditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_neg,
                crossattn_cache=self.crossattn_cache_neg,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length,
                viewmats=vm_chunk,
                Ks=ks_chunk,
                prope_kv_cache=self.prope_kv_cache_neg,
                prope_window_viewmats=prope_window_viewmats,
                cache_keep_frames=self._cache_keep,
                cache_sink_frames=self._sink_promote,
                prope_window_Ks=prope_window_Ks
            )

            # Commit only the final clean rerun to long-term appearance
            # memory. Denoising passes may overwrite the resident local cache,
            # but they never enter this bank, preventing noisy/error feedback.
            for cache in (self.kv_cache_pos, self.kv_cache_neg):
                for layer_cache in cache:
                    token_count = current_num_frames * self.frame_seq_length
                    end = int(layer_cache["local_end_index"].item())
                    start = end - token_count
                    block_k = layer_cache["k"][:, start:end].reshape(
                        batch_size, current_num_frames, self.frame_seq_length,
                        *layer_cache["k"].shape[2:])
                    block_v = layer_cache["v"][:, start:end].reshape(
                        batch_size, current_num_frames, self.frame_seq_length,
                        *layer_cache["v"].shape[2:])
                    summary_start = 0
                    if layer_cache["memory_ref_k"] is None:
                        # Preserve full spatial reference tokens permanently.
                        layer_cache["memory_ref_k"] = block_k[:, 0].detach().clone()
                        layer_cache["memory_ref_v"] = block_v[:, 0].detach().clone()
                        summary_start = 1
                    if summary_start < current_num_frames:
                        summary_k = block_k[:, summary_start:].mean(dim=2).detach()
                        summary_v = block_v[:, summary_start:].mean(dim=2).detach()
                        if layer_cache["memory_summary_k"] is None:
                            layer_cache["memory_summary_k"] = summary_k.clone()
                            layer_cache["memory_summary_v"] = summary_v.clone()
                        else:
                            layer_cache["memory_summary_k"] = torch.cat(
                                [layer_cache["memory_summary_k"], summary_k], dim=1)
                            layer_cache["memory_summary_v"] = torch.cat(
                                [layer_cache["memory_summary_v"], summary_v], dim=1)

            if is_reference_block:
                ref_scale = float(getattr(self.args, "ref_cond_scale", 1.0))
                if ref_scale != 1.0:
                    # Match the conditioning-strength control already used by
                    # the few-step pipeline.  The reference is the persistent
                    # sink token in sequence-I2V; scaling only its clean cached
                    # values strengthens appearance/identity without changing
                    # positions, later motion tokens, or the generated pixels.
                    span = self.frame_seq_length
                    for cache in (self.kv_cache_pos, self.kv_cache_neg):
                        for layer_cache in cache:
                            layer_cache["v"][:, :span] *= ref_scale
                    # The camera branch uses a separate cache.  Scaling only
                    # the ordinary cache makes the two branches disagree about
                    # reference importance.  Keep the reference weight
                    # symmetric.
                    if self.prope_kv_cache_pos is not None:
                        for cache in (self.prope_kv_cache_pos,
                                      self.prope_kv_cache_neg):
                            for layer_cache in cache:
                                layer_cache["v"][:, :span] *= ref_scale
                    for cache in (self.kv_cache_pos, self.kv_cache_neg):
                        for layer_cache in cache:
                            if layer_cache["memory_ref_v"] is not None:
                                layer_cache["memory_ref_v"] *= ref_scale
                    print(f"[cond] reference values scaled by {ref_scale}",
                          flush=True)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames

        # Step 4: Decode the output
        if return_video:
            stitch = float(getattr(self.args, "latent_stitch_strength", 0.0))
            if stitch > 0 and self.num_frame_per_block > 1:
                # Causal blocks are independently denoised and otherwise hard
                # concatenated. Match the first latent of each new block to a
                # conservative continuation of the previous block, then fade
                # that offset to zero inside the block. This is overlap-add in
                # latent space: it does not duplicate or mutate KV entries, so
                # the model remains on its trained cache distribution.
                output = output.clone()
                block = int(self.num_frame_per_block)
                for start in range(block, output.shape[1], block):
                    if start < 2:
                        continue
                    velocity = output[:, start - 1] - output[:, start - 2]
                    expected = output[:, start - 1] + 0.5 * velocity
                    offset = expected - output[:, start]
                    # A rare bad block must not drag the whole following block
                    # with it. Clamp relative to the current latent scale.
                    scale = output[:, start].float().std(
                        dim=(1, 2, 3), keepdim=True).to(output.dtype)
                    limit = 0.35 * scale
                    offset = offset.clamp(min=-limit, max=limit)
                    highpass_kernel = int(getattr(
                        self.args, "latent_stitch_highpass_kernel", 0))
                    if highpass_kernel > 1:
                        if highpass_kernel % 2 == 0:
                            raise ValueError(
                                "latent_stitch_highpass_kernel must be odd")
                        radius = highpass_kernel // 2
                        low_frequency = F.avg_pool2d(
                            F.pad(offset, (radius, radius, radius, radius),
                                  mode="reflect"),
                            highpass_kernel, stride=1)
                        offset = offset - low_frequency
                    if bool(getattr(
                            self.args, "latent_stitch_preserve_mean", False)):
                        # The spatially constant component primarily changes
                        # exposure.  Keep the structural/motion correction but
                        # make it luminance-neutral for every latent channel.
                        offset = offset - offset.mean(
                            dim=(-2, -1), keepdim=True)
                    end = min(start + block, output.shape[1])
                    count = end - start
                    weights = torch.linspace(
                        1.0, 0.0, count + 1, device=output.device,
                        dtype=output.dtype)[:-1]
                    weights = weights.view(1, count, 1, 1, 1)
                    output[:, start:end] += stitch * weights * offset[:, None]
            temporal_smooth = float(getattr(
                self.args, "latent_temporal_smooth", 0.0))
            if temporal_smooth > 0 and output.shape[1] > 2:
                # Penalise temporal acceleration, not velocity. A constant
                # camera pan/forward move is unchanged; only alternating
                # frame-to-frame latent jitter (visible as VAE cadence noise)
                # is attenuated. Use an out-of-place source so this is one
                # symmetric pass rather than a direction-dependent filter.
                source = output.clone()
                linear_mid = 0.5 * (source[:, :-2] + source[:, 2:])
                correction = linear_mid - source[:, 1:-1]
                scale = source[:, 1:-1].float().std(
                    dim=(2, 3, 4), keepdim=True).to(source.dtype)
                correction = correction.clamp(
                    min=-0.2 * scale, max=0.2 * scale)
                output[:, 1:-1] = (source[:, 1:-1]
                                    + temporal_smooth * correction)
            if os.environ.get("WAN_SKIP_DECODE") == "1":
                empty_video = torch.empty((1, 0, 3, 1, 1), dtype=torch.float32)
                return (empty_video, output) if return_latents else empty_video

            # With the VAE on its own card (`vae_device`, set by a caller that
            # split the model), the latents go to it for the decode. Unset, this
            # is the single-card call it always was.
            _vae_dev = getattr(self, "vae_device", None)
            _offloaded = self._release_denoiser_for_decode()
            video = self.vae.decode_to_pixel(
                output.to(_vae_dev) if _vae_dev is not None else output)
            if _offloaded is not None:
                self.generator.to(_offloaded)
            video = (video * 0.5 + 0.5).clamp(0, 1)

            chroma_strength = float(getattr(
                self.args, "rgb_chroma_stabilize_strength", 0.0))
            if chroma_strength > 0 and video.shape[1] > 8:
                # Smooth only the temporal trajectory of spatial RGB means.
                # Pixels are never mixed spatially or temporally: every frame
                # receives one constant per-channel offset, preserving all
                # texture while removing multi-frame colour stair-steps.
                source = video.clone()
                means = source.float().mean(dim=(-2, -1))
                radius = max(1, int(getattr(
                    self.args, "rgb_chroma_stabilize_fade", 4)))
                radius = min(radius, video.shape[1] - 1)
                rising = torch.arange(
                    1, radius + 2, device=video.device, dtype=torch.float32)
                kernel = torch.cat([rising, rising[:-1].flip(0)])
                kernel = (kernel / kernel.sum()).view(1, 1, -1).repeat(3, 1, 1)
                mean_signal = means.transpose(1, 2)
                padded = F.pad(mean_signal, (radius, radius), mode="reflect")
                smooth = F.conv1d(padded, kernel, groups=3).transpose(1, 2)
                correction = (smooth - means).clamp(-0.06, 0.06)
                video = (source + chroma_strength
                         * correction.to(video.dtype)[..., None, None]).clamp(0, 1)
                print(f"[rgb-chroma] smoothed RGB means radius={radius} "
                      f"max_correction={correction.abs().max().item():.6f}")

            if return_latents:
                return video, output
            else:
                return video
        else:
            return output

    
    def inference_for_cd(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        record_step_indices: List[int],
        initial_latent: Optional[torch.Tensor] = None,
        start_frame_index: int = 0
    ):
        """
        Causal-forcing inference + record selected diffusion steps (per-chunk) for consistency distillation data.
        Record semantics: record xt BEFORE scheduler.step() at the specified progress_id (index in timesteps list).
        Also record the final latent of each chunk after the denoising loop.

        Returns:
            if return_video:
                (video, output_latents, cd_pack)
            else:
                (output_latents, cd_pack)

        cd_pack:
            {
            "record_step_indices": [...],
            "record_t_values": [t_i ...]  # same for all chunks
            "chunks": [
                {
                    "frame_start": int,
                    "frame_len": int,
                    "latents": Tensor [B, R, T, C, H, W]  (R = len(record_step_indices)+1, last one is final)
                }, ...
            ]
            }
        """
        self.sampling_steps = 48
        batch_size, num_frames, num_channels, height, width = noise.shape

        # ---- block counting (same logic as inference) ----
        if (not self.independent_first_frame) or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block

        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames

        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        unconditional_dict = self.text_encoder(text_prompts=[self.args.negative_prompt] * len(text_prompts))

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # ---- Step 1: init/reset caches (same as inference) ----
        if self.kv_cache_pos is None:
            self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
            self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache_pos)):
                self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache_neg[block_index]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache_neg[block_index]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)

        # ---- validate record indices against scheduler length ----
        sample_scheduler_probe = self._initialize_sample_scheduler(noise)
        T = len(sample_scheduler_probe.timesteps)
        record_step_indices = sorted(set(int(i) for i in record_step_indices))
        if len(record_step_indices) == 0:
            raise ValueError("record_step_indices must be non-empty")
        if record_step_indices[0] < 0 or record_step_indices[-1] >= T:
            raise ValueError(f"record_step_indices out of range: valid=[0,{T-1}], got={record_step_indices}")
        record_set = set(record_step_indices)

        # ---- Step 2: cache context from initial_latent (same as inference) ----
        current_start_frame = start_frame_index
        cache_start_frame = 0

        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            
            # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
            assert num_input_frames % self.num_frame_per_block == 0
            num_input_blocks = num_input_frames // self.num_frame_per_block

            for block_index in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, cache_start_frame:cache_start_frame + self.num_frame_per_block]
                output[:, cache_start_frame:cache_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                current_start_frame += self.num_frame_per_block
                cache_start_frame += self.num_frame_per_block

        # ---- Step 3: causal-forcing denoising per chunk + record ----
        all_num_frames = [self.num_frame_per_block] * num_blocks

        full_chunk_record = []
        for current_num_frames in all_num_frames:
            # noise slice for current window (same as inference)
            noisy_input = noise[:, cache_start_frame - num_input_frames:cache_start_frame + current_num_frames - num_input_frames]
            latents = noisy_input

            # record list for this chunk
            chunk_records = []

            sample_scheduler = self._initialize_sample_scheduler(noise)
            for progress_id, t in enumerate(tqdm(sample_scheduler.timesteps)):
                if progress_id in record_set:
                    print(f'{progress_id}: {t} saved')
                    chunk_records.append(latents.detach().clone())

                timestep = t * torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.float32)

                flow_pred_cond, _ = self.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                flow_pred_uncond, _ = self.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )

                flow_pred = flow_pred_uncond + self.args.guidance_scale * (flow_pred_cond - flow_pred_uncond)
                latents = sample_scheduler.step(flow_pred, t, latents, return_dict=False)[0]

            # always append final latent of this chunk (like "-2")
            chunk_records.append(latents.detach().clone())
            chunk_records = torch.stack(chunk_records, dim=1)  # [B, R, T, C, H, W]

            full_chunk_record.append(chunk_records)
            # write output
            output[:, cache_start_frame:cache_start_frame + current_num_frames] = latents

            # rerun at t=0 to update cache using clean context (same as inference)
            timestep0 = torch.zeros([batch_size, current_num_frames], device=noise.device, dtype=torch.float32)
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=timestep0,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length
            )
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=unconditional_dict,
                timestep=timestep0,
                kv_cache=self.kv_cache_neg,
                crossattn_cache=self.crossattn_cache_neg,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length
            )

            
            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames

        
        full_chunk_record = torch.cat(full_chunk_record, dim=2)
        # ---- Step 4: decode if needed ----
        
        return full_chunk_record
    
    
    def inference_for_genuine_cd(
        self,
        noisy_input: torch.Tensor,
        conditional_dict = None,
        unconditional_dict = None,
        text_prompts = None,
        initial_latent: Optional[torch.Tensor] = None,
        timestep_idx=0,
        sampling_steps=48,
        chunksize = 3
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noisy_input.shape
        assert num_frames == chunksize
        if initial_latent is not None:
            num_input_frames = initial_latent.shape[1]
            assert num_input_frames % chunksize == 0
            num_output_frames = num_frames + num_input_frames
        else:
            num_output_frames = num_frames
            
        if conditional_dict is None:
            assert text_prompts is not None
            conditional_dict = self.text_encoder(
                text_prompts=text_prompts
            )
            unconditional_dict = self.text_encoder(
                text_prompts=[self.args.negative_prompt] * len(text_prompts)
            )
            
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noisy_input.device,
            dtype=noisy_input.dtype
        )

        if self.kv_cache_pos is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noisy_input.dtype,
                device=noisy_input.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noisy_input.dtype,
                device=noisy_input.device
            )
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache_pos)):
                self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noisy_input.device)
                self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noisy_input.device)
                self.kv_cache_neg[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noisy_input.device)
                self.kv_cache_neg[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noisy_input.device)

        current_start_frame = 0
        cache_start_frame = 0
        timestep = torch.ones([batch_size, 1], device=noisy_input.device, dtype=torch.int64) * 0
        

        if initial_latent is not None:
            num_input_blocks = num_input_frames // chunksize
            for block_index in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, cache_start_frame:cache_start_frame + chunksize]
                output[:, cache_start_frame:cache_start_frame + chunksize] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                current_start_frame += chunksize
                cache_start_frame += chunksize

    
        latents = noisy_input
        sample_scheduler = self._initialize_sample_scheduler(noisy_input, sampling_steps=sampling_steps)
        t = sample_scheduler.timesteps[timestep_idx]
        latent_model_input = latents
        timestep = t * torch.ones(
            [batch_size, chunksize], device=noisy_input.device, dtype=torch.float32
        )
        flow_pred_cond, _ = self.generator(
            noisy_image_or_video=latent_model_input,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=self.kv_cache_pos,
            crossattn_cache=self.crossattn_cache_pos,
            current_start=current_start_frame * self.frame_seq_length,
            cache_start=cache_start_frame * self.frame_seq_length
        )
        flow_pred_uncond, _ = self.generator(
            noisy_image_or_video=latent_model_input,
            conditional_dict=unconditional_dict,
            timestep=timestep,
            kv_cache=self.kv_cache_neg,
            crossattn_cache=self.crossattn_cache_neg,
            current_start=current_start_frame * self.frame_seq_length,
            cache_start=cache_start_frame * self.frame_seq_length
        )
        flow_pred = flow_pred_uncond + self.args.guidance_scale * (
            flow_pred_cond - flow_pred_uncond)

        latents = sample_scheduler.step(
            flow_pred,
            t,
            latents,
            return_dict=False)[0]
        
        return latents

    

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.

        Under SP, KV cache is stored in head-parallel domain (post all-to-all),
        so each rank only stores num_heads // sp_size heads.
        """
        num_heads = self._get_sp_num_heads(self.model_num_heads)
        kv_cache_pos = []
        kv_cache_neg = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 20 * self.frame_seq_length  # 20 frames, this model's tokens

        for _ in range(self.num_transformer_blocks):
            kv_cache_pos.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, self.model_head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, self.model_head_dim], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "memory_ref_k": None,
                "memory_ref_v": None,
                "memory_summary_k": None,
                "memory_summary_v": None,
                "history_full_k": None,
                "history_full_v": None,
            })
            kv_cache_neg.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, self.model_head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, self.model_head_dim], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "memory_ref_k": None,
                "memory_ref_v": None,
                "memory_summary_k": None,
                "memory_summary_v": None,
                "history_full_k": None,
                "history_full_v": None,
            })

        self.kv_cache_pos = kv_cache_pos  # always store the clean cache
        self.kv_cache_neg = kv_cache_neg  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache_pos = []
        crossattn_cache_neg = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache_pos.append({
                "k": torch.zeros([batch_size, self.text_len, self._get_sp_num_heads(self.model_num_heads), self.model_head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.text_len, self._get_sp_num_heads(self.model_num_heads), self.model_head_dim], dtype=dtype, device=device),
                "is_init": False
            })
            crossattn_cache_neg.append({
                "k": torch.zeros([batch_size, self.text_len, self._get_sp_num_heads(self.model_num_heads), self.model_head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.text_len, self._get_sp_num_heads(self.model_num_heads), self.model_head_dim], dtype=dtype, device=device),
                "is_init": False
            })

        self.crossattn_cache_pos = crossattn_cache_pos  # always store the clean cache
        self.crossattn_cache_neg = crossattn_cache_neg  # always store the clean cache

    @staticmethod
    def _get_sp_num_heads(full_num_heads):
        """Return per-rank num_heads under SP (head-parallel domain)."""
        try:
            from utils.sp.parallel_states import get_parallel_state
            ps = get_parallel_state()
            if ps.sp_enabled:
                return full_num_heads // ps.sp
        except (ImportError, AttributeError):
            pass
        return full_num_heads

    def _initialize_prope_kv_cache(self, batch_size, dtype, device):
        num_heads = self._get_sp_num_heads(self.model_num_heads)
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = 20 * self.frame_seq_length  # 20 frames, this model's tokens
        entry = lambda: {
            "k": torch.zeros([batch_size, kv_cache_size, num_heads, self.model_head_dim], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, kv_cache_size, num_heads, self.model_head_dim], dtype=dtype, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        }
        self.prope_kv_cache_pos = [entry() for _ in range(self.num_transformer_blocks)]
        self.prope_kv_cache_neg = [entry() for _ in range(self.num_transformer_blocks)]

    def _initialize_sample_scheduler(self, noise, sampling_steps=-1):
        if sampling_steps == -1:
            sampling_steps = self.sampling_steps
        if self.sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=noise.device, shift=self.shift)
            self.timesteps = sample_scheduler.timesteps
        elif self.sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, self.shift)
            self.timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=noise.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        return sample_scheduler
