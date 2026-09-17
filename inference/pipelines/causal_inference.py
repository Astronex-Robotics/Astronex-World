import os
import time
from typing import List, Optional
import torch

from models.wan_wrapper import WanDiffusionWrapper, build_text_encoder, build_vae

from inference.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation
import tqdm

class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = build_text_encoder(args) if text_encoder is None else text_encoder
        self.vae = build_vae(args) if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        # Derive cache geometry from the instantiated backbone.  The old
        # constants described Wan2.1-1.3B (12 heads, 128 dim, 1560
        # tokens/frame) and silently constructed invalid caches for Wan2.2.
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

        self.kv_cache1 = None
        self.prope_kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size
        self.prope_chunk_anchor = bool(getattr(
            self.generator.model.blocks[0].self_attn, "prope_chunk_anchor", False))
        # Diagnostic: pin the sink/reference frame's camera to the current
        # window anchor so its PRoPE transform stays identity regardless of
        # chunk distance (old-scheme behaviour for the ref). The ref's pairwise
        # geometry vs queries is then off by the anchor<-ref motion, but its
        # projected features never grow, which keeps the most-attended token
        # inside the distribution the weights were trained on.
        self.prope_pin_sink = bool(getattr(args, "prope_pin_sink", False))
        self.retrieval_window = bool(getattr(args, "retrieval_window", False))
        self.retrieval_recent_frames = int(getattr(args, "retrieval_recent_frames", 4))
        self._cache_keep = []
        # Diagnostic: keep the raw-K/V window re-projection mechanism but feed
        # the window cameras in their original (data) gauge, i.e. no per-chunk
        # anchoring. Isolates "re-projection" from "anchoring".
        self.prope_no_anchor = bool(getattr(args, "prope_no_anchor", False))
        # Match reference-conditioned teacher forcing exactly: the first
        # causal block is [clean reference, noisy target, noisy target,
        # noisy target].  The reference stays at timestep zero throughout
        # denoising instead of being filed into the KV cache as a separate
        # history block.
        self.ref_sequence_conditioning = bool(getattr(
            args, "ref_sequence_conditioning", False))
        self.prope_sink_frames = int(getattr(
            self.generator.model.blocks[0].self_attn, "sink_size", 1))
        # Mirror of which frames are currently in the PRoPE cache, kept in
        # cache order (sink first), so the pipeline can hand the model the
        # per-frame window cameras for re-projection under a fresh anchor.
        self._prope_cache_frames = None
        self._prope_last_window = None

        # Latency of producing the first chunk (set by inference()).
        self.last_chunk0_latency = None

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        rectified_tf = False,
        viewmats: Optional[torch.Tensor] = None,  # (B, F, 4, 4) PRoPE camera extrinsics
        Ks: Optional[torch.Tensor] = None,         # (B, F, 3, 3) PRoPE camera intrinsics
        event_prompts: Optional[List[str]] = None,  # independent event-text channel
        event_start_frame: int = 0,  # latent frame the event is injected from
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
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if viewmats is not None and bool(getattr(self.args, "causal_prope_anchor", False)) and \
                not self.prope_chunk_anchor:
            from models.prope import anchor_viewmats_to_first
            # Normalize before any prefill/generation slicing: every cached
            # block and every current query then shares the same camera gauge.
            viewmats = anchor_viewmats_to_first(viewmats)

        # Reset the per-call PRoPE window mirror (chunk-relative anchoring).
        self._prope_cache_frames = []
        self._prope_last_window = None

        def chunk_cameras(start, n):
            """Return (chunk viewmats, chunk Ks, window viewmats, window Ks).

            In chunk-anchor mode the window covers the frames that will be in
            the PRoPE cache after this call (sink + previous chunk + current
            chunk) and is re-anchored to the last frame before `start`, so
            every transform magnitude stays inside the training window. The
            chunk cameras are the anchored window's tail, keeping queries and
            cached keys in one gauge.
            """
            if viewmats is None:
                return None, None, None, None
            if self.prope_chunk_anchor:
                if self._prope_last_window is not None and \
                        self._prope_last_window[0] == start:
                    window_frames = self._prope_last_window[1]
                else:
                    window_frames = self._prope_window_for(
                        list(range(start, start + n)), viewmats, Ks)
                    self._prope_last_window = (start, window_frames)
                anchor_frame = max(0, start - 1)
                vm_window, ks_window = self._prope_window_cameras(
                    viewmats, Ks, window_frames, anchor_frame)
                vm_chunk = vm_window[:, -n:] if vm_window is not None else None
                ks_chunk = ks_window[:, -n:] if ks_window is not None else None
                return vm_chunk, ks_chunk, vm_window, ks_window
            vm_chunk = viewmats[:, start:start + n]
            ks_chunk = Ks[:, start:start + n] if Ks is not None else None
            return vm_chunk, ks_chunk, None, None

        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
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
            num_blocks = (num_frames + 1) // self.num_frame_per_block
        elif not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            # default here
            # self.independent_first_frame: False
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        from inference.pipelines.event_prompt import add_event_embeds
        add_event_embeds(conditional_dict, self.text_encoder,
                         event_prompts, batch_size)

        # The event is auxiliary and schedulable: the caption conditions the
        # whole rollout, the event only the chunks from `event_start_frame` on.
        # That is what "insertable at any point" means here -- the branch simply
        # is not reached before then, so earlier chunks are bit-identical to a
        # run with no event at all, and the KV cache they filled carries the
        # pre-event scene forward. Held off the dict and re-attached per chunk
        # rather than sliced, because the branch is all-or-nothing per forward.
        event_embeds = conditional_dict.pop("event_embeds", None)
        if event_embeds is not None and event_start_frame > 0:
            print(f"[event] injecting from latent frame {event_start_frame}")

        # Releasing the i2v reference part-way through the rollout.
        #
        # The reference frame is prefilled into the KV cache and pinned there by
        # `sink_size`, so it conditions every block to the end. That is what the
        # caller wants for a short clip and the wrong thing for exploration: with
        # it pinned, content distance from the first frame *shrinks* over a 15 s
        # rollout (0.052 -> 0.044), while the same model given no reference at
        # all moves away from where it started (0.036 -> 0.131). The scene is
        # held, not explored.
        #
        # `cond_release_frame: N` unpins it after N latent frames. The opening
        # establishes the scene from the image; after that the model runs as it
        # does in t2v, and the reference is evicted by the ordinary rolling
        # window. 0 (the default) keeps the current behaviour.
        release_at = int(getattr(self.args, "cond_release_frame", 0) or 0)
        self._reference_released = False

        def release_reference(start_frame):
            if (not release_at) or self._reference_released or start_frame < release_at:
                return
            for blk in self.generator.model.blocks:
                blk.self_attn.sink_size = 0
            self.prope_sink_frames = 0
            self._reference_released = True
            print(f"[cond] reference released at latent frame {start_frame}",
                  flush=True)

        def apply_event(start_frame):
            """Attach or detach the event memory for a chunk at `start_frame`.

            Detaching is a `pop`, not a None: the model skips its event branch
            entirely when the key is absent, which is both cheaper and the exact
            computation an event-free run performs.
            """
            if event_embeds is not None and start_frame >= event_start_frame:
                conditional_dict["event_embeds"] = event_embeds
            else:
                conditional_dict.pop("event_embeds", None)

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

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

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
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
            self._initialize_prope_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
            # reset prope kv cache
            for block_index in range(len(self.prope_kv_cache1)):
                self.prope_kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.prope_kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None and not sequence_ref:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                vm_chunk, ks_chunk, vm_window, ks_window = chunk_cameras(
                    current_start_frame, 1)
                apply_event(current_start_frame)
                # The reference frame is prefilled at timestep 0 -- clean, no
                # noise -- while every later block's K/V comes out of a denoising
                # pass. Two different distributions sit in one cache, and the
                # attention flips between them; that is a candidate cause of the
                # colour flicker (0.81 here against 0.26 for the same model with
                # no reference frame at all). `cond_prefill_timestep` prefills it
                # at a low but non-zero timestep instead, so its K/V statistics
                # sit closer to the blocks that follow.
                prefill_t = int(getattr(self.args, "cond_prefill_timestep", 0))
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0 + prefill_t,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    viewmats=vm_chunk,
                    Ks=ks_chunk,
                    prope_kv_cache=self.prope_kv_cache1,
                    prope_window_viewmats=vm_window,
                    cache_keep_frames=(self._cache_keep if getattr(self, 'retrieval_window', False) else None),
                    prope_window_Ks=ks_window,
                )
                # Conditioning strength for the reference image.
                #
                # The three conditioning channels are not comparable as written:
                # the caption and the event prompt each enter through their own
                # cross-attention (the event one already has `event_scale`),
                # while the reference image is prefilled into the KV cache and
                # its weight is whatever the attention softmax gives it. Measured
                # on case 171, that weight wins -- the prompt describes side
                # aisles and a reading alcove to discover, and over 15 s the
                # content distance from the first frame *shrinks* (0.052 ->
                # 0.044) instead of growing as it does with no reference at all.
                #
                # Scaling the cached values scales exactly the reference's
                # contribution to every later block's attention output, which is
                # the knob that was missing. 1.0 is the old behaviour.
                ref_scale = float(getattr(self.args, "ref_cond_scale", 1.0))
                if ref_scale != 1.0:
                    span = self.frame_seq_length
                    for c in self.kv_cache1:
                        c["v"][:, :span] *= ref_scale
                    print(f"[cond] reference values scaled by {ref_scale}", flush=True)

                current_start_frame += 1
                if self.prope_chunk_anchor and self._prope_last_window is not None:
                    self._prope_cache_frames = list(self._prope_last_window[1])
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                vm_chunk, ks_chunk, vm_window, ks_window = chunk_cameras(
                    current_start_frame, self.num_frame_per_block)
                apply_event(current_start_frame)
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    viewmats=vm_chunk,
                    Ks=ks_chunk,
                    prope_kv_cache=self.prope_kv_cache1,
                    prope_window_viewmats=vm_window,
                    cache_keep_frames=(self._cache_keep if getattr(self, 'retrieval_window', False) else None),
                    prope_window_Ks=ks_window,
                )
                current_start_frame += self.num_frame_per_block
                if self.prope_chunk_anchor and self._prope_last_window is not None:
                    self._prope_cache_frames = list(self._prope_last_window[1])

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        if sequence_ref:
            # The first model call consumes one reference frame and block-1
            # noise frames. Later calls consume a full noise block. Keeping
            # current_start_frame in output coordinates makes the existing
            # noise slicing correct for both cases.
            all_num_frames = [self.num_frame_per_block] * (
                (num_frames + 1) // self.num_frame_per_block)
        else:
            all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_num_frames in tqdm.tqdm(all_num_frames):
            if profile:
                block_start.record()

            apply_event(current_start_frame)
            release_reference(current_start_frame)

            is_reference_block = sequence_ref and current_start_frame == 0
            if is_reference_block:
                noisy_input = noise[:, :current_num_frames - 1]
                reference_latent = initial_latent[:, :1].to(noisy_input.dtype)
                noisy_input = torch.cat([reference_latent, noisy_input], dim=1)
            else:
                noisy_input = noise[
                    :, current_start_frame - num_input_frames:
                    current_start_frame + current_num_frames - num_input_frames]

            vm_chunk, ks_chunk, vm_window, ks_window = chunk_cameras(
                current_start_frame, current_num_frames)

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                # print(f"current_timestep: {current_timestep}")
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep
                if is_reference_block:
                    timestep[:, :1] = 0

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        viewmats=vm_chunk,
                        Ks=ks_chunk,
                        prope_kv_cache=self.prope_kv_cache1,
                        prope_window_viewmats=vm_window,
                        cache_keep_frames=(self._cache_keep if getattr(self, 'retrieval_window', False) else None),
                        prope_window_Ks=ks_window,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    if is_reference_block:
                        generated_pred = denoised_pred[:, 1:]
                        generated_noisy = self.scheduler.add_noise(
                            generated_pred.flatten(0, 1),
                            torch.randn_like(generated_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * (current_num_frames - 1)],
                                device=noise.device, dtype=torch.long)
                        ).unflatten(0, generated_pred.shape[:2])
                        # The clean reference is a sequence member at every
                        # denoising step, exactly as in training.
                        noisy_input = torch.cat(
                            [reference_latent, generated_noisy], dim=1)
                    else:
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        viewmats=vm_chunk,
                        Ks=ks_chunk,
                        prope_kv_cache=self.prope_kv_cache1,
                        prope_window_viewmats=vm_window,
                        cache_keep_frames=(self._cache_keep if getattr(self, 'retrieval_window', False) else None),
                        prope_window_Ks=ks_window,
                    )

            # Step 3.2: record the model's output
            if is_reference_block:
                denoised_pred[:, :1] = reference_latent
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Capture chunk0 latency: stop timer once the first chunk's denoised output is ready.
            if self.last_chunk0_latency is None:
                torch.cuda.synchronize()
                self.last_chunk0_latency = time.perf_counter() - _chunk0_t0

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                viewmats=vm_chunk,
                Ks=ks_chunk,
                prope_kv_cache=self.prope_kv_cache1,
                prope_window_viewmats=vm_window,
                cache_keep_frames=(self._cache_keep if getattr(self, 'retrieval_window', False) else None),
                prope_window_Ks=ks_window,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames
            if self.prope_chunk_anchor and self._prope_last_window is not None:
                self._prope_cache_frames = list(self._prope_last_window[1])

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()
        if rectified_tf: 
            mean = torch.load('laboratory/mean.pt').to(output.device) 
            std = torch.load('laboratory/std.pt').to(output.device) 
            noise = torch.randn_like(output).to(output.device) 
            output -= mean 
        # Latent-only mode separates expensive diffusion from VAE decoding.
        # This is useful at 720p when a 48 GB card cannot hold the 5B backbone,
        # attention caches and the full-clip decoder workspace together.
        if os.environ.get("WAN_SKIP_DECODE") == "1":
            empty_video = torch.empty((1, 0, 3, 1, 1), dtype=torch.float32)
            return (empty_video, output) if return_latents else empty_video

        # At 720p the diffusion backbone, KV caches and VAE decoder do not fit
        # on a 48 GB card at the same time.  Sampling is complete here, so the
        # backbone and text encoder can be offloaded before GPU VAE decode.
        # This keeps decoding fast without changing the sampled latent.
        if os.environ.get("WAN_OFFLOAD_GENERATOR_FOR_DECODE") == "1":
            self.generator.to("cpu")
            self.text_encoder.to("cpu")
            # Attention caches are owned by the pipeline rather than the
            # generator module, so moving the module alone does not release
            # them.  They are no longer needed once the final latent exists.
            self.kv_cache1 = None
            self.crossattn_cache = None
            self.prope_kv_cache1 = None
            self._cache_keep = []
            self._prope_cache_frames = None
            self._prope_last_window = None
            torch.cuda.empty_cache()
        # Step 4: Decode the output.
        #
        # Chunk-by-chunk with the VAE's own causal-conv cache, not one call over
        # the whole clip. Measured on case 171, decoding was 52.6% of inference
        # -- 8.45 s against 7.47 s for all four denoising steps combined -- so
        # this is the larger half of the cost, and it was being paid in a single
        # blocking call at the end.
        #
        # Streaming it matters for more than throughput: a caller can display
        # each chunk as it lands instead of waiting for the last frame to be
        # generated, which is the difference between ~16 s to first pixel and
        # roughly one chunk's worth.
        #
        # `use_cache=True` continues the previous chunk's temporal state rather
        # than restarting it, so the seams between chunks carry the same
        # receptive field a whole-clip decode would have given them.
        # `stream_decode: true` decodes chunk by chunk, continuing the decoder's
        # temporal state across calls (see `Wan22VAEWrapper._decode_streaming`).
        # The first attempt at this went through `AutoencoderKLWan.decode`, which
        # clears its own cache on entry and exit, so every chunk restarted blank
        # and the output drifted from a whole-clip decode by 10.5 mean / 255 max
        # pixel levels, growing along the clip. Driving the decoder's per-frame
        # loop directly keeps the state and makes the two paths agree.
        stream = bool(getattr(self.args, "stream_decode", False)) and output.shape[0] == 1
        if stream:
            self.vae.reset_stream()
            chunks, start = [], 0
            # The i2v prefix frame is decoded on its own, matching how it was
            # generated; the rest follow in generation-sized blocks.
            sizes = ([1] if self.independent_first_frame and num_input_frames == 0
                     else [])
            while sum(sizes) < output.shape[1]:
                sizes.append(min(self.num_frame_per_block,
                                 output.shape[1] - sum(sizes)))
            for n in sizes:
                chunks.append(self.vae.decode_to_pixel(
                    output[:, start:start + n], use_cache=True))
                start += n
            video = torch.cat(chunks, dim=1)
        else:
            video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.

        Under SP, KV cache is stored in head-parallel domain (post all-to-all),
        so each rank only stores num_heads // sp_size heads.
        """
        num_heads = self._get_sp_num_heads(self.model_num_heads)
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 20 * self.frame_seq_length

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, self.model_head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, self.model_head_dim], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.

        NOTE: Cross-attention does NOT use SP all-to-all, so cache keeps full num_heads.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, self.text_len, self.model_num_heads, self.model_head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.text_len, self.model_num_heads, self.model_head_dim], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def _initialize_prope_kv_cache(self, batch_size, dtype, device):
        num_heads = self._get_sp_num_heads(self.model_num_heads)
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = 20 * self.frame_seq_length
        self.prope_kv_cache1 = [{
            "k": torch.zeros([batch_size, kv_cache_size, num_heads, self.model_head_dim], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, kv_cache_size, num_heads, self.model_head_dim], dtype=dtype, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        } for _ in range(self.num_transformer_blocks)]

    def _prope_window_for(self, new_frames, viewmats=None, Ks=None):
        """Frames that will occupy the PRoPE cache after caching `new_frames`.

        Mirrors the backbone's eviction rule (keep `sink_size` leading frames,
        evict the oldest of the rest) so the window cameras handed to the
        model line up 1:1 with the raw K/V actually in the cache.
        """
        cache_frames = self.local_attn_size if self.local_attn_size != -1 else 20
        combined = list(self._prope_cache_frames) + list(new_frames)
        if len(combined) <= cache_frames:
            self._cache_keep = [f for f in combined if f < min(new_frames)]
            return combined
        evict = len(combined) - cache_frames
        sink = max(0, min(self.prope_sink_frames, len(combined)))
        if not (getattr(self, "retrieval_window", False) and viewmats is not None):
            combined = combined[:sink] + combined[sink + evict:]
            self._cache_keep = [f for f in combined if f < min(new_frames)]
            return combined
        # Keep what the incoming block is about to look at rather than what is
        # merely recent. Evicting by age is why a corridor walked forty frames
        # ago cannot be redrawn consistently on the way back -- the frames that
        # saw it are already gone, so nothing constrains the redraw. The newest
        # few are pinned regardless: Memory Forcing reports that leaning wholly
        # on spatial memory degrades genuinely new scenery, and those frames
        # are the temporal half of that trade.
        body = combined[sink:len(combined) - len(new_frames)]
        keep_n = cache_frames - sink - len(new_frames)
        recent = int(getattr(self, "retrieval_recent_frames", 4))
        pinned = body[-recent:] if recent > 0 else []
        free = [f for f in body if f not in pinned]
        room = max(0, keep_n - len(pinned))
        if room and free:
            # Imported here: camera_diffusion pulls in this module, so a
            # top-level import closes the cycle and breaks both.
            from post_train.objectives.camera_diffusion import _frustum_overlap
            scores = _frustum_overlap(viewmats, Ks, list(new_frames)[-1], free)
            free = sorted(free, key=lambda f: -scores[f])[:room]
        else:
            free = []
        body = sorted(set(free) | set(pinned))
        self._cache_keep = list(body)
        return combined[:sink] + body + list(new_frames)

    def _prope_window_cameras(self, viewmats, Ks, window_frames, anchor_frame):
        """Per-frame window cameras in the gauge of `anchor_frame`."""
        vm = viewmats[:, window_frames]
        ks = Ks[:, window_frames] if Ks is not None else None
        if self.prope_chunk_anchor:
            from models.prope import anchor_viewmats_to_frame
            # The anchor is the previous chunk's last frame, and with the
            # default 20-frame window it is always still resident. A small
            # window evicts it -- `local_attn_size: 5` keeps one sink plus the
            # current block -- and `.index()` then raised "4 is not in list",
            # which reads as a data error rather than "your window is shorter
            # than your anchor distance".
            #
            # Falling back to the newest retained frame at or before the anchor
            # keeps the gauge as close to the intended one as the cache allows;
            # the sink is the last resort, and it is always present.
            if anchor_frame in window_frames:
                anchor_idx = window_frames.index(anchor_frame)
            else:
                older = [i for i, f in enumerate(window_frames) if f <= anchor_frame]
                anchor_idx = older[-1] if older else 0
            if not self.prope_no_anchor:
                vm = anchor_viewmats_to_frame(vm, anchor_idx)
            if self.prope_pin_sink and 0 in window_frames:
                sink_idx = window_frames.index(0)
                if sink_idx != anchor_idx:
                    vm = vm.clone()
                    vm[:, sink_idx] = vm[:, anchor_idx]
        return vm, ks

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
