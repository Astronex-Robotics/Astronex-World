from tqdm import tqdm
from typing import List, Optional
import time
import torch

from models.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from models.fm_solvers_unipc import FlowUniPCMultistepScheduler
from models.wan_wrapper import (
    WanDiffusionWrapper,
    build_text_encoder,
    build_vae,
)


class BidirectionalDiffusionInferencePipeline(torch.nn.Module):
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
            **getattr(args, "model_kwargs", {}), is_causal=False) if generator is None else generator
        self.text_encoder = build_text_encoder(args) if text_encoder is None else text_encoder
        self.vae = build_vae(args) if vae is None else vae

        # Step 2: Initialize scheduler
        self.num_train_timesteps = args.num_train_timestep
        self.sampling_steps = int(getattr(args, "sampling_steps", 50))
        self.sample_solver = 'unipc'
        self.shift = 5.0

        self.args = args

        # Latency to first denoised latent (excludes VAE decode), set per-call.
        self.last_chunk0_latency = None

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        return_latents=False,
        initial_latent: Optional[torch.Tensor] = None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        event_prompts: Optional[List[str]] = None,
        event_start_frame: int = 0,
        concat_initial_latent: bool = False,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """

        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * len(text_prompts)
        )

        if event_start_frame:
            # Accepted so the CLI flag is uniform, but this pipeline conditions
            # the whole clip in one pass -- there is no "before the event" to be
            # identical to. Failing loudly beats silently ignoring a schedule
            # the caller asked for.
            raise ValueError(
                "event_start_frame is only supported by CausalInferencePipeline; "
                "this pipeline conditions the whole rollout at once")
        from inference.pipelines.event_prompt import add_event_embeds
        add_event_embeds(conditional_dict, self.text_encoder,
                         event_prompts, len(text_prompts))

        torch.cuda.synchronize()
        _chunk0_t0 = time.perf_counter()
        self.last_chunk0_latency = None

        latents = (torch.cat([initial_latent, noise], dim=1)
                   if initial_latent is not None and concat_initial_latent
                   else noise)
        condition = None
        first_frame_mask = None
        if initial_latent is not None:
            condition = initial_latent
            first_frame_mask = torch.ones(
                latents.shape[0], latents.shape[1], 1, 1, 1,
                device=condition.device, dtype=condition.dtype)
            prefix_len = (condition.shape[1] if concat_initial_latent else 1)
            first_frame_mask[:, :prefix_len] = 0

        sample_scheduler = self._initialize_sample_scheduler(noise)
        timesteps = list(sample_scheduler.timesteps)
        for step_idx, t in enumerate(tqdm(timesteps)):
            if initial_latent is not None:
                if concat_initial_latent:
                    latent_model_input = latents.clone()
                    latent_model_input[:, :condition.shape[1]] = condition
                else:
                    latent_model_input = (
                        (1 - first_frame_mask) * condition
                        + first_frame_mask * latents)
            else:
                latent_model_input = latents

            timestep = t * torch.ones([latents.shape[0], latents.shape[1]], device=noise.device, dtype=torch.float32)
            if initial_latent is not None:
                timestep[:, :prefix_len] = 0

            flow_pred_cond, _ = self.generator(latent_model_input, conditional_dict, timestep, viewmats=viewmats, Ks=Ks)
            flow_pred_uncond, _ = self.generator(latent_model_input, unconditional_dict, timestep, viewmats=viewmats, Ks=Ks)

            flow_pred = flow_pred_uncond + self.args.guidance_scale * (
                flow_pred_cond - flow_pred_uncond)

            latents = sample_scheduler.step(
                flow_pred, t, latents, return_dict=False)[0]

        if initial_latent is not None:
            if concat_initial_latent:
                latents[:, :condition.shape[1]] = condition
            else:
                latents = (1 - first_frame_mask) * condition + first_frame_mask * latents

        x0 = latents

        # Stop chunk0 timer once the first (and only) denoised latent is ready,
        # before VAE decode — matches the reported latency definition.
        torch.cuda.synchronize()
        self.last_chunk0_latency = time.perf_counter() - _chunk0_t0

        vae_device = getattr(self, "vae_device", None)
        video = self.vae.decode_to_pixel(
            x0.to(vae_device) if vae_device is not None else x0)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        del sample_scheduler

        if return_latents:
            return video, latents
        else:
            return video

    def _initialize_sample_scheduler(self, noise):
        if self.sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                self.sampling_steps, device=noise.device, shift=self.shift)
            self.timesteps = sample_scheduler.timesteps
        elif self.sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(self.sampling_steps, self.shift)
            self.timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=noise.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        return sample_scheduler
