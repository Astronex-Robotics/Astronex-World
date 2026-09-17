"""The frozen components Astronex runs on: the Wan2.2 VAE and UMT5-xxl.

Both are loaded from the released weights directory (``vae/``,
``text_encoder/``, ``tokenizer/``).

* **VAE.** ``AutoencoderKLWan`` from diffusers: 48 channels, 16x16x4. Used as
  is rather than reimplemented -- a hand-derived copy would put a subtly
  different tokenizer under the trained denoiser.
* **Text encoder.** umT5-xxl as a HF ``UMT5EncoderModel``.

Interfaces: ``Wan22TextEncoder(text_prompts) -> {"prompt_embeds"}``,
``encode_to_latent(pixel[B,C,F,H,W]) -> [B,F,C,H,W]``,
``decode_to_pixel(latent[B,F,C,H,W]) -> [B,F,C,H,W]``.

Latent normalisation follows diffusers' AutoencoderKLWan convention --
``(z - latents_mean) / latents_std`` -- read from the checkpoint's own config
rather than hardcoded, because a stale copy of those 48 numbers is invisible
until sample quality is mysteriously bad.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import torch


def wan22_root() -> Path:
    """Directory holding vae/, text_encoder/ and tokenizer/: the released weights."""
    return Path(os.environ.get("ASTRONEX_WEIGHTS", "../Astronex"))


class Wan22TextEncoder(torch.nn.Module):
    """Frozen umT5-xxl encoder from the released weights."""

    def __init__(self, root: str | Path | None = None, dtype=torch.bfloat16):
        super().__init__()
        from transformers import AutoTokenizer, UMT5EncoderModel

        root = Path(root) if root is not None else wan22_root()
        self.text_len = 512
        self.tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
        self.text_encoder = (
            UMT5EncoderModel.from_pretrained(str(root / "text_encoder"), torch_dtype=dtype)
            .eval()
            .requires_grad_(False)
        )
        # Prompt embeddings are a pure function of the prompt string, and a
        # control-labelled dataset typically carries very few distinct prompts
        # (the HM3D export carries exactly one). Caching them turns umT5 from a
        # per-step cost into a one-off.
        self._cache: dict[str, torch.Tensor] = {}
        self.cache_limit = 64

    @property
    def device(self):
        # DynamicSwap keeps params on CPU but swaps them up lazily, so inputs
        # must be placed on the swap target rather than the storage device.
        swap = self.text_encoder.__dict__.get('forge_swap_kwargs')
        if swap is not None and 'device' in swap:
            return swap['device']
        # Follow the actual encoder placement.  Returning the current CUDA
        # device breaks CPU offload by moving token ids away from the weights.
        return next(self.text_encoder.parameters()).device

    def forward(self, text_prompts: List[str]) -> dict:
        if all(p in self._cache for p in text_prompts):
            return {"prompt_embeds": torch.stack(
                [self._cache[p] for p in text_prompts]).to(self.device)}

        batch = self.tokenizer(
            text_prompts,
            padding="max_length",
            max_length=self.text_len,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        ids = batch.input_ids.to(self.device)
        mask = batch.attention_mask.to(self.device)
        context = self.text_encoder(ids, attention_mask=mask).last_hidden_state

        # Wan's DiT expects padding to be exactly zero, not whatever the encoder
        # produced for pad tokens: `text_embedding` runs over the full padded
        # length and nonzero padding leaks into cross-attention.
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = context.clone()
        for u, v in zip(context, seq_lens):
            u[v:] = 0.0

        if len(self._cache) < self.cache_limit:
            for prompt, embed in zip(text_prompts, context):
                self._cache[prompt] = embed.detach()

        return {"prompt_embeds": context}


class Wan22VAEWrapper(torch.nn.Module):
    """``AutoencoderKLWan`` (48ch, 16x16x4) behind the runtime's VAE interface.

    This is the tokenizer Cosmos3-Nano was trained against, so latents produced
    here can be handed to the Cosmos teacher unmodified -- the "latent bridge"
    that cross-model distillation normally needs reduces to the identity.
    """

    def __init__(self, root: str | Path | None = None, dtype=torch.float32):
        super().__init__()
        from diffusers import AutoencoderKLWan

        root = Path(root) if root is not None else wan22_root()
        self.model = (
            AutoencoderKLWan.from_pretrained(str(root / "vae"), torch_dtype=dtype)
            .eval()
            .requires_grad_(False)
        )
        cfg = self.model.config
        # (1, C, 1, 1, 1) so they broadcast over [B, C, F, H, W].
        self.register_buffer(
            "latents_mean",
            torch.tensor(cfg.latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(cfg.latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1),
            persistent=False,
        )
        self.z_dim = int(cfg.z_dim)
        self.scale_factor_spatial = int(cfg.scale_factor_spatial)
        self.scale_factor_temporal = int(cfg.scale_factor_temporal)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        """[B, C, F, H, W] pixels in [-1, 1] -> [B, F, z, H/16, W/16] latents."""
        self.model.clear_cache()
        dist = self.model.encode(pixel.to(self.model.dtype)).latent_dist
        z = dist.mode().float()
        mean = self.latents_mean.to(z.device)
        std = self.latents_std.to(z.device)
        z = (z - mean) / std
        return z.permute(0, 2, 1, 3, 4)

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """[B, F, z, h, w] -> [B, F, C, H, W] pixels in [-1, 1].

        ``use_cache`` mirrors the 2.1 wrapper's streaming decode. AutoencoderKLWan
        carries the causal-conv cache internally, so the only thing that differs
        is whether it is cleared first: a cached call continues the previous
        chunk's temporal state instead of restarting it.
        """
        z = latent.permute(0, 2, 1, 3, 4).float()
        mean = self.latents_mean.to(z.device)
        std = self.latents_std.to(z.device)
        z = z * std + mean

        if use_cache:
            if latent.shape[0] != 1:
                raise ValueError("batch size must be 1 when decoding with cache")
            out = self._decode_streaming(z.to(self.model.dtype))
        else:
            self.model.clear_cache()
            out = self.model.decode(z.to(self.model.dtype)).sample.float().clamp_(-1, 1)
        return out.permute(0, 2, 1, 3, 4)

    def reset_stream(self):
        """Start a new streaming decode. Call once before the first chunk."""
        self.model.clear_cache()
        self._stream_first = True

    def _decode_streaming(self, z):
        """Decode `z` continuing the previous call's temporal state.

        `AutoencoderKLWan._decode` cannot be used for this. It already decodes
        one latent frame at a time against an internal `feat_cache`, but it
        calls `clear_cache()` on entry and on exit, so every call restarts from
        a blank cache and re-runs the `first_chunk=True` path. Passing
        `use_cache=True` to it therefore did nothing: measured against a
        whole-clip decode, chunked output drifted by 10.5 mean / 255 max pixel
        levels, growing along the clip because each chunk began from the wrong
        state.
        """
        from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

        m = self.model
        if getattr(self, "_stream_first", True):
            m.clear_cache()
        x = m.post_quant_conv(z)
        outs = []
        for i in range(x.shape[2]):
            m._conv_idx = [0]
            first = bool(getattr(self, "_stream_first", True)) and i == 0
            outs.append(m.decoder(
                x[:, :, i:i + 1, :, :], feat_cache=m._feat_map,
                feat_idx=m._conv_idx, first_chunk=first))
            self._stream_first = False
        out = torch.cat(outs, dim=2)
        if m.config.patch_size is not None:
            out = unpatchify(out, patch_size=m.config.patch_size)
        return out.float().clamp_(-1, 1)
