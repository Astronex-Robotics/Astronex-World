# Wan2.2 components

Astronex replaces one thing in the Wan2.2 stack — the denoiser — and reuses
the rest. What the release ships and what it borrows:

| | comes from |
|---|---|
| denoiser (5.35B, 892 tensors) | `../../Astronex`, this release |
| VAE (48-channel, 4x16x16) | Wan2.2 diffusers release |
| UMT5-xxl text encoder | Wan2.2 diffusers release |
| module definition (`config.json`) | `ckpts/Wan22/Wan2.2-TI2V-5B` |

```bash
python wan/check_components.py
```

## Two roots, two spellings

This is the one piece of the layout worth reading before a first run. The
components resolve from **two different directories**:

- **`ckpts/Wan22/Wan2.2-TI2V-5B`** — set by `model_root` in the configs. Holds
  the backbone `config.json` and the original shards. Astronex supplies the
  weights, but this `config.json` is still what constructs the module, so the
  directory has to exist even though its shards go unused.
- **`ckpts/wan2.2-ti2v-5b`** — the diffusers-format release, holding
  `tokenizer/`, `text_encoder/` and `vae/`. Overridden with
  `WAN22_DIFFUSERS_ROOT`.

Note the different capitalisation and punctuation. They are separate paths, not
two names for one directory, and a missing text encoder does not fail at
startup — it fails several minutes in, as a tokenizer error, after the denoiser
has already loaded.

## Latent geometry

The VAE compresses 4x temporally and 16x16 spatially into 48 channels, so a
latent frame is 4 RGB frames and the tensors are
`(batch, frames, 48, height/16, width/16)`. At the released resolution that is
`(1, F, 48, 30, 52)` — 480x832 RGB — and one latent frame is 390 tokens after
the 2x2 patch embedding.

The frame counts in the configs are therefore *latent* frames: 20 latent frames
is 77 RGB frames, a little over three seconds at 24fps. This is also why the
causal window of 20 is described as ~3.3s of history.

## Why the VAE is fp32 by default

`build_vae` loads it in fp32 because training uses it to *encode* targets and
those feed a loss. For inference only, `vae_dtype: bfloat16` in the config is
worth setting: decoding was measured at 52.6% of total generation time, more
than all the denoising steps combined, and nothing downstream of the decode is
differentiated.
