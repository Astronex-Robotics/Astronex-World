"""Check that the Wan2.2 components Astronex needs are present.

The released Astronex weights are the denoiser only. Generation also needs the
Wan2.2 VAE and the UMT5-xxl text encoder, and training needs the VAE to encode
targets. They resolve from two different roots, which is the thing that trips
people up:

* the *backbone* config and its shards come from `model_root` in the config --
  `ckpts/Wan22/Wan2.2-TI2V-5B`. Astronex replaces these weights, but the
  `config.json` beside them is still what builds the module.
* the *VAE, tokenizer and text encoder* come from the diffusers-format release
  at `WAN22_DIFFUSERS_ROOT`, default `ckpts/wan2.2-ti2v-5b` -- a different
  directory with a different spelling.

Run this before a first generation; a missing text encoder otherwise surfaces
several minutes in, as a tokenizer error.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import astronex_env  # noqa: E402

BACKBONE = ("config.json", "diffusion_pytorch_model.safetensors.index.json")
DIFFUSERS = ("tokenizer", "text_encoder", "vae")


def main():
    root = astronex_env.setup()
    ok = True

    backbone = os.path.join(root, "ckpts", "Wan22", "Wan2.2-TI2V-5B")
    print(f"backbone config   {backbone}")
    for f in BACKBONE:
        here = os.path.exists(os.path.join(backbone, f))
        ok &= here
        print(f"  {'ok  ' if here else 'MISS'}  {f}")

    diff = os.environ.get("WAN22_DIFFUSERS_ROOT", "ckpts/wan2.2-ti2v-5b")
    diff_abs = diff if os.path.isabs(diff) else os.path.join(root, diff)
    print(f"vae / text encoder {diff_abs}")
    for d in DIFFUSERS:
        here = os.path.isdir(os.path.join(diff_abs, d))
        ok &= here
        print(f"  {'ok  ' if here else 'MISS'}  {d}/")

    # One directory serves both forms: the causal checkpoint is a superset of
    # what full-attention sampling needs (see astronex_env.weights).
    try:
        w = astronex_env.weights("causal")
        shards = [f for f in os.listdir(w) if f.endswith(".safetensors")]
        print(f"weights (causal + bidirectional) {w}  ({len(shards)} shard(s))")
        ok &= bool(shards)
    except SystemExit as e:
        print(f"weights MISSING -- {e}")
        ok = False

    print("\nall present" if ok else "\nsomething is missing; see above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
