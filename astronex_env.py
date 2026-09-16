"""Locate the minWM source tree and the Wan2.2 component checkpoints.

The released weights in `../Astronex` are the generator only. Generation also
needs the Wan2.2 VAE and the UMT5 text encoder, which live in the minWM
checkout under `ckpts/Wan22` and are not duplicated here.

Set MINWM_ROOT to point elsewhere; the default is the sibling checkout.
"""

import os
import sys

MINWM_ROOT = os.environ.get(
    "MINWM_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "minWM"))

WEIGHTS_ROOT = os.environ.get(
    "ASTRONEX_WEIGHTS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "Astronex"))


def setup():
    """Put minWM's packages on the path and chdir into it.

    The chdir is not decoration: `model_root: ckpts/Wan22` in every config is
    relative, and the VAE and text encoder are resolved against the working
    directory.
    """
    for sub in ("Wan21", "shared"):
        p = os.path.join(MINWM_ROOT, sub)
        if not os.path.isdir(p):
            raise SystemExit(
                f"{p} not found. Point MINWM_ROOT at the minWM checkout "
                f"(currently {MINWM_ROOT}).")
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(MINWM_ROOT)
    return MINWM_ROOT


def weights(mode):
    """Directory holding the safetensors shards for `mode`.

    Both inference forms run from the same released directory. The causal
    checkpoint is a strict superset of what full-attention sampling needs --
    892 tensors against 885, and the extra seven are `action_embedder`, which
    the bidirectional config leaves unused because it sets `use_action: false`.
    Load it with `--weights` on the CLI, or point ASTRONEX_WEIGHTS elsewhere.
    """
    if mode not in ("causal", "bidirectional"):
        raise SystemExit(f"unknown mode {mode!r}: expected causal or bidirectional")
    if not os.path.isdir(WEIGHTS_ROOT):
        raise SystemExit(f"weights not found at {WEIGHTS_ROOT}")
    return WEIGHTS_ROOT
