"""Locate the released weights.

Everything this repo runs lives inside it -- ``models/``, ``inference/``,
``post_train/``, ``utils/`` -- and everything it loads lives in one released
directory, ``../Astronex`` by default (``ASTRONEX_WEIGHTS`` overrides):

    model-*.safetensors      the denoiser, every tensor it needs
    transformer/config.json  the module definition
    vae/                     Wan2.2 VAE
    text_encoder/, tokenizer/  UMT5-xxl
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_ROOT = os.path.abspath(os.environ.get(
    "ASTRONEX_WEIGHTS", os.path.join(os.path.dirname(REPO_ROOT), "Astronex")))

COMPONENTS = ("transformer/config.json", "vae", "text_encoder", "tokenizer")


def setup():
    """Put the repo root on the path and chdir into it."""
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    os.chdir(REPO_ROOT)
    return REPO_ROOT


def child_env(weights_dir):
    """Environment for the subprocess: repo on the path, weights located."""
    env = dict(os.environ)
    env["ASTRONEX_WEIGHTS"] = weights_dir
    env["PYTHONPATH"] = os.pathsep.join(
        [REPO_ROOT, env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def resolve_config(src, dst, weights_dir=None, data=None):
    """Write `src` to `dst` with paths filled in.

    `model_root` and `"@weights"` placeholders become the weights directory,
    and a relative `data_path` is anchored at the repo root. Returns `dst`.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(src)
    weights_dir = weights_dir or weights()
    if "model_kwargs" in cfg:
        cfg.model_kwargs.model_root = weights_dir
        cfg.model_kwargs.model_name = "transformer"
    if cfg.get("generator_ckpt") == "@weights":
        cfg.generator_ckpt = weights_dir
    if data:
        cfg.data_path = os.path.abspath(data)
    elif cfg.get("data_path") and not os.path.isabs(cfg.data_path):
        cfg.data_path = os.path.join(REPO_ROOT, cfg.data_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    OmegaConf.save(cfg, dst)
    return dst


def weights(mode="causal"):
    """The released weights directory, checked for every component.

    Both inference forms run from it: 898 tensors -- the 825 of the Wan2.2
    backbone, the camera and action grafts, and the action head. Override
    with `--weights` on the CLI, or point ASTRONEX_WEIGHTS elsewhere.
    """
    if mode not in ("causal", "bidirectional"):
        raise SystemExit(f"unknown mode {mode!r}: expected causal or bidirectional")
    missing = [c for c in COMPONENTS
               if not os.path.exists(os.path.join(WEIGHTS_ROOT, c))]
    if not os.path.isdir(WEIGHTS_ROOT) or missing:
        raise SystemExit(f"weights incomplete at {WEIGHTS_ROOT} (missing: "
                         f"{', '.join(missing) or 'directory'}); download "
                         f"Astronex-Lab/Astronex-World or set ASTRONEX_WEIGHTS")
    return WEIGHTS_ROOT
