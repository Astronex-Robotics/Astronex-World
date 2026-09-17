"""Checkpoint loading for Astronex-World releases.

The released weights are ``.safetensors``; everything this repo trains is a
``.pt`` produced by ``torch.save``. The two formats differ in one structural
way that matters here:

* a ``.pt`` checkpoint is a *nested* dict -- ``{"generator": {...},
  "generator_ema": {...}}`` -- or, for a merged export, a bare state dict;
* safetensors is flat by construction. It cannot hold nesting, so the section
  travels in the key: ``generator.model.blocks.0...``.

``load_checkpoint`` returns the nested form for either, so callers keep the
one code path they already had.
"""

from pathlib import Path

import torch

SECTIONS = ("generator", "generator_ema")


def _nest(flat: dict) -> dict:
    """Split a flat safetensors dict back into its sections.

    A file with no section prefix is a bare state dict -- a merged, ready to
    sample export -- and is returned under ``generator`` so that both release
    layouts load through the same branch.
    """
    out: dict = {}
    bare: dict = {}
    for key, value in flat.items():
        for section in SECTIONS:
            if key.startswith(section + "."):
                out.setdefault(section, {})[key[len(section) + 1:]] = value
                break
        else:
            bare[key] = value
    if bare:
        if out:
            raise ValueError(
                "checkpoint mixes prefixed and unprefixed keys; "
                f"unprefixed example: {next(iter(bare))}")
        out["generator"] = bare
    return out


def load_checkpoint(path, map_location="cpu") -> dict:
    """Load ``.safetensors`` or ``.pt``, always returning the nested form.

    A directory is treated as a sharded release: the shards named by
    ``model.safetensors.index.json`` are merged into one state dict. A 5B model
    in bf16 is ten gigabytes, which is past the five the safetensors
    convention shards at, so the released weights arrive this way rather than
    as a single file.
    """
    path = Path(path)
    device = "cpu" if map_location in (None, "cpu") else str(map_location)

    if path.is_dir():
        from safetensors.torch import load_file
        index = path / "model.safetensors.index.json"
        if index.exists():
            import json
            names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        else:
            names = sorted(f.name for f in path.glob("*.safetensors"))
        if not names:
            raise FileNotFoundError(f"no safetensors shards in {path}")
        flat = {}
        for name in names:
            flat.update(load_file(str(path / name), device=device))
        return _nest(flat)

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return _nest(load_file(str(path), device=device))
    return torch.load(str(path), map_location=map_location)


def flatten(nested: dict) -> dict:
    """Inverse of ``_nest``: nested sections -> flat, prefixed keys.

    Tensors are made contiguous and detached; safetensors refuses views, and
    an FSDP gather leaves plenty of them.
    """
    flat = {}
    for section, sd in nested.items():
        if not isinstance(sd, dict):
            continue
        for key, value in sd.items():
            if not torch.is_tensor(value):
                continue
            flat[f"{section}.{key}"] = value.detach().contiguous()
    return flat
