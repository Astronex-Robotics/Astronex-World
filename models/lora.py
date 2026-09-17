"""LoRA with multiple adapters over one shared frozen base.

This module is what makes distillation fit on a 16 GB card, and the reason is
structural rather than incidental.

DMD keeps three networks resident: the student, a critic that tracks the
student's current output distribution, and the frozen teacher. The critic is
*initialised from the student* -- that is not an implementation shortcut, it is
what the method prescribes, since the critic has to model whatever the student
currently produces.

So with LoRA, the student and the critic are the same frozen base weights plus
two different adapters. Keeping one copy of the base and swapping adapters
removes an entire model from VRAM: 10 GB for a 5B student in bf16, which is
precisely the difference between fitting and not.

Budget:

    naive:   student base 10.0 + critic base 10.0 + adapters 1.0  = 21.0 GB
    shared:  shared base  10.0 +                     adapters 1.0 = 11.0 GB
    shared, FP8 base:                                             =  6.0 GB

The adapter stays in bf16 even when the base is quantised: the base is frozen so
its precision only costs accuracy, while the adapter carries every gradient and
wants the dynamic range.
"""

from __future__ import annotations

import logging

import math
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    rank: int = 32
    alpha: float = 64.0  # scaling = alpha / rank
    dropout: float = 0.0
    # Substrings of module names to adapt. Attention and MLP projections carry
    # nearly all the parameters and nearly all the useful adaptation.
    target_patterns: tuple[str, ...] = ("qkv", "to_q", "to_kv", "proj", "mlp.0", "mlp.2")
    # Never adapt these. Same sensitivity argument as quantisation: they are
    # tiny, so adapting them saves nothing, and they are load-bearing.
    exclude_patterns: tuple[str, ...] = ("final", "t_embed", "modulation", "shift_table")
    # Adapters that must exist from the start. For DMD: the student and the critic.
    adapters: tuple[str, ...] = ("student", "critic")
    init_scale: float = 0.01
    # Per-layer rank overrides as ``(name substring, rank)``, first match wins.
    # TMD's flow head is recurrent: the same blocks are unrolled N times and have
    # to behave differently at each inner step, while the backbone is only a
    # semantic feature extractor run once per outer transition. A uniform rank
    # spends the same capacity on both. See :func:`head_rank_overrides`.
    rank_overrides: tuple[tuple[str, int], ...] = ()

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    def rank_for(self, name: str) -> int:
        for pattern, rank in self.rank_overrides:
            if pattern in name:
                return rank
        return self.rank


def head_rank_overrides(depth: int, head_blocks: int, rank: int) -> tuple[tuple[str, int], ...]:
    """Give the last ``head_blocks`` transformer blocks their own LoRA rank.

    Widening only the head keeps every existing adapter loadable: ranks grow
    rather than shrink, and :func:`load_adapter_state_dict` zero-pads, so a
    checkpoint trained at uniform rank is preserved exactly.
    """
    if not 0 < head_blocks < depth:
        raise ValueError("head_blocks must be between zero and depth")
    return tuple((f"blocks.{i}.", rank) for i in range(depth - head_blocks, depth))


class LoRALinear(nn.Module):
    """A frozen base ``nn.Linear`` plus a set of named low-rank adapters.

    Only one adapter is active at a time. ``None`` means the base alone, which is
    what an unconditional or reference pass wants.
    """

    def __init__(self, base: nn.Linear, cfg: LoRAConfig, rank: int | None = None):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.cfg = cfg
        self.rank = cfg.rank if rank is None else rank
        # Deliberately alpha/cfg.rank, not alpha/self.rank: the scaling is held
        # fixed across per-layer ranks so that widening a layer leaves the
        # already-trained directions contributing exactly what they did before.
        # Dividing by the layer's own rank would rescale a loaded adapter.
        self.scaling = cfg.scaling
        self.in_features = base.in_features
        self.out_features = base.out_features

        self.lora_a = nn.ParameterDict()
        self.lora_b = nn.ParameterDict()
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.active: str | None = None

        for name in cfg.adapters:
            self.add_adapter(name)
        if cfg.adapters:
            self.active = cfg.adapters[0]

    def add_adapter(self, name: str) -> None:
        if name in self.lora_a:
            raise KeyError(f"adapter {name!r} already exists on this layer")
        device = self.base.weight.device
        # A is Kaiming-initialised and B is zero, so the adapter starts as an
        # exact no-op. The model is bit-identical to the frozen base at step 0,
        # which is what lets a pretrained backbone be fine-tuned without an
        # initial quality dip.
        a = nn.Parameter(torch.empty(self.rank, self.in_features, device=device))
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        b = nn.Parameter(torch.zeros(self.out_features, self.rank, device=device))
        self.lora_a[name] = a
        self.lora_b[name] = b

    def set_active(self, name: str | None) -> None:
        if name is not None and name not in self.lora_a:
            raise KeyError(f"unknown adapter {name!r}; have {sorted(self.lora_a)}")
        self.active = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.active is None:
            return out
        a, b = self.lora_a[self.active], self.lora_b[self.active]
        # Cast to the adapter dtype: the base may be FP8/INT8 while the adapter
        # is bf16, and the low-rank product must not inherit the base's
        # quantisation error.
        delta = self.dropout(x.to(a.dtype)) @ a.t() @ b.t()
        return out + delta.to(out.dtype) * self.scaling

    def merged_weight(self, name: str) -> torch.Tensor:
        """Base + adapter, for exporting a standalone deployable model."""
        a, b = self.lora_a[name], self.lora_b[name]
        return self.base.weight + (b @ a) * self.scaling

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, rank={self.rank}, "
            f"adapters={sorted(self.lora_a)}, active={self.active}"
        )


# ---------------------------------------------------------------------------
# Model surgery
# ---------------------------------------------------------------------------


def _matches(name: str, cfg: LoRAConfig) -> bool:
    if any(p in name for p in cfg.exclude_patterns):
        return False
    return any(p in name for p in cfg.target_patterns)


def apply_lora(model: nn.Module, cfg: LoRAConfig | None = None) -> tuple[nn.Module, dict[str, float]]:
    """Replace matching ``nn.Linear`` layers with :class:`LoRALinear`, in place.

    Freezes every base parameter in the model, not just the adapted ones --
    anything left trainable would silently reintroduce the full-fine-tune
    optimiser cost that this whole approach exists to avoid.
    """
    cfg = cfg or LoRAConfig()

    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and _matches(name, cfg)
    ]
    if not targets:
        raise ValueError(
            f"no modules matched {cfg.target_patterns} (excluding {cfg.exclude_patterns}). "
            f"Check the patterns against the model's actual layer names."
        )

    for param in model.parameters():
        param.requires_grad_(False)

    for name, module in targets:
        parent_path, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, attr, LoRALinear(module, cfg, rank=cfg.rank_for(name)))

    base_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    adapter_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    per_adapter = adapter_params / max(len(cfg.adapters), 1)

    stats = {
        "adapted_layers": float(len(targets)),
        "base_params_b": base_params / 1e9,
        "adapter_params_total_b": adapter_params / 1e9,
        "adapter_params_each_b": per_adapter / 1e9,
        "trainable_fraction": per_adapter / max(base_params, 1),
    }
    return model, stats


def set_adapter(model: nn.Module, name: str | None) -> None:
    """Switch every LoRA layer in the model to ``name``."""
    found = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.set_active(name)
            found = True
    if not found:
        raise ValueError("no LoRALinear layers found; was apply_lora called?")


@contextmanager
def using_adapter(model: nn.Module, name: str | None):
    """Temporarily activate an adapter, restoring the previous one afterwards.

    The DMD step needs this: within one iteration the same weights are read as
    the generator and as the critic. Forgetting to switch back is a silent bug
    -- the critic's gradient would be applied through the generator's adapter
    and both would slowly converge to the same thing.
    """
    previous = [
        (m, m.active) for m in model.modules() if isinstance(m, LoRALinear)
    ]
    try:
        set_adapter(model, name)
        yield model
    finally:
        for module, old in previous:
            module.active = old


class AdapterView(nn.Module):
    """Presents one adapter of a shared base as if it were a standalone model.

    DMD's machinery wants a ``critic`` object it can call. With a shared base
    there is no separate critic module -- only a second adapter over the same
    weights. This wrapper closes that gap: calling it activates its adapter for
    the duration of the forward pass and restores the previous one afterwards.

    The restore matters. Within one DMD iteration the same weights are read as
    generator and as critic; leaking the critic's adapter into the generator
    pass would apply the critic's updates through the generator and collapse the
    adversarial structure the method depends on.
    """

    def __init__(self, root: nn.Module, callable_module: nn.Module, adapter: str):
        super().__init__()
        # Assigned via object.__setattr__ so nn.Module does not register them as
        # children -- otherwise the shared base would appear in this wrapper's
        # parameters() and be double-counted by any optimiser built from it.
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_target", callable_module)
        self.adapter = adapter

    def forward(self, *args, **kwargs):
        with using_adapter(self._root, self.adapter):
            return self._target(*args, **kwargs)

    def adapter_parameters(self) -> list[nn.Parameter]:
        return adapter_parameters(self._root, self.adapter)

    def __getattr__(self, name):
        # The view forwards calls, but DMD also reads plain attributes off what
        # it thinks is a model -- `use_camera`, the scheduler hooks. Registered
        # params, buffers and children go through nn.Module as usual; anything
        # else falls through to the shared module, so the view does not have to
        # restate a surface it is only standing in for.
        try:
            return super().__getattr__(name)
        except AttributeError:
            target = object.__getattribute__(self, "__dict__").get("_target")
            if target is None:
                raise
            return getattr(target, name)

    def extra_repr(self) -> str:
        return f"adapter={self.adapter!r} over a shared base"


def adapter_parameters(model: nn.Module, name: str) -> list[nn.Parameter]:
    """Parameters of one adapter, for building a per-adapter optimiser.

    The student and the critic must have separate optimisers. Sharing one would
    let the critic's updates drive the generator, which breaks the adversarial
    structure DMD depends on.
    """
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, LoRALinear) and name in module.lora_a:
            params.append(module.lora_a[name])
            params.append(module.lora_b[name])
    if not params:
        raise KeyError(f"no parameters found for adapter {name!r}")
    return params


def adapter_state_dict(model: nn.Module, name: str) -> dict[str, torch.Tensor]:
    """Just the adapter, which is what should be checkpointed.

    Two orders of magnitude smaller than the full model, and it keeps the
    pretrained base as a single shared artefact rather than copying it into
    every checkpoint.
    """
    out: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        if isinstance(module, LoRALinear) and name in module.lora_a:
            # .clone() is load-bearing. On CPU, .cpu() is a no-op that returns the
            # *same* tensor, so without it this "snapshot" aliases the live
            # parameters: training on after a save silently rewrites the saved
            # checkpoint. Only visible when the model is already on CPU, which is
            # exactly where tests and small runs live.
            out[f"{module_name}.lora_a"] = module.lora_a[name].detach().cpu().clone()
            out[f"{module_name}.lora_b"] = module.lora_b[name].detach().cpu().clone()
    return out


def load_adapter_state_dict(model: nn.Module, name: str, state: dict[str, torch.Tensor]) -> None:
    """Load an adapter, allowing a layer to have been widened since it was saved.

    A saved rank smaller than the layer's current rank is written into the
    leading rank slots and the rest is left at its init -- and since ``B`` is
    zero-initialised, those extra directions contribute nothing, so the loaded
    model is functionally identical to the one that was saved. This is what lets
    the flow head's rank be raised mid-pipeline without discarding stage 1.
    Shrinking is refused: truncating trained directions is not a no-op.
    """
    loaded = 0
    widened = 0
    for module_name, module in model.named_modules():
        if isinstance(module, LoRALinear) and f"{module_name}.lora_a" in state:
            if name not in module.lora_a:
                module.add_adapter(name)
            saved_a = state[f"{module_name}.lora_a"]
            saved_b = state[f"{module_name}.lora_b"]
            saved_rank = saved_a.shape[0]
            if saved_rank > module.rank:
                raise ValueError(
                    f"{module_name}: checkpoint rank {saved_rank} exceeds the layer's "
                    f"rank {module.rank}; shrinking a LoRA adapter would drop trained "
                    "directions rather than preserve the saved model"
                )
            with torch.no_grad():
                module.lora_a[name][:saved_rank].copy_(saved_a)
                module.lora_b[name][:, :saved_rank].copy_(saved_b)
                if saved_rank < module.rank:
                    # Keep the widened part an exact no-op at load time.
                    module.lora_b[name][:, saved_rank:].zero_()
                    widened += 1
            loaded += 1
    if loaded == 0:
        raise ValueError(
            "no adapter tensors matched the model. The checkpoint was probably saved "
            "from a different architecture or a different set of target_patterns."
        )


def load_checkpoint_state(path: str | Path) -> tuple[str, dict[str, torch.Tensor], int]:
    """Return (mode, state, lora_rank) from a train_causal_adapt checkpoint.

    ``mode`` is ``"full"`` for a full-fine-tuned model or ``"lora"`` for an
    adapter checkpoint. The LoRA rank is read from the file when present and
    otherwise inferred from the first ``lora_a`` tensor; full checkpoints report
    rank 0 because there is no adapter to size.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "model" in blob:
        return "full", blob["model"], 0

    state = blob.get("adapters", blob) if isinstance(blob, dict) else blob
    if not state:
        raise ValueError(f"{path} contains no model or adapter tensors")
    rank = int(blob.get("args", {}).get("lora_rank", 0))
    if not rank:
        a = next(v for k, v in state.items() if "lora_a" in k)
        rank = a.shape[0]
    return "lora", state, rank


@torch.no_grad()
def merge_adapter(model: nn.Module, name: str) -> nn.Module:
    """Fold an adapter into the base weights and drop the LoRA wrappers.

    Do this before deployment: a merged model has zero adapter overhead at
    inference and exports cleanly to ONNX/TensorRT, which cannot represent the
    adapter-switching indirection.
    """
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        merged = nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.base.bias is not None,
            device=module.base.weight.device,
            dtype=module.base.weight.dtype,
        )
        merged.weight.copy_(module.merged_weight(name).to(merged.weight.dtype))
        if module.base.bias is not None:
            merged.bias.copy_(module.base.bias)
        parent_path, _, attr = module_name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, attr, merged)
    return model


def memory_report(stats: dict[str, float], num_adapters: int = 2, base_dtype: str = "bf16") -> str:
    # Bytes per parameter, by storage dtype. Inlined rather than imported so
    # this module has no dependency outside torch.
    DTYPE_BYTES = {"fp32": 4.0, "bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.0}

    base_gb = stats["base_params_b"] * DTYPE_BYTES[base_dtype]
    # 18 bytes/param of optimiser state, but only on the adapters.
    adapters_gb = stats["adapter_params_each_b"] * 18 * num_adapters
    naive_gb = (base_gb + stats["adapter_params_each_b"] * 18) * num_adapters
    return "\n".join(
        [
            f"adapted layers        {int(stats['adapted_layers'])}",
            f"frozen base           {stats['base_params_b']:.2f}B  ({base_gb:.1f} GB @ {base_dtype})",
            f"adapter, each         {stats['adapter_params_each_b'] * 1000:.1f}M  "
            f"({stats['trainable_fraction']:.2%} of base)",
            f"adapters x{num_adapters} w/ Adam    {adapters_gb:.1f} GB",
            "",
            f"shared base total     {base_gb + adapters_gb:.1f} GB",
            f"separate bases would  {naive_gb:.1f} GB",
            f"saved                 {naive_gb - base_gb - adapters_gb:.1f} GB",
        ]
    )
