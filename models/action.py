"""Action conditioning, aligned with Cosmos3's ``action_gen`` channel.

Why additive per-frame modulation
---------------------------------
Cosmos3-Nano conditions on a 64-dimensional continuous action vector per frame
plus an embodiment-domain id (``action_dim=64``, ``num_embodiment_domains=32``
in its config), and packs those as tokens in its joint sequence where mRoPE
binds each action to its frame.

Wan has no joint sequence, so the equivalent binding here is the *modulation*
path: Wan's blocks already consume a per-frame ``e`` of shape [B, F, 6, dim]
(shift/scale/gate for attention and FFN), and ``CausalWanAttentionBlock`` derives
its frame stride from ``e.shape[1]``. Adding the action's contribution there
gives each latent frame exactly the control signal for that frame, with no
positional ambiguity and no extra tokens -- the cheapest possible place to put
it, and the one that cannot drift out of alignment during an autoregressive
rollout.

This is deliberately orthogonal to camera control. Camera goes through PRoPE,
which edits *attention geometry* (where tokens are relative to each other);
actions go through modulation, which edits *feature statistics* (what happens
next). Routing both through one mechanism would make them compete for the same
capacity, and PRoPE's projective structure is what gives camera control its
generalisation to unseen trajectories -- there is nothing analogous to exploit
for a robot action vector.

Zero-init and why it matters
----------------------------
``to_modulation``'s final layer is zero-initialised, so a freshly extended model
is *bit-identical* to the pretrained Wan checkpoint on the first step. Stage 0
therefore starts from Wan's real prior rather than from a perturbed one, and any
early loss spike is a bug in the data pipeline rather than an artefact of
surgery on the network.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def camera_plan_actions(viewmats: torch.Tensor, action_dim: int = 64) -> torch.Tensor:
    """Encode a known full camera trajectory without exposing future video."""
    if viewmats.dim() != 4 or viewmats.shape[-2:] != (4, 4):
        raise ValueError(f"viewmats must be [B,F,4,4], got {tuple(viewmats.shape)}")
    b, f = viewmats.shape[:2]
    first_inv = torch.linalg.inv(viewmats[:, :1].float())
    rel = viewmats.float() @ first_inv
    cur = rel[..., :3, :4].reshape(b, f, 12)
    end = rel[:, -1:, :3, :4].reshape(b, 1, 12).expand(-1, f, -1)
    velocity = (torch.diff(cur, dim=1, prepend=cur[:, :1])
                if f > 1 else torch.zeros_like(cur))
    progress = torch.linspace(0, 1, f, device=viewmats.device).view(1, f, 1).expand(b, -1, -1)
    feat = torch.cat((cur, end, end - cur, velocity, progress), dim=-1)
    out = torch.zeros((b, f, action_dim), device=viewmats.device, dtype=viewmats.dtype)
    width = min(action_dim, feat.shape[-1])
    out[..., :width] = feat[..., :width].to(out.dtype)
    return out


class ActionEmbedder(nn.Module):
    """(B, F, action_dim) [+ embodiment id] -> additive [B, F, 6, dim] modulation.

    Args:
        dim: transformer hidden size.
        action_dim: width of the continuous action vector. 64 matches Cosmos3,
            which is the point: an action encoded for the teacher can be fed to
            the student unchanged, so distillation transfers *controllability*
            and not just appearance.
        num_embodiments: size of the embodiment-domain table (32 in Cosmos3).
            0 disables it, for single-embodiment data such as camera-only video.
        hidden: width of the MLP; defaults to ``dim``.
    """

    def __init__(
        self,
        dim: int,
        action_dim: int = 64,
        num_embodiments: int = 32,
        hidden: Optional[int] = None,
    ):
        super().__init__()
        hidden = hidden or dim
        self.dim = dim
        self.action_dim = action_dim
        self.num_embodiments = num_embodiments

        self.proj = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.embodiment = (
            nn.Embedding(num_embodiments, hidden) if num_embodiments > 0 else None
        )
        self.to_modulation = nn.Linear(hidden, dim * 6)

        # Identity at init: the extended model reproduces base Wan exactly.
        nn.init.zeros_(self.to_modulation.weight)
        nn.init.zeros_(self.to_modulation.bias)
        if self.embodiment is not None:
            nn.init.zeros_(self.embodiment.weight)

    def forward(
        self,
        actions: torch.Tensor,
        embodiment_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """actions: [B, F, action_dim]; embodiment_id: [B] or [B, F] longs."""
        if actions.dim() != 3 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"actions must be [B, F, {self.action_dim}], got {tuple(actions.shape)}"
            )

        # No dtype casting here on purpose. Under FSDP mixed precision a
        # parameter's `.dtype` is its *master* dtype (fp32), while the value
        # actually served inside the forward is the reduced one (bf16), so
        # matching `weight.dtype` reliably produces the opposite of what is
        # needed. The caller casts `actions` to the activation dtype instead --
        # see apply_action_modulation -- which is the dtype FSDP serves.
        h = self.proj(actions)

        if self.embodiment is not None and embodiment_id is not None:
            emb = self.embodiment(embodiment_id.long())
            if emb.dim() == 2:  # [B, hidden] -> broadcast over frames
                emb = emb.unsqueeze(1)
            h = h + emb.to(h.dtype)

        return self.to_modulation(h).unflatten(-1, (6, self.dim))


def add_action_parameters(
    model,
    action_dim: int = 64,
    num_embodiments: int = 32,
    hidden: Optional[int] = None,
) -> ActionEmbedder:
    """Attach an :class:`ActionEmbedder` to a WanModel / CausalWanModel in place.

    Mirrors ``add_prope_parameters``: the base checkpoint is loaded first and the
    new parameters are grafted on afterwards, so ``from_pretrained`` never sees
    keys it does not know about and no ``strict=False`` load is needed anywhere.
    """
    if getattr(model, "action_embedder", None) is not None:
        return model.action_embedder

    ref = next(model.parameters())
    # One scalar per modulation slot. Zero-initialised, so tanh(0) = 0 and a
    # model that has not trained this reproduces the un-grafted network exactly
    # -- the same identity-at-init property the embedder itself relies on.
    if getattr(model, "action_cross_norm", False) and not hasattr(model, "action_gate"):
        model.register_parameter(
            "action_gate",
            torch.nn.Parameter(torch.zeros(6, 1, dtype=ref.dtype, device=ref.device)))
    embedder = ActionEmbedder(
        dim=model.dim, action_dim=action_dim, num_embodiments=num_embodiments, hidden=hidden
    ).to(device=ref.device, dtype=ref.dtype)
    model.action_embedder = embedder
    model.action_dim = action_dim
    return embedder


def apply_action_modulation(
    model,
    e0: torch.Tensor,
    actions: Optional[torch.Tensor],
    embodiment_id: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Add the action term to Wan's time modulation ``e0``.

    ``e0`` is [B, F, 6, dim] on the causal path (per-frame timesteps) and
    [B, 6, dim] on the bidirectional path (one timestep for the whole clip). In
    the second case the result is promoted to per-frame, which the blocks accept
    because they derive their frame stride from ``e.shape[1]``.
    """
    embedder = getattr(model, "action_embedder", None)
    if embedder is None or actions is None:
        return e0

    # Run under autocast rather than casting by hand. The embedder's parameters
    # may be bf16 (plain mixed-precision) or fp32 (FSDP master weights, or a
    # freshly grafted module that has not been cast), and its submodules need
    # not agree with each other. Reading `weight.dtype` to decide is actively
    # wrong under FSDP: before a submodule's forward runs, `.weight` is still the
    # fp32 master, while the value served inside the forward is the reduced one.
    # Autocast resolves each linear against whatever it is actually given.
    target = e0.dtype
    autocast_ok = actions.is_cuda and target in (torch.float16, torch.bfloat16)
    with torch.autocast(device_type="cuda", dtype=target, enabled=autocast_ok):
        delta = embedder(actions.to(target), embodiment_id)  # [B, F, 6, dim]

    if e0.dim() == 3:  # [B, 6, dim] -> [B, F, 6, dim]
        e0 = e0.unsqueeze(1).expand(-1, delta.shape[1], -1, -1)
    if e0.shape[1] != delta.shape[1]:
        raise ValueError(
            f"action frames {delta.shape[1]} do not match modulation frames {e0.shape[1]}"
        )

    # Cross-normalise the control against the branch it is being added to.
    #
    # `e0` is the timestep modulation: it tells every block how noisy the input
    # is and how hard to denoise. A raw additive delta perturbs that schedule,
    # and its usable magnitude depends on the trunk's own scale -- which is why
    # an embedder trained on one trunk wrecks another. Measured here: the
    # CrossFPS-trained branch dropped final-third sharpness to 260 on this
    # trunk against a 1151 baseline, and 125 steps of self-forcing "fixed" that
    # only by driving the action response to zero (held camera + W produced
    # +0.02% zoom, i.e. nothing). Magnitude and effect were in direct
    # opposition, with no working point between them.
    #
    # ControlNeXt (arXiv 2408.06070) reports the same failure for control
    # branches added to a pretrained backbone -- "directly adding the controls
    # to the inputs results in training collapse" -- and fixes it by
    # normalising the control features with the main branch's statistics. That
    # is what this does: the delta carries direction, `e0` sets the scale, and a
    # zero-initialised gate decides how much. The graft becomes invariant to
    # which trunk it lands on.
    # Opt-in. Every checkpoint trained before this existed carries no
    # action_gate, and tanh(0) = 0 would silently zero their action pathway --
    # the runs would still produce video, and the measurement would read as
    # "action does nothing" rather than as a broken comparison. Old checkpoints
    # keep the raw additive path until a config asks for the new one.
    gate = getattr(model, "action_gate", None)
    if gate is not None and getattr(model, "action_cross_norm", False):
        d = delta.float()
        std = d.std(dim=-1, keepdim=True).clamp_min(1e-4)
        ref = e0.float().std(dim=-1, keepdim=True)
        # tanh bounds the gate, so no amount of training can make the control
        # outgrow the modulation it is perturbing.
        delta = (d / std) * ref * torch.tanh(gate.to(d.dtype))
    return e0 + delta.to(e0.dtype)
