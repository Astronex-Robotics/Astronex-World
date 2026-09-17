"""Shared plumbing for the control channels: camera, actions, events.

The trainer does two things with them -- pick a dataset that carries control
labels, and move those labels onto the device before the loss call. They live
here so that the conditional and unconditional branches cannot drift apart.

The invariant these helpers exist to protect: **actions and camera must be
present in the unconditional dict too.** Classifier-free guidance is supposed to
vary the *text*, not the controls. Dropping controls from the unconditional
branch turns the guidance term into "conditioned minus uncontrolled", which
amplifies the control signal by the guidance scale -- a wildly over-steered
camera at scale 3.0, with nothing in the logs to explain it.
"""

from __future__ import annotations

from typing import Optional

import torch

from post_train.data.dataset import (
    ActionCameraLatentLMDBDataset,
    CameraLatentLMDBDataset,
)


# Substrings naming every parameter a control stage trains: the LoRA adapters
# plus the grafted control pathways. `save_trainable_only` keeps exactly these,
# so a graft missing from this tuple trains for the whole run and is then
# dropped at save time -- the checkpoint looks fine, is the right size, and has
# simply lost the thing the run existed to produce. Add new grafts here.
TRAINABLE_MARKERS = (
    "lora",
    "action_embedder",      # action conditioning (in)
    "action_head",          # action prediction (out)
    "prope_o",              # camera / PRoPE
    "camera_film",
    "history_cross_o",
    "event_embedding",      # event prompt: projection ...
    "event_cross_attn",     # ... and its own cross-attention
    "norm_event",
)


def build_control_dataset(config):
    """Dataset for a control-conditioned stage.

    ``use_action: true`` in the config selects the action-aware variant, which
    reads ``actions``/``embodiment`` from the LMDB when present and silently
    omits them when not -- so turning the flag on against a camera-only dataset
    degrades to camera-only training rather than crashing, and the log line
    below is how you find out that is what happened.
    """
    use_action = bool(getattr(config, "use_action", False))
    cls = ActionCameraLatentLMDBDataset if use_action else CameraLatentLMDBDataset
    # Control2V carries poses and no actions, so `use_action` alone degrades it
    # to camera-only and the action branch sees nothing. Its poses decompose to
    # exactly the +-0.08 / +-3 degrees of camera_trajectory._MOTIONS, so the
    # WASD behind them can be recovered and handed to the action branch.
    from_poses = bool(getattr(config, "actions_from_poses", False))

    def _make(path, max_pair):
        source = cls(path, max_pair=int(max_pair))
        source.actions_from_poses = from_poses
        return source

    data_paths = getattr(config, "data_paths", None)
    if not data_paths:
        return _make(config.data_path, int(1e8))

    # Long streaming clips are scarce, while using them alone caused a strong
    # CrossFPS texture/exposure domain shift.  Allow a deterministic mixture
    # of LMDBs without physically copying multi-gigabyte datasets. Repetition
    # controls source weight; max_pairs can cap a large short-clip source.
    from torch.utils.data import ConcatDataset, Subset
    paths = list(data_paths)
    repeats = list(getattr(config, "data_repeats", [1] * len(paths)))
    max_pairs = list(getattr(config, "data_max_pairs", [int(1e8)] * len(paths)))
    if not (len(paths) == len(repeats) == len(max_pairs)):
        raise ValueError("data_paths, data_repeats and data_max_pairs must have equal lengths")
    parts = []
    for path, repeat, max_pair in zip(paths, repeats, max_pairs):
        source = _make(path, int(max_pair))
        if int(max_pair) < len(source):
            source = Subset(source, range(int(max_pair)))
        parts.extend([source] * int(repeat))
    if not parts:
        raise ValueError("dataset mixture is empty")
    return ConcatDataset(parts)


def describe_control(dataset) -> str:
    """One line saying which control channels the data actually carries."""
    try:
        sample = dataset[0]
    except Exception as exc:  # pragma: no cover - diagnostics only
        return f"could not inspect dataset: {exc}"
    channels = [k for k in ("viewmats", "Ks", "actions", "embodiment_id") if k in sample]
    return "control channels present in data: " + (", ".join(channels) or "none")


def control_tensors(batch, device, dtype, config=None) -> dict:
    """Pull the control channels out of a batch and move them to the device.

    When ``config.use_action`` is set, a missing action stream is replaced by
    zeros rather than left absent. That is not cosmetic. Under FSDP every rank
    must execute the same modules in the same order: the collectives are
    matched by call sequence, not by name. ``apply_action_modulation`` is a
    no-op when ``actions`` is None, so a batch that mixes sources -- Control2V
    rows carry no actions, DROID rows do -- makes rank 0 skip the action
    embedder while rank 1 runs it, and the two ranks issue *different*
    collectives at the same sequence number.

    That failure does not raise. It hangs, for the full 30-minute NCCL timeout,
    and only then reports a mismatch between an ALLREDUCE on one rank and a
    REDUCE_SCATTER on the other -- with nothing pointing at the data.

    Zeros are also the semantically right filler: the action graft is
    zero-initialised precisely so that a zero action is the identity.
    """
    out = {}
    if "viewmats" in batch:
        out["viewmats"] = batch["viewmats"].to(device=device, dtype=dtype)
    if "Ks" in batch:
        out["Ks"] = batch["Ks"].to(device=device, dtype=dtype)
    if "actions" in batch:
        out["actions"] = batch["actions"].to(device=device, dtype=dtype)
    if "embodiment_id" in batch:
        out["embodiment_id"] = batch["embodiment_id"].to(device=device).long()

    if config is not None and getattr(config, "use_action", False):
        if getattr(config, "camera_plan_condition", False) and "viewmats" in out:
            from models.action import camera_plan_actions
            dim = int(getattr(config, "model_kwargs", {}).get("action_dim", 64))
            out["actions"] = camera_plan_actions(out["viewmats"], dim)
        if "actions" not in out:
            shape = list(config.image_or_video_shape)
            batch_size = len(batch["prompts"]) if "prompts" in batch else shape[0]
            frames = shape[1]
            dim = int(getattr(config, "model_kwargs", {}).get("action_dim", 64))
            out["actions"] = torch.zeros(
                (batch_size, frames, dim), device=device, dtype=dtype)
        if "embodiment_id" not in out:
            out["embodiment_id"] = torch.zeros(
                (out["actions"].shape[0],), device=device, dtype=torch.long)
    return out


def extend_controls(control: dict, target_frames: int) -> dict:
    """Continue the camera trajectory (and hold the actions) out to `target_frames`.

    Self-forcing needs no ground-truth video past the conditioning frame -- the
    student rolls out on its own predictions and DMD scores that rollout against
    the teacher's distribution, not against a paired clip. What a longer rollout
    *does* need is a control signal for every frame it generates, and the camera
    datasets here are all 20 latents (Control2V, its variants, and CrossFPS,
    whose poses are identity anyway).

    So the poses are continued rather than fetched: take the last frame-to-frame
    camera-centre step and keep applying it, holding the rotation. The result is
    a constant-velocity extension of whatever the clip was doing. That is not a
    liberty -- it is the inference distribution: every trajectory this model is
    driven with at test time is synthesised the same way, by parse_trajectory or
    by WBench's case_to_poses, both of which emit constant per-frame deltas.

    Actions hold their last command instead of being zeroed: a zero action is
    "stick released", a distinct command, and the rollout would read the
    extension as the player letting go.
    """
    import torch as _t
    out = dict(control)
    vm = out.get("viewmats")
    if vm is None or vm.shape[1] >= target_frames:
        return out
    have = vm.shape[1]
    if have < 2:
        out["viewmats"] = vm[:, :1].repeat(1, target_frames, 1, 1)
    else:
        c2w = _t.linalg.inv(vm.float())
        step = c2w[:, -1] @ _t.linalg.inv(c2w[:, -2])   # last relative motion
        tail = [c2w[:, -1]]
        for _ in range(target_frames - have):
            tail.append(step @ tail[-1])
        ext = _t.stack(tail[1:], dim=1)
        out["viewmats"] = _t.cat(
            [vm, _t.linalg.inv(ext).to(vm.dtype)], dim=1)
    for key in ("Ks", "actions"):
        v = out.get(key)
        if v is not None and v.shape[1] < target_frames:
            pad = v[:, -1:].repeat(1, target_frames - v.shape[1],
                                   *([1] * (v.dim() - 2)))
            out[key] = _t.cat([v, pad], dim=1)
    return out


def inject_control(
    conditional_dict: dict,
    unconditional_dict: Optional[dict],
    batch,
    device,
    dtype,
    config=None,
    text_prompts=None,
    event_encoder=None,
) -> dict:
    """Put every control channel on both dicts; return the control tensors.
    """
    control = control_tensors(batch, device, dtype, config)
    # Extend once, to the longest rollout that can be drawn, so any drawn
    # length is covered without re-deriving this per step.
    target = int(getattr(config, "num_training_frames", 0) or 0)
    if target:
        control = extend_controls(control, target)
    conditional_dict.update(control)
    if unconditional_dict is not None:
        unconditional_dict.update(control)

    # The event channel is text, so it is encoded rather than moved; the
    # encoder is passed in because it belongs to the model, not to the batch.
    inject_event(conditional_dict, unconditional_dict, batch,
                 event_encoder, device, dtype, config)

    return control


def inject_event(
    conditional_dict: dict,
    unconditional_dict: Optional[dict],
    batch,
    encoder,
    device,
    dtype,
    config=None,
) -> None:
    """Encode the batch's event prompts into ``event_embeds`` on both dicts.

    The event prompt is a second, independent text channel: it reaches the DiT
    through its own cross-attention (``enable_event_conditioning``) rather than
    being concatenated onto the caption, so it can be dropped, swapped or
    guided without touching the caption -- which is the whole point of having
    it. Concatenating instead would make "no event" and "an event" different
    lengths of one string and give CFG no handle on the event alone.

    ``event_guidance`` decides what the unconditional branch sees:

    * ``False`` (default) -- the same event embedding as the conditional
      branch. Guidance then varies the caption only, which is the invariant the
      rest of this module protects.
    * ``True`` -- the empty event embedding, making event strength a second
      guidance axis at inference. Only turn this on if you actually want the
      event over-steered relative to the caption; it is the same failure mode
      as guiding the camera, just on a channel where it is sometimes wanted.
    """
    if encoder is None:
        return
    prompts = batch.get("event_prompts") if hasattr(batch, "get") else None
    if prompts is None:
        return
    prompts = [prompts] if isinstance(prompts, str) else list(prompts)

    # Whether the branch runs must not depend on what this rank happened to
    # draw. Under FSDP every rank executes the same modules in the same order --
    # the collectives are matched by call sequence, not by name -- so on mixed
    # data, a rank whose rows all have blank event text would skip the event
    # cross-attention while its peers run it, and the two issue different
    # collectives at the same sequence number. That does not raise. It hangs for
    # the full NCCL timeout and then reports a mismatch with nothing pointing at
    # the data.
    #
    # So with the channel enabled, every rank always attaches an embedding, and
    # rows with no event text get the embedding of the empty string. The branch
    # is zero-initialised at `o`, so that is the identity at the start of
    # training and a learned near-no-op after it.
    always_on = config is not None and getattr(config, "use_event", False)
    if not any(p and p.strip() for p in prompts) and not always_on:
        # Inference, or a stage with the channel off: leave `event_embeds`
        # absent so the branch is skipped entirely, which is both cheaper and
        # exactly what an event-free run computes.
        return

    with torch.no_grad():
        embeds = encoder(text_prompts=prompts)["prompt_embeds"]
    embeds = embeds.to(device=device, dtype=dtype)
    conditional_dict["event_embeds"] = embeds

    if unconditional_dict is None:
        return
    if config is not None and getattr(config, "event_guidance", False):
        with torch.no_grad():
            empty = encoder(text_prompts=[""] * len(prompts))["prompt_embeds"]
        unconditional_dict["event_embeds"] = empty.to(device=device, dtype=dtype)
    else:
        unconditional_dict["event_embeds"] = embeds


def assert_action_shape(actions: torch.Tensor, num_frames: int, action_dim: int) -> None:
    """Fail loudly on a control stream that is misaligned with the latents.

    An action tensor with the wrong frame count is the single easiest way to
    train a model that appears to work and ignores its controls: broadcasting
    will happily accept [B, 1, D] and apply one action to the whole clip.
    """
    if actions.dim() != 3:
        raise ValueError(f"actions must be [B, F, D], got {tuple(actions.shape)}")
    if actions.shape[1] != num_frames:
        raise ValueError(
            f"actions cover {actions.shape[1]} frames but the latents have {num_frames}; "
            "a per-clip action silently becomes per-frame under broadcasting"
        )
    if actions.shape[2] != action_dim:
        raise ValueError(
            f"actions are {actions.shape[2]}-wide but the embedder expects {action_dim}"
        )
