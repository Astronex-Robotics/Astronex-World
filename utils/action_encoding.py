"""Build the 64-wide action vectors the CrossFPS-trained embedder reads.

The lane layout comes from build_crossfps_lmdb.py:

    [0:2]   left stick   (LX, LY)  -- movement, continuous in [-1, 1]
    [2:4]   right stick  (RX, RY)  -- camera,   continuous in [-1, 1]
    [4:21]  17 buttons, binary
    [21:]   zero, reserved

The dataset ships no sign convention, so each axis was read off the footage by
correlating the recorded stick value against optical flow. Every number below
is that measurement, not an assumption:

    LX  positive = strafe right.  LX +0.81 -> content drifts left  (-157 px)
                                  LX -0.65 -> content drifts right (+85 px)
    LY  positive = move forward.  LY +0.91 -> +150% radial zoom
                                  LY -0.91 ->  +13% (players rarely back up,
                                  so the negative end is weak evidence)
    RX  positive = turn right.    n=20, r = -0.707 against horizontal flow
    RY  positive = look down.     n=16, r = -0.640 against vertical flow,
                                  measured on clips where both left-stick axes
                                  and RX are near zero so pitch is isolated

Frame 0 is always all-zero. The encoder does the same thing, for the reason its
docstring gives: nothing has happened at t=0, and seeding it with the first
command shifts the whole control stream one frame early.
"""
from __future__ import annotations

import numpy as np

ACTION_DIM = 64
LX, LY, RX, RY = 0, 1, 2, 3
N_BUTTONS = 17
BUTTON_BASE = 4

# WBench's navigation vocabulary onto the sticks. Its W/A/S/D are translations
# and its arrows are rotations, which is exactly the left/right stick split.
WBENCH_ACTION = {
    "W": {LY: +1.0}, "S": {LY: -1.0},
    "D": {LX: +1.0}, "A": {LX: -1.0},
    "right": {RX: +1.0}, "left": {RX: -1.0},
    "down": {RY: +1.0}, "up": {RY: -1.0},
}

# A keyboard/mouse binding over the same lanes, so a live session can drive the
# model with the keys players actually use.
KEYBOARD = {
    "w": {LY: +1.0}, "s": {LY: -1.0},
    "d": {LX: +1.0}, "a": {LX: -1.0},
    "arrowright": {RX: +1.0}, "arrowleft": {RX: -1.0},
    "arrowdown": {RY: +1.0}, "arrowup": {RY: -1.0},
}


def from_axes(frames: int, lx=0.0, ly=0.0, rx=0.0, ry=0.0,
              buttons=None) -> np.ndarray:
    """(frames, 64) holding one constant stick position, frame 0 excepted."""
    out = np.zeros((frames, ACTION_DIM), dtype=np.float32)
    if frames > 1:
        out[1:, LX] = lx
        out[1:, LY] = ly
        out[1:, RX] = rx
        out[1:, RY] = ry
        for b in (buttons or ()):
            if not 0 <= b < N_BUTTONS:
                raise ValueError(f"button index {b} outside 0..{N_BUTTONS - 1}")
            out[1:, BUTTON_BASE + b] = 1.0
    return out


def from_tokens(tokens, table=WBENCH_ACTION, magnitude: float = 1.0) -> np.ndarray:
    """(len(tokens)+1, 64) from one action name per latent frame.

    ``tokens`` is a per-frame sequence, so a four-turn case is expanded to its
    frames before being passed here rather than being summarised into one
    vector. Unknown names raise: a silently zeroed action is a held stick, which
    is a legitimate command, and would hide a typo as "the model ignored it".
    """
    tokens = list(tokens)
    out = np.zeros((len(tokens) + 1, ACTION_DIM), dtype=np.float32)
    for i, tok in enumerate(tokens):
        for part in str(tok).split("+"):
            part = part.strip()
            if not part or part in ("h", "hold", "none"):
                continue
            if part not in table:
                raise ValueError(f"unknown action {part!r}; known: {sorted(table)}")
            for lane, v in table[part].items():
                # Compound actions address distinct lanes; a repeat would mean
                # two commands on one axis, which the sticks cannot express.
                if out[i + 1, lane] != 0.0:
                    raise ValueError(f"{tok!r} drives lane {lane} twice")
                out[i + 1, lane] = v * magnitude
    return out


def from_viewmats(viewmats, step: float = 0.08, rot_step_deg: float = 3.0):
    """(F, 64) stick lanes recovered from a w2c camera trajectory.

    Control2V's poses are not free-form camera paths: decomposed into the
    previous camera's frame they quantise to exactly +-0.08 translation and
    +-3.0 degrees rotation, which is `camera_trajectory._MOTIONS` verbatim. The
    clips *are* WASD command sequences, so this recovers the commands rather
    than estimating them, and `up` is always zero because that vocabulary's
    u/dn were never used.

    Lane assignment matches the module's measured convention: LY forward, LX
    strafe right, RX yaw right, RY look down. Frame 0 is zero, as everywhere
    else here -- nothing has been commanded before the first frame.
    """
    import numpy as _np
    vm = _np.asarray(viewmats, dtype=_np.float64)
    if vm.ndim != 3 or vm.shape[-2:] != (4, 4):
        raise ValueError(f"viewmats must be (F,4,4), got {vm.shape}")
    c2w = _np.linalg.inv(vm)
    R, t = c2w[:, :3, :3], c2w[:, :3, 3]
    # Each step read in the frame the camera occupied before it, which is the
    # frame the command was issued in.
    local = _np.einsum('fij,fj->fi', _np.transpose(R[:-1], (0, 2, 1)),
                       _np.diff(t, axis=0))
    rel = _np.einsum('fij,fkj->fik', R[1:], R[:-1])
    yaw = _np.degrees(_np.arctan2(rel[:, 0, 2], rel[:, 2, 2]))
    pitch = _np.degrees(_np.arcsin(_np.clip(-rel[:, 1, 2], -1.0, 1.0)))

    out = _np.zeros((len(vm), ACTION_DIM), dtype=_np.float32)
    out[1:, LX] = _np.clip(local[:, 0] / step, -1.0, 1.0)
    out[1:, LY] = _np.clip(local[:, 2] / step, -1.0, 1.0)
    out[1:, RX] = _np.clip(yaw / rot_step_deg, -1.0, 1.0)
    out[1:, RY] = _np.clip(pitch / rot_step_deg, -1.0, 1.0)
    return out
