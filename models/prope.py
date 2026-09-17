# MIT License
#
# Copyright (c) Authors of
# "PRoPE: Projective Positional Encoding for Multiview Transformers"
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# How to use PRoPE attention for self-attention:
#
# 1. Easiest way (fast):
#    attn = PropeDotProductAttention(...)
#    o = attn(q, k, v, viewmats, Ks)
#
# 2. More flexible way (fast):
#    attn = PropeDotProductAttention(...)
#    attn._precompute_and_cache_apply_fns(viewmats, Ks)
#    q = attn._apply_to_q(q)
#    k = attn._apply_to_kv(k)
#    v = attn._apply_to_kv(v)
#    o = F.scaled_dot_product_attention(q, k, v, **kwargs)
#    o = attn._apply_to_o(o)
#
# 3. The most flexible way (but slower because repeated computation of RoPE coefficients):
#    o = prope_dot_product_attention(q, k, v, ...)
#
# How to use PRoPE attention for cross-attention:
#
#    attn_src = PropeDotProductAttention(...)
#    attn_tgt = PropeDotProductAttention(...)
#    attn_src._precompute_and_cache_apply_fns(viewmats_src, Ks_src)
#    attn_tgt._precompute_and_cache_apply_fns(viewmats_tgt, Ks_tgt)
#    q_src = attn_src._apply_to_q(q_src)
#    k_tgt = attn_tgt._apply_to_kv(k_tgt)
#    v_tgt = attn_tgt._apply_to_kv(v_tgt)
#    o_src = F.scaled_dot_product_attention(q_src, k_tgt, v_tgt, **kwargs)
#    o_src = attn_src._apply_to_o(o_src)

from functools import partial
from typing import Callable, Optional, Tuple, List

import os
import torch
import torch.nn.functional as F


def anchor_viewmats_to_frame(
    viewmats: torch.Tensor, anchor_index: int = 0
) -> torch.Tensor:
    """Express camera<-world poses in the gauge of an arbitrary reference camera.

    Right-multiplying every pose by the inverse of the anchor camera is a pure
    gauge change: pairwise relative projective geometry is preserved, but the
    magnitude of every transform stays bounded around the anchor instead of
    growing with the distance from a fixed first frame.  For chunked causal
    generation, anchoring each chunk to the last frame of the previous chunk
    keeps the magnitudes inside the training window regardless of sequence
    length.
    """
    if viewmats is None or viewmats.shape[1] == 0:
        return viewmats
    return torch.matmul(
        viewmats, _invert_SE3(viewmats[:, anchor_index:anchor_index + 1])
    )


def anchor_viewmats_to_first(viewmats: torch.Tensor) -> torch.Tensor:
    """Express camera<-world poses in a persistent first-camera gauge."""
    return anchor_viewmats_to_frame(viewmats, 0)


def prope_qkv(
    q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    *,
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
    patches_x: int = None,  # How many patches wide is each image?
    patches_y: int = None,  # How many patches tall is each image?
    image_width: int = None,  # Width of the image. Used to normalize intrinsics.
    image_height: int = None,  # Height of the image. Used to normalize intrinsics.
    coeffs_x: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    coeffs_y: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    mask: Optional[torch.Tensor] = None,
    kv_cache=None,
    is_cache: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Similar to torch.nn.functional.scaled_dot_product_attention, but applies PRoPE-style
    positional encoding.

    Currently, we assume that the sequence length is equal to:

        cameras * patches_x * patches_y

    And token ordering allows the `(seqlen,)` axis to be reshaped into
    `(cameras, patches_x, patches_y)`.
    """
    # We're going to assume self-attention: all inputs are the same shape.
    (batch, num_heads, seqlen, head_dim) = q.shape
    cameras = viewmats.shape[1]
    assert q.shape == k.shape == v.shape
    assert viewmats.shape == (batch, cameras, 4, 4)
    assert Ks is None or Ks.shape == (batch, cameras, 3, 3)
    # assert seqlen == cameras * patches_x * patches_y

    apply_fn_q, apply_fn_kv, apply_fn_o = _prepare_apply_fns_all_dim(
        head_dim=head_dim,
        viewmats=viewmats,
        Ks=Ks,
        patches_x=patches_x,
        patches_y=patches_y,
        image_width=image_width,
        image_height=image_height,
        coeffs_x=coeffs_x,
        coeffs_y=coeffs_y,
    )

    query = apply_fn_q(q)
    key = apply_fn_kv(k)
    value = apply_fn_kv(v)

    return query, key, value, apply_fn_o


def _prepare_apply_fns_all_dim(
    head_dim: int,  # Q/K/V will have this last dimension
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
    patches_x: int,  # How many patches wide is each image?
    patches_y: int,  # How many patches tall is each image?
    image_width: int,  # Width of the image. Used to normalize intrinsics.
    image_height: int,  # Height of the image. Used to normalize intrinsics.
    coeffs_x: Optional[torch.Tensor] = None,
    coeffs_y: Optional[torch.Tensor] = None,
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Prepare transforms for PRoPE-style positional encoding."""
    device = viewmats.device
    (batch, cameras, _, _) = viewmats.shape

    # Normalize camera intrinsics.
    if Ks is not None:
        Ks_norm = torch.zeros_like(Ks)
        Ks_norm[..., 0, 0] = Ks[..., 0, 0]
        Ks_norm[..., 1, 1] = Ks[..., 1, 1]
        Ks_norm[..., 0, 2] = 0
        Ks_norm[..., 1, 2] = 0
        Ks_norm[..., 2, 2] = 1.0
        Ks_norm = Ks_norm.to(dtype=Ks.dtype)
        del Ks

        # Compute the camera projection matrices we use in PRoPE.
        # - K is an `image<-camera` transform.
        # - viewmats is a `camera<-world` transform.
        # - P = lift(K) @ viewmats is an `image<-world` transform.
        P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
        P_T = P.transpose(-1, -2).to(dtype=viewmats.dtype)
        P_inv = torch.einsum(
            "...ij,...jk->...ik",
            _invert_SE3(viewmats),
            _lift_K(_invert_K(Ks_norm)),
        ).to(dtype=viewmats.dtype)

    else:
        # GTA formula. P is `camera<-world` transform.
        P = viewmats
        P_T = P.transpose(-1, -2)
        P_inv = _invert_SE3(viewmats)

    assert P.shape == P_inv.shape == (batch, cameras, 4, 4)

    # Block-diagonal transforms to the inputs and outputs of the attention operator.
    assert head_dim % 4 == 0
    transforms_q = [
        (partial(_apply_tiled_projmat, matrix=P_T), head_dim),
    ]
    transforms_kv = [
        (partial(_apply_tiled_projmat, matrix=P_inv), head_dim),
    ]
    transforms_o = [
        (partial(_apply_tiled_projmat, matrix=P), head_dim),
    ]

    apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
    apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
    apply_fn_o = partial(_apply_block_diagonal, func_size_pairs=transforms_o)
    return apply_fn_q, apply_fn_kv, apply_fn_o


# Opt-in: PRoPE's projective injection changes token norms, and renormalising
# after it changes what every existing checkpoint sees. Off by default so a
# checkpoint's behaviour does not silently shift; set PROPE_NORM_PRESERVING=1 to
# evaluate the norm-preserving variant.
_NORM_PRESERVING = os.environ.get("PROPE_NORM_PRESERVING") == "1"


def _apply_tiled_projmat(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    matrix: torch.Tensor,  # (batch, cameras, D, D)
) -> torch.Tensor:
    """Apply projection matrix to features."""
    # - seqlen => (cameras, patches_x * patches_y)
    # - feat_dim => (feat_dim // 4, 4)
    (batch, num_heads, seqlen, feat_dim) = feats.shape
    cameras = matrix.shape[1]
    assert seqlen >= cameras and seqlen % cameras == 0
    D = matrix.shape[-1]
    assert matrix.shape == (batch, cameras, D, D)
    assert feat_dim % D == 0
    out = torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix,
        feats.reshape((batch, num_heads, cameras, -1, feat_dim // D, D)),
    ).reshape(feats.shape)

    if _NORM_PRESERVING:
        # Restore each token's norm, keeping only the direction the projection
        # produced.
        #
        # The 4x4 projective matrices are not norm-preserving, and the damage is
        # not uniform: measured on a w*11 trajectory anchored on the last frame,
        # |q_p|/|q| runs 0.872 at frame 0 up to 0.989 at the anchor, while |k_p|
        # is inflated 1.35x. So the further a frame sits from the anchor -- the
        # larger its camera transform -- the smaller its query norm, and the
        # less weight it carries in the softmax. That is a built-in suppression
        # of exactly the frames the camera signal lives in, present before any
        # training.
        #
        # OrthoMotion (arXiv 2606.22835, Thm. 2) shows the norm-preserving
        # injection is the unique one that leaves token norms, softmax
        # temperature and output magnitude intact, and measures projective-RoPE
        # injection of this kind at 7.4 cross-talk against 4.6 for an SO(d)
        # phase. Renormalising here is the cheap half of that: it keeps PRoPE's
        # angular information and drops its effect on magnitude, without the
        # architecture change an SO(d) phase would need.
        scale = feats.norm(dim=-1, keepdim=True) / out.norm(
            dim=-1, keepdim=True).clamp_min(1e-6)
        out = out * scale
    return out


def _apply_block_diagonal(
    feats: torch.Tensor,  # (..., dim)
    func_size_pairs: List[Tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Apply a block-diagonal function to an input array.

    Each function is specified as a tuple with form:

        ((Tensor) -> Tensor, int)

    Where the integer is the size of the input to the function.
    """
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat(
        [f(x_block) for f, x_block in zip(funcs, x_blocks)],
        dim=-1,
    )
    assert out.shape == feats.shape, "Input/output shapes should match."
    return out


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    out = out.to(dtype=transforms.dtype)
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """Lift 3x3 matrices to homogeneous 4x4 matrices."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    out = out.to(dtype=Ks.dtype)
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Invert 3x3 intrinsics matrices. Assumes no skew."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    out = out.to(dtype=Ks.dtype)
    return out


def add_prope_parameters(model, zero_init: bool = True, layer_start: int = 0, layer_end: int = None):
    """Add learnable PRoPE parameters to selected self-attention blocks.

    Registers per-layer:
    - prope_o: nn.Linear(dim, dim), zero-initialised output projection
      for the PRoPE attention path. Output = rope_out + prope_o(prope_out).

    Args:
        model: WanModel or CausalWanModel instance
        zero_init: If True, initialize prope_o weights and biases to zero
        layer_start: first transformer block index to equip (inclusive).
        layer_end: one-past-the-last transformer block index to equip
            (exclusive). ``None`` means every block, matching the legacy
            behaviour.
    """
    import torch.nn as nn

    # Import here to avoid circular dependency
    try:
        from models.wan_model import WanSelfAttention, WanT2VCrossAttention, WanI2VCrossAttention, WanGanCrossAttention
        cross_attn_types = (WanT2VCrossAttention, WanI2VCrossAttention, WanGanCrossAttention)
    except ImportError:
        WanSelfAttention = None
        cross_attn_types = ()

    try:
        from models.causal_model import CausalWanSelfAttention
    except ImportError:
        CausalWanSelfAttention = None

    attn_types = tuple(filter(None, [WanSelfAttention, CausalWanSelfAttention]))

    blocks = getattr(model, "blocks", None)
    if blocks is not None:
        total = len(blocks)
        start = max(0, int(layer_start or 0))
        end = total if layer_end is None else min(int(layer_end), total)
        targets = [
            (f"blocks.{i}.self_attn", blocks[i].self_attn)
            for i in range(start, end)
        ]
    else:
        # Legacy fallback for layouts without a named ``blocks`` sequence.
        start = max(0, int(layer_start or 0))
        end = None if layer_end is None else int(layer_end)
        targets = []
        for name, module in model.named_modules():
            if not isinstance(module, attn_types) or isinstance(module, cross_attn_types):
                continue
            block_index = None
            for part in name.split("."):
                if part.isdigit():
                    block_index = int(part)
                    break
            if block_index is None:
                continue
            if block_index < start or (end is not None and block_index >= end):
                continue
            targets.append((name, module))

    for name, module in targets:
        if not hasattr(module, 'prope_o'):
            # Get dim from existing projection layer
            dim = module.o.out_features
            prope_o = nn.Linear(dim, dim)

            if zero_init:
                nn.init.zeros_(prope_o.weight)
                nn.init.zeros_(prope_o.bias)

            # Move to same device/dtype as existing parameters
            prope_o = prope_o.to(
                device=module.o.weight.device,
                dtype=module.o.weight.dtype
            )

            module.prope_o = prope_o
