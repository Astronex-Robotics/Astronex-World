# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import repeat

from models.attention import flash_attention

try:
    from utils.sp.communication_op import (
        sequence_model_parallel_all_gather,
        sequence_model_parallel_all_to_all_4D,
    )
    from utils.sp.parallel_states import get_parallel_state
    _HAS_SP_INFRA = True
except ImportError:
    _HAS_SP_INFRA = False

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    freqs = torch.pow(10000, -torch.arange(half).to(position).div(half))
    sinusoid = position.unsqueeze(-1) * freqs
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=-1)
    return x


# @amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


# @amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, viewmats=None, Ks=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            viewmats(Tensor, optional): Camera extrinsics, shape [B, L, 4, 4]
            Ks(Tensor, optional): Camera intrinsics, shape [B, L, 3, 3]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        # SP: scatter heads, gather sequence for attention
        if _HAS_SP_INFRA:
            sp_enabled = get_parallel_state().sp_enabled
        else:
            sp_enabled = False

        # PRoPE: apply projective encoding BEFORE SP all-to-all (if viewmats provided)
        if viewmats is not None:
            from models.prope import prope_qkv
            # q/k/v: [B, L, H, D] → [B, H, L, D] for prope_qkv
            q_p, k_p, v_p, apply_fn_o = prope_qkv(
                q.permute(0, 2, 1, 3),
                k.permute(0, 2, 1, 3),
                v.permute(0, 2, 1, 3),
                viewmats=viewmats,
                Ks=Ks,
            )
            # [B, H, L, D] → [B, L, H, D]
            q_p = q_p.permute(0, 2, 1, 3)
            k_p = k_p.permute(0, 2, 1, 3)
            v_p = v_p.permute(0, 2, 1, 3)

        # SP all-to-all: scatter heads, gather sequence for both paths
        if sp_enabled:
            q = sequence_model_parallel_all_to_all_4D(q, scatter_dim=2, gather_dim=1)
            k = sequence_model_parallel_all_to_all_4D(k, scatter_dim=2, gather_dim=1)
            v = sequence_model_parallel_all_to_all_4D(v, scatter_dim=2, gather_dim=1)
            if viewmats is not None:
                q_p = sequence_model_parallel_all_to_all_4D(q_p, scatter_dim=2, gather_dim=1)
                k_p = sequence_model_parallel_all_to_all_4D(k_p, scatter_dim=2, gather_dim=1)
                v_p = sequence_model_parallel_all_to_all_4D(v_p, scatter_dim=2, gather_dim=1)

        # Standard RoPE attention path
        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # PRoPE attention path (if viewmats provided)
        if viewmats is not None:
            x_prope = flash_attention(
                q=q_p, k=k_p, v=v_p,
                k_lens=seq_lens,
                window_size=self.window_size)

        # SP: scatter sequence, gather heads back
        if sp_enabled:
            x = sequence_model_parallel_all_to_all_4D(x, scatter_dim=1, gather_dim=2)
            if viewmats is not None:
                x_prope = sequence_model_parallel_all_to_all_4D(
                    x_prope, scatter_dim=1, gather_dim=2)

        # RoPE output
        x = x.flatten(2)
        x = self.o(x)

        # PRoPE output: apply_fn_o correction then zero-init projection
        if viewmats is not None:
            # [B, L, H, D] → [B, H, L, D] → correct → [B, H, L, D] → [B, L, H, D]
            x_prope = apply_fn_o(
                x_prope.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            x_prope = x_prope.flatten(2)
            x_prope = self.prope_o(x_prope)
            if getattr(self, "capture_camera_context", False):
                # Spatially pooled, direction-sensitive PRoPE context.  Keep
                # the graph for the causal student; the frozen teacher call is
                # already under no_grad in the distillation model.
                self.camera_context = x_prope.mean(dim=1)
            x = x + x_prope

        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanGanCrossAttention(WanSelfAttention):

    def forward(self, x, context, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        qq = self.norm_q(self.q(context)).view(b, 1, -1, d)

        kk = self.norm_k(self.k(x)).view(b, -1, n, d)
        vv = self.v(x).view(b, -1, n, d)

        # compute attention
        x = flash_attention(qq, kk, vv)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(
            dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        # Event-prompt branch: installed on demand by
        # `enable_event_conditioning`, absent (and free) otherwise.
        self.cross_attn_type = cross_attn_type
        self.norm_event = None
        self.event_cross_attn = None
        self.event_scale = 1.0
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def enable_event_conditioning(self, cross_attn_type=None):
        """Graft the independent event-prompt cross-attention onto this block.

        See ``CausalWanAttentionBlock.enable_event_conditioning``: the branch is
        grafted after the base checkpoint loads and zero-initialised at ``o``,
        so it neither breaks existing loads nor perturbs the model it is added
        to.
        """
        if self.event_cross_attn is not None:
            return
        ref = next(self.parameters())
        cross_attn_type = cross_attn_type or self.cross_attn_type
        norm = (WanLayerNorm(self.dim, self.eps, elementwise_affine=True)
                if self.cross_attn_norm else nn.Identity())
        attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            self.dim, self.num_heads, (-1, -1), self.qk_norm, self.eps)
        nn.init.zeros_(attn.o.weight)
        nn.init.zeros_(attn.o.bias)
        self.norm_event = norm.to(device=ref.device, dtype=ref.dtype)
        self.event_cross_attn = attn.to(device=ref.device, dtype=ref.dtype)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        viewmats=None,
        Ks=None,
        event_context=None,
        event_context_lens=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            viewmats(Tensor, optional): Shape [B, L, 4, 4] camera extrinsics for PRoPE
            Ks(Tensor, optional): Shape [B, L, 3, 3] camera intrinsics for PRoPE
        """
        # `e` is [B, 6, C] for a clip-uniform timestep, or [B, F, 6, C] once a
        # per-frame signal is present -- which action conditioning always is
        # (see wan/modules/action.py). The causal blocks have always supported
        # the per-frame form; this path mirrors them so stage 0 (bidirectional)
        # can be action-conditioned too, instead of forcing actions to be pooled
        # over the clip and losing exactly the per-frame control we are after.
        if e.dim() == 4:
            num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
            e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)  # each [B, F, 1, C]

            def _mod(h, i_scale, i_shift):
                return (h.unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                        * (1 + e[i_scale]) + e[i_shift]).flatten(1, 2)

            def _gate(h, i):
                return (h.unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                        * e[i]).flatten(1, 2)
        else:
            e = (self.modulation + e).chunk(6, dim=1)

            def _mod(h, i_scale, i_shift):
                return h * (1 + e[i_scale]) + e[i_shift]

            def _gate(h, i):
                return h * e[i]

        # self-attention
        y = self.self_attn(
            _mod(self.norm1(x), 1, 0), seq_lens, grid_sizes,
            freqs, viewmats=viewmats, Ks=Ks)
        # with amp.autocast(dtype=torch.float32):
        x = x + _gate(y, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(_mod(self.norm2(x), 4, 3))
            # with amp.autocast(dtype=torch.float32):
            x = x + _gate(y, 5)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)

        # Independent event prompt: second cross-attention so event text can be
        # guided separately from the main caption.
        if event_context is not None and self.event_cross_attn is not None:
            # `event_scale` is the auxiliary knob: the caption drives the scene
            # through the main cross-attention, the event rides on top as a
            # residual that can be dialled down (or up) at inference without
            # retraining. 1.0 is the trained strength.
            x = x + self.event_scale * self.event_cross_attn(
                self.norm_event(x), event_context, event_context_lens)

        return x


class GanAttentionBlock(nn.Module):

    def __init__(self,
                 dim=1536,
                 ffn_dim=8192,
                 num_heads=12,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        # self.norm1 = WanLayerNorm(dim, eps)
        # self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
        #   eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()

        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        self.cross_attn = WanGanCrossAttention(dim, num_heads,
                                               (-1, -1),
                                               qk_norm,
                                               eps)

        # modulation
        # self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        context,
        # seq_lens,
        # grid_sizes,
        # freqs,
        # context,
        # context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        # e = (self.modulation + e).chunk(6, dim=1)
        # assert e[0].dtype == torch.float32

        # # self-attention
        # y = self.self_attn(
        #     self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes,
        #     freqs)
        # # with amp.autocast(dtype=torch.float32):
        # x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context):
            token = context + self.cross_attn(self.norm3(x), context)
            y = self.ffn(self.norm2(token)) + token  # * (1 + e[4]) + e[3])
            # with amp.autocast(dtype=torch.float32):
            # x = x + y * e[5]
            return y

        x = cross_attn_ffn(x, context)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        if e.dim() == 3:
            num_frames = e.shape[1]
            frame_seqlen = x.shape[1] // num_frames
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = self.norm(x)
            x = x.unflatten(1, (num_frames, frame_seqlen)) * (1 + e[1]) + e[0]
            x = self.head(x.flatten(1, 2))
        else:
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class RegisterTokens(nn.Module):
    def __init__(self, num_registers: int, dim: int):
        super().__init__()
        self.register_tokens = nn.Parameter(torch.randn(num_registers, dim) * 0.02)
        self.rms_norm = WanRMSNorm(dim, eps=1e-6)

    def forward(self):
        return self.rms_norm(self.register_tokens)

    def reset_parameters(self):
        nn.init.normal_(self.register_tokens, std=0.02)


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.local_attn_size = 21

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))
        # Independent event-prompt embedding, grafted on demand together with
        # the per-block event cross-attention. Kept separate from
        # `text_embedding` so event and main caption can be conditioned and
        # guided independently.
        self.event_embedding = None

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)
        self.action_head = None
        self.event_enabled = False
        self.action_dim = 0

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

    def enable_event_conditioning(self, num_layers=None, layer_start=0):
        """Turn on the independent event-prompt channel: projection + per-block
        cross-attention.

        Grafted rather than built in ``__init__`` so a model that never uses
        events neither carries nor saves the parameters, and so existing
        checkpoints keep loading against the key set they were written with.

        ``num_layers`` limits the graft to ``blocks[layer_start:layer_start +
        num_layers]``, mirroring ``prope_num_layers``. On the 5B backbone the
        full graft is +1.16 B parameters, and because these are new
        zero-initialised weights they train *unadapted* -- fp32 masters plus
        AdamW moments, roughly 14 GB of optimiser state on top of the
        backbone. Text conditioning does most of its work in the early blocks,
        so a dozen layers is usually enough and costs a third of that. ``None``
        (the default) grafts every block.
        """
        if getattr(self, "event_enabled", False):
            return
        ref = next(self.parameters())
        self.event_embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.dim), nn.GELU(approximate='tanh'),
            nn.Linear(self.dim, self.dim),
        ).to(device=ref.device, dtype=ref.dtype)
        start = int(layer_start)
        end = len(self.blocks) if num_layers is None else start + int(num_layers)
        if start >= end or start < 0:
            raise ValueError(
                f"event layers [{start}, {end}) selects no block of "
                f"{len(self.blocks)}")
        for block in self.blocks[start:end]:
            block.enable_event_conditioning()
        self.event_enabled = True
        self.set_event_scale(getattr(self, "event_scale", 1.0))
        self.event_layers = (start, min(end, len(self.blocks)))

    def set_event_scale(self, scale: float):
        """Strength of the event channel relative to the caption, at inference."""
        self.event_scale = float(scale)
        for block in self.blocks:
            block.event_scale = float(scale)

    def add_action_head(self, action_dim: int, hidden: int = None):
        """Attach the action-output head; zero-init keeps the graft identity."""
        if self.action_head is not None:
            return
        hidden = hidden or self.dim
        self.action_dim = int(action_dim)
        head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(action_dim)),
        )
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        # Match the backbone: a head left on the CPU in fp32 crashes the first
        # forward on a model that has already been moved to the GPU.
        ref = next(self.parameters())
        self.action_head = head.to(device=ref.device, dtype=ref.dtype)

    def _predict_actions(self, hidden: torch.Tensor, grid_sizes: torch.Tensor):
        b = hidden.shape[0]
        f, hp, wp = grid_sizes[0].tolist()
        tokens_per_frame = hp * wp
        pooled = hidden[:, :f * tokens_per_frame].reshape(
            b, f, tokens_per_frame, self.dim).mean(dim=2)
        # Run under autocast rather than casting by hand, for the same reason
        # `apply_action_modulation` does. The head is cast to fp32 while
        # training (bf16 masters round small updates away at these learning
        # rates), so a checkpoint carries fp32 weights into a bf16 backbone at
        # inference -- and reading `weight.dtype` to decide is actively wrong
        # under FSDP, where the master dtype is not the one served in the
        # forward. Without this, loading a trained head raises "expected
        # BFloat16 but found Float" from the head's LayerNorm.
        target = pooled.dtype
        autocast_ok = pooled.is_cuda and target in (torch.float16, torch.bfloat16)
        with torch.autocast(device_type="cuda", dtype=target, enabled=autocast_ok):
            out = self.action_head(pooled)
        return out.to(target)

    def _set_gradient_checkpointing(self, module=None, value=False, enable=None, gradient_checkpointing_func=None):
        if enable is not None:
            value = enable
        self.gradient_checkpointing = value

    def forward(
        self,
        *args,
        **kwargs
    ):
        # if kwargs.get('classify_mode', False) is True:
        # kwargs.pop('classify_mode')
        # return self._forward_classify(*args, **kwargs)
        # else:
        return self._forward(*args, **kwargs)

    def _forward(
        self,
        x,
        t,
        context,
        seq_len,
        classify_mode=False,
        concat_time_embeddings=False,
        register_tokens=None,
        cls_pred_branch=None,
        gan_ca_blocks=None,
        clip_fea=None,
        y=None,
        viewmats=None,
        Ks=None,
        actions=None,
        embodiment_id=None,
        event_context=None,
        return_actions=False,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))
        if t.dim() > 1:
            e0 = self.time_projection(e).unflatten(-1, (6, self.dim))
        else:
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        # Action conditioning: [B, 6, C] -> [B, F, 6, C] with the per-frame term
        # added. No-op (and shape-preserving) when the model has no embedder or
        # the batch carries no actions.
        from models.action import apply_action_modulation
        e0 = apply_action_modulation(self, e0, actions, embodiment_id)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        event_ctx = None
        if event_context is not None:
            if self.event_embedding is None:
                raise RuntimeError(
                    "an event prompt was supplied but this model has no event "
                    "branch; set `use_event: true` in model_kwargs (or call "
                    "enable_event_conditioning()) before loading a checkpoint "
                    "that was trained with one")
            event_ctx = self.event_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in event_context
                ]))

        # SP: chunk sequence along token dimension
        if _HAS_SP_INFRA:
            sp_state = get_parallel_state()
            sp_enabled = sp_state.sp_enabled
        else:
            sp_enabled = False
        if sp_enabled:
            sp_size = sp_state.sp
            sp_rank = sp_state.sp_rank
            chunk_size = x.shape[1] // sp_size
            chunk_start = sp_rank * chunk_size
            chunk_end = chunk_start + chunk_size
            x = x[:, chunk_start:chunk_end]

        # PRoPE: expand viewmats/Ks from (B, F, *, *) to (B, seq_len, *, *)
        if viewmats is not None:
            expanded_vm = []
            expanded_ks = []
            for i, (f, h, w) in enumerate(grid_sizes.tolist()):
                vm = viewmats[i, :f]  # (F, 4, 4)
                vm = vm[:, None, None].expand(-1, h, w, -1, -1).reshape(f * h * w, 4, 4)
                pad_len = seq_len - f * h * w
                if pad_len > 0:
                    pad_vm = torch.eye(4, device=vm.device, dtype=vm.dtype).unsqueeze(0).expand(pad_len, -1, -1)
                    vm = torch.cat([vm, pad_vm])
                expanded_vm.append(vm)

                ks = Ks[i, :f]  # (F, 3, 3)
                ks = ks[:, None, None].expand(-1, h, w, -1, -1).reshape(f * h * w, 3, 3)
                if pad_len > 0:
                    pad_ks = torch.eye(3, device=ks.device, dtype=ks.dtype).unsqueeze(0).expand(pad_len, -1, -1)
                    ks = torch.cat([ks, pad_ks])
                expanded_ks.append(ks)

            viewmats = torch.stack(expanded_vm)  # (B, seq_len, 4, 4)
            Ks = torch.stack(expanded_ks)        # (B, seq_len, 3, 3)

            # SP chunk: only take current rank's tokens
            if sp_enabled:
                viewmats = viewmats[:, chunk_start:chunk_end]
                Ks = Ks[:, chunk_start:chunk_end]

        # arguments
        # NOTE: seq_lens stays as the original full lengths — after all-to-all
        # inside self-attention, q/k/v are full-length and need the original
        # seq_lens for flash_attention's varlen masking.
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)
        if viewmats is not None:
            kwargs['viewmats'] = viewmats
            kwargs['Ks'] = Ks
        if event_ctx is not None:
            kwargs['event_context'] = event_ctx
            kwargs['event_context_lens'] = None

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        # TODO: Tune the number of blocks for feature extraction
        final_x = None
        if classify_mode:
            assert cls_pred_branch is not None
            # `register_tokens`/`gan_ca_blocks` are optional: the pooled head
            # reduces each tapped layer by mean over the sequence and has no
            # parameters of its own per tap.
            final_x = []
            registers = (
                repeat(register_tokens(), "n d -> b n d", b=x.shape[0])
                if register_tokens is not None else None)

        gan_idx = 0
        for ii, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

            if classify_mode and ii in [13, 21, 29]:
                # The discriminator head is held in fp32 -- it is what the
                # optimiser updates, and bf16 masters round lr-sized steps away
                # -- while the backbone runs bf16 under FSDP's mixed precision.
                # Cast at the boundary, the same way LoRALinear does for its
                # adapters. Casting the head down instead would cost the
                # precision it is kept in fp32 for.
                if gan_ca_blocks is None:
                    head_dtype = next(cls_pred_branch.parameters()).dtype
                    final_x.append(x.mean(dim=1, keepdim=True).to(head_dtype))
                else:
                    gan_token = registers[:, gan_idx: gan_idx + 1]
                    blk = gan_ca_blocks[gan_idx]
                    blk_dtype = next(blk.parameters()).dtype
                    final_x.append(blk(x.to(blk_dtype), gan_token.to(blk_dtype)))
                gan_idx += 1

        if classify_mode:
            head_dtype = next(cls_pred_branch.parameters()).dtype
            final_x = torch.cat(final_x, dim=1).to(head_dtype)
            if concat_time_embeddings:
                final_x = cls_pred_branch(
                    torch.cat([final_x, 10 * e[:, None, :].to(head_dtype)], dim=1
                              ).view(final_x.shape[0], -1))
            else:
                final_x = cls_pred_branch(final_x.view(final_x.shape[0], -1))

        # head
        # SP: gather sequence from all ranks before head
        if sp_enabled:
            x = sequence_model_parallel_all_gather(x, dim=1)

        action_pred = None
        if self.action_head is not None and return_actions:
            action_pred = self._predict_actions(x, grid_sizes)

        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)

        if classify_mode:
            video = torch.stack(x)
            if action_pred is not None:
                return video, final_x, action_pred
            return video, final_x

        video = torch.stack(x)
        if action_pred is not None:
            return video, action_pred
        return video

    def _forward_classify(
        self,
        x,
        t,
        context,
        seq_len,
        register_tokens,
        cls_pred_branch,
        clip_fea=None,
        y=None,
        actions=None,
        embodiment_id=None,
    ):
        r"""
        Feature extraction through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of video features with original input shapes [C_block, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))
        if t.dim() > 1:
            e0 = self.time_projection(e).unflatten(-1, (6, self.dim))
        else:
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        # Action conditioning: [B, 6, C] -> [B, F, 6, C] with the per-frame term
        # added. No-op (and shape-preserving) when the model has no embedder or
        # the batch carries no actions.
        from models.action import apply_action_modulation
        e0 = apply_action_modulation(self, e0, actions, embodiment_id)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        # TODO: Tune the number of blocks for feature extraction
        for block in self.blocks[:16]:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        # unpatchify
        x = self.unpatchify(x, grid_sizes, c=self.dim // 4)
        return torch.stack(x)

    def unpatchify(self, x, grid_sizes, c=None):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim if c is None else c
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
