from models.attention import attention
from models.wan_model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import os
import torch.distributed as dist

try:
    from models.causal_wan_config import CausalWanConfig
    from utils.sp.communication_op import (
        sequence_model_parallel_all_gather,
        sequence_model_parallel_all_to_all_4D,
    )
    from utils.sp.parallel_states import get_parallel_state
    _HAS_CLEANCODE_INFRA = True
except ImportError as _e:
    _HAS_CLEANCODE_INFRA = False
    _CLEANCODE_IMPORT_ERROR = _e


def _require_cleancode_infra(context: str = ""):
    """Hard-fail if SP > 1 but CleanCode SP infra failed to import.

    Called at every SP decision point so that a silent fallback to
    sp_enabled=False can never happen when the user explicitly requested SP.
    """
    if _HAS_CLEANCODE_INFRA:
        return
    # Only crash when distributed training is active (world_size > 1).
    # Single-GPU / non-distributed usage can degrade gracefully.
    sp_size_env = int(os.environ.get("SP_SIZE", "1"))
    if dist.is_initialized() and dist.get_world_size() > 1 and sp_size_env > 1:
        msg = (
            f"[FATAL] CleanCode SP infra import failed but distributed "
            f"training is active (world_size={dist.get_world_size()}).\n"
            f"Import error: {_CLEANCODE_IMPORT_ERROR}\n"
            f"Context: {context}\n"
            f"SP > 1 requires sp.communication_op, sp.parallel_states, and "
            f"wan.configs.causal_wan_config. Fix the import or run with SP=1."
        )
        raise RuntimeError(msg)

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention,
    dynamic=False,
    mode="default"
)



def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
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
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
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


def _roll_cache_by_frames(cache, keep_frames, frame_seqlen, sink_size,
                          num_rolled_tokens, sink_frames=None):
    """Compact the cache down to `keep_frames`, in the order given.

    `cache["frame_ids"]` maps each resident slot to the global frame it holds,
    so a frame that left the recent window but is about to be looked at again
    can be kept where age-based eviction would have dropped it. Slots are
    gathered rather than shifted; the survivors land contiguously starting
    after the sink, which is what the re-roping below expects.
    """
    ids = cache.get("frame_ids")
    if ids is None:
        raise RuntimeError("Cache frame IDs must be tracked before retrieval")
    resident = ids.tolist()
    where = {f: i for i, f in enumerate(resident) if f >= 0}
    # Sink slots are pinned: they are the reference the window-relative RoPE
    # measures everything against.
    wanted = list(keep_frames)
    room = num_rolled_tokens // frame_seqlen
    if len(wanted) != room or len(set(wanted)) != room:
        raise RuntimeError(f"Expected {room} distinct history frames, got {wanted}")
    # Frames promoted into the sink are exempt from the "outside the sink"
    # rule below: moving one there is the whole point.
    promote = list(sink_frames) if sink_frames else []
    missing_sink = [f for f in promote if f not in where]
    if missing_sink:
        raise RuntimeError(
            f"Sink refresh asked for non-resident frames: {missing_sink}; "
            f"resident={resident}")
    missing = [f for f in wanted
               if f not in where or (where[f] < sink_size and f not in promote)]
    if missing:
        raise RuntimeError(f"Requested history is not resident: {missing}; resident={resident}")

    k, v = cache["k"], cache["v"]
    # Everything is cloned before anything is written: a frame being promoted
    # into the sink may currently sit where a surviving history frame is about
    # to land, and vice versa.
    src_k = [k[:, where[f] * frame_seqlen:(where[f] + 1) * frame_seqlen].clone()
             for f in wanted]
    src_v = [v[:, where[f] * frame_seqlen:(where[f] + 1) * frame_seqlen].clone()
             for f in wanted]
    sink_k = [k[:, where[f] * frame_seqlen:(where[f] + 1) * frame_seqlen].clone()
              for f in promote]
    sink_v = [v[:, where[f] * frame_seqlen:(where[f] + 1) * frame_seqlen].clone()
              for f in promote]
    for j, (bk, bv) in enumerate(zip(src_k, src_v)):
        dst = (sink_size + j) * frame_seqlen
        k[:, dst:dst + frame_seqlen] = bk
        v[:, dst:dst + frame_seqlen] = bv
    for j, (bk, bv) in enumerate(zip(sink_k, sink_v)):
        dst = j * frame_seqlen
        k[:, dst:dst + frame_seqlen] = bk
        v[:, dst:dst + frame_seqlen] = bv
    new_ids = ids.clone()
    for j, f in enumerate(promote):
        new_ids[j] = f
    for j, f in enumerate(wanted):
        new_ids[sink_size + j] = f
    new_ids[sink_size + len(wanted):] = -1
    cache["frame_ids"] = new_ids


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 31200 if local_attn_size == -1 else local_attn_size * 1560
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        # Optional history-context branch. It is installed explicitly by the
        # wrapper so legacy checkpoints keep exactly their original parameter
        # set. The zero-init output makes enabling it an exact no-op before
        # training.
        self.history_cross_enabled = False
        self.history_cross_topk = 0
        self.history_cross_block_frames = 1
        self.history_cross_independent_first_frame = False
        # Keep appearance/history retrieval in the ordinary content feature
        # space.  PRoPE is a camera-geometry residual; using its transformed
        # Q/K/V as identity memory lets motion overwrite subject appearance.
        self.history_cross_content_space = False
        self.history_cross_scale = 1.0
        self.history_cross_use_memory = False
        self.history_cross_full_memory = False
        # Parameter-free sink logit bias for inference. Repeating the exact
        # reference K/V N times adds log(N) to its aggregate softmax mass;
        # unlike value scaling, this prevents the sink probability itself
        # from vanishing as generated blocks accumulate.
        self.sink_attention_repeat = 1
        # The inverse knob. `ref_cond_scale` dims what the reference *says* by
        # scaling its cached values, but leaves the attention probability spent
        # on it untouched, so a clean i2v reference keeps the same share of the
        # softmax however faint it is made. Softmax only reads relative logits,
        # so repeating the current block instead divides sink+history weight by
        # this factor without touching them -- the same concat trick as
        # sink_attention_repeat, in the opposite direction, and unlike an
        # additive logit bias it survives FlashAttention, which takes no bias.
        # 1 leaves attention exactly as it was.
        self.rollout_attention_repeat = 1
        # Same relative-weight trick, aimed at clean content instead. Repeating
        # the current block downweights the sink but doubles the influence of a
        # block that is still mid-denoising, and the result flickers. The window
        # between the sink and the current block is already fully denoised, so
        # repeating that lowers the sink's share without amplifying noise.
        self.history_attention_repeat = 1
        # Spatial (per-token) MoBA routing for history selection.  The pooled
        # baseline averages a small foreground object (hand-held lantern ~4%)
        # into the background and nearly always drops its identity block.
        self.history_cross_per_token_routing = False
        # Route each query token/head independently across temporal blocks.
        # This is distinct from ``per_token_routing`` above, which only uses a
        # bag of tokens to make one clip-wide block decision.
        self.history_cross_tokenwise_blocks = False
        self.history_cross_route_heads = 0  # 0 = all heads
        # Explicit multiplicative boost for tokens belonging to the persistent
        # sink / reference frame block (frame 0), which is the only frame in
        # the cache whose content we know is the "clean identity" of the
        # subject.  Bidirectional attention always has the whole clip to
        # re-anchor on; causal needs us to deliberately protect this one.
        self.history_cross_sink_boost = 1.0
        self.history_cross_route_pool_tokens = 64  # spatial diversity, not 1
        # Which fraction of the query rep bag votes on block selection.  A
        # small foreground object (2-5 %) only owns a few reps; we keep their
        # best max-match scores and discard the overwhelming background.
        self.history_cross_route_topk_frac = 0.125
        # Contiguous history mode: keep the sink block plus the most recent
        # topk-1 blocks, in temporal order, with no similarity scoring.
        # Subject identity is read from a continuous frame sequence; letting
        # topk routing pick scattered high-score blocks turns the motion
        # record into jump-cut snapshots and the model re-imagines the
        # subject between anchors.
        self.history_cross_contiguous = False

    def enable_history_cross_attention(self, topk=0, block_frames=1, rank=64):
        if not hasattr(self, "history_cross_o"):
            rank = min(int(rank), self.dim)
            self.history_cross_o = nn.Sequential(
                nn.Linear(self.dim, rank, bias=False),
                nn.Linear(rank, self.dim, bias=False),
            )
            # Non-zero down projection permits gradient to reach the zero-init
            # up projection on step one while preserving an exact no-op.
            nn.init.normal_(self.history_cross_o[0].weight, std=rank ** -0.5)
            nn.init.zeros_(self.history_cross_o[1].weight)
        self.history_cross_enabled = True
        self.history_cross_topk = int(topk)
        self.history_cross_block_frames = max(1, int(block_frames))

    def _split_history_chunks(self, key, value, frame_seqlen):
        block_tokens = self.history_cross_block_frames * frame_seqlen
        if self.history_cross_independent_first_frame and key.shape[1] >= frame_seqlen:
            chunks_k = [key[:, :frame_seqlen]] + list(
                key[:, frame_seqlen:].split(block_tokens, dim=1))
            chunks_v = [value[:, :frame_seqlen]] + list(
                value[:, frame_seqlen:].split(block_tokens, dim=1))
        else:
            chunks_k = list(key.split(block_tokens, dim=1))
            chunks_v = list(value.split(block_tokens, dim=1))
        # A partially initialized absolute history bank can leave a zero-token
        # tail during the first I2V rollout calls. Concatenation tolerated it,
        # but token-wise max routing has no valid reduction over an empty
        # block. Empty cache capacity is not history and must not be routed.
        nonempty = [
            (chunk_k, chunk_v) for chunk_k, chunk_v in zip(chunks_k, chunks_v)
            if chunk_k.shape[1] > 0
        ]
        return ([pair[0] for pair in nonempty],
                [pair[1] for pair in nonempty])

    def _routed_history_attention(self, query, key, value, frame_seqlen):
        """Cross-attend query tokens to selected historical frame blocks.

        The first (sink / reference) and most-recent history blocks are always
        retained.  Remaining blocks are selected by Q/K routing.
        """
        if key.shape[1] == 0:
            return torch.zeros_like(query)
        chunks_k, chunks_v = self._split_history_chunks(key, value, frame_seqlen)
        count = len(chunks_k)
        topk = int(self.history_cross_topk)
        keep = set(range(count))

        if self.history_cross_tokenwise_blocks and count > 1:
            # A single clip-wide routing vote is dominated by static
            # background.  Produce one history read per block, then let every
            # query token and head blend those reads using its own strongest
            # content match.  A lantern token can therefore stay attached to
            # the reference/lantern history while neighbouring shelf tokens
            # prefer recent shelf history.
            pool_tokens = max(1, int(self.history_cross_route_pool_tokens))
            block_outputs = []
            block_scores = []
            qf = query.float()
            for block_index, (chunk_k, chunk_v) in enumerate(
                    zip(chunks_k, chunks_v)):
                block_outputs.append(attention(query, chunk_k, chunk_v))
                length = chunk_k.shape[1]
                if length <= pool_tokens:
                    indices = torch.arange(length, device=chunk_k.device)
                else:
                    stride = length / float(pool_tokens)
                    indices = (torch.arange(
                        pool_tokens, device=chunk_k.device) * stride
                    ).floor().long()
                representatives = chunk_k[:, indices].float()
                # B,Lq,H,D x B,R,H,D -> B,Lq,H,R.  Max rather than mean keeps
                # a small structured object from being averaged into the
                # background of its block.
                score = torch.einsum(
                    "blhd,brhd->blhr", qf, representatives
                ).amax(dim=-1) / math.sqrt(query.shape[-1])
                if block_index == 0:
                    score = score + math.log(max(
                        1.0, float(self.history_cross_sink_boost)))
                block_scores.append(score)
            scores = torch.stack(block_scores, dim=-1)
            if topk > 0 and count > topk:
                best = scores.topk(topk, dim=-1)
                masked = torch.full_like(scores, float("-inf"))
                scores = masked.scatter(-1, best.indices, best.values)
            weights = scores.softmax(dim=-1).to(query.dtype)
            stacked = torch.stack(block_outputs, dim=-1)
            return (stacked * weights.unsqueeze(-2)).sum(dim=-1)

        if os.environ.get("WAN_HISTORY_DIAG") == "1":
            seen = getattr(self, "_hist_route_seen", set())
            key_len = key.shape[1]
            if key_len not in seen:
                seen.add(key_len)
                self._hist_route_seen = seen
                print(
                    f"[hist-route] topk={self.history_cross_topk} "
                    f"block_frames={self.history_cross_block_frames} "
                    f"chunks={count} key_tokens={key_len} "
                    f"tokenwise={self.history_cross_tokenwise_blocks} "
                    f"per_token={self.history_cross_per_token_routing} "
                    f"sink_boost={self.history_cross_sink_boost}",
                    flush=True)

        if topk > 0 and count > topk:
            if self.history_cross_contiguous:
                # Sink (identity) + contiguous tail (motion).  Selection is
                # deterministic; no routing score gets to break continuity.
                keep = {0} | set(range(count - (topk - 1), count))
                keep = {i for i in keep if 0 <= i < count}
                selected = sorted(keep)
                hist_k = torch.cat([chunks_k[i] for i in selected], dim=1)
                hist_v = torch.cat([chunks_v[i] for i in selected], dim=1)
                out = attention(query, hist_k, hist_v)
                sink_boost = float(self.history_cross_sink_boost)
                if sink_boost != 1.0 and 0 in selected:
                    ref_attn = attention(query, chunks_k[0], chunks_v[0])
                    out = out + (sink_boost - 1.0) * ref_attn
                return out
            # Always keep frame 0 (reference / sink) and the immediately
            # preceding block regardless of routing score.  These two are the
            # causal analogue of what bidirectional gets for free — identity +
            # local motion continuity.
            mandatory = {0, count - 1}
            candidates = [i for i in range(count) if i not in mandatory]
            budget = max(0, topk - len(mandatory))
            keep = set(mandatory)

            if budget and candidates:
                B, Lq, Hq, Dq = query.shape
                if self.history_cross_per_token_routing:
                    # Build a representative bag: every head, spatial token
                    # subsample sized by ``history_cross_route_pool_tokens``.
                    pool_tokens = min(Lq, max(1, int(
                        self.history_cross_route_pool_tokens)))
                    route_heads = Hq if (
                        int(self.history_cross_route_heads) <= 0 or
                        int(self.history_cross_route_heads) > Hq
                    ) else int(self.history_cross_route_heads)
                    # Deterministic spatial stride sampling: stride avoids
                    # picking only a corner of the frame.
                    if Lq <= pool_tokens:
                        q_idx = torch.arange(Lq, device=query.device)
                    else:
                        stride = Lq / float(pool_tokens)
                        q_idx = (torch.arange(pool_tokens, device=query.device) *
                                 stride).floor().to(torch.long)
                    q_bag = query[:, q_idx, :route_heads, :].float()  # B, R, Hr, D
                    q_bag = q_bag.reshape(B, -1, Dq)  # B, R*Hr, D — rep bag
                    scores_cand = []
                    for i in candidates:
                        ki = chunks_k[i].float()
                        Lki = ki.shape[1]
                        pool_ki = min(Lki, pool_tokens)
                        if Lki <= pool_ki:
                            k_idx = torch.arange(Lki, device=ki.device)
                        else:
                            stride_k = Lki / float(pool_ki)
                            k_idx = (torch.arange(pool_ki, device=ki.device) *
                                     stride_k).floor().to(torch.long)
                        k_rep = ki[:, k_idx, :route_heads, :]  # B, Rk, Hr, D
                        k_rep = k_rep.reshape(B, -1, Dq)  # B, Rk*Hr, D
                        # MAX over candidate-side reps picks the strongest
                        # single spatial-token match found in the block; a
                        # mean would dilute a small foreground match with
                        # background.
                        sim = torch.einsum(
                            "brd,bcd->brc", q_bag, k_rep)  # B, Rq, Rc
                        per_rep_max, _ = sim.max(dim=-1)  # B, Rq
                        # Rank reps by their max-match and average the
                        # TOP-fraction winners.
                        route_topk_frac = float(getattr(
                            self, "history_cross_route_topk_frac", 0.125))
                        route_topk = max(
                            1, min(per_rep_max.shape[1], int(round(
                                per_rep_max.shape[1] * route_topk_frac))))
                        top_per_rep = torch.topk(
                            per_rep_max, route_topk, dim=1).values  # B, K
                        scores_cand.append(top_per_rep.mean(dim=1))  # B
                    stacked = torch.stack(scores_cand, dim=-1)  # B, C
                    scores = stacked.mean(dim=0)  # C (avg across batch)
                    chosen = scores.topk(min(budget, len(candidates))).indices.tolist()
                    keep.update(candidates[i] for i in chosen)
                else:
                    # Original LongLive-style pooled routing (kept as exact
                    # fallback).
                    q_pool = query.float().mean(dim=1)  # B, H, D
                    scores = []
                    for i in candidates:
                        k_pool = chunks_k[i].float().mean(dim=1)  # B, H, D
                        scores.append((q_pool * k_pool).mean(dim=(-1, -2)))
                    scores = torch.stack(scores, dim=-1).mean(dim=0)
                    chosen = scores.topk(min(budget, len(candidates))).indices.tolist()
                    keep.update(candidates[i] for i in chosen)

        selected = sorted(keep)
        hist_k = torch.cat([chunks_k[i] for i in selected], dim=1)
        hist_v = torch.cat([chunks_v[i] for i in selected], dim=1)

        out = attention(query, hist_k, hist_v)

        sink_boost = float(self.history_cross_sink_boost)
        if sink_boost != 1.0 and 0 in selected:
            # Recompute the reference-block component separately so we can
            # scale it.  Block 0 is the sink in this cache layout.
            ref_k = chunks_k[0]
            ref_v = chunks_v[0]
            ref_attn = attention(query, ref_k, ref_v)
            out = out + (sink_boost - 1.0) * ref_attn

        if os.environ.get("WAN_HISTORY_DIAG") == "1":
            seen = getattr(self, "_hist_keep_seen", set())
            key_len = key.shape[1]
            if key_len not in seen:
                seen.add(key_len)
                self._hist_keep_seen = seen
                print(
                    f"[hist-keep] key_tokens={key_len} count={count} "
                    f"selected={selected}",
                    flush=True)

        return out

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        viewmats=None,
        Ks=None,
        prope_kv_cache=None,
        prope_window_viewmats=None,
        prope_window_Ks=None,
        replace_sink=False,
        cache_keep_frames=None,
        cache_sink_frames=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
            viewmats(Tensor, optional): Shape [B, L, 4, 4] camera extrinsics for PRoPE
            Ks(Tensor, optional): Shape [B, L, 3, 3] camera intrinsics for PRoPE
            prope_kv_cache(dict, optional): PRoPE KV cache for inference, same structure as kv_cache
            prope_window_viewmats(Tensor, optional): Shape [B, F_window, 4, 4] camera
                extrinsics for every frame currently in the PRoPE cache (sink +
                previous chunk + current chunk), in the current chunk's anchor
                gauge. When set together with ``prope_chunk_anchor`` the cache
                stores raw (unprojected) K/V and re-projects the whole window
                under this gauge at every call, keeping cross-chunk geometry
                exact while bounding transform magnitudes.
            prope_window_Ks(Tensor, optional): Shape [B, F_window, 3, 3]
            replace_sink(bool, optional): Overwrite the persistent sink region
                (positions [0, num_new_tokens)) with this call's clean frame,
                leaving history and valid-length indices untouched. Used for
                rolling per-block conditioning: block k's anchor = block k-1's
                last frame (block 0 = reference).
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        history_cross = None
        if self.history_cross_enabled and kv_cache is None:
            # Packed TF layout is [clean prefix, noisy targets].  This branch
            # is deliberately separate from self-attention: generated tokens
            # query a routed memory built only from already-observed blocks.
            full_s = q.shape[1]
            if full_s > seq_lens[0].item() * 1.5:
                half = full_s // 2
                frame_seqlen_h = math.prod(grid_sizes[0][1:]).item()
                clean_k, clean_v = k[:, :half], v[:, :half]
                noisy_q = q[:, half:]
                history_noisy = torch.zeros_like(noisy_q)
                frames = int(grid_sizes[0, 0].item())
                block = self.history_cross_block_frames
                spans = ([(0, 1)] + [(s, min(frames, s + block))
                         for s in range(1, frames, block)]
                         if self.history_cross_independent_first_frame
                         else [(s, min(frames, s + block))
                               for s in range(0, frames, block)])
                for frame_start, frame_end in spans:
                    token_start = frame_start * frame_seqlen_h
                    token_end = frame_end * frame_seqlen_h
                    history_noisy[:, token_start:token_end] = self._routed_history_attention(
                        noisy_q[:, token_start:token_end],
                        clean_k[:, :token_start], clean_v[:, :token_start],
                        frame_seqlen_h)
                history_cross = torch.cat([torch.zeros_like(q[:, :half]), history_noisy], dim=1)

        # PRoPE: apply BEFORE SP all-to-all so viewmats/Ks and q/k/v have matching lengths
        prope_enabled = (viewmats is not None) and hasattr(self, 'prope_o') and \
                        (kv_cache is None or prope_kv_cache is not None)
        chunk_anchor = bool(getattr(self, "prope_chunk_anchor", False)) and \
            prope_window_viewmats is not None
        if (os.environ.get("WAN_HISTORY_DIAG") == "1" and
                not hasattr(self, "_history_status_printed")):
            self._history_status_printed = True
            print(
                f"[history-status] enabled={self.history_cross_enabled} "
                f"content={self.history_cross_content_space} "
                f"prope={prope_enabled} chunk_anchor={chunk_anchor} "
                f"kv_cache={kv_cache is not None}", flush=True)
        window_apply_kv = None
        window_apply_kv_sink = None
        if prope_enabled:
            from models.prope import prope_qkv
            if chunk_anchor:
                from models.prope import _prepare_apply_fns_all_dim
                # Per-frame window transforms (cache order == window order).
                # Both use the same anchor gauge as the current chunk, so
                # queries and every cached key share one camera frame.
                window_apply_kv = _prepare_apply_fns_all_dim(
                    head_dim=self.head_dim,
                    viewmats=prope_window_viewmats,
                    Ks=prope_window_Ks,
                    patches_x=None, patches_y=None,
                    image_width=None, image_height=None,
                )[1]
                if getattr(self, "prope_sink_dual", False):
                    # Second hypothesis for the reference frame.  The window is
                    # already in the anchor's gauge, so the anchor pose is the
                    # identity; giving the sink that pose says "the reference
                    # was taken from where the camera is now", i.e. its content
                    # is rigidly attached to the camera rather than to the
                    # world.  Both projections of the sink are offered to the
                    # camera softmax and each query token picks per content:
                    # shelf tokens match the world-projected sink, the
                    # first-person hand matches the camera-attached one.  A
                    # single global choice cannot serve both -- excluding the
                    # sink holds case 171's hand but inflates it and weakens
                    # the direction signal, keeping it drops the hand.
                    sink_vm = prope_window_viewmats.clone()
                    eye = torch.eye(
                        4, dtype=sink_vm.dtype, device=sink_vm.device)
                    sink_vm[:, :max(1, self.sink_size)] = eye
                    window_apply_kv_sink = _prepare_apply_fns_all_dim(
                        head_dim=self.head_dim,
                        viewmats=sink_vm,
                        Ks=prope_window_Ks,
                        patches_x=None, patches_y=None,
                        image_width=None, image_height=None,
                    )[1]
            q_p, k_p, v_p, apply_fn_o = prope_qkv(
                q.permute(0, 2, 1, 3),
                k.permute(0, 2, 1, 3),
                v.permute(0, 2, 1, 3),
                viewmats=viewmats,
                Ks=Ks,
            )
            q_p = q_p.permute(0, 2, 1, 3)
            k_p = k_p.permute(0, 2, 1, 3)
            v_p = v_p.permute(0, 2, 1, 3)

        # SP: scatter heads, gather sequence for attention
        if _HAS_CLEANCODE_INFRA:
            parallel_dims = get_parallel_state()
            sp_enabled = parallel_dims.sp_enabled
        else:
            _require_cleancode_infra("CausalWanSelfAttention.forward SP check")
            sp_enabled = False

        if sp_enabled:
            q = sequence_model_parallel_all_to_all_4D(q, scatter_dim=2, gather_dim=1)
            k = sequence_model_parallel_all_to_all_4D(k, scatter_dim=2, gather_dim=1)
            v = sequence_model_parallel_all_to_all_4D(v, scatter_dim=2, gather_dim=1)
            if prope_enabled:
                q_p = sequence_model_parallel_all_to_all_4D(q_p, scatter_dim=2, gather_dim=1)
                k_p = sequence_model_parallel_all_to_all_4D(k_p, scatter_dim=2, gather_dim=1)
                v_p = sequence_model_parallel_all_to_all_4D(v_p, scatter_dim=2, gather_dim=1)

        is_tf = False
        sp_world_size = 0
        per_rank_half = 0
        if kv_cache is None:
            # if it is teacher forcing training?
            # Use q.shape[1] (post all-to-all full seq len) for TF detection
            # With SP padding, full_s may be slightly larger than seq_lens[0]*2,
            # so use > 1.5x threshold instead of exact equality.
            full_s = q.shape[1]
            is_tf = (full_s > seq_lens[0].item() * 1.5)
            if is_tf:
                # SP + TF: after all-to-all, sequence is interleaved:
                # [clean_r0, noisy_r0, clean_r1, noisy_r1, ...]
                # Reorder to contiguous [all_clean, all_noisy] for correct chunk(2).
                if sp_enabled:
                    sp_world_size = parallel_dims.sp
                    chunk_size = full_s // sp_world_size  # tokens per SP rank = 2 * (L/sp)
                    per_rank_half = chunk_size // 2

                    def _interleaved_to_contiguous(x):
                        B, S, H, D = x.shape
                        return x.reshape(B, sp_world_size, 2, per_rank_half, H, D) \
                                .permute(0, 2, 1, 3, 4, 5) \
                                .reshape(B, S, H, D)

                    q = _interleaved_to_contiguous(q)
                    k = _interleaved_to_contiguous(k)
                    v = _interleaved_to_contiguous(v)
                    if prope_enabled:
                        q_p = _interleaved_to_contiguous(q_p)
                        k_p = _interleaved_to_contiguous(k_p)
                        v_p = _interleaved_to_contiguous(v_p)

                    # Strip SP padding before attention.
                    # After contiguous reorder: [clean_valid, clean_pad, noisy_valid, noisy_pad]
                    unpadded_half = seq_lens[0].item()
                    sp_pad_per_half = full_s // 2 - unpadded_half
                    if sp_pad_per_half > 0:
                        # Remove padding from each half
                        def _strip_sp_pad(t):
                            c = t[:, :unpadded_half]
                            n = t[:, full_s // 2:full_s // 2 + unpadded_half]
                            return torch.cat([c, n], dim=1)
                        q = _strip_sp_pad(q)
                        k = _strip_sp_pad(k)
                        v = _strip_sp_pad(v)
                        if prope_enabled:
                            q_p = _strip_sp_pad(q_p)
                            k_p = _strip_sp_pad(k_p)
                            v_p = _strip_sp_pad(v_p)

                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )
                x = x[:, :, :q.shape[1]].transpose(2, 1) if padded_length > 0 else x.transpose(2, 1)

                # PRoPE attention path (TF mode)
                if prope_enabled:
                    padded_length_p = padded_length
                    if padded_length_p > 0:
                        pad_zeros_p = torch.zeros([q_p.shape[0], padded_length_p, q_p.shape[2], q_p.shape[3]],
                                                  device=q_p.device, dtype=q_p.dtype)
                        padded_q_p = torch.cat([q_p, pad_zeros_p], dim=1)
                        padded_k_p = torch.cat([k_p, pad_zeros_p], dim=1)
                        padded_v_p = torch.cat([v_p, pad_zeros_p], dim=1)
                    else:
                        padded_q_p, padded_k_p, padded_v_p = q_p, k_p, v_p

                    x_prope = flex_attention(
                        query=padded_q_p.transpose(2, 1),
                        key=padded_k_p.transpose(2, 1),
                        value=padded_v_p.transpose(2, 1),
                        block_mask=block_mask
                    )
                    x_prope = x_prope[:, :, :q_p.shape[1]].transpose(2, 1) if padded_length_p > 0 else x_prope.transpose(2, 1)

                # Restore SP padding after attention so reverse reorder works
                if sp_enabled and sp_pad_per_half > 0:
                    B_x, S_x, H_x, D_x = x.shape
                    half_valid = S_x // 2  # == unpadded_half
                    pad_t = x.new_zeros(B_x, sp_pad_per_half, H_x, D_x)
                    x = torch.cat([
                        x[:, :half_valid], pad_t,
                        x[:, half_valid:], pad_t
                    ], dim=1)
                    if prope_enabled:
                        x_prope = torch.cat([
                            x_prope[:, :half_valid], pad_t,
                            x_prope[:, half_valid:], pad_t
                        ], dim=1)

            else:
                # DF path removed — only Teacher Forcing is supported.
                assert False, "Diffusion Forcing is currently not supported. Only Teacher Forcing is supported."
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            # self.max_attention_size is local_attn_size * 1560, a token count
            # for the resolution this file was written against. sink_tokens two
            # lines down is computed from the real frame_seqlen, so at any other
            # resolution the two disagree and the window silently stops being
            # local_attn_size frames wide. Here frame_seqlen is 390 (30x52
            # latents, 2x2 patches), so a configured window of 12 was really 48.
            # Everything the cache path does -- eviction, window-relative RoPE,
            # the PRoPE window -- keys off this number.
            max_attention_size = (self.max_attention_size
                                  if self.local_attn_size == -1
                                  else self.local_attn_size * frame_seqlen)
            num_new_tokens = q.shape[1]
            current_end = current_start + num_new_tokens
            sink_tokens = self.sink_size * frame_seqlen
            if self.history_cross_enabled and self.history_cross_full_memory:
                # Absolute raw-QKV history bank. Repeated denoising calls for
                # one block overwrite the same slice; the timestep-zero clean
                # commit is the last writer before the next block reads it.
                # Thus noisy intermediate states never accumulate as extra
                # segments, while early clean object details are never evicted.
                old_k = kv_cache.get("history_full_k")
                old_v = kv_cache.get("history_full_v")
                if old_k is None or old_k.shape[1] < current_end:
                    new_k = k.new_zeros(k.shape[0], current_end, *k.shape[2:])
                    new_v = v.new_zeros(v.shape[0], current_end, *v.shape[2:])
                    if old_k is not None:
                        new_k[:, :old_k.shape[1]] = old_k
                        new_v[:, :old_v.shape[1]] = old_v
                    kv_cache["history_full_k"] = new_k
                    kv_cache["history_full_v"] = new_v
                kv_cache["history_full_k"][:, current_start:current_end] = k.detach()
                kv_cache["history_full_v"][:, current_start:current_end] = v.detach()
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            if "frame_ids" not in kv_cache:
                # Track from the first write, before the first eviction.
                # Retrieval cannot reconstruct IDs from unlabelled slots.
                kv_cache["frame_ids"] = torch.full(
                    (kv_cache_size // frame_seqlen,), -1,
                    dtype=torch.long, device=kv_cache["k"].device)
            if replace_sink:
                # Rolling conditioning prefill: only the sink region is
                # rewritten. The attention output here is discarded by the
                # caller, so it only needs to be shape-correct.
                kv_cache["k"][:, :num_new_tokens] = k
                kv_cache["v"][:, :num_new_tokens] = v
                attn_end = kv_cache["local_end_index"].item()
            else:
                if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                        num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                    # Calculate the number of new tokens added in this step
                    # Shift existing cache content left to discard oldest tokens
                    # Clone the source slice to avoid overlapping memory error
                    num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                    num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                    if cache_keep_frames is not None:
                        # Evict by what the next block is about to look at
                        # rather than by age. Dropping the oldest is what makes
                        # a corridor walked forty frames ago unrecoverable: the
                        # frames that saw it are gone, so nothing constrains
                        # the redraw on the way back. The caller picks the
                        # survivors by frustum overlap and hands the same list
                        # to PRoPE, which is required anyway -- the backbone
                        # rejects a window that disagrees with the residents.
                        #
                        # Reordering is safe here because the resident window
                        # is re-roped from slot zero every step, so position
                        # comes from the slot, not the frame's true age. The
                        # sink already relies on this.
                        _roll_cache_by_frames(
                            kv_cache, cache_keep_frames, frame_seqlen,
                            self.sink_size, num_rolled_tokens,
                            sink_frames=cache_sink_frames)
                    else:
                        kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                            kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                        kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                            kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                        n_keep = num_rolled_tokens // frame_seqlen
                        n_drop = num_evicted_tokens // frame_seqlen
                        ids = kv_cache["frame_ids"]
                        ids[self.sink_size:self.sink_size + n_keep] = ids[
                            self.sink_size + n_drop:self.sink_size + n_drop + n_keep].clone()
                        ids[self.sink_size + n_keep:] = -1
                    # Insert the new keys/values at the end
                    local_end_index = kv_cache["local_end_index"].item() + current_end - \
                        kv_cache["global_end_index"].item() - num_evicted_tokens
                    local_start_index = local_end_index - num_new_tokens
                    kv_cache["k"][:, local_start_index:local_end_index] = \
                        k
                    kv_cache["v"][:, local_start_index:local_end_index] = v
                else:
                    # Assign new keys/values directly up to current_end
                    local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                    local_start_index = local_end_index - num_new_tokens
                    kv_cache["k"][:, local_start_index:local_end_index] = \
                        k
                    kv_cache["v"][:, local_start_index:local_end_index] = v
                if "frame_ids" in kv_cache:
                    ids = kv_cache["frame_ids"]
                    first_slot = local_start_index // frame_seqlen
                    first_frame = current_start // frame_seqlen
                    for j in range(num_new_tokens // frame_seqlen):
                        if first_slot + j < ids.shape[0]:
                            ids[first_slot + j] = first_frame + j
                attn_end = local_end_index
                kv_cache["global_end_index"].fill_(current_end)
                kv_cache["local_end_index"].fill_(local_end_index)

            # Window-relative RoPE: the cache stores RAW keys; at every step
            # the resident window is re-roped with the window start as frame 0
            # and the query takes the tail positions. This pins the
            # sink<->query rotation distance at the window size forever, so
            # the reference frame never decays out of attention.
            window_start = max(0, attn_end - max_attention_size)
            window_frames = (attn_end - window_start) // frame_seqlen
            window_grid = grid_sizes.clone()
            window_grid[:, 0] = window_frames
            roped_key = causal_rope_apply(
                kv_cache["k"][:, window_start:attn_end], window_grid, freqs,
                start_frame=0).type_as(v)
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs,
                start_frame=window_frames - num_new_tokens // frame_seqlen).type_as(v)
            attention_key = roped_key
            attention_value = kv_cache["v"][:, window_start:attn_end]
            sink_repeat = max(1, int(self.sink_attention_repeat))
            if sink_tokens > 0 and sink_repeat > 1 and window_start == 0:
                attention_key = torch.cat(
                    [roped_key[:, :sink_tokens]] * (sink_repeat - 1) +
                    [roped_key], dim=1)
                attention_value = torch.cat(
                    [attention_value[:, :sink_tokens]] * (sink_repeat - 1) +
                    [attention_value], dim=1)
            hist_repeat = max(1, int(
                getattr(self, "history_attention_repeat", 1)))
            if hist_repeat > 1:
                hist_end = attention_key.shape[1] - num_new_tokens
                if hist_end > sink_tokens:
                    h_k = attention_key[:, sink_tokens:hist_end]
                    h_v = attention_value[:, sink_tokens:hist_end]
                    attention_key = torch.cat(
                        [attention_key] + [h_k] * (hist_repeat - 1), dim=1)
                    attention_value = torch.cat(
                        [attention_value] + [h_v] * (hist_repeat - 1), dim=1)
            rollout_repeat = max(1, int(
                getattr(self, "rollout_attention_repeat", 1)))
            if rollout_repeat > 1 and num_new_tokens > 0:
                # The window's tail is the block being generated. Repeating it
                # is a +ln(repeat) logit bias on the rollout, which is the same
                # as -ln(repeat) on sink and history: they stay in the window
                # and keep contributing, they just stop dominating.
                cur_k = attention_key[:, -num_new_tokens:]
                cur_v = attention_value[:, -num_new_tokens:]
                attention_key = torch.cat(
                    [attention_key] + [cur_k] * (rollout_repeat - 1), dim=1)
                attention_value = torch.cat(
                    [attention_value] + [cur_v] * (rollout_repeat - 1), dim=1)
            x = attention(roped_query, attention_key, attention_value)

            # History-cross over the resident window (everything before the
            # current block), in the same window-relative coordinates as the
            # main attention.  With PRoPE chunk anchoring, content-space
            # history is computed below from that cache's raw (unrotated)
            # K/V instead.
            if (self.history_cross_enabled and not replace_sink and
                    (not prope_enabled or self.history_cross_content_space)):
                memory_ref_k = kv_cache.get("memory_ref_k")
                if (self.history_cross_content_space
                        and self.history_cross_full_memory
                        and current_start > 0):
                    if os.environ.get("WAN_HISTORY_DIAG") == "1":
                        seen = getattr(self, "_hist_bank_seen", set())
                        if current_start not in seen:
                            seen.add(current_start)
                            self._hist_bank_seen = seen
                            print(
                                f"[hist-bank] current_start={current_start} "
                                f"frames={current_start // frame_seqlen} "
                                f"window_frames={max_attention_size // frame_seqlen}",
                                flush=True)
                    history_cross = self._routed_history_attention(
                        q, kv_cache["history_full_k"][:, :current_start],
                        kv_cache["history_full_v"][:, :current_start],
                        frame_seqlen)
                elif (self.history_cross_content_space
                        and self.history_cross_use_memory
                        and memory_ref_k is not None):
                    memory_k = memory_ref_k
                    memory_v = kv_cache["memory_ref_v"]
                    if kv_cache.get("memory_summary_k") is not None:
                        memory_k = torch.cat(
                            [memory_k, kv_cache["memory_summary_k"]], dim=1)
                        memory_v = torch.cat(
                            [memory_v, kv_cache["memory_summary_v"]], dim=1)
                    history_cross = attention(q, memory_k, memory_v)
                else:
                    hist_len = (attn_end - window_start) - num_new_tokens
                    if hist_len > 0:
                        history_cross = self._routed_history_attention(
                            q if self.history_cross_content_space else roped_query,
                            (kv_cache["k"][:, window_start:window_start + hist_len]
                             if self.history_cross_content_space
                             else roped_key[:, :hist_len]),
                            kv_cache["v"][:, window_start:window_start + hist_len],
                            frame_seqlen)

            # PRoPE second attention path (inference mode with KV cache)
            if prope_enabled and prope_kv_cache is not None:
                # q_p, k_p, v_p were already computed above by prope_qkv.
                # In chunk-anchor mode the cache stores RAW k/v (gauge-free) so
                # the whole window can be re-projected under a new anchor at
                # every chunk boundary; otherwise store the projected k_p/v_p.
                # Use the same eviction logic as the RoPE cache
                prope_cache_size = prope_kv_cache["k"].shape[1]
                if "frame_ids" not in prope_kv_cache:
                    prope_kv_cache["frame_ids"] = torch.full(
                        (prope_cache_size // frame_seqlen,), -1,
                        dtype=torch.long, device=prope_kv_cache["k"].device)
                if replace_sink:
                    prope_kv_cache["k"][:, :num_new_tokens] = k_p
                    prope_kv_cache["v"][:, :num_new_tokens] = v_p
                    p_attn_end = prope_kv_cache["local_end_index"].item()
                    x_prope = attention(
                        q_p,
                        prope_kv_cache["k"][:, max(0, p_attn_end - max_attention_size):p_attn_end],
                        prope_kv_cache["v"][:, max(0, p_attn_end - max_attention_size):p_attn_end]
                    )
                elif self.local_attn_size != -1 and (current_end > prope_kv_cache["global_end_index"].item()) and (
                        num_new_tokens + prope_kv_cache["local_end_index"].item() > prope_cache_size):
                    p_num_evicted = num_new_tokens + prope_kv_cache["local_end_index"].item() - prope_cache_size
                    p_num_rolled = prope_kv_cache["local_end_index"].item() - p_num_evicted - sink_tokens
                    if cache_keep_frames is not None:
                        # The PRoPE cache must be promoted identically or the
                        # two caches disagree on which frame sits in each slot,
                        # which the order check below catches.
                        _roll_cache_by_frames(
                            prope_kv_cache, cache_keep_frames, frame_seqlen,
                            self.sink_size, p_num_rolled,
                            sink_frames=cache_sink_frames)
                    else:
                        prope_kv_cache["k"][:, sink_tokens:sink_tokens + p_num_rolled] = \
                            prope_kv_cache["k"][:, sink_tokens + p_num_evicted:sink_tokens + p_num_evicted + p_num_rolled].clone()
                        prope_kv_cache["v"][:, sink_tokens:sink_tokens + p_num_rolled] = \
                            prope_kv_cache["v"][:, sink_tokens + p_num_evicted:sink_tokens + p_num_evicted + p_num_rolled].clone()
                        n_keep = p_num_rolled // frame_seqlen
                        n_drop = p_num_evicted // frame_seqlen
                        ids = prope_kv_cache["frame_ids"]
                        ids[self.sink_size:self.sink_size + n_keep] = ids[
                            self.sink_size + n_drop:self.sink_size + n_drop + n_keep].clone()
                        ids[self.sink_size + n_keep:] = -1
                    p_local_end = prope_kv_cache["local_end_index"].item() + current_end - \
                        prope_kv_cache["global_end_index"].item() - p_num_evicted
                    p_local_start = p_local_end - num_new_tokens
                    if chunk_anchor:
                        prope_kv_cache["k"][:, p_local_start:p_local_end] = k
                        prope_kv_cache["v"][:, p_local_start:p_local_end] = v
                    else:
                        prope_kv_cache["k"][:, p_local_start:p_local_end] = k_p
                        prope_kv_cache["v"][:, p_local_start:p_local_end] = v_p
                else:
                    p_local_end = prope_kv_cache["local_end_index"].item() + current_end - prope_kv_cache["global_end_index"].item()
                    p_local_start = p_local_end - num_new_tokens
                    if chunk_anchor:
                        prope_kv_cache["k"][:, p_local_start:p_local_end] = k
                        prope_kv_cache["v"][:, p_local_start:p_local_end] = v
                    else:
                        prope_kv_cache["k"][:, p_local_start:p_local_end] = k_p
                        prope_kv_cache["v"][:, p_local_start:p_local_end] = v_p

                if not replace_sink:
                    first_slot = p_local_start // frame_seqlen
                    first_frame = current_start // frame_seqlen
                    n_frames = num_new_tokens // frame_seqlen
                    prope_kv_cache["frame_ids"][first_slot:first_slot + n_frames] = torch.arange(
                        first_frame, first_frame + n_frames,
                        device=prope_kv_cache["frame_ids"].device)
                    if cache_keep_frames is not None and not torch.equal(
                            kv_cache["frame_ids"], prope_kv_cache["frame_ids"]):
                        raise RuntimeError("Content and PRoPE cache frame order disagree")
                    window_start = max(0, p_local_end - max_attention_size)
                    if chunk_anchor:
                        # Re-project every cached frame under the current window
                        # gauge so cached keys and current queries share one frame.
                        window_frames = (p_local_end - window_start) // frame_seqlen
                        if window_frames != prope_window_viewmats.shape[1]:
                            raise RuntimeError(
                                f"PRoPE window mismatch: cache holds {window_frames} "
                                f"frames but prope_window_viewmats has "
                                f"{prope_window_viewmats.shape[1]}")
                        raw_k_w = prope_kv_cache["k"][:, window_start:p_local_end]
                        raw_v_w = prope_kv_cache["v"][:, window_start:p_local_end]
                        k_p_w = window_apply_kv(
                            raw_k_w.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
                        v_p_w = window_apply_kv(
                            raw_v_w.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
                        # Match the non-chunk-anchor path: a clean I2V sink is
                        # identity/content memory, not world-camera memory.
                        # Previously prope_exclude_sink was silently ignored
                        # whenever chunk anchoring was enabled (the default in
                        # our causal camera configs), so the hand/held object
                        # was transformed with the background and drifted out
                        # on W while remaining on S.
                        rel_pr_start = 0
                        if getattr(self, "prope_exclude_sink", False):
                            rel_pr_start = min(sink_tokens, k_p_w.shape[1])
                        pr_k = k_p_w[:, rel_pr_start:]
                        pr_v = v_p_w[:, rel_pr_start:]
                        if window_apply_kv_sink is not None:
                            n_sink = min(sink_tokens, raw_k_w.shape[1])
                            if n_sink > 0:
                                k_sink = window_apply_kv_sink(
                                    raw_k_w.permute(0, 2, 1, 3)
                                ).permute(0, 2, 1, 3)[:, :n_sink]
                                v_sink = window_apply_kv_sink(
                                    raw_v_w.permute(0, 2, 1, 3)
                                ).permute(0, 2, 1, 3)[:, :n_sink]
                                # Repeating the camera-attached copy is a
                                # +ln(repeat) logit bias on that hypothesis --
                                # the same trick as sink_attention_repeat, and
                                # for the same reason: what needs raising is
                                # the softmax probability, not the value.
                                repeat = max(1, int(getattr(
                                    self, "prope_sink_dual_repeat", 1)))
                                pr_k = torch.cat(
                                    [pr_k] + [k_sink] * repeat, dim=1)
                                pr_v = torch.cat(
                                    [pr_v] + [v_sink] * repeat, dim=1)
                        x_prope = attention(q_p, pr_k, pr_v)
                        if (self.history_cross_enabled
                                and not self.history_cross_content_space):
                            history_start = max(0, p_local_start - max_attention_size)
                            rel_s = max(0, history_start - window_start)
                            rel_e = max(0, p_local_start - window_start)
                            history_cross = self._routed_history_attention(
                                q_p, k_p_w[:, rel_s:rel_e], v_p_w[:, rel_s:rel_e],
                                frame_seqlen)
                        elif (self.history_cross_enabled
                              and self.history_cross_content_space
                              and history_cross is None):
                            # `history_cross is None` guard: with
                            # history_cross_full_memory the read above already
                            # covered every observed frame from the absolute
                            # bank.  Recomputing here would silently replace it
                            # with the resident local window (sink + one block
                            # at local_attn_size=9), which is not what the
                            # branch was trained on -- training (kv_cache is
                            # None) gives it clean_k[:, :token_start], the
                            # whole prefix.  That overwrite made
                            # history_cross_full_memory a dead switch whenever
                            # prope_chunk_anchor was on, i.e. in every causal
                            # camera config, and no history_cross_scale can
                            # compensate for a read that never contains the
                            # early frames holding the subject.
                            # step175 trained this adapter before PRoPE/RoPE:
                            # noisy raw Q attends already-observed clean raw
                            # K/V.  Reproduce that feature distribution at
                            # inference instead of feeding temporally rotated
                            # ordinary-cache keys into the trained projection.
                            rel_s = 0
                            rel_e = max(0, p_local_start - window_start)
                            if os.environ.get("WAN_HISTORY_DIAG") == "1":
                                print(
                                    f"[hist-window] current_start={current_start} "
                                    f"p_local_start={p_local_start} "
                                    f"window_start={window_start} "
                                    f"rel_s={rel_s} rel_e={rel_e} "
                                    f"frames={rel_e // frame_seqlen} "
                                    f"includes_ref={window_start == 0}",
                                    flush=True)
                            history_cross = self._routed_history_attention(
                                q, raw_k_w[:, rel_s:rel_e],
                                raw_v_w[:, rel_s:rel_e], frame_seqlen)
                    else:
                        pr_start = window_start
                        if getattr(self, "prope_exclude_sink", False):
                            # Keep the static sink (reference image) out of the
                            # camera branch so its identity camera / clean
                            # content cannot crowd out the PRoPE signal.
                            pr_start = max(pr_start, sink_tokens)
                        x_prope = attention(
                            q_p,
                            prope_kv_cache["k"][:, pr_start:p_local_end],
                            prope_kv_cache["v"][:, pr_start:p_local_end]
                        )
                        if (self.history_cross_enabled
                                and not self.history_cross_content_space):
                            history_start = max(0, p_local_start - max_attention_size)
                            history_cross = self._routed_history_attention(
                                q_p,
                                prope_kv_cache["k"][:, history_start:p_local_start],
                                prope_kv_cache["v"][:, history_start:p_local_start],
                                frame_seqlen)
                    prope_kv_cache["global_end_index"].fill_(current_end)
                    prope_kv_cache["local_end_index"].fill_(p_local_end)

        # SP: scatter sequence, gather heads back
        if sp_enabled:
            # TF + SP: reorder from contiguous [all_clean, all_noisy] back to
            # interleaved [clean_r0, noisy_r0, ...] before reverse all-to-all
            if is_tf:
                B_x, S_x, H_x, D_x = x.shape
                x = x.reshape(B_x, 2, sp_world_size, per_rank_half, H_x, D_x) \
                      .permute(0, 2, 1, 3, 4, 5) \
                      .reshape(B_x, S_x, H_x, D_x)
                if prope_enabled:
                    x_prope = x_prope.reshape(B_x, 2, sp_world_size, per_rank_half, H_x, D_x) \
                                     .permute(0, 2, 1, 3, 4, 5) \
                                     .reshape(B_x, S_x, H_x, D_x)
            x = sequence_model_parallel_all_to_all_4D(x, scatter_dim=1, gather_dim=2)
            if prope_enabled:
                x_prope = sequence_model_parallel_all_to_all_4D(x_prope, scatter_dim=1, gather_dim=2)

        # output
        base_attention_pre = x.flatten(2)
        x = self.o(base_attention_pre)

        if history_cross is not None:
            history_projected = self.history_cross_o(history_cross.flatten(2))
            if os.environ.get("WAN_HISTORY_DIAG") == "1":
                diag = getattr(self, "_history_diag", None)
                if diag is None:
                    diag = self._history_diag = {}
                diag_key = int(current_start)
                if diag_key not in diag:
                    with torch.no_grad():
                        base_rms = x.float().square().mean().sqrt().item()
                        history_rms = history_projected.float().square().mean().sqrt().item()
                        diag[diag_key] = (base_rms, history_rms)
                        print(
                            f"[history-diag] start={diag_key} "
                            f"base_rms={base_rms:.6g} history_rms={history_rms:.6g} "
                            f"ratio={history_rms / max(base_rms, 1e-12):.6g} "
                            f"scale={self.history_cross_scale:.4g}",
                            flush=True)
            x = x + self.history_cross_scale * history_projected

        # PRoPE output path: fuse with zero-init projection
        if prope_enabled:
            # apply_fn_o correction: [B, L, H, D] → [B, H, L, D] → correct → [B, L, H, D]
            x_prope = apply_fn_o(x_prope.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            x_prope_pre = x_prope.flatten(2)
            x_prope = self.prope_o(x_prope_pre)
            if getattr(self, "capture_camera_context", False):
                context_tokens = x_prope
                # Teacher-forcing packs [clean, noisy].  The generated/noisy
                # half is the one that must match inference-time context.
                if is_tf:
                    context_tokens = context_tokens[:, context_tokens.shape[1] // 2:]
                self.camera_context = context_tokens.mean(dim=1)
            if os.environ.get("WAN_CAMERA_DIAG") == "1":
                # One sample per temporal chunk is enough to identify whether
                # camera attention or its output projection suppresses the
                # signal, without adding reductions to every denoising step.
                diag = getattr(self, "_camera_diag", None)
                if diag is None:
                    diag = self._camera_diag = {}
                diag_key = int(current_start)
                if diag_key not in diag:
                    with torch.no_grad():
                        base_rms = x.float().square().mean().sqrt().item()
                        pre_rms = x_prope_pre.float().square().mean().sqrt().item()
                        out_rms = x_prope.float().square().mean().sqrt().item()
                        diag[diag_key] = {
                            "base_rms": base_rms,
                            "prope_attention_rms": pre_rms,
                            "prope_projected_rms": out_rms,
                            "projected_to_base": out_rms / max(base_rms, 1e-12),
                            # Compact direction-sensitive summaries. PRoPE's
                            # geometric transforms can preserve norms while
                            # rotating the feature, so RMS alone cannot prove
                            # that two trajectories produce the same signal.
                            "base_signature": x[0].float().mean(dim=0).cpu(),
                            "prope_attention_signature": x_prope_pre[0].float().mean(dim=0).cpu(),
                            "projected_signature": x_prope[0].float().mean(dim=0).cpu(),
                        }
            # Default 1.0 preserves checkpoint behavior. A diagnostic/control
            # scale lets us test whether causal conversion merely weakened an
            # otherwise trajectory-sensitive PRoPE branch.
            prope_scale = float(os.environ.get(
                "WAN_PROPE_SCALE", getattr(self, "prope_scale", 1.0)))
            x = x + prope_scale * x_prope

        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
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
        # Keep caption guidance strong while establishing the shot, then let
        # visual history dominate long rollouts instead of re-sampling a new
        # prompt-compatible scene when local visual KV becomes uncertain.
        self.text_cross_attn_late_scale = 1.0
        self.text_cross_attn_decay_start = 10**9
        self.text_cross_attn_decay_frames = 1
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.camera_film = None

    def enable_event_conditioning(self, cross_attn_type=None):
        """Graft the independent event-prompt cross-attention onto this block.

        Deliberately not built in ``__init__``. The branch is a full extra
        cross-attention -- ~4*dim^2 per block, over a billion parameters across
        the stack -- so building it unconditionally would make every existing
        checkpoint load with a billion randomly initialised weights no loss ever
        touches, and would inflate every saved checkpoint by the same amount.
        Grafting it afterwards follows the same rule as PRoPE and the action
        embedder: the base checkpoint loads first and knows every key it sees,
        and ``o`` is zero-initialised so the grafted model is numerically
        identical to the one it was grafted onto.
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

    def enable_camera_film(self, rank=64):
        if self.camera_film is not None:
            return
        rank = min(int(rank), self.dim)
        self.camera_film = nn.Sequential(
            nn.Linear(6, rank), nn.SiLU(), nn.Linear(rank, self.dim * 2))
        nn.init.zeros_(self.camera_film[-1].weight)
        nn.init.zeros_(self.camera_film[-1].bias)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        viewmats=None,
        Ks=None,
        prope_kv_cache=None,
        prope_window_viewmats=None,
        prope_window_Ks=None,
        replace_sink=False,
        cache_keep_frames=None,
        cache_sink_frames=None,
        camera_film_tokens=None,
        event_context=None,
        event_context_lens=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C] (frame-level) or [B, L, 6, C] (token-level, SP mode)
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        token_level_e = (e.shape[1] == x.shape[1])
        e_num_frames = e.shape[1]  # save before chunk() turns e into a tuple
        cross_frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        current_frame = current_start // cross_frame_seqlen
        decay = max(1, int(self.text_cross_attn_decay_frames))
        alpha = min(1.0, max(0.0, (
            current_frame - int(self.text_cross_attn_decay_start)) / decay))
        text_cross_scale = 1.0 + alpha * (
            float(self.text_cross_attn_late_scale) - 1.0)
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        if token_level_e:
            # SP token-level: e[i] is [B, L, 1, C], squeeze to [B, L, C]
            y = self.self_attn(
                self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
                seq_lens, grid_sizes,
                freqs, block_mask, kv_cache, current_start, cache_start,
                viewmats=viewmats, Ks=Ks, prope_kv_cache=prope_kv_cache,
                prope_window_viewmats=prope_window_viewmats,
                prope_window_Ks=prope_window_Ks, replace_sink=replace_sink,
                cache_keep_frames=cache_keep_frames,
                cache_sink_frames=cache_sink_frames)
            x = x + y * e[2].squeeze(2)

            # cross-attention & ffn function
            def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
                x = x + text_cross_scale * self.cross_attn(
                    self.norm3(x), context, context_lens,
                    crossattn_cache=crossattn_cache)
                y = self.ffn(
                    self.norm2(x) * (1 + e[4].squeeze(2)) + e[3].squeeze(2)
                )
                x = x + y * e[5].squeeze(2)
                return x
        else:
            # Original frame-level path
            num_frames, frame_seqlen = e_num_frames, x.shape[1] // e_num_frames

            # self-attention
            y = self.self_attn(
                (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
                seq_lens, grid_sizes,
                freqs, block_mask, kv_cache, current_start, cache_start,
                viewmats=viewmats, Ks=Ks, prope_kv_cache=prope_kv_cache,
                prope_window_viewmats=prope_window_viewmats,
                prope_window_Ks=prope_window_Ks, replace_sink=replace_sink,
                cache_keep_frames=cache_keep_frames,
                cache_sink_frames=cache_sink_frames)

            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

            # cross-attention & ffn function
            def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
                x = x + text_cross_scale * self.cross_attn(
                    self.norm3(x), context, context_lens,
                    crossattn_cache=crossattn_cache)
                y = self.ffn(
                    (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
                )
                # with amp.autocast(dtype=torch.float32):
                x = x + (y.unflatten(dim=1, sizes=(num_frames,
                         frame_seqlen)) * e[5]).flatten(1, 2)
                return x

        # Camera pose is injected as feature-wise scale/shift, not as a
        # cross-attention memory.  Zero-init makes this branch an exact no-op
        # when grafted onto an existing checkpoint.
        if self.camera_film is not None and camera_film_tokens is not None:
            if camera_film_tokens.shape[1] != x.shape[1]:
                raise RuntimeError(
                    f"camera FiLM tokens {camera_film_tokens.shape[1]} != video tokens {x.shape[1]}")
            scale, shift = self.camera_film(camera_film_tokens).chunk(2, dim=-1)
            x = x * (1 + scale) + shift

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)

        # Independent event prompt: a second text memory injected through its
        # own cross-attention so it can be guided separately from the main
        # caption (drop one while keeping the other).
        if event_context is not None and self.event_cross_attn is not None:
            # `event_scale` is the auxiliary knob: the caption drives the scene
            # through the main cross-attention, the event rides on top as a
            # residual that can be dialled down (or up) at inference without
            # retraining. 1.0 is the trained strength.
            x = x + self.event_scale * self.event_cross_attn(
                self.norm_event(x), event_context, event_context_lens)

        return x


class CausalHead(nn.Module):

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
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    # FSDP shard conditions — infra only, no model change
    _fsdp_shard_conditions = CausalWanConfig().arch_config._fsdp_shard_conditions if _HAS_CLEANCODE_INFRA else []

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
                 local_attn_size=-1,
                 sink_size=0,
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
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
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
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

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
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # Optional action output head (video + action prediction). Grafted
        # lazily by `add_action_head` so existing checkpoints load unchanged.
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

        self.block_mask = None
        self.moba_block_masks = {}
        self.moba_bidirectional_ratio = 0.0

        self.num_frame_per_block = 1
        self.independent_first_frame = False

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
        """(B, L, dim) hidden tokens -> (B, F, action_dim) pooled per frame."""
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

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1,
        independent_first_frame=False, local_attn_size=-1, sink_size=0,
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # debug
        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        sink_tokens = max(0, int(sink_size)) * frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        if independent_first_frame:
            spans = [(0, frame_seqlen)]
            spans += [(s, min(s + attention_block_size, clean_ends))
                      for s in range(frame_seqlen, clean_ends, attention_block_size)]
        else:
            spans = [(s, min(s + attention_block_size, clean_ends))
                     for s in range(0, clean_ends, attention_block_size)]

        # attention for clean context frames
        for start, end in spans:
            context_ends[start:end] = end
            if local_attn_size != -1:
                # Inference's cache capacity includes both the persistent sink
                # and the current block.  Keep the same total token budget in
                # teacher forcing: [sink | recent history | current block].
                local_tokens = max(
                    0, (int(local_attn_size) - int(sink_size)) * frame_seqlen)
                context_starts[start:end] = max(sink_tokens, end - local_tokens)

        # attention for noisy frames
        for clean_start, clean_end in spans:
            start, end = clean_ends + clean_start, clean_ends + clean_end
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = clean_start
            if local_attn_size != -1:
                current_block_tokens = clean_end - clean_start
                history_tokens = max(
                    0,
                    (int(local_attn_size) - int(sink_size)) * frame_seqlen
                    - current_block_tokens)
                noise_context_starts[start:end] = max(
                    sink_tokens, clean_start - history_tokens)

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = ((q_idx < clean_ends) &
                          (kv_idx < context_ends[q_idx]) &
                          (kv_idx >= context_starts[q_idx]))
            clean_sink_mask = ((q_idx < clean_ends) &
                               (kv_idx < sink_tokens))
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_sink = (kv_idx < sink_tokens)
            noise_mask = (q_idx >= clean_ends) & (C1 | C2 | noise_sink)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | clean_sink_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                                padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_bidirectional_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560,
    ) -> BlockMask:
        """MoBA's bidirectional component in the packed [clean, noisy] layout.

        The noisy half is an ordinary full-attention video and cannot read the
        clean teacher-forcing half.  This uses the same QKV/SDPA operator as
        the causal component; only the mask changes.
        """
        video_length = num_frames * frame_seqlen
        total_length = video_length * 2
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        def attention_mask(b, h, q_idx, kv_idx):
            clean = (q_idx < video_length) & (kv_idx < video_length)
            noisy = ((q_idx >= video_length) & (q_idx < total_length) &
                     (kv_idx >= video_length) & (kv_idx < total_length))
            return clean | noisy | (q_idx == kv_idx)

        return create_block_mask(
            attention_mask, B=None, H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False, device=device)

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        viewmats=None,
        Ks=None,
        prope_kv_cache=None,
        prope_window_viewmats=None,
        prope_window_Ks=None,
        replace_sink=False,
        cache_keep_frames=None,
        cache_sink_frames=None,
        actions=None,
        embodiment_id=None,
        event_context=None,
        return_actions=False,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

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

        for block in self.blocks:
            block.self_attn.history_cross_independent_first_frame = bool(
                self.independent_first_frame or getattr(
                    self, "history_cross_reference_only", False))

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # Action conditioning (wan/modules/action.py). e0 is already per-frame
        # here, so this is a straight additive term on the same [B, F, 6, C].
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

        # SP: chunk sequence along token dimension (same pattern as _forward_train)
        # KV cache is stored in head-parallel domain (post all-to-all inside attention),
        # following CleanCode's DMD pipeline design.
        if _HAS_CLEANCODE_INFRA:
            parallel_dims = get_parallel_state()
            sp_enabled = parallel_dims.sp_enabled
        else:
            _require_cleancode_infra("CausalWanModel._forward_infer SP check")
            sp_enabled = False
        if sp_enabled:
            sp_size = parallel_dims.sp
            sp_rank = parallel_dims.sp_rank
            x = torch.chunk(x, sp_size, dim=1)[sp_rank]
            # NOTE: e0 is NOT chunked — it has shape [B, 1, 6, dim] (per-frame),
            # and block.forward derives frame_seqlen from x.shape[1] // e.shape[1].

        camera_film_tokens = None
        if viewmats is not None and getattr(self, "camera_film_enabled", False):
            from models.camera_film import camera_ray_tokens
            camera_film_tokens = camera_ray_tokens(
                viewmats, Ks, grid_sizes, dtype=x.dtype)

        # PRoPE: expand viewmats/Ks from (B, F_chunk, *, *) to (B, seq_len, *, *)
        if viewmats is not None:
            expanded_vm, expanded_ks = [], []
            single_seq_len = seq_lens[0].item()
            for i, (f, h, w) in enumerate(grid_sizes.tolist()):
                vm = viewmats[i, :f]  # (F_chunk, 4, 4)
                vm = vm[:, None, None].expand(-1, h, w, -1, -1).reshape(f * h * w, 4, 4)
                pad_len = single_seq_len - f * h * w
                if pad_len > 0:
                    vm = torch.cat([vm, torch.eye(4, device=vm.device, dtype=vm.dtype).unsqueeze(0).expand(pad_len, -1, -1)])
                expanded_vm.append(vm)

                ks = Ks[i, :f]  # (F_chunk, 3, 3)
                ks = ks[:, None, None].expand(-1, h, w, -1, -1).reshape(f * h * w, 3, 3)
                if pad_len > 0:
                    ks = torch.cat([ks, torch.eye(3, device=ks.device, dtype=ks.dtype).unsqueeze(0).expand(pad_len, -1, -1)])
                expanded_ks.append(ks)

            viewmats = torch.stack(expanded_vm)  # (B, single_seq_len, 4, 4)
            Ks = torch.stack(expanded_ks)        # (B, single_seq_len, 3, 3)

            # SP: chunk viewmats/Ks along sequence dim to match x's per-rank slice.
            # x was chunked above; viewmats/Ks must follow so prope_qkv sees matching seqlen.
            if sp_enabled:
                viewmats = torch.chunk(viewmats, sp_size, dim=1)[sp_rank]
                Ks = torch.chunk(Ks, sp_size, dim=1)[sp_rank]

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )
        if viewmats is not None:
            kwargs['viewmats'] = viewmats
            kwargs['Ks'] = Ks
        if prope_window_viewmats is not None:
            kwargs['prope_window_viewmats'] = prope_window_viewmats
            kwargs['prope_window_Ks'] = prope_window_Ks
        kwargs['replace_sink'] = replace_sink
        kwargs['cache_keep_frames'] = cache_keep_frames
        kwargs['cache_sink_frames'] = cache_sink_frames
        if camera_film_tokens is not None:
            kwargs['camera_film_tokens'] = camera_film_tokens
        if event_ctx is not None:
            kwargs['event_context'] = event_ctx
            kwargs['event_context_lens'] = None

        def create_custom_forward(module, via_attr=False):
            if not via_attr:
                def custom_forward(*inputs, **kwargs):
                    return module(*inputs, **kwargs)
                return custom_forward

            # The keyword arguments travel on the module rather than through
            # the checkpoint call. Non-reentrant checkpointing saves whatever
            # it is passed so it can recompute, and the KV cache rides in those
            # kwargs -- 30 frames a step, one per block, each pinning a 5.6 GiB
            # cache that nothing could then free. Reached through `module`, the
            # frame pins only the module, whose attribute can be emptied once
            # the step's backward is done.
            def custom_forward(*inputs):
                return module(*inputs, **module._ckpt_kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "prope_kv_cache": prope_kv_cache[block_index] if prope_kv_cache is not None else None
                    }
                )
                # Non-reentrant, because reentrant cannot be used here.
                #
                # `use_reentrant=True` only runs the checkpointed region's
                # backward when one of the positional inputs requires grad.
                # Block 0's input comes off a frozen patch embedding, so it
                # does not, and the whole 30-block chain silently built no
                # graph: 600 trainable tensors, all 600 with `grad is None`,
                # the optimiser moving nothing, 26 steps of checkpoints
                # byte-identical to their starting point. It reads as a memory
                # fix because a run that computes no gradients has little to
                # hold.
                block._ckpt_kwargs = dict(kwargs)
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block, via_attr=True),
                    x,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "prope_kv_cache": prope_kv_cache[block_index] if prope_kv_cache is not None else None
                    }
                )
                x = block(x, **kwargs)

        # SP: gather sequence from all ranks before head
        if sp_enabled:
            x = sequence_model_parallel_all_gather(x, dim=1)

        action_pred = None
        if self.action_head is not None and return_actions:
            action_pred = self._predict_actions(x, grid_sizes)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        video = torch.stack(x)
        if action_pred is not None:
            return video, action_pred
        return video

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
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

        # Construct blockwise causal attention mask.  LingBot-style MoBA mixes
        # the causal teacher-forcing view with a full bidirectional view while
        # sharing every QKV/operator parameter.  We sample one view per
        # micro-batch to keep activation memory unchanged on 48-GB cards; over
        # an optimizer accumulation window both masks contribute gradients.
        use_moba_bidir = False
        if clean_x is not None and self.training and self.moba_bidirectional_ratio > 0:
            draw = torch.rand((), device=device)
            if dist.is_initialized():
                dist.broadcast(draw, src=0)
            use_moba_bidir = bool(draw.item() < self.moba_bidirectional_ratio)

        if clean_x is not None and self.moba_bidirectional_ratio > 0:
            frame_seqlen_mask = x.shape[-2] * x.shape[-1] // (
                self.patch_size[1] * self.patch_size[2])
            key = ("bidir" if use_moba_bidir else "causal", x.shape[2],
                   frame_seqlen_mask, self.num_frame_per_block,
                   self.independent_first_frame, self.local_attn_size,
                   self.sink_size)
            if key not in self.moba_block_masks:
                if use_moba_bidir:
                    self.moba_block_masks[key] = self._prepare_bidirectional_teacher_forcing_mask(
                        device, num_frames=x.shape[2], frame_seqlen=frame_seqlen_mask)
                else:
                    self.moba_block_masks[key] = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2], frame_seqlen=frame_seqlen_mask,
                        num_frame_per_block=self.num_frame_per_block,
                        independent_first_frame=self.independent_first_frame,
                        local_attn_size=self.local_attn_size,
                        sink_size=self.sink_size)
            self.block_mask = self.moba_block_masks[key]
            self.last_moba_mask = key[0]
        elif self.block_mask is None:
            if clean_x is not None: # TF
                self.block_mask = self._prepare_teacher_forcing_mask(
                    device, num_frames=x.shape[2],
                    frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                    num_frame_per_block=self.num_frame_per_block,
                    independent_first_frame=self.independent_first_frame,
                    local_attn_size=self.local_attn_size,
                    sink_size=self.sink_size,
                )
            else: # DF?
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

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
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # Action conditioning (wan/modules/action.py). e0 is already per-frame
        # here, so this is a straight additive term on the same [B, F, 6, C].
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

        # Independent event prompt: project the separately-tokenised event text
        # into transformer width. Padded to the same fixed text length as the
        # main caption so the cross-attention shape is static.
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

        if clean_x is not None:
            # clean_x.detach()
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        camera_film_tokens = None
        if viewmats is not None and getattr(self, "camera_film_enabled", False):
            from models.camera_film import camera_ray_tokens
            camera_film_tokens = camera_ray_tokens(
                viewmats, Ks, grid_sizes, dtype=x.dtype)
            if clean_x is not None:
                camera_film_tokens = torch.cat(
                    [camera_film_tokens, camera_film_tokens], dim=1)

        # PRoPE: expand viewmats/Ks from (B, F, *, *) to (B, seq_len, *, *)
        # Must happen before SP chunking so token counts align.
        if viewmats is not None:
            expanded_vm, expanded_ks = [], []
            single_seq_len = seq_lens[0].item()  # tokens per video (noisy half)
            for i, (f, h, w) in enumerate(grid_sizes.tolist()):
                vm = viewmats[i, :f]  # (F, 4, 4)
                vm = vm[:, None, None].expand(-1, h, w, -1, -1).reshape(f * h * w, 4, 4)
                pad_len = single_seq_len - f * h * w
                if pad_len > 0:
                    vm = torch.cat([vm, torch.eye(4, device=vm.device, dtype=vm.dtype).unsqueeze(0).expand(pad_len, -1, -1)])
                expanded_vm.append(vm)

                ks = Ks[i, :f]  # (F, 3, 3)
                ks = ks[:, None, None].expand(-1, h, w, -1, -1).reshape(f * h * w, 3, 3)
                if pad_len > 0:
                    ks = torch.cat([ks, torch.eye(3, device=ks.device, dtype=ks.dtype).unsqueeze(0).expand(pad_len, -1, -1)])
                expanded_ks.append(ks)

            viewmats = torch.stack(expanded_vm)  # (B, single_seq_len, 4, 4)
            Ks = torch.stack(expanded_ks)        # (B, single_seq_len, 3, 3)

            # TF mode: clean and noisy share the same camera trajectory
            if clean_x is not None:
                viewmats = torch.cat([viewmats, viewmats], dim=1)  # (B, 2*single_seq_len, 4, 4)
                Ks = torch.cat([Ks, Ks], dim=1)                    # (B, 2*single_seq_len, 3, 3)

        # SP: token-level chunk (like HunyuanVideo)
        if _HAS_CLEANCODE_INFRA:
            parallel_dims = get_parallel_state()
            sp_enabled = parallel_dims.sp_enabled
        else:
            _require_cleancode_infra("CausalWanModel._forward_train SP check")
            sp_enabled = False
        sp_pad_len = 0
        sp_seq_len_orig = x.shape[1]  # before any SP padding (includes clean+noisy if TF)
        if sp_enabled:
            sp_size = parallel_dims.sp
            sp_rank = parallel_dims.sp_rank
            num_frames_total = e0.shape[1]
            frame_seqlen = x.shape[1] // num_frames_total
            # Expand e0 from frame-level [B, F, 6, C] to token-level [B, L, 6, C]
            e0 = e0.unsqueeze(2).expand(-1, -1, frame_seqlen, -1, -1).flatten(1, 2)

            if clean_x is not None:
                # TF mode: chunk clean and noisy halves separately (like HunyuanVideo),
                # so each rank's local x stays [clean_chunk, noisy_chunk].
                half = sp_seq_len_orig // 2
                x_clean_half = x[:, :half]
                x_noisy_half = x[:, half:]
                e0_clean_half = e0[:, :half]
                e0_noisy_half = e0[:, half:]
                # Pad each half to sp_size multiple
                sp_pad_len = (sp_size - half % sp_size) % sp_size
                if sp_pad_len > 0:
                    x_clean_half = F.pad(x_clean_half, (0, 0, 0, sp_pad_len))
                    x_noisy_half = F.pad(x_noisy_half, (0, 0, 0, sp_pad_len))
                    e0_clean_half = F.pad(e0_clean_half, (0, 0, 0, 0, 0, sp_pad_len))
                    e0_noisy_half = F.pad(e0_noisy_half, (0, 0, 0, 0, 0, sp_pad_len))
                    if viewmats is not None:
                        vm_clean = F.pad(viewmats[:, :half], (0, 0, 0, 0, 0, sp_pad_len))
                        vm_noisy = F.pad(viewmats[:, half:], (0, 0, 0, 0, 0, sp_pad_len))
                        ks_clean = F.pad(Ks[:, :half], (0, 0, 0, 0, 0, sp_pad_len))
                        ks_noisy = F.pad(Ks[:, half:], (0, 0, 0, 0, 0, sp_pad_len))
                        viewmats = torch.cat([vm_clean, vm_noisy], dim=1)
                        Ks = torch.cat([ks_clean, ks_noisy], dim=1)
                    if camera_film_tokens is not None:
                        cf_clean = F.pad(
                            camera_film_tokens[:, :half], (0, 0, 0, sp_pad_len))
                        cf_noisy = F.pad(
                            camera_film_tokens[:, half:], (0, 0, 0, sp_pad_len))
                        camera_film_tokens = torch.cat([cf_clean, cf_noisy], dim=1)
                # Chunk each half
                x_clean_half = torch.chunk(x_clean_half, sp_size, dim=1)[sp_rank]
                x_noisy_half = torch.chunk(x_noisy_half, sp_size, dim=1)[sp_rank]
                e0_clean_half = torch.chunk(e0_clean_half, sp_size, dim=1)[sp_rank]
                e0_noisy_half = torch.chunk(e0_noisy_half, sp_size, dim=1)[sp_rank]
                if viewmats is not None:
                    vm_clean_chunk = torch.chunk(viewmats[:, :half + sp_pad_len], sp_size, dim=1)[sp_rank]
                    vm_noisy_chunk = torch.chunk(viewmats[:, half + sp_pad_len:], sp_size, dim=1)[sp_rank]
                    ks_clean_chunk = torch.chunk(Ks[:, :half + sp_pad_len], sp_size, dim=1)[sp_rank]
                    ks_noisy_chunk = torch.chunk(Ks[:, half + sp_pad_len:], sp_size, dim=1)[sp_rank]
                    viewmats = torch.cat([vm_clean_chunk, vm_noisy_chunk], dim=1)
                    Ks = torch.cat([ks_clean_chunk, ks_noisy_chunk], dim=1)
                if camera_film_tokens is not None:
                    cf_half = half + sp_pad_len
                    cf_clean_chunk = torch.chunk(
                        camera_film_tokens[:, :cf_half], sp_size, dim=1)[sp_rank]
                    cf_noisy_chunk = torch.chunk(
                        camera_film_tokens[:, cf_half:], sp_size, dim=1)[sp_rank]
                    camera_film_tokens = torch.cat(
                        [cf_clean_chunk, cf_noisy_chunk], dim=1)
                # Reassemble [clean_chunk, noisy_chunk]
                x = torch.cat([x_clean_half, x_noisy_half], dim=1)
                e0 = torch.cat([e0_clean_half, e0_noisy_half], dim=1)
            else:
                # DF mode: single chunk
                sp_pad_len = (sp_size - x.shape[1] % sp_size) % sp_size
                if sp_pad_len > 0:
                    x = F.pad(x, (0, 0, 0, sp_pad_len))
                    e0 = F.pad(e0, (0, 0, 0, 0, 0, sp_pad_len))
                x = torch.chunk(x, sp_size, dim=1)[sp_rank]
                e0 = torch.chunk(e0, sp_size, dim=1)[sp_rank]
                if camera_film_tokens is not None:
                    camera_film_tokens = torch.chunk(
                        camera_film_tokens, sp_size, dim=1)[sp_rank]

        # arguments
        block_mask = self.block_mask
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=block_mask)
        if viewmats is not None:
            kwargs['viewmats'] = viewmats
            kwargs['Ks'] = Ks
        if camera_film_tokens is not None:
            kwargs['camera_film_tokens'] = camera_film_tokens
        if event_ctx is not None:
            kwargs['event_context'] = event_ctx
            kwargs['event_context_lens'] = None

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]
            # [1,31200,1536]

        # SP: gather sequence from all ranks and remove padding
        if sp_enabled:
            x = sequence_model_parallel_all_gather(x, dim=1)
            # Determine unpadded target length
            # TF mode: clean half was discarded, target = original_seq_len / 2
            # DF mode: target = original_seq_len
            sp_target_len = sp_seq_len_orig // 2 if clean_x is not None else sp_seq_len_orig
            if x.shape[1] > sp_target_len:
                x = x[:, :sp_target_len]

        # Optional action output: pool the final hidden tokens per latent frame
        # before the denoising head consumes them.
        action_pred = None
        if self.action_head is not None and return_actions:
            action_pred = self._predict_actions(x, grid_sizes)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        video = torch.stack(x)
        if action_pred is not None:
            return video, action_pred
        return video

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            # TF or DF
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
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

        c = self.out_dim
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
