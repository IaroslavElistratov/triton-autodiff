# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/fla-org/flash-linear-attention
# Source-Files: fla/ops/simple_gla/chunk.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rh2l8zei/flash-linear-attention-main/fla/ops/simple_gla/chunk.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

# Common helper imports
from torch.cuda.amp import custom_bwd
from torch.cuda.amp import custom_fwd

def chunk_fwd_h(k: torch.Tensor, v: torch.Tensor, g: Optional[torch.Tensor]
    =None, g_gamma: Optional[torch.Tensor]=None, gk: Optional[torch.Tensor]
    =None, gv: Optional[torch.Tensor]=None, h0: Optional[torch.Tensor]=None,
    output_final_state: bool=False, cu_seqlens: Optional[torch.Tensor]=None,
    chunk_size: int=64, split_size: Optional[int]=None, states_in_fp32:
    bool=False) ->Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BS = BT if split_size is None else min(split_size, max(16, triton.
        next_power_of_2(T)))
    assert BS % BT == 0, f'The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}'
    if cu_seqlens is None:
        N, NS, split_offsets = B, triton.cdiv(T, BS), None
    else:
        split_offsets = prepare_chunk_offsets(cu_seqlens, BS)
        N, NS = len(cu_seqlens) - 1, split_offsets[-1].item()
    h = k.new_empty(B, NS, H, K, V, dtype=k.dtype if not states_in_fp32 else
        torch.float)
    ht = k.new_empty(N, H, K, V, dtype=torch.float
        ) if output_final_state else None

    def grid(meta):
        return triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * H
    chunk_fwd_kernel_h[grid](k=k, v=v, h=h, g=g, g_gamma=g_gamma, gk=gk, gv
        =gv, h0=h0, ht=ht, cu_seqlens=cu_seqlens, split_offsets=
        split_offsets, T=T, H=H, K=K, V=V, BT=BT, BS=BS, USE_G=g is not
        None, USE_G_GAMMA=g_gamma is not None, USE_GK=gk is not None,
        USE_GV=gv is not None)
    return h, ht


@triton.heuristics({'USE_INITIAL_STATE': lambda args: args['h0'] is not
    None, 'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BK': BK, 'BV': BV}, num_warps=
    num_warps, num_stages=num_stages) for BK in BKV_LIST for BV in BKV_LIST for
    num_warps in [1, 2, 4, 8] for num_stages in [2, 3, 4]], key=['BT',
    'USE_G', 'USE_GK', 'USE_GV'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_h(k, v, h, g, g_gamma, gk, gv, h0, ht, cu_seqlens,
    split_offsets, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT:
    tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_G: tl.constexpr, USE_G_GAMMA: tl.constexpr, USE_GK: tl.constexpr,
    USE_GV: tl.constexpr, USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT, NS = tl.cdiv(T, BT), tl.cdiv(T, BS)
        boh = tl.load(split_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT, NS = tl.cdiv(T, BT), tl.cdiv(T, BS)
        boh = i_n * NS
    NTS = BS // BT
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_nh * K * V, (K, V), (V, 1), (i_k *
            BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
    for i_t in range(NT):
        i_s = i_t // NTS
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H * K),
            (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1),
            (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        o_h = ((boh + i_s) * H + i_h).to(tl.int64) * K * V
        p_h = tl.make_block_ptr(h + o_h, (K, V), (V, 1), (i_k * BK, i_v *
            BV), (BK, BV), (1, 0))
        if i_t % NTS == 0:
            tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = g + bos * H + (i_t * BT + tl.arange(0, BT)) * H + i_h
            b_g = tl.load(p_g, mask=i_t * BT + tl.arange(0, BT) < T, other=0.0)
            b_h *= exp(b_g_last)
            b_v = (b_v * exp(b_g_last - b_g)[:, None]).to(b_v.dtype)
        if USE_G_GAMMA:
            b_g_last = b_gamma * min(BT, T - i_t * BT)
            b_h *= exp(b_g_last)
            b_v = (b_v * exp(b_g_last - b_g)[:, None]).to(b_v.dtype)
        if USE_GK:
            p_gk = tl.make_block_ptr(gk + (bos * H + i_h) * K, (K, T), (1, 
                H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_gk_last = gk + (bos + last_idx
                ) * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
            b_gk_last = tl.load(p_gk_last, mask=i_k * BK + tl.arange(0, BK) <
                K, other=0.0)
            b_h *= exp(b_gk_last)[:, None]
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_k = (b_k * exp(b_gk_last[:, None] - b_gk)).to(b_k.dtype)
        if USE_GV:
            p_gv = tl.make_block_ptr(gv + (bos * H + i_h) * V, (T, V), (H *
                V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_gv_last = gv + (bos + last_idx
                ) * H * V + i_h * V + i_v * BV + tl.arange(0, BV)
            b_gv_last = tl.load(p_gv_last, mask=i_v * BV + tl.arange(0, BV) <
                V, other=0.0)
            b_h *= exp(b_gv_last)[None, :]
            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            b_v = (b_v * exp(b_gv_last[None, :] - b_gv)).to(b_v.dtype)
        b_h += tl.dot(b_k, b_v)
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_nh * K * V, (K, V), (V, 1), (i_k *
            BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@input_guard
def chunk_local_cumsum(g: torch.Tensor, chunk_size: int, reverse: bool=
    False, scale: float=None, cu_seqlens: Optional[torch.Tensor]=None,
    head_first: bool=False, output_dtype: Optional[torch.dtype]=torch.float,
    **kwargs) ->torch.Tensor:
    if cu_seqlens is not None:
        assert g.shape[0
            ] == 1, 'Only batch size 1 is supported when cu_seqlens are provided'
    if len(g.shape) == 3:
        return chunk_local_cumsum_scalar(g=g, chunk_size=chunk_size,
            reverse=reverse, scale=scale, cu_seqlens=cu_seqlens, head_first
            =head_first, output_dtype=output_dtype)
    elif len(g.shape) == 4:
        return chunk_local_cumsum_vector(g=g, chunk_size=chunk_size,
            reverse=reverse, scale=scale, cu_seqlens=cu_seqlens, head_first
            =head_first, output_dtype=output_dtype)
    else:
        raise ValueError(
            f'Unsupported input shape {g.shape}, which should be (B, T, H, D) if `head_first=False` or (B, H, T, D) otherwise'
            )


def chunk_local_cumsum_scalar(g: torch.Tensor, chunk_size: int, reverse:
    bool=False, scale: float=None, cu_seqlens: Optional[torch.Tensor]=None,
    head_first: bool=False, output_dtype: Optional[torch.dtype]=torch.float
    ) ->torch.Tensor:
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1
        ), 'chunk_size must be a power of 2'
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    grid = NT, B * H
    chunk_local_cumsum_scalar_kernel[grid](s=g_org, o=g, scale=scale,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T, B=B, H=H,
        BT=BT, HEAD_FIRST=head_first, REVERSE=reverse)
    return g


@triton.heuristics({'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for
    num_warps in [1, 2, 4, 8]], key=['B', 'H', 'BT', 'IS_VARLEN', 'REVERSE'
    ], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_local_cumsum_scalar_kernel(s, o, scale, cu_seqlens, chunk_indices,
    T, B: tl.constexpr, H: tl.constexpr, BT: tl.constexpr, REVERSE: tl.
    constexpr, HAS_SCALE: tl.constexpr, IS_VARLEN: tl.constexpr, HEAD_FIRST:
    tl.constexpr):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + bos * H + i_h * T, (T,), (1,), (i_t *
            BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos * H + i_h * T, (T,), (1,), (i_t *
            BT,), (BT,), (0,))
    else:
        p_s = tl.make_block_ptr(s + bos * H + i_h, (T,), (H,), (i_t * BT,),
            (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t * BT,),
            (BT,), (0,))
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


def chunk_local_cumsum_vector(g: torch.Tensor, chunk_size: int, reverse:
    bool=False, scale: float=None, cu_seqlens: Optional[torch.Tensor]=None,
    head_first: bool=False, output_dtype: Optional[torch.dtype]=torch.float
    ) ->torch.Tensor:
    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1
        ), 'chunk_size must be a power of 2'
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)

    def grid(meta):
        return triton.cdiv(meta['S'], meta['BS']), NT, B * H
    chunk_local_cumsum_vector_kernel[grid](s=g_org, o=g, scale=scale,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T, B=B, H=H,
        S=S, BT=BT, HEAD_FIRST=head_first, REVERSE=reverse)
    return g


@triton.heuristics({'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BS': BS}, num_warps=num_warps) for
    BS in BS_LIST for num_warps in [2, 4, 8]], key=['B', 'H', 'S', 'BT',
    'IS_VARLEN', 'REVERSE'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_local_cumsum_vector_kernel(s, o, scale, cu_seqlens, chunk_indices,
    T, B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, BT: tl.constexpr,
    BS: tl.constexpr, REVERSE: tl.constexpr, HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr, HEAD_FIRST: tl.constexpr):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_i = tl.arange(0, BT)
    if REVERSE:
        m_s = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
    else:
        m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + (bos * H + i_h * T) * S, (T, S), (S, 1),
            (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + (bos * H + i_h * T) * S, (T, S), (S, 1),
            (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    else:
        p_s = tl.make_block_ptr(s + (bos * H + i_h) * S, (T, S), (H * S, 1),
            (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + (bos * H + i_h) * S, (T, S), (H * S, 1),
            (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int
    ) ->torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(
        cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens
        )


@tensor_cache
def prepare_chunk_offsets(cu_seqlens: torch.LongTensor, chunk_size: int
    ) ->torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(
        cu_seqlens), chunk_size)]).cumsum(-1)


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) ->torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.heuristics({'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None, 'IS_VARLEN': 
    lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BK': 128, 'BV': 128}, num_warps=8,
    num_stages=3), triton.Config({'BK': 64, 'BV': 64}, num_warps=4,
    num_stages=3), triton.Config({'BK': 32, 'BV': 32}, num_warps=2,
    num_stages=3)], key=['H', 'K', 'V', 'BT'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_o(q, k, v, h, g, g_gamma, o, cu_seqlens, chunk_indices,
    scale, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.
    constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK),
            (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (i_k * BK, i_t * BT),
            (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (
            BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_h)
        b_A += tl.dot(b_q, b_k)
    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_o = b_o * exp(b_g)[:, None]
        b_A = b_A * exp(b_g[:, None] - b_g[None, :])
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_o = b_o * exp(b_g)[:, None]
        b_A = b_A * exp(b_g[:, None] - b_g[None, :])
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (
        BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (
        BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_o(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, h: torch
    .Tensor, g: Optional[torch.Tensor]=None, g_gamma: Optional[torch.Tensor
    ]=None, scale: Optional[float]=None, cu_seqlens: Optional[torch.
    LongTensor]=None, chunk_size: int=64) ->torch.Tensor:
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o = torch.empty_like(v)

    def grid(meta):
        return triton.cdiv(V, meta['BV']), NT, B * H
    chunk_fwd_kernel_o[grid](q=q, k=k, v=v, h=h, g=g, g_gamma=g_gamma, o=o,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, scale=scale, T=
        T, H=H, K=K, V=V, BT=BT)
    return o


def chunk_simple_gla_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    g: Optional[torch.Tensor]=None, g_gamma: Optional[torch.Tensor]=None,
    scale: Optional[float]=None, initial_state: Optional[torch.Tensor]=None,
    output_final_state: bool=False, cu_seqlens: Optional[torch.LongTensor]=
    None, chunk_size: int=64) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor
    ]:
    h, ht = chunk_fwd_h(k=k, v=v, g=g, g_gamma=g_gamma, gk=None, gv=None,
        h0=initial_state, output_final_state=output_final_state,
        states_in_fp32=False, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    o = chunk_fwd_o(q=q, k=k, v=v, g=g, g_gamma=g_gamma, h=h, scale=scale,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    return o, ht


# Forward method (kernel launch code)
@input_guard
@autocast_custom_fwd
def _ChunkSimpleGLAFunction_forward(ctx, q, k, v, g, g_gamma, scale,
    initial_state, output_final_state, cu_seqlens):
    T = q.shape[1]
    chunk_size = min(64, max(16, triton.next_power_of_2(T)))
    g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens
        ) if g is not None else None
    o, ht = chunk_simple_gla_fwd(q=q, k=k, v=v, g=g, g_gamma=g_gamma, scale
        =scale, initial_state=initial_state, output_final_state=
        output_final_state, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    ctx.save_for_backward(q, k, v, g, g_gamma, initial_state)
    ctx.chunk_size = chunk_size
    ctx.scale = scale
    ctx.cu_seqlens = cu_seqlens
    return o.to(q.dtype), ht


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.heuristics({'STORE_INITIAL_STATE_GRADIENT': lambda args: args['dh0'
    ] is not None, 'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not
    None, 'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BK': BK, 'BV': BV}, num_warps=
    num_warps, num_stages=num_stages) for BK in BKV_LIST for BV in BKV_LIST for
    num_warps in [1, 2, 4, 8] for num_stages in [2, 3, 4]], key=['BT',
    'USE_G', 'USE_GK', 'USE_GV'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dh(q, g, g_gamma, gk, gv, do, dh, dht, dh0, cu_seqlens,
    split_offsets, scale, T, HQ: tl.constexpr, H: tl.constexpr, K: tl.
    constexpr, V: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr, BK: tl.
    constexpr, BV: tl.constexpr, NG: tl.constexpr, USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr, USE_GK: tl.constexpr, USE_GV: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr, USE_FINAL_STATE_GRADIENT:
    tl.constexpr, IS_VARLEN: tl.constexpr):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // NG
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        NS = tl.cdiv(T, BS)
        boh = tl.load(split_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        NS = tl.cdiv(T, BS)
        boh = i_n * NS
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = tl.make_block_ptr(dht + i_nh * K * V, (K, V), (V, 1), (i_k *
            BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dht, boundary_check=(0, 1)).to(tl.float32)
    for i_t in range(NT - 1, -1, -1):
        i_s = i_t // (BS // BT)
        o_dh = ((boh + i_s) * H + i_h).to(tl.int64) * K * V
        p_dh = tl.make_block_ptr(dh + o_dh, (K, V), (V, 1), (i_k * BK, i_v *
            BV), (BK, BV), (1, 0))
        if i_t % (BS // BT) == 0:
            tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(
                0, 1))
        last_idx = min(i_t * BT + BT, T) - 1
        p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (K, T), (1, HQ *
            K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ *
            V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        b_do = tl.load(p_do, boundary_check=(0, 1))
        if USE_G:
            p_g = g + (bos + i_t * BT + tl.arange(0, BT)) * H + i_h
            b_g_last = tl.load(g + (bos + last_idx) * H + i_h)
            b_g = tl.load(p_g, mask=i_t * BT + tl.arange(0, BT) < T, other=0.0)
            b_q = (b_q * exp(b_g)[None, :]).to(b_q.dtype)
            b_dh *= exp(b_g_last)
        if USE_G_GAMMA:
            b_g_last = b_gamma * min(BT, T - i_t * BT)
            b_q = (b_q * exp(b_g)[None, :]).to(b_q.dtype)
            b_dh *= exp(b_g_last)
        if USE_GK:
            p_gk = tl.make_block_ptr(gk + (bos * H + i_h) * K, (K, T), (1, 
                H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_gk_last = gk + (bos + last_idx
                ) * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_q = (b_q * exp(b_gk)).to(b_q.dtype)
            b_gk_last = tl.load(p_gk_last, mask=i_k * BK + tl.arange(0, BK) <
                K, other=0.0)
            b_dh *= exp(b_gk_last)[:, None]
        if USE_GV:
            p_gv = tl.make_block_ptr(gv + (bos * H + i_h) * V, (T, V), (H *
                V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_gv_last = gv + (bos + last_idx
                ) * H * V + i_h * V + i_v * BV + tl.arange(0, BV)
            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            b_do = b_do * exp(b_gv)
            b_gv_last = tl.load(p_gv_last, mask=i_v * BV + tl.arange(0, BV) <
                V, other=0.0)
            b_dh *= exp(b_gv_last)[None, :]
        b_dh += tl.dot(b_q, b_do.to(b_q.dtype))
    if STORE_INITIAL_STATE_GRADIENT:
        p_dh0 = tl.make_block_ptr(dh0 + i_nh * K * V, (K, V), (V, 1), (i_k *
            BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None, 'USE_DW': lambda
    args: args['dw'] is not None, 'IS_VARLEN': lambda args: args[
    'cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages
    =num_stages) for num_warps in NUM_WARPS for num_stages in [2, 3, 4]],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'USE_G', 'USE_G_GAMMA', 'USE_DW'],
    **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dqkwg(q, k, v, h, g, g_gamma, do, dh, dq, dk, dg, w,
    dv, dw, cu_seqlens, chunk_indices, scale, B: tl.constexpr, T, H: tl.
    constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.
    constexpr, BV: tl.constexpr, USE_G: tl.constexpr, USE_G_GAMMA: tl.
    constexpr, USE_DW: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        all = T
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T
        all = B * T
    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V
    dh += (i_tg * H + i_h).to(tl.int64) * K * V
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    if USE_DW:
        w += (bos * H + i_h) * K
        dw += (bos * H + i_h) * K
        dv += (bos * H + i_h) * V
    if USE_G:
        dg += i_k * all * H
        b_dg_last = tl.zeros([1], dtype=tl.float32) if USE_G else None
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_g_last = b_gamma * min(BT, T - i_t * BT)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32) if USE_DW else None
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV),
            (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_t * BT, i_v *
            BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (
            BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK),
            (BV, BK), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        if USE_G:
            b_dg_last += tl.sum(b_h * b_dh)
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))
        if USE_DW:
            p_dv = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t * BT, i_v *
                BV), (BT, BV), (1, 0))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_dw += tl.dot(b_dv.to(b_v.dtype), b_h.to(b_v.dtype))
    if USE_DW:
        p_dw = tl.make_block_ptr(dw, (T, K), (H * K, 1), (i_t * BT, i_k *
            BK), (BT, BK), (1, 0))
        tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), boundary_check=(0, 1))
    tl.debug_barrier()
    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (
        BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (
        BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    p_dq = tl.make_block_ptr(dq, (T, K), (H * K, 1), (i_t * BT, i_k * BK),
        (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk, (T, K), (H * K, 1), (i_t * BT, i_k * BK),
        (BT, BK), (1, 0))
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    if USE_G:
        b_dg = tl.zeros([BT], dtype=tl.float32)
        g += bos * H + i_h
        dg += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * H)
        b_dg_last *= exp(b_g_last)
        b_dq = b_dq * exp(b_g)[:, None] * scale
        b_dg += tl.sum(b_dq * b_q, axis=1)
        b_dk = b_dk * tl.where(m_t, exp(-b_g + b_g_last), 0)[:, None]
        b_dg -= tl.sum(b_k * b_dk, axis=1)
        b_dg_last += tl.sum(b_dk * b_k)
        b_ds = tl.where(m_A, b_ds * exp(b_g[:, None] - b_g[None, :]), 0
            ) * scale
        b_ds2 = b_ds * tl.dot(b_q, tl.trans(b_k))
        b_dg += tl.sum(b_ds2, axis=1)
        b_dg -= tl.sum(b_ds2, axis=0)
        b_ds = b_ds.to(b_k.dtype)
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q)
        p_dg = tl.make_block_ptr(dg, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_dg = tl.where(o_t < min(i_t * BT + BT, T) - 1, b_dg, b_dg + b_dg_last
            )
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))
    elif USE_G_GAMMA:
        b_dq = b_dq * exp(b_g)[:, None] * scale
        b_dk = b_dk * tl.where(m_t, exp(-b_g + b_g_last), 0)[:, None]
        b_ds = tl.where(m_A, b_ds * exp(b_g[:, None] - b_g[None, :]), 0
            ) * scale
        b_ds = b_ds.to(b_k.dtype)
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q)
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    else:
        b_ds = tl.where(m_A, b_ds, 0)
        b_ds = b_ds.to(b_k.dtype)
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q) * scale
        b_dq *= scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None, 'IS_VARLEN': 
    lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages
    =num_stages) for num_warps in NUM_WARPS for num_stages in [2, 3, 4]],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'USE_G', 'USE_G_GAMMA'], **
    autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dv(q, k, g, g_gamma, do, dv, dh, cu_seqlens,
    chunk_indices, scale, T, H: tl.constexpr, K: tl.constexpr, V: tl.
    constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_G:
    tl.constexpr, USE_G_GAMMA: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    do += (bos * H + i_h) * V
    dv += (bos * H + i_h) * V
    dh += (i_tg * H + i_h).to(tl.int64) * K * V
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, i_k * BK),
            (BT, BK), (1, 0))
        p_q = tl.make_block_ptr(q, (K, T), (1, H * K), (i_k * BK, i_t * BT),
            (BK, BT), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, b_q)
        p_dh = tl.make_block_ptr(dh, (K, V), (V, 1), (i_k * BK, i_v * BV),
            (BK, BV), (1, 0))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dv += tl.dot(b_k, b_dh.to(b_k.dtype))
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * H)
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_g_last = b_gamma * min(BT, T - i_t * BT)
    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
    if USE_G or USE_G_GAMMA:
        b_A = tl.where(m_A, b_A * exp(b_g[None, :] - b_g[:, None]) * scale, 0
            ).to(do.dtype.element_ty)
        b_dv *= tl.where(m_t, exp(-b_g + b_g_last), 0)[:, None]
    else:
        b_A = tl.where(m_A, b_A * scale, 0).to(do.dtype.element_ty)
    p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_t * BT, i_v * BV),
        (BT, BV), (1, 0))
    p_dv = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t * BT, i_v * BV),
        (BT, BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dv += tl.dot(b_A.to(b_do.dtype), b_do)
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@classmethod
def get_shared_memory(cls, arch: str) ->int:
    try:
        return cls[arch.upper()].value
    except KeyError:
        return cls.DEFAULT.value


def _cpu_device_warning():
    warnings.warn(
        'Triton is not supported on current platform, roll back to CPU.',
        stacklevel=1)


@lru_cache(maxsize=None)
def check_shared_mem(arch: str='none', tensor_idx: int=0) ->bool:
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False


def get_all_max_shared_mem():
    try:
        return [triton.runtime.driver.active.utils.get_device_properties(i)
            ['max_shared_mem'] for i in range(device_torch_lib.device_count())]
    except BaseException:
        _cpu_device_warning()
        return [-1]


def chunk_bwd_dh(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, do:
    torch.Tensor, h0: torch.Tensor, dht: torch.Tensor, scale: float, g:
    Optional[torch.Tensor]=None, g_gamma: Optional[torch.Tensor]=None, gk:
    Optional[torch.Tensor]=None, gv: Optional[torch.Tensor]=None,
    cu_seqlens: Optional[torch.Tensor]=None, chunk_size: int=64, split_size:
    Optional[int]=None, states_in_fp32: bool=False) ->Tuple[torch.Tensor,
    torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BS = BT if split_size is None else min(split_size, max(16, triton.
        next_power_of_2(T)))
    assert BS % BT == 0, f'The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}'
    if cu_seqlens is None:
        N, NS, split_offsets = B, triton.cdiv(T, BS), None
    else:
        split_offsets = prepare_chunk_offsets(cu_seqlens, BS)
        N, NS = len(cu_seqlens) - 1, split_offsets[-1].item()
    NG = HQ // H
    dh = k.new_empty(B, NS, HQ, K, V, dtype=k.dtype if not states_in_fp32 else
        torch.float)
    dh0 = torch.empty_like(h0, dtype=torch.float) if h0 is not None else None

    def grid(meta):
        return triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * H
    chunk_bwd_kernel_dh[grid](q=q, g=g, g_gamma=g_gamma, gk=gk, gv=gv, do=
        do, dh=dh, dht=dht, dh0=dh0, cu_seqlens=cu_seqlens, split_offsets=
        split_offsets, scale=scale, T=T, HQ=HQ, H=H, K=K, V=V, BT=BT, BS=BS,
        NG=NG, USE_G=g is not None, USE_G_GAMMA=g_gamma is not None, USE_GK
        =gk is not None, USE_GV=gv is not None)
    return dh, dh0


def chunk_bwd_dqkwg(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, do:
    torch.Tensor, h: torch.Tensor, dh: torch.Tensor, g: Optional[torch.
    Tensor]=None, g_gamma: Optional[torch.Tensor]=None, dv: Optional[torch.
    Tensor]=None, w: Optional[torch.Tensor]=None, cu_seqlens: Optional[
    torch.LongTensor]=None, chunk_size: int=64, scale: float=1.0) ->Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NK = triton.cdiv(K, BK)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dg = torch.empty(NK, *g.shape, dtype=torch.float32, device=g.device
        ) if g is not None else None
    dw = torch.empty_like(w) if w is not None else None
    grid = NK, NT, B * H
    chunk_bwd_kernel_dqkwg[grid](q=q, k=k, v=v, h=h, g=g, g_gamma=g_gamma,
        do=do, dh=dh, dv=dv, w=w, dw=dw, dq=dq, dk=dk, dg=dg, cu_seqlens=
        cu_seqlens, chunk_indices=chunk_indices, scale=scale, B=B, T=T, H=H,
        K=K, V=V, BT=BT, BK=BK, BV=BV)
    if dg is not None:
        dg = dg.sum(0)
    return dq, dk, dw, dg


def chunk_bwd_dv(q: torch.Tensor, k: torch.Tensor, do: torch.Tensor, dh:
    torch.Tensor, g: Optional[torch.Tensor]=None, g_gamma: Optional[torch.
    Tensor]=None, scale: Optional[float]=None, cu_seqlens: Optional[torch.
    LongTensor]=None, chunk_size: int=64) ->torch.Tensor:
    B, T, H, K, V = *k.shape, do.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    if check_shared_mem('hopper', k.device.index):
        CONST_TILING = 128
    elif check_shared_mem:
        CONST_TILING = 64
    else:
        CONST_TILING = 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NV = triton.cdiv(V, BV)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    dv = torch.empty_like(do)
    grid = NV, NT, B * H
    chunk_bwd_kernel_dv[grid](q=q, k=k, g=g, g_gamma=g_gamma, do=do, dv=dv,
        dh=dh, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, scale=
        scale, T=T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV)
    return dv


def chunk_simple_gla_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    g: torch.Tensor, g_gamma: torch.Tensor, initial_state: torch.Tensor, do:
    torch.Tensor, dht: torch.Tensor, scale: float, cu_seqlens: Optional[
    torch.LongTensor]=None, chunk_size: int=64) ->Tuple[torch.Tensor, torch
    .Tensor, torch.Tensor]:
    h, _ = chunk_fwd_h(k=k, v=v, g=g, g_gamma=g_gamma, gk=None, gv=None, h0
        =initial_state, output_final_state=False, states_in_fp32=True,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    dh, dh0 = chunk_bwd_dh(q=q, k=k, v=v, g=g, g_gamma=g_gamma, gk=None, gv
        =None, do=do, h0=initial_state, dht=dht, scale=scale,
        states_in_fp32=True, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    dq, dk, _, dg = chunk_bwd_dqkwg(q=q, k=k, v=v, g=g, g_gamma=g_gamma, h=
        h, do=do, dh=dh, scale=scale, cu_seqlens=cu_seqlens, chunk_size=
        chunk_size)
    dv = chunk_bwd_dv(q=q, k=k, g=g, g_gamma=g_gamma, do=do, dh=dh, scale=
        scale, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    return dq, dk, dv, dg, dh0


# Backward method (kernel launch code)
@input_guard
@autocast_custom_bwd
def _ChunkSimpleGLAFunction_backward(ctx, do, dht):
    chunk_size, scale, cu_seqlens = ctx.chunk_size, ctx.scale, ctx.cu_seqlens
    q, k, v, g, g_gamma, initial_state = ctx.saved_tensors
    dq, dk, dv, dg, dh0 = chunk_simple_gla_bwd(q=q, k=k, v=v, g=g, g_gamma=
        g_gamma, initial_state=initial_state, do=do, dht=dht, scale=scale,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    if g is not None:
        dg = chunk_local_cumsum(dg, chunk_size=chunk_size, reverse=True,
            cu_seqlens=cu_seqlens).to(g)
    else:
        dg = None
    return dq.to(q), dk.to(k), dv.to(v), dg, None, None, dh0, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class ChunkSimpleGLAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, g, g_gamma, scale, initial_state,
        output_final_state, cu_seqlens):
        T = q.shape[1]
        chunk_size = min(64, max(16, triton.next_power_of_2(T)))
        g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens
            ) if g is not None else None
        o, ht = chunk_simple_gla_fwd(q=q, k=k, v=v, g=g, g_gamma=g_gamma,
            scale=scale, initial_state=initial_state, output_final_state=
            output_final_state, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
        ctx.save_for_backward(q, k, v, g, g_gamma, initial_state)
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        return o.to(q.dtype), ht

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        chunk_size, scale, cu_seqlens = (ctx.chunk_size, ctx.scale, ctx.
            cu_seqlens)
        q, k, v, g, g_gamma, initial_state = ctx.saved_tensors
        dq, dk, dv, dg, dh0 = chunk_simple_gla_bwd(q=q, k=k, v=v, g=g,
            g_gamma=g_gamma, initial_state=initial_state, do=do, dht=dht,
            scale=scale, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
        if g is not None:
            dg = chunk_local_cumsum(dg, chunk_size=chunk_size, reverse=True,
                cu_seqlens=cu_seqlens).to(g)
        else:
            dg = None
        return dq.to(q), dk.to(k), dv.to(v), dg, None, None, dh0, None, None
