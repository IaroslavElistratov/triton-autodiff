# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/fla-org/flash-linear-attention
# Source-Files: fla/ops/gla/chunk.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rh2l8zei/flash-linear-attention-main/fla/ops/gla/chunk.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

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

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BK': BK}, num_warps=num_warps,
    num_stages=num_stages) for BK in [32, 64] for num_warps in [1, 2, 4, 8] for
    num_stages in [2, 3, 4]], key=['BC'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_gla_fwd_A_kernel_intra_sub_inter(q, k, g, A, cu_seqlens,
    chunk_indices, scale, T, H: tl.constexpr, K: tl.constexpr, BT: tl.
    constexpr, BC: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr,
    IS_VARLEN: tl.constexpr):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_i, i_j = i_c // NC, i_c % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H * K, 1),
            (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + (bos * H + i_h) * K, (T, K), (H * K, 1),
            (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H * K),
            (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(g + (bos * H + i_h) * K, (K, T), (1, H * K
            ), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gn = g + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = b_q * exp(b_g - b_gn[None, :]) * scale
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = b_k * exp(b_gn[:, None] - b_gk)
        b_A += tl.dot(b_qg, b_kg)
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1),
        (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8)], key=['BK', 'BT'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_gla_fwd_A_kernel_intra_sub_intra(q, k, g, A, cu_seqlens,
    chunk_indices, scale, T, H: tl.constexpr, K: tl.constexpr, BT: tl.
    constexpr, BC: tl.constexpr, BK: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_j = i_i
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    if i_t * BT + i_i * BC >= T:
        return
    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_A = i_t * BT + i_i * BC + tl.arange(0, BC) < T
    o_A = (bos + i_t * BT + i_i * BC + tl.arange(0, BC)
        ) * H * BT + i_h * BT + i_j * BC
    p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H * K, 1), (
        i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g + (bos * H + i_h) * K, (T, K), (H * K, 1), (
        i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_k = k + (bos + i_t * BT + i_j * BC) * H * K + i_h * K + o_k
    p_gk = g + (bos + i_t * BT + i_j * BC) * H * K + i_h * K + o_k
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A = tl.sum(b_q * b_k[None, :] * exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i >= j, b_A * scale, 0.0)
        tl.store(A + o_A + j, b_A, mask=m_A)
        p_k += H * K
        p_gk += H * K


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8)], key=['BC'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_gla_fwd_A_kernel_intra_sub_intra_merge(A, A2, cu_seqlens,
    chunk_indices, T, B: tl.constexpr, H: tl.constexpr, BT: tl.constexpr,
    BC: tl.constexpr, NK: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        all = T
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
        all = B * T
    if i_t * BT + i_c * BC >= T:
        return
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(0, NK):
        p_A = tl.make_block_ptr(A + (i_k * all + bos) * H * BC + i_h * BC,
            (T, BC), (H * BC, 1), (i_t * BT + i_c * BC, 0), (BC, BC), (1, 0))
        b_A += tl.load(p_A, boundary_check=(0, 1))
    p_A2 = tl.make_block_ptr(A2 + (bos * H + i_h) * BT, (T, BT), (H * BT, 1
        ), (i_t * BT + i_c * BC, i_c * BC), (BC, BC), (1, 0))
    tl.store(p_A2, b_A.to(A2.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8)], key=['BC', 'BK'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_gla_fwd_A_kernel_intra_sub_intra_split(q, k, g, A, cu_seqlens,
    chunk_indices, scale, T, B: tl.constexpr, H: tl.constexpr, K: tl.
    constexpr, BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, NC: tl
    .constexpr, IS_VARLEN: tl.constexpr):
    i_k, i_tc, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_t, i_i = i_tc // NC, i_tc % NC
    i_j = i_i
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        all = T
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
        all = B * T
    if i_t * BT + i_i * BC >= T:
        return
    o_i = tl.arange(0, BC)
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    m_A = i_t * BT + i_i * BC + tl.arange(0, BC) < T
    o_A = (i_k * all + bos + i_t * BT + i_i * BC + tl.arange(0, BC)
        ) * H * BC + i_h * BC
    p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H * K, 1), (
        i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g + (bos * H + i_h) * K, (T, K), (H * K, 1), (
        i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_k = k + (bos + i_t * BT + i_j * BC) * H * K + i_h * K + o_k
    p_gk = g + (bos + i_t * BT + i_j * BC) * H * K + i_h * K + o_k
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_A = tl.zeros([BC], dtype=tl.float32)
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A += tl.sum(b_q * b_k[None, :] * exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i >= j, b_A * scale, 0.0)
        tl.store(A + o_A + j, b_A, mask=m_A)
        p_k += H * K
        p_gk += H * K


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BK': BK, 'BV': BV}, num_warps=
    num_warps) for BK in [32, 64] for BV in [64, 128] for num_warps in [2, 
    4, 8]], key=['BT'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_gla_fwd_kernel_o(q, v, g, h, o, A, cu_seqlens, chunk_indices,
    scale, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.
    constexpr, BK: tl.constexpr, BV: tl.constexpr, IS_VARLEN: tl.constexpr):
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
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H * K, 1),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g + (bos * H + i_h) * K, (T, K), (H * K, 1),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + (i_tg * H + i_h) * K * V, (K, V), (V, 1
            ), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = (b_q * exp(b_g)).to(b_q.dtype)
        b_h = tl.load(p_h, boundary_check=(0, 1))
        if i_k >= 0:
            b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (
        i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (bos * H + i_h) * V, (T, V), (H * V, 1), (
        i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1),
        (i_t * BT, 0), (BT, BT), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_gla_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g:
    torch.Tensor, g_cumsum: Optional[torch.Tensor], scale: float,
    initial_state: torch.Tensor, output_final_state: bool, cu_seqlens:
    Optional[torch.LongTensor]=None, chunk_size: int=64) ->Tuple[torch.
    Tensor, torch.Tensor, torch.Tensor]:
    T = q.shape[1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    if g_cumsum is None:
        g_cumsum = chunk_local_cumsum(g, BT, cu_seqlens=cu_seqlens)
    h, ht = chunk_fwd_h(k=k, v=v, g=None, gk=g_cumsum, gv=None, h0=
        initial_state, output_final_state=output_final_state,
        states_in_fp32=False, cu_seqlens=cu_seqlens, chunk_size=BT)
    A = chunk_gla_fwd_intra_gk(q=q, k=k, g=g_cumsum, scale=scale,
        cu_seqlens=cu_seqlens, chunk_size=BT)
    o = chunk_gla_fwd_o_gk(q=q, v=v, g=g_cumsum, A=A, h=h, scale=scale,
        cu_seqlens=cu_seqlens, chunk_size=BT)
    return g_cumsum, A, h, ht, o


def chunk_gla_fwd_intra_gk(q: torch.Tensor, k: torch.Tensor, g: torch.
    Tensor, scale: float, cu_seqlens: Optional[torch.LongTensor]=None,
    chunk_size: int=64):
    B, T, H, K = k.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BC = min(16, BT)
    NC = triton.cdiv(BT, BC)
    A = q.new_empty(B, T, H, BT, dtype=torch.float)
    grid = NT, NC * NC, B * H
    chunk_gla_fwd_A_kernel_intra_sub_inter[grid](q, k, g, A, cu_seqlens,
        chunk_indices, scale, T=T, H=H, K=K, BT=BT, BC=BC, NC=NC)
    grid = NT, NC, B * H
    if K <= 256:
        BK = max(triton.next_power_of_2(K), 16)
        chunk_gla_fwd_A_kernel_intra_sub_intra[grid](q, k, g, A, cu_seqlens,
            chunk_indices, scale, T=T, H=H, K=K, BT=BT, BC=BC, BK=BK)
    else:
        BK = min(128, triton.next_power_of_2(K))
        NK = triton.cdiv(K, BK)
        A_intra = q.new_empty(NK, B, T, H, BC, dtype=torch.float)
        grid = NK, NT * NC, B * H
        chunk_gla_fwd_A_kernel_intra_sub_intra_split[grid](q, k, g, A_intra,
            cu_seqlens, chunk_indices, scale, T=T, B=B, H=H, K=K, BT=BT, BC
            =BC, BK=BK, NC=NC)
        grid = NT, NC, B * H
        chunk_gla_fwd_A_kernel_intra_sub_intra_merge[grid](A_intra, A,
            cu_seqlens, chunk_indices, T=T, B=B, H=H, BT=BT, BC=BC, NK=NK)
    return A


def chunk_gla_fwd_o_gk(q: torch.Tensor, v: torch.Tensor, g: torch.Tensor, A:
    torch.Tensor, h: torch.Tensor, scale: float, cu_seqlens: Optional[torch
    .LongTensor]=None, chunk_size: int=64):
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    o = torch.empty_like(v)

    def grid(meta):
        return triton.cdiv(V, meta['BV']), NT, B * H
    chunk_gla_fwd_kernel_o[grid](q, v, g, h, o, A, cu_seqlens,
        chunk_indices, scale, T=T, H=H, K=K, V=V, BT=BT)
    return o


# Forward method (kernel launch code)
@input_guard
def _ChunkGLAFunction_forward(ctx, q, k, v, g, scale, initial_state,
    output_final_state, cu_seqlens):
    T = q.shape[1]
    chunk_size = min(64, max(16, triton.next_power_of_2(T)))
    g_cumsum, A, h, ht, o = chunk_gla_fwd(q=q, k=k, v=v, g=g, g_cumsum=None,
        scale=scale, initial_state=initial_state, output_final_state=
        output_final_state, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    if g.dtype != torch.float:
        g_cumsum = None
    else:
        g = None
    ctx.save_for_backward(q, k, v, g, g_cumsum, initial_state, A)
    ctx.chunk_size = chunk_size
    ctx.scale = scale
    ctx.cu_seqlens = cu_seqlens
    return o, ht


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


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8)], key=['BV', 'BT'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_gla_bwd_kernel_dA(v, do, dA, cu_seqlens, chunk_indices, scale, T,
    H: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr,
    IS_VARLEN: tl.constexpr):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_do = tl.make_block_ptr(do + (bos * H + i_h) * V, (T, V), (H * V, 
            1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T), (1, H * V),
            (i_v * BV, i_t * BT), (BV, BT), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dA += tl.dot(b_do, b_v)
    p_dA = tl.make_block_ptr(dA + (bos * H + i_h) * BT, (T, BT), (H * BT, 1
        ), (i_t * BT, 0), (BT, BT), (1, 0))
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    b_dA = tl.where(m_s, b_dA * scale, 0.0)
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BK': BK, 'BV': BV}, num_warps=
    num_warps) for BK in BK_LIST for BV in BV_LIST for num_warps in [2, 4, 
    8]], key=['BT'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_gla_bwd_kernel_dv(k, g, A, do, dh, dv, cu_seqlens, chunk_indices,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, IS_VARLEN: tl.constexpr):
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
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (BT, T), (1, H * BT),
        (0, i_t * BT), (BT, BT), (0, 1))
    p_do = tl.make_block_ptr(do + (bos * H + i_h) * V, (T, V), (H * V, 1),
        (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * H + i_h) * V, (T, V), (H * V, 1),
        (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :],
        b_A, 0.0)
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dv = tl.dot(b_A, b_do.to(b_A.dtype), allow_tf32=False)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_gk = tl.make_block_ptr(g + (bos * H + i_h) * K, (T, K), (H * K, 1
            ), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_gn = g + (bos + min(i_t * BT + BT, T) - 1) * H * K + i_h * K + o_k
        p_dh = tl.make_block_ptr(dh + (i_tg * H + i_h) * K * V, (K, V), (V,
            1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_gn = exp(tl.load(p_gn, mask=m_k, other=0)[None, :] - b_gk)
        b_k = (b_k * b_gn).to(b_k.dtype)
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dv += tl.dot(b_k, b_dh.to(b_k.dtype))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BK': BK, 'BV': BV}, num_warps=
    num_warps) for BK in BK_LIST for BV in BV_LIST for num_warps in [2, 4, 
    8]], key=['BT'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_gla_bwd_kernel_inter(q, k, v, h, g, do, dh, dq, dk, dq2, dk2, dg,
    cu_seqlens, chunk_indices, scale, T, H: tl.constexpr, K: tl.constexpr,
    V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    IS_VARLEN: tl.constexpr):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    p_gk = tl.make_block_ptr(g + (bos * H + i_h) * K, (T, K), (H * K, 1), (
        i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_gn = g + (bos + min(T, i_t * BT + BT) - 1) * H * K + i_h * K + o_k
    b_gn = tl.load(p_gn, mask=m_k, other=0)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dgk = tl.zeros([BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1),
            (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + (bos * H + i_h) * V, (T, V), (H * V, 
            1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + (i_tg * H + i_h) * K * V, (V, K), (1, V
            ), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh + (i_tg * H + i_h) * K * V, (V, K), (1,
            V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dgk += tl.sum(b_h * b_dh, axis=0)
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))
    b_dgk *= exp(b_gn)
    b_dq *= scale
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_dq = b_dq * exp(b_gk)
    b_dk = b_dk * exp(b_gn[None, :] - b_gk)
    p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H * K, 1), (
        i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (
        i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq + (bos * H + i_h) * K, (T, K), (H * K, 1),
        (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + (bos * H + i_h) * K, (T, K), (H * K, 1),
        (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dgk += tl.sum(b_dk * b_k, axis=0)
    b_dq += tl.load(p_dq, boundary_check=(0, 1))
    b_dk += tl.load(p_dk, boundary_check=(0, 1))
    b_dg = b_q * b_dq - b_k * b_dk
    b_dg = b_dg - tl.cumsum(b_dg, axis=0) + tl.sum(b_dg, axis=0)[None, :
        ] + b_dgk[None, :]
    p_dq = tl.make_block_ptr(dq2 + (bos * H + i_h) * K, (T, K), (H * K, 1),
        (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk2 + (bos * H + i_h) * K, (T, K), (H * K, 1),
        (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg + (bos * H + i_h) * K, (T, K), (H * K, 1),
        (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for
    num_warps in [1, 2, 4, 8]], key=['BK', 'NC', 'BT'], **autotune_cache_kwargs
    )
@triton.jit(do_not_specialize=['T'])
def chunk_gla_bwd_kernel_intra(q, k, g, dA, dq, dk, cu_seqlens,
    chunk_indices, T, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr,
    BC: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr, IS_VARLEN: tl.
    constexpr):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_t, i_i = i_c // NC, i_c % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos
    if i_t * BT + i_i * BC >= T:
        return
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    p_g = tl.make_block_ptr(g + (bos * H + i_h) * K, (T, K), (H * K, 1), (
        i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        p_gn = g + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(0, i_i):
            p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K,
                1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g + (bos * H + i_h) * K, (T, K), (H *
                K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dA = tl.make_block_ptr(dA + (bos * H + i_h) * BT, (T, BT), (H *
                BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_kg = b_k * exp(b_gn[None, :] - b_gk)
            b_dA = tl.load(p_dA, boundary_check=(0, 1))
            b_dq += tl.dot(b_dA, b_kg)
        b_dq *= exp(b_g - b_gn[None, :])
    o_i = tl.arange(0, BC)
    m_dA = i_t * BT + i_i * BC + tl.arange(0, BC) < T
    o_dA = bos * H * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)
        ) * H * BT + i_h * BT + i_i * BC
    p_kj = k + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
    p_gkj = g + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
    p_dq = tl.make_block_ptr(dq + (bos * H + i_h) * K, (T, K), (H * K, 1),
        (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_dA = tl.load(dA + o_dA + j, mask=m_dA, other=0)
        b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
        m_i = o_i[:, None] >= j
        b_dq += tl.where(m_i, b_dA[:, None] * b_kj[None, :] * exp(b_g -
            b_gkj[None, :]), 0.0)
        p_kj += H * K
        p_gkj += H * K
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.debug_barrier()
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        p_gn = g + (bos + min(i_t * BT + i_i * BC + BC, T) - 1
            ) * H * K + i_h * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(i_i + 1, NC):
            p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H * K,
                1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gq = tl.make_block_ptr(g + (bos * H + i_h) * K, (T, K), (H *
                K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dA = tl.make_block_ptr(dA + (bos * H + i_h) * BT, (BT, T), (1,
                H * BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_gq = tl.load(p_gq, boundary_check=(0, 1))
            b_qg = b_q * tl.where(m_j[:, None], exp(b_gq - b_gn[None, :]), 0)
            b_dA = tl.load(p_dA, boundary_check=(0, 1))
            b_dk += tl.dot(b_dA, b_qg)
        b_dk *= exp(b_gn[None, :] - b_g)
    o_dA = bos * H * BT + (i_t * BT + i_i * BC
        ) * H * BT + i_h * BT + i_i * BC + tl.arange(0, BC)
    p_qj = q + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
    p_gqj = g + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
    p_dk = tl.make_block_ptr(dk + (bos * H + i_h) * K, (T, K), (H * K, 1),
        (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_dA = tl.load(dA + o_dA + j * H * BT)
        b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
        b_gqj = tl.load(p_gqj, mask=m_k, other=0).to(tl.float32)
        m_i = o_i[:, None] <= j
        b_dk += tl.where(m_i, b_dA[:, None] * b_qj[None, :] * exp(b_gqj[
            None, :] - b_g), 0.0)
        p_qj += H * K
        p_gqj += H * K
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


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


def chunk_gla_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g:
    torch.Tensor, g_cumsum: Optional[torch.Tensor], scale: float,
    initial_state: torch.Tensor, h: torch.Tensor, A: torch.Tensor, do:
    torch.Tensor, dht: torch.Tensor, cu_seqlens: Optional[torch.LongTensor]
    =None, chunk_size: int=64):
    T = q.shape[1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    if g_cumsum is None:
        g_cumsum = chunk_local_cumsum(g, BT, cu_seqlens=cu_seqlens)
    if h is None:
        h, _ = chunk_fwd_h(k=k, v=v, g=None, gk=g_cumsum, gv=None, h0=
            initial_state, output_final_state=False, cu_seqlens=cu_seqlens,
            chunk_size=BT, states_in_fp32=True)
    dh, dh0 = chunk_bwd_dh(q=q, k=k, v=v, g=None, gk=g_cumsum, gv=None, do=
        do, h0=initial_state, dht=dht, scale=scale, cu_seqlens=cu_seqlens,
        chunk_size=BT, states_in_fp32=True)
    dv = chunk_gla_bwd_dv(k=k, g=g_cumsum, A=A, do=do, dh=dh, cu_seqlens=
        cu_seqlens, chunk_size=BT)
    dA = chunk_gla_bwd_dA(v=v, do=do, scale=scale, cu_seqlens=cu_seqlens,
        chunk_size=BT)
    dq, dk = chunk_gla_bwd_dqk_intra(q=q, k=k, g=g_cumsum, dA=dA,
        cu_seqlens=cu_seqlens, chunk_size=BT)
    dq, dk, dg = chunk_gla_bwd_dqkg(q=q, k=k, v=v, h=h, g=g_cumsum, do=do,
        dh=dh, dq=dq, dk=dk, scale=scale, cu_seqlens=cu_seqlens, chunk_size=BT)
    return dq, dk, dv, dg, dh0


def chunk_gla_bwd_dA(v: torch.Tensor, do: torch.Tensor, scale: float,
    cu_seqlens: Optional[torch.LongTensor]=None, chunk_size: int=64):
    B, T, H, V = v.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BV = min(64, triton.next_power_of_2(V))
    dA = v.new_empty(B, T, H, BT, dtype=torch.float)
    grid = NT, B * H
    chunk_gla_bwd_kernel_dA[grid](v, do, dA, cu_seqlens, chunk_indices,
        scale, T=T, H=H, V=V, BT=BT, BV=BV)
    return dA


def chunk_gla_bwd_dqk_intra(q: torch.Tensor, k: torch.Tensor, g: torch.
    Tensor, dA: torch.Tensor, cu_seqlens: Optional[torch.LongTensor]=None,
    chunk_size: int=64):
    B, T, H, K = q.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BC = min(16, BT)
    BK = min(64, triton.next_power_of_2(K))
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)
    dq = torch.empty_like(q, dtype=torch.float)
    dk = torch.empty_like(k, dtype=torch.float)
    grid = NK, NT * NC, B * H
    chunk_gla_bwd_kernel_intra[grid](q, k, g, dA, dq, dk, cu_seqlens,
        chunk_indices, T=T, H=H, K=K, BT=BT, BC=BC, BK=BK, NC=NC)
    return dq, dk


def chunk_gla_bwd_dqkg(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, h:
    torch.Tensor, g: torch.Tensor, do: torch.Tensor, dh: torch.Tensor, dq:
    torch.Tensor, dk: torch.Tensor, scale: float, cu_seqlens: Optional[
    torch.LongTensor]=None, chunk_size: int=64):
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    dg = torch.empty_like(g)
    dq2 = torch.empty_like(dq)
    dk2 = torch.empty_like(dk)

    def grid(meta):
        return triton.cdiv(K, meta['BK']), NT, B * H
    chunk_gla_bwd_kernel_inter[grid](q, k, v, h, g, do, dh, dq, dk, dq2,
        dk2, dg, cu_seqlens, chunk_indices, scale, T=T, H=H, K=K, V=V, BT=BT)
    return dq2, dk2, dg


def chunk_gla_bwd_dv(k: torch.Tensor, g: torch.Tensor, A: torch.Tensor, do:
    torch.Tensor, dh: torch.Tensor, cu_seqlens: Optional[torch.LongTensor]=
    None, chunk_size: int=64):
    B, T, H, K, V = *k.shape, do.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    dv = torch.empty_like(do)

    def grid(meta):
        return triton.cdiv(V, meta['BV']), NT, B * H
    chunk_gla_bwd_kernel_dv[grid](k, g, A, do, dh, dv, cu_seqlens,
        chunk_indices, T=T, H=H, K=K, V=V, BT=BT)
    return dv


# Backward method (kernel launch code)
@input_guard
def _ChunkGLAFunction_backward(ctx, do, dht):
    q, k, v, g, g_cumsum, initial_state, A = ctx.saved_tensors
    chunk_size, scale, cu_seqlens = ctx.chunk_size, ctx.scale, ctx.cu_seqlens
    dq, dk, dv, dg, dh0 = chunk_gla_bwd(q=q, k=k, v=v, g=g, g_cumsum=
        g_cumsum, scale=scale, h=None, A=A, initial_state=initial_state, do
        =do, dht=dht, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    return dq.to(q), dk.to(k), dv.to(v), dg, None, dh0, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class ChunkGLAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(ctx, q, k, v, g, scale, initial_state, output_final_state,
        cu_seqlens):
        T = q.shape[1]
        chunk_size = min(64, max(16, triton.next_power_of_2(T)))
        g_cumsum, A, h, ht, o = chunk_gla_fwd(q=q, k=k, v=v, g=g, g_cumsum=
            None, scale=scale, initial_state=initial_state,
            output_final_state=output_final_state, cu_seqlens=cu_seqlens,
            chunk_size=chunk_size)
        if g.dtype != torch.float:
            g_cumsum = None
        else:
            g = None
        ctx.save_for_backward(q, k, v, g, g_cumsum, initial_state, A)
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        return o, ht

    @staticmethod
    @input_guard
    def backward(ctx, do, dht):
        q, k, v, g, g_cumsum, initial_state, A = ctx.saved_tensors
        chunk_size, scale, cu_seqlens = (ctx.chunk_size, ctx.scale, ctx.
            cu_seqlens)
        dq, dk, dv, dg, dh0 = chunk_gla_bwd(q=q, k=k, v=v, g=g, g_cumsum=
            g_cumsum, scale=scale, h=None, A=A, initial_state=initial_state,
            do=do, dht=dht, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
        return dq.to(q), dk.to(k), dv.to(v), dg, None, dh0, None, None
