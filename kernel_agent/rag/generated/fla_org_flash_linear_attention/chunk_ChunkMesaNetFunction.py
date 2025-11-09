# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/fla-org/flash-linear-attention
# Source-Files: fla/ops/mesa_net/chunk.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rh2l8zei/flash-linear-attention-main/fla/ops/mesa_net/chunk.py
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

def chunk_mesa_fwd_h(k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
    beta: torch.Tensor, h_init: torch.Tensor, h_kv_init: torch.Tensor,
    output_final_state: bool, cu_seqlens: Optional[torch.Tensor]=None,
    chunk_size: int=64, split_size: Optional[int]=None, states_in_fp32:
    bool=False) ->Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    assert K == V, 'K must be equal to V for now'
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
    h_kv = k.new_empty(B, NS, H, K, V, dtype=k.dtype if not states_in_fp32 else
        torch.float)
    h_final = k.new_empty(N, H, K, V, dtype=torch.float
        ) if output_final_state else None
    h_kv_final = k.new_empty(N, H, K, V, dtype=torch.float)

    def grid(meta):
        return triton.cdiv(K, 64), triton.cdiv(V, 64), N * H
    chunk_mesa_net_fwd_kernel_h[grid](k=k, v=v, beta=beta, g=g, h=h, h_kv=
        h_kv, h_init=h_init, h_kv_init=h_kv_init, h_final=h_final,
        h_kv_final=h_kv_final, cu_seqlens=cu_seqlens, split_offsets=
        split_offsets, T=T, H=H, K=K, V=V, BT=BT, BS=BS, BK=64, BV=64)
    return h, h_kv, h_final, h_kv_final


@triton.heuristics({'USE_INITIAL_STATE': lambda args: args['h_init'] is not
    None, 'STORE_FINAL_STATE': lambda args: args['h_final'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages
    =num_stages) for num_warps in [1, 2, 4, 8] for num_stages in [2, 3, 4]],
    key=['BT'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_mesa_net_fwd_kernel_h(k, v, beta, g, h, h_kv, h_init, h_kv_init,
    h_final, h_kv_final, cu_seqlens, split_offsets, T, H: tl.constexpr, K:
    tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr, BK:
    tl.constexpr, BV: tl.constexpr, USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
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
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    b_h_kv = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h_init + i_nh * K * V, (K, V), (V, 1), (
            i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
        p_h_kv0 = tl.make_block_ptr(h_kv_init + i_nh * K * V, (K, V), (V, 1
            ), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h_kv = tl.load(p_h_kv0, boundary_check=(0, 1)).to(tl.float32)
    for i_t in range(NT):
        i_s = i_t // (BS // BT)
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k2 = tl.make_block_ptr(k + (bos * H + i_h) * V, (T, V), (H * V, 1
            ), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1),
            (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_beta = tl.make_block_ptr(beta + (bos * H + i_h), (T,), (H,), (i_t *
            BT,), (BT,), (0,))
        b_beta = tl.load(p_beta, boundary_check=(0,))
        o_h = ((boh + i_s) * H + i_h).to(tl.int64) * K * V
        p_h = tl.make_block_ptr(h + o_h, (K, V), (V, 1), (i_k * BK, i_v *
            BV), (BK, BV), (1, 0))
        p_h_kv = tl.make_block_ptr(h_kv + o_h, (K, V), (V, 1), (i_k * BK, 
            i_v * BV), (BK, BV), (1, 0))
        if i_t % (BS // BT) == 0:
            tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_h_kv, b_h_kv.to(p_h_kv.dtype.element_ty),
                boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_k2 = tl.load(p_k2, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
        p_g = g + bos * H + (i_t * BT + tl.arange(0, BT)) * H + i_h
        b_h *= exp(b_g_last)
        b_h_kv *= exp(b_g_last)
        b_g = tl.load(p_g, mask=i_t * BT + tl.arange(0, BT) < T, other=0.0)
        b_k_decay = (b_k * exp(b_g_last - b_g)[:, None] * b_beta[:, None]).to(
            b_k2.dtype)
        b_h += tl.dot(tl.trans(b_k_decay), b_k2)
        b_h_kv += tl.dot(tl.trans(b_k_decay), b_v.to(b_k2.dtype))
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(h_final + i_nh * K * V, (K, V), (V, 1), (
            i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        p_h_kv_final = tl.make_block_ptr(h_kv_final + i_nh * K * V, (K, V),
            (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h_kv_final, b_h_kv.to(p_h_kv_final.dtype.element_ty),
            boundary_check=(0, 1))


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

@triton.autotune(configs=[triton.Config({'BT': BT}, num_warps=num_warps) for
    num_warps in [1, 2, 4, 8, 16] for BT in BT_LIST], key=['D', 'NB'], **
    autotune_cache_kwargs)
@triton.jit
def l2norm_fwd_kernel(x, y, rstd, eps, T: tl.constexpr, D: tl.constexpr, BD:
    tl.constexpr, NB: tl.constexpr, BT: tl.constexpr):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
    b_y = b_x * b_rstd[:, None]
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))


@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for
    num_warps in NUM_WARPS_AUTOTUNE], key=['D'], **autotune_cache_kwargs)
@triton.jit
def l2norm_fwd_kernel1(x, y, rstd, eps, D, BD: tl.constexpr):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    cols = tl.arange(0, BD)
    mask = cols < D
    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x) + eps)
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)
    tl.store(rstd + i_t, b_rstd)


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_mesa_cg_dim64_kernel(q, q_final, k, h, o, v, h_kv, g, beta,
    lamb, cu_seqlens, chunk_indices, T, max_CG_iteration: tl.constexpr, H:
    tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
    IS_VARLEN: tl.constexpr):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
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
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    q += (bos * H + i_h) * K
    q_final += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    h += (i_tg * H + i_h).to(tl.int64) * K * K
    g += bos * H + i_h
    beta += bos * H + i_h
    lamb += i_h * K
    o += (bos * H + i_h) * K
    v += (bos * H + i_h) * K
    h_kv += (i_tg * H + i_h).to(tl.int64) * K * K
    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK),
        (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK),
        (1, 0))
    p_h = tl.make_block_ptr(h, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
    b_h = tl.load(p_h, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
    p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
    p_lamb = tl.make_block_ptr(lamb, (K,), (1,), (0,), (BK,), (0,))
    b_lamb = tl.load(p_lamb, boundary_check=(0,)).to(tl.float32)
    b_m = exp(b_g[:, None] - b_g[None, :]) * b_beta[None, :]
    b_m = tl.where((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[
        None, :]), b_m, 0)
    b_g_exp_q = tl.exp(b_g)[:, None]
    b_x = tl.zeros([BT, BK], dtype=tl.float32)
    b_p = tl.zeros([BT, BK], dtype=tl.float32)
    b_r = tl.zeros([BT, BK], dtype=tl.float32)
    b_x += b_q * 0.0
    b_r += b_q
    b_p += b_r
    b_delta_old = tl.sum(b_r * b_r, axis=1)
    for i in range(max_CG_iteration):
        b_o = chunk_update_once(b_p, b_k, b_k, b_m, b_g_exp_q, b_h, b_lamb)
        alpha = b_delta_old / (tl.sum(b_p * b_o, axis=1) + 1e-05)
        b_x += alpha[:, None] * b_p
        b_r = b_r - alpha[:, None] * b_o
        b_delta_new = tl.sum(b_r * b_r, axis=1)
        b_p = b_r + (b_delta_new / (b_delta_old + 1e-05))[:, None] * b_p
        b_delta_old = b_delta_new
    p_q_final = tl.make_block_ptr(q_final, (T, K), (H * K, 1), (i_t * BT, 0
        ), (BT, BK), (1, 0))
    tl.store(p_q_final, b_x.to(p_q_final.dtype.element_ty), boundary_check=
        (0, 1))
    p_h_kv = tl.make_block_ptr(h_kv, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
    b_h_kv = tl.load(p_h_kv, boundary_check=(0, 1))
    p_v = tl.make_block_ptr(v, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK),
        (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = chunk_update_once(b_x, b_k, b_v, b_m, b_g_exp_q, b_h_kv, None)
    p_o = tl.make_block_ptr(o, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK),
        (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit()
def chunk_update_once(b_p, b_k, b_v, b_m, b_g_exp_q, b_h, b_lamb):
    b_o = tl.dot((tl.dot(b_p.to(b_k.dtype), tl.trans(b_k)) * b_m).to(b_v.
        dtype), b_v)
    b_o += tl.dot((b_p * b_g_exp_q).to(b_h.dtype), b_h)
    if b_lamb is not None:
        b_o += b_lamb[None, :] * b_p
    return b_o


def l2norm_fwd(x: torch.Tensor, eps: float=1e-06, output_dtype: Optional[
    torch.dtype]=None):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")
    rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    if D <= 512:
        NB = triton.cdiv(T, 2048)

        def grid(meta):
            return triton.cdiv(T, meta['BT']),
        l2norm_fwd_kernel[grid](x=x, y=y, rstd=rstd, eps=eps, T=T, D=D, BD=
            BD, NB=NB)
    else:
        l2norm_fwd_kernel1[T,](x=x, y=y, rstd=rstd, eps=eps, D=D, BD=BD)
    return y.view(x_shape_og), rstd.view(x_shape_og[:-1])


def chunk_fwd_mesa_net_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.
    Tensor, g: torch.Tensor, beta: torch.Tensor, lamb: torch.Tensor,
    cu_seqlens: torch.Tensor, max_CG_iteration: int=30, chunk_size: int=64,
    h_kk_init: Optional[torch.Tensor]=None, h_kv_init: Optional[torch.
    Tensor]=None, output_final_state: bool=False) ->torch.Tensor:
    g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens
        ) if g is not None else None
    h_kk, h_kv, h_kk_final, h_kv_final = chunk_mesa_fwd_h(k=k, v=v, g=g,
        beta=beta, h_init=h_kk_init, h_kv_init=h_kv_init,
        output_final_state=output_final_state, states_in_fp32=False,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    q_star, o = chunk_mesa_cg_fwd(q=q, k=k, h=h_kk, h_kv=h_kv, v=v,
        g_local_cumsum=g, beta=beta, lamb=lamb, cu_seqlens=cu_seqlens,
        chunk_size=chunk_size, max_CG_iteration=max_CG_iteration)
    return g, q_star, o, (h_kk_final, h_kv_final)


def chunk_mesa_cg_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, h:
    torch.Tensor, h_kv: torch.Tensor, g_local_cumsum: torch.Tensor, beta:
    torch.Tensor, lamb: torch.Tensor, cu_seqlens: Optional[torch.Tensor]=
    None, chunk_size: int=64, max_CG_iteration: int=30, output_dtype:
    Optional[torch.dtype]=None) ->torch.Tensor:
    B, T, H, K = q.shape
    assert K <= 128, 'head dimension must be less than 128'
    assert chunk_size <= 64 or K <= 64, 'either chunk size or head dimension must be no greater than 64'
    q_final = torch.empty_like(q, dtype=q.dtype if output_dtype is None else
        output_dtype)
    assert v is not None, 'v must be provided if calculate_output is True'
    assert h_kv is not None, 'h_kv must be provided if calculate_output is True'
    o = torch.empty_like(v)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, chunk_size) if cu_seqlens is None else len(
        chunk_indices)
    BK = max(triton.next_power_of_2(K), 16)
    grid = NT, H * B
    chunk_fwd_mesa_cg_dim64_kernel[grid](q=q, q_final=q_final, o=o, v=v,
        h_kv=h_kv, k=k, h=h, g=g_local_cumsum, beta=beta, lamb=lamb,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        max_CG_iteration=max_CG_iteration, T=T, H=H, K=K, BT=chunk_size, BK
        =BK, num_warps=4, num_stages=1)
    return q_final, o


# Forward method (kernel launch code)
@input_guard
@autocast_custom_fwd
def _ChunkMesaNetFunction_forward(ctx, q, k, v, g, beta, lamb, cu_seqlens,
    max_CG_iteration, h_kk_init, h_kv_init, output_final_state,
    use_qk_l2norm_in_kernel):
    chunk_size = 64
    if use_qk_l2norm_in_kernel:
        q, q_rstd = l2norm_fwd(q, output_dtype=torch.float16)
        k, k_rstd = l2norm_fwd(k, output_dtype=torch.float16)
    else:
        q_rstd, k_rstd = None, None
        q = q.to(torch.float16)
        k = k.to(torch.float16)
    g_cumsum, q_star, o, (h_kk_final, h_kv_final) = chunk_fwd_mesa_net_fwd(q
        =q, k=k, v=v, g=g, beta=beta, lamb=lamb, cu_seqlens=cu_seqlens,
        max_CG_iteration=max_CG_iteration, chunk_size=chunk_size, h_kk_init
        =h_kk_init, h_kv_init=h_kv_init, output_final_state=output_final_state)
    ctx.max_CG_iteration = max_CG_iteration
    ctx.chunk_size = chunk_size
    ctx.cu_seqlens = cu_seqlens
    ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
    ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g_cumsum, beta, lamb,
        h_kk_init, h_kv_init, q_star, o)
    return o, h_kk_final, h_kv_final


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=[triton.Config({'BT': BT}, num_warps=num_warps) for
    num_warps in [1, 2, 4, 8, 16] for BT in BT_LIST], key=['D', 'NB'], **
    autotune_cache_kwargs)
@triton.jit
def l2norm_bwd_kernel(y, rstd, dy, dx, eps, T: tl.constexpr, D: tl.
    constexpr, BD: tl.constexpr, NB: tl.constexpr, BT: tl.constexpr):
    i_t = tl.program_id(0)
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))
    p_dy = tl.make_block_ptr(dy, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (
        1, 0))
    p_dx = tl.make_block_ptr(dx, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (
        1, 0))
    b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
    b_rstd = tl.load(p_rstd, boundary_check=(0,)).to(tl.float32)
    b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
    b_dx = b_dy * b_rstd[:, None] - tl.sum(b_dy * b_y, 1)[:, None
        ] * b_y * b_rstd[:, None]
    tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for
    num_warps in NUM_WARPS_AUTOTUNE], key=['D'], **autotune_cache_kwargs)
@triton.jit
def l2norm_bwd_kernel1(y, rstd, dy, dx, eps, D, BD: tl.constexpr):
    i_t = tl.program_id(0)
    y += i_t * D
    dx += i_t * D
    dy += i_t * D
    cols = tl.arange(0, BD)
    mask = cols < D
    b_y = tl.load(y + cols, mask=mask, other=0.0).to(tl.float32)
    b_rstd = tl.load(rstd + i_t).to(tl.float32)
    b_dy = tl.load(dy + cols, mask=mask, other=0.0).to(tl.float32)
    b_dx = b_dy * b_rstd - tl.sum(b_dy * b_y) * b_y * b_rstd
    tl.store(dx + cols, b_dx, mask=mask)


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
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_mesa_cg_dim64_kernel(dq, dq_final, k, h, g, beta, lamb,
    cu_seqlens, chunk_indices, T, max_CG_iteration: tl.constexpr, H: tl.
    constexpr, K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
    IS_VARLEN: tl.constexpr):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
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
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    dq += (bos * H + i_h) * K
    dq_final += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    h += (i_tg * H + i_h).to(tl.int64) * K * K
    g += bos * H + i_h
    beta += bos * H + i_h
    lamb += i_h * K
    p_q = tl.make_block_ptr(dq, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK),
        (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK),
        (1, 0))
    p_h = tl.make_block_ptr(h, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
    b_h = tl.load(p_h, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
    p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
    p_lamb = tl.make_block_ptr(lamb, (K,), (1,), (0,), (BK,), (0,))
    b_lamb = tl.load(p_lamb, boundary_check=(0,)).to(tl.float32)
    b_m = exp(b_g[:, None] - b_g[None, :]) * b_beta[None, :]
    b_m = tl.where((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[
        None, :]), b_m, 0)
    b_g_exp_q = tl.exp(b_g)[:, None]
    b_x = tl.zeros([BT, BK], dtype=tl.float32)
    b_p = tl.zeros([BT, BK], dtype=tl.float32)
    b_r = tl.zeros([BT, BK], dtype=tl.float32)
    b_x += b_q * 0.0
    b_r += b_q
    b_p += b_q
    b_delta_old = tl.sum(b_r * b_r, axis=1)
    for _ in range(max_CG_iteration):
        b_o = chunk_update_once(b_p, b_k, b_k, b_m, b_g_exp_q, b_h, b_lamb)
        alpha = b_delta_old / (tl.sum(b_p * b_o, axis=1) + 1e-05)
        b_x += alpha[:, None] * b_p
        b_r = b_r - alpha[:, None] * b_o
        b_delta_new = tl.sum(b_r * b_r, axis=1)
        b_p = b_r + (b_delta_new / (b_delta_old + 1e-05))[:, None] * b_p
        b_delta_old = b_delta_new
    p_q_final = tl.make_block_ptr(dq_final, (T, K), (H * K, 1), (i_t * BT, 
        0), (BT, BK), (1, 0))
    tl.store(p_q_final, b_x.to(p_q_final.dtype.element_ty), boundary_check=
        (0, 1))


@triton.jit()
def chunk_update_once(b_p, b_k, b_v, b_m, b_g_exp_q, b_h, b_lamb):
    b_o = tl.dot((tl.dot(b_p.to(b_k.dtype), tl.trans(b_k)) * b_m).to(b_v.
        dtype), b_v)
    b_o += tl.dot((b_p * b_g_exp_q).to(b_h.dtype), b_h)
    if b_lamb is not None:
        b_o += b_lamb[None, :] * b_p
    return b_o


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=['T'])
def chunk_mesa_net_h_kk_bwd_intra_kernel(k, beta, h, dh, g, q_star, dq, dk,
    dg, dbeta, dk_beta, dlamb, cu_seqlens, chunk_indices, B: tl.constexpr,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
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
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    q_star += (bos * H + i_h) * V
    dq += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V
    dh += (i_tg * H + i_h).to(tl.int64) * K * V
    k += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dk_beta += (bos * H + i_h) * K
    dlamb += (i_tg * H + i_h).to(tl.int64) * K
    beta += bos * H + i_h
    dbeta += bos * H + i_h
    g += bos * H + i_h
    dg += bos * H + i_h
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dv = tl.zeros([BT, BK], dtype=tl.float32)
    b_dbeta = tl.zeros([BT], dtype=tl.float32)
    b_dg_last = tl.zeros([1], dtype=tl.float32)
    b_dg = tl.zeros([BT], dtype=tl.float32)
    p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * H)
    b_gk = tl.where(m_t, exp(b_g_last - b_g), 0)
    p_q_star = tl.make_block_ptr(q_star, (T, V), (H * V, 1), (i_t * BT, 0),
        (BT, BV), (1, 0))
    b_q_star = tl.load(p_q_star, boundary_check=(0, 1))
    p_dq = tl.make_block_ptr(dq, (T, V), (H * V, 1), (i_t * BT, 0), (BT, BV
        ), (1, 0))
    b_dq = tl.load(p_dq, boundary_check=(0, 1))
    b_dlamb = -tl.sum(b_q_star * b_dq, axis=0)
    p_dlamb = tl.make_block_ptr(dlamb, (K,), (1,), (0,), (BK,), (0,))
    tl.store(p_dlamb, b_dlamb.to(p_dlamb.dtype.element_ty), boundary_check=(0,)
        )
    p_h = tl.make_block_ptr(h, (V, K), (1, V), (0, 0), (BV, BK), (0, 1))
    p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (0, 0), (BV, BK), (0, 1))
    p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK),
        (1, 0))
    b_v = tl.load(p_k, boundary_check=(0, 1))
    b_k = (b_v * b_beta[:, None]).to(b_v.dtype)
    b_m = tl.where((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[
        None, :]), exp(b_g[:, None] - b_g[None, :]), 0)
    b_s = tl.dot(b_q_star, tl.trans(b_k)) * b_m
    b_ds = tl.dot(b_dq, tl.trans(b_v))
    b_dv += tl.dot(tl.trans(b_s.to(b_dq.dtype)), b_dq)
    b_dm = b_s * b_ds
    b_dm = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :],
        b_dm, 0)
    b_dg += tl.sum(b_dm, axis=1)
    b_dg -= tl.sum(b_dm, axis=0)
    b_ds = b_ds * b_m
    b_dk += tl.dot(tl.trans(b_ds.to(b_q_star.dtype)), b_q_star)
    b_h = tl.load(p_h, boundary_check=(0, 1))
    b_dg += tl.sum(tl.dot(b_dq, tl.trans(b_h)) * tl.exp(b_g)[:, None] *
        b_q_star, axis=1)
    b_dh = tl.load(p_dh, boundary_check=(0, 1))
    b_dk2 = tl.dot(b_v, b_dh.to(b_v.dtype)) * b_gk[:, None]
    b_dg -= tl.sum(b_dk2 * b_k, axis=1)
    b_dg_last += tl.sum(b_dk2 * b_k)
    b_dk += b_dk2
    b_dv += tl.dot(b_k, tl.trans(b_dh).to(b_k.dtype)) * b_gk[:, None]
    b_dh = b_dh * b_h
    b_dg_last += tl.sum(b_dh) * exp(b_g_last)
    p_dk_beta = tl.make_block_ptr(dk_beta, (T, K), (H * K, 1), (i_t * BT, 0
        ), (BT, BK), (1, 0))
    b_dk -= tl.load(p_dk_beta, boundary_check=(0, 1))
    b_dbeta = tl.sum(b_dk * b_v, axis=1)
    b_dk = b_dk * b_beta[:, None] + b_dv
    b_dk = -b_dk
    b_dg = tl.where(o_t < min(i_t * BT + BT, T) - 1, b_dg, b_dg + b_dg_last)
    p_dk = tl.make_block_ptr(dk, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK
        ), (1, 0))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    p_dg = tl.make_block_ptr(dg, (T,), (H,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dg, -b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))
    p_dbeta = tl.make_block_ptr(dbeta, (T,), (H,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dbeta, -b_dbeta.to(p_dbeta.dtype.element_ty), boundary_check
        =(0,))


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages
    =num_stages) for num_warps in NUM_WARPS for num_stages in [2, 3, 4]],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_mesa_net_h_kv_bwd_intra_kernel(q_star, k, v, beta, h_kv, g, do,
    dh_kv, dq, dk_beta, dg, dv, cu_seqlens, chunk_indices, B: tl.constexpr,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
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
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    h_kv += (i_tg * H + i_h).to(tl.int64) * K * V
    dh_kv += (i_tg * H + i_h).to(tl.int64) * K * V
    q_star += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    beta += bos * H + i_h
    g += bos * H + i_h
    dg += bos * H + i_h
    dq += (bos * H + i_h) * K
    dk_beta += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dv = tl.zeros([BT, BK], dtype=tl.float32)
    b_dg_last = tl.zeros([1], dtype=tl.float32)
    b_dg = tl.zeros([BT], dtype=tl.float32)
    p_q = tl.make_block_ptr(q_star, (T, K), (H * K, 1), (i_t * BT, 0), (BT,
        BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK),
        (1, 0))
    p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, 0), (BT, BV),
        (1, 0))
    p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_t * BT, 0), (BT, BV
        ), (1, 0))
    p_h = tl.make_block_ptr(h_kv, (V, K), (1, V), (0, 0), (BV, BK), (0, 1))
    p_dh = tl.make_block_ptr(dh_kv, (V, K), (1, V), (0, 0), (BV, BK), (0, 1))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_h = tl.load(p_h, boundary_check=(0, 1))
    b_dh = tl.load(p_dh, boundary_check=(0, 1))
    b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * H)
    b_dg_last += tl.sum(b_h * b_dh)
    b_dg_last *= exp(b_g_last)
    b_m = tl.where((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[
        None, :]), exp(b_g[:, None] - b_g[None, :]), 0)
    b_k = (b_k * b_beta[:, None]).to(b_k.dtype)
    b_s = tl.dot(b_q, tl.trans(b_k)) * b_m
    b_ds = tl.dot(b_do, tl.trans(b_v))
    b_dm = b_s * b_ds
    b_dm = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :],
        b_dm, 0)
    b_dg += tl.sum(b_dm, axis=1)
    b_dg -= tl.sum(b_dm, axis=0)
    b_g_exp_q = exp(b_g)
    b_g_exp_k = tl.where(m_t, exp(-b_g + b_g_last), 0)
    b_ds = b_ds * b_m
    b_dq += tl.dot(b_do, b_h.to(b_do.dtype)) * b_g_exp_q[:, None]
    b_dk += tl.dot(b_v, b_dh.to(b_v.dtype)) * b_g_exp_k[:, None]
    b_dg_last += tl.sum(b_dk * b_k)
    b_dg -= tl.sum(b_dk * b_k, axis=1)
    b_dg += tl.sum(b_dq * b_q, axis=1)
    b_dq += tl.dot(b_ds.to(b_k.dtype), b_k)
    b_dv += tl.dot(b_k, tl.trans(b_dh).to(b_k.dtype)) * b_g_exp_k[:, None
        ] + tl.dot(tl.trans(b_s.to(b_do.dtype)), b_do)
    b_dk += tl.dot(tl.trans(b_ds.to(b_q.dtype)), b_q)
    b_dg = tl.where(o_t < min(i_t * BT + BT, T) - 1, b_dg, b_dg + b_dg_last)
    p_dq = tl.make_block_ptr(dq, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK
        ), (1, 0))
    p_dk = tl.make_block_ptr(dk_beta, (T, K), (H * K, 1), (i_t * BT, 0), (
        BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t * BT, 0), (BT, BV
        ), (1, 0))
    p_dg = tl.make_block_ptr(dg, (T,), (H,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages
    =num_stages) for num_warps in NUM_WARPS for num_stages in [2, 3, 4]],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_mesa_net_h_kv_bwd_intra_kernel_dkv(q_star, k, v, beta, h_kv, g,
    do, dh_kv, dk_beta, dg, dv, cu_seqlens, chunk_indices, B: tl.constexpr,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
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
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    h_kv += (i_tg * H + i_h).to(tl.int64) * K * V
    dh_kv += (i_tg * H + i_h).to(tl.int64) * K * V
    q_star += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    beta += bos * H + i_h
    g += bos * H + i_h
    dg += bos * H + i_h
    dk_beta += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dv = tl.zeros([BT, BK], dtype=tl.float32)
    b_dg_last = tl.zeros([1], dtype=tl.float32)
    b_dg = tl.zeros([BT], dtype=tl.float32)
    p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, 0), (BT, BV),
        (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK),
        (1, 0))
    p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_t * BT, 0), (BT, BV
        ), (1, 0))
    p_h = tl.make_block_ptr(h_kv, (V, K), (1, V), (0, 0), (BV, BK), (0, 1))
    p_dh = tl.make_block_ptr(dh_kv, (V, K), (1, V), (0, 0), (BV, BK), (0, 1))
    p_q = tl.make_block_ptr(q_star, (T, K), (H * K, 1), (i_t * BT, 0), (BT,
        BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_h = tl.load(p_h, boundary_check=(0, 1))
    b_dh = tl.load(p_dh, boundary_check=(0, 1))
    b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * H)
    b_dg_last += tl.sum(b_h * b_dh)
    b_dg_last *= exp(b_g_last)
    b_m = tl.where((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[
        None, :]), exp(b_g[:, None] - b_g[None, :]), 0)
    b_k = (b_k * b_beta[:, None]).to(b_k.dtype)
    b_s = tl.dot(b_q, tl.trans(b_k)) * b_m
    b_ds = tl.dot(b_do, tl.trans(b_v))
    b_dm = b_s * b_ds
    b_dm = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :],
        b_dm, 0)
    b_dg += tl.sum(b_dm, axis=1)
    b_dg -= tl.sum(b_dm, axis=0)
    b_g_exp_k = tl.where(m_t, exp(-b_g + b_g_last), 0)
    b_ds = b_ds * b_m
    b_dk += tl.dot(b_v, b_dh.to(b_v.dtype)) * b_g_exp_k[:, None]
    b_dg_last += tl.sum(b_dk * b_k)
    b_dg -= tl.sum(b_dk * b_k, axis=1)
    b_dv += tl.dot(b_k, tl.trans(b_dh).to(b_k.dtype)) * b_g_exp_k[:, None
        ] + tl.dot(tl.trans(b_s.to(b_do.dtype)), b_do)
    b_dk += tl.dot(tl.trans(b_ds.to(b_q.dtype)), b_q)
    b_dg = tl.where(o_t < min(i_t * BT + BT, T) - 1, b_dg, b_dg + b_dg_last)
    p_dk = tl.make_block_ptr(dk_beta, (T, K), (H * K, 1), (i_t * BT, 0), (
        BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t * BT, 0), (BT, BV
        ), (1, 0))
    p_dg = tl.make_block_ptr(dg, (T,), (H,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages
    =num_stages) for num_warps in NUM_WARPS for num_stages in [2, 3, 4]],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_mesa_net_h_kv_bwd_intra_kernel_dq(q_star, k, v, beta, h_kv, g, do,
    dq, dg_prev, dg, cu_seqlens, chunk_indices, B: tl.constexpr, T, H: tl.
    constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.
    constexpr, BV: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
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
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    h_kv += (i_tg * H + i_h).to(tl.int64) * K * V
    q_star += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    beta += bos * H + i_h
    g += bos * H + i_h
    dg_prev += bos * H + i_h
    dg += bos * H + i_h
    dq += (bos * H + i_h) * K
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, 0), (BT, BV),
        (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK),
        (1, 0))
    p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_t * BT, 0), (BT, BV
        ), (1, 0))
    p_h = tl.make_block_ptr(h_kv, (V, K), (1, V), (0, 0), (BV, BK), (0, 1))
    p_q = tl.make_block_ptr(q_star, (T, K), (H * K, 1), (i_t * BT, 0), (BT,
        BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dg_prev = tl.make_block_ptr(dg_prev, (T,), (H,), (i_t * BT,), (BT,), (0,)
        )
    p_dg = tl.make_block_ptr(dg, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dq = tl.make_block_ptr(dq, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK
        ), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_h = tl.load(p_h, boundary_check=(0, 1))
    b_m = tl.where((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[
        None, :]), exp(b_g[:, None] - b_g[None, :]), 0)
    b_k = (b_k * b_beta[:, None]).to(b_k.dtype)
    b_ds = tl.dot(b_do, tl.trans(b_v)) * b_m
    b_g_exp_q = exp(b_g)
    b_dq = tl.dot(b_do, b_h.to(b_do.dtype)) * b_g_exp_q[:, None]
    b_dg = tl.sum(b_dq * b_q, axis=1) + tl.load(p_dg_prev, boundary_check=(0,))
    b_dq += tl.dot(b_ds.to(b_k.dtype), b_k)
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


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


def l2norm_bwd(y: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor, eps:
    float=1e-06):
    y_shape_og = y.shape
    y = y.view(-1, dy.shape[-1])
    dy = dy.view(-1, dy.shape[-1])
    assert dy.shape == y.shape
    dx = torch.empty_like(y)
    T, D = y.shape[0], y.shape[-1]
    MAX_FUSED_SIZE = 65536 // y.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    if D <= 512:
        NB = triton.cdiv(T, 2048)

        def grid(meta):
            return triton.cdiv(T, meta['BT']),
        l2norm_bwd_kernel[grid](y=y, rstd=rstd, dy=dy, dx=dx, eps=eps, T=T,
            D=D, BD=BD, NB=NB)
    else:
        l2norm_bwd_kernel1[T,](y=y, rstd=rstd, dy=dy, dx=dx, eps=eps, D=D,
            BD=BD)
    return dx.view(y_shape_og)


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


def chunk_fwd_mesa_net_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.
    Tensor, g: torch.Tensor, beta: torch.Tensor, lamb: torch.Tensor, q_star:
    torch.Tensor, do: torch.Tensor, cu_seqlens: torch.Tensor,
    max_CG_iteration: int=30, chunk_size: int=64, h_kk_init: Optional[torch
    .Tensor]=None, h_kv_init: Optional[torch.Tensor]=None, dh_kv_final:
    Optional[torch.Tensor]=None, dh_kk_final: Optional[torch.Tensor]=None
    ) ->torch.Tensor:
    h_kk, h_kv, _, _ = chunk_mesa_fwd_h(k=k, v=v, g=g, beta=beta, h_init=
        h_kk_init, h_kv_init=h_kv_init, output_final_state=False,
        states_in_fp32=False, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    dh_kv, dh0_kv = chunk_bwd_dh(q=q_star, k=k, v=v, g=g, gk=None, gv=None,
        do=do, h0=h_kv_init, dht=dh_kv_final, states_in_fp32=False,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size, scale=1)
    dq, dk_beta, dv, dg = chunk_mesa_net_h_kv_bwd_intra_fn(q_star=q_star, k
        =k, v=v, beta=beta, h_kv=h_kv, dh_kv=dh_kv, g=g, do=do, cu_seqlens=
        cu_seqlens, chunk_size=chunk_size)
    dq = chunk_mesa_cg_bwd(dq=dq, k=k, h=h_kk, g_local_cumsum=g, beta=beta,
        lamb=lamb, cu_seqlens=cu_seqlens, chunk_size=chunk_size,
        max_CG_iteration=max_CG_iteration, output_dtype=torch.float16)
    dh_kk, dh0_kk = chunk_bwd_dh(q=dq, k=k, v=k, g=g, gk=None, gv=None, do=
        q_star, h0=h_kk_init, dht=-dh_kk_final if dh_kk_final is not None else
        None, states_in_fp32=False, cu_seqlens=cu_seqlens, chunk_size=
        chunk_size, scale=1)
    dk, dg2, dlamb, dbeta = chunk_mesa_net_h_kk_bwd_intra_fn(k=k, g=g, beta
        =beta, h=h_kk, dh=dh_kk, dk_beta=dk_beta, q_star=q_star, dq=dq,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    dg.add_(dg2)
    dg = chunk_local_cumsum(dg, chunk_size=chunk_size, reverse=True,
        cu_seqlens=cu_seqlens).to(g)
    return (dq, dk, dv, dg, dbeta, dlamb, -dh0_kk if dh0_kk is not None else
        None, dh0_kv if dh0_kv is not None else None)


def chunk_mesa_cg_bwd(dq: torch.Tensor, k: torch.Tensor, h: torch.Tensor,
    g_local_cumsum: torch.Tensor, beta: torch.Tensor, lamb: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor]=None, chunk_size: int=64,
    max_CG_iteration: int=30, output_dtype: Optional[torch.dtype]=None
    ) ->torch.Tensor:
    B, T, H, K = dq.shape
    assert K <= 128, 'head dimension must be less than 128'
    assert chunk_size <= 64 or K <= 64, 'either chunk size or head dimension must be no greater than 64'
    dq_final = torch.empty_like(dq, dtype=dq.dtype if output_dtype is None else
        output_dtype)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, chunk_size) if cu_seqlens is None else len(
        chunk_indices)
    BK = max(triton.next_power_of_2(K), 16)
    grid = NT, H * B
    chunk_fwd_mesa_cg_dim64_kernel[grid](dq=dq, dq_final=dq_final, k=k, h=h,
        g=g_local_cumsum, beta=beta, lamb=lamb, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, max_CG_iteration=max_CG_iteration, T=T,
        H=H, K=K, BT=chunk_size, BK=BK, num_warps=4, num_stages=1)
    return dq_final


def chunk_mesa_net_h_kk_bwd_intra_fn(k: torch.Tensor, beta: torch.Tensor, g:
    torch.Tensor, h: torch.Tensor, dh: torch.Tensor, q_star: torch.Tensor,
    dq: torch.Tensor, dk_beta: torch.Tensor, cu_seqlens: Optional[torch.
    LongTensor]=None, chunk_size: int=64) ->Tuple[torch.Tensor, torch.
    Tensor, torch.Tensor]:
    B, T, H, K = k.shape
    V = K
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = max(triton.next_power_of_2(K), 16)
    BV = max(triton.next_power_of_2(V), 16)
    dk = torch.empty_like(k)
    dg = torch.empty_like(g)
    dbeta = torch.empty_like(beta)
    dlamb = torch.empty(B, NT, H, K, dtype=torch.float32, device=k.device)
    grid = NT, B * H
    chunk_mesa_net_h_kk_bwd_intra_kernel[grid](k=k, h=h, dh=dh, g=g, q_star
        =q_star, beta=beta, dbeta=dbeta, dq=dq, dk=dk, dk_beta=dk_beta, dg=
        dg, dlamb=dlamb, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        B=B, T=T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV)
    dlamb = dlamb.sum([0, 1])
    return dk, dg, dlamb, dbeta


def chunk_mesa_net_h_kv_bwd_intra_fn(q_star, k, v, beta, h_kv, dh_kv, g, do,
    cu_seqlens, chunk_size=64):
    if not check_shared_mem('ampere'):
        return chunk_mesa_net_h_kv_bwd_intra_separate_fn(q_star=q_star, k=k,
            v=v, beta=beta, h_kv=h_kv, dh_kv=dh_kv, g=g, do=do, cu_seqlens=
            cu_seqlens, chunk_size=chunk_size)
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = max(triton.next_power_of_2(K), 16)
    BV = max(triton.next_power_of_2(V), 16)
    dq = torch.empty_like(q_star, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dg = torch.empty_like(g)
    grid = NT, B * H
    chunk_mesa_net_h_kv_bwd_intra_kernel[grid](q_star=q_star, k=k, v=v,
        beta=beta, h_kv=h_kv, g=g, do=do, dh_kv=dh_kv, dq=dq, dk_beta=dk,
        dg=dg, dv=dv, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, B
        =B, T=T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV)
    return dq, dk, dv, dg


def chunk_mesa_net_h_kv_bwd_intra_separate_fn(q_star, k, v, beta, h_kv,
    dh_kv, g, do, cu_seqlens, chunk_size=64):
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = max(triton.next_power_of_2(K), 16)
    BV = max(triton.next_power_of_2(V), 16)
    dq = torch.empty_like(q_star, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dg = torch.empty_like(g)
    grid = NT, B * H
    chunk_mesa_net_h_kv_bwd_intra_kernel_dkv[grid](q_star=q_star, k=k, v=v,
        beta=beta, h_kv=h_kv, g=g, do=do, dh_kv=dh_kv, dk_beta=dk, dg=dg,
        dv=dv, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, B=B, T=T,
        H=H, K=K, V=V, BT=BT, BK=BK, BV=BV)
    dg_final = torch.empty_like(dg)
    chunk_mesa_net_h_kv_bwd_intra_kernel_dq[grid](q_star=q_star, k=k, v=v,
        beta=beta, h_kv=h_kv, g=g, do=do, dg=dg_final, dg_prev=dg, dq=dq,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, B=B, T=T, H=H,
        K=K, V=V, BT=BT, BK=BK, BV=BV)
    return dq, dk, dv, dg_final


# Backward method (kernel launch code)
@input_guard
@autocast_custom_bwd
def _ChunkMesaNetFunction_backward(ctx, do, dh_kk_final=None, dh_kv_final=None
    ):
    (q, q_rstd, k, k_rstd, v, g, beta, lamb, h_kk_init, h_kv_init, q_star, o
        ) = ctx.saved_tensors
    max_CG_iteration = ctx.max_CG_iteration
    chunk_size = ctx.chunk_size
    cu_seqlens = ctx.cu_seqlens
    dq, dk, dv, dg, dbeta, dlamb, dh0_kk, dh0_kv = chunk_fwd_mesa_net_bwd(q
        =q, k=k, v=v, g=g, beta=beta, lamb=lamb, q_star=q_star, do=do,
        cu_seqlens=cu_seqlens, max_CG_iteration=max_CG_iteration,
        chunk_size=chunk_size, h_kk_init=h_kk_init, h_kv_init=h_kv_init,
        dh_kv_final=dh_kv_final, dh_kk_final=dh_kk_final)
    if ctx.use_qk_l2norm_in_kernel:
        dq = l2norm_bwd(q, q_rstd, dq)
        dk = l2norm_bwd(k, k_rstd, dk)
    return dq, dk, dv.to(v), dg.to(g), dbeta.to(beta), dlamb.to(lamb
        ), None, None, dh0_kk, dh0_kv, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class ChunkMesaNetFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, g, beta, lamb, cu_seqlens, max_CG_iteration,
        h_kk_init, h_kv_init, output_final_state, use_qk_l2norm_in_kernel):
        chunk_size = 64
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q, output_dtype=torch.float16)
            k, k_rstd = l2norm_fwd(k, output_dtype=torch.float16)
        else:
            q_rstd, k_rstd = None, None
            q = q.to(torch.float16)
            k = k.to(torch.float16)
        g_cumsum, q_star, o, (h_kk_final, h_kv_final) = chunk_fwd_mesa_net_fwd(
            q=q, k=k, v=v, g=g, beta=beta, lamb=lamb, cu_seqlens=cu_seqlens,
            max_CG_iteration=max_CG_iteration, chunk_size=chunk_size,
            h_kk_init=h_kk_init, h_kv_init=h_kv_init, output_final_state=
            output_final_state)
        ctx.max_CG_iteration = max_CG_iteration
        ctx.chunk_size = chunk_size
        ctx.cu_seqlens = cu_seqlens
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g_cumsum, beta, lamb,
            h_kk_init, h_kv_init, q_star, o)
        return o, h_kk_final, h_kv_final

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dh_kk_final=None, dh_kv_final=None):
        (q, q_rstd, k, k_rstd, v, g, beta, lamb, h_kk_init, h_kv_init,
            q_star, o) = ctx.saved_tensors
        max_CG_iteration = ctx.max_CG_iteration
        chunk_size = ctx.chunk_size
        cu_seqlens = ctx.cu_seqlens
        dq, dk, dv, dg, dbeta, dlamb, dh0_kk, dh0_kv = chunk_fwd_mesa_net_bwd(q
            =q, k=k, v=v, g=g, beta=beta, lamb=lamb, q_star=q_star, do=do,
            cu_seqlens=cu_seqlens, max_CG_iteration=max_CG_iteration,
            chunk_size=chunk_size, h_kk_init=h_kk_init, h_kv_init=h_kv_init,
            dh_kv_final=dh_kv_final, dh_kk_final=dh_kk_final)
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq, dk, dv.to(v), dg.to(g), dbeta.to(beta), dlamb.to(lamb
            ), None, None, dh0_kk, dh0_kv, None, None
