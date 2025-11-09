# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/fla-org/flash-linear-attention
# Source-Files: fla/ops/path_attn/parallel.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rh2l8zei/flash-linear-attention-main/fla/ops/path_attn/parallel.py
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
from einops import reduce

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


@input_guard
def chunk_global_cumsum(s: torch.Tensor, reverse: bool=False, cu_seqlens:
    Optional[torch.Tensor]=None, scale: float=None, head_first: bool=False,
    output_dtype: Optional[torch.dtype]=torch.float) ->torch.Tensor:
    if cu_seqlens is not None:
        assert s.shape[0
            ] == 1, 'Only batch size 1 is supported when cu_seqlens are provided'
    if len(s.shape) == 3:
        return chunk_global_cumsum_scalar(s=s, reverse=reverse, cu_seqlens=
            cu_seqlens, scale=scale, head_first=head_first, output_dtype=
            output_dtype)
    elif len(s.shape) == 4:
        return chunk_global_cumsum_vector(s=s, reverse=reverse, cu_seqlens=
            cu_seqlens, scale=scale, head_first=head_first, output_dtype=
            output_dtype)
    else:
        raise ValueError(
            f'Unsupported input shape {s.shape}, which should be [B, T, H]/[B, T, H, D] if `head_first=False` or [B, H, T]/[B, H, T, D] otherwise'
            )


@input_guard
def chunk_global_cumsum_scalar(s: torch.Tensor, reverse: bool=False,
    cu_seqlens: Optional[torch.Tensor]=None, scale: float=None, head_first:
    bool=False, output_dtype: Optional[torch.dtype]=torch.float
    ) ->torch.Tensor:
    if head_first:
        B, H, T = s.shape
    else:
        B, T, H = s.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    z = torch.empty_like(s, dtype=output_dtype or s.dtype)
    grid = N * H,
    chunk_global_cumsum_scalar_kernel[grid](s=s, o=z, scale=scale,
        cu_seqlens=cu_seqlens, T=T, B=B, H=H, HEAD_FIRST=head_first,
        REVERSE=reverse)
    return z


@triton.heuristics({'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BT': BT}, num_warps=num_warps,
    num_stages=num_stages) for BT in [32, 64, 128, 256] for num_warps in [2,
    4, 8] for num_stages in [1, 2, 3, 4]], key=['B', 'H', 'IS_VARLEN',
    'REVERSE'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_global_cumsum_scalar_kernel(s, o, scale, cu_seqlens, T, B: tl.
    constexpr, H: tl.constexpr, BT: tl.constexpr, REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr, IS_VARLEN: tl.constexpr, HEAD_FIRST: tl.constexpr
    ):
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
    T = eos - bos
    b_z = tl.zeros([], dtype=tl.float32)
    NT = tl.cdiv(T, BT)
    for i_c in range(NT):
        i_t = NT - 1 - i_c if REVERSE else i_c
        if HEAD_FIRST:
            p_s = tl.make_block_ptr(s + bos * H + i_h * T, (T,), (1,), (i_t *
                BT,), (BT,), (0,))
            p_o = tl.make_block_ptr(o + bos * H + i_h * T, (T,), (1,), (i_t *
                BT,), (BT,), (0,))
        else:
            p_s = tl.make_block_ptr(s + bos * H + i_h, (T,), (H,), (i_t *
                BT,), (BT,), (0,))
            p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t *
                BT,), (BT,), (0,))
        b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
        b_o = tl.cumsum(b_s, axis=0)
        b_ss = tl.sum(b_s, 0)
        if REVERSE:
            b_o = -b_o + b_ss + b_s
        b_o += b_z
        if i_c >= 0:
            b_z += b_ss
        if HAS_SCALE:
            b_o *= scale
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@input_guard
def chunk_global_cumsum_vector(s: torch.Tensor, reverse: bool=False,
    cu_seqlens: Optional[torch.Tensor]=None, scale: float=None, head_first:
    bool=False, output_dtype: Optional[torch.dtype]=torch.float
    ) ->torch.Tensor:
    if head_first:
        B, H, T, S = s.shape
    else:
        B, T, H, S = s.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BS = min(32, triton.next_power_of_2(S))
    z = torch.empty_like(s, dtype=output_dtype or s.dtype)
    grid = triton.cdiv(S, BS), N * H
    chunk_global_cumsum_vector_kernel[grid](s=s, o=z, scale=scale,
        cu_seqlens=cu_seqlens, T=T, B=B, H=H, S=S, BS=BS, HEAD_FIRST=
        head_first, REVERSE=reverse)
    return z


@triton.heuristics({'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BT': BT}, num_warps=num_warps,
    num_stages=num_stages) for BT in [16, 32, 64, 128] for num_warps in [2,
    4, 8] for num_stages in [1, 2, 3, 4]], key=['B', 'H', 'S', 'IS_VARLEN',
    'REVERSE'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_global_cumsum_vector_kernel(s, o, scale, cu_seqlens, T, B: tl.
    constexpr, H: tl.constexpr, S: tl.constexpr, BT: tl.constexpr, BS: tl.
    constexpr, REVERSE: tl.constexpr, HAS_SCALE: tl.constexpr, IS_VARLEN:
    tl.constexpr, HEAD_FIRST: tl.constexpr):
    i_s, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
    T = eos - bos
    o_i = tl.arange(0, BT)
    if REVERSE:
        m_s = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
    else:
        m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
    b_z = tl.zeros([BS], dtype=tl.float32)
    NT = tl.cdiv(T, BT)
    for i_c in range(NT):
        i_t = NT - 1 - i_c if REVERSE else i_c
        if HEAD_FIRST:
            p_s = tl.make_block_ptr(s + (bos * H + i_h * T) * S, (T, S), (S,
                1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
            p_o = tl.make_block_ptr(o + (bos * H + i_h * T) * S, (T, S), (S,
                1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        else:
            p_s = tl.make_block_ptr(s + (bos * H + i_h) * S, (T, S), (H * S,
                1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
            p_o = tl.make_block_ptr(o + (bos * H + i_h) * S, (T, S), (H * S,
                1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=False)
        if HAS_SCALE:
            b_c *= scale
        tl.store(p_o, b_c.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        b_z += tl.sum(b_s, 0)


@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int
    ) ->torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(
        cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens
        )


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) ->torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.heuristics({'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BK': BK}, num_warps=num_warps,
    num_stages=num_stages) for BK in [32, 64, 128] for num_warps in [2, 4, 
    8] for num_stages in [2, 3, 4]], key=['H', 'K', 'BT', 'IS_VARLEN'], **
    autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel(k, g, beta, A, cu_seqlens,
    chunk_indices, T, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr,
    BK: tl.constexpr, IS_VARLEN: tl.constexpr, USE_G: tl.constexpr):
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
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,
        ), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, tl.trans(b_k))
    if USE_G:
        p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,),
            (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A *= exp(b_g_diff)
    b_A *= b_beta[:, None]
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (BT * H, 1),
        (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BK': BK}, num_warps=num_warps,
    num_stages=num_stages) for BK in [32, 64] for num_warps in [1, 2, 4, 8] for
    num_stages in [2, 3, 4]], key=['BC'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel_intra_sub_inter(k, g, beta, A,
    cu_seqlens, chunk_indices, T, H: tl.constexpr, K: tl.constexpr, BT: tl.
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
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    A += (bos * H + i_h) * BT
    p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT +
        i_i * BC,), (BC,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT + i_i * BC,
            i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT + i_i * BC,
            i_k * BK), (BC, BK), (1, 0))
        b_kt = tl.make_block_ptr(k, (K, T), (1, H * K), (i_k * BK, i_t * BT +
            i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(g, (K, T), (1, H * K), (i_k * BK, i_t * BT +
            i_j * BC), (BK, BC), (0, 1))
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_gn = tl.load(g + (i_t * BT + i_i * BC) * H * K + o_k, mask=m_k,
            other=0)
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1)) * exp(b_g - b_gn[None, :])
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kt = tl.load(b_kt, boundary_check=(0, 1)) * exp(b_gn[:, None] - b_gk)
        b_A += tl.dot(b_k, b_kt)
    b_A *= b_beta[:, None]
    p_A = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + i_i * BC, 
        i_j * BC), (BC, BC), (1, 0))
    tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8)], key=['BK', 'BT'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel_intra_sub_intra(k, g, beta, A,
    cu_seqlens, chunk_indices, T, H: tl.constexpr, K: tl.constexpr, BT: tl.
    constexpr, BC: tl.constexpr, BK: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
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
    m_A = i_t * BT + i_i * BC + o_i < T
    o_A = (bos + i_t * BT + i_i * BC + o_i) * H * BT + i_h * BT + i_i * BC
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (
        i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g + (bos * H + i_h) * K, (T, K), (H * K, 1), (
        i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_beta = beta + (bos + i_t * BT + i_i * BC + o_i) * H + i_h
    b_k = tl.load(p_k, boundary_check=(0, 1)) * tl.load(p_beta, mask=m_A,
        other=0)[:, None]
    b_g = tl.load(p_g, boundary_check=(0, 1))
    p_kt = k + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
    p_gk = g + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_kt = tl.load(p_kt, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A = tl.sum(b_k * b_kt[None, :] * exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i > j, b_A, 0.0)
        tl.store(A + o_A + j, b_A, mask=m_A)
        p_kt += H * K
        p_gk += H * K


@triton.heuristics({'USE_G': lambda args: args['g_cumsum'] is not None,
    'IS_VARLEN': lambda args: args['offsets'] is not None})
@triton.jit(do_not_specialize=['T'])
def intra_chunk_preprocess_fwd_kernel(q, k, v, w, beta, g_cumsum, o, A, L,
    M, w2, q_new, k_new, scale, indices, offsets, T, H: tl.constexpr, G: tl
    .constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.
    constexpr, BV: tl.constexpr, BT: tl.constexpr, IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr):
    i_t, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // G
    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices +
            i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets +
            i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    sm_scale = scale * 1.44269504
    A += (bos * H + i_h) * BT
    q += (bos * HQ + i_hq) * K
    q_new += (bos * HQ + i_hq) * K
    k += (bos * H + i_h) * K
    k_new += (bos * H + i_h) * K
    w2 += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    o += (bos * HQ + i_hq) * V
    beta += bos * H + i_h
    if USE_G:
        g_cumsum += bos * HQ + i_hq
    L += bos * HQ + i_hq
    M += bos * HQ + i_hq
    p_q = tl.make_block_ptr(q, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK),
        (1, 0))
    p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_t * BT), (BK, BT),
        (0, 1))
    p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK),
        (1, 0))
    p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, 0), (BT, BV),
        (1, 0))
    p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_T = tl.make_block_ptr(A, (T, BT), (BT * H, 1), (i_t * BT, 0), (BT, BT
        ), (1, 0))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_kt = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_w = tl.load(p_w, boundary_check=(0, 1))
    b_T = tl.load(p_T, boundary_check=(0, 1))
    b_T = b_T * b_beta[None, :]
    o_i = tl.arange(0, BT)
    m_t = o_i[:, None] >= o_i[None, :]
    b_qw = tl.where(m_t, tl.dot(b_q, tl.trans(b_w.to(b_q.dtype))), 0).to(b_q
        .dtype)
    b_qwT = tl.dot(b_qw, b_T.to(b_q.dtype)).to(b_q.dtype)
    b_wbk = tl.where(o_i[:, None] > o_i[None, :], tl.dot(b_w.to(b_q.dtype),
        b_kt), 0).to(b_q.dtype)
    b_A = tl.where(m_t, tl.dot(b_q, b_kt) - tl.dot(b_qwT.to(b_q.dtype),
        b_wbk), 0)
    b_q = b_q.to(tl.float32) - tl.dot(b_qwT, b_w.to(b_q.dtype))
    p_q_new = tl.make_block_ptr(q_new, (T, K), (K * HQ, 1), (i_t * BT, 0),
        (BT, K), (1, 0))
    tl.store(p_q_new, b_q.to(p_q_new.dtype.element_ty), boundary_check=(0, 1))
    if i_hq % G == 0:
        b_Twb = tl.dot(b_T, b_w)
        p_w2 = tl.make_block_ptr(w2, (T, K), (K * H, 1), (i_t * BT, 0), (BT,
            BK), (1, 0))
        tl.store(p_w2, b_Twb.to(p_w2.dtype.element_ty), boundary_check=(0, 1))
        b_T_wbk = tl.dot(b_T.to(b_kt.dtype), b_wbk).to(b_kt.dtype)
        p_k_new = tl.make_block_ptr(k_new, (K, T), (1, K * H), (0, i_t * BT
            ), (BK, BT), (0, 1))
        tl.store(p_k_new, (b_kt - tl.dot(tl.trans(b_w.to(b_kt.dtype)),
            b_T_wbk)).to(p_k_new.dtype.element_ty), boundary_check=(0, 1))
    if USE_G:
        p_g_cumsum = tl.make_block_ptr(g_cumsum, (T,), (HQ,), (i_t * BT,),
            (BT,), (0,))
        b_g_cumsum = tl.load(p_g_cumsum, boundary_check=(0,))
        b_A = b_A + (b_g_cumsum[:, None] - b_g_cumsum[None, :])
        b_A = tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], b_A,
            float('-inf'))
    b_qkT_softmax = tl.where(o_i[:, None] >= o_i[None, :], b_A * sm_scale,
        float('-inf'))
    m_i = tl.max(b_qkT_softmax, 1)
    b_qkT_softmax = tl.math.exp2(b_qkT_softmax - m_i[:, None])
    l_i = tl.sum(b_qkT_softmax, 1)
    b_o = tl.dot(b_qkT_softmax.to(b_v.dtype), b_v)
    p_o = tl.make_block_ptr(o, (T, V), (V * HQ, 1), (i_t * BT, 0), (BT, BV),
        (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    p_l = tl.make_block_ptr(L, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    p_m = tl.make_block_ptr(M, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    tl.store(p_m, m_i.to(p_m.dtype.element_ty), boundary_check=(0,))
    tl.store(p_l, l_i.to(p_l.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({'USE_GATE': lambda args: args['g_cumsum'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=['T'])
def parallel_path_fwd_kernel(q, k, v, o, o_new, g_cumsum, w1, w2, scale, L,
    L_new, M, cu_seqlens, indices, T, G: tl.constexpr, HQ: tl.constexpr, H:
    tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BS:
    tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_GATE: tl.
    constexpr, IS_VARLEN: tl.constexpr):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices +
            i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T
    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1),
        (i_t * BT, 0), (BT, BK), (1, 0))
    b_q = tl.zeros([BT, BK], dtype=tl.float32)
    b_q += tl.load(p_q, boundary_check=(0, 1))
    sm_scale = scale * 1.44269504
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * V, (T, V), (HQ * V, 1),
        (i_t * BT, 0), (BT, BV), (1, 0))
    b_o += tl.load(p_o, boundary_check=(0, 1))
    p_L = tl.make_block_ptr(L + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,),
        (BT,), (0,))
    p_M = tl.make_block_ptr(M + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,),
        (BT,), (0,))
    b_l = tl.load(p_L, boundary_check=(0,))
    b_m = tl.load(p_M, boundary_check=(0,))
    if USE_GATE:
        p_g_cumsum_q = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,),
            (HQ,), (i_t * BT,), (BT,), (0,))
        b_g_cumsum_q = tl.load(p_g_cumsum_q, boundary_check=(0,))
    else:
        b_g_cumsum_q = None
    for offset in range((i_t + 1) * BT - 2 * BS, i_t * BT - BS, -BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, K * H),
            (0, offset), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (V * H, 1),
            (offset, 0), (BS, BV), (1, 0))
        p_w1 = tl.make_block_ptr(w1 + (bos * H + i_h) * K, (K, T), (1, K *
            H), (0, offset), (BK, BS), (0, 1))
        p_w2 = tl.make_block_ptr(w2 + (bos * H + i_h) * K, (T, K), (K * H, 
            1), (offset, 0), (BS, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_w2 = tl.load(p_w2, boundary_check=(0, 1))
        m_s = i_t * BT + tl.arange(0, BT) >= offset + BS
        b_s = tl.dot(b_q.to(b_k.dtype), b_k)
        if USE_GATE:
            p_g_cumsum_k = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq),
                (T,), (HQ,), (offset,), (BS,), (0,))
            b_g_cumsum_k = tl.load(p_g_cumsum_k, boundary_check=(0,))
            b_s = b_s + b_g_cumsum_q[:, None] - b_g_cumsum_k[None, :]
        b_s = tl.where(m_s[:, None], b_s * sm_scale, float('-inf'))
        b_m_new = tl.maximum(b_m, tl.max(b_s, 1))
        alpha = tl.math.exp2(b_m - b_m_new)
        b_s = tl.math.exp2(b_s - b_m_new[:, None])
        b_o *= alpha[:, None]
        b_l = b_l * alpha + tl.sum(b_s, 1)
        b_m = b_m_new
        b_o += tl.dot(b_s.to(b_v.dtype), b_v)
        b_s2 = tl.dot(b_q.to(b_w1.dtype), b_w1)
        b_s2 = tl.where(m_s[:, None], b_s2, 0)
        b_q -= tl.dot(b_s2.to(b_w2.dtype), b_w2)
    tl.debug_barrier()
    for offset in range(i_t * BT - BS, -BS, -BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, K * H),
            (0, offset), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (V * H, 1),
            (offset, 0), (BS, BV), (1, 0))
        p_w1 = tl.make_block_ptr(w1 + (bos * H + i_h) * K, (K, T), (1, K *
            H), (0, offset), (BK, BS), (0, 1))
        p_w2 = tl.make_block_ptr(w2 + (bos * H + i_h) * K, (T, K), (K * H, 
            1), (offset, 0), (BS, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_w2 = tl.load(p_w2, boundary_check=(0, 1))
        b_s = tl.dot(b_q.to(b_k.dtype), b_k)
        if USE_GATE:
            p_g_cumsum_k = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq),
                (T,), (HQ,), (offset,), (BS,), (0,))
            b_g_cumsum_k = tl.load(p_g_cumsum_k, boundary_check=(0,))
            b_s = b_s + b_g_cumsum_q[:, None] - b_g_cumsum_k[None, :]
        b_s = b_s * sm_scale
        b_m_new = tl.maximum(b_m, tl.max(b_s, 1))
        alpha = tl.math.exp2(b_m - b_m_new)
        b_s = tl.math.exp2(b_s - b_m_new[:, None])
        b_o *= alpha[:, None]
        b_l = b_l * alpha + tl.sum(b_s, 1)
        b_m = b_m_new
        b_o += tl.dot(b_s.to(b_v.dtype), b_v)
        b_s2 = tl.dot(b_q.to(b_w1.dtype), b_w1)
        b_q -= tl.dot(b_s2.to(b_w2.dtype), b_w2)
    b_o = b_o / b_l[:, None]
    p_o_new = tl.make_block_ptr(o_new + (bos * HQ + i_hq) * V, (T, V), (HQ *
        V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    tl.store(p_o_new, b_o.to(p_o_new.dtype.element_ty), boundary_check=(0, 1))
    b_l = tl.math.log2(b_l) + b_m
    p_L_new = tl.make_block_ptr(L_new + (bos * HQ + i_hq), (T,), (HQ,), (
        i_t * BT,), (BT,), (0,))
    tl.store(p_L_new, b_l.to(p_L_new.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({'IS_VARLEN': lambda args: args['offsets'] is not None})
@triton.jit(do_not_specialize=['T'])
def parallel_path_fwd_kernel_prepare_k_cache(k, k_new, w1, w2, offsets,
    indices, T, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BK: tl.
    constexpr, IS_VARLEN: tl.constexpr):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices +
            i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets +
            i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T
    k += (bos * H + i_h) * K
    k_new += (bos * H + i_h) * K
    w1 += (bos * H + i_h) * K
    w2 += (bos * H + i_h) * K
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK),
        (1, 0))
    b_k = tl.zeros([BT, BK], dtype=tl.float32)
    b_k += tl.load(p_k, boundary_check=(0, 1))
    for k_block_idx in range(i_t + 1, tl.cdiv(T, BT)):
        p_w1 = tl.make_block_ptr(w1, (T, K), (H * K, 1), (k_block_idx * BT,
            0), (BT, BK), (1, 0))
        p_w2 = tl.make_block_ptr(w2, (T, K), (H * K, 1), (k_block_idx * BT,
            0), (BT, BK), (1, 0))
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_w2 = tl.load(p_w2, boundary_check=(0, 1))
        b_A = tl.dot(b_k.to(b_w2.dtype), tl.trans(b_w2))
        b_k = b_k - tl.dot(b_A.to(b_w1.dtype), b_w1)
    p_k_new = tl.make_block_ptr(k_new, (T, K), (H * K, 1), (i_t * BT, 0), (
        BT, BK), (1, 0))
    tl.store(p_k_new, b_k.to(p_k_new.dtype.element_ty), boundary_check=(0, 1))


def chunk_scaled_dot_kkt_fwd(k: torch.Tensor, g: Optional[torch.Tensor]=
    None, gk: Optional[torch.Tensor]=None, beta: Optional[torch.Tensor]=
    None, cu_seqlens: Optional[torch.LongTensor]=None, chunk_size: int=64,
    output_dtype: torch.dtype=torch.float32) ->torch.Tensor:
    """
    Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        g (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H]`. Default: `None`.
        gk (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H, K]` applied to the key tensor. Default: `None`.
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`

    Returns:
        beta * K * K^T of shape `[B, T, H, BT]` where `BT` is the chunk size.
    """
    B, T, H, K = k.shape
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if gk is None:
        A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)
        chunk_scaled_dot_kkt_fwd_kernel[NT, B * H](k=k, g=g, beta=beta, A=A,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T, H=H, K
            =K, BT=BT)
        return A
    BC = min(16, BT)
    NC = triton.cdiv(BT, BC)
    BK = max(triton.next_power_of_2(K), 16)
    A = torch.zeros(B, T, H, BT, device=k.device, dtype=output_dtype)
    grid = NT, NC * NC, B * H
    chunk_scaled_dot_kkt_fwd_kernel_intra_sub_inter[grid](k=k, g=gk, beta=
        beta, A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T,
        H=H, K=K, BT=BT, BC=BC, NC=NC)
    grid = NT, NC, B * H
    chunk_scaled_dot_kkt_fwd_kernel_intra_sub_intra[grid](k=k, g=gk, beta=
        beta, A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T,
        H=H, K=K, BT=BT, BC=BC, BK=BK)
    return A


def intra_chunk_preprocess_fwd_fn(q, k, v, w, beta, g_cumsum, A, scale, BT,
    cu_seqlens):
    HQ = q.shape[-2]
    B, T, H, K = k.shape
    V = v.shape[-1]
    q_new = torch.empty_like(q, dtype=torch.float32)
    k_new = torch.empty_like(k)
    o = torch.empty(B, T, HQ, V, device=q.device, dtype=torch.float32)
    indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)
    grid = NT, B * HQ
    L = torch.empty(B, T, HQ, dtype=torch.float32, device=q.device)
    M = torch.empty(B, T, HQ, dtype=torch.float32, device=q.device)
    w2 = torch.empty_like(w)
    G = HQ // H
    intra_chunk_preprocess_fwd_kernel[grid](q=q, k=k, v=v, w=w, beta=beta,
        g_cumsum=g_cumsum, o=o, A=A, L=L, M=M, w2=w2, q_new=q_new, k_new=
        k_new, scale=scale, offsets=cu_seqlens, indices=indices, T=T, H=H,
        G=G, HQ=HQ, K=K, V=V, BK=triton.next_power_of_2(K), BV=triton.
        next_power_of_2(V), BT=BT, num_warps=4 if BT == 64 else 2)
    return q_new, k_new, w2, o, L, M


def parallel_path_fwd_fn(q, k, v, o, g_cumsum, w1, w2, scale, L, M,
    cu_seqlens, BT, BS):
    B, T, HQ, K = q.shape
    V = v.shape[-1]
    H = k.shape[-2]
    G = HQ // H
    indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)
    grid = NT, B * HQ
    o_new = torch.empty_like(o, dtype=v.dtype)
    L_new = torch.empty_like(L)
    parallel_path_fwd_kernel[grid](q=q, k=k, v=v, o=o, o_new=o_new, w1=w1,
        w2=w2, g_cumsum=g_cumsum, scale=scale, cu_seqlens=cu_seqlens,
        indices=indices, L=L, L_new=L_new, M=M, T=T, K=K, V=V, BK=triton.
        next_power_of_2(K), BV=triton.next_power_of_2(V), G=G, HQ=HQ, H=H,
        BS=BS, BT=BT, num_warps=8 if BT == 128 and K == 128 else 4)
    return o_new, L_new


def prepare_k_cache_fn(k, w1, w2, cu_seqlens, BS, use_cache=False):
    if not use_cache:
        return None
    else:
        B, T, H, K = k.shape
        k_new = torch.empty_like(k)
        indices = prepare_chunk_indices(cu_seqlens, BS
            ) if cu_seqlens is not None else None
        NT = triton.cdiv(T, BS) if cu_seqlens is None else len(indices)
        grid = NT, B * H
        parallel_path_fwd_kernel_prepare_k_cache[grid](k=k, k_new=k_new, w1
            =w1, w2=w2, offsets=cu_seqlens, indices=indices, H=H, T=T, K=K,
            BT=BS, BK=triton.next_power_of_2(K))
        return k_new


@input_guard
def solve_tril(A: torch.Tensor, cu_seqlens: Optional[torch.Tensor]=None,
    output_dtype: torch.dtype=torch.float) ->torch.Tensor:
    """
    Compute the inverse of the matrix I + A
    A should be strictly lower triangular, i.e., A.triu() == 0.

    Args:
        A (torch.Tensor):
            [B, T, H, BT], where BT should only be 16, 32, or 64.
        cu_seqlens (torch.Tensor):
            The cumulative sequence lengths of the input tensor. Default: `None`.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float`.
            If `None`, the output dtype will be the same as the input dtype.

    Returns:
        (I + A)^-1 with the same shape as A
    """
    assert A.shape[-1] in [16, 32, 64]
    output_dtype = A.dtype if output_dtype is None else output_dtype
    B, T, H, BT = A.shape
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    Ai = torch.zeros_like(A, dtype=output_dtype)
    if BT == 16:
        merge_fn = solve_tril_16x16_kernel
    elif BT == 32:
        merge_fn = merge_16x16_to_32x32_inverse_kernel
    elif BT == 64:
        merge_fn = merge_16x16_to_64x64_inverse_kernel
    merge_fn[NT, B * H](A=A, Ai=Ai, cu_seqlens=cu_seqlens, chunk_indices=
        chunk_indices, T=T, H=H, BT=BT, USE_TMA=is_tma_supported,
        DOT_PRECISION=FLA_TRIL_PRECISION)
    return Ai


# Forward method (kernel launch code)
@input_guard
@autocast_custom_fwd
def _ParallelPATHAttentionFunction_forward(ctx, q, k, v, w, beta, g, scale,
    cu_seqlens, use_cache=False):
    g_cumsum = chunk_global_cumsum(g, cu_seqlens=cu_seqlens, output_dtype=
        torch.float32) if g is not None else None
    BS = 64 if check_shared_mem('hopper') else 32
    BT = 128 if check_shared_mem('ampere') else 64
    A = chunk_scaled_dot_kkt_fwd(k=w, beta=beta, cu_seqlens=cu_seqlens,
        chunk_size=BS, output_dtype=torch.float32)
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=w.dtype)
    q_new, k_new, w2, o, L, M = intra_chunk_preprocess_fwd_fn(q=q, k=k, v=v,
        w=w, beta=beta, g_cumsum=g_cumsum, A=A, scale=scale, BT=BS,
        cu_seqlens=cu_seqlens)
    w_fp16 = w.to(torch.float16)
    w2_fp16 = w2.to(torch.float16)
    o, L = parallel_path_fwd_fn(q=q_new, k=k_new, v=v, L=L, w1=w_fp16, w2=
        w2_fp16, M=M, o=o, g_cumsum=g_cumsum, scale=scale, cu_seqlens=
        cu_seqlens, BT=BT, BS=BS)
    k_cache = prepare_k_cache_fn(k=k_new, w1=w, w2=w2, cu_seqlens=
        cu_seqlens, BS=BS, use_cache=use_cache)
    ctx.save_for_backward(q, k, v, w, g_cumsum, o, beta, L, A)
    ctx.scale = scale
    ctx.cu_seqlens = cu_seqlens
    return o, k_cache


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def parallel_attn_bwd_kernel_preprocess(o, do, delta, B: tl.constexpr, V:
    tl.constexpr):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < V
    b_o = tl.load(o + i_n * V + o_d, mask=m_d, other=0)
    b_do = tl.load(do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    b_delta = tl.sum(b_o * b_do)
    tl.store(delta + i_n, b_delta.to(delta.dtype.element_ty))


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=['T'])
def chunk_cumprod_householder_bwd_kernel(hc_suffix, dhc_whole, k, dk, w1,
    w2, dw1, dw2, dk_new, cu_seqlens, split_indices, chunk_offsets,
    split_offsets, BT: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, T:
    tl.constexpr, S: tl.constexpr, G: tl.constexpr, H: tl.constexpr, HQ: tl
    .constexpr, IS_VARLEN: tl.constexpr):
    i_ss, i_hq = tl.program_id(0), tl.program_id(1)
    i_h = i_hq // G
    if IS_VARLEN:
        i_n, i_s = tl.load(split_indices + i_ss * 2).to(tl.int32), tl.load(
            split_indices + i_ss * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NS = tl.cdiv(T, S)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
        boh_large = tl.load(split_offsets + i_n).to(tl.int32)
    else:
        NS = tl.cdiv(T, S)
        i_n, i_s = i_ss // NS, i_ss % NS
        bos, eos = i_n * T, i_n * T + T
        boh = i_n * tl.cdiv(T, BT)
        boh_large = i_n * tl.cdiv(T, S)
    dhc_whole += ((boh_large + i_s) * HQ + i_hq) * K * K
    hc_suffix += ((boh + tl.cdiv(i_s * S, BT)) * H + i_h) * K * K
    k += (bos * H + i_h) * K
    w1 += (bos * H + i_h) * K
    w2 += (bos * H + i_h) * K
    dw1 += (bos * HQ + i_hq) * K
    dw2 += (bos * HQ + i_hq) * K
    dk += (bos * HQ + i_hq) * K
    dk_new += (bos * HQ + i_hq) * K
    stride_h = H * K * K
    NT_small = tl.cdiv(min(S, T - i_s * S), BT)
    p_dhc_whole = tl.make_block_ptr(dhc_whole, (K, K), (K, 1), (0, 0), (BK,
        BK), (1, 0))
    b_dhc = tl.zeros([BK, BK], dtype=tl.float32)
    b_dhc += tl.load(p_dhc_whole, boundary_check=(0, 1))
    for i_t_small in range(0, NT_small):
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_s * S + i_t_small *
            BT, 0), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk, (T, K), (HQ * K, 1), (i_s * S + 
            i_t_small * BT, 0), (BT, BK), (1, 0))
        p_dk_new = tl.make_block_ptr(dk_new, (T, K), (HQ * K, 1), (i_s * S +
            i_t_small * BT, 0), (BT, BK), (1, 0))
        p_hc = tl.make_block_ptr(hc_suffix + i_t_small * stride_h, (K, K),
            (K, 1), (0, 0), (BK, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_dk = tl.load(p_dk, boundary_check=(0, 1))
        p_w1 = tl.make_block_ptr(w1, (T, K), (H * K, 1), (i_s * S + 
            i_t_small * BT, 0), (BT, BK), (1, 0))
        p_w2 = tl.make_block_ptr(w2, (T, K), (H * K, 1), (i_s * S + 
            i_t_small * BT, 0), (BT, BK), (1, 0))
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_w2 = tl.load(p_w2, boundary_check=(0, 1))
        b_hc = tl.load(p_hc, boundary_check=(0, 1))
        b_dk_new = b_dk - tl.dot(b_dk.to(b_hc.dtype), b_hc)
        tl.store(p_dk_new, b_dk_new.to(dk_new.dtype.element_ty),
            boundary_check=(0, 1))
        b_dh = b_dhc - tl.dot(tl.trans(b_hc), b_dhc.to(b_hc.dtype))
        b_dw2 = tl.dot(b_w1, b_dh.to(b_w1.dtype))
        b_dw1 = tl.dot(b_w2, tl.trans(b_dh.to(b_w2.dtype)))
        p_dw1 = tl.make_block_ptr(dw1, (T, K), (HQ * K, 1), (i_s * S + 
            i_t_small * BT, 0), (BT, BK), (1, 0))
        p_dw2 = tl.make_block_ptr(dw2, (T, K), (HQ * K, 1), (i_s * S + 
            i_t_small * BT, 0), (BT, BK), (1, 0))
        tl.store(p_dw1, b_dw1.to(dw1.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dw2, b_dw2.to(dw2.dtype.element_ty), boundary_check=(0, 1))
        b_dhc = b_dhc - tl.dot(tl.dot(b_dhc.to(b_w2.dtype), tl.trans(b_w2))
            .to(b_w1.dtype), b_w1)
        b_dhc -= tl.dot(tl.trans(b_dk).to(b_k.dtype), b_k)


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit
def chunk_cumprod_householder_fwd_kernel(k, k_new, w1, w2, hc_suffix,
    hc_whole, cu_seqlens, split_indices, chunk_offsets, split_offsets, BT:
    tl.constexpr, K: tl.constexpr, H: tl.constexpr, BK: tl.constexpr, T: tl
    .constexpr, S: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_ss, i_h = tl.program_id(0), tl.program_id(1)
    if IS_VARLEN:
        i_n, i_s = tl.load(split_indices + i_ss * 2).to(tl.int32), tl.load(
            split_indices + i_ss * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NS = tl.cdiv(T, S)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
        boh_large = tl.load(split_offsets + i_n).to(tl.int32)
    else:
        NS = tl.cdiv(T, S)
        i_n, i_s = i_ss // NS, i_ss % NS
        bos, eos = i_n * T, i_n * T + T
        boh = i_n * tl.cdiv(T, BT)
        boh_large = i_n * tl.cdiv(T, S)
    NT_small = tl.cdiv(min(S, T - i_s * S), BT)
    stride_h = H * K * K
    hc_whole += ((boh_large + i_s) * H + i_h) * K * K
    hc_suffix += ((boh + tl.cdiv(i_s * S, BT)) * H + i_h) * K * K
    k += (bos * H + i_h) * K
    k_new += (bos * H + i_h) * K
    w1 += (bos * H + i_h) * K
    w2 += (bos * H + i_h) * K
    b_h = tl.zeros([BK, BK], dtype=tl.float32)
    for i_t_small in range(NT_small - 1, -1, -1):
        p_hc_suffix = tl.make_block_ptr(hc_suffix + i_t_small * stride_h, (
            K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
        tl.store(p_hc_suffix, b_h.to(hc_suffix.dtype.element_ty),
            boundary_check=(0, 1))
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_s * S + i_t_small *
            BT, 0), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_k = b_k - tl.dot(b_k, tl.trans(b_h.to(b_k.dtype)))
        p_w1 = tl.make_block_ptr(w1, (K, T), (1, H * K), (0, i_s * S + 
            i_t_small * BT), (BK, BT), (0, 1))
        p_w2 = tl.make_block_ptr(w2, (T, K), (H * K, 1), (i_s * S + 
            i_t_small * BT, 0), (BT, BK), (1, 0))
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_w2 = tl.load(p_w2, boundary_check=(0, 1))
        b_v_new = (b_w1 - tl.dot(b_h.to(b_w1.dtype), b_w1)).to(b_w2.dtype)
        b_h += tl.dot(b_v_new, b_w2)
        p_k_new = tl.make_block_ptr(k_new, (T, K), (H * K, 1), (i_s * S + 
            i_t_small * BT, 0), (BT, BK), (1, 0))
        tl.store(p_k_new, b_k.to(k_new.dtype.element_ty), boundary_check=(0, 1)
            )
    p_hc_whole = tl.make_block_ptr(hc_whole, (K, K), (K, 1), (0, 0), (BK,
        BK), (1, 0))
    tl.store(p_hc_whole, b_h.to(hc_whole.dtype.element_ty), boundary_check=
        (0, 1))


@triton.heuristics({'IS_VARLEN': lambda args: args['offsets'] is not None})
@triton.jit(do_not_specialize=['T'])
def intra_chunk_preprocess_bwd_kernel(q, k, w, w2, beta, AT, dA_local, dq,
    dq_new, dk, dk_new, dw, dbeta, dw1, dw2, T, offsets, indices, HQ: tl.
    constexpr, G: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BT: tl.
    constexpr, BK: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_t, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // G
    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices +
            i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets +
            i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw_beta = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32)
    b_dT = tl.zeros([BT, BT], dtype=tl.float32)
    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (K * HQ, 1),
        (i_t * BT, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (K * H, 1), (
        i_t * BT, 0), (BT, BK), (1, 0))
    p_w = tl.make_block_ptr(w + (bos * H + i_h) * K, (T, K), (K * H, 1), (
        i_t * BT, 0), (BT, BK), (1, 0))
    p_w2 = tl.make_block_ptr(w2 + (bos * H + i_h) * K, (T, K), (K * H, 1),
        (i_t * BT, 0), (BT, BK), (1, 0))
    p_beta = tl.make_block_ptr(beta + (bos * H + i_h), (T,), (H,), (i_t *
        BT,), (BT,), (0,))
    p_T = tl.make_block_ptr(AT + (bos * H + i_h) * BT, (T, BT), (BT * H, 1),
        (i_t * BT, 0), (BT, BT), (1, 0))
    b_w = tl.load(p_w, boundary_check=(0, 1))
    b_Twb = tl.load(p_w2, boundary_check=(0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_T = tl.load(p_T, boundary_check=(0, 1))
    b_w_beta = (b_w * b_beta[:, None]).to(b_w.dtype)
    o_i = tl.arange(0, BT)
    b_qw = tl.where(o_i[:, None] >= o_i[None, :], tl.dot(b_q, tl.trans(b_w)), 0
        ).to(b_q.dtype)
    b_wbk = tl.where(o_i[:, None] > o_i[None, :], tl.dot(b_w_beta, tl.trans
        (b_k)), 0).to(b_k.dtype)
    b_Twbk = tl.dot(b_T, b_wbk).to(b_w.dtype)
    p_dA_local = tl.make_block_ptr(dA_local + (bos * HQ + i_hq) * BT, (T,
        BT), (BT * HQ, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_dA_local = tl.load(p_dA_local, boundary_check=(0, 1))
    p_dq = tl.make_block_ptr(dq + (bos * HQ + i_hq) * K, (T, K), (K * HQ, 1
        ), (i_t * BT, 0), (BT, BK), (1, 0))
    b_dq = tl.load(p_dq, boundary_check=(0, 1))
    p_dw1 = tl.make_block_ptr(dw1 + (bos * HQ + i_hq) * K, (T, K), (K * HQ,
        1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_dw += tl.load(p_dw1, boundary_check=(0, 1))
    b_dqw = -tl.dot(b_dA_local, tl.trans(b_Twbk)) - tl.dot(b_dq.to(b_Twb.
        dtype), tl.trans(b_Twb))
    p_dw2 = tl.make_block_ptr(dw2 + (bos * HQ + i_hq) * K, (T, K), (K * HQ,
        1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_dTwb = -tl.dot(tl.trans(b_qw), b_dq) + tl.load(p_dw2, boundary_check=
        (0, 1))
    b_dT += tl.dot(b_dTwb.to(b_w_beta.dtype), tl.trans(b_w_beta))
    b_dw_beta += tl.dot(tl.trans(b_T), b_dTwb.to(b_T.dtype))
    b_dqw = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :],
        b_dqw, 0)
    b_dq += tl.dot(b_dA_local.to(b_k.dtype), b_k)
    b_dq += tl.dot(b_dqw.to(b_w.dtype), b_w)
    b_dw += tl.dot(tl.trans(b_dqw.to(b_q.dtype)), b_q)
    p_q_new = tl.make_block_ptr(dq_new + (bos * HQ + i_hq) * K, (T, K), (K *
        HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_q_new, b_dq.to(dq_new.dtype.element_ty), boundary_check=(0, 1))
    p_dk = tl.make_block_ptr(dk + (bos * HQ + i_hq) * K, (T, K), (K * HQ, 1
        ), (i_t * BT, 0), (BT, BK), (1, 0))
    b_dk = tl.load(p_dk, boundary_check=(0, 1))
    b_dTwbk = -tl.dot(tl.trans(b_qw), b_dA_local.to(b_qw.dtype)) - tl.dot(b_w,
        tl.trans(b_dk.to(b_w.dtype)))
    b_dw -= tl.dot(b_Twbk, b_dk.to(b_w.dtype))
    b_dT += tl.dot(b_dTwbk.to(b_wbk.dtype), tl.trans(b_wbk))
    b_dwbk = tl.where(o_i[:, None] > o_i[None, :], tl.dot(tl.trans(b_T),
        b_dTwbk.to(b_T.dtype)), 0).to(b_w.dtype)
    b_dw_beta += tl.dot(b_dwbk, b_k)
    b_dk += tl.dot(tl.trans(b_dwbk), b_w_beta)
    b_dk += tl.dot(tl.trans(b_dA_local), b_q)
    p_dk_new = tl.make_block_ptr(dk_new + (bos * HQ + i_hq) * K, (T, K), (K *
        HQ, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_dk_new, b_dk.to(dk_new.dtype.element_ty), boundary_check=(0, 1))
    p_T = tl.make_block_ptr(AT + (bos * H + i_h) * BT, (BT, T), (1, BT * H),
        (0, i_t * BT), (BT, BT), (0, 1))
    b_Tt = tl.load(p_T, boundary_check=(0, 1))
    b_dT = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :],
        b_dT, 0).to(b_w.dtype)
    b_dT = tl.dot(b_Tt, b_dT).to(b_w.dtype)
    b_dT = tl.dot(b_dT, b_Tt)
    b_dT = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], 
        -b_dT, 0).to(b_k.dtype)
    b_dw_beta += tl.dot(b_dT, b_w)
    b_dw += tl.dot(tl.trans(b_dT), b_w_beta)
    b_dw += b_dw_beta * b_beta[:, None]
    b_dbeta = tl.sum(b_dw_beta * b_w, axis=1)
    p_dw = tl.make_block_ptr(dw + (bos * HQ + i_hq) * K, (T, K), (K * HQ, 1
        ), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_dw, b_dw.to(dw.dtype.element_ty), boundary_check=(0, 1))
    p_dbeta = tl.make_block_ptr(dbeta + (bos * HQ + i_hq), (T,), (HQ,), (
        i_t * BT,), (BT,), (0,))
    tl.store(p_dbeta, b_dbeta.to(dbeta.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({'USE_GATE': lambda args: args['g_cumsum'] is not None,
    'IS_VARLEN': lambda args: args['offsets'] is not None})
@triton.jit(do_not_specialize=['T'])
def chunk_transform_qk_bwd_kernel_prepare(q, k, v, w, beta, g_cumsum, L, D,
    h, q_new, k_new, AT, dA_local, dv, do, dg_cumsum, scale, indices,
    offsets, chunk_offsets, T, G: tl.constexpr, HQ: tl.constexpr, H: tl.
    constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.
    constexpr, BT: tl.constexpr, IS_VARLEN: tl.constexpr, USE_GATE: tl.
    constexpr, RETURN_H: tl.constexpr):
    i_t, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // G
    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices +
            i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets +
            i_n + 1).to(tl.int32)
        T = eos - bos
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT
    sm_scale = scale * 1.44269504
    dA_local += (bos * HQ + i_hq) * BT
    AT += (bos * H + i_h) * BT
    q += (bos * HQ + i_hq) * K
    q_new += (bos * HQ + i_hq) * K
    k += (bos * H + i_h) * K
    k_new += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    do += (bos * HQ + i_hq) * V
    dv += (bos * HQ + i_hq) * V
    beta += bos * H + i_h
    if RETURN_H:
        h += ((boh + i_t) * H + i_h) * K * K
    else:
        h += (bos * H + i_h) * K
    if USE_GATE:
        g_cumsum += bos * HQ + i_hq
        dg_cumsum += bos * HQ + i_hq
    L += bos * HQ + i_hq
    D += bos * HQ + i_hq
    p_q = tl.make_block_ptr(q, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK),
        (1, 0))
    p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_t * BT), (BK, BT),
        (0, 1))
    p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK),
        (1, 0))
    p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_kt = tl.load(p_k, boundary_check=(0, 1))
    b_w = tl.load(p_w, boundary_check=(0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    p_T = tl.make_block_ptr(AT, (T, BT), (BT * H, 1), (i_t * BT, 0), (BT,
        BT), (1, 0))
    b_T = tl.load(p_T, boundary_check=(0, 1)) * b_beta[None, :]
    o_i = tl.arange(0, BT)
    m_t = o_i[:, None] >= o_i[None, :]
    b_qw = tl.where(m_t, tl.dot(b_q, tl.trans(b_w.to(b_q.dtype))), 0).to(b_q
        .dtype)
    b_qwT = tl.dot(b_qw, b_T.to(b_q.dtype)).to(b_q.dtype)
    b_wbk = tl.where(o_i[:, None] > o_i[None, :], tl.dot(b_w.to(b_kt.dtype),
        b_kt), 0).to(b_q.dtype)
    b_A = tl.where(m_t, tl.dot(b_q, b_kt) - tl.dot(b_qwT, b_wbk), 0)
    b_q = b_q.to(tl.float32) - tl.dot(b_qwT, b_w.to(b_qwT.dtype))
    p_q_new = tl.make_block_ptr(q_new, (T, K), (K * HQ, 1), (i_t * BT, 0),
        (BT, K), (1, 0))
    tl.store(p_q_new, b_q.to(p_q_new.dtype.element_ty), boundary_check=(0, 1))
    if i_hq % G == 0:
        b_Twb = tl.dot(b_T, b_w)
        p_h = tl.make_block_ptr(h, (T, K), (K * H, 1), (i_t * BT, 0), (BT,
            BK), (1, 0))
        tl.store(p_h, b_Twb.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        b_T_wbk = tl.dot(b_T.to(b_wbk.dtype), b_wbk).to(b_kt.dtype)
        p_k_new = tl.make_block_ptr(k_new, (K, T), (1, K * H), (0, i_t * BT
            ), (BK, BT), (0, 1))
        tl.store(p_k_new, (b_kt - tl.dot(tl.trans(b_w.to(b_kt.dtype)),
            b_T_wbk)).to(p_k_new.dtype.element_ty), boundary_check=(0, 1))
    if USE_GATE:
        p_g_cumsum = tl.make_block_ptr(g_cumsum, (T,), (HQ,), (i_t * BT,),
            (BT,), (0,))
        b_g_cumsum = tl.load(p_g_cumsum, boundary_check=(0,))
        b_A = b_A + (b_g_cumsum[:, None] - b_g_cumsum[None, :])
        b_A = tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], b_A,
            float('-inf'))
    p_l = tl.make_block_ptr(L, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    b_l = tl.load(p_l, boundary_check=(0,))
    p_delta = tl.make_block_ptr(D, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    delta = tl.load(p_delta, boundary_check=(0,))
    b_A_softmax = tl.exp2(tl.where(o_i[:, None] >= o_i[None, :], b_A *
        sm_scale - b_l[:, None], float('-inf')))
    p_do = tl.make_block_ptr(do, (T, V), (HQ * V, 1), (i_t * BT, 0), (BT,
        BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dv = tl.dot(tl.trans(b_A_softmax.to(b_do.dtype)), b_do)
    p_dv = tl.make_block_ptr(dv, (T, V), (HQ * V, 1), (i_t * BT, 0), (BT,
        BV), (1, 0))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    p_v = tl.make_block_ptr(v, (V, T), (1, H * V), (0, i_t * BT), (BV, BT),
        (0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dp = tl.dot(b_do, b_v)
    b_dA = (b_dp - delta[:, None]) * b_A_softmax * scale
    if USE_GATE:
        b_dgq = tl.sum(b_dA, axis=1) - tl.sum(b_dA, axis=0)
        p_dg = tl.make_block_ptr(dg_cumsum, (T,), (HQ,), (i_t * BT,), (BT,),
            (0,))
        tl.store(p_dg, b_dgq.to(p_dg.dtype.element_ty), boundary_check=(0,))
    p_dA = tl.make_block_ptr(dA_local, (T, BT), (BT * HQ, 1), (i_t * BT, 0),
        (BT, BT), (1, 0))
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({'USE_GATE': lambda args: args['g_cumsum'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=['T'])
def parallel_path_bwd_dkv_kernel(q, k, v, g_cumsum, hc_whole, scale, L, D,
    dk, dv, do, dg_cumsum, cu_seqlens, indices, split_offsets, T, G: tl.
    constexpr, HQ: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.
    constexpr, BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl
    .constexpr, S: tl.constexpr, IS_VARLEN: tl.constexpr, USE_GATE: tl.
    constexpr, NUM_BLOCKS: tl.constexpr):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices +
            i_t * 2 + 1).to(tl.int32)
        boh_large = tl.load(split_offsets + i_n).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T
        boh_large = i_n * tl.cdiv(T, S)
    do += (bos * HQ + i_hq) * V
    dk += (bos * HQ + i_hq) * K
    dv += (bos * HQ + i_hq) * K
    L += bos * HQ + i_hq
    D += bos * HQ + i_hq
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    hc_whole += (boh_large * H + i_h) * K * K
    if USE_GATE:
        g_cumsum += bos * HQ + i_hq
        dg_cumsum += bos * HQ + i_hq
    sm_scale = scale * 1.44269504
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK),
        (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    p_v = tl.make_block_ptr(v, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK),
        (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    if USE_GATE:
        b_g_cumsum_k = tl.zeros([BT], dtype=tl.float32)
        p_g_cumsum_k = tl.make_block_ptr(g_cumsum, (T,), (HQ,), (i_t * BT,),
            (BT,), (0,))
        b_g_cumsum_k += tl.load(p_g_cumsum_k, boundary_check=(0,))
        b_dg_cumsum_k = tl.zeros([BT], dtype=tl.float32)
    else:
        b_g_cumsum_k = None
        b_dg_cumsum_k = None
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dv = tl.zeros([BT, BK], dtype=tl.float32)
    last_chunk_start = tl.floor(i_t * BT / S).to(tl.int32) * S
    idx_j = (tl.floor(i_t * BT / S).to(tl.int32) + 1).to(tl.int32)
    last_chunk_end = tl.ceil(T / BS).to(tl.int32) * BS - BS
    for offset in range(last_chunk_end, last_chunk_start + S - BS, -BS):
        p_delta = tl.make_block_ptr(D, (T,), (HQ,), (offset,), (BS,), (0,))
        p_l = tl.make_block_ptr(L, (T,), (HQ,), (offset,), (BS,), (0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        b_l = tl.load(p_l, boundary_check=(0,))
        p_q = tl.make_block_ptr(q + ((bos * NUM_BLOCKS + idx_j) * HQ + i_hq
            ) * K, (T, K), (HQ * K * NUM_BLOCKS, 1), (offset, 0), (BS, BK),
            (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_A = tl.dot(b_k, tl.trans(b_q).to(b_k.dtype))
        if USE_GATE:
            p_g_cumsum_q = tl.make_block_ptr(g_cumsum, (T,), (HQ,), (offset
                ,), (BS,), (0,))
            b_g_cumsum_q = tl.load(p_g_cumsum_q, boundary_check=(0,))
            b_A = b_A + b_g_cumsum_q[None, :] - b_g_cumsum_k[:, None]
            b_A = tl.where((offset + tl.arange(0, BS) < T)[None, :], b_A,
                float('-inf'))
        b_A_softmax = tl.math.exp2(b_A * sm_scale - b_l[None, :])
        p_do = tl.make_block_ptr(do, (T, V), (HQ * V, 1), (offset, 0), (BS,
            BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dv += tl.dot(b_A_softmax.to(b_do.dtype), b_do)
        b_dp = tl.dot(b_v, tl.trans(b_do))
        b_dA = (b_dp - b_delta[None, :]) * b_A_softmax * scale
        if USE_GATE:
            b_dg_cumsum_k -= tl.sum(b_dA, axis=1)
        b_dk += tl.dot(b_dA.to(b_q.dtype), b_q)
    p_dk = tl.make_block_ptr(dk, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT,
        BK), (1, 0))
    tl.store(p_dk, b_dk.to(dk.dtype.element_ty), boundary_check=(0, 1))
    mask = i_t * BT + tl.arange(0, BT) < T
    tl.atomic_add(dv + (i_t * BT + tl.arange(0, BT))[:, None] * HQ * K + tl
        .arange(0, BK)[None, :], b_dv, mask=mask[:, None], sem='relaxed')
    if USE_GATE:
        tl.atomic_add(dg_cumsum + (i_t * BT + tl.arange(0, BT)) * HQ,
            b_dg_cumsum_k, mask=mask, sem='relaxed')


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not
    None, 'USE_GATE': lambda args: args['g_cumsum'] is not None})
@triton.jit(do_not_specialize=['T'])
def parallel_path_bwd_dq_kernel(q, k, v, g_cumsum, hc_whole, scale, L, D,
    dq, do, dhc_whole, dg_cumsum, cu_seqlens, indices, split_offsets, T, G:
    tl.constexpr, HQ: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl
    .constexpr, BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV:
    tl.constexpr, S: tl.constexpr, NUM_BLOCKS: tl.constexpr, IS_VARLEN: tl.
    constexpr, USE_GATE: tl.constexpr):
    i_t, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // G
    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices +
            i_t * 2 + 1).to(tl.int32)
        boh_large = tl.load(split_offsets + i_n).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        boh_large = i_n * tl.cdiv(T, S)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    do += (bos * HQ + i_hq) * V
    dq += (bos * HQ + i_hq) * K
    hc_whole += (boh_large * H + i_h) * K * K
    dhc_whole += (boh_large * HQ + i_hq) * K * K
    L += bos * HQ + i_hq
    D += bos * HQ + i_hq
    if USE_GATE:
        g_cumsum += bos * HQ + i_hq
        dg_cumsum += bos * HQ + i_hq
    stride_h = H * K * K
    stride_hq = HQ * K * K
    sm_scale = scale * 1.44269504
    p_do = tl.make_block_ptr(do, (T, V), (HQ * V, 1), (i_t * BT, 0), (BT,
        BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    p_l = tl.make_block_ptr(L, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    p_d = tl.make_block_ptr(D, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    b_l = tl.load(p_l, boundary_check=(0,))
    b_delta = tl.load(p_d, boundary_check=(0,))
    if USE_GATE:
        p_g_cumsum_q = tl.make_block_ptr(g_cumsum, (T,), (HQ,), (i_t * BT,),
            (BT,), (0,))
        b_g_cumsum_q = tl.load(p_g_cumsum_q, boundary_check=(0,)).to(tl.float32
            )
        b_dg_cumsum_q = tl.zeros([BT], dtype=tl.float32)
    else:
        b_g_cumsum_q = None
        b_dg_cumsum_q = None
    curr_end = (i_t * BT // S * S).to(tl.int32)
    b_dq = tl.zeros([BT, K], dtype=tl.float32)
    for offset_outer in range(0, curr_end, S):
        idx_j = offset_outer // S
        p_q = tl.make_block_ptr(q + ((bos * NUM_BLOCKS + idx_j + 1) * HQ +
            i_hq) * K, (T, K), (HQ * K * NUM_BLOCKS, 1), (i_t * BT, 0), (BT,
            BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_dh = -tl.dot(tl.trans(b_q), b_dq.to(b_q.dtype))
        tl.atomic_add(dhc_whole + idx_j * stride_hq + tl.arange(0, K)[:,
            None] * K + tl.arange(0, K)[None, :], b_dh, sem='relaxed')
        p_h = tl.make_block_ptr(hc_whole + idx_j * stride_h, (K, K), (K, 1),
            (0, 0), (BK, BK), (1, 0))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dq = b_dq - tl.dot(b_dq.to(b_h.dtype), tl.trans(b_h))
        for offset in range(offset_outer, min(offset_outer + S, i_t * BT), BS):
            p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (offset, 0), (BS,
                BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_A = tl.dot(b_q, tl.trans(b_k).to(b_q.dtype))
            if USE_GATE:
                p_g_cumsum_k = tl.make_block_ptr(g_cumsum, (T,), (HQ,), (
                    offset,), (BS,), (0,))
                b_g_cumsum_k = tl.load(p_g_cumsum_k, boundary_check=(0,)).to(tl
                    .float32)
                b_A = b_A + b_g_cumsum_q[:, None] - b_g_cumsum_k[None, :]
            b_A = exp2(b_A * sm_scale - b_l[:, None])
            b_A = tl.where(m_t[:, None], b_A, 0)
            p_v = tl.make_block_ptr(v, (V, T), (1, V * H), (0, offset), (BK,
                BS), (0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_dp = tl.dot(b_do, b_v.to(b_do.dtype))
            b_dA = (b_dp - b_delta[:, None]) * b_A * scale
            b_dq += tl.dot(b_dA.to(b_k.dtype), b_k)
            if USE_GATE:
                b_dg_cumsum_q += tl.sum(b_dA, axis=1)
    p_dq = tl.make_block_ptr(dq, (T, K), (K * HQ, 1), (i_t * BT, 0), (BT,
        BK), (1, 0))
    tl.store(p_dq, b_dq.to(dq.dtype.element_ty), boundary_check=(0, 1))
    if USE_GATE:
        tl.atomic_add(dg_cumsum + o_t * HQ, b_dg_cumsum_q, mask=m_t, sem=
            'relaxed')


@triton.heuristics({'IS_VARLEN': lambda args: args['offsets'] is not None,
    'USE_GATE': lambda args: args['g_cumsum'] is not None})
@triton.jit(do_not_specialize=['T'])
def parallel_path_bwd_intra_chunk_kernel(q, k, v, g_cumsum, w1, w2, L, D,
    dq, dq_new, dk, dv, dw1, dw2, do, dg_cumsum, offsets, indices, T, scale,
    G: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V:
    tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BT: tl.constexpr, S:
    tl.constexpr, IS_VARLEN: tl.constexpr, USE_GATE: tl.constexpr):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices +
            i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets +
            i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    w1 += (bos * H + i_h) * K
    w2 += (bos * H + i_h) * K
    q += (bos * HQ + i_hq) * K
    dq += (bos * HQ + i_hq) * K
    dq_new += (bos * HQ + i_hq) * K
    dk += (bos * HQ + i_hq) * K
    dv += (bos * HQ + i_hq) * V
    do += (bos * HQ + i_hq) * V
    dw1 += (bos * HQ + i_hq) * K
    dw2 += (bos * HQ + i_hq) * K
    L += bos * HQ + i_hq
    D += bos * HQ + i_hq
    if USE_GATE:
        g_cumsum += bos * HQ + i_hq
        dg_cumsum += bos * HQ + i_hq
    sm_scale = scale * 1.44269504
    p_do = tl.make_block_ptr(do, (T, V), (HQ * V, 1), (i_t * BT, 0), (BT,
        BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    p_delta = tl.make_block_ptr(D, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))
    p_l = tl.make_block_ptr(L, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    b_l = tl.load(p_l, boundary_check=(0,))
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    p_dq = tl.make_block_ptr(dq, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT,
        BK), (1, 0))
    b_dq += tl.load(p_dq, boundary_check=(0, 1))
    p_q = tl.make_block_ptr(q, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK),
        (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    if USE_GATE:
        p_gq_cumsum = tl.make_block_ptr(g_cumsum, (T,), (HQ,), (i_t * BT,),
            (BT,), (0,))
        b_gq_cumsum = tl.load(p_gq_cumsum, boundary_check=(0,))
        b_dgq = tl.zeros([BT], dtype=tl.float32)
    else:
        b_dgq = None
    curr_start = (tl.floor(i_t * BT / S).to(tl.int32) * S).to(tl.int32)
    for offset in range(curr_start, i_t * BT, BT):
        mask = offset + tl.arange(0, BT) < T
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (offset, 0), (BT, BK
            ), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_q_tmp = tl.zeros([BT, BK], dtype=tl.float32)
        b_q_tmp += b_q
        for i_t_small in range(i_t * BT - BT, offset, -BT):
            p_w1 = tl.make_block_ptr(w1, (T, K), (H * K, 1), (i_t_small, 0),
                (BT, BK), (1, 0))
            b_w1 = tl.load(p_w1, boundary_check=(0, 1))
            p_w2 = tl.make_block_ptr(w2, (T, K), (H * K, 1), (i_t_small, 0),
                (BT, BK), (1, 0))
            b_w2 = tl.load(p_w2, boundary_check=(0, 1))
            b_A_tmp = tl.dot(b_q_tmp.to(b_w1.dtype), tl.trans(b_w1))
            b_q_tmp -= tl.dot(b_A_tmp.to(b_w1.dtype), b_w2)
        b_q2 = b_q_tmp.to(b_k.dtype)
        b_A = tl.dot(b_q2, tl.trans(b_k))
        if USE_GATE:
            p_gk_cumsum = tl.make_block_ptr(g_cumsum, (T,), (HQ,), (offset,
                ), (BT,), (0,))
            b_gk_cumsum = tl.load(p_gk_cumsum, boundary_check=(0,))
            b_A = b_A + b_gq_cumsum[:, None] - b_gk_cumsum[None, :]
            b_A = tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], b_A,
                float('-inf'))
        b_A_softmax = tl.math.exp2(b_A * sm_scale - b_l[:, None])
        b_dv = tl.dot(tl.trans(b_A_softmax.to(b_do.dtype)), b_do)
        tl.atomic_add(dv + ((offset + tl.arange(0, BT)) * HQ * V)[:, None] +
            tl.arange(0, BV)[None, :], b_dv.to(dv.dtype.element_ty), mask=
            mask[:, None], sem='relaxed')
        p_v = tl.make_block_ptr(v, (T, V), (V * H, 1), (offset, 0), (BT, BV
            ), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_dp = tl.dot(b_do, tl.trans(b_v))
        b_dA = (b_dp - b_delta[:, None]) * b_A_softmax * scale
        if USE_GATE:
            b_dgk = -tl.sum(b_dA, axis=0)
            tl.atomic_add(dg_cumsum + (offset + tl.arange(0, BT)) * HQ,
                b_dgk, mask=mask, sem='relaxed')
            b_dgq += tl.sum(b_dA, axis=1)
        b_dA = b_dA.to(b_v.dtype)
        b_dk = tl.dot(tl.trans(b_dA), b_q2)
        tl.atomic_add(dk + (offset + tl.arange(0, BT))[:, None] * HQ * K +
            tl.arange(0, BK)[None, :], b_dk, mask=mask[:, None], sem='relaxed')
        p_w1 = tl.make_block_ptr(w1, (T, K), (H * K, 1), (offset, 0), (BT,
            BK), (1, 0))
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        p_w2 = tl.make_block_ptr(w2, (T, K), (H * K, 1), (offset, 0), (BT,
            BK), (1, 0))
        b_w2 = tl.load(p_w2, boundary_check=(0, 1))
        b_dA2 = tl.dot(b_dq.to(b_w2.dtype), tl.trans(b_w2)).to(b_v.dtype)
        b_A2 = tl.dot(b_q2.to(b_w1.dtype), tl.trans(b_w1)).to(b_v.dtype)
        b_dw2 = -tl.dot(tl.trans(b_A2), b_dq.to(b_v.dtype))
        tl.atomic_add(dw2 + (offset + tl.arange(0, BT))[:, None] * HQ * K +
            tl.arange(0, BK)[None, :], b_dw2, mask=mask[:, None], sem='relaxed'
            )
        b_dw1 = -tl.dot(tl.trans(b_dA2), b_q2.to(b_v.dtype))
        tl.atomic_add(dw1 + (offset + tl.arange(0, BT))[:, None] * HQ * K +
            tl.arange(0, BK)[None, :], b_dw1, mask=mask[:, None], sem='relaxed'
            )
        b_dq -= tl.dot(b_dA2, b_w1.to(b_v.dtype))
        b_dq += tl.dot(b_dA.to(b_k.dtype), b_k)
    p_dq_new = tl.make_block_ptr(dq_new, (T, K), (HQ * K, 1), (i_t * BT, 0),
        (BT, BK), (1, 0))
    tl.store(p_dq_new, b_dq.to(dq_new.dtype.element_ty), boundary_check=(0, 1))
    mask = i_t * BT + tl.arange(0, BT) < T
    if USE_GATE:
        tl.atomic_add(dg_cumsum + (i_t * BT + tl.arange(0, BT)) * HQ, b_dgq,
            mask=mask, sem='relaxed')


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=['T'])
def transform_q_fwd_kernel(q, q_new, w1, w2, cu_seqlens, indices, T, S: tl.
    constexpr, G: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr, K: tl.
    constexpr, BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr,
    NUM_BLOCKS: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices +
            i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T
    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1),
        (i_t * BT, 0), (BT, BK), (1, 0))
    b_q = tl.zeros([BT, BK], dtype=tl.float32)
    b_q += tl.load(p_q, boundary_check=(0, 1))
    if BS == BT:
        if i_t * BT % S == 0:
            p_q_new = tl.make_block_ptr(q_new + ((bos * NUM_BLOCKS + i_t *
                BT // S) * HQ + i_hq) * K, (T, K), (HQ * K * NUM_BLOCKS, 1),
                (i_t * BT, 0), (BT, BK), (1, 0))
            tl.store(p_q_new, b_q.to(q_new.dtype.element_ty),
                boundary_check=(0, 1))
    for offset in range((i_t + 1) * BT - 2 * BS, S - BS, -BS):
        p_w1 = tl.make_block_ptr(w1 + (bos * H + i_h) * K, (K, T), (1, K *
            H), (0, offset), (BK, BS), (0, 1))
        p_w2 = tl.make_block_ptr(w2 + (bos * H + i_h) * K, (T, K), (K * H, 
            1), (offset, 0), (BS, BK), (1, 0))
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_w2 = tl.load(p_w2, boundary_check=(0, 1))
        m_s = i_t * BT + tl.arange(0, BT) >= offset + BS
        b_s2 = tl.dot(b_q.to(b_w1.dtype), b_w1)
        b_s2 = tl.where(m_s[:, None], b_s2, 0)
        b_q -= tl.dot(b_s2.to(b_w2.dtype), b_w2)
        if offset % S == 0:
            p_q_new = tl.make_block_ptr(q_new + ((bos * NUM_BLOCKS + offset //
                S) * HQ + i_hq) * K, (T, K), (HQ * K * NUM_BLOCKS, 1), (i_t *
                BT, 0), (BT, BK), (1, 0))
            tl.store(p_q_new, b_q.to(q_new.dtype.element_ty),
                boundary_check=(0, 1))


def parallel_attn_bwd_preprocess(o: torch.Tensor, do: torch.Tensor):
    V = o.shape[-1]
    delta = torch.empty_like(o[..., 0], dtype=torch.float)
    parallel_attn_bwd_kernel_preprocess[delta.numel(),](o=o, do=do, delta=
        delta, B=triton.next_power_of_2(V), V=V)
    return delta


def chunk_cumprod_householder_bwd_fn(w1: torch.Tensor, w2: torch.Tensor,
    hc_suffix: torch.Tensor, dhc_whole: torch.Tensor, k: torch.Tensor, dk:
    torch.Tensor, S: int, BT: int, cu_seqlens: torch.Tensor=None):
    B, T, HQ, K = dk.shape
    H = k.shape[2]
    G = HQ // H
    split_indices = prepare_chunk_indices(cu_seqlens, S
        ) if cu_seqlens is not None else None
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    split_offsets = prepare_chunk_offsets(cu_seqlens, S
        ) if cu_seqlens is not None else None
    if cu_seqlens is None:
        N = B
        NS = N * triton.cdiv(T, S)
    else:
        N = len(cu_seqlens) - 1
        NS = split_offsets[-1].item()
    grid = NS, HQ
    dw1 = torch.empty_like(dk, dtype=torch.float32)
    dw2 = torch.empty_like(dk, dtype=torch.float32)
    dk_new = torch.empty_like(dk, dtype=torch.float32)
    chunk_cumprod_householder_bwd_kernel[grid](hc_suffix=hc_suffix,
        dhc_whole=dhc_whole, k=k, dk=dk, w1=w1, w2=w2, dw1=dw1, dw2=dw2,
        dk_new=dk_new, cu_seqlens=cu_seqlens, split_indices=split_indices,
        chunk_offsets=chunk_offsets, split_offsets=split_offsets, BT=BT, K=
        K, G=G, H=H, HQ=HQ, BK=K, T=T, S=S, num_warps=8 if K == 128 else 4,
        num_stages=2 if check_shared_mem('ampere') else 1)
    return dw1, dw2, dk_new


def chunk_cumprod_householder_fwd_fn(k: torch.Tensor, w1: torch.Tensor, w2:
    torch.Tensor, S: int, BT: int, cu_seqlens: torch.Tensor=None):
    B, T, H, K = k.shape
    split_indices = prepare_chunk_indices(cu_seqlens, S
        ) if cu_seqlens is not None else None
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    split_offsets = prepare_chunk_offsets(cu_seqlens, S
        ) if cu_seqlens is not None else None
    if cu_seqlens is None:
        N = B
        NS = N * triton.cdiv(T, S)
        NT = N * triton.cdiv(T, BT)
    else:
        N = len(cu_seqlens) - 1
        NS = split_offsets[-1]
        NT = chunk_offsets[-1]
    grid = NS, H
    hc_whole = torch.empty((NS, H, K, K), device=k.device, dtype=w1.dtype)
    k_new = torch.empty_like(k, dtype=k.dtype)
    hc_suffix = torch.empty((NT, H, K, K), device=k.device, dtype=w1.dtype)
    chunk_cumprod_householder_fwd_kernel[grid](k=k, k_new=k_new, w1=w1, w2=
        w2, hc_whole=hc_whole, hc_suffix=hc_suffix, cu_seqlens=cu_seqlens,
        split_indices=split_indices, chunk_offsets=chunk_offsets,
        split_offsets=split_offsets, BT=BT, K=K, H=H, BK=K, T=T, S=S,
        num_warps=8 if K == 128 else 4, num_stages=3 if check_shared_mem(
        'ampere') else 1)
    return k_new, hc_suffix, hc_whole


def intra_chunk_preprocess_bwd_fn(q, k, w, w2, beta, dq, dk, dA_local, dw1,
    dw2, A, L, D, do, scale, cu_seqlens=None):
    BT = A.shape[-1]
    HQ = q.shape[-2]
    B, T, H, K = k.shape
    G = HQ // H
    indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)
    grid = NT, B * HQ
    dbeta = torch.empty(B, T, HQ, device=q.device, dtype=k.dtype if G == 1 else
        torch.float32)
    dw = torch.empty(B, T, HQ, K, device=q.device, dtype=k.dtype if G == 1 else
        torch.float32)
    dk_new = torch.empty_like(dk, dtype=k.dtype if G == 1 else torch.float32)
    dq_new = torch.empty_like(dq, dtype=q.dtype)
    intra_chunk_preprocess_bwd_kernel[grid](q=q, k=k, w=w, w2=w2, beta=beta,
        AT=A, dA_local=dA_local, dq=dq, dq_new=dq_new, dk=dk, dk_new=dk_new,
        dw=dw, dbeta=dbeta, dw1=dw1, dw2=dw2, T=T, offsets=cu_seqlens,
        indices=indices, HQ=HQ, G=G, H=H, K=K, BT=BT, BK=triton.
        next_power_of_2(K), num_stages=3 if check_shared_mem('hopper') else 1)
    return dq_new, dk_new, dbeta, dw


def intra_chunk_preprocess_bwd_prepare_fn(q, k, v, w, beta, g_cumsum, A, L,
    D, do, scale, return_h=True, cu_seqlens=None):
    BT = A.shape[-1]
    HQ = q.shape[-2]
    B, T, H, K = k.shape
    G = HQ // H
    V = v.shape[-1]
    q_new = torch.empty_like(q)
    k_new = torch.empty_like(k)
    indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)
    grid = NT, B * HQ
    h = torch.empty_like(w)
    dA_local = torch.empty(B, T, HQ, BT, dtype=q.dtype, device=q.device)
    dv = torch.empty(B, T, HQ, V, device=q.device, dtype=torch.float32)
    dg_cumsum = torch.empty_like(g_cumsum) if g_cumsum is not None else None
    chunk_transform_qk_bwd_kernel_prepare[grid](q=q, k=k, v=v, w=w, beta=
        beta, g_cumsum=g_cumsum, AT=A, dA_local=dA_local, dv=dv, dg_cumsum=
        dg_cumsum, do=do, L=L, D=D, h=h, q_new=q_new, k_new=k_new, scale=
        scale, offsets=cu_seqlens, indices=indices, chunk_offsets=
        chunk_offsets, T=T, H=H, G=G, HQ=HQ, K=K, V=V, BK=triton.
        next_power_of_2(K), BV=triton.next_power_of_2(V), BT=BT, RETURN_H=
        return_h)
    return q_new, k_new, h, dA_local, dv, dg_cumsum


def parallel_path_bwd_dkv_fn(q, k, v, g_cumsum, do, dv, dg_cumsum, hc_whole,
    scale, L, D, cu_seqlens, S, BT, BS):
    B, T, num_blocks, HQ, K = q.shape
    V = v.shape[-1]
    H = k.shape[-2]
    G = HQ // H
    indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    split_offsets = prepare_chunk_offsets(cu_seqlens, S
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)
    if cu_seqlens is not None:
        assert split_offsets[-1] == hc_whole.shape[0]
    dk = torch.empty(B, T, HQ, K, dtype=torch.float32, device=q.device)
    parallel_path_bwd_dkv_kernel[NT, B * HQ](q=q, k=k, v=v, g_cumsum=
        g_cumsum, hc_whole=hc_whole, scale=scale, L=L, D=D, dk=dk, dv=dv,
        do=do, dg_cumsum=dg_cumsum, cu_seqlens=cu_seqlens, indices=indices,
        split_offsets=split_offsets, T=T, S=S, BT=BT, BS=BS, G=G, HQ=HQ, H=
        H, K=K, V=V, BK=triton.next_power_of_2(K), BV=triton.
        next_power_of_2(V), num_warps=8 if BT == 128 and K == 128 else 4,
        NUM_BLOCKS=num_blocks)
    return dk, dv, dg_cumsum


def parallel_path_bwd_dq_fn(q, k, v, g_cumsum, do, dg_cumsum, hc_whole,
    scale, L, D, cu_seqlens, S, BT, BS):
    B, T, num_blocks, HQ, K = q.shape
    H, V = v.shape[-2:]
    G = HQ // H
    BK, BV = triton.next_power_of_2(K), triton.next_power_of_2(V)
    indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    split_offsets = prepare_chunk_offsets(cu_seqlens, S
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)
    if cu_seqlens is not None:
        assert split_offsets[-1] == hc_whole.shape[0]
    dq = torch.empty(B, T, HQ, K, dtype=torch.float32, device=q.device)
    dhc_whole = torch.zeros(hc_whole.shape[0], HQ, K, K, dtype=torch.
        float32, device=q.device)
    parallel_path_bwd_dq_kernel[NT, B * HQ](q=q, k=k, v=v, g_cumsum=
        g_cumsum, hc_whole=hc_whole, scale=scale, L=L, D=D, dq=dq, do=do,
        dhc_whole=dhc_whole, dg_cumsum=dg_cumsum, cu_seqlens=cu_seqlens,
        indices=indices, split_offsets=split_offsets, T=T, S=S, BT=BT, BS=
        BS, G=G, HQ=HQ, H=H, K=K, V=V, BK=BK, BV=BV, NUM_BLOCKS=num_blocks,
        num_warps=8 if BT == 128 and K == 128 else 4, num_stages=3 if
        check_shared_mem('ampere') else 2)
    return dq, dhc_whole, dg_cumsum


def parallel_path_bwd_intra_chunk_fn(q, k, v, g_cumsum, w1, w2, dq, dk, dv,
    dg_cumsum, dw1, dw2, do, scale, L, D, cu_seqlens, S, BT):
    assert dk.dtype == dv.dtype == dw1.dtype == dw2.dtype == torch.float32, 'atomic_add requires float32'
    B, T, HQ, K = q.shape
    assert dk.shape == dq.shape
    V = v.shape[-1]
    H = k.shape[-2]
    G = HQ // H
    indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)
    dq_new = torch.empty_like(dq, dtype=q.dtype)
    parallel_path_bwd_intra_chunk_kernel[NT, B * HQ](q=q, k=k, v=v,
        g_cumsum=g_cumsum, w1=w1, w2=w2, L=L, D=D, dq=dq, dq_new=dq_new, dk
        =dk, dv=dv, dw1=dw1, dw2=dw2, do=do, dg_cumsum=dg_cumsum, offsets=
        cu_seqlens, indices=indices, T=T, S=S, BT=BT, scale=scale, G=G, HQ=
        HQ, H=H, K=K, V=V, BK=triton.next_power_of_2(K), BV=triton.
        next_power_of_2(V))
    return dq_new, dk, dv, dw1, dw2, dg_cumsum


def transform_q_fwd_fn(q, w1, w2, cu_seqlens, BT, BS, S):
    B, T, HQ, K = q.shape
    H = w1.shape[-2]
    G = HQ // H
    indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)
    num_blocks = triton.cdiv(T, S
        ) if cu_seqlens is None else get_max_num_splits(cu_seqlens, S)
    q_new = torch.zeros(B, T, num_blocks, HQ, K, dtype=q.dtype, device=q.device
        )
    transform_q_fwd_kernel[NT, B * HQ](q=q, q_new=q_new, w1=w1, w2=w2,
        cu_seqlens=cu_seqlens, indices=indices, T=T, K=K, BK=triton.
        next_power_of_2(K), G=G, HQ=HQ, H=H, BS=BS, BT=BT, S=S, NUM_BLOCKS=
        num_blocks, num_warps=8 if BT == 128 and K == 128 else 4)
    return q_new


@tensor_cache
def get_max_num_splits(cu_seqlens: torch.LongTensor, chunk_size: int) ->int:
    return triton.cdiv(int(max(prepare_lens(cu_seqlens))), chunk_size)


@tensor_cache
def prepare_chunk_offsets(cu_seqlens: torch.LongTensor, chunk_size: int
    ) ->torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(
        cu_seqlens), chunk_size)]).cumsum(-1)


# Backward method (kernel launch code)
@input_guard
@autocast_custom_bwd
def _ParallelPATHAttentionFunction_backward(ctx, do, dk_new):
    q, k, v, w, g_cumsum, o, beta, L, A = ctx.saved_tensors
    BT = 128 if check_shared_mem('ampere') else 64
    BS = 64 if check_shared_mem('hopper') else 32
    S = 512
    cu_seqlens = ctx.cu_seqlens
    delta = parallel_attn_bwd_preprocess(o, do)
    q_new, k_new, h, dA_local, dv, dg_cumsum = (
        intra_chunk_preprocess_bwd_prepare_fn(q=q, k=k, v=v, w=w, beta=beta,
        g_cumsum=g_cumsum, A=A, L=L, D=delta, do=do, scale=ctx.scale,
        cu_seqlens=cu_seqlens, return_h=False))
    w_fp16 = w.to(torch.float16)
    h_fp16 = h.to(torch.float16)
    k_new_large, hc_suffix, hc_whole = chunk_cumprod_householder_fwd_fn(k=
        k_new, w1=w_fp16, w2=h_fp16, S=S, BT=BS, cu_seqlens=cu_seqlens)
    q_new_large = transform_q_fwd_fn(q=q_new, w1=w_fp16, w2=h_fp16,
        cu_seqlens=cu_seqlens, BT=BT, BS=BS, S=S)
    w = w.to(q.dtype)
    h = h.to(q.dtype)
    A = A.to(q.dtype)
    dk, dv, _ = parallel_path_bwd_dkv_fn(q=q_new_large, k=k_new_large, v=v,
        g_cumsum=g_cumsum, do=do, dv=dv, dg_cumsum=dg_cumsum, hc_whole=
        hc_whole, scale=ctx.scale, cu_seqlens=cu_seqlens, L=L, D=delta, S=S,
        BT=BT, BS=BS)
    dq, dhc_whole, dg_cumsum = parallel_path_bwd_dq_fn(q=q_new_large, k=
        k_new_large, v=v, g_cumsum=g_cumsum, do=do, dg_cumsum=dg_cumsum,
        hc_whole=hc_whole, scale=ctx.scale, cu_seqlens=cu_seqlens, L=L, D=
        delta, S=S, BT=BT, BS=BS)
    dw1, dw2, dk = chunk_cumprod_householder_bwd_fn(w1=w, w2=h, k=k_new, dk
        =dk, hc_suffix=hc_suffix, dhc_whole=dhc_whole, cu_seqlens=
        cu_seqlens, S=S, BT=BS)
    dq, dk, dv, dw1, dw2, dg_cumsum = parallel_path_bwd_intra_chunk_fn(q=
        q_new, k=k_new, v=v, g_cumsum=g_cumsum, w1=w, w2=h, L=L, D=delta,
        scale=ctx.scale, dw1=dw1, dw2=dw2, dq=dq, dk=dk, dv=dv, do=do,
        dg_cumsum=dg_cumsum, cu_seqlens=cu_seqlens, S=S, BT=BS)
    dq, dk, dbeta, dw = intra_chunk_preprocess_bwd_fn(q=q, k=k, w=w, w2=h,
        beta=beta, dq=dq, dk=dk, dw1=dw1, dw2=dw2, dA_local=dA_local, A=A,
        L=L, D=delta, do=do, scale=ctx.scale, cu_seqlens=cu_seqlens)
    G = q.shape[-2] // k.shape[-2]
    if G > 1:
        assert dk.dtype == dv.dtype == dw.dtype == dbeta.dtype == torch.float32, 'reduction requires float32'
        dk = reduce(dk, 'b t (h g) k -> b t h k', g=G, reduction='sum')
        dv = reduce(dv, 'b t (h g) k -> b t h k', g=G, reduction='sum')
        dw = reduce(dw, 'b t (h g) k -> b t h k', g=G, reduction='sum')
        dbeta = reduce(dbeta, 'b t (h g) -> b t h', g=G, reduction='sum')
    if dg_cumsum is not None:
        dg_cumsum = chunk_global_cumsum(dg_cumsum, cu_seqlens=cu_seqlens,
            reverse=True)
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dw.to(w.dtype
        ), dbeta.to(beta.dtype), dg_cumsum.to(g_cumsum.dtype
        ) if g_cumsum is not None else None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class ParallelPATHAttentionFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, w, beta, g, scale, cu_seqlens, use_cache=False):
        g_cumsum = chunk_global_cumsum(g, cu_seqlens=cu_seqlens,
            output_dtype=torch.float32) if g is not None else None
        BS = 64 if check_shared_mem('hopper') else 32
        BT = 128 if check_shared_mem('ampere') else 64
        A = chunk_scaled_dot_kkt_fwd(k=w, beta=beta, cu_seqlens=cu_seqlens,
            chunk_size=BS, output_dtype=torch.float32)
        A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=w.dtype)
        q_new, k_new, w2, o, L, M = intra_chunk_preprocess_fwd_fn(q=q, k=k,
            v=v, w=w, beta=beta, g_cumsum=g_cumsum, A=A, scale=scale, BT=BS,
            cu_seqlens=cu_seqlens)
        w_fp16 = w.to(torch.float16)
        w2_fp16 = w2.to(torch.float16)
        o, L = parallel_path_fwd_fn(q=q_new, k=k_new, v=v, L=L, w1=w_fp16,
            w2=w2_fp16, M=M, o=o, g_cumsum=g_cumsum, scale=scale,
            cu_seqlens=cu_seqlens, BT=BT, BS=BS)
        k_cache = prepare_k_cache_fn(k=k_new, w1=w, w2=w2, cu_seqlens=
            cu_seqlens, BS=BS, use_cache=use_cache)
        ctx.save_for_backward(q, k, v, w, g_cumsum, o, beta, L, A)
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        return o, k_cache

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dk_new):
        q, k, v, w, g_cumsum, o, beta, L, A = ctx.saved_tensors
        BT = 128 if check_shared_mem('ampere') else 64
        BS = 64 if check_shared_mem('hopper') else 32
        S = 512
        cu_seqlens = ctx.cu_seqlens
        delta = parallel_attn_bwd_preprocess(o, do)
        q_new, k_new, h, dA_local, dv, dg_cumsum = (
            intra_chunk_preprocess_bwd_prepare_fn(q=q, k=k, v=v, w=w, beta=
            beta, g_cumsum=g_cumsum, A=A, L=L, D=delta, do=do, scale=ctx.
            scale, cu_seqlens=cu_seqlens, return_h=False))
        w_fp16 = w.to(torch.float16)
        h_fp16 = h.to(torch.float16)
        k_new_large, hc_suffix, hc_whole = chunk_cumprod_householder_fwd_fn(k
            =k_new, w1=w_fp16, w2=h_fp16, S=S, BT=BS, cu_seqlens=cu_seqlens)
        q_new_large = transform_q_fwd_fn(q=q_new, w1=w_fp16, w2=h_fp16,
            cu_seqlens=cu_seqlens, BT=BT, BS=BS, S=S)
        w = w.to(q.dtype)
        h = h.to(q.dtype)
        A = A.to(q.dtype)
        dk, dv, _ = parallel_path_bwd_dkv_fn(q=q_new_large, k=k_new_large,
            v=v, g_cumsum=g_cumsum, do=do, dv=dv, dg_cumsum=dg_cumsum,
            hc_whole=hc_whole, scale=ctx.scale, cu_seqlens=cu_seqlens, L=L,
            D=delta, S=S, BT=BT, BS=BS)
        dq, dhc_whole, dg_cumsum = parallel_path_bwd_dq_fn(q=q_new_large, k
            =k_new_large, v=v, g_cumsum=g_cumsum, do=do, dg_cumsum=
            dg_cumsum, hc_whole=hc_whole, scale=ctx.scale, cu_seqlens=
            cu_seqlens, L=L, D=delta, S=S, BT=BT, BS=BS)
        dw1, dw2, dk = chunk_cumprod_householder_bwd_fn(w1=w, w2=h, k=k_new,
            dk=dk, hc_suffix=hc_suffix, dhc_whole=dhc_whole, cu_seqlens=
            cu_seqlens, S=S, BT=BS)
        dq, dk, dv, dw1, dw2, dg_cumsum = parallel_path_bwd_intra_chunk_fn(q
            =q_new, k=k_new, v=v, g_cumsum=g_cumsum, w1=w, w2=h, L=L, D=
            delta, scale=ctx.scale, dw1=dw1, dw2=dw2, dq=dq, dk=dk, dv=dv,
            do=do, dg_cumsum=dg_cumsum, cu_seqlens=cu_seqlens, S=S, BT=BS)
        dq, dk, dbeta, dw = intra_chunk_preprocess_bwd_fn(q=q, k=k, w=w, w2
            =h, beta=beta, dq=dq, dk=dk, dw1=dw1, dw2=dw2, dA_local=
            dA_local, A=A, L=L, D=delta, do=do, scale=ctx.scale, cu_seqlens
            =cu_seqlens)
        G = q.shape[-2] // k.shape[-2]
        if G > 1:
            assert dk.dtype == dv.dtype == dw.dtype == dbeta.dtype == torch.float32, 'reduction requires float32'
            dk = reduce(dk, 'b t (h g) k -> b t h k', g=G, reduction='sum')
            dv = reduce(dv, 'b t (h g) k -> b t h k', g=G, reduction='sum')
            dw = reduce(dw, 'b t (h g) k -> b t h k', g=G, reduction='sum')
            dbeta = reduce(dbeta, 'b t (h g) -> b t h', g=G, reduction='sum')
        if dg_cumsum is not None:
            dg_cumsum = chunk_global_cumsum(dg_cumsum, cu_seqlens=
                cu_seqlens, reverse=True)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dw.to(w.dtype
            ), dbeta.to(beta.dtype), dg_cumsum.to(g_cumsum.dtype
            ) if g_cumsum is not None else None, None, None, None, None
