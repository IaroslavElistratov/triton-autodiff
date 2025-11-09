# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/fla-org/flash-linear-attention
# Source-Files: fla/ops/attn/parallel.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rh2l8zei/flash-linear-attention-main/fla/ops/attn/parallel.py
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

@triton.heuristics({'USE_G': lambda args: args['g_cumsum'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit
def parallel_attn_fwd_kernel(q, k, v, o, g_cumsum, lse, scale, cu_seqlens,
    chunk_indices, T, B: tl.constexpr, H: tl.constexpr, HQ: tl.constexpr, G:
    tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BS:
    tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T
    RCP_LN2: tl.constexpr = 1.4426950216
    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1),
        (i_t * BT, 0), (BT, BK), (1, 0))
    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * V, (T, V), (HQ * V, 1),
        (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT
        ,), (BT,), (0,))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_m = tl.full([BT], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)
    if USE_G:
        p_g = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (
            i_t * BT,), (BT,), (0,))
        b_gq = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    else:
        b_gq = None
    for i_s in range(0, i_t * BT, BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H * K),
            (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1),
            (i_s, i_v * BV), (BS, BV), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2
        if USE_G:
            o_k = i_s + tl.arange(0, BS)
            m_k = o_k < T
            b_gk = tl.load(g_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k,
                other=0).to(tl.float32)
            b_s += b_gq[:, None] - b_gk[None, :]
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_r = exp2(b_mp - b_m)
        b_p = exp2(b_s - b_m[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)
        b_mp = b_m
    o_q = i_t * BT + tl.arange(0, BT)
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H * K),
            (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1),
            (i_s, i_v * BV), (BS, BV), (1, 0))
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2
        if USE_G:
            b_gk = tl.load(g_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k,
                other=0).to(tl.float32)
            b_s += b_gq[:, None] - b_gk[None, :]
        b_s = tl.where((o_q[:, None] >= o_k[None, :]) & m_k[None, :], b_s,
            float('-inf'))
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_r = exp2(b_mp - b_m)
        b_p = exp2(b_s - b_m[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)
        b_mp = b_m
    b_o = b_o / b_acc[:, None]
    b_m += log2(b_acc)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty), boundary_check=(0,))


def parallel_attn_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    g_cumsum: torch.Tensor, scale: float, cu_seqlens: Optional[torch.
    LongTensor]=None):
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BT = 128
    if check_shared_mem('hopper', q.device.index):
        BS = min(64, max(16, triton.next_power_of_2(T)))
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(256, max(16, triton.next_power_of_2(V)))
        num_warps = 8
    elif check_shared_mem('ampere', q.device.index):
        BS = min(32, max(16, triton.next_power_of_2(T)))
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(128, max(16, triton.next_power_of_2(V)))
        num_warps = 4
    else:
        BS = min(32, max(16, triton.next_power_of_2(T)))
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(64, max(16, triton.next_power_of_2(V)))
        num_warps = 2
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    assert NK == 1, 'The key dimension can not be larger than 256'
    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
    grid = NV, NT, B * HQ
    parallel_attn_fwd_kernel[grid](q=q, k=k, v=v, o=o, g_cumsum=g_cumsum,
        lse=lse, scale=scale, cu_seqlens=cu_seqlens, chunk_indices=
        chunk_indices, B=B, T=T, H=H, HQ=HQ, G=G, K=K, V=V, BT=BT, BS=BS,
        BK=BK, BV=BV, num_warps=num_warps)
    return o, lse


# Forward method (kernel launch code)
@contiguous
@autocast_custom_fwd
def _ParallelAttentionFunction_forward(ctx, q, k, v, g, scale, cu_seqlens):
    ctx.dtype = q.dtype
    RCP_LN2: float = 1.4426950216
    g_cumsum = chunk_global_cumsum(g, cu_seqlens=cu_seqlens, scale=RCP_LN2
        ) if g is not None else None
    o, lse = parallel_attn_fwd(q=q, k=k, v=v, g_cumsum=g_cumsum, scale=
        scale, cu_seqlens=cu_seqlens)
    ctx.save_for_backward(q, k, v, o, g_cumsum, lse)
    ctx.cu_seqlens = cu_seqlens
    ctx.scale = scale
    return o.to(q.dtype)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.heuristics({'USE_G': lambda args: args['g_cumsum'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=['T'])
def parallel_attn_bwd_kernel_dkv(q, k, v, g_cumsum, lse, delta, do, dk, dv,
    dg_cumsum, cu_seqlens, chunk_indices, scale, T, B: tl.constexpr, H: tl.
    constexpr, HQ: tl.constexpr, G: tl.constexpr, K: tl.constexpr, V: tl.
    constexpr, BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl
    .constexpr, USE_G: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T
    RCP_LN2: tl.constexpr = 1.4426950216
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (
        i_t * BT, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (
        i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dk = tl.make_block_ptr(dk + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1
        ), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * HQ + i_hq) * V, (T, V), (HQ * V, 1
        ), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)
    o_k = i_t * BT + tl.arange(0, BT)
    if USE_G:
        p_gk = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (
            i_t * BT,), (BT,), (0,))
        b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
        b_dg = tl.zeros([BT], dtype=tl.float32)
    else:
        b_gk = None
        b_dg = None
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ * K,
            1), (i_s, 0), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ *
            V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_s,
            ), (BS,), (0,))
        p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (
            i_s,), (BS,), (0,))
        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_lse = tl.load(p_lse, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        b_s = tl.dot(b_k, tl.trans(b_q)) * scale * RCP_LN2
        if USE_G:
            p_gq = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,
                ), (i_s,), (BS,), (0,))
            b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
            b_s += b_gq[None, :] - b_gk[:, None]
        b_p = tl.where((o_k[:, None] <= o_q[None, :]) & m_q[None, :], exp2(
            b_s - b_lse[None, :]), 0)
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        b_dp = tl.dot(b_v, tl.trans(b_do))
        b_ds = b_p * (b_dp - b_delta[None, :])
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
        if USE_G:
            b_dg -= tl.sum(b_ds, 1)
    for i_s in range((i_t + 1) * BT, tl.cdiv(T, BS) * BS, BS):
        p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ * K,
            1), (i_s, 0), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ *
            V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_s,
            ), (BS,), (0,))
        p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (
            i_s,), (BS,), (0,))
        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_lse = tl.load(p_lse, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        b_s = tl.dot(b_k, tl.trans(b_q)) * scale * RCP_LN2
        if USE_G:
            p_gq = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,
                ), (i_s,), (BS,), (0,))
            b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
            b_s += b_gq[None, :] - b_gk[:, None]
        b_p = tl.where(m_q[None, :], exp2(b_s - b_lse[None, :]), 0)
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        b_dp = tl.dot(b_v, tl.trans(b_do))
        b_ds = b_p * (b_dp - b_delta[None, :])
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
        if USE_G:
            b_dg -= tl.sum(b_ds, 1)
    b_dk = b_dk * scale
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    if USE_G:
        p_dg = tl.make_block_ptr(dg_cumsum + bos * HQ + i_hq, (T,), (HQ,),
            (i_t * BT,), (BT,), (0,))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({'USE_G': lambda args: args['g_cumsum'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=['T'])
def parallel_attn_bwd_kernel_dq(q, k, v, lse, delta, do, dq, dg_cumsum,
    g_cumsum, scale, cu_seqlens, chunk_indices, T, B: tl.constexpr, H: tl.
    constexpr, HQ: tl.constexpr, G: tl.constexpr, K: tl.constexpr, V: tl.
    constexpr, BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl
    .constexpr, IS_VARLEN: tl.constexpr, USE_G: tl.constexpr):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T
    RCP_LN2: tl.constexpr = 1.4426950216
    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1),
        (i_t * BT, 0), (BT, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1
        ), (i_t * BT, 0), (BT, BK), (1, 0))
    p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ * V, 1
        ), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT
        ,), (BT,), (0,))
    p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_t *
        BT,), (BT,), (0,))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_lse = tl.load(p_lse, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    if USE_G:
        b_dg = tl.zeros([BT], dtype=tl.float32)
        p_gq = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (
            i_t * BT,), (BT,), (0,))
        b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
    else:
        b_gq = None
        b_dg = None
    o_q = i_t * BT + tl.arange(0, BT)
    for i_s in range(0, i_t * BT, BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H * K),
            (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T), (1, H * V),
            (i_v * BV, i_s), (BV, BS), (0, 1))
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2
        if USE_G:
            b_gk = tl.load(g_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k,
                other=0).to(tl.float32)
            b_s += b_gq[:, None] - b_gk[None, :]
        b_s = tl.where((o_q[:, None] >= o_k[None, :]) & m_k[None, :], b_s,
            float('-inf'))
        b_p = exp2(b_s - b_lse[:, None])
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
        if USE_G:
            b_dg += tl.sum(b_ds, 1)
    o_q = i_t * BT + tl.arange(0, BT)
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H * K),
            (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T), (1, H * V),
            (i_v * BV, i_s), (BV, BS), (0, 1))
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2
        if USE_G:
            p_gk = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,
                ), (i_s,), (BS,), (0,))
            b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
            b_s += b_gq[:, None] - b_gk[None, :]
        b_p = tl.where((o_q[:, None] >= o_k[None, :]) & m_k[None, :], exp2(
            b_s - b_lse[:, None]), 0)
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
        if USE_G:
            b_dg += tl.sum(b_ds, 1)
    b_dq *= scale
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    if USE_G:
        p_dg = tl.make_block_ptr(dg_cumsum + bos * HQ + i_hq, (T,), (HQ,),
            (i_t * BT,), (BT,), (0,))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


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


def parallel_attn_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o:
    torch.Tensor, g_cumsum: torch.Tensor, lse: torch.Tensor, do: torch.
    Tensor, scale: float=None, chunk_size: int=128, cu_seqlens: Optional[
    torch.LongTensor]=None):
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    if check_shared_mem('hopper'):
        BT = 128
        BS = 64
        BK = max(triton.next_power_of_2(K), 16)
        BV = max(triton.next_power_of_2(V), 16)
        num_warps = 8
    elif check_shared_mem('ampere'):
        BS = 32
        BK = max(triton.next_power_of_2(K), 16)
        BV = max(triton.next_power_of_2(V), 16)
        BT = 128 if K <= 64 else 64
        num_warps = 4
    else:
        BT = 64
        BS = 32
        BK = max(triton.next_power_of_2(K), 16)
        BV = min(max(triton.next_power_of_2(V), 16), 64)
        num_warps = 2
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NV = triton.cdiv(V, BV)
    delta = parallel_attn_bwd_preprocess(o, do)
    dq = torch.empty(B, T, HQ, K, dtype=k.dtype if H == HQ else torch.float,
        device=q.device)
    dk = torch.empty(B, T, HQ, K, dtype=k.dtype if H == HQ else torch.float,
        device=q.device)
    dv = torch.empty(B, T, HQ, V, dtype=v.dtype if H == HQ else torch.float,
        device=q.device)
    grid = NV, NT, B * HQ
    dg_cumsum, dg_cumsum_k = None, None
    if g_cumsum is not None:
        dg_cumsum = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
        dg_cumsum_k = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
    parallel_attn_bwd_kernel_dq[grid](q=q, k=k, v=v, g_cumsum=g_cumsum, lse
        =lse, delta=delta, do=do, dq=dq, dg_cumsum=dg_cumsum, cu_seqlens=
        cu_seqlens, chunk_indices=chunk_indices, scale=scale, T=T, B=B, H=H,
        HQ=HQ, G=G, K=K, V=V, BT=BT, BS=BS, BK=BK, BV=BV, num_warps=num_warps)
    parallel_attn_bwd_kernel_dkv[grid](q=q, k=k, v=v, g_cumsum=g_cumsum,
        lse=lse, delta=delta, do=do, dk=dk, dv=dv, dg_cumsum=dg_cumsum_k,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, scale=scale, T=
        T, B=B, H=H, HQ=HQ, G=G, K=K, V=V, BT=BT, BS=BS, BK=BK, BV=BV,
        num_warps=num_warps)
    dk = reduce(dk, 'b t (h g) k -> b t h k', g=G, reduction='sum')
    dv = reduce(dv, 'b t (h g) v -> b t h v', g=G, reduction='sum')
    if g_cumsum is not None:
        dg_cumsum.add_(dg_cumsum_k)
    return dq, dk, dv, dg_cumsum


def parallel_attn_bwd_preprocess(o: torch.Tensor, do: torch.Tensor):
    V = o.shape[-1]
    delta = torch.empty_like(o[..., 0], dtype=torch.float)
    parallel_attn_bwd_kernel_preprocess[delta.numel(),](o=o, do=do, delta=
        delta, B=triton.next_power_of_2(V), V=V)
    return delta


# Backward method (kernel launch code)
@contiguous
@autocast_custom_bwd
def _ParallelAttentionFunction_backward(ctx, do):
    q, k, v, o, g_cumsum, lse = ctx.saved_tensors
    dq, dk, dv, dg = parallel_attn_bwd(q=q, k=k, v=v, o=o, g_cumsum=
        g_cumsum, lse=lse, do=do, scale=ctx.scale, cu_seqlens=ctx.cu_seqlens)
    if dg is not None:
        dg = chunk_global_cumsum(dg, cu_seqlens=ctx.cu_seqlens, reverse=True)
    return dq.to(q), dk.to(k), dv.to(v), dg, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

@torch.compile
class ParallelAttentionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, g, scale, cu_seqlens):
        ctx.dtype = q.dtype
        RCP_LN2: float = 1.4426950216
        g_cumsum = chunk_global_cumsum(g, cu_seqlens=cu_seqlens, scale=RCP_LN2
            ) if g is not None else None
        o, lse = parallel_attn_fwd(q=q, k=k, v=v, g_cumsum=g_cumsum, scale=
            scale, cu_seqlens=cu_seqlens)
        ctx.save_for_backward(q, k, v, o, g_cumsum, lse)
        ctx.cu_seqlens = cu_seqlens
        ctx.scale = scale
        return o.to(q.dtype)

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do):
        q, k, v, o, g_cumsum, lse = ctx.saved_tensors
        dq, dk, dv, dg = parallel_attn_bwd(q=q, k=k, v=v, o=o, g_cumsum=
            g_cumsum, lse=lse, do=do, scale=ctx.scale, cu_seqlens=ctx.
            cu_seqlens)
        if dg is not None:
            dg = chunk_global_cumsum(dg, cu_seqlens=ctx.cu_seqlens, reverse
                =True)
        return dq.to(q), dk.to(k), dv.to(v), dg, None, None
