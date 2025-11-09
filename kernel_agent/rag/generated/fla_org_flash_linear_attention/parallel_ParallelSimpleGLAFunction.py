# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/fla-org/flash-linear-attention
# Source-Files: fla/ops/simple_gla/parallel.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rh2l8zei/flash-linear-attention-main/fla/ops/simple_gla/parallel.py
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

@triton.heuristics({'NV': lambda args: triton.cdiv(args['V'], args['BV']),
    'OUTPUT_ATTENTIONS': lambda args: args['attn'] is not None, 'USE_G': lambda
    args: args['g'] is not None, 'IS_VARLEN': lambda args: args[
    'cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages
    =num_stages) for num_warps in [2, 4, 8, 16] for num_stages in [2, 3, 4]
    ], key=['BT', 'BS', 'BK', 'BV', 'USE_G'], **autotune_cache_kwargs)
@triton.jit
def parallel_simple_gla_fwd_kernel(q, k, v, g, o, attn, scale, cu_seqlens,
    chunk_indices, T, B: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V:
    tl.constexpr, BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV:
    tl.constexpr, NV: tl.constexpr, OUTPUT_ATTENTIONS: tl.constexpr,
    IS_VARLEN: tl.constexpr, USE_G: tl.constexpr):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_k, i_v = i_kv // NV, i_kv % NV
    i_b, i_h = i_bh // H, i_bh % H
    all = B * T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    o += ((i_k * all + bos) * H + i_h) * V
    if USE_G:
        g += bos * H + i_h
    if OUTPUT_ATTENTIONS:
        attn += i_k * B * H * T * T + (bos * H + i_h * T) * T
    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (
        BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    o_q = i_t * BT + tl.arange(0, BT)
    m_q = o_q < T
    if USE_G:
        b_gq = tl.load(g + o_q * H, mask=m_q, other=float('-inf')).to(tl.
            float32)
    else:
        b_gq = None
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (i_k * BK, i_s), (BK,
            BS), (0, 1))
        p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_s, i_v * BV), (BS,
            BV), (1, 0))
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        m_s = (o_q[:, None] >= o_k[None, :]) & (m_q[:, None] & m_k[None, :])
        b_s = tl.dot(b_q, b_k)
        if USE_G:
            b_gk = tl.load(g + o_k * H, mask=m_k, other=0)
            b_s *= exp(b_gq[:, None] - b_gk[None, :])
        b_s = tl.where(m_s, b_s, 0)
        if i_s >= 0:
            b_o += tl.dot(b_s.to(b_q.dtype), b_v)
        if OUTPUT_ATTENTIONS:
            p_a = tl.make_block_ptr(attn, (T, T), (T, 1), (i_t * BT, i_s),
                (BT, BS), (1, 0))
            tl.store(p_a, b_s.to(p_a.dtype.element_ty), boundary_check=(0, 1))
    for i_s in range(i_t * BT - BS, -BS, -BS):
        p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (i_k * BK, i_s), (BK,
            BS), (0, 1))
        p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_s, i_v * BV), (BS,
            BV), (1, 0))
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        m_s = m_q[:, None] & m_k[None, :]
        b_s = tl.dot(b_q, b_k)
        if USE_G:
            b_g = tl.load(g + o_k * H, mask=m_k, other=0)
            b_gn = tl.load(g + (min(i_s + BS, T) - 1) * H)
            b_gp = tl.load(g + (i_s - 1) * H) if i_s % BT > 0 else 0.0
            b_s *= exp(b_gq[:, None] + (b_gn - b_g)[None, :])
            b_gq += b_gn - b_gp
        b_s = tl.where(m_s, b_s, 0)
        if OUTPUT_ATTENTIONS:
            p_a = tl.make_block_ptr(attn, (T, T), (T, 1), (i_t * BT, i_s),
                (BT, BS), (1, 0))
            tl.store(p_a, b_s.to(p_a.dtype.element_ty), boundary_check=(0, 1))
        if i_s >= 0:
            b_o += tl.dot(b_s.to(b_v.dtype), b_v)
    p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (
        BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


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


def parallel_simple_gla_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.
    Tensor, g: torch.Tensor, scale: float, output_attentions: bool=False,
    chunk_size: int=128, cu_seqlens: Optional[torch.LongTensor]=None):
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT, BS = chunk_size, 32
    if check_shared_mem('hopper', k.device.index):
        BK = min(256, triton.next_power_of_2(K))
        BV = min(256, triton.next_power_of_2(V))
    elif check_shared_mem('ampere', k.device.index):
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
    else:
        BK = min(64, triton.next_power_of_2(K))
        BV = min(64, triton.next_power_of_2(V))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert BT % BS == 0
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if g is not None:
        g = chunk_local_cumsum(g, chunk_size, cu_seqlens=cu_seqlens)
    grid = NK * NV, NT, B * H
    o = torch.empty(NK, *v.shape, dtype=v.dtype if NK == 1 else torch.float,
        device=q.device)
    attn = q.new_zeros(NK, B, H, T, T) if output_attentions else None
    parallel_simple_gla_fwd_kernel[grid](q=q, k=k, v=v, g=g, o=o, attn=attn,
        scale=scale, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, B=
        B, H=H, T=T, K=K, V=V, BT=BT, BS=BS, BK=BK, BV=BV)
    o = o.sum(0)
    if output_attentions:
        attn = attn.sum(0)
    return o, g, attn


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


# Forward method (kernel launch code)
@input_guard
@autocast_custom_fwd
def _ParallelSimpleGLAFunction_forward(ctx, q, k, v, g, scale,
    output_attentions, cu_seqlens):
    chunk_size = 128
    ctx.dtype = q.dtype
    o, g, attn = parallel_simple_gla_fwd(q=q, k=k, v=v, g=g, scale=scale,
        output_attentions=output_attentions, chunk_size=chunk_size,
        cu_seqlens=cu_seqlens)
    ctx.save_for_backward(q, k, v, g, cu_seqlens)
    ctx.scale = scale
    ctx.chunk_size = chunk_size
    return o.to(q.dtype), attn


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.heuristics({'NV': lambda args: triton.cdiv(args['V'], args['BV']),
    'USE_G': lambda args: args['g'] is not None, 'IS_VARLEN': lambda args: 
    args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config(triton_config, num_warps=num_warps) for
    num_warps in NUM_WARPS], key=['BT', 'BS', 'BK', 'BV', 'USE_G'], **
    autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def parallel_simple_gla_bwd_kernel(q, k, v, g, do, dq, dk, dv, dg, scale,
    cu_seqlens, chunk_indices, T, B: tl.constexpr, H: tl.constexpr, K: tl.
    constexpr, V: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr, BK: tl.
    constexpr, BV: tl.constexpr, NV: tl.constexpr, IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_k, i_v = i_kv // NV, i_kv % NV
    i_b, i_h = i_bh // H, i_bh % H
    dq += i_v * B * H * T * K
    dk += i_v * B * H * T * K
    dv += i_k * B * H * T * V
    if USE_G:
        dg += i_kv * B * H * T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V
    if USE_G:
        g += bos * H + i_h
        dg += bos * H + i_h
    parallel_simple_gla_bwd_kernel_dq(i_t=i_t, i_k=i_k, i_v=i_v, q=q, k=k,
        v=v, g=g, do=do, dq=dq, dg=dg, scale=scale, T=T, H=H, K=K, V=V, BT=
        BT, BS=BS, BK=BK, BV=BV, USE_G=USE_G)
    tl.debug_barrier()
    parallel_simple_gla_bwd_kernel_dkv(i_t=i_t, i_k=i_k, i_v=i_v, q=q, k=k,
        v=v, g=g, do=do, dk=dk, dv=dv, dg=dg, scale=scale, T=T, H=H, K=K, V
        =V, BT=BT, BS=BS, BK=BK, BV=BV, USE_G=USE_G)


@triton.jit(do_not_specialize=['T'])
def parallel_simple_gla_bwd_kernel_dkv(i_t, i_k, i_v, q, k, v, g, do, dk,
    dv, dg, scale, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT:
    tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_G: tl.constexpr):
    o_k = i_t * BT + tl.arange(0, BT)
    m_k = o_k < T
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (
        BT, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (
        BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)
    if USE_G:
        b_gk = tl.load(g + o_k * H, mask=m_k, other=0)
    NTS = tl.cdiv(T, BS)
    for i_s in range(NTS * BS - BS, (i_t + 1) * BT - BS, -BS):
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_s, i_k * BK), (BS,
            BK), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_s, i_v * BV), (
            BS, BV), (1, 0))
        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_ds = tl.dot(b_v, tl.trans(b_do))
        b_s = tl.dot(b_k, tl.trans(b_q))
        if USE_G:
            b_gq = tl.load(g + o_q * H, mask=m_q, other=float('-inf'))
            b_gp = tl.load(g + (min(i_s + BS, T) - 1) * H)
            b_gn = tl.load(g + (i_s - 1) * H) if i_s % BT > 0 else 0.0
            if i_s >= 0:
                b_gpn = exp(b_gp - b_gn)
                b_dk *= b_gpn
                b_dv *= b_gpn
                b_gqn = exp(b_gq - b_gn)
                b_ds *= b_gqn[None, :]
                b_s *= b_gqn[None, :]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
        b_dv += tl.dot(b_s.to(b_do.dtype), b_do)
    if USE_G:
        b_gn = tl.load(g + (min(i_t * BT + BT, T) - 1) * H)
        if i_t >= 0:
            b_gpn = exp(b_gn - b_gk)[:, None]
            b_dk *= b_gpn
            b_dv *= b_gpn
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_s, i_k * BK), (BS,
            BK), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_s, i_v * BV), (
            BS, BV), (1, 0))
        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_s = tl.dot(b_k, tl.trans(b_q))
        b_ds = tl.dot(b_v, tl.trans(b_do))
        if USE_G:
            b_gq = tl.load(g + o_q * H, mask=m_q, other=float('-inf'))
            if i_s >= 0:
                b_gkq = exp(-b_gk[:, None] + b_gq[None, :])
                b_ds *= b_gkq
                b_s *= b_gkq
        m_s = o_k[:, None] <= o_q[None, :]
        b_s = tl.where(m_s, b_s, 0)
        b_ds = tl.where(m_s, b_ds, 0)
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
        b_dv += tl.dot(b_s.to(b_do.dtype), b_do)
    b_dk *= scale
    b_dv *= scale
    p_dk = tl.make_block_ptr(dk, (T, K), (H * K, 1), (i_t * BT, i_k * BK),
        (BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t * BT, i_v * BV),
        (BT, BV), (1, 0))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    if USE_G:
        b_dg = tl.load(dg + o_k * H, mask=m_k, other=0)
        b_dg -= tl.sum(b_dk * b_k, 1)
        tl.store(dg + o_k * H, b_dg.to(dg.dtype.element_ty), mask=m_k)


@triton.jit(do_not_specialize=['T'])
def parallel_simple_gla_bwd_kernel_dq(i_t, i_k, i_v, q, k, v, g, do, dq, dg,
    scale, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.
    constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_G:
    tl.constexpr):
    p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_t * BT, i_v * BV),
        (BT, BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    o_q = i_t * BT + tl.arange(0, BT)
    m_q = o_q < T
    for i_s in range(0, i_t * BT, BS):
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_s, i_k * BK), (BS,
            BK), (1, 0))
        p_v = tl.make_block_ptr(v, (V, T), (1, H * V), (i_v * BV, i_s), (BV,
            BS), (0, 1))
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_ds = tl.dot(b_do, b_v)
        if USE_G:
            b_g = tl.load(g + o_k * H, mask=m_k, other=0)
            b_gn = tl.load(g + (min(i_s + BS, T) - 1) * H)
            b_gp = tl.load(g + (i_s - 1) * H) if i_s % BT > 0 else 0.0
            b_ds *= tl.where(m_k, exp(b_gn - b_g), 0)[None, :]
            if i_s > 0:
                b_dq *= exp(b_gn - b_gp)
        b_dq += tl.dot(b_ds.to(b_v.dtype), b_k)
    if USE_G:
        b_gq = tl.load(g + o_q * H, mask=m_q, other=float('-inf'))
        b_dq *= exp(b_gq)[:, None]
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_s, i_k * BK), (BS,
            BK), (1, 0))
        p_v = tl.make_block_ptr(v, (V, T), (1, H * V), (i_v * BV, i_s), (BV,
            BS), (0, 1))
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_ds = tl.dot(b_do, b_v)
        if USE_G:
            b_gk = tl.load(g + o_k * H, mask=m_k, other=0)
            b_ds *= exp(b_gq[:, None] - b_gk[None, :])
        m_s = (o_q[:, None] >= o_k[None, :]) & (m_q[:, None] & m_k[None, :])
        b_ds = tl.where(m_s, b_ds, 0)
        b_dq += tl.dot(b_ds.to(b_k.dtype), b_k)
    b_dq *= scale
    p_dq = tl.make_block_ptr(dq, (T, K), (H * K, 1), (i_t * BT, i_k * BK),
        (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    if USE_G:
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK),
            (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_dg = tl.sum(b_dq * b_q, 1)
        p_dg = tl.make_block_ptr(dg, (T,), (H,), (i_t * BT,), (BT,), (0,))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


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


def parallel_simple_gla_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.
    Tensor, g: torch.Tensor, do: torch.Tensor, scale: float, chunk_size:
    int=128, cu_seqlens: Optional[torch.LongTensor]=None):
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT, BS = chunk_size, 32
    if check_shared_mem('hopper', k.device.index):
        BK = min(256, triton.next_power_of_2(K))
        BV = min(256, triton.next_power_of_2(V))
    elif check_shared_mem('ampere', k.device.index):
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
    elif check_shared_mem('ada', k.device.index):
        BK = min(64, triton.next_power_of_2(K))
        BV = min(64, triton.next_power_of_2(V))
    else:
        BK = min(32, triton.next_power_of_2(K))
        BV = min(32, triton.next_power_of_2(V))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert BT % BS == 0
    dq = torch.empty(NV, *q.shape, dtype=q.dtype if NV == 1 else torch.
        float, device=q.device)
    dk = torch.empty(NV, *k.shape, dtype=k.dtype if NV == 1 else torch.
        float, device=q.device)
    dv = torch.empty(NK, *v.shape, dtype=v.dtype if NK == 1 else torch.
        float, device=q.device)
    dg = torch.empty(NK * NV, *g.shape, dtype=torch.float, device=q.device
        ) if g is not None else None
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    grid = NK * NV, NT, B * H
    parallel_simple_gla_bwd_kernel[grid](q=q, k=k, v=v, g=g, do=do, dq=dq,
        dk=dk, dv=dv, dg=dg, cu_seqlens=cu_seqlens, chunk_indices=
        chunk_indices, scale=scale, T=T, B=B, H=H, K=K, V=V, BT=BT, BS=BS,
        BK=BK, BV=BV)
    dq = dq.sum(0)
    dk = dk.sum(0)
    dv = dv.sum(0)
    dg = chunk_global_cumsum(dg.sum(0), reverse=True, cu_seqlens=cu_seqlens
        ) if g is not None else None
    return dq, dk, dv, dg


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


# Backward method (kernel launch code)
@input_guard
@autocast_custom_bwd
def _ParallelSimpleGLAFunction_backward(ctx, do, da=None):
    q, k, v, g, cu_seqlens = ctx.saved_tensors
    dq, dk, dv, dg = parallel_simple_gla_bwd(q=q, k=k, v=v, g=g, do=do,
        scale=ctx.scale, chunk_size=ctx.chunk_size, cu_seqlens=cu_seqlens)
    return dq.to(q), dk.to(k), dv.to(v), dg.to(ctx.dtype
        ) if dg is not None else None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class ParallelSimpleGLAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, g, scale, output_attentions, cu_seqlens):
        chunk_size = 128
        ctx.dtype = q.dtype
        o, g, attn = parallel_simple_gla_fwd(q=q, k=k, v=v, g=g, scale=
            scale, output_attentions=output_attentions, chunk_size=
            chunk_size, cu_seqlens=cu_seqlens)
        ctx.save_for_backward(q, k, v, g, cu_seqlens)
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        return o.to(q.dtype), attn

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, da=None):
        q, k, v, g, cu_seqlens = ctx.saved_tensors
        dq, dk, dv, dg = parallel_simple_gla_bwd(q=q, k=k, v=v, g=g, do=do,
            scale=ctx.scale, chunk_size=ctx.chunk_size, cu_seqlens=cu_seqlens)
        return dq.to(q), dk.to(k), dv.to(v), dg.to(ctx.dtype
            ) if dg is not None else None, None, None, None
