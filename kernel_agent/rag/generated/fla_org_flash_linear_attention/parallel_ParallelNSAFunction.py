# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/fla-org/flash-linear-attention
# Source-Files: fla/ops/nsa/parallel.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rh2l8zei/flash-linear-attention-main/fla/ops/nsa/parallel.py
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

@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) ->torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not
    None, 'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'],
    torch.Tensor)})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for
    num_warps in [1, 2, 4]], key=['BS', 'BK', 'BV'], **autotune_cache_kwargs)
@triton.jit
def parallel_nsa_fwd_kernel(q, k, v, o, lse, scale, block_indices,
    block_counts, cu_seqlens, token_indices, T, H: tl.constexpr, HQ: tl.
    constexpr, G: tl.constexpr, K: tl.constexpr, V: tl.constexpr, S: tl.
    constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    IS_VARLEN: tl.constexpr, USE_BLOCK_COUNTS: tl.constexpr):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(
            token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    block_indices += (bos + i_t) * H * S + i_h * S
    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * H + i_h)
    else:
        NS = S
    p_q = tl.make_block_ptr(q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h *
        G, 0), (G, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    p_o = tl.make_block_ptr(o + (bos + i_t) * HQ * V, (HQ, V), (V, 1), (i_h *
        G, i_v * BV), (G, BV), (1, 0))
    p_lse = lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G)
    b_o = tl.zeros([G, BV], dtype=tl.float32)
    b_m = tl.full([G], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([G], dtype=tl.float32)
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS
        if i_s <= i_t and i_s >= 0:
            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK,
                BS), (0, 1))
            p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_s, i_v * BV),
                (BS, BV), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_s = tl.dot(b_q, b_k)
            b_s = tl.where((i_t >= i_s + tl.arange(0, BS))[None, :], b_s,
                float('-inf'))
            b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
            b_r = exp(b_mp - b_m)
            b_p = exp(b_s - b_m[:, None])
            b_acc = b_acc * b_r + tl.sum(b_p, 1)
            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)
            b_mp = b_m
    b_o = b_o / b_acc[:, None]
    b_m += log(b_acc)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty))


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


def parallel_nsa_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    block_indices: torch.LongTensor, block_counts: Union[torch.LongTensor,
    int], block_size: int, scale: float, cu_seqlens: Optional[torch.
    LongTensor]=None, token_indices: Optional[torch.LongTensor]=None):
    B, T, H, K, V, S = *k.shape, v.shape[-1], block_indices.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BS = block_size
    if check_shared_mem('hopper', q.device.index):
        BK = min(256, triton.next_power_of_2(K))
        BV = min(256, triton.next_power_of_2(V))
    else:
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, 'The key dimension can not be larger than 256'
    grid = T, NV, B * H
    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
    parallel_nsa_fwd_kernel[grid](q=q, k=k, v=v, o=o, lse=lse, scale=scale,
        block_indices=block_indices, block_counts=block_counts, cu_seqlens=
        cu_seqlens, token_indices=token_indices, T=T, H=H, HQ=HQ, G=G, K=K,
        V=V, S=S, BS=BS, BK=BK, BV=BV)
    return o, lse


@tensor_cache
def prepare_position_ids(cu_seqlens: torch.LongTensor) ->torch.LongTensor:
    return torch.cat([torch.arange(n, dtype=cu_seqlens.dtype, device=
        cu_seqlens.device) for n in prepare_lens(cu_seqlens).unbind()])


@tensor_cache
def prepare_sequence_ids(cu_seqlens: torch.LongTensor) ->torch.LongTensor:
    return prepare_position_ids(cu_seqlens).eq(0).cumsum(0) - 1


@tensor_cache
def prepare_token_indices(cu_seqlens: torch.LongTensor) ->torch.LongTensor:
    position_ids = prepare_position_ids(cu_seqlens)
    return torch.stack([prepare_sequence_ids(cu_seqlens), position_ids], 1).to(
        cu_seqlens)


# Forward method (kernel launch code)
@contiguous
@autocast_custom_fwd
def _ParallelNSAFunction_forward(ctx, q, k, v, block_indices, block_counts,
    block_size, scale, cu_seqlens):
    ctx.dtype = q.dtype
    token_indices = prepare_token_indices(cu_seqlens
        ) if cu_seqlens is not None else None
    o, lse = parallel_nsa_fwd(q=q, k=k, v=v, block_indices=block_indices,
        block_counts=block_counts, block_size=block_size, scale=scale,
        cu_seqlens=cu_seqlens, token_indices=token_indices)
    ctx.save_for_backward(q, k, v, o, lse)
    ctx.block_indices = block_indices
    ctx.block_counts = block_counts
    ctx.cu_seqlens = cu_seqlens
    ctx.token_indices = token_indices
    ctx.block_size = block_size
    ctx.scale = scale
    return o.to(q.dtype)


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
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for
    num_warps in [1, 2, 4]], key=['BS', 'BK', 'BV'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def parallel_nsa_bwd_kernel_dkv(q, k, v, lse, delta, do, dk, dv, block_mask,
    cu_seqlens, chunk_indices, scale, T, B: tl.constexpr, H: tl.constexpr,
    HQ: tl.constexpr, G: tl.constexpr, K: tl.constexpr, V: tl.constexpr, M:
    tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    IS_VARLEN: tl.constexpr):
    i_v, i_s, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    all = B * T
    if IS_VARLEN:
        i_n, i_s = tl.load(chunk_indices + i_s * 2).to(tl.int32), tl.load(
            chunk_indices + i_s * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (
        i_s * BS, 0), (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (
        i_s * BS, i_v * BV), (BS, BV), (1, 0))
    p_dk = tl.make_block_ptr(dk + (i_v * all * H + bos * H + i_h) * K, (T,
        K), (H * K, 1), (i_s * BS, 0), (BS, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * H + i_h) * V, (T, V), (H * V, 1),
        (i_s * BS, i_v * BV), (BS, BV), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BS, BK], dtype=tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BS, BV], dtype=tl.float32)
    for i in range(i_s * BS, T):
        b_m = tl.load(block_mask + (bos + i) * H * M + i_h * M + i_s)
        if b_m:
            p_q = tl.make_block_ptr(q + (bos + i) * HQ * K, (HQ, K), (K, 1),
                (i_h * G, 0), (G, BK), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = (b_q * scale).to(b_q.dtype)
            p_do = tl.make_block_ptr(do + (bos + i) * HQ * V, (HQ, V), (V, 
                1), (i_h * G, i_v * BV), (G, BV), (1, 0))
            p_lse = lse + (bos + i) * HQ + i_h * G + tl.arange(0, G)
            p_delta = delta + (bos + i) * HQ + i_h * G + tl.arange(0, G)
            b_do = tl.load(p_do, boundary_check=(0, 1))
            b_lse = tl.load(p_lse)
            b_delta = tl.load(p_delta)
            b_s = tl.dot(b_k, tl.trans(b_q))
            b_p = exp(b_s - b_lse[None, :])
            b_p = tl.where((i >= i_s * BS + tl.arange(0, BS))[:, None], b_p, 0)
            b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
            b_dp = tl.dot(b_v, tl.trans(b_do))
            b_ds = b_p * (b_dp - b_delta[None, :])
            b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not
    None, 'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'],
    torch.Tensor)})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for
    num_warps in [1, 2, 4]], key=['BS', 'BK', 'BV'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def parallel_nsa_bwd_kernel_dq(q, k, v, lse, delta, do, dq, scale,
    block_indices, block_counts, cu_seqlens, token_indices, T, B: tl.
    constexpr, H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr, K: tl.
    constexpr, V: tl.constexpr, S: tl.constexpr, BS: tl.constexpr, BK: tl.
    constexpr, BV: tl.constexpr, IS_VARLEN: tl.constexpr, USE_BLOCK_COUNTS:
    tl.constexpr):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    all = B * T
    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(
            token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    q += (bos + i_t) * HQ * K
    do += (bos + i_t) * HQ * V
    lse += (bos + i_t) * HQ
    delta += (bos + i_t) * HQ
    dq += (i_v * all + bos + i_t) * HQ * K
    block_indices += (bos + i_t) * H * S + i_h * S
    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * H + i_h)
    else:
        NS = S
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    p_q = tl.make_block_ptr(q, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0)
        )
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    p_do = tl.make_block_ptr(do, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G,
        BV), (1, 0))
    p_lse = lse + i_h * G + tl.arange(0, G)
    p_delta = delta + i_h * G + tl.arange(0, G)
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_lse = tl.load(p_lse)
    b_delta = tl.load(p_delta)
    b_dq = tl.zeros([G, BK], dtype=tl.float32)
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS
        if i_s <= i_t and i_s >= 0:
            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK,
                BS), (0, 1))
            p_v = tl.make_block_ptr(v, (V, T), (1, H * V), (i_v * BV, i_s),
                (BV, BS), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_s = tl.dot(b_q, b_k)
            b_p = exp(b_s - b_lse[:, None])
            b_p = tl.where((i_t >= i_s + tl.arange(0, BS))[None, :], b_p, 0)
            b_dp = tl.dot(b_do, b_v)
            b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
            b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
    b_dq *= scale
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({'USE_BLOCK_COUNTS': lambda args: isinstance(args[
    'block_counts'], torch.Tensor)})
@triton.jit(do_not_specialize=['T'])
def parallel_nsa_kernel_mask(block_indices, block_counts, block_mask, T, H:
    tl.constexpr, S: tl.constexpr, BS: tl.constexpr, NS: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr):
    i_t, i_b, i_hs = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_s = i_hs // S, i_hs % S
    b_i = tl.load(block_indices + i_b * T * H * S + i_t * H * S + i_h * S + i_s
        )
    if USE_BLOCK_COUNTS:
        b_m = b_i * BS <= i_t and i_s < tl.load(block_counts + i_b * T * H +
            i_t * H + i_h)
    else:
        b_m = b_i * BS <= i_t
    if b_i < NS and b_i >= 0:
        tl.store(block_mask + i_b * T * H * NS + i_t * H * NS + i_h * NS +
            b_i, b_m.to(block_mask.dtype.element_ty))


def parallel_attn_bwd_preprocess(o: torch.Tensor, do: torch.Tensor):
    V = o.shape[-1]
    delta = torch.empty_like(o[..., 0], dtype=torch.float)
    parallel_attn_bwd_kernel_preprocess[delta.numel(),](o=o, do=do, delta=
        delta, B=triton.next_power_of_2(V), V=V)
    return delta


def parallel_nsa_block_mask(block_indices: torch.LongTensor, block_counts:
    Union[torch.LongTensor, int], cu_seqlens: torch.LongTensor, block_size: int
    ):
    B, T, H, S = block_indices.shape
    BS = block_size
    if cu_seqlens is not None:
        NS = triton.cdiv(prepare_lens(cu_seqlens).max().item(), BS)
    else:
        NS = triton.cdiv(T, BS)
    block_mask = torch.zeros(B, T, H, NS, dtype=torch.bool, device=
        block_indices.device)
    parallel_nsa_kernel_mask[T, B, H * S](block_indices=block_indices,
        block_counts=block_counts, block_mask=block_mask, T=T, H=H, S=S, BS
        =BS, NS=NS)
    return block_mask


def parallel_nsa_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o:
    torch.Tensor, lse: torch.Tensor, do: torch.Tensor, block_indices: torch
    .Tensor, block_counts: Union[torch.LongTensor, int], block_size: int=64,
    scale: float=None, cu_seqlens: Optional[torch.LongTensor]=None,
    token_indices: Optional[torch.LongTensor]=None):
    B, T, H, K, V, S = *k.shape, v.shape[-1], block_indices.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BS = block_size
    BK = max(triton.next_power_of_2(K), 16)
    BV = min(128, max(triton.next_power_of_2(v.shape[-1]), 16))
    NV = triton.cdiv(V, BV)
    delta = parallel_attn_bwd_preprocess(o, do)
    dq = torch.empty(NV, *q.shape, dtype=q.dtype if NV == 1 else torch.
        float, device=q.device)
    grid = T, NV, B * H
    parallel_nsa_bwd_kernel_dq[grid](q=q, k=k, v=v, lse=lse, delta=delta,
        do=do, dq=dq, block_indices=block_indices, block_counts=
        block_counts, cu_seqlens=cu_seqlens, token_indices=token_indices,
        scale=scale, T=T, B=B, H=H, HQ=HQ, G=G, K=K, V=V, S=S, BS=BS, BK=BK,
        BV=BV)
    dq = dq.sum(0)
    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BS)
        NS = len(chunk_indices)
    else:
        chunk_indices = None
        NS = triton.cdiv(T, BS)
    block_mask = parallel_nsa_block_mask(block_indices, block_counts,
        cu_seqlens, block_size)
    dk = torch.empty(NV, *k.shape, dtype=k.dtype if NV == 1 else torch.
        float, device=q.device)
    dv = torch.empty(v.shape, dtype=v.dtype, device=q.device)
    grid = NV, NS, B * H
    parallel_nsa_bwd_kernel_dkv[grid](q=q, k=k, v=v, lse=lse, delta=delta,
        do=do, dk=dk, dv=dv, block_mask=block_mask, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, scale=scale, T=T, B=B, H=H, HQ=HQ, G=G,
        K=K, V=V, M=block_mask.shape[-1], BS=BS, BK=BK, BV=BV)
    dk = dk.sum(0)
    return dq, dk, dv


@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int
    ) ->torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(
        cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens
        )


# Backward method (kernel launch code)
@contiguous
@autocast_custom_bwd
def _ParallelNSAFunction_backward(ctx, do):
    q, k, v, o, lse = ctx.saved_tensors
    dq, dk, dv = parallel_nsa_bwd(q=q, k=k, v=v, o=o, lse=lse, do=do,
        block_indices=ctx.block_indices, block_counts=ctx.block_counts,
        block_size=ctx.block_size, scale=ctx.scale, cu_seqlens=ctx.
        cu_seqlens, token_indices=ctx.token_indices)
    return dq.to(q), dk.to(k), dv.to(v
        ), None, None, None, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

@torch.compile
class ParallelNSAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, block_indices, block_counts, block_size,
        scale, cu_seqlens):
        ctx.dtype = q.dtype
        token_indices = prepare_token_indices(cu_seqlens
            ) if cu_seqlens is not None else None
        o, lse = parallel_nsa_fwd(q=q, k=k, v=v, block_indices=
            block_indices, block_counts=block_counts, block_size=block_size,
            scale=scale, cu_seqlens=cu_seqlens, token_indices=token_indices)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.block_indices = block_indices
        ctx.block_counts = block_counts
        ctx.cu_seqlens = cu_seqlens
        ctx.token_indices = token_indices
        ctx.block_size = block_size
        ctx.scale = scale
        return o.to(q.dtype)

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = parallel_nsa_bwd(q=q, k=k, v=v, o=o, lse=lse, do=do,
            block_indices=ctx.block_indices, block_counts=ctx.block_counts,
            block_size=ctx.block_size, scale=ctx.scale, cu_seqlens=ctx.
            cu_seqlens, token_indices=ctx.token_indices)
        return dq.to(q), dk.to(k), dv.to(v
            ), None, None, None, None, None, None, None, None
