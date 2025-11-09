# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/fla-org/native-sparse-attention
# Source-Files: native_sparse_attention/ops/parallel.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rnwedfp2/native-sparse-attention-main/native_sparse_attention/ops/parallel.py
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

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.heuristics({'USE_OFFSETS': lambda args: args['offsets'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for
    num_warps in [1, 2, 4]], key=['BS', 'BK', 'BV'])
@triton.jit
def parallel_nsa_compression_fwd_kernel(q, k, v, o, lse, scale, offsets,
    token_indices, chunk_offsets, T, H: tl.constexpr, HQ: tl.constexpr, G:
    tl.constexpr, K: tl.constexpr, V: tl.constexpr, BC: tl.constexpr, BS:
    tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_OFFSETS: tl.constexpr
    ):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(
            token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets +
            i_n + 1).to(tl.int32)
        T = eos - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
        boc = i_b * tl.cdiv(T, BS)
    p_q = tl.make_block_ptr(q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h *
        G, 0), (G, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    TC = tl.cdiv(T, BS)
    NC = (i_t + 1) // BS
    p_o = tl.make_block_ptr(o + (bos + i_t) * HQ * V, (HQ, V), (V, 1), (i_h *
        G, i_v * BV), (G, BV), (1, 0))
    b_o = tl.zeros([G, BV], dtype=tl.float32)
    b_m = tl.full([G], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([G], dtype=tl.float32)
    for i_c in range(0, NC, BC):
        o_c = i_c + tl.arange(0, BC)
        p_k = tl.make_block_ptr(k + (boc * H + i_h) * K, (K, TC), (1, H * K
            ), (0, i_c), (BK, BC), (0, 1))
        p_v = tl.make_block_ptr(v + (boc * H + i_h) * V, (TC, V), (H * V, 1
            ), (i_c, i_v * BV), (BC, BV), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k)
        b_s = tl.where((o_c < NC)[None, :], b_s, float('-inf'))
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_r = tl.exp(b_mp - b_m)
        b_p = tl.exp(b_s - b_m[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)
        b_mp = b_m
    if NC == 0:
        b_lse = tl.zeros([G], dtype=tl.float32)
    else:
        b_o = b_o / b_acc[:, None]
        b_lse = b_m + tl.log(b_acc)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    if i_v == 0:
        tl.store(lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G), b_lse.
            to(lse.dtype.element_ty))


def parallel_nsa_compression_fwd(q: torch.Tensor, k: torch.Tensor, v: torch
    .Tensor, block_size: int, scale: float, offsets: Optional[torch.
    LongTensor]=None, token_indices: Optional[torch.LongTensor]=None):
    B, T, HQ, K, V = *q.shape, v.shape[-1]
    H = k.shape[2]
    G = HQ // H
    BC = BS = block_size
    if torch.cuda.get_device_capability()[0] >= 9:
        BK = min(256, triton.next_power_of_2(K))
        BV = min(256, triton.next_power_of_2(V))
    else:
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, 'The key dimension can not be larger than 256'
    chunk_offsets = prepare_chunk_offsets(offsets, BS
        ) if offsets is not None else None
    grid = T, NV, B * H
    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
    parallel_nsa_compression_fwd_kernel[grid](q=q, k=k, v=v, o=o, lse=lse,
        scale=scale, offsets=offsets, token_indices=token_indices,
        chunk_offsets=chunk_offsets, T=T, H=H, HQ=HQ, G=G, K=K, V=V, BC=BC,
        BS=BS, BK=BK, BV=BV)
    return o, lse


# Forward method (kernel launch code)
@contiguous
@autocast_custom_fwd
def _ParallelNSACompressionFunction_forward(ctx, q, k, v, block_size, scale,
    offsets):
    ctx.dtype = q.dtype
    token_indices = prepare_token_indices(offsets
        ) if offsets is not None else None
    o, lse = parallel_nsa_compression_fwd(q=q, k=k, v=v, block_size=
        block_size, scale=scale, offsets=offsets, token_indices=token_indices)
    ctx.save_for_backward(q, k, v, o, lse)
    ctx.offsets = offsets
    ctx.token_indices = token_indices
    ctx.block_size = block_size
    ctx.scale = scale
    return o.to(q.dtype), lse


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def parallel_nsa_bwd_kernel_preprocess(o, do, delta, B: tl.constexpr, V: tl
    .constexpr):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < V
    b_o = tl.load(o + i_n * V + o_d, mask=m_d, other=0)
    b_do = tl.load(do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    b_delta = tl.sum(b_o * b_do)
    tl.store(delta + i_n, b_delta.to(delta.dtype.element_ty))


@triton.heuristics({'USE_OFFSETS': lambda args: args['offsets'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for
    num_warps in [1, 2, 4]], key=['BS', 'BK', 'BV'])
@triton.jit(do_not_specialize=['T'])
def parallel_nsa_compression_bwd_kernel_dkv(q, k, v, lse, delta, do, dk, dv,
    offsets, chunk_indices, chunk_offsets, scale, T, B: tl.constexpr, H: tl
    .constexpr, HQ: tl.constexpr, G: tl.constexpr, K: tl.constexpr, V: tl.
    constexpr, BC: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl
    .constexpr, USE_OFFSETS: tl.constexpr):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_n, i_c = tl.load(chunk_indices + i_c * 2).to(tl.int32), tl.load(
            chunk_indices + i_c * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets +
            i_n + 1).to(tl.int32)
        T = eos - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
        boc = i_b * tl.cdiv(T, BS)
    TC = tl.cdiv(T, BS)
    p_k = tl.make_block_ptr(k + (boc * H + i_h) * K, (TC, K), (H * K, 1), (
        i_c * BC, 0), (BC, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (boc * H + i_h) * V, (TC, V), (H * V, 1), (
        i_c * BC, i_v * BV), (BC, BV), (1, 0))
    p_dk = tl.make_block_ptr(dk + (i_v * B * T * H + boc * H + i_h) * K, (
        TC, K), (H * K, 1), (i_c * BC, 0), (BC, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (i_v * B * T * H + boc * H + i_h) * V, (
        TC, V), (H * V, 1), (i_c * BC, i_v * BV), (BC, BV), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BC, BV], dtype=tl.float32)
    for i in range(i_c * BC * BS, T):
        o_c = i_c * BC + tl.arange(0, BC)
        p_q = tl.make_block_ptr(q + (bos + i) * HQ * K, (HQ, K), (K, 1), (
            i_h * G, 0), (G, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        p_do = tl.make_block_ptr(do + (bos + i) * HQ * V, (HQ, V), (V, 1),
            (i_h * G, i_v * BV), (G, BV), (1, 0))
        p_lse = lse + (bos + i) * HQ + i_h * G + tl.arange(0, G)
        p_delta = delta + (bos + i) * HQ + i_h * G + tl.arange(0, G)
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_lse = tl.load(p_lse)
        b_delta = tl.load(p_delta)
        b_s = tl.dot(b_k, tl.trans(b_q))
        b_p = tl.exp(b_s - b_lse[None, :])
        b_p = tl.where((i >= max(0, (o_c + 1) * BS - 1))[:, None], b_p, 0)
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        b_dp = tl.dot(b_v, tl.trans(b_do))
        b_ds = b_p * (b_dp - b_delta[None, :])
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({'USE_OFFSETS': lambda args: args['offsets'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for
    num_warps in [1, 2, 4]], key=['BS', 'BK', 'BV'])
@triton.jit(do_not_specialize=['T'])
def parallel_nsa_compression_bwd_kernel_dq(q, k, v, lse, delta, do, dq,
    scale, offsets, token_indices, chunk_offsets, T, B: tl.constexpr, H: tl
    .constexpr, HQ: tl.constexpr, G: tl.constexpr, K: tl.constexpr, V: tl.
    constexpr, BC: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl
    .constexpr, USE_OFFSETS: tl.constexpr):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(
            token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets +
            i_n + 1).to(tl.int32)
        T = eos - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
        boc = i_b * tl.cdiv(T, BS)
    q += (bos + i_t) * HQ * K
    do += (bos + i_t) * HQ * V
    lse += (bos + i_t) * HQ
    delta += (bos + i_t) * HQ
    dq += (i_v * B * T + bos + i_t) * HQ * K
    p_q = tl.make_block_ptr(q, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0)
        )
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    p_do = tl.make_block_ptr(do, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G,
        BV), (1, 0))
    p_lse = lse + i_h * G + tl.arange(0, G)
    p_delta = delta + i_h * G + tl.arange(0, G)
    TC = tl.cdiv(T, BS)
    NC = (i_t + 1) // BS
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_lse = tl.load(p_lse)
    b_delta = tl.load(p_delta)
    b_dq = tl.zeros([G, BK], dtype=tl.float32)
    for i_c in range(0, NC, BC):
        o_c = i_c + tl.arange(0, BC)
        p_k = tl.make_block_ptr(k + (boc * H + i_h) * K, (K, TC), (1, H * K
            ), (0, i_c), (BK, BC), (0, 1))
        p_v = tl.make_block_ptr(v + (boc * H + i_h) * V, (V, TC), (1, H * V
            ), (i_v * BV, i_c), (BV, BC), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k)
        b_p = tl.exp(b_s - b_lse[:, None])
        b_p = tl.where((o_c < NC)[None, :], b_p, 0)
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
    b_dq *= scale
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


def parallel_nsa_bwd_preprocess(o: torch.Tensor, do: torch.Tensor):
    V = o.shape[-1]
    delta = torch.empty_like(o[..., 0], dtype=torch.float32)
    parallel_nsa_bwd_kernel_preprocess[delta.numel(),](o=o, do=do, delta=
        delta, B=triton.next_power_of_2(V), V=V)
    return delta


def parallel_nsa_compression_bwd(q: torch.Tensor, k: torch.Tensor, v: torch
    .Tensor, o: torch.Tensor, lse: torch.Tensor, do: torch.Tensor,
    block_size: int=64, scale: float=None, offsets: Optional[torch.
    LongTensor]=None, token_indices: Optional[torch.LongTensor]=None):
    B, T, HQ, K, V = *q.shape, v.shape[-1]
    H = k.shape[2]
    G = HQ // H
    BC = BS = block_size
    BK = triton.next_power_of_2(K)
    BV = min(128, triton.next_power_of_2(v.shape[-1]))
    NV = triton.cdiv(V, BV)
    if offsets is not None:
        lens = prepare_lens(offsets)
        chunk_indices = torch.cat([torch.arange(n) for n in triton.cdiv(
            triton.cdiv(lens, BS), BC).tolist()])
        chunk_indices = torch.stack([chunk_indices.eq(0).cumsum(0) - 1,
            chunk_indices], 1).to(offsets)
        chunk_offsets = prepare_chunk_offsets(offsets, BS)
        NC = len(chunk_indices)
    else:
        chunk_indices, chunk_offsets = None, None
        NC = triton.cdiv(triton.cdiv(T, BS), BC)
    delta = parallel_nsa_bwd_preprocess(o, do)
    dq = torch.empty(NV, *q.shape, dtype=q.dtype if NV == 1 else torch.
        float, device=q.device)
    grid = T, NV, B * H
    parallel_nsa_compression_bwd_kernel_dq[grid](q=q, k=k, v=v, lse=lse,
        delta=delta, do=do, dq=dq, scale=scale, offsets=offsets,
        token_indices=token_indices, chunk_offsets=chunk_offsets, T=T, B=B,
        H=H, HQ=HQ, G=G, K=K, V=V, BC=BC, BS=BS, BK=BK, BV=BV)
    dq = dq.sum(0)
    dk = torch.empty(NV, *k.shape, dtype=k.dtype if NV == 1 else torch.
        float, device=q.device)
    dv = torch.empty(v.shape, dtype=v.dtype, device=q.device)
    grid = NV, NC, B * H
    parallel_nsa_compression_bwd_kernel_dkv[grid](q=q, k=k, v=v, lse=lse,
        delta=delta, do=do, dk=dk, dv=dv, offsets=offsets, chunk_indices=
        chunk_indices, chunk_offsets=chunk_offsets, scale=scale, T=T, B=B,
        H=H, HQ=HQ, G=G, K=K, V=V, BC=BC, BS=BS, BK=BK, BV=BV)
    dk = dk.sum(0)
    return dq, dk, dv


# Backward method (kernel launch code)
@contiguous
@autocast_custom_bwd
def _ParallelNSACompressionFunction_backward(ctx, do, *args):
    q, k, v, o, lse = ctx.saved_tensors
    dq, dk, dv = parallel_nsa_compression_bwd(q=q, k=k, v=v, o=o, lse=lse,
        do=do, block_size=ctx.block_size, scale=ctx.scale, offsets=ctx.
        offsets, token_indices=ctx.token_indices)
    return dq.to(q), dk.to(k), dv.to(v), None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class ParallelNSACompressionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, block_size, scale, offsets):
        ctx.dtype = q.dtype
        token_indices = prepare_token_indices(offsets
            ) if offsets is not None else None
        o, lse = parallel_nsa_compression_fwd(q=q, k=k, v=v, block_size=
            block_size, scale=scale, offsets=offsets, token_indices=
            token_indices)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.offsets = offsets
        ctx.token_indices = token_indices
        ctx.block_size = block_size
        ctx.scale = scale
        return o.to(q.dtype), lse

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, *args):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = parallel_nsa_compression_bwd(q=q, k=k, v=v, o=o, lse=
            lse, do=do, block_size=ctx.block_size, scale=ctx.scale, offsets
            =ctx.offsets, token_indices=ctx.token_indices)
        return dq.to(q), dk.to(k), dv.to(v), None, None, None
