# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/fla-org/flash-bidirectional-linear-attention
# Source-Files: fbi_la/ops/simple_la/attention.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_zdgo1av7/flash-bidirectional-linear-attention-main/fbi_la/ops/simple_la/attention.py
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
from triton import cdiv

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=[triton.Config({'BK': BK, 'BV': BV}, num_warps=
    num_warps, num_stages=num_stages) for BK in [32, 64] for BV in [32, 64] for
    num_warps in [2, 4, 8] for num_stages in [2, 3, 4]], key=['L'])
@triton.jit
def _fwd_kv_kernel(K, V, S, stride_qk_bh, stride_qk_l, stride_qk_d,
    stride_vo_bh, stride_vo_l, stride_vo_d, stride_s_bh, stride_s_dk,
    stride_s_dv, scale, L: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
    BL: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    start_v, start_k, off_bs_head = tl.program_id(0), tl.program_id(1
        ), tl.program_id(2)
    K_block_ptr = tl.make_block_ptr(base=K + off_bs_head * stride_qk_bh,
        shape=(DK, L), strides=(stride_qk_d, stride_qk_l), offsets=(start_k *
        BK, 0), block_shape=(BK, BL), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(base=V + off_bs_head * stride_vo_bh,
        shape=(L, DV), strides=(stride_vo_l, stride_vo_d), offsets=(0, 
        start_v * BV), block_shape=(BL, BV), order=(1, 0))
    s = tl.zeros([BK, BV], dtype=tl.float32)
    for _ in range(0, L, BL):
        k = tl.load(K_block_ptr, boundary_check=(0, 1))
        v = tl.load(V_block_ptr, boundary_check=(0, 1))
        v = (v * scale).to(v.dtype)
        s += tl.dot(k, v, allow_tf32=False)
        K_block_ptr = tl.advance(K_block_ptr, (0, BL))
        V_block_ptr = tl.advance(V_block_ptr, (BL, 0))
    S_block_ptr = tl.make_block_ptr(base=S + off_bs_head * stride_s_bh,
        shape=(DK, DV), strides=(stride_s_dk, stride_s_dv), offsets=(
        start_k * BK, start_v * BV), block_shape=(BK, BV), order=(1, 0))
    tl.store(S_block_ptr, s.to(S.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({'BL': BL, 'BV': BV}, num_warps=
    num_warps, num_stages=num_stages) for BL in [32, 64] for BV in [32, 64] for
    num_warps in [2, 4, 8] for num_stages in [2, 3, 4]], key=['L'])
@triton.jit
def _fwd_qs_kernel(Q, S, O, stride_qk_bh, stride_qk_l, stride_qk_d,
    stride_vo_bh, stride_vo_l, stride_vo_d, stride_s_bh, stride_s_dk,
    stride_s_dv, L: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BL:
    tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    start_v, start_m, off_bs_head = tl.program_id(0), tl.program_id(1
        ), tl.program_id(2)
    Q_block_ptr = tl.make_block_ptr(base=Q + off_bs_head * stride_qk_bh,
        shape=(L, DK), strides=(stride_qk_l, stride_qk_d), offsets=(start_m *
        BL, 0), block_shape=(BL, BK), order=(1, 0))
    S_block_ptr = tl.make_block_ptr(base=S + off_bs_head * stride_s_bh,
        shape=(DK, DV), strides=(stride_s_dk, stride_s_dv), offsets=(0, 
        start_v * BV), block_shape=(BK, BV), order=(1, 0))
    o = tl.zeros([BL, BV], dtype=tl.float32)
    for _ in range(0, DK, BK):
        q = tl.load(Q_block_ptr, boundary_check=(0, 1))
        s = tl.load(S_block_ptr, boundary_check=(0, 1))
        o += tl.dot(q, s, allow_tf32=False)
        Q_block_ptr = tl.advance(Q_block_ptr, (0, BK))
        S_block_ptr = tl.advance(S_block_ptr, (BK, 0))
    O_block_ptr = tl.make_block_ptr(base=O + off_bs_head * stride_vo_bh,
        shape=(L, DV), strides=(stride_vo_l, stride_vo_d), offsets=(start_m *
        BL, start_v * BV), block_shape=(BL, BV), order=(1, 0))
    tl.store(O_block_ptr, o.to(O.dtype.element_ty), boundary_check=(0, 1))


# Forward method (kernel launch code)
def _SimpleLinearAttnFunction_forward(ctx, q, k, v, scale):
    B, H, L, K, V = *q.shape, v.shape[-1]
    BL = min(128, max(16, triton.next_power_of_2(L)))
    grid = lambda meta: (triton.cdiv(V, meta['BV']), triton.cdiv(K, meta[
        'BK']), B * H)
    s = torch.empty(B, H, K, V, device=k.device)
    _fwd_kv_kernel[grid](k, v, s, k.stride(1), k.stride(2), k.stride(3), v.
        stride(1), v.stride(2), v.stride(3), s.stride(1), s.stride(2), s.
        stride(3), scale, L=L, DK=K, DV=V, BL=BL)
    s = s.to(k.dtype)
    BK = min(64, max(16, triton.next_power_of_2(K)))
    grid = lambda meta: (triton.cdiv(V, meta['BV']), triton.cdiv(L, meta[
        'BL']), B * H)
    o = torch.empty_like(v)
    _fwd_qs_kernel[grid](q, s, o, q.stride(1), q.stride(2), q.stride(3), v.
        stride(1), v.stride(2), v.stride(3), s.stride(1), s.stride(2), s.
        stride(3), L=L, DK=K, DV=V, BK=BK)
    ctx.save_for_backward(q, k, v, s)
    ctx.scale = scale
    return o


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=[triton.Config({'BL': BL, 'BK': BK}, num_warps=
    num_warps, num_stages=num_stages) for BL in [32, 64] for BK in [32, 64] for
    num_warps in [2, 4, 8] for num_stages in [2, 3, 4]], key=['L'])
@triton.jit
def _bwd_dqk_kernel(V, S, dQ, dK, dS, dO, stride_qk_bh, stride_qk_l,
    stride_qk_d, stride_vo_bh, stride_vo_l, stride_vo_d, stride_s_bh,
    stride_s_dk, stride_s_dv, scale, L: tl.constexpr, DK: tl.constexpr, DV:
    tl.constexpr, BL: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    start_k, start_m, off_bs_head = tl.program_id(0), tl.program_id(1
        ), tl.program_id(2)
    V_block_ptr = tl.make_block_ptr(base=V + off_bs_head * stride_vo_bh,
        shape=(L, DV), strides=(stride_vo_l, stride_vo_d), offsets=(start_m *
        BL, 0), block_shape=(BL, BV), order=(1, 0))
    dO_block_ptr = tl.make_block_ptr(base=dO + off_bs_head * stride_vo_bh,
        shape=(L, DV), strides=(stride_vo_l, stride_vo_d), offsets=(start_m *
        BL, 0), block_shape=(BL, BV), order=(1, 0))
    S_block_ptr = tl.make_block_ptr(base=S + off_bs_head * stride_s_bh,
        shape=(DV, DK), strides=(stride_s_dv, stride_s_dk), offsets=(0, 
        start_k * BK), block_shape=(BV, BK), order=(0, 1))
    dS_block_ptr = tl.make_block_ptr(base=dS + off_bs_head * stride_s_bh,
        shape=(DV, DK), strides=(stride_s_dv, stride_s_dk), offsets=(0, 
        start_k * BK), block_shape=(BV, BK), order=(0, 1))
    dq = tl.zeros([BL, BK], dtype=tl.float32)
    dk = tl.zeros([BL, BK], dtype=tl.float32)
    for _ in range(0, DV, BV):
        v = tl.load(V_block_ptr, boundary_check=(0, 1))
        do = tl.load(dO_block_ptr, boundary_check=(0, 1))
        s = tl.load(S_block_ptr, boundary_check=(0, 1))
        ds = tl.load(dS_block_ptr, boundary_check=(0, 1))
        v = (v * scale).to(v.dtype)
        dq += tl.dot(do, s.to(do.dtype), allow_tf32=False)
        dk += tl.dot(v, ds.to(v.dtype), allow_tf32=False)
        V_block_ptr = tl.advance(V_block_ptr, (0, BV))
        dS_block_ptr = tl.advance(dS_block_ptr, (BV, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (0, BV))
        S_block_ptr = tl.advance(S_block_ptr, (BV, 0))
    dQ_block_ptr = tl.make_block_ptr(base=dQ + off_bs_head * stride_qk_bh,
        shape=(L, DK), strides=(stride_qk_l, stride_qk_d), offsets=(start_m *
        BL, start_k * BK), block_shape=(BL, BK), order=(1, 0))
    dK_block_ptr = tl.make_block_ptr(base=dK + off_bs_head * stride_qk_bh,
        shape=(L, DK), strides=(stride_qk_l, stride_qk_d), offsets=(start_m *
        BL, start_k * BK), block_shape=(BL, BK), order=(1, 0))
    tl.store(dQ_block_ptr, dq.to(dQ.dtype.element_ty), boundary_check=(0, 1))
    tl.store(dK_block_ptr, dk.to(dK.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({'BK': BK, 'BV': BV}, num_warps=
    num_warps, num_stages=num_stages) for BK in [32, 64] for BV in [32, 64] for
    num_warps in [2, 4, 8] for num_stages in [2, 3, 4]], key=['L'])
@triton.jit
def _bwd_ds_kernel(Q, DO, DS, stride_qk_bh, stride_qk_l, stride_qk_d,
    stride_vo_bh, stride_vo_l, stride_vo_d, stride_s_bh, stride_s_dk,
    stride_s_dv, L: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BL:
    tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    start_v, start_k, off_bs_head = tl.program_id(0), tl.program_id(1
        ), tl.program_id(2)
    Q_block_ptr = tl.make_block_ptr(base=Q + off_bs_head * stride_qk_bh,
        shape=(DK, L), strides=(stride_qk_d, stride_qk_l), offsets=(start_k *
        BK, 0), block_shape=(BK, BL), order=(0, 1))
    DO_block_ptr = tl.make_block_ptr(base=DO + off_bs_head * stride_vo_bh,
        shape=(L, DV), strides=(stride_vo_l, stride_vo_d), offsets=(0, 
        start_v * BV), block_shape=(BL, BV), order=(1, 0))
    ds = tl.zeros([BK, BV], dtype=tl.float32)
    for i in range(0, L, BL):
        q = tl.load(Q_block_ptr, boundary_check=(0, 1))
        do = tl.load(DO_block_ptr, boundary_check=(0, 1))
        ds += tl.dot(q, do, allow_tf32=False)
        Q_block_ptr = tl.advance(Q_block_ptr, (0, BL))
        DO_block_ptr = tl.advance(DO_block_ptr, (BL, 0))
    DS_block_ptr = tl.make_block_ptr(base=DS + off_bs_head * stride_s_bh,
        shape=(DK, DV), strides=(stride_s_dk, stride_s_dv), offsets=(
        start_k * BK, start_v * BV), block_shape=(BK, BV), order=(1, 0))
    tl.store(DS_block_ptr, ds.to(DS.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({'BL': BL, 'BV': BV}, num_warps=
    num_warps, num_stages=num_stages) for BL in [32, 64] for BV in [32, 64] for
    num_warps in [2, 4, 8] for num_stages in [2, 3, 4]], key=['L'])
@triton.jit
def _bwd_dv_kernel(K, dV, dS, stride_qk_bh, stride_qk_l, stride_qk_d,
    stride_vo_bh, stride_vo_l, stride_vo_d, stride_s_bh, stride_s_dk,
    stride_s_dv, scale, L: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
    BL: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    start_v, start_m, off_bs_head = tl.program_id(0), tl.program_id(1
        ), tl.program_id(2)
    K_block_ptr = tl.make_block_ptr(base=K + off_bs_head * stride_qk_bh,
        shape=(L, DK), strides=(stride_qk_l, stride_qk_d), offsets=(start_m *
        BL, 0), block_shape=(BL, BK), order=(1, 0))
    dS_block_ptr = tl.make_block_ptr(base=dS + off_bs_head * stride_s_bh,
        shape=(DK, DV), strides=(stride_s_dk, stride_s_dv), offsets=(0, 
        start_v * BV), block_shape=(BK, BV), order=(1, 0))
    dv = tl.zeros([BL, BV], dtype=tl.float32)
    for _ in range(0, DK, BK):
        k = tl.load(K_block_ptr, boundary_check=(0, 1))
        ds = tl.load(dS_block_ptr, boundary_check=(0, 1))
        dv += tl.dot(k, ds.to(k.dtype), allow_tf32=False) * scale
        K_block_ptr = tl.advance(K_block_ptr, (0, BK))
        dS_block_ptr = tl.advance(dS_block_ptr, (BK, 0))
    dV_block_ptr = tl.make_block_ptr(base=dV + off_bs_head * stride_vo_bh,
        shape=(L, DV), strides=(stride_vo_l, stride_vo_d), offsets=(start_m *
        BL, start_v * BV), block_shape=(BL, BV), order=(1, 0))
    tl.store(dV_block_ptr, dv.to(dV.dtype.element_ty), boundary_check=(0, 1))


# Backward method (kernel launch code)
def _SimpleLinearAttnFunction_backward(ctx, do):
    q, k, v, s = ctx.saved_tensors
    B, H, L, K, V = *q.shape, v.shape[-1]
    BL = min(128, max(16, triton.next_power_of_2(L)))
    grid = lambda meta: (triton.cdiv(V, meta['BV']), triton.cdiv(K, meta[
        'BK']), B * H)
    ds = torch.empty(B, H, K, V, device=k.device)
    _bwd_ds_kernel[grid](q, do, ds, q.stride(1), q.stride(2), q.stride(3),
        v.stride(1), v.stride(2), v.stride(3), ds.stride(1), ds.stride(2),
        ds.stride(3), L=L, DK=K, DV=V, BL=BL)
    BV = min(64, max(16, triton.next_power_of_2(V)))
    grid = lambda meta: (triton.cdiv(K, meta['BK']), triton.cdiv(L, meta[
        'BL']), B * H)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    _bwd_dqk_kernel[grid](v, s, dq, dk, ds, do, k.stride(1), k.stride(2), k
        .stride(3), v.stride(1), v.stride(2), v.stride(3), ds.stride(1), ds
        .stride(2), ds.stride(3), ctx.scale, L=L, DK=K, DV=V, BV=BV)
    BK = min(64, max(16, triton.next_power_of_2(K)))
    grid = lambda meta: (triton.cdiv(V, meta['BV']), triton.cdiv(L, meta[
        'BL']), B * H)
    _bwd_dv_kernel[grid](k, dv, ds, k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3), ds.stride(1), ds.stride(2),
        ds.stride(3), ctx.scale, L=L, DK=K, DV=V, BK=BK)
    return dq, dk, dv, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class SimpleLinearAttnFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, scale):
        B, H, L, K, V = *q.shape, v.shape[-1]
        BL = min(128, max(16, triton.next_power_of_2(L)))
        grid = lambda meta: (triton.cdiv(V, meta['BV']), triton.cdiv(K,
            meta['BK']), B * H)
        s = torch.empty(B, H, K, V, device=k.device)
        _fwd_kv_kernel[grid](k, v, s, k.stride(1), k.stride(2), k.stride(3),
            v.stride(1), v.stride(2), v.stride(3), s.stride(1), s.stride(2),
            s.stride(3), scale, L=L, DK=K, DV=V, BL=BL)
        s = s.to(k.dtype)
        BK = min(64, max(16, triton.next_power_of_2(K)))
        grid = lambda meta: (triton.cdiv(V, meta['BV']), triton.cdiv(L,
            meta['BL']), B * H)
        o = torch.empty_like(v)
        _fwd_qs_kernel[grid](q, s, o, q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3), s.stride(1), s.stride(2),
            s.stride(3), L=L, DK=K, DV=V, BK=BK)
        ctx.save_for_backward(q, k, v, s)
        ctx.scale = scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, s = ctx.saved_tensors
        B, H, L, K, V = *q.shape, v.shape[-1]
        BL = min(128, max(16, triton.next_power_of_2(L)))
        grid = lambda meta: (triton.cdiv(V, meta['BV']), triton.cdiv(K,
            meta['BK']), B * H)
        ds = torch.empty(B, H, K, V, device=k.device)
        _bwd_ds_kernel[grid](q, do, ds, q.stride(1), q.stride(2), q.stride(
            3), v.stride(1), v.stride(2), v.stride(3), ds.stride(1), ds.
            stride(2), ds.stride(3), L=L, DK=K, DV=V, BL=BL)
        BV = min(64, max(16, triton.next_power_of_2(V)))
        grid = lambda meta: (triton.cdiv(K, meta['BK']), triton.cdiv(L,
            meta['BL']), B * H)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        _bwd_dqk_kernel[grid](v, s, dq, dk, ds, do, k.stride(1), k.stride(2
            ), k.stride(3), v.stride(1), v.stride(2), v.stride(3), ds.
            stride(1), ds.stride(2), ds.stride(3), ctx.scale, L=L, DK=K, DV
            =V, BV=BV)
        BK = min(64, max(16, triton.next_power_of_2(K)))
        grid = lambda meta: (triton.cdiv(V, meta['BV']), triton.cdiv(L,
            meta['BL']), B * H)
        _bwd_dv_kernel[grid](k, dv, ds, k.stride(1), k.stride(2), k.stride(
            3), v.stride(1), v.stride(2), v.stride(3), ds.stride(1), ds.
            stride(2), ds.stride(3), ctx.scale, L=L, DK=K, DV=V, BK=BK)
        return dq, dk, dv, None, None, None
