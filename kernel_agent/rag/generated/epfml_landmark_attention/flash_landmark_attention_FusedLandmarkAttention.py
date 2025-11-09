# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/epfml/landmark-attention
# Source-Files: llama/ltriton/flash_landmark_attention.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_vujmck4r/landmark-attention-main/llama/ltriton/flash_landmark_attention.py
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

@triton.jit
def _fwd_kernel(Q, K, V, sm_scale, Out, sqz, sqh, sqm, sqd, skz, skh, skn,
    skd, svz, svh, svn, svd, soz, soh, som, sod, L, M, Z, H, N_CTX_Q,
    N_CTX_KV, BLOCK: tl.constexpr, BLOCK_DMODEL: tl.constexpr, N_PREFIX_Q:
    tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    BLOCK_M: tl.constexpr = BLOCK
    BLOCK_N: tl.constexpr = BLOCK
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m_real = (start_m + N_PREFIX_Q) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m_real += tl.where(tl.arange(0, BLOCK_M) == BLOCK_M - 1, -1, 0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_q = off_hz * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd
    offs_k = off_hz * skh + offs_n[None, :] * skn + offs_d[:, None] * skd
    offs_v = off_hz * svh + offs_n[:, None] * svn + offs_d[None, :] * svd
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    q_vals = tl.load(Q + offs_q, mask=offs_m[:, None] < N_CTX_Q, other=0)
    for start_n in range(0, N_PREFIX_Q + start_m):
        k_vals = tl.load(K + offs_k, mask=offs_n[None, :] < N_CTX_KV, other=0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=q_vals.dtype)
        qk += tl.dot(q_vals, k_vals, allow_tf32=False)
        qk *= sm_scale
        qk = tl.where(offs_m_real[:, None] >= offs_n[None, :], qk, float(
            '-inf'))
        landmark_qk = tl.max(tl.where(tl.arange(0, BLOCK_N)[None, :] == 
            BLOCK_N - 1, qk, float('-inf')), 1)
        normal_qk = tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1,
            float('-inf'), qk)
        normal_m = tl.max(normal_qk, 1)
        normal_p = tl.exp(normal_qk - normal_m[:, None])
        normal_denom = tl.sum(normal_p, 1)
        m_curr = tl.maximum(landmark_qk, m_prev)
        m_curr_ = m_curr
        l_prev *= tl.exp(m_prev - m_curr_)
        landmark_p = tl.exp(landmark_qk - m_curr_)
        l_curr = landmark_p + l_prev
        l_rcp = 1.0 / l_curr
        landmark_p *= l_rcp
        acc *= (l_prev * l_rcp)[:, None]
        v_vals = tl.load(V + offs_v, mask=offs_n[:, None] < N_CTX_KV, other=0)
        acc += tl.dot((landmark_p[:, None] * normal_p / normal_denom[:,
            None]).to(Q.dtype.element_ty), v_vals, allow_tf32=False)
        l_prev = l_curr
        m_prev = m_curr
        offs_n += BLOCK_N
        offs_k += BLOCK_N * skn
        offs_v += BLOCK_N * svn
    k_vals = tl.load(K + offs_k, mask=offs_n[None, :] < N_CTX_KV, other=0)
    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=q_vals.dtype)
    qk += tl.dot(q_vals, k_vals, allow_tf32=False)
    qk *= sm_scale
    qk = tl.where(offs_m_real[:, None] >= offs_n[None, :], qk, float('-inf'))
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    m_curr_ = m_curr
    l_prev *= tl.exp(m_prev - m_curr_)
    p = tl.exp(qk - m_curr_[:, None])
    l_curr = tl.sum(p, 1) + l_prev
    l_rcp = 1.0 / l_curr
    p *= l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None]
    p = p.to(Q.dtype.element_ty)
    v_vals = tl.load(V + offs_v, mask=offs_n[:, None] < N_CTX_KV, other=0)
    acc += tl.dot(p, v_vals, allow_tf32=False)
    l_prev = l_curr
    m_prev = m_curr
    offs_L = off_hz * N_CTX_Q + offs_m
    offs_M = off_hz * N_CTX_Q + offs_m
    tl.store(L + offs_L, l_prev, mask=offs_m < N_CTX_Q)
    tl.store(M + offs_M, m_prev, mask=offs_m < N_CTX_Q)
    offs_o = off_hz * soh + offs_m[:, None] * som + offs_d[None, :] * sod
    tl.store(Out + offs_o, acc, mask=offs_m[:, None] < N_CTX_Q)


# Forward method (kernel launch code)
def _FusedLandmarkAttention_forward(ctx, q, k, v, n_prefix_q, sm_scale,
    block_size):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    batch, nheads, seqlen_q, d = q.shape
    _, _, seqlen_k, _ = k.shape
    assert k.shape == (batch, nheads, seqlen_k, d)
    assert v.shape == (batch, nheads, seqlen_k, d)
    assert d <= 128, 'FlashAttention only support head dimensions up to 128'
    assert q.dtype == k.dtype == v.dtype, 'All tensors must have the same type'
    assert q.is_cuda and k.is_cuda and v.is_cuda
    BLOCK = block_size
    o = torch.empty_like(q)
    grid = triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1
    L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device,
        dtype=torch.float32)
    m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device,
        dtype=torch.float32)
    num_warps = 4 if d <= 64 else 8
    _fwd_kernel[grid](q, k, v, sm_scale, o, q.stride(0), q.stride(1), q.
        stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2), k.
        stride(3), v.stride(0), v.stride(1), v.stride(2), v.stride(3), o.
        stride(0), o.stride(1), o.stride(2), o.stride(3), L, m, q.shape[0],
        q.shape[1], q.shape[2], k.shape[2], BLOCK=BLOCK, BLOCK_DMODEL=d,
        N_PREFIX_Q=n_prefix_q, num_warps=num_warps, num_stages=2)
    ctx.save_for_backward(q, k, v, o, L, m)
    ctx.grid = grid
    ctx.sm_scale = sm_scale
    ctx.BLOCK_DMODEL = d
    ctx.N_PREFIX_Q = n_prefix_q
    ctx.BLOCK = BLOCK
    return o


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _bwd_kernel(Q, K, V, sm_scale, Out, DO, DQ, DK, DV, L, M, D, sqz, sqh,
    sqm, sqd, skz, skh, skn, skd, svz, svh, svn, svd, Z, H, N_CTX_Q,
    N_CTX_KV, BLOCK: tl.constexpr, BLOCK_DMODEL: tl.constexpr, N_PREFIX_Q:
    tl.constexpr):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    BLOCK_M: tl.constexpr = BLOCK
    BLOCK_N: tl.constexpr = BLOCK
    Q += off_z * sqz + off_h * sqh
    K += off_z * skz + off_h * skh
    V += off_z * svz + off_h * svh
    DO += off_z * sqz + off_h * sqh
    DQ += off_z * sqz + off_h * sqh
    DK += off_z * skz + off_h * skh
    DV += off_z * svz + off_h * svh
    offs_d = tl.arange(0, BLOCK_DMODEL)
    D_ptrs = D + off_hz * N_CTX_Q
    m_ptrs = M + off_hz * N_CTX_Q
    for start_n in range(0, N_CTX_KV, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K + (offs_n[:, None] * skn + offs_d[None, :] * skd)
        v_ptrs = V + (offs_n[:, None] * svn + offs_d[None, :] * svd)
        dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        if start_n < N_PREFIX_Q * BLOCK_M:
            start_q_index = 0
        elif N_CTX_Q <= start_n - N_PREFIX_Q * BLOCK_M:
            start_q_index = start_n - N_PREFIX_Q * BLOCK_M
        else:
            first_start_m = start_n - N_PREFIX_Q * BLOCK_M
            first_start_m = tl.multiple_of(first_start_m, BLOCK_M)
            offs_m = first_start_m + tl.arange(0, BLOCK_M)
            offs_m_real = offs_m + N_PREFIX_Q * BLOCK_M
            offs_m_real += tl.where(tl.arange(0, BLOCK_M) == BLOCK_M - 1, -1, 0
                )
            q_ptrs = Q + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            do_ptrs = DO + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            dq_ptrs = DQ + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            q = tl.load(q_ptrs)
            qk = tl.dot(q, tl.trans(k), allow_tf32=False)
            qk = tl.where(offs_m_real[:, None] >= offs_n[None, :], qk,
                float('-inf'))
            m = tl.load(m_ptrs + offs_m)
            m_ = m
            last_p = tl.exp(qk * sm_scale - m_[:, None])
            do = tl.load(do_ptrs)
            dv += tl.dot(tl.trans(last_p.to(Q.dtype.element_ty)), do,
                allow_tf32=False)
            Di = tl.load(D_ptrs + offs_m)
            last_dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:,
                None]
            last_dp += tl.dot(do, tl.trans(v), allow_tf32=False)
            ds = last_p * last_dp * sm_scale
            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q, allow_tf32
                =False)
            dq = tl.load(dq_ptrs)
            dq += tl.dot(ds.to(Q.dtype.element_ty), k, allow_tf32=False)
            tl.store(dq_ptrs, dq)
            start_q_index = first_start_m + BLOCK_M
        for start_m in range(start_q_index, N_CTX_Q, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            offs_m = start_m + tl.arange(0, BLOCK_M)
            q_ptrs = Q + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            do_ptrs = DO + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            dq_ptrs = DQ + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            q = tl.load(q_ptrs)
            qk = tl.dot(q, tl.trans(k), allow_tf32=False)
            qk *= sm_scale
            landmark_qk = tl.max(tl.where(tl.arange(0, BLOCK_N)[None, :] ==
                BLOCK_N - 1, qk, float('-inf')), 1)
            normal_qk = tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N -
                1, float('-inf'), qk)
            m = tl.load(m_ptrs + offs_m)
            m_ = m
            p = tl.exp(landmark_qk - m_)
            do = tl.load(do_ptrs)
            normal_m = tl.max(normal_qk, 1)
            normal_p = tl.exp(normal_qk - normal_m[:, None])
            normal_p_normalized = normal_p / tl.sum(normal_p, 1)[:, None]
            normal_kv = tl.dot(normal_p_normalized.to(Q.dtype.element_ty),
                v, allow_tf32=False)
            normal_D = tl.sum(do * normal_kv, 1)
            dv += tl.dot(tl.trans((p[:, None] * normal_p_normalized).to(Q.
                dtype.element_ty)), do, allow_tf32=False)
            Di = tl.load(D_ptrs + offs_m)
            dp = tl.zeros([BLOCK_M], dtype=tl.float32) - Di
            dp += normal_D
            landmark_ds = p * dp
            normal_dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32
                ) - normal_D[:, None]
            normal_dp += tl.dot(do, tl.trans(v), allow_tf32=False)
            normal_ds = p[:, None] * normal_p_normalized * normal_dp
            ds = tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1,
                landmark_ds[:, None], normal_ds)
            ds *= sm_scale
            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q, allow_tf32
                =False)
            dq = tl.load(dq_ptrs)
            dq += tl.dot(ds.to(Q.dtype.element_ty), k, allow_tf32=False)
            tl.store(dq_ptrs, dq)
        dv_ptrs = DV + (offs_n[:, None] * svn + offs_d[None, :] * svd)
        dk_ptrs = DK + (offs_n[:, None] * skn + offs_d[None, :] * skd)
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)


@triton.jit
def _bwd_preprocess(Out, soz, soh, som, sod, DO, L, slzh, slm, NewDO, Delta,
    N_CTX_Q, BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, D_HEAD)
    off_o = off_hz * soh + off_m[:, None] * som + off_d[None, :] * sod
    off_l = off_hz * slzh + off_m * slm
    o = tl.load(Out + off_o).to(tl.float32)
    do = tl.load(DO + off_o).to(tl.float32)
    denom = tl.load(L + off_l).to(tl.float32)
    do = do / denom[:, None]
    delta = tl.sum(o * do, axis=1)
    tl.store(NewDO + off_o, do)
    tl.store(Delta + off_l, delta)


# Backward method (kernel launch code)
def _FusedLandmarkAttention_backward(ctx, do):
    BLOCK = ctx.BLOCK
    q, k, v, o, l, m = ctx.saved_tensors
    assert q.shape[2] % BLOCK == 0, 'Backward supported only for full blocks'
    assert k.shape[2] % BLOCK == 0, 'Backward supported only for full blocks'
    do = do.contiguous()
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    do_scaled = torch.empty_like(do)
    delta = torch.empty_like(l)
    _bwd_preprocess[ctx.grid[0], ctx.grid[1]](o, o.stride(0), o.stride(1),
        o.stride(2), o.stride(3), do, l, l.stride(0), l.stride(1),
        do_scaled, delta, q.shape[2], BLOCK_M=BLOCK, D_HEAD=ctx.BLOCK_DMODEL)
    _bwd_kernel[ctx.grid[1],](q, k, v, ctx.sm_scale, o, do_scaled, dq, dk,
        dv, l, m, delta, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3), v.stride(0), v.
        stride(1), v.stride(2), v.stride(3), q.shape[0], q.shape[1], q.
        shape[2], k.shape[2], BLOCK=BLOCK, BLOCK_DMODEL=ctx.BLOCK_DMODEL,
        N_PREFIX_Q=ctx.N_PREFIX_Q, num_warps=8, num_stages=1)
    return dq, dk, dv, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class FusedLandmarkAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, n_prefix_q, sm_scale, block_size):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        batch, nheads, seqlen_q, d = q.shape
        _, _, seqlen_k, _ = k.shape
        assert k.shape == (batch, nheads, seqlen_k, d)
        assert v.shape == (batch, nheads, seqlen_k, d)
        assert d <= 128, 'FlashAttention only support head dimensions up to 128'
        assert q.dtype == k.dtype == v.dtype, 'All tensors must have the same type'
        assert q.is_cuda and k.is_cuda and v.is_cuda
        BLOCK = block_size
        o = torch.empty_like(q)
        grid = triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.
            device, dtype=torch.float32)
        m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.
            device, dtype=torch.float32)
        num_warps = 4 if d <= 64 else 8
        _fwd_kernel[grid](q, k, v, sm_scale, o, q.stride(0), q.stride(1), q
            .stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2),
            k.stride(3), v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3), L, m, q.
            shape[0], q.shape[1], q.shape[2], k.shape[2], BLOCK=BLOCK,
            BLOCK_DMODEL=d, N_PREFIX_Q=n_prefix_q, num_warps=num_warps,
            num_stages=2)
        ctx.save_for_backward(q, k, v, o, L, m)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = d
        ctx.N_PREFIX_Q = n_prefix_q
        ctx.BLOCK = BLOCK
        return o

    @staticmethod
    def backward(ctx, do):
        BLOCK = ctx.BLOCK
        q, k, v, o, l, m = ctx.saved_tensors
        assert q.shape[2
            ] % BLOCK == 0, 'Backward supported only for full blocks'
        assert k.shape[2
            ] % BLOCK == 0, 'Backward supported only for full blocks'
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        do_scaled = torch.empty_like(do)
        delta = torch.empty_like(l)
        _bwd_preprocess[ctx.grid[0], ctx.grid[1]](o, o.stride(0), o.stride(
            1), o.stride(2), o.stride(3), do, l, l.stride(0), l.stride(1),
            do_scaled, delta, q.shape[2], BLOCK_M=BLOCK, D_HEAD=ctx.
            BLOCK_DMODEL)
        _bwd_kernel[ctx.grid[1],](q, k, v, ctx.sm_scale, o, do_scaled, dq,
            dk, dv, l, m, delta, q.stride(0), q.stride(1), q.stride(2), q.
            stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3), q.shape[0],
            q.shape[1], q.shape[2], k.shape[2], BLOCK=BLOCK, BLOCK_DMODEL=
            ctx.BLOCK_DMODEL, N_PREFIX_Q=ctx.N_PREFIX_Q, num_warps=8,
            num_stages=1)
        return dq, dk, dv, None, None, None
