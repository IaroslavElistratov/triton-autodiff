# SPDX-License-Identifier: BSD-3-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/Dao-AILab/flash-attention
# Source-Files: flash_attn/flash_attn_triton_og.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_aqv8vcou/flash-attention-main/flash_attn/flash_attn_triton_og.py
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
def _fwd_kernel(Q, K, V, sm_scale, TMP, L, M, Out, stride_qz, stride_qh,
    stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn, stride_oz, stride_oh,
    stride_om, stride_on, Z, H, N_CTX, BLOCK_M: tl.constexpr, BLOCK_DMODEL:
    tl.constexpr, BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :
        ] * stride_qk
    off_k = off_hz * stride_qh + offs_n[:, None] * stride_kn + offs_d[None, :
        ] * stride_kk
    off_v = off_hz * stride_qh + offs_n[:, None] * stride_qm + offs_d[None, :
        ] * stride_qk
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    t_ptrs = TMP + off_hz * N_CTX + offs_m
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    q = tl.load(q_ptrs)
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + start_n * stride_kn)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k, trans_b=True)
        qk *= sm_scale
        qk += tl.where(offs_m[:, None] >= start_n + offs_n[None, :], 0,
            float('-inf'))
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        tl.store(t_ptrs, acc_scale)
        acc_scale = tl.load(t_ptrs)
        acc = acc * acc_scale[:, None]
        v = tl.load(v_ptrs + start_n * stride_vk)
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        l_i = l_i_new
        m_i = m_i_new
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    l_ptrs = L + off_hz * N_CTX + offs_m
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, l_i)
    tl.store(m_ptrs, m_i)
    offs_n = tl.arange(0, BLOCK_DMODEL)
    off_o = off_hz * stride_oh + offs_m[:, None] * stride_om + offs_n[None, :
        ] * stride_on
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)


# Forward method (kernel launch code)
def __attention_forward(ctx, q, k, v, sm_scale):
    BLOCK = 128
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.empty_like(q)
    grid = triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1]
    tmp = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.
        device, dtype=torch.float32)
    L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device,
        dtype=torch.float32)
    m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device,
        dtype=torch.float32)
    num_warps = 4 if Lk <= 64 else 8
    _fwd_kernel[grid](q, k, v, sm_scale, tmp, L, m, o, q.stride(0), q.
        stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride(1), k.
        stride(2), k.stride(3), v.stride(0), v.stride(1), v.stride(2), v.
        stride(3), o.stride(0), o.stride(1), o.stride(2), o.stride(3), q.
        shape[0], q.shape[1], q.shape[2], BLOCK_M=BLOCK, BLOCK_N=BLOCK,
        BLOCK_DMODEL=Lk, num_warps=num_warps, num_stages=1)
    ctx.save_for_backward(q, k, v, o, L, m)
    ctx.BLOCK = BLOCK
    ctx.grid = grid
    ctx.sm_scale = sm_scale
    ctx.BLOCK_DMODEL = Lk
    return o


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _bwd_kernel(Q, K, V, sm_scale, Out, DO, DQ, DK, DV, L, M, D, stride_qz,
    stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn,
    stride_kk, stride_vz, stride_vh, stride_vk, stride_vn, Z, H, N_CTX,
    num_block, BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N:
    tl.constexpr):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_qz + off_h * stride_qh
    V += off_z * stride_qz + off_h * stride_qh
    DO += off_z * stride_qz + off_h * stride_qh
    DQ += off_z * stride_qz + off_h * stride_qh
    DK += off_z * stride_qz + off_h * stride_qh
    DV += off_z * stride_qz + off_h * stride_qh
    for start_n in range(0, num_block):
        lo = start_n * BLOCK_M
        offs_qm = lo + tl.arange(0, BLOCK_M)
        offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)
        q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_k[None, :] *
            stride_qk)
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
            )
        v_ptrs = V + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk
            )
        do_ptrs = DO + (offs_qm[:, None] * stride_qm + offs_k[None, :] *
            stride_qk)
        dq_ptrs = DQ + (offs_qm[:, None] * stride_qm + offs_k[None, :] *
            stride_qk)
        D_ptrs = D + off_hz * N_CTX
        m_ptrs = M + off_hz * N_CTX
        dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        for start_m in range(lo, num_block * BLOCK_M, BLOCK_M):
            offs_m_curr = start_m + offs_m
            q = tl.load(q_ptrs)
            qk = tl.dot(q, k, trans_b=True)
            qk = tl.where(offs_m_curr[:, None] >= offs_n[None, :], qk,
                float('-inf'))
            m = tl.load(m_ptrs + offs_m_curr)
            p = tl.exp(qk * sm_scale - m[:, None])
            do = tl.load(do_ptrs)
            dv += tl.dot(p.to(do.dtype), do, trans_a=True)
            Di = tl.load(D_ptrs + offs_m_curr)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do, v, trans_b=True)
            ds = p * dp * sm_scale
            dk += tl.dot(ds.to(q.dtype), q, trans_a=True)
            dq = tl.load(dq_ptrs, eviction_policy='evict_last')
            dq += tl.dot(ds.to(k.dtype), k)
            tl.store(dq_ptrs, dq, eviction_policy='evict_last')
            dq_ptrs += BLOCK_M * stride_qm
            q_ptrs += BLOCK_M * stride_qm
            do_ptrs += BLOCK_M * stride_qm
        dv_ptrs = DV + (offs_n[:, None] * stride_qm + offs_k[None, :] *
            stride_qk)
        dk_ptrs = DK + (offs_n[:, None] * stride_kn + offs_k[None, :] *
            stride_kk)
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)


@triton.jit
def _bwd_preprocess(Out, DO, L, NewDO, Delta, BLOCK_M: tl.constexpr, D_HEAD:
    tl.constexpr):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    denom = tl.load(L + off_m).to(tl.float32)
    do = do / denom[:, None]
    delta = tl.sum(o * do, axis=1)
    tl.store(NewDO + off_m[:, None] * D_HEAD + off_n[None, :], do)
    tl.store(Delta + off_m, delta)


# Backward method (kernel launch code)
def __attention_backward(ctx, do):
    q, k, v, o, l, m = ctx.saved_tensors
    do = do.contiguous()
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    do_scaled = torch.empty_like(do)
    delta = torch.empty_like(l)
    _bwd_preprocess[ctx.grid[0] * ctx.grid[1],](o, do, l, do_scaled, delta,
        BLOCK_M=ctx.BLOCK, D_HEAD=ctx.BLOCK_DMODEL)
    num_warps = 8
    _bwd_kernel[ctx.grid[1],](q, k, v, ctx.sm_scale, o, do_scaled, dq, dk,
        dv, l, m, delta, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3), v.stride(0), v.
        stride(1), v.stride(2), v.stride(3), q.shape[0], q.shape[1], q.
        shape[2], ctx.grid[0], BLOCK_M=ctx.BLOCK, BLOCK_N=ctx.BLOCK,
        BLOCK_DMODEL=ctx.BLOCK_DMODEL, num_warps=num_warps, num_stages=1)
    return dq.to(q.dtype), dk, dv, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        BLOCK = 128
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        grid = triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1]
        tmp = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.
            device, dtype=torch.float32)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.
            device, dtype=torch.float32)
        m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.
            device, dtype=torch.float32)
        num_warps = 4 if Lk <= 64 else 8
        _fwd_kernel[grid](q, k, v, sm_scale, tmp, L, m, o, q.stride(0), q.
            stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride(1),
            k.stride(2), k.stride(3), v.stride(0), v.stride(1), v.stride(2),
            v.stride(3), o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2], BLOCK_M=BLOCK, BLOCK_N=
            BLOCK, BLOCK_DMODEL=Lk, num_warps=num_warps, num_stages=1)
        ctx.save_for_backward(q, k, v, o, L, m)
        ctx.BLOCK = BLOCK
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, l, m = ctx.saved_tensors
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        do_scaled = torch.empty_like(do)
        delta = torch.empty_like(l)
        _bwd_preprocess[ctx.grid[0] * ctx.grid[1],](o, do, l, do_scaled,
            delta, BLOCK_M=ctx.BLOCK, D_HEAD=ctx.BLOCK_DMODEL)
        num_warps = 8
        _bwd_kernel[ctx.grid[1],](q, k, v, ctx.sm_scale, o, do_scaled, dq,
            dk, dv, l, m, delta, q.stride(0), q.stride(1), q.stride(2), q.
            stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3), q.shape[0],
            q.shape[1], q.shape[2], ctx.grid[0], BLOCK_M=ctx.BLOCK, BLOCK_N
            =ctx.BLOCK, BLOCK_DMODEL=ctx.BLOCK_DMODEL, num_warps=num_warps,
            num_stages=1)
        return dq.to(q.dtype), dk, dv, None
