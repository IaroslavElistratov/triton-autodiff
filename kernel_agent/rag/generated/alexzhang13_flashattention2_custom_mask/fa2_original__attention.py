# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/alexzhang13/flashattention2-custom-mask
# Source-Files: fa2_original.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_er8kleg9/flashattention2-custom-mask-main/fa2_original.py
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

@triton.autotune(list(filter(keep, configs)), key=['N_CTX', 'HEAD_DIM'])
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out, stride_qz, stride_qh, stride_qm,
    stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
    stride_vh, stride_vk, stride_vn, stride_oz, stride_oh, stride_om,
    stride_on, Z, H, N_CTX, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, STAGE: tl.constexpr):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64
        ) * stride_qh
    Q_block_ptr = tl.make_block_ptr(base=Q + qvk_offset, shape=(N_CTX,
        HEAD_DIM), strides=(stride_qm, stride_qk), offsets=(start_m *
        BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (
        1, 0)
    V_block_ptr = tl.make_block_ptr(base=V + qvk_offset, shape=(N_CTX,
        HEAD_DIM), strides=(stride_vk, stride_vn), offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM), order=v_order)
    K_block_ptr = tl.make_block_ptr(base=K + qvk_offset, shape=(HEAD_DIM,
        N_CTX), strides=(stride_kk, stride_kn), offsets=(0, 0), block_shape
        =(HEAD_DIM, BLOCK_N), order=(0, 1))
    O_block_ptr = tl.make_block_ptr(base=Out + qvk_offset, shape=(N_CTX,
        HEAD_DIM), strides=(stride_om, stride_on), offsets=(start_m *
        BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale
    qk_scale *= 1.44269504
    q = tl.load(Q_block_ptr)
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr,
            V_block_ptr, start_m, qk_scale, BLOCK_M, HEAD_DIM, BLOCK_N, 4 -
            STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5)
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr,
            V_block_ptr, start_m, qk_scale, BLOCK_M, HEAD_DIM, BLOCK_N, 2,
            offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5)
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m,
    qk_scale, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.
    constexpr, STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.
    constexpr, N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= start_n + offs_n[None, :]
            qk = qk * qk_scale + tl.where(mask, 0, -1000000.0)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(V_block_ptr)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == 'hip'


# Forward method (kernel launch code)
def __attention_forward(ctx, q, k, v, causal, sm_scale):
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    o = torch.empty_like(q)
    stage = 3 if causal else 1
    extra_kern_args = {}
    if is_hip():
        waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
        extra_kern_args = {'waves_per_eu': waves_per_eu,
            'allow_flush_denorm': True}
    grid = lambda args: (triton.cdiv(q.shape[2], args['BLOCK_M']), q.shape[
        0] * q.shape[1], 1)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device,
        dtype=torch.float32)
    _attn_fwd[grid](q, k, v, sm_scale, M, o, q.stride(0), q.stride(1), q.
        stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2), k.
        stride(3), v.stride(0), v.stride(1), v.stride(2), v.stride(3), o.
        stride(0), o.stride(1), o.stride(2), o.stride(3), q.shape[0], q.
        shape[1], N_CTX=q.shape[2], HEAD_DIM=HEAD_DIM_K, STAGE=stage, **
        extra_kern_args)
    ctx.save_for_backward(q, k, v, o, M)
    ctx.grid = grid
    ctx.sm_scale = sm_scale
    ctx.HEAD_DIM = HEAD_DIM_K
    ctx.causal = causal
    return o


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _attn_bwd(Q, K, V, sm_scale, DO, DQ, DK, DV, M, D, stride_z, stride_h,
    stride_tok, stride_d, H, N_CTX, BLOCK_M1: tl.constexpr, BLOCK_N1: tl.
    constexpr, BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr, HEAD_DIM: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996
    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz
    offs_k = tl.arange(0, HEAD_DIM)
    start_n = pid * BLOCK_N1
    start_m = start_n
    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    num_steps = BLOCK_N1 // MASK_BLOCK_M1
    dk, dv = _attn_bwd_dkdv(dk, dv, Q, k, v, sm_scale, DO, M, D, stride_tok,
        stride_d, H, N_CTX, MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM, start_n,
        start_m, num_steps, MASK=True)
    start_m += num_steps * MASK_BLOCK_M1
    num_steps = (N_CTX - start_m) // BLOCK_M1
    dk, dv = _attn_bwd_dkdv(dk, dv, Q, k, v, sm_scale, DO, M, D, stride_tok,
        stride_d, H, N_CTX, BLOCK_M1, BLOCK_N1, HEAD_DIM, start_n, start_m,
        num_steps, MASK=False)
    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2
    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
        )
    m = tl.load(M + offs_m)
    m = m[:, None]
    num_steps = BLOCK_M2 // MASK_BLOCK_N2
    dq = _attn_bwd_dq(dq, q, K, V, do, m, D, stride_tok, stride_d, H, N_CTX,
        BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM, start_m, end_n - num_steps *
        MASK_BLOCK_N2, num_steps, MASK=True)
    end_n -= num_steps * MASK_BLOCK_N2
    num_steps = end_n // BLOCK_N2
    dq = _attn_bwd_dq(dq, q, K, V, do, m, D, stride_tok, stride_d, H, N_CTX,
        BLOCK_M2, BLOCK_N2, HEAD_DIM, start_m, end_n - num_steps * BLOCK_N2,
        num_steps, MASK=False)
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


@triton.jit
def _attn_bwd_dkdv(dk, dv, Q, k, v, sm_scale, DO, M, D, stride_tok,
    stride_d, H, N_CTX, BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr,
    HEAD_DIM: tl.constexpr, start_n, start_m, num_steps, MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        if MASK:
            mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        ppT = pT
        ppT = ppT.to(tl.float16)
        dv += tl.dot(ppT, do)
        Di = tl.load(D + offs_m)
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


@triton.jit
def _attn_bwd_dq(dq, q, K, V, do, m, D, stride_tok, stride_d, H, N_CTX,
    BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr, HEAD_DIM: tl.constexpr,
    start_m, start_n, num_steps, MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    Di = tl.load(D + offs_m)
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = offs_m[:, None] >= offs_n[None, :]
            p = tl.where(mask, p, 0.0)
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        dq += tl.dot(ds, tl.trans(kT))
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq


@triton.jit
def _attn_bwd_preprocess(O, DO, Delta, Z, H, N_CTX, BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM +
        off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM +
        off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# Backward method (kernel launch code)
def __attention_backward(ctx, do):
    q, k, v, o, M = ctx.saved_tensors
    assert do.is_contiguous()
    assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    BATCH, N_HEAD, N_CTX = q.shape[:3]
    PRE_BLOCK = 128
    NUM_WARPS, NUM_STAGES = 4, 5
    BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
    BLK_SLICE_FACTOR = 2
    RCP_LN2 = 1.4426950408889634
    arg_k = k
    arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
    PRE_BLOCK = 128
    assert N_CTX % PRE_BLOCK == 0
    pre_grid = N_CTX // PRE_BLOCK, BATCH * N_HEAD
    delta = torch.empty_like(M)
    _attn_bwd_preprocess[pre_grid](o, do, delta, BATCH, N_HEAD, N_CTX,
        BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM)
    grid = N_CTX // BLOCK_N1, 1, BATCH * N_HEAD
    _attn_bwd[grid](q, arg_k, v, ctx.sm_scale, do, dq, dk, dv, M, delta, q.
        stride(0), q.stride(1), q.stride(2), q.stride(3), N_HEAD, N_CTX,
        BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1, BLOCK_M2=BLOCK_M2, BLOCK_N2=
        BLOCK_N2, BLK_SLICE_FACTOR=BLK_SLICE_FACTOR, HEAD_DIM=ctx.HEAD_DIM,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    return dq, dk, dv, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {'waves_per_eu': waves_per_eu,
                'allow_flush_denorm': True}
        grid = lambda args: (triton.cdiv(q.shape[2], args['BLOCK_M']), q.
            shape[0] * q.shape[1], 1)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.
            device, dtype=torch.float32)
        _attn_fwd[grid](q, k, v, sm_scale, M, o, q.stride(0), q.stride(1),
            q.stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2),
            k.stride(3), v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3), q.shape[0],
            q.shape[1], N_CTX=q.shape[2], HEAD_DIM=HEAD_DIM_K, STAGE=stage,
            **extra_kern_args)
        ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride(
            ) == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 5
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = N_CTX // PRE_BLOCK, BATCH * N_HEAD
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](o, do, delta, BATCH, N_HEAD, N_CTX,
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM)
        grid = N_CTX // BLOCK_N1, 1, BATCH * N_HEAD
        _attn_bwd[grid](q, arg_k, v, ctx.sm_scale, do, dq, dk, dv, M, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3), N_HEAD,
            N_CTX, BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1, BLOCK_M2=BLOCK_M2,
            BLOCK_N2=BLOCK_N2, BLK_SLICE_FACTOR=BLK_SLICE_FACTOR, HEAD_DIM=
            ctx.HEAD_DIM, num_warps=NUM_WARPS, num_stages=NUM_STAGES)
        return dq, dk, dv, None, None
