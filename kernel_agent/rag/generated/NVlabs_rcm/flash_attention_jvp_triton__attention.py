# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/NVlabs/rcm
# Source-Files: rcm/utils/flash_attention_jvp_triton.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_ykw7krs6/rcm-main/rcm/utils/flash_attention_jvp_triton.py
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

@triton.autotune(configs, key=['SEQ_LEN_Q', 'SEQ_LEN_KV', 'HEAD_DIM_QK',
    'HEAD_DIM_V'])
@triton.jit
def _attn_fwd(Q, K, V, tQ, tK, tV, sm_scale, M, Out, tOut, stride_qz,
    stride_qh, stride_qm, stride_qd, stride_kz, stride_kh, stride_kn,
    stride_kd, stride_vz, stride_vh, stride_vn, stride_vd, stride_oz,
    stride_oh, stride_om, stride_od, Z, H, SEQ_LEN_Q, SEQ_LEN_KV,
    HEAD_DIM_QK: tl.constexpr, HEAD_DIM_V: tl.constexpr, BLOCK_M: tl.
    constexpr, BLOCK_N: tl.constexpr, STAGE: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    start_m_idx = start_m * BLOCK_M
    Q_block_ptr = tl.make_block_ptr(base=Q + q_offset, shape=(SEQ_LEN_Q,
        HEAD_DIM_QK), strides=(stride_qm, stride_qd), offsets=(start_m_idx,
        0), block_shape=(BLOCK_M, HEAD_DIM_QK), order=(1, 0))
    V_block_ptr = tl.make_block_ptr(base=V + v_offset, shape=(SEQ_LEN_KV,
        HEAD_DIM_V), strides=(stride_vn, stride_vd), offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM_V), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + k_offset, shape=(HEAD_DIM_QK,
        SEQ_LEN_KV), strides=(stride_kd, stride_kn), offsets=(0, 0),
        block_shape=(HEAD_DIM_QK, BLOCK_N), order=(0, 1))
    O_block_ptr = tl.make_block_ptr(base=Out + o_offset, shape=(SEQ_LEN_Q,
        HEAD_DIM_V), strides=(stride_om, stride_od), offsets=(start_m_idx, 
        0), block_shape=(BLOCK_M, HEAD_DIM_V), order=(1, 0))
    tQ_block_ptr = tl.make_block_ptr(base=tQ + q_offset, shape=(SEQ_LEN_Q,
        HEAD_DIM_QK), strides=(stride_qm, stride_qd), offsets=(start_m_idx,
        0), block_shape=(BLOCK_M, HEAD_DIM_QK), order=(1, 0))
    tV_block_ptr = tl.make_block_ptr(base=tV + v_offset, shape=(SEQ_LEN_KV,
        HEAD_DIM_V), strides=(stride_vn, stride_vd), offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM_V), order=(1, 0))
    tK_block_ptr = tl.make_block_ptr(base=tK + k_offset, shape=(HEAD_DIM_QK,
        SEQ_LEN_KV), strides=(stride_kd, stride_kn), offsets=(0, 0),
        block_shape=(HEAD_DIM_QK, BLOCK_N), order=(0, 1))
    tO_block_ptr = tl.make_block_ptr(base=tOut + o_offset, shape=(SEQ_LEN_Q,
        HEAD_DIM_V), strides=(stride_om, stride_od), offsets=(start_m_idx, 
        0), block_shape=(BLOCK_M, HEAD_DIM_V), order=(1, 0))
    offs_m = start_m_idx + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d_qk, offs_d_v = tl.arange(0, HEAD_DIM_QK), tl.arange(0, HEAD_DIM_V)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM_V], dtype=tl.float32)
    r_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_A = tl.zeros([BLOCK_M, HEAD_DIM_V], dtype=tl.float32)
    acc_B = tl.zeros([BLOCK_M, HEAD_DIM_V], dtype=tl.float32)
    q, tq = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero'
        ), tl.load(tQ_block_ptr, boundary_check=(0, 1), padding_option='zero')
    if STAGE & 1:
        acc, acc_A, acc_B, l_i, m_i, r_i = _attn_fwd_inner(acc, acc_A,
            acc_B, l_i, m_i, r_i, q, tq, K_block_ptr, V_block_ptr,
            tK_block_ptr, tV_block_ptr, start_m, sm_scale, BLOCK_M, BLOCK_N,
            4 - STAGE, offs_m, offs_n, SEQ_LEN_KV, HEAD_DIM_V, V.dtype.
            element_ty == tl.bfloat16)
    if STAGE & 2:
        acc, acc_A, acc_B, l_i, m_i, r_i = _attn_fwd_inner(acc, acc_A,
            acc_B, l_i, m_i, r_i, q, tq, K_block_ptr, V_block_ptr,
            tK_block_ptr, tV_block_ptr, start_m, sm_scale, BLOCK_M, BLOCK_N,
            2, offs_m, offs_n, SEQ_LEN_KV, HEAD_DIM_V, V.dtype.element_ty ==
            tl.bfloat16)
    empty_mask = l_i == 0.0
    l_i = tl.where(empty_mask, 1.0, l_i)
    m_i = m_i + tl.where(empty_mask, 0.0, tl.math.log2(l_i))
    acc = acc / l_i[:, None]
    tO_i = (acc_A + acc_B - r_i[:, None] * acc) / l_i[:, None]
    m_ptrs = M + off_hz * SEQ_LEN_Q + offs_m
    O_block_ptr = Out + o_offset + offs_m[:, None] * stride_om + offs_d_v[
        None, :] * stride_od
    tO_block_ptr = tOut + o_offset + offs_m[:, None] * stride_om + offs_d_v[
        None, :] * stride_od
    mask_lse = offs_m < SEQ_LEN_Q
    mask = offs_m[:, None] < SEQ_LEN_Q
    tl.store(m_ptrs, m_i * 0.69314718, mask=mask_lse)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=mask)
    tl.store(tO_block_ptr, tO_i.to(tOut.type.element_ty), mask=mask)


@triton.jit
def _attn_fwd_inner(acc, acc_A, acc_B, l_i, m_i, r_i, q, tq, K_block_ptr,
    V_block_ptr, tK_block_ptr, tV_block_ptr, start_m, sm_scale, BLOCK_M: tl
    .constexpr, BLOCK_N: tl.constexpr, STAGE: tl.constexpr, offs_m: tl.
    constexpr, offs_n: tl.constexpr, SEQ_LEN_KV: tl.constexpr, HEAD_DIM_V:
    tl.constexpr, bf16_v: tl.constexpr):
    if STAGE == 1:
        lo, hi = 0, min(start_m * BLOCK_M, SEQ_LEN_KV)
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, min((start_m + 1) * BLOCK_M, SEQ_LEN_KV)
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, SEQ_LEN_KV
    qk_scale = sm_scale * 1.44269504
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    tK_block_ptr = tl.advance(tK_block_ptr, (0, lo))
    tV_block_ptr = tl.advance(tV_block_ptr, (lo, 0))
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k, tk = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option=
            'zero'), tl.load(tK_block_ptr, boundary_check=(0, 1),
            padding_option='zero')
        qk = tl.dot(q, k)
        tS_ij = tl.dot(tq, k)
        tS_ij = tl.dot(q, tk, tS_ij)
        tS_ij *= sm_scale
        if STAGE == 2:
            causal_mask = offs_m[:, None] >= start_n + offs_n[None, :]
            qk = qk * qk_scale + tl.where(causal_mask, 0, -1000000.0)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        boundary_m = tl.full([BLOCK_M], hi, dtype=tl.int32)
        size_n = start_n + offs_n[None, :]
        mask = size_n < boundary_m[:, None]
        qk = tl.where(mask, qk, float('-inf'))
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        tS_ij = tl.where(mask, tS_ij, float('0'))
        H_ij = p * tS_ij
        r_ij = tl.sum(H_ij, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        r_i = r_i * alpha + r_ij
        acc = acc * alpha[:, None]
        acc_A = acc_A * alpha[:, None]
        acc_B = acc_B * alpha[:, None]
        v, tv = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option=
            'zero'), tl.load(tV_block_ptr, boundary_check=(0, 1),
            padding_option='zero')
        if bf16_v:
            p = p.to(tl.bfloat16)
            H_ij = H_ij.to(tl.bfloat16)
            v = v.to(tl.bfloat16)
            tv = tv.to(tl.bfloat16)
        else:
            p = p.to(tl.float16)
            H_ij = H_ij.to(tl.float16)
            v = v.to(tl.float16)
            tv = tv.to(tl.float16)
        acc = tl.dot(p, v, acc)
        acc_A = tl.dot(p, tv, acc_A)
        acc_B = tl.dot(H_ij, v, acc_B)
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        tV_block_ptr = tl.advance(tV_block_ptr, (BLOCK_N, 0))
        tK_block_ptr = tl.advance(tK_block_ptr, (0, BLOCK_N))
    return acc, acc_A, acc_B, l_i, m_i, r_i


# Forward method (kernel launch code)
def __attention_forward(ctx, q, k, v, tq, tk, tv, causal=False, sm_scale=None):
    is_grad = any(x.requires_grad for x in [q, k, v])
    assert q.shape[:-2] == k.shape[:-2] and k.shape[:-2] == v.shape[:-2]
    assert k.shape[-2] == v.shape[-2] and q.shape[-1] == k.shape[-1]
    Z, H = q.shape[:-2]
    SEQ_LEN_Q, SEQ_LEN_KV = q.shape[-2], k.shape[-2]
    HEAD_DIM_QK, HEAD_DIM_V = q.shape[-1], v.shape[-1]
    assert HEAD_DIM_QK in {16, 32, 64, 128, 256}
    assert HEAD_DIM_V in {16, 32, 64, 128, 256}
    assert SEQ_LEN_Q == SEQ_LEN_KV or not causal, 'Causal cross-attention is currently not supported.'
    assert tq.shape == q.shape and tk.shape == k.shape and tv.shape == v.shape
    assert tq.stride() == q.stride() and tk.stride() == k.stride(
        ) and tv.stride() == v.stride()
    if sm_scale is None:
        sm_scale = HEAD_DIM_QK ** -0.5
    o = torch.empty((Z, H, SEQ_LEN_Q, HEAD_DIM_V), device=q.device, dtype=q
        .dtype)
    to = torch.empty_like(o)
    stage = 3 if causal else 1
    M = torch.empty((Z, H, SEQ_LEN_Q), device=q.device, dtype=torch.float32)

    def grid(args):
        return triton.cdiv(SEQ_LEN_Q, args['BLOCK_M']), Z * H, 1
    ctx.grid = grid
    _attn_fwd[grid](q, k, v, tq, tk, tv, sm_scale, M, o, to, q.stride(0), q
        .stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride(1), k.
        stride(2), k.stride(3), v.stride(0), v.stride(1), v.stride(2), v.
        stride(3), o.stride(0), o.stride(1), o.stride(2), o.stride(3), Z, H,
        SEQ_LEN_Q, SEQ_LEN_KV, HEAD_DIM_QK, HEAD_DIM_V, STAGE=stage)
    if is_grad:
        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
    return o, to


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

def generate_qkv(q, k, v):
    """
    Arguments:
        q: (batch_size, nheads, seqlen_q, d)
        k: (batch_size, nheads_k, seqlen_k, d)
        v: (batch_size, nheads_k, seqlen_k, d)
    """
    batch_size, _, seqlen_q, d = q.shape
    _, nheads_k, seqlen_k, _ = k.shape
    assert k.shape == (batch_size, nheads_k, seqlen_k, d)
    assert v.shape == (batch_size, nheads_k, seqlen_k, d)

    def unpad_fn(x):
        return rearrange(x, 'b h s d -> (b s) h d')

    def lse_unpad_fn(x):
        return rearrange(x, 'b h s -> (b s) h')

    def pad_fn(x):
        return rearrange(x, '(b s) h d -> b h s d', b=batch_size)
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=
        seqlen_q, dtype=torch.int32, device=q.device)
    max_seqlen_q = seqlen_q
    cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=
        seqlen_k, dtype=torch.int32, device=q.device)
    max_seqlen_k = seqlen_k
    return (unpad_fn, lse_unpad_fn, pad_fn, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k)


# Backward method (kernel launch code)
def __attention_backward(ctx, dout, *args):
    q, k, v, out, softmax_lse = ctx.saved_tensors
    assert q.shape[-1] == k.shape[-1] and k.shape[-1] == v.shape[-1
        ], 'Backward not supported with different headdim.'
    if q.shape[-2] == k.shape[-2]:
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k
            ), torch.empty_like(v)
        _flash_attn_backward(dout.transpose(1, 2), q.transpose(1, 2), k.
            transpose(1, 2), v.transpose(1, 2), out.transpose(1, 2),
            softmax_lse, dq.transpose(1, 2), dk.transpose(1, 2), dv.
            transpose(1, 2), dropout_p=0.0, softmax_scale=ctx.sm_scale,
            causal=ctx.causal, window_size=(-1, -1), alibi_slopes=None,
            deterministic=False)
    else:
        (unpad_fn, lse_unpad_fn, pad_fn, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k) = generate_qkv(q, k, v)
        q_unpad, k_unpad, v_unpad = unpad_fn(q), unpad_fn(k), unpad_fn(v)
        dq, dk, dv = torch.empty_like(q_unpad), torch.empty_like(k_unpad
            ), torch.empty_like(v_unpad)
        _flash_attn_varlen_backward(unpad_fn(dout), q_unpad, k_unpad,
            v_unpad, unpad_fn(out), lse_unpad_fn(softmax_lse), dq, dk, dv,
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            dropout_p=0.0, softmax_scale=ctx.sm_scale, causal=ctx.causal,
            window_size=(-1, -1), alibi_slopes=None, deterministic=False)
        dq, dk, dv = pad_fn(dq), pad_fn(dk), pad_fn(dv)
    return dq, dk, dv, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _attention(torch.autograd.Function):
    """
    Arguments:
        q, tq: (batch_size, nheads, seqlen_q, d_qk)
        k, tk: (batch_size, nheads, seqlen_kv, d_qk)
        v, tv: (batch_size, nheads, seqlen_kv, d_v)
    Returns:
        o, to: (batch_size, nheads, seqlen_q, d_v)

    Backward is only supported when d_qk=d_v.
    """

    @staticmethod
    def forward(ctx, q, k, v, tq, tk, tv, causal=False, sm_scale=None):
        is_grad = any(x.requires_grad for x in [q, k, v])
        assert q.shape[:-2] == k.shape[:-2] and k.shape[:-2] == v.shape[:-2]
        assert k.shape[-2] == v.shape[-2] and q.shape[-1] == k.shape[-1]
        Z, H = q.shape[:-2]
        SEQ_LEN_Q, SEQ_LEN_KV = q.shape[-2], k.shape[-2]
        HEAD_DIM_QK, HEAD_DIM_V = q.shape[-1], v.shape[-1]
        assert HEAD_DIM_QK in {16, 32, 64, 128, 256}
        assert HEAD_DIM_V in {16, 32, 64, 128, 256}
        assert SEQ_LEN_Q == SEQ_LEN_KV or not causal, 'Causal cross-attention is currently not supported.'
        assert tq.shape == q.shape and tk.shape == k.shape and tv.shape == v.shape
        assert tq.stride() == q.stride() and tk.stride() == k.stride(
            ) and tv.stride() == v.stride()
        if sm_scale is None:
            sm_scale = HEAD_DIM_QK ** -0.5
        o = torch.empty((Z, H, SEQ_LEN_Q, HEAD_DIM_V), device=q.device,
            dtype=q.dtype)
        to = torch.empty_like(o)
        stage = 3 if causal else 1
        M = torch.empty((Z, H, SEQ_LEN_Q), device=q.device, dtype=torch.float32
            )

        def grid(args):
            return triton.cdiv(SEQ_LEN_Q, args['BLOCK_M']), Z * H, 1
        ctx.grid = grid
        _attn_fwd[grid](q, k, v, tq, tk, tv, sm_scale, M, o, to, q.stride(0
            ), q.stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride
            (1), k.stride(2), k.stride(3), v.stride(0), v.stride(1), v.
            stride(2), v.stride(3), o.stride(0), o.stride(1), o.stride(2),
            o.stride(3), Z, H, SEQ_LEN_Q, SEQ_LEN_KV, HEAD_DIM_QK,
            HEAD_DIM_V, STAGE=stage)
        if is_grad:
            ctx.save_for_backward(q, k, v, o, M)
            ctx.sm_scale = sm_scale
            ctx.causal = causal
        return o, to

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        assert q.shape[-1] == k.shape[-1] and k.shape[-1] == v.shape[-1
            ], 'Backward not supported with different headdim.'
        if q.shape[-2] == k.shape[-2]:
            dq, dk, dv = torch.empty_like(q), torch.empty_like(k
                ), torch.empty_like(v)
            _flash_attn_backward(dout.transpose(1, 2), q.transpose(1, 2), k
                .transpose(1, 2), v.transpose(1, 2), out.transpose(1, 2),
                softmax_lse, dq.transpose(1, 2), dk.transpose(1, 2), dv.
                transpose(1, 2), dropout_p=0.0, softmax_scale=ctx.sm_scale,
                causal=ctx.causal, window_size=(-1, -1), alibi_slopes=None,
                deterministic=False)
        else:
            (unpad_fn, lse_unpad_fn, pad_fn, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k) = generate_qkv(q, k, v)
            q_unpad, k_unpad, v_unpad = unpad_fn(q), unpad_fn(k), unpad_fn(v)
            dq, dk, dv = torch.empty_like(q_unpad), torch.empty_like(k_unpad
                ), torch.empty_like(v_unpad)
            _flash_attn_varlen_backward(unpad_fn(dout), q_unpad, k_unpad,
                v_unpad, unpad_fn(out), lse_unpad_fn(softmax_lse), dq, dk,
                dv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                dropout_p=0.0, softmax_scale=ctx.sm_scale, causal=ctx.
                causal, window_size=(-1, -1), alibi_slopes=None,
                deterministic=False)
            dq, dk, dv = pad_fn(dq), pad_fn(dk), pad_fn(dv)
        return dq, dk, dv, None, None, None, None, None
