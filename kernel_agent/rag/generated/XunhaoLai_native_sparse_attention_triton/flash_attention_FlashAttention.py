# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/XunhaoLai/native-sparse-attention-triton
# Source-Files: native_sparse_attention/ops/triton/flash_attention.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_ottww3ye/native-sparse-attention-triton-main/native_sparse_attention/ops/triton/flash_attention.py
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
from math import sqrt

def get_num_warps_stages(head_dim, block_size, is_hopper_gpu):
    """
    Returns recommended num_warps and num_stages for a Sparse Attention kernel in Triton.

    Args:
        head_dim (int): Size of the head dimension.
        block_size (int): Size of the block in the attention matrix.
        is_hopper_gpu (bool): True if Hopper GPU, False if Ampere GPU.

    Returns:
        tuple: (num_warps, num_stages) recommended values.
    """
    head_large = head_dim > 64
    block_large = block_size > 64
    if is_hopper_gpu:
        if head_large and block_large:
            num_warps = 8
            num_stages = 3
        elif head_large or block_large:
            num_warps = 4
            num_stages = 3
        else:
            num_warps = 2
            num_stages = 2
    elif head_large and block_large:
        num_warps = 8
        num_stages = 3
    elif head_large or block_large:
        num_warps = 8
        num_stages = 3
    else:
        num_warps = 2
        num_stages = 2
    if head_dim > 128:
        num_stages = 2
    return num_warps, num_stages


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def forward_kernel(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, cu_seqlens_q,
    cu_seqlens_k, NUM_KV_HEADS, NUM_SHARE_Q_HEADS, qk_head_dim, v_head_dim,
    sm_scale, causal, gqa_interleave, stride_qn, stride_qh, stride_qd,
    stride_kn, stride_kh, stride_kd, stride_vn, stride_vh, stride_vd,
    stride_on, stride_oh, stride_od, stride_lh, stride_ln, BLOCK_SIZE_Q: tl
    .constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_KD: tl.constexpr,
    BLOCK_SIZE_VD: tl.constexpr):
    qk_scale = sm_scale * 1.44269504
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)
    if gqa_interleave:
        pid_kh = pid_h % NUM_KV_HEADS
    else:
        pid_kh = pid_h // NUM_SHARE_Q_HEADS
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start
    if BLOCK_SIZE_Q * pid_q >= q_len:
        return
    q_ptrs = tl.make_block_ptr(base=q_ptr + q_start * stride_qn + pid_h *
        stride_qh, shape=(q_len, qk_head_dim), strides=(stride_qn,
        stride_qd), offsets=(pid_q * BLOCK_SIZE_Q, 0), block_shape=(
        BLOCK_SIZE_Q, BLOCK_SIZE_KD), order=(1, 0))
    k_ptrs = tl.make_block_ptr(base=k_ptr + k_start * stride_kn + pid_kh *
        stride_kh, shape=(qk_head_dim, k_len), strides=(stride_kd,
        stride_kn), offsets=(0, 0), block_shape=(BLOCK_SIZE_KD,
        BLOCK_SIZE_K), order=(0, 1))
    v_ptrs = tl.make_block_ptr(base=v_ptr + k_start * stride_vn + pid_kh *
        stride_vh, shape=(k_len, v_head_dim), strides=(stride_vn, stride_vd
        ), offsets=(0, 0), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_VD), order
        =(1, 0))
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option='zero')
    off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q
    off_k = tl.arange(0, BLOCK_SIZE_K)
    m_i = tl.full((BLOCK_SIZE_Q,), float('-inf'), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_Q,), float('-inf'), dtype=tl.float32)
    acc_o = tl.full((BLOCK_SIZE_Q, BLOCK_SIZE_VD), 0, dtype=tl.float32)
    lo = 0
    if causal:
        hi = min(k_len, (pid_q + 1) * BLOCK_SIZE_Q)
    else:
        hi = k_len
    for i in range(lo, hi, BLOCK_SIZE_K):
        i = tl.multiple_of(i, BLOCK_SIZE_K)
        k = tl.load(k_ptrs, boundary_check=(1, 0), padding_option='zero')
        qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        if causal:
            qk += tl.where(off_q[:, None] >= (i + off_k)[None, :], 0, float
                ('-inf'))
        else:
            qk += tl.where((off_k < k_len - i)[None, :], 0, float('-inf'))
        qk += tl.dot(q, k) * qk_scale
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        acc_o_scale = tl.math.exp2(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option='zero')
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)
        m_i = m_ij
        lse_i = m_ij + tl.math.log2(tl.math.exp2(lse_i - m_ij) + l_ij)
        k_ptrs = tl.advance(k_ptrs, (0, BLOCK_SIZE_K))
        v_ptrs = tl.advance(v_ptrs, (BLOCK_SIZE_K, 0))
    acc_o = acc_o * tl.math.exp2(m_i - lse_i)[:, None]
    o_ptrs = tl.make_block_ptr(base=o_ptr + q_start * stride_on + pid_h *
        stride_oh, shape=(q_len, v_head_dim), strides=(stride_on, stride_od
        ), offsets=(pid_q * BLOCK_SIZE_Q, 0), block_shape=(BLOCK_SIZE_Q,
        BLOCK_SIZE_VD), order=(1, 0))
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    l_ptrs = (lse_ptr + q_start * stride_ln + pid_h * stride_lh + off_q *
        stride_ln)
    tl.store(l_ptrs, lse_i, mask=off_q < q_len)


def _flash_attention_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, max_seqlen_q:
    int, max_seqlen_k: int, causal: bool, sm_scale: float, gqa_interleave:
    bool=False):
    assert q.dtype == torch.bfloat16 or q.dtype == torch.float16
    assert k.dtype == q.dtype and v.dtype == q.dtype
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    q_len, num_q_heads, qk_head_dim = q.shape
    k_len, num_k_heads, qk_head_dim = k.shape
    v_len, num_v_heads, v_head_dim = v.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    assert qk_head_dim <= 256 and v_head_dim <= 256, 'head_dim must be less than 256'
    assert q_len == k_len and k_len == v_len
    assert num_k_heads == num_v_heads
    assert num_q_heads % num_k_heads == 0
    num_share_q_heads = num_q_heads // num_k_heads
    o = torch.empty(q.shape[0], q.shape[1], v.shape[-1], dtype=q.dtype,
        device=q.device)
    lse = torch.empty(num_q_heads, q_len, dtype=torch.float32, device=q.device)
    grid = lambda META: (batch_size, num_q_heads, triton.cdiv(max_seqlen_q,
        META['BLOCK_SIZE_Q']))
    BLOCK_SIZE_Q = 128
    BLOCK_SIZE_K = 64
    BLOCK_SIZE_KD = triton.next_power_of_2(qk_head_dim)
    BLOCK_SIZE_VD = triton.next_power_of_2(v_head_dim)
    num_warps, num_stages = get_num_warps_stages(max(qk_head_dim,
        v_head_dim), BLOCK_SIZE_Q, IS_HOPPER_GPU)
    forward_kernel[grid](q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k,
        num_k_heads, num_share_q_heads, qk_head_dim, v_head_dim, sm_scale,
        causal, gqa_interleave, q.stride(0), q.stride(1), q.stride(2), k.
        stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.
        stride(2), o.stride(0), o.stride(1), o.stride(2), lse.stride(0),
        lse.stride(1), BLOCK_SIZE_Q=BLOCK_SIZE_Q, BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_KD=BLOCK_SIZE_KD, BLOCK_SIZE_VD=BLOCK_SIZE_VD, num_warps
        =num_warps, num_stages=num_stages)
    return o, lse


# Forward method (kernel launch code)
def _FlashAttention_forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch
    .Tensor, cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int, max_seqlen_k: int, causal=True, sm_scale=None,
    gqa_interleave=False):
    if sm_scale is None:
        sm_scale = 1 / math.sqrt(q.shape[-1])
    o, lse = _flash_attention_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, causal, sm_scale, gqa_interleave)
    ctx.save_for_backward(q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k)
    ctx.sm_scale = sm_scale
    ctx.max_seqlen_q = max_seqlen_q
    ctx.max_seqlen_k = max_seqlen_k
    ctx.causal = causal
    ctx.gqa_interleave = gqa_interleave
    return o


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def backward_dkdv(q_ptr, k_ptr, v_ptr, lse_ptr, d_ptr, do_ptr, dk_ptr,
    dv_ptr, cu_seqlens_q, cu_seqlens_k, NUM_KV_HEADS, NUM_SHARE_Q_HEADS,
    qk_head_dim, v_head_dim, sm_scale, causal, gqa_interleave, stride_qn,
    stride_qh, stride_qd, stride_kn, stride_kh, stride_kd, stride_vn,
    stride_vh, stride_vd, stride_lh, stride_ln, stride_dh, stride_dn,
    stride_don, stride_doh, stride_dod, stride_dks, stride_dkn, stride_dkh,
    stride_dkd, stride_dvs, stride_dvn, stride_dvh, stride_dvd,
    BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_KD:
    tl.constexpr, BLOCK_SIZE_VD: tl.constexpr):
    qk_scale = sm_scale * 1.44269504
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    if gqa_interleave:
        pid_kh = pid_h % NUM_SHARE_Q_HEADS
        pid_sh = pid_h // NUM_SHARE_Q_HEADS
    else:
        pid_kh = pid_h // NUM_SHARE_Q_HEADS
        pid_sh = pid_h % NUM_SHARE_Q_HEADS
    pid_k = tl.program_id(2)
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start
    if BLOCK_SIZE_K * pid_k >= k_len:
        return
    k_ptrs = tl.make_block_ptr(base=k_ptr + k_start * stride_kn + pid_kh *
        stride_kh, shape=(k_len, qk_head_dim), strides=(stride_kn,
        stride_kd), offsets=(pid_k * BLOCK_SIZE_K, 0), block_shape=(
        BLOCK_SIZE_K, BLOCK_SIZE_KD), order=(1, 0))
    dk_ptrs = tl.make_block_ptr(base=dk_ptr + k_start * stride_dkn + pid_kh *
        stride_dkh + pid_sh * stride_dks, shape=(k_len, qk_head_dim),
        strides=(stride_dkn, stride_dkd), offsets=(pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_KD), order=(1, 0))
    v_ptrs = tl.make_block_ptr(base=v_ptr + k_start * stride_vn + pid_kh *
        stride_vh, shape=(k_len, v_head_dim), strides=(stride_vn, stride_vd
        ), offsets=(pid_k * BLOCK_SIZE_K, 0), block_shape=(BLOCK_SIZE_K,
        BLOCK_SIZE_VD), order=(1, 0))
    dv_ptrs = tl.make_block_ptr(base=dv_ptr + k_start * stride_dvn + pid_kh *
        stride_dvh + pid_sh * stride_dvs, shape=(k_len, v_head_dim),
        strides=(stride_dvn, stride_dvd), offsets=(pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_VD), order=(1, 0))
    off_q = tl.arange(0, BLOCK_SIZE_Q)
    off_k = tl.arange(0, BLOCK_SIZE_K) + pid_k * BLOCK_SIZE_K
    k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option='zero')
    v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option='zero')
    dk = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_KD), dtype=tl.float32)
    dv = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_VD), dtype=tl.float32)
    if causal:
        q_lo = pid_k * BLOCK_SIZE_K
    else:
        q_lo = 0
    q_ptrs = tl.make_block_ptr(base=q_ptr + q_start * stride_qn + pid_h *
        stride_qh, shape=(q_len, qk_head_dim), strides=(stride_qn,
        stride_qd), offsets=(q_lo, 0), block_shape=(BLOCK_SIZE_Q,
        BLOCK_SIZE_KD), order=(1, 0))
    do_ptrs = tl.make_block_ptr(base=do_ptr + q_start * stride_don + pid_h *
        stride_doh, shape=(q_len, v_head_dim), strides=(stride_don,
        stride_dod), offsets=(q_lo, 0), block_shape=(BLOCK_SIZE_Q,
        BLOCK_SIZE_VD), order=(1, 0))
    d_ptrs = tl.make_block_ptr(base=d_ptr + q_start * stride_dn + pid_h *
        stride_dh, shape=(q_len, 1), strides=(stride_dn, stride_dh),
        offsets=(q_lo, 0), block_shape=(BLOCK_SIZE_Q, 1), order=(0, 1))
    lse_ptrs = tl.make_block_ptr(base=lse_ptr + q_start * stride_ln + pid_h *
        stride_lh, shape=(q_len, 1), strides=(stride_ln, stride_lh),
        offsets=(q_lo, 0), block_shape=(BLOCK_SIZE_Q, 1), order=(0, 1))
    for i in range(q_lo, q_len, BLOCK_SIZE_Q):
        q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option='zero')
        do = tl.load(do_ptrs, boundary_check=(0, 1), padding_option='zero')
        lse = tl.load(lse_ptrs, boundary_check=(0, 1), padding_option='zero')
        d = tl.load(d_ptrs, boundary_check=(0, 1), padding_option='zero')
        if causal:
            qk = tl.where((off_q + i)[:, None] >= off_k[None, :], float(0.0
                ), float('-inf'))
        else:
            qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.dot(q, k.T) * qk_scale
        p = tl.math.exp2(qk - lse)
        dp = tl.dot(do, v.T)
        ds = sm_scale * p * (dp - d)
        p = p.to(do.dtype)
        ds = ds.to(q.dtype)
        dk += tl.dot(ds.T, q)
        dv += tl.dot(p.T, do)
        q_ptrs = tl.advance(q_ptrs, (BLOCK_SIZE_Q, 0))
        do_ptrs = tl.advance(do_ptrs, (BLOCK_SIZE_Q, 0))
        lse_ptrs = tl.advance(lse_ptrs, (BLOCK_SIZE_Q, 0))
        d_ptrs = tl.advance(d_ptrs, (BLOCK_SIZE_Q, 0))
    tl.store(dk_ptrs, dk.to(dk_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(dv_ptrs, dv.to(dv_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def backward_dq(q_ptr, k_ptr, v_ptr, lse_ptr, d_ptr, do_ptr, dq_ptr,
    cu_seqlens_q, cu_seqlens_k, NUM_KV_HEADS, NUM_SHARE_Q_HEADS,
    qk_head_dim, v_head_dim, sm_scale, causal, gqa_interleave, stride_qn,
    stride_qh, stride_qd, stride_kn, stride_kh, stride_kd, stride_vn,
    stride_vh, stride_vd, stride_lh, stride_ln, stride_dh, stride_dn,
    stride_don, stride_doh, stride_dod, stride_dqn, stride_dqh, stride_dqd,
    BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_KD:
    tl.constexpr, BLOCK_SIZE_VD: tl.constexpr):
    qk_scale = sm_scale * 1.44269504
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)
    if gqa_interleave:
        pid_kh = pid_h % NUM_KV_HEADS
    else:
        pid_kh = pid_h // NUM_SHARE_Q_HEADS
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start
    if BLOCK_SIZE_Q * pid_q >= q_len:
        return
    q_ptrs = tl.make_block_ptr(base=q_ptr + q_start * stride_qn + pid_h *
        stride_qh, shape=(q_len, qk_head_dim), strides=(stride_qn,
        stride_qd), offsets=(pid_q * BLOCK_SIZE_Q, 0), block_shape=(
        BLOCK_SIZE_Q, BLOCK_SIZE_KD), order=(1, 0))
    dq_ptrs = tl.make_block_ptr(base=dq_ptr + q_start * stride_dqn + pid_h *
        stride_dqh, shape=(q_len, qk_head_dim), strides=(stride_dqn,
        stride_dqd), offsets=(pid_q * BLOCK_SIZE_Q, 0), block_shape=(
        BLOCK_SIZE_Q, BLOCK_SIZE_KD), order=(1, 0))
    k_ptrs = tl.make_block_ptr(base=k_ptr + k_start * stride_kn + pid_kh *
        stride_kh, shape=(k_len, qk_head_dim), strides=(stride_kn,
        stride_kd), offsets=(0, 0), block_shape=(BLOCK_SIZE_K,
        BLOCK_SIZE_KD), order=(1, 0))
    v_ptrs = tl.make_block_ptr(base=v_ptr + k_start * stride_vn + pid_kh *
        stride_vh, shape=(k_len, qk_head_dim), strides=(stride_vn,
        stride_vd), offsets=(0, 0), block_shape=(BLOCK_SIZE_K,
        BLOCK_SIZE_VD), order=(1, 0))
    do_ptrs = tl.make_block_ptr(base=do_ptr + q_start * stride_don + pid_h *
        stride_doh, shape=(q_len, qk_head_dim), strides=(stride_don,
        stride_dod), offsets=(pid_q * BLOCK_SIZE_Q, 0), block_shape=(
        BLOCK_SIZE_Q, BLOCK_SIZE_VD), order=(1, 0))
    d_ptrs = tl.make_block_ptr(base=d_ptr + q_start * stride_dn + pid_h *
        stride_dh, shape=(q_len, 1), strides=(stride_dn, stride_dh),
        offsets=(pid_q * BLOCK_SIZE_Q, 0), block_shape=(BLOCK_SIZE_Q, 1),
        order=(0, 1))
    lse_ptrs = tl.make_block_ptr(base=lse_ptr + q_start * stride_ln + pid_h *
        stride_lh, shape=(q_len, 1), strides=(stride_ln, stride_lh),
        offsets=(pid_q * BLOCK_SIZE_Q, 0), block_shape=(BLOCK_SIZE_Q, 1),
        order=(0, 1))
    off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q
    off_k = tl.arange(0, BLOCK_SIZE_K)
    q = tl.load(q_ptrs, boundary_check=(1, 0), padding_option='zero')
    do = tl.load(do_ptrs, boundary_check=(0, 1), padding_option='zero')
    lse = tl.load(lse_ptrs, boundary_check=(0, 1), padding_option='zero')
    d = tl.load(d_ptrs, boundary_check=(0, 1), padding_option='zero')
    dq = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_KD), dtype=tl.float32)
    if causal:
        k_hi = (pid_q + 1) * BLOCK_SIZE_Q
    else:
        k_hi = k_len
    for j in range(0, k_hi, BLOCK_SIZE_K):
        k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option='zero')
        v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option='zero')
        if causal:
            qk = tl.where(off_q[:, None] >= (off_k + j)[None, :], float(0.0
                ), float('-inf'))
        else:
            qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.dot(q, k.T) * qk_scale
        p = tl.math.exp2(qk - lse)
        dp = tl.dot(do, v.T)
        ds = sm_scale * p * (dp - d)
        ds = ds.to(q.dtype)
        dq += tl.dot(ds, k)
        k_ptrs = tl.advance(k_ptrs, (BLOCK_SIZE_K, 0))
        v_ptrs = tl.advance(v_ptrs, (BLOCK_SIZE_K, 0))
    tl.store(dq_ptrs, dq.to(dq_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def backward_sum_o_do(o_ptr, do_ptr, delta_ptr, o_len, HEAD_DIM, stride_on,
    stride_oh, stride_od, stride_don, stride_doh, stride_dod, stride_dh,
    stride_dn, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_D: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    off_n = pid_n * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    o = tl.load(o_ptr + off_n[:, None] * stride_on + pid_h * stride_oh + 
        off_d[None, :] * stride_od, mask=(off_n[:, None] < o_len) & (off_d[
        None, :] < HEAD_DIM), other=0).to(tl.float32)
    do = tl.load(do_ptr + off_n[:, None] * stride_don + pid_h * stride_doh +
        off_d[None, :] * stride_dod, mask=(off_n[:, None] < o_len) & (off_d
        [None, :] < HEAD_DIM), other=0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(delta_ptr + pid_h * stride_dh + off_n * stride_dn, delta, mask
        =off_n < o_len)


def _flash_attention_bwd(o: torch.Tensor, do: torch.Tensor, lse: torch.
    Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens_q:
    torch.Tensor, cu_seqlens_k: torch.Tensor, max_seqlen_q: int,
    max_seqlen_k: int, causal: bool, sm_scale: float, gqa_interleave: bool=
    False):
    q_len, num_q_heads, qk_head_dim = q.shape
    k_len, num_k_heads, qk_head_dim = k.shape
    v_len, num_v_heads, v_head_dim = v.shape
    o_len, num_o_heads, v_head_dim = o.shape
    num_share_q_heads = num_q_heads // num_k_heads
    delta = torch.empty([num_o_heads, o_len], device=o.device, dtype=torch.
        float32)
    grid = lambda META: (triton.cdiv(o_len, META['BLOCK_SIZE_O']), num_o_heads)
    BLOCK_SIZE_O = 256
    BLOCK_SIZE_VD = triton.next_power_of_2(v_head_dim)
    num_warps, num_stages = get_num_warps_stages(max(qk_head_dim,
        v_head_dim), BLOCK_SIZE_O, IS_HOPPER_GPU)
    backward_sum_o_do[grid](o, do, delta, o_len, v_head_dim, o.stride(0), o
        .stride(1), o.stride(2), do.stride(0), do.stride(1), do.stride(2),
        delta.stride(0), delta.stride(1), BLOCK_SIZE_O=BLOCK_SIZE_O,
        BLOCK_SIZE_D=BLOCK_SIZE_VD, num_warps=num_warps, num_stages=num_stages)
    dk = torch.empty(num_share_q_heads, k_len, num_k_heads, qk_head_dim,
        device=k.device, dtype=k.dtype)
    dv = torch.empty(num_share_q_heads, k_len, num_k_heads, v_head_dim,
        device=k.device, dtype=k.dtype)
    batch_size = cu_seqlens_q.shape[0] - 1
    grid = lambda META: (batch_size, num_q_heads, triton.cdiv(max_seqlen_k,
        META['BLOCK_SIZE_K']))
    BLOCK_SIZE_Q = 64
    BLOCK_SIZE_K = 64
    BLOCK_SIZE_KD = triton.next_power_of_2(qk_head_dim)
    BLOCK_SIZE_VD = triton.next_power_of_2(v_head_dim)
    num_warps, num_stages = get_num_warps_stages(max(qk_head_dim,
        v_head_dim), BLOCK_SIZE_K, IS_HOPPER_GPU)
    backward_dkdv[grid](q, k, v, lse, delta, do, dk, dv, cu_seqlens_q,
        cu_seqlens_k, num_k_heads, num_share_q_heads, qk_head_dim,
        v_head_dim, sm_scale, causal, gqa_interleave, q.stride(0), q.stride
        (1), q.stride(2), k.stride(0), k.stride(1), k.stride(2), v.stride(0
        ), v.stride(1), v.stride(2), lse.stride(0), lse.stride(1), delta.
        stride(0), delta.stride(1), do.stride(0), do.stride(1), do.stride(2
        ), dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3), dv.
        stride(0), dv.stride(1), dv.stride(2), dv.stride(3), BLOCK_SIZE_Q=
        BLOCK_SIZE_Q, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_KD=
        BLOCK_SIZE_KD, BLOCK_SIZE_VD=BLOCK_SIZE_VD, num_warps=num_warps,
        num_stages=num_stages)
    dk = dk.sum(0)
    dv = dv.sum(0)
    dq = torch.empty_like(q)
    grid = lambda META: (batch_size, num_q_heads, triton.cdiv(max_seqlen_q,
        META['BLOCK_SIZE_Q']))
    BLOCK_SIZE_Q = 64 if max(qk_head_dim, v_head_dim) > 128 else 128
    BLOCK_SIZE_K = 64
    num_warps, num_stages = get_num_warps_stages(max(qk_head_dim,
        v_head_dim), BLOCK_SIZE_Q, IS_HOPPER_GPU)
    backward_dq[grid](q, k, v, lse, delta, do, dq, cu_seqlens_q,
        cu_seqlens_k, num_k_heads, num_share_q_heads, qk_head_dim,
        v_head_dim, sm_scale, causal, gqa_interleave, q.stride(0), q.stride
        (1), q.stride(2), k.stride(0), k.stride(1), k.stride(2), v.stride(0
        ), v.stride(1), v.stride(2), lse.stride(0), lse.stride(1), delta.
        stride(0), delta.stride(1), do.stride(0), do.stride(1), do.stride(2
        ), dq.stride(0), dq.stride(1), dq.stride(2), BLOCK_SIZE_Q=
        BLOCK_SIZE_Q, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_KD=
        BLOCK_SIZE_KD, BLOCK_SIZE_VD=BLOCK_SIZE_VD, num_warps=num_warps,
        num_stages=num_stages)
    return dq, dk, dv


# Backward method (kernel launch code)
def _FlashAttention_backward(ctx, do: torch.Tensor, *args) ->Any:
    q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
    max_seqlen_q = ctx.max_seqlen_q
    max_seqlen_k = ctx.max_seqlen_k
    sm_scale = ctx.sm_scale
    causal = ctx.causal
    gqa_interleave = ctx.gqa_interleave
    dq, dk, dv = _flash_attention_bwd(o, do, lse, q, k, v, cu_seqlens_q,
        cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, sm_scale,
        gqa_interleave)
    return dq, dk, dv, None, None, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class FlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int, max_seqlen_k: int, causal=True, sm_scale=None,
        gqa_interleave=False):
        if sm_scale is None:
            sm_scale = 1 / math.sqrt(q.shape[-1])
        o, lse = _flash_attention_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, causal, sm_scale, gqa_interleave)
        ctx.save_for_backward(q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.sm_scale = sm_scale
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.causal = causal
        ctx.gqa_interleave = gqa_interleave
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor, *args) ->Any:
        q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        sm_scale = ctx.sm_scale
        causal = ctx.causal
        gqa_interleave = ctx.gqa_interleave
        dq, dk, dv = _flash_attention_bwd(o, do, lse, q, k, v, cu_seqlens_q,
            cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, sm_scale,
            gqa_interleave)
        return dq, dk, dv, None, None, None, None, None, None, None
