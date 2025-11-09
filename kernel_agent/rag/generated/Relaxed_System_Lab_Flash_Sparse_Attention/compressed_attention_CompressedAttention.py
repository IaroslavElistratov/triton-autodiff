# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention
# Source-Files: nsa_ref/ops/compressed_attention.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_vvmnxazz/Flash-Sparse-Attention-main/nsa_ref/ops/compressed_attention.py
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
    return num_warps, num_stages


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def forward_kernel(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, kernel_size,
    kernel_stride, cu_seqlens_q, cu_seqlens_k, NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS, HEAD_DIM, sm_scale, stride_qn, stride_qh, stride_qd,
    stride_kn, stride_kh, stride_kd, stride_vn, stride_vh, stride_vd,
    stride_on, stride_oh, stride_od, stride_lh, stride_ln, BLOCK_SIZE_Q: tl
    .constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_D: tl.constexpr):
    qk_scale = sm_scale * 1.44269504
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)
    pid_kh = pid_h // NUM_SHARE_Q_HEADS
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start
    q_start_in_seq = pid_q * BLOCK_SIZE_Q + kernel_size - 1
    if q_start_in_seq >= q_len:
        return
    q_ptrs = tl.make_block_ptr(base=q_ptr + q_start * stride_qn + pid_h *
        stride_qh, shape=(q_len, HEAD_DIM), strides=(stride_qn, stride_qd),
        offsets=(q_start_in_seq, 0), block_shape=(BLOCK_SIZE_Q,
        BLOCK_SIZE_D), order=(1, 0))
    k_ptrs = tl.make_block_ptr(base=k_ptr + k_start * stride_kn + pid_kh *
        stride_kh, shape=(HEAD_DIM, k_len), strides=(stride_kd, stride_kn),
        offsets=(0, 0), block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K), order=(0, 1))
    v_ptrs = tl.make_block_ptr(base=v_ptr + k_start * stride_vn + pid_kh *
        stride_vh, shape=(k_len, HEAD_DIM), strides=(stride_vn, stride_vd),
        offsets=(0, 0), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D), order=(1, 0))
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option='zero')
    off_q = tl.arange(0, BLOCK_SIZE_Q) + q_start_in_seq
    off_k = tl.arange(0, BLOCK_SIZE_K) * kernel_stride + kernel_size - 1
    m_i = tl.full((BLOCK_SIZE_Q,), float('-inf'), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_Q,), float('-inf'), dtype=tl.float32)
    acc_o = tl.full((BLOCK_SIZE_Q, BLOCK_SIZE_D), 0, dtype=tl.float32)
    lo = 0
    hi = min(k_len, (q_start_in_seq + BLOCK_SIZE_Q - kernel_size) //
        kernel_stride + 1)
    for i in range(lo, hi, BLOCK_SIZE_K):
        i = tl.multiple_of(i, BLOCK_SIZE_K)
        k = tl.load(k_ptrs, boundary_check=(1, 0), padding_option='zero')
        qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where(off_q[:, None] >= (i * kernel_stride + off_k)[None,
            :], 0, float('-inf'))
        qk += tl.dot(q, k) * qk_scale
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        acc_o_scale = tl.exp2(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option='zero')
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)
        m_i = m_ij
        lse_i = m_ij + tl.math.log2(tl.exp2(lse_i - m_ij) + l_ij)
        k_ptrs = tl.advance(k_ptrs, (0, BLOCK_SIZE_K))
        v_ptrs = tl.advance(v_ptrs, (BLOCK_SIZE_K, 0))
    acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]
    o_ptrs = tl.make_block_ptr(base=o_ptr + q_start * stride_on + pid_h *
        stride_oh, shape=(q_len, HEAD_DIM), strides=(stride_on, stride_od),
        offsets=(q_start_in_seq, 0), block_shape=(BLOCK_SIZE_Q,
        BLOCK_SIZE_D), order=(1, 0))
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    l_ptrs = (lse_ptr + q_start * stride_ln + pid_h * stride_lh + off_q *
        stride_ln)
    tl.store(l_ptrs, lse_i, mask=off_q < q_len)


def _compressed_attention_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.
    Tensor, kernel_size: int, kernel_stride: int, cu_seqlens_q: torch.
    Tensor, cu_seqlens_k: torch.Tensor, max_seqlen_q: torch.Tensor,
    max_seqlen_k: torch.Tensor, sm_scale: float):
    assert k.dtype == q.dtype and v.dtype == q.dtype
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    q_len, num_q_heads, head_dim = q.shape
    k_len, num_k_heads, head_dim = k.shape
    v_len, num_v_heads, head_dim = v.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    assert k_len == v_len and q_len > k_len
    assert num_k_heads == num_v_heads
    assert num_q_heads % num_k_heads == 0
    num_share_q_heads = num_q_heads // num_k_heads
    o = torch.zeros_like(q)
    lse = torch.full((num_q_heads, q_len), fill_value=-torch.inf, dtype=
        torch.float32, device=q.device)
    grid = lambda META: (batch_size, num_q_heads, triton.cdiv(max_seqlen_q,
        META['BLOCK_SIZE_Q']))
    BLOCK_SIZE_Q = 128
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_Q,
        IS_HOPPER_GPU)
    forward_kernel[grid](q, k, v, o, lse, kernel_size, kernel_stride,
        cu_seqlens_q, cu_seqlens_k, num_k_heads, num_share_q_heads,
        head_dim, sm_scale, q.stride(0), q.stride(1), q.stride(2), k.stride
        (0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2
        ), o.stride(0), o.stride(1), o.stride(2), lse.stride(0), lse.stride
        (1), BLOCK_SIZE_Q=BLOCK_SIZE_Q, BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D, num_warps=num_warps, num_stages=num_stages)
    return o, lse


# Forward method (kernel launch code)
def _CompressedAttention_forward(ctx, q: torch.Tensor, k: torch.Tensor, v:
    torch.Tensor, kernel_size: int, kernel_stride: int, cu_seqlens_q: torch
    .Tensor, cu_seqlens_k: torch.Tensor, max_seqlen_q: torch.Tensor,
    max_seqlen_k: torch.Tensor, sm_scale=None):
    assert q.dtype == torch.bfloat16 or q.dtype == torch.float16
    assert q.dtype == k.dtype and k.dtype == v.dtype
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    if sm_scale is None:
        sm_scale = 1 / math.sqrt(q.shape[-1])
    o, lse = _compressed_attention_fwd(q, k, v, kernel_size, kernel_stride,
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, sm_scale)
    ctx.save_for_backward(q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k)
    ctx.sm_scale = sm_scale
    ctx.max_seqlen_q = max_seqlen_q
    ctx.max_seqlen_k = max_seqlen_k
    ctx.kernel_size = kernel_size
    ctx.kernel_stride = kernel_stride
    return o, lse


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def backward_dkdv(q_ptr, k_ptr, v_ptr, lse_ptr, d_ptr, do_ptr, dk_ptr,
    dv_ptr, kernel_size, kernel_stride, cu_seqlens_q, cu_seqlens_k,
    NUM_KV_HEADS, NUM_SHARE_Q_HEADS, HEAD_DIM, sm_scale, stride_qn,
    stride_qh, stride_qd, stride_kn, stride_kh, stride_kd, stride_vn,
    stride_vh, stride_vd, stride_lh, stride_ln, stride_dh, stride_dn,
    stride_don, stride_doh, stride_dod, stride_dks, stride_dkn, stride_dkh,
    stride_dkd, stride_dvs, stride_dvn, stride_dvh, stride_dvd,
    BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_D:
    tl.constexpr):
    qk_scale = sm_scale * 1.44269504
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
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
        stride_kh, shape=(k_len, HEAD_DIM), strides=(stride_kn, stride_kd),
        offsets=(pid_k * BLOCK_SIZE_K, 0), block_shape=(BLOCK_SIZE_K,
        BLOCK_SIZE_D), order=(1, 0))
    dk_ptrs = tl.make_block_ptr(base=dk_ptr + k_start * stride_dkn + pid_kh *
        stride_dkh + pid_sh * stride_dks, shape=(k_len, HEAD_DIM), strides=
        (stride_dkn, stride_dkd), offsets=(pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D), order=(1, 0))
    v_ptrs = tl.make_block_ptr(base=v_ptr + k_start * stride_vn + pid_kh *
        stride_vh, shape=(k_len, HEAD_DIM), strides=(stride_vn, stride_vd),
        offsets=(pid_k * BLOCK_SIZE_K, 0), block_shape=(BLOCK_SIZE_K,
        BLOCK_SIZE_D), order=(1, 0))
    dv_ptrs = tl.make_block_ptr(base=dv_ptr + k_start * stride_dvn + pid_kh *
        stride_dvh + pid_sh * stride_dvs, shape=(k_len, HEAD_DIM), strides=
        (stride_dvn, stride_dvd), offsets=(pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D), order=(1, 0))
    off_q = tl.arange(0, BLOCK_SIZE_Q)
    off_k = pid_k * BLOCK_SIZE_K * kernel_stride + tl.arange(0, BLOCK_SIZE_K
        ) * kernel_stride + kernel_size - 1
    k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option='zero')
    v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option='zero')
    dk = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_D), dtype=tl.float32)
    q_lo = pid_k * BLOCK_SIZE_K * kernel_stride + kernel_size - 1
    q_ptrs = tl.make_block_ptr(base=q_ptr + q_start * stride_qn + pid_h *
        stride_qh, shape=(HEAD_DIM, q_len), strides=(stride_qd, stride_qn),
        offsets=(0, q_lo), block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_Q), order=
        (0, 1))
    do_ptrs = tl.make_block_ptr(base=do_ptr + q_start * stride_don + pid_h *
        stride_doh, shape=(HEAD_DIM, q_len), strides=(stride_dod,
        stride_don), offsets=(0, q_lo), block_shape=(BLOCK_SIZE_D,
        BLOCK_SIZE_Q), order=(0, 1))
    d_ptrs = tl.make_block_ptr(base=d_ptr + q_start * stride_dn + pid_h *
        stride_dh, shape=(1, q_len), strides=(0, stride_dn), offsets=(0,
        q_lo), block_shape=(1, BLOCK_SIZE_Q), order=(1, 0))
    lse_ptrs = tl.make_block_ptr(base=lse_ptr + q_start * stride_ln + pid_h *
        stride_lh, shape=(1, q_len), strides=(0, stride_ln), offsets=(0,
        q_lo), block_shape=(1, BLOCK_SIZE_Q), order=(0, 1))
    for i in range(q_lo, q_len, BLOCK_SIZE_Q):
        q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option='zero')
        do = tl.load(do_ptrs, boundary_check=(0, 1), padding_option='zero')
        lse = tl.load(lse_ptrs, boundary_check=(0, 1), padding_option='zero')
        d = tl.load(d_ptrs, boundary_check=(0, 1), padding_option='zero')
        qk = tl.where(off_k[:, None] <= (off_q + i)[None, :], float(0.0),
            float('-inf'))
        qk += tl.dot(k, q) * qk_scale
        p = tl.exp2(qk - lse)
        dp = tl.dot(v, do)
        ds = sm_scale * p * (dp - d)
        p = p.to(do.dtype)
        ds = ds.to(q.dtype)
        dk += tl.dot(ds, tl.trans(q))
        dv += tl.dot(p, tl.trans(do))
        q_ptrs = tl.advance(q_ptrs, (0, BLOCK_SIZE_Q))
        do_ptrs = tl.advance(do_ptrs, (0, BLOCK_SIZE_Q))
        lse_ptrs = tl.advance(lse_ptrs, (0, BLOCK_SIZE_Q))
        d_ptrs = tl.advance(d_ptrs, (0, BLOCK_SIZE_Q))
    tl.store(dk_ptrs, dk.to(dk_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(dv_ptrs, dv.to(dv_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def backward_dq(q_ptr, k_ptr, v_ptr, lse_ptr, d_ptr, do_ptr, dq_ptr,
    kernel_size, kernel_stride, cu_seqlens_q, cu_seqlens_k, NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS, HEAD_DIM, sm_scale, stride_qn, stride_qh, stride_qd,
    stride_kn, stride_kh, stride_kd, stride_vn, stride_vh, stride_vd,
    stride_lh, stride_ln, stride_dh, stride_dn, stride_don, stride_doh,
    stride_dod, stride_dqn, stride_dqh, stride_dqd, BLOCK_SIZE_Q: tl.
    constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_D: tl.constexpr):
    qk_scale = sm_scale * 1.44269504
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)
    pid_kh = pid_h // NUM_SHARE_Q_HEADS
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start
    q_start_in_seq = pid_q * BLOCK_SIZE_Q + kernel_size - 1
    if q_start_in_seq >= q_len:
        return
    q_ptrs = tl.make_block_ptr(base=q_ptr + q_start * stride_qn + pid_h *
        stride_qh, shape=(q_len, HEAD_DIM), strides=(stride_qn, stride_qd),
        offsets=(q_start_in_seq, 0), block_shape=(BLOCK_SIZE_Q,
        BLOCK_SIZE_D), order=(1, 0))
    dq_ptrs = tl.make_block_ptr(base=dq_ptr + q_start * stride_dqn + pid_h *
        stride_dqh, shape=(q_len, HEAD_DIM), strides=(stride_dqn,
        stride_dqd), offsets=(q_start_in_seq, 0), block_shape=(BLOCK_SIZE_Q,
        BLOCK_SIZE_D), order=(1, 0))
    k_ptrs = tl.make_block_ptr(base=k_ptr + k_start * stride_kn + pid_kh *
        stride_kh, shape=(k_len, HEAD_DIM), strides=(stride_kn, stride_kd),
        offsets=(0, 0), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D), order=(1, 0))
    v_ptrs = tl.make_block_ptr(base=v_ptr + k_start * stride_vn + pid_kh *
        stride_vh, shape=(HEAD_DIM, k_len), strides=(stride_vd, stride_vn),
        offsets=(0, 0), block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K), order=(0, 1))
    do_ptrs = tl.make_block_ptr(base=do_ptr + q_start * stride_don + pid_h *
        stride_doh, shape=(q_len, HEAD_DIM), strides=(stride_don,
        stride_dod), offsets=(q_start_in_seq, 0), block_shape=(BLOCK_SIZE_Q,
        BLOCK_SIZE_D), order=(1, 0))
    d_ptrs = tl.make_block_ptr(base=d_ptr + q_start * stride_dn + pid_h *
        stride_dh, shape=(q_len, 1), strides=(stride_dn, stride_dh),
        offsets=(q_start_in_seq, 0), block_shape=(BLOCK_SIZE_Q, 1), order=(
        0, 1))
    lse_ptrs = tl.make_block_ptr(base=lse_ptr + q_start * stride_ln + pid_h *
        stride_lh, shape=(q_len, 1), strides=(stride_ln, stride_lh),
        offsets=(q_start_in_seq, 0), block_shape=(BLOCK_SIZE_Q, 1), order=(
        0, 1))
    off_q = tl.arange(0, BLOCK_SIZE_Q) + q_start_in_seq
    off_k = tl.arange(0, BLOCK_SIZE_K) * kernel_stride + kernel_size - 1
    q = tl.load(q_ptrs, boundary_check=(1, 0), padding_option='zero')
    do = tl.load(do_ptrs, boundary_check=(0, 1), padding_option='zero')
    lse = tl.load(lse_ptrs, boundary_check=(0, 1), padding_option='zero')
    d = tl.load(d_ptrs, boundary_check=(0, 1), padding_option='zero')
    dq = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_D), dtype=tl.float32)
    lo = 0
    hi = min(k_len, (q_start_in_seq + BLOCK_SIZE_Q - kernel_size) //
        kernel_stride + 1)
    for i in range(lo, hi, BLOCK_SIZE_K):
        k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option='zero')
        v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option='zero')
        qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where(off_q[:, None] >= (i * kernel_stride + off_k)[None,
            :], 0, float('-inf'))
        qk += tl.dot(q, tl.trans(k)) * qk_scale
        p = tl.exp2(qk - lse)
        dp = tl.dot(do, v)
        ds = sm_scale * p * (dp - d)
        ds = ds.to(q.dtype)
        dq += tl.dot(ds, k)
        k_ptrs = tl.advance(k_ptrs, (BLOCK_SIZE_K, 0))
        v_ptrs = tl.advance(v_ptrs, (0, BLOCK_SIZE_K))
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


def _compressed_attention_bwd(o: torch.Tensor, do: torch.Tensor, lse: torch
    .Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kernel_size:
    int, kernel_stride: int, cu_seqlens_q: torch.Tensor, cu_seqlens_k:
    torch.Tensor, max_seqlen_q: torch.Tensor, max_seqlen_k: torch.Tensor,
    sm_scale: float):
    q_len, num_q_heads, head_dim = q.shape
    k_len, num_k_heads, head_dim = k.shape
    v_len, num_v_heads, head_dim = v.shape
    o_len, num_o_heads, head_dim = o.shape
    num_share_q_heads = num_q_heads // num_k_heads
    delta = torch.zeros([num_o_heads, o_len], device=o.device, dtype=torch.
        float32)
    grid = lambda META: (triton.cdiv(o_len, META['BLOCK_SIZE_O']), num_o_heads)
    BLOCK_SIZE_O = 256
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_O,
        IS_HOPPER_GPU)
    backward_sum_o_do[grid](o, do, delta, o_len, head_dim, o.stride(0), o.
        stride(1), o.stride(2), do.stride(0), do.stride(1), do.stride(2),
        delta.stride(0), delta.stride(1), BLOCK_SIZE_O=BLOCK_SIZE_O,
        BLOCK_SIZE_D=BLOCK_SIZE_D, num_warps=num_warps, num_stages=num_stages)
    dk = torch.zeros(num_share_q_heads, k_len, num_k_heads, head_dim,
        device=k.device, dtype=k.dtype)
    dv = torch.zeros(num_share_q_heads, k_len, num_k_heads, head_dim,
        device=k.device, dtype=k.dtype)
    batch_size = cu_seqlens_q.shape[0] - 1
    grid = lambda META: (batch_size, num_q_heads, triton.cdiv(max_seqlen_k,
        META['BLOCK_SIZE_K']))
    BLOCK_SIZE_Q = 64
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_K,
        IS_HOPPER_GPU)
    backward_dkdv[grid](q, k, v, lse, delta, do, dk, dv, kernel_size,
        kernel_stride, cu_seqlens_q, cu_seqlens_k, num_k_heads,
        num_share_q_heads, head_dim, sm_scale, q.stride(0), q.stride(1), q.
        stride(2), k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.
        stride(1), v.stride(2), lse.stride(0), lse.stride(1), delta.stride(
        0), delta.stride(1), do.stride(0), do.stride(1), do.stride(2), dk.
        stride(0), dk.stride(1), dk.stride(2), dk.stride(3), dv.stride(0),
        dv.stride(1), dv.stride(2), dv.stride(3), BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_D=BLOCK_SIZE_D, num_warps=
        num_warps, num_stages=num_stages)
    dk = dk.sum(0)
    dv = dv.sum(0)
    dq = torch.zeros_like(q)
    grid = lambda META: (batch_size, num_q_heads, triton.cdiv(max_seqlen_q,
        META['BLOCK_SIZE_Q']))
    BLOCK_SIZE_Q = 128
    BLOCK_SIZE_K = 64
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_Q,
        IS_HOPPER_GPU)
    backward_dq[grid](q, k, v, lse, delta, do, dq, kernel_size,
        kernel_stride, cu_seqlens_q, cu_seqlens_k, num_k_heads,
        num_share_q_heads, head_dim, sm_scale, q.stride(0), q.stride(1), q.
        stride(2), k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.
        stride(1), v.stride(2), lse.stride(0), lse.stride(1), delta.stride(
        0), delta.stride(1), do.stride(0), do.stride(1), do.stride(2), dq.
        stride(0), dq.stride(1), dq.stride(2), BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_D=BLOCK_SIZE_D, num_warps=
        num_warps, num_stages=num_stages)
    return dq, dk, dv


# Backward method (kernel launch code)
def _CompressedAttention_backward(ctx, do: torch.Tensor, *args) ->Any:
    q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
    max_seqlen_q = ctx.max_seqlen_q
    max_seqlen_k = ctx.max_seqlen_k
    sm_scale = ctx.sm_scale
    kernel_size = ctx.kernel_size
    kernel_stride = ctx.kernel_stride
    dq, dk, dv = _compressed_attention_bwd(o, do, lse, q, k, v, kernel_size,
        kernel_stride, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
        max_seqlen_k, sm_scale)
    return dq, dk, dv, None, None, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class CompressedAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        kernel_size: int, kernel_stride: int, cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor, max_seqlen_q: torch.Tensor,
        max_seqlen_k: torch.Tensor, sm_scale=None):
        assert q.dtype == torch.bfloat16 or q.dtype == torch.float16
        assert q.dtype == k.dtype and k.dtype == v.dtype
        assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
        if sm_scale is None:
            sm_scale = 1 / math.sqrt(q.shape[-1])
        o, lse = _compressed_attention_fwd(q, k, v, kernel_size,
            kernel_stride, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
            max_seqlen_k, sm_scale)
        ctx.save_for_backward(q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.sm_scale = sm_scale
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.kernel_size = kernel_size
        ctx.kernel_stride = kernel_stride
        return o, lse

    @staticmethod
    def backward(ctx, do: torch.Tensor, *args) ->Any:
        q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        sm_scale = ctx.sm_scale
        kernel_size = ctx.kernel_size
        kernel_stride = ctx.kernel_stride
        dq, dk, dv = _compressed_attention_bwd(o, do, lse, q, k, v,
            kernel_size, kernel_stride, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, sm_scale)
        return dq, dk, dv, None, None, None, None, None, None, None
