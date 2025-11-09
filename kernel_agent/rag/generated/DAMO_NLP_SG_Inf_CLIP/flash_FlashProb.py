# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/DAMO-NLP-SG/Inf-CLIP
# Source-Files: inf_cl/flash.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_wg7pjl5w/Inf-CLIP-main/inf_cl/flash.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def _prob_fwd_kernel(Q, K, LSE, nheads, seqlen_q, seqlen_k, BLOCK_HEADDIM:
    tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    ndims = nheads * BLOCK_HEADDIM
    offs_m = tl.arange(0, BLOCK_M) + start_m * BLOCK_M
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = Q + ndims * offs_m[:, None]
    k_ptrs = K + ndims * offs_n[:, None]
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    end_n = seqlen_k
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off_h in range(nheads):
            offs_hd = (offs_d + off_h * BLOCK_HEADDIM)[None, :]
            q = tl.load(q_ptrs + offs_hd, mask=offs_m[:, None] < seqlen_q,
                other=0.0)
            k = tl.load(k_ptrs + offs_hd + start_n * ndims, mask=(start_n +
                offs_n)[:, None] < seqlen_k, other=0.0)
            qk += tl.dot(q, tl.trans(k))
        m_ij = tl.maximum(tl.max(qk, 1), m_i)
        p = tl.exp(qk - m_ij[:, None])
        p = tl.where((start_n + offs_n)[None, :] < seqlen_k, p, 0.0)
        lse_i = tl.exp(m_i - m_ij) * lse_i + tl.sum(p, 1)
        m_i = m_ij
    lse_i = m_i + tl.log(lse_i)
    lse_i = tl.where(offs_m < seqlen_q, lse_i, 0.0)
    tl.store(LSE + offs_m, lse_i)


def _flash_prob_forward(q, k):
    seqlen_q, nheads, d = q.shape
    seqlen_k, _, _ = k.shape
    assert k.shape == (seqlen_k, nheads, d)
    assert q.dtype == k.dtype, 'All tensors must have the same type'
    assert q.is_cuda and k.is_cuda
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty(seqlen_q_rounded, device=q.device, dtype=torch.float32)
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK_M = 64
    BLOCK_N = 64
    num_warps = 8
    num_stages = 1
    grid = lambda META: (triton.cdiv(seqlen_q, META['BLOCK_M']), 1)
    _prob_fwd_kernel[grid](q, k, lse, nheads, seqlen_q, seqlen_k,
        BLOCK_HEADDIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=
        num_warps, num_stages=num_stages)
    lse = lse[:seqlen_q]
    return lse


# Forward method (kernel launch code)
def _FlashProb_forward(ctx, q, k):
    lse = _flash_prob_forward(q, k)
    ctx.save_for_backward(q, k, lse)
    return lse


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _dk_prob_bwd_kernel(Q, K, dK, LSE, dLSE, nheads, seqlen_q, seqlen_k,
    BLOCK_HEADDIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    ASM: tl.constexpr = 'cvt.rna.tf32.f32 $0, $1;'
    start_n = tl.program_id(0)
    ndims = nheads * BLOCK_HEADDIM
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N) + start_n * BLOCK_N
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = Q + ndims * offs_m[:, None]
    k_ptrs = K + ndims * offs_n[:, None]
    dk_ptrs = dK + ndims * offs_n[:, None]
    end_m = seqlen_q
    for start_m in range(0, end_m, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        lse = tl.load(LSE + offs_m + start_m, mask=offs_m < seqlen_q, other=0.0
            )
        dlse = tl.load(dLSE + offs_m + start_m, mask=offs_m < seqlen_q,
            other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off_h in range(nheads):
            offs_hd = (offs_d + off_h * BLOCK_HEADDIM)[None, :]
            q = tl.load(q_ptrs + offs_hd + start_m * ndims, mask=(offs_m +
                start_m)[:, None] < seqlen_q, other=0.0)
            k = tl.load(k_ptrs + offs_hd, mask=offs_n[:, None] < seqlen_k,
                other=0.0)
            qk += tl.dot(q, tl.trans(k))
        qk_grad = tl.exp(qk - lse[:, None])
        qk_grad = tl.where((start_m + offs_m)[:, None] < seqlen_q, qk_grad, 0.0
            )
        qk_grad = qk_grad * dlse[:, None]
        qk_grad = tl.inline_asm_elementwise(ASM, '=r, r', [qk_grad], dtype=
            tl.float32, is_pure=True, pack=1)
        for off_h in range(nheads):
            offs_hd = (offs_d + off_h * BLOCK_HEADDIM)[None, :]
            q = tl.load(q_ptrs + offs_hd + start_m * ndims, mask=(start_m +
                offs_m)[:, None] < seqlen_q, other=0.0)
            k = tl.load(k_ptrs + offs_hd, mask=offs_n[:, None] < seqlen_k,
                other=0.0)
            q = tl.inline_asm_elementwise(ASM, '=r, r', [q], dtype=tl.
                float32, is_pure=True, pack=1)
            k_grad = tl.dot(tl.trans(qk_grad), q)
            dk_h = tl.load(dk_ptrs + offs_hd, mask=offs_n[:, None] <
                seqlen_k, other=0.0)
            tl.store(dk_ptrs + offs_hd, dk_h + k_grad, mask=offs_n[:, None] <
                seqlen_k)


@triton.jit
def _dq_prob_bwd_kernel(Q, K, dQ, LSE, dLSE, nheads, seqlen_q, seqlen_k,
    BLOCK_HEADDIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    ASM: tl.constexpr = 'cvt.rna.tf32.f32 $0, $1;'
    start_m = tl.program_id(0)
    ndims = nheads * BLOCK_HEADDIM
    offs_m = tl.arange(0, BLOCK_M) + start_m * BLOCK_M
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = Q + ndims * offs_m[:, None]
    dq_ptrs = dQ + ndims * offs_m[:, None]
    k_ptrs = K + ndims * offs_n[:, None]
    lse = tl.load(LSE + offs_m, mask=offs_m < seqlen_q, other=0.0)
    dlse = tl.load(dLSE + offs_m, mask=offs_m < seqlen_q, other=0.0)
    end_n = seqlen_k
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off_h in range(nheads):
            offs_hd = (offs_d + off_h * BLOCK_HEADDIM)[None, :]
            q = tl.load(q_ptrs + offs_hd, mask=offs_m[:, None] < seqlen_q,
                other=0.0)
            k = tl.load(k_ptrs + offs_hd + start_n * ndims, mask=(start_n +
                offs_n)[:, None] < seqlen_k, other=0.0)
            qk += tl.dot(q, tl.trans(k))
        qk_grad = tl.exp(qk - lse[:, None])
        qk_grad = tl.where((start_n + offs_n)[None, :] < seqlen_k, qk_grad, 0.0
            )
        qk_grad = qk_grad * dlse[:, None]
        qk_grad = tl.inline_asm_elementwise(ASM, '=r, r', [qk_grad], dtype=
            tl.float32, is_pure=True, pack=1)
        for off_h in range(nheads):
            offs_hd = (offs_d + off_h * BLOCK_HEADDIM)[None, :]
            q = tl.load(q_ptrs + offs_hd, mask=offs_m[:, None] < seqlen_q,
                other=0.0)
            k = tl.load(k_ptrs + offs_hd + start_n * ndims, mask=(start_n +
                offs_n)[:, None] < seqlen_k, other=0.0)
            k = tl.inline_asm_elementwise(ASM, '=r, r', [k], dtype=tl.
                float32, is_pure=True, pack=1)
            q_grad = tl.dot(qk_grad, k)
            dq_h = tl.load(dq_ptrs + offs_hd, mask=offs_m[:, None] <
                seqlen_q, other=0.0)
            tl.store(dq_ptrs + offs_hd, dq_h + q_grad, mask=offs_m[:, None] <
                seqlen_q)


def _flash_prob_backward(q, k, lse, dlse):
    seqlen_q, nheads, d = q.shape
    seqlen_k, _, _ = k.shape
    assert k.shape == (seqlen_k, nheads, d)
    assert q.dtype == k.dtype, 'All tensors must have the same type'
    assert q.is_cuda and k.is_cuda
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    q = q.contiguous()
    k = k.contiguous()
    dlse = dlse.contiguous()
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK_M = 64
    BLOCK_N = 64
    num_warps = 8
    num_stages = 1
    grid = lambda META: (triton.cdiv(seqlen_q, META['BLOCK_M']), 1)
    _dq_prob_bwd_kernel[grid](q, k, dq, lse, dlse, nheads, seqlen_q,
        seqlen_k, BLOCK_HEADDIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages)
    BLOCK_N = BLOCK_M
    BLOCK_M = BLOCK_N
    grid = lambda META: (triton.cdiv(seqlen_k, META['BLOCK_N']), 1)
    _dk_prob_bwd_kernel[grid](q, k, dk, lse, dlse, nheads, seqlen_q,
        seqlen_k, BLOCK_HEADDIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages)
    dq = dq[:seqlen_q]
    dk = dk[:seqlen_k]
    return dq, dk


# Backward method (kernel launch code)
def _FlashProb_backward(ctx, dlse):
    q, k, lse = ctx.saved_tensors
    dq, dk = _flash_prob_backward(q, k, lse, dlse)
    return dq, dk


# ============================================================
# autograd.Function Class Definition
# ============================================================

class FlashProb(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k):
        lse = _flash_prob_forward(q, k)
        ctx.save_for_backward(q, k, lse)
        return lse

    @staticmethod
    def backward(ctx, dlse):
        q, k, lse = ctx.saved_tensors
        dq, dk = _flash_prob_backward(q, k, lse, dlse)
        return dq, dk
