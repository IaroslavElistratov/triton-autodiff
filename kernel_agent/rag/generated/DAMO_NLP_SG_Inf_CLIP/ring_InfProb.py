# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/DAMO-NLP-SG/Inf-CLIP
# Source-Files: inf_cl/ring.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_wg7pjl5w/Inf-CLIP-main/inf_cl/ring.py
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
from math import log

def __init__(self, process_group: dist.ProcessGroup):
    self._process_group = process_group
    self._ops = []
    self.rank = dist.get_rank(self._process_group)
    self.world_size = dist.get_world_size(self._process_group)
    self._reqs = None
    self.send_rank = (self.rank + 1) % self.world_size
    self.recv_rank = (self.rank - 1) % self.world_size
    if process_group is not None:
        self.send_rank = dist.get_global_rank(self._process_group, self.
            send_rank)
        self.recv_rank = dist.get_global_rank(self._process_group, self.
            recv_rank)


def commit(self):
    if self._reqs is not None:
        raise RuntimeError('commit called twice')
    self._reqs = dist.batch_isend_irecv(self._ops)


def send_recv(self, to_send, recv_tensor=None):
    if recv_tensor is None:
        res = torch.empty_like(to_send)
    else:
        res = recv_tensor
    send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self.
        _process_group)
    recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self.
        _process_group)
    self._ops.append(send_op)
    self._ops.append(recv_op)
    return res


def wait(self):
    if self._reqs is None:
        raise RuntimeError('wait called before commit')
    for req in self._reqs:
        req.wait()
    self._reqs = None
    self._ops = []


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
def _InfProb_forward(ctx, q, k, group):
    rank = dist.get_rank()
    k = k.contiguous()
    comm = RingComm(group)
    colle = [q, k]
    lse = None
    next_k = None
    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(k)
            comm.commit()
        block_lse = _flash_prob_forward(q, k)
        if step == 0:
            lse = block_lse
        else:
            lse = lse - F.logsigmoid(lse - block_lse)
        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
    colle.append(lse)
    ctx.save_for_backward(*colle)
    ctx.group = group
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
def _InfProb_backward(ctx, dlse):
    rank = dist.get_rank()
    q, k, lse = ctx.saved_tensors
    k_comm = RingComm(ctx.group)
    d_k_comm = RingComm(ctx.group)
    dq, dk = None, None
    next_dk = None
    block_dq_buffer = torch.empty(q.shape, dtype=torch.float32, device=q.device
        )
    block_dk_buffer = torch.empty(k.shape, dtype=torch.float32, device=k.device
        )
    next_dk, next_k = None, None
    for step in range(k_comm.world_size):
        if step + 1 != k_comm.world_size:
            next_k = k_comm.send_recv(k)
            k_comm.commit()
        block_dq_buffer, block_dk_buffer = _flash_prob_backward(q, k, lse, dlse
            )
        if step == 0:
            dq = block_dq_buffer
            dk = block_dk_buffer
        else:
            dq += block_dq_buffer
            d_k_comm.wait()
            dk = block_dk_buffer + next_dk
        if step + 1 != k_comm.world_size:
            k_comm.wait()
            k = next_k
        next_dk = d_k_comm.send_recv(dk)
        d_k_comm.commit()
    d_k_comm.wait()
    return dq, next_dk, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class InfProb(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, group):
        rank = dist.get_rank()
        k = k.contiguous()
        comm = RingComm(group)
        colle = [q, k]
        lse = None
        next_k = None
        for step in range(comm.world_size):
            if step + 1 != comm.world_size:
                next_k: torch.Tensor = comm.send_recv(k)
                comm.commit()
            block_lse = _flash_prob_forward(q, k)
            if step == 0:
                lse = block_lse
            else:
                lse = lse - F.logsigmoid(lse - block_lse)
            if step + 1 != comm.world_size:
                comm.wait()
                k = next_k
        colle.append(lse)
        ctx.save_for_backward(*colle)
        ctx.group = group
        return lse

    @staticmethod
    def backward(ctx, dlse):
        rank = dist.get_rank()
        q, k, lse = ctx.saved_tensors
        k_comm = RingComm(ctx.group)
        d_k_comm = RingComm(ctx.group)
        dq, dk = None, None
        next_dk = None
        block_dq_buffer = torch.empty(q.shape, dtype=torch.float32, device=
            q.device)
        block_dk_buffer = torch.empty(k.shape, dtype=torch.float32, device=
            k.device)
        next_dk, next_k = None, None
        for step in range(k_comm.world_size):
            if step + 1 != k_comm.world_size:
                next_k = k_comm.send_recv(k)
                k_comm.commit()
            block_dq_buffer, block_dk_buffer = _flash_prob_backward(q, k,
                lse, dlse)
            if step == 0:
                dq = block_dq_buffer
                dk = block_dk_buffer
            else:
                dq += block_dq_buffer
                d_k_comm.wait()
                dk = block_dk_buffer + next_dk
            if step + 1 != k_comm.world_size:
                k_comm.wait()
                k = next_k
            next_dk = d_k_comm.send_recv(dk)
            d_k_comm.commit()
        d_k_comm.wait()
        return dq, next_dk, None
