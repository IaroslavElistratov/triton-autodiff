# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/MzeroMiko/VMamba
# Source-Files: classification/models/mamba2/ssd_chunk_state.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_1v8ylcm9/VMamba-main/classification/models/mamba2/ssd_chunk_state.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_M': 128,
    'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K':
    32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128,
    'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K':
    32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K':
    32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 
    32}, num_stages=5, num_warps=2), triton.Config({'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2)],
    key=['hdim', 'dstate', 'chunk_size'])
@triton.jit
def _chunk_state_fwd_kernel(x_ptr, b_ptr, states_ptr, dt_ptr, dA_cumsum_ptr,
    seq_idx_ptr, hdim, dstate, chunk_size, batch, seqlen,
    nheads_ngroups_ratio, stride_x_batch, stride_x_seqlen, stride_x_head,
    stride_x_hdim, stride_b_batch, stride_b_seqlen, stride_b_head,
    stride_b_dstate, stride_states_batch, stride_states_chunk,
    stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head,
    stride_dA_cs_csize, stride_seq_idx_batch, stride_seq_idx_seqlen,
    HAS_SEQ_IDX: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl
    .constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    b_ptr += (pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen +
        pid_h // nheads_ngroups_ratio * stride_b_head)
    x_ptr += (pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen +
        pid_h * stride_x_head)
    dt_ptr += (pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h *
        stride_dt_head)
    dA_cumsum_ptr += (pid_b * stride_dA_cs_batch + pid_c *
        stride_dA_cs_chunk + pid_h * stride_dA_cs_head)
    if HAS_SEQ_IDX:
        seq_idx_ptr += (pid_b * stride_seq_idx_batch + pid_c * chunk_size *
            stride_seq_idx_seqlen)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_hdim + offs_k[None, :] *
        stride_x_seqlen)
    b_ptrs = b_ptr + (offs_n[None, :] * stride_b_dstate + offs_k[:, None] *
        stride_b_seqlen)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize
        ).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    if HAS_SEQ_IDX:
        seq_idx_ptrs = seq_idx_ptr + offs_k * stride_seq_idx_seqlen
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    if HAS_SEQ_IDX:
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) *
            stride_seq_idx_seqlen)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < hdim) & (offs_k[None, :
            ] < chunk_size_limit - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) &
            (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size_limit -
            k, other=0.0).to(tl.float32)
        if HAS_SEQ_IDX:
            seq_idx_k = tl.load(seq_idx_ptrs, mask=offs_k < 
                chunk_size_limit - k, other=-1)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0
            ).to(tl.float32)
        if not HAS_SEQ_IDX:
            scale = tl.exp(dA_cs_last - dA_cs_k) * dt_k
        else:
            scale = tl.where(seq_idx_k == seq_idx_last, tl.exp(dA_cs_last -
                dA_cs_k) * dt_k, 0.0)
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
        if HAS_SEQ_IDX:
            seq_idx_ptrs += BLOCK_SIZE_K * stride_seq_idx_seqlen
    states = acc.to(states_ptr.dtype.element_ty)
    states_ptr += (pid_b * stride_states_batch + pid_c *
        stride_states_chunk + pid_h * stride_states_head)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (offs_m[:, None] * stride_states_hdim + 
        offs_n[None, :] * stride_states_dstate)
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)


def _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=None, states=None,
    states_in_fp32=True):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if states is not None:
        assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    else:
        states_dtype = torch.float32 if states_in_fp32 else B.dtype
        states = torch.empty((batch, nchunks, nheads, headdim, dstate),
            device=x.device, dtype=states_dtype)
    grid = lambda META: (triton.cdiv(headdim, META['BLOCK_SIZE_M']) *
        triton.cdiv(dstate, META['BLOCK_SIZE_N']), batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_state_fwd_kernel[grid](x, B, states, dt, dA_cumsum, seq_idx,
            int(headdim), int(dstate), int(chunk_size), int(batch), int(
            seqlen), int(nheads // ngroups), x.stride(0), x.stride(1), x.
            stride(2), x.stride(3), B.stride(0), B.stride(1), B.stride(2),
            B.stride(-1), states.stride(0), states.stride(1), states.stride
            (2), states.stride(3), states.stride(4), dt.stride(0), dt.
            stride(2), dt.stride(1), dt.stride(3), dA_cumsum.stride(0),
            dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None
             else (0, 0)), HAS_SEQ_IDX=seq_idx is not None)
    return states


# Forward method (kernel launch code)
def _ChunkStateFn_forward(ctx, B, x, dt, dA_cumsum, states_in_fp32=True):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen <= nchunks * chunk_size
    _, _, ngroups, dstate = B.shape
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if x.stride(-1) != 1 and x.stride(1) != 1:
        x = x.contiguous()
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, states_in_fp32=
        states_in_fp32)
    ctx.save_for_backward(B, x, dt, dA_cumsum)
    return states


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N':
    128}, num_stages=3, num_warps=4, pre_hook=init_to_zero([
    'ddA_cumsum_ptr'])), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N':
    32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(['ddA_cumsum_ptr'
    ])), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128},
    num_stages=3, num_warps=4, pre_hook=init_to_zero(['ddA_cumsum_ptr'])),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=3,
    num_warps=4, pre_hook=init_to_zero(['ddA_cumsum_ptr'])), triton.Config(
    {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4,
    pre_hook=init_to_zero(['ddA_cumsum_ptr'])), triton.Config({
    'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4,
    pre_hook=init_to_zero(['ddA_cumsum_ptr'])), triton.Config({
    'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4,
    pre_hook=init_to_zero(['ddA_cumsum_ptr'])), triton.Config({
    'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4,
    pre_hook=init_to_zero(['ddA_cumsum_ptr']))], key=['chunk_size',
    'dstate', 'hdim'])
@triton.jit
def _chunk_state_bwd_db_kernel(x_ptr, dstates_ptr, b_ptr, dt_ptr,
    dA_cumsum_ptr, seq_idx_ptr, db_ptr, ddA_cumsum_ptr, chunk_size, dstate,
    hdim, batch, seqlen, nheads, nheads_per_program, ngroups,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_dstates_batch, stride_dstates_chunk, stride_states_head,
    stride_states_hdim, stride_states_dstate, stride_b_batch,
    stride_b_seqlen, stride_b_head, stride_b_dstate, stride_dt_batch,
    stride_dt_chunk, stride_dt_head, stride_dt_csize, stride_dA_cs_batch,
    stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen, stride_db_batch,
    stride_db_seqlen, stride_db_split, stride_db_group, stride_db_dstate,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head,
    stride_ddA_cs_csize, HAS_DDA_CS: tl.constexpr, HAS_SEQ_IDX: tl.
    constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_sg = tl.program_id(axis=2)
    pid_s = pid_sg // ngroups
    pid_g = pid_sg - pid_s * ngroups
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + (
        pid_g * (nheads // ngroups) + pid_s * nheads_per_program
        ) * stride_x_head
    db_ptr += (pid_b * stride_db_batch + pid_c * chunk_size *
        stride_db_seqlen + pid_g * stride_db_group + pid_s * stride_db_split)
    dstates_ptr += (pid_b * stride_dstates_batch + pid_c *
        stride_dstates_chunk + (pid_g * (nheads // ngroups) + pid_s *
        nheads_per_program) * stride_states_head)
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + (pid_g *
        (nheads // ngroups) + pid_s * nheads_per_program) * stride_dt_head
    dA_cumsum_ptr += (pid_b * stride_dA_cs_batch + pid_c *
        stride_dA_cs_chunk + (pid_g * (nheads // ngroups) + pid_s *
        nheads_per_program) * stride_dA_cs_head)
    if HAS_DDA_CS:
        b_ptr += (pid_b * stride_b_batch + pid_c * chunk_size *
            stride_b_seqlen + pid_g * stride_b_head)
        ddA_cumsum_ptr += (pid_b * stride_ddA_cs_batch + pid_c *
            stride_ddA_cs_chunk + (pid_g * (nheads // ngroups) + pid_s *
            nheads_per_program) * stride_ddA_cs_head)
    if HAS_SEQ_IDX:
        seq_idx_ptr += (pid_b * stride_seq_idx_batch + pid_c * chunk_size *
            stride_seq_idx_seqlen)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_k[None, :] *
        stride_x_hdim)
    dstates_ptrs = dstates_ptr + (offs_n[None, :] * stride_states_dstate + 
        offs_k[:, None] * stride_states_hdim)
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_m * stride_dA_cs_csize
    if HAS_DDA_CS:
        b_ptrs = b_ptr + (offs_m[:, None] * stride_b_seqlen + offs_n[None,
            :] * stride_b_dstate)
        ddA_cumsum_ptrs = ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if HAS_DDA_CS:
        b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (
            offs_n[None, :] < dstate), other=0.0).to(tl.float32)
    if HAS_SEQ_IDX:
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen,
            mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) *
            stride_seq_idx_seqlen)
    nheads_iter = min(nheads_per_program, nheads // ngroups - pid_s *
        nheads_per_program)
    for h in range(nheads_iter):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (
            offs_k[None, :] < hdim), other=0.0)
        dstates = tl.load(dstates_ptrs, mask=(offs_k[:, None] < hdim) & (
            offs_n[None, :] < dstate), other=0.0)
        dstates = dstates.to(x_ptrs.dtype.element_ty)
        db = tl.dot(x, dstates)
        dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) *
            stride_dA_cs_csize).to(tl.float32)
        dA_cs_m = tl.load(dA_cumsum_ptrs, mask=offs_m < chunk_size, other=0.0
            ).to(tl.float32)
        dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size, other=0.0).to(tl.
            float32)
        if not HAS_SEQ_IDX:
            scale = tl.exp(dA_cs_last - dA_cs_m)
        else:
            scale = tl.where(seq_idx_m == seq_idx_last, tl.exp(dA_cs_last -
                dA_cs_m), 0.0)
        db *= (scale * dt_m)[:, None]
        if HAS_DDA_CS:
            ddA_cs = tl.sum(db * b, axis=1)
            tl.atomic_add(ddA_cumsum_ptrs + stride_ddA_cs_csize, ddA_cs,
                mask=offs_m < chunk_size - 1)
        acc += db
        x_ptrs += stride_x_head
        dstates_ptrs += stride_states_head
        dt_ptrs += stride_dt_head
        dA_cumsum_ptr += stride_dA_cs_head
        dA_cumsum_ptrs += stride_dA_cs_head
        if HAS_DDA_CS:
            ddA_cumsum_ptrs += stride_ddA_cs_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    db_ptrs = db_ptr + (offs_m[:, None] * stride_db_seqlen + offs_n[None, :
        ] * stride_db_dstate)
    tl.store(db_ptrs, acc, mask=(offs_m[:, None] < chunk_size_limit) & (
        offs_n[None, :] < dstate))


@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_M': 128,
    'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8,
    pre_hook=init_to_zero(['ddt_ptr', 'ddA_cumsum_ptr'])), triton.Config({
    'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32},
    num_stages=4, num_warps=4, pre_hook=init_to_zero(['ddt_ptr',
    'ddA_cumsum_ptr'])), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N':
    128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=
    init_to_zero(['ddt_ptr', 'ddA_cumsum_ptr'])), triton.Config({
    'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32},
    num_stages=4, num_warps=4, pre_hook=init_to_zero(['ddt_ptr',
    'ddA_cumsum_ptr'])), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N':
    128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=
    init_to_zero(['ddt_ptr', 'ddA_cumsum_ptr'])), triton.Config({
    'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32},
    num_stages=4, num_warps=4, pre_hook=init_to_zero(['ddt_ptr',
    'ddA_cumsum_ptr'])), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N':
    32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4, pre_hook=
    init_to_zero(['ddt_ptr', 'ddA_cumsum_ptr'])), triton.Config({
    'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages
    =5, num_warps=4, pre_hook=init_to_zero(['ddt_ptr', 'ddA_cumsum_ptr'])),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 
    32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(['ddt_ptr',
    'ddA_cumsum_ptr']))], key=['chunk_size', 'hdim', 'dstate'])
@triton.jit
def _chunk_state_bwd_dx_kernel(x_ptr, b_ptr, dstates_ptr, dt_ptr,
    dA_cumsum_ptr, dx_ptr, ddt_ptr, ddA_cumsum_ptr, chunk_size, hdim,
    dstate, batch, seqlen, nheads_ngroups_ratio, stride_x_batch,
    stride_x_seqlen, stride_x_head, stride_x_hdim, stride_b_batch,
    stride_b_seqlen, stride_b_head, stride_b_dstate, stride_dstates_batch,
    stride_dstates_chunk, stride_states_head, stride_states_hdim,
    stride_states_dstate, stride_dt_batch, stride_dt_chunk, stride_dt_head,
    stride_dt_csize, stride_dA_cs_batch, stride_dA_cs_chunk,
    stride_dA_cs_head, stride_dA_cs_csize, stride_dx_batch,
    stride_dx_seqlen, stride_dx_head, stride_dx_hdim, stride_ddt_batch,
    stride_ddt_chunk, stride_ddt_head, stride_ddt_csize,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head,
    stride_ddA_cs_csize, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.
    constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_DSTATE: tl.constexpr):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    x_ptr += (pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen +
        pid_h * stride_x_head)
    b_ptr += (pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen +
        pid_h // nheads_ngroups_ratio * stride_b_head)
    dstates_ptr += (pid_b * stride_dstates_batch + pid_c *
        stride_dstates_chunk + pid_h * stride_states_head)
    dt_ptr += (pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h *
        stride_dt_head)
    ddt_ptr += (pid_b * stride_ddt_batch + pid_c * stride_ddt_chunk + pid_h *
        stride_ddt_head)
    ddA_cumsum_ptr += (pid_b * stride_ddA_cs_batch + pid_c *
        stride_ddA_cs_chunk + pid_h * stride_ddA_cs_head)
    dA_cumsum_ptr += (pid_b * stride_dA_cs_batch + pid_c *
        stride_dA_cs_chunk + pid_h * stride_dA_cs_head)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    offs_k = tl.arange(0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else
        BLOCK_SIZE_K)
    b_ptrs = b_ptr + (offs_m[:, None] * stride_b_seqlen + offs_k[None, :] *
        stride_b_dstate)
    dstates_ptrs = dstates_ptr + (offs_n[None, :] * stride_states_hdim + 
        offs_k[:, None] * stride_states_dstate)
    if BLOCK_SIZE_DSTATE <= 128:
        b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (
            offs_k[None, :] < dstate), other=0.0)
        dstates = tl.load(dstates_ptrs, mask=(offs_k[:, None] < dstate) & (
            offs_n[None, :] < hdim), other=0.0)
        dstates = dstates.to(b_ptr.dtype.element_ty)
        acc = tl.dot(b, dstates)
    else:
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, dstate, BLOCK_SIZE_K):
            b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) &
                (offs_k[None, :] < dstate - k), other=0.0)
            dstates = tl.load(dstates_ptrs, mask=(offs_k[:, None] < dstate -
                k) & (offs_n[None, :] < hdim), other=0.0)
            dstates = dstates.to(b_ptr.dtype.element_ty)
            acc += tl.dot(b, dstates)
            b_ptrs += BLOCK_SIZE_K * stride_b_dstate
            dstates_ptrs += BLOCK_SIZE_K * stride_states_dstate
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize
        ).to(tl.float32)
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_m * stride_dA_cs_csize
    dA_cs_m = tl.load(dA_cumsum_ptrs, mask=offs_m < chunk_size, other=0.0).to(
        tl.float32)
    dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
    acc *= tl.exp(dA_cs_last - dA_cs_m)[:, None]
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] *
        stride_x_hdim)
    x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n
        [None, :] < hdim), other=0.0).to(tl.float32)
    ddt = tl.sum(acc * x, axis=1)
    ddt_ptrs = ddt_ptr + offs_m * stride_ddt_csize
    tl.atomic_add(ddt_ptrs, ddt, mask=offs_m < chunk_size)
    ddA_cs = -(ddt * dt_m)
    ddA_cs_last = -tl.sum(ddA_cs)
    ddA_cumsum_ptrs = ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize
    tl.atomic_add(ddA_cumsum_ptrs, ddA_cs, mask=offs_m < chunk_size)
    tl.atomic_add(ddA_cumsum_ptr + (chunk_size - 1) * stride_ddA_cs_csize,
        ddA_cs_last)
    dx = (acc * dt_m[:, None]).to(dx_ptr.dtype.element_ty)
    dx_ptr += (pid_b * stride_dx_batch + pid_c * chunk_size *
        stride_dx_seqlen + pid_h * stride_dx_head)
    dx_ptrs = dx_ptr + (offs_m[:, None] * stride_dx_seqlen + offs_n[None, :
        ] * stride_dx_hdim)
    tl.store(dx_ptrs, dx, mask=(offs_m[:, None] < chunk_size_limit) & (
        offs_n[None, :] < hdim))


def _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, seq_idx=None, B=None,
    ngroups=1):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    dstate = dstates.shape[-1]
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if B is not None:
        assert B.shape == (batch, seqlen, ngroups, dstate)
        B_strides = B.stride(0), B.stride(1), B.stride(2), B.stride(3)
        ddA_cumsum = torch.empty(batch, nheads, nchunks, chunk_size, device
            =x.device, dtype=torch.float32)
        ddA_cumsum_strides = ddA_cumsum.stride(0), ddA_cumsum.stride(2
            ), ddA_cumsum.stride(1), ddA_cumsum.stride(3)
    else:
        B_strides = 0, 0, 0, 0
        ddA_cumsum = None
        ddA_cumsum_strides = 0, 0, 0, 0
    nheads_ngroups_ratio = nheads // ngroups
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    nheads_per_program = max(min(math.ceil(batch * nchunks * nheads /
        sm_count), nheads_ngroups_ratio), 1)
    nsplits = triton.cdiv(nheads_ngroups_ratio, nheads_per_program)
    dB = torch.empty(batch, seqlen, nsplits, ngroups, dstate, device=x.
        device, dtype=torch.float32)
    grid_db = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) *
        triton.cdiv(dstate, META['BLOCK_SIZE_N']), batch * nchunks, nsplits *
        ngroups)
    with torch.cuda.device(x.device.index):
        _chunk_state_bwd_db_kernel[grid_db](x, dstates, B, dt, dA_cumsum,
            seq_idx, dB, ddA_cumsum, int(chunk_size), int(dstate), int(
            headdim), int(batch), int(seqlen), int(nheads), int(
            nheads_per_program), int(ngroups), x.stride(0), x.stride(1), x.
            stride(2), x.stride(3), dstates.stride(0), dstates.stride(1),
            dstates.stride(2), dstates.stride(3), dstates.stride(4), *
            B_strides, dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(
            3), dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(
            1), dA_cumsum.stride(3), *((seq_idx.stride(0), seq_idx.stride(1
            )) if seq_idx is not None else (0, 0)), dB.stride(0), dB.stride
            (1), dB.stride(2), dB.stride(3), dB.stride(4), *
            ddA_cumsum_strides, HAS_DDA_CS=ddA_cumsum is not None,
            HAS_SEQ_IDX=seq_idx is not None, BLOCK_SIZE_K=max(triton.
            next_power_of_2(headdim), 16))
    dB = dB.sum(2)
    if ddA_cumsum is not None:
        torch.cumsum(ddA_cumsum, dim=-1, out=ddA_cumsum)
    return dB if B is None else (dB, ddA_cumsum)


def _chunk_state_bwd_dx(B, x, dt, dA_cumsum, dstates, dx=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    if dx is not None:
        assert dx.shape == x.shape
    else:
        dx = torch.empty_like(x)
    ddt = torch.empty(batch, nheads, nchunks, chunk_size, device=dt.device,
        dtype=torch.float32)
    ddA_cumsum = torch.empty(batch, nheads, nchunks, chunk_size, device=
        dA_cumsum.device, dtype=torch.float32)
    grid_dx = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) *
        triton.cdiv(headdim, META['BLOCK_SIZE_N']), batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_state_bwd_dx_kernel[grid_dx](x, B, dstates, dt, dA_cumsum,
            dx, ddt, ddA_cumsum, int(chunk_size), int(headdim), int(dstate),
            int(batch), int(seqlen), int(nheads // ngroups), x.stride(0), x
            .stride(1), x.stride(2), x.stride(3), B.stride(0), B.stride(1),
            B.stride(2), B.stride(-1), dstates.stride(0), dstates.stride(1),
            dstates.stride(2), dstates.stride(3), dstates.stride(4), dt.
            stride(0), dt.stride(2), dt.stride(1), dt.stride(3), dA_cumsum.
            stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.
            stride(3), dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(
            3), ddt.stride(0), ddt.stride(2), ddt.stride(1), ddt.stride(3),
            ddA_cumsum.stride(0), ddA_cumsum.stride(2), ddA_cumsum.stride(1
            ), ddA_cumsum.stride(3), BLOCK_SIZE_DSTATE=max(triton.
            next_power_of_2(dstate), 16))
    return dx, ddt.to(dt.dtype), ddA_cumsum.to(dA_cumsum.dtype)


# Backward method (kernel launch code)
def _ChunkStateFn_backward(ctx, dstates):
    B, x, dt, dA_cumsum = ctx.saved_tensors
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    if dstates.stride(-1) != 1:
        dstates = dstates.contiguous()
    dx, ddt, ddA_cumsum = _chunk_state_bwd_dx(B, x, dt, dA_cumsum, dstates)
    dB = _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, ngroups=ngroups)
    dB = dB.to(B.dtype)
    return dB, dx, ddt, ddA_cumsum, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class ChunkStateFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, B, x, dt, dA_cumsum, states_in_fp32=True):
        batch, seqlen, nheads, headdim = x.shape
        _, _, nchunks, chunk_size = dt.shape
        assert seqlen <= nchunks * chunk_size
        _, _, ngroups, dstate = B.shape
        assert B.shape == (batch, seqlen, ngroups, dstate)
        assert dt.shape == (batch, nheads, nchunks, chunk_size)
        assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
        if B.stride(-1) != 1:
            B = B.contiguous()
        if x.stride(-1) != 1 and x.stride(1) != 1:
            x = x.contiguous()
        states = _chunk_state_fwd(B, x, dt, dA_cumsum, states_in_fp32=
            states_in_fp32)
        ctx.save_for_backward(B, x, dt, dA_cumsum)
        return states

    @staticmethod
    def backward(ctx, dstates):
        B, x, dt, dA_cumsum = ctx.saved_tensors
        batch, seqlen, nheads, headdim = x.shape
        _, _, nchunks, chunk_size = dt.shape
        _, _, ngroups, dstate = B.shape
        assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
        if dstates.stride(-1) != 1:
            dstates = dstates.contiguous()
        dx, ddt, ddA_cumsum = _chunk_state_bwd_dx(B, x, dt, dA_cumsum, dstates)
        dB = _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, ngroups=ngroups)
        dB = dB.to(B.dtype)
        return dB, dx, ddt, ddA_cumsum, None
