# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/MzeroMiko/VMamba
# Source-Files: classification/models/mamba2/ssd_combined.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_1v8ylcm9/VMamba-main/classification/models/mamba2/ssd_combined.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

def _bmm_chunk_fwd(a, b, chunk_size, seq_idx=None, causal=False,
    output_dtype=None):
    """
    Argument:
        a: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        b: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        seq_idx: (batch, seqlen) or None. out[i, j] for seq_idx[i] != seq_idx[j] will be zeroed out.
        causal: if True, then out[i, j] for i > j will be arbitrary, only out[i, j] for i <= j are
            guaranteed to be correct.
    Return:
        out: (batch, nchunks, chunk_size, chunk_size) or (batch, nchunks, ngroups, chunk_size, chunk_size)
    """
    has_groups = a.dim() == 4
    if not has_groups:
        batch, seqlen, k = a.shape
    else:
        batch, seqlen, ngroups, k = a.shape
    assert b.shape == a.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if a.stride(-1) != 1 and a.stride(1) != 1:
        a = a.contiguous()
    if b.stride(-1) != 1 and b.stride(1) != 1:
        b = b.contiguous()
    nchunks = math.ceil(seqlen / chunk_size)
    out_dtype = a.dtype if output_dtype is None else output_dtype
    out = torch.empty((batch, nchunks, chunk_size, chunk_size) if not
        has_groups else (batch, nchunks, ngroups, chunk_size, chunk_size),
        device=a.device, dtype=out_dtype)
    dot_dtype = (tl.bfloat16 if a.dtype == torch.bfloat16 or b.dtype ==
        torch.bfloat16 else tl.float16 if a.dtype == torch.float16 or b.
        dtype == torch.float16 else tl.float32)
    grid = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) *
        triton.cdiv(chunk_size, META['BLOCK_SIZE_N']), batch, nchunks if 
        not has_groups else nchunks * ngroups)
    with torch.cuda.device(a.device.index):
        _bmm_chunk_fwd_kernel[grid](a, b, out, seq_idx, int(seqlen), int(
            chunk_size), int(k), int(ngroups if has_groups else 1), a.
            stride(0), a.stride(1), 0 if not has_groups else a.stride(2), a
            .stride(-1), b.stride(0), b.stride(1), 0 if not has_groups else
            b.stride(2), b.stride(-1), out.stride(0), out.stride(1), 0 if 
            not has_groups else out.stride(2), out.stride(-2), out.stride(-
            1), *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not
            None else (0, 0)), causal, dot_dtype, HAS_SEQ_IDX=seq_idx is not
            None)
    return out


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
    key=['chunk_size', 'K', 'IS_CAUSAL'])
@triton.jit
def _bmm_chunk_fwd_kernel(a_ptr, b_ptr, out_ptr, seq_idx_ptr, seqlen,
    chunk_size, K, ngroups, stride_a_batch, stride_a_seqlen, stride_a_head,
    stride_ak, stride_b_batch, stride_b_seqlen, stride_b_head, stride_bk,
    stride_out_batch, stride_out_chunk, stride_out_head, stride_outm,
    stride_outn, stride_seq_idx_batch, stride_seq_idx_seqlen, IS_CAUSAL: tl
    .constexpr, dot_dtype: tl.constexpr, HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K:
    tl.constexpr):
    pid_b = tl.program_id(axis=1)
    pid_ch = tl.program_id(axis=2)
    pid_c = pid_ch // ngroups
    pid_h = pid_ch - pid_c * ngroups
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    if IS_CAUSAL:
        if pid_n * BLOCK_SIZE_N >= (pid_m + 1) * BLOCK_SIZE_M:
            return
    a_ptr += (pid_b * stride_a_batch + pid_c * chunk_size * stride_a_seqlen +
        pid_h * stride_a_head)
    b_ptr += (pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen +
        pid_h * stride_b_head)
    if HAS_SEQ_IDX:
        seq_idx_ptr += (pid_b * stride_seq_idx_batch + pid_c * chunk_size *
            stride_seq_idx_seqlen)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_a_seqlen + offs_k[None, :] *
        stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] *
        stride_b_seqlen)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (
            offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0).to(dot_dtype)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) &
            (offs_n[None, :] < chunk_size_limit), other=0.0).to(dot_dtype)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_SEQ_IDX:
        chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen,
            mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_n = tl.load(seq_idx_ptr + offs_n * stride_seq_idx_seqlen,
            mask=offs_n < chunk_size_limit, other=-2)
        acc = tl.where(seq_idx_m[:, None] == seq_idx_n[None, :], acc, 0.0)
    out = acc.to(out_ptr.dtype.element_ty)
    out_ptr += (pid_b * stride_out_batch + pid_c * stride_out_chunk + pid_h *
        stride_out_head)
    out_ptrs = out_ptr + (stride_outm * offs_m[:, None] + offs_n[None, :] *
        stride_outn)
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < chunk_size) & (offs_n[
        None, :] < chunk_size))


def _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=None, dt_softplus=False,
    dt_limit=(0.0, float('inf'))):
    batch, seqlen, nheads = dt.shape
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
    nchunks = math.ceil(seqlen / chunk_size)
    dt_out = torch.empty(batch, nheads, nchunks, chunk_size, device=dt.
        device, dtype=torch.float32)
    dA_cumsum = torch.empty(batch, nheads, nchunks, chunk_size, device=dt.
        device, dtype=torch.float32)
    grid_chunk_cs = lambda META: (batch, nchunks, triton.cdiv(nheads, META[
        'BLOCK_SIZE_H']))
    with torch.cuda.device(dt.device.index):
        _chunk_cumsum_fwd_kernel[grid_chunk_cs](dt, A, dt_bias, dt_out,
            dA_cumsum, int(batch), int(seqlen), int(nheads), int(chunk_size
            ), dt_limit[0], dt_limit[1], dt.stride(0), dt.stride(1), dt.
            stride(2), A.stride(0), dt_bias.stride(0) if dt_bias is not
            None else 0, dt_out.stride(0), dt_out.stride(2), dt_out.stride(
            1), dt_out.stride(3), dA_cumsum.stride(0), dA_cumsum.stride(2),
            dA_cumsum.stride(1), dA_cumsum.stride(3), dt_softplus,
            HAS_DT_BIAS=dt_bias is not None, BLOCK_SIZE_CHUNK=triton.
            next_power_of_2(chunk_size))
    return dA_cumsum, dt_out


@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_H': 1}), triton.Config
    ({'BLOCK_SIZE_H': 2}), triton.Config({'BLOCK_SIZE_H': 4}), triton.
    Config({'BLOCK_SIZE_H': 8}), triton.Config({'BLOCK_SIZE_H': 16}),
    triton.Config({'BLOCK_SIZE_H': 32}), triton.Config({'BLOCK_SIZE_H': 64}
    )], key=['chunk_size', 'nheads'])
@triton.jit
def _chunk_cumsum_fwd_kernel(dt_ptr, A_ptr, dt_bias_ptr, dt_out_ptr,
    dA_cumsum_ptr, batch, seqlen, nheads, chunk_size, dt_min, dt_max,
    stride_dt_batch, stride_dt_seqlen, stride_dt_head, stride_A_head,
    stride_dt_bias_head, stride_dt_out_batch, stride_dt_out_chunk,
    stride_dt_out_head, stride_dt_out_csize, stride_dA_cs_batch,
    stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize, DT_SOFTPLUS:
    tl.constexpr, HAS_DT_BIAS: tl.constexpr, BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_CHUNK: tl.constexpr):
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    dt_ptr += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    dt_out_ptr += pid_b * stride_dt_out_batch + pid_c * stride_dt_out_chunk
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    dt_ptrs = dt_ptr + (offs_h[:, None] * stride_dt_head + offs_c[None, :] *
        stride_dt_seqlen)
    A_ptrs = A_ptr + offs_h * stride_A_head
    dt_out_ptrs = dt_out_ptr + (offs_h[:, None] * stride_dt_out_head + 
        offs_c[None, :] * stride_dt_out_csize)
    dA_cs_ptrs = dA_cumsum_ptr + (offs_h[:, None] * stride_dA_cs_head + 
        offs_c[None, :] * stride_dA_cs_csize)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    dt = tl.load(dt_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :
        ] < chunk_size_limit), other=0.0).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + offs_h * stride_dt_bias_head, mask=
            offs_h < nheads, other=0.0).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    dt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] <
        chunk_size_limit), dt, 0.0)
    tl.store(dt_out_ptrs, dt, mask=(offs_h[:, None] < nheads) & (offs_c[
        None, :] < chunk_size))
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    dA = dt * A[:, None]
    dA_cs = tl.cumsum(dA, axis=1)
    tl.store(dA_cs_ptrs, dA_cs, mask=(offs_h[:, None] < nheads) & (offs_c[
        None, :] < chunk_size))


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


def _state_passing_fwd(states, dA_chunk_cumsum, initial_states=None,
    seq_idx=None, chunk_size=None, out_dtype=None):
    batch, nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, dim)
    if seq_idx is not None:
        assert chunk_size is not None
        seqlen = seq_idx.shape[-1]
        assert seq_idx.shape == (batch, seqlen)
    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.empty((batch, nchunks, nheads, dim), device=states.device,
        dtype=out_dtype)
    final_states = torch.empty((batch, nheads, dim), device=states.device,
        dtype=torch.float32)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE']), batch, nheads)
    with torch.cuda.device(states.device.index):
        _state_passing_fwd_kernel[grid](states, out, final_states,
            dA_chunk_cumsum, initial_states, seq_idx, int(dim), int(nchunks
            ), int(seqlen if seq_idx is not None else 0), int(chunk_size if
            seq_idx is not None else 0), states.stride(0), states.stride(1),
            states.stride(2), states.stride(3), out.stride(0), out.stride(1
            ), out.stride(2), out.stride(3), final_states.stride(0),
            final_states.stride(1), final_states.stride(2), dA_chunk_cumsum
            .stride(0), dA_chunk_cumsum.stride(2), dA_chunk_cumsum.stride(1
            ), *((initial_states.stride(0), initial_states.stride(1),
            initial_states.stride(2)) if initial_states is not None else (0,
            0, 0)), *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not
            None else (0, 0)), HAS_INITSTATES=initial_states is not None,
            HAS_SEQ_IDX=seq_idx is not None)
    return out, final_states


@triton.autotune(configs=[triton.Config({'BLOCK_SIZE': 64}), triton.Config(
    {'BLOCK_SIZE': 128}), triton.Config({'BLOCK_SIZE': 256}), triton.Config
    ({'BLOCK_SIZE': 512}), triton.Config({'BLOCK_SIZE': 1024}), triton.
    Config({'BLOCK_SIZE': 2048})], key=['dim'])
@triton.jit
def _state_passing_fwd_kernel(states_ptr, out_ptr, final_states_ptr,
    dA_cs_ptr, initstates_ptr, seq_idx_ptr, dim, nchunks, seqlen,
    chunk_size, stride_states_batch, stride_states_chunk,
    stride_states_head, stride_states_dim, stride_out_batch,
    stride_out_chunk, stride_out_head, stride_out_dim,
    stride_final_states_batch, stride_final_states_head,
    stride_final_states_dim, stride_dA_cs_batch, stride_dA_cs_chunk,
    stride_dA_cs_head, stride_initstates_batch, stride_initstates_head,
    stride_initstates_dim, stride_seq_idx_batch, stride_seq_idx_seqlen,
    HAS_INITSTATES: tl.constexpr, HAS_SEQ_IDX: tl.constexpr, BLOCK_SIZE: tl
    .constexpr):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    dA_cs_ptr += pid_b * stride_dA_cs_batch + pid_h * stride_dA_cs_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head
    final_states_ptr += (pid_b * stride_final_states_batch + pid_h *
        stride_final_states_head)
    if HAS_INITSTATES:
        initstates_ptr += (pid_b * stride_initstates_batch + pid_h *
            stride_initstates_head)
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_m * stride_states_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim
    final_states_ptrs = final_states_ptr + offs_m * stride_final_states_dim
    if not HAS_INITSTATES:
        states = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    else:
        initstates_ptrs = initstates_ptr + offs_m * stride_initstates_dim
        states = tl.load(initstates_ptrs, mask=offs_m < dim, other=0.0).to(tl
            .float32)
    tl.store(out_ptrs, states, mask=offs_m < dim)
    out_ptrs += stride_out_chunk
    seq_idx = 0
    for c in range(nchunks):
        new_states = tl.load(states_ptrs, mask=offs_m < dim, other=0.0).to(tl
            .float32)
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        scale = tl.exp(dA_cs)
        if HAS_SEQ_IDX:
            seq_idx_new = tl.load(seq_idx_ptr + (min((c + 1) * chunk_size,
                seqlen) - 1) * stride_seq_idx_seqlen)
            scale = tl.where(seq_idx_new == seq_idx, scale, 0.0)
            seq_idx = seq_idx_new
        states = scale * states + new_states
        if c < nchunks - 1:
            tl.store(out_ptrs, states, mask=offs_m < dim)
        else:
            tl.store(final_states_ptrs, states, mask=offs_m < dim)
        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk
        out_ptrs += stride_out_chunk


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
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K':
    64}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K':
    32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 
    32}, num_stages=5, num_warps=2), triton.Config({'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2)],
    key=['chunk_size', 'hdim', 'dstate', 'IS_CAUSAL'])
@triton.jit
def _chunk_scan_fwd_kernel(cb_ptr, x_ptr, z_ptr, out_ptr, out_x_ptr, dt_ptr,
    dA_cumsum_ptr, seq_idx_ptr, C_ptr, prev_states_ptr, D_ptr, chunk_size,
    hdim, dstate, batch, seqlen, nheads_ngroups_ratio, stride_cb_batch,
    stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_k,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_z_batch, stride_z_seqlen, stride_z_head, stride_z_hdim,
    stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head,
    stride_dA_cs_csize, stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_C_batch, stride_C_seqlen, stride_C_head, stride_C_dstate,
    stride_states_batch, stride_states_chunk, stride_states_head,
    stride_states_hdim, stride_states_dstate, stride_D_head, IS_CAUSAL: tl.
    constexpr, HAS_D: tl.constexpr, D_HAS_HDIM: tl.constexpr, HAS_Z: tl.
    constexpr, HAS_SEQ_IDX: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr, IS_TRITON_22: tl.constexpr):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    cb_ptr += (pid_b * stride_cb_batch + pid_c * stride_cb_chunk + pid_h //
        nheads_ngroups_ratio * stride_cb_head)
    x_ptr += (pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen +
        pid_h * stride_x_head)
    dt_ptr += (pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h *
        stride_dt_head)
    dA_cumsum_ptr += (pid_b * stride_dA_cs_batch + pid_c *
        stride_dA_cs_chunk + pid_h * stride_dA_cs_head)
    C_ptr += (pid_b * stride_C_batch + pid_c * chunk_size * stride_C_seqlen +
        pid_h // nheads_ngroups_ratio * stride_C_head)
    prev_states_ptr += (pid_b * stride_states_batch + pid_c *
        stride_states_chunk + pid_h * stride_states_head)
    if HAS_SEQ_IDX:
        seq_idx_ptr += (pid_b * stride_seq_idx_batch + pid_c * chunk_size *
            stride_seq_idx_seqlen)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=
        offs_m < chunk_size, other=0.0).to(tl.float32)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    if HAS_SEQ_IDX:
        seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen, mask=
            pid_c >= 1, other=0)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen,
            mask=offs_m < chunk_size_limit, other=-1)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if IS_TRITON_22 or pid_c > -1:
        offs_k_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <=
            128 else BLOCK_SIZE_K)
        C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_k_dstate
            [None, :] * stride_C_dstate)
        prev_states_ptrs = prev_states_ptr + (offs_n[None, :] *
            stride_states_hdim + offs_k_dstate[:, None] * stride_states_dstate)
        if not HAS_SEQ_IDX:
            scale_m = tl.exp(dA_cs_m)
        else:
            scale_m = tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)
        if BLOCK_SIZE_DSTATE <= 128:
            C = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) &
                (offs_k_dstate[None, :] < dstate), other=0.0)
            prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:,
                None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
            prev_states = prev_states.to(C_ptr.dtype.element_ty)
            acc = tl.dot(C, prev_states) * scale_m[:, None]
        else:
            for k in range(0, dstate, BLOCK_SIZE_K):
                C = tl.load(C_ptrs, mask=(offs_m[:, None] <
                    chunk_size_limit) & (offs_k_dstate[None, :] < dstate -
                    k), other=0.0)
                prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate
                    [:, None] < dstate - k) & (offs_n[None, :] < hdim),
                    other=0.0)
                prev_states = prev_states.to(C_ptr.dtype.element_ty)
                acc += tl.dot(C, prev_states)
                C_ptrs += BLOCK_SIZE_K
                prev_states_ptrs += BLOCK_SIZE_K
            acc *= scale_m[:, None]
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None,
        :] * stride_cb_csize_k)
    x_ptrs = x_ptr + (offs_k[:, None] * stride_x_seqlen + offs_n[None, :] *
        stride_x_hdim)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    K_MAX = chunk_size_limit if not IS_CAUSAL else min((pid_m + 1) *
        BLOCK_SIZE_M, chunk_size_limit)
    for k in range(0, K_MAX, BLOCK_SIZE_K):
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_k
            [None, :] < chunk_size - k), other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size - k,
            other=0.0).to(tl.float32)
        cb *= tl.exp(dA_cs_m[:, None] - dA_cs_k[None, :])
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl
            .float32)
        cb *= dt_k
        if IS_CAUSAL:
            mask = offs_m[:, None] >= k + offs_k[None, :]
            cb = tl.where(mask, cb, 0.0)
        cb = cb.to(x_ptr.dtype.element_ty)
        x = tl.load(x_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) &
            (offs_n[None, :] < hdim), other=0.0)
        acc += tl.dot(cb, x)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_D:
        if D_HAS_HDIM:
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n <
                hdim, other=0.0).to(tl.float32)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        x_residual = tl.load(x_ptr + (offs_m[:, None] * stride_x_seqlen + 
            offs_n[None, :] * stride_x_hdim), mask=(offs_m[:, None] <
            chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.
            float32)
        acc += x_residual * D
    if HAS_Z:
        out_x_ptr += (pid_b * stride_out_batch + pid_c * chunk_size *
            stride_out_seqlen + pid_h * stride_out_head)
        out_x_ptrs = out_x_ptr + (stride_out_seqlen * offs_out_m[:, None] +
            offs_out_n[None, :])
        tl.store(out_x_ptrs, acc, mask=(offs_out_m[:, None] <
            chunk_size_limit) & (offs_out_n[None, :] < hdim))
        z_ptr += (pid_b * stride_z_batch + pid_c * chunk_size *
            stride_z_seqlen + pid_h * stride_z_head)
        z_ptrs = z_ptr + (stride_z_seqlen * offs_out_m[:, None] + 
            stride_z_hdim * offs_out_n[None, :])
        z = tl.load(z_ptrs, mask=(offs_out_m[:, None] < chunk_size_limit) &
            (offs_out_n[None, :] < hdim), other=0.0).to(tl.float32)
        acc *= z * tl.sigmoid(z)
    out_ptr += (pid_b * stride_out_batch + pid_c * chunk_size *
        stride_out_seqlen + pid_h * stride_out_head)
    out_ptrs = out_ptr + (stride_out_seqlen * offs_out_m[:, None] + 
        offs_out_n[None, :] * stride_out_hdim)
    tl.store(out_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) &
        (offs_out_n[None, :] < hdim))


def _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, D=None, z=None,
    seq_idx=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (batch, seqlen, ngroups, dstate)
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    out = torch.empty(batch, seqlen, nheads, headdim, device=x.device,
        dtype=x.dtype)
    if z is not None:
        out_x = torch.empty(batch, seqlen, nheads, headdim, device=x.device,
            dtype=x.dtype)
        assert out_x.stride() == out.stride()
    else:
        out_x = None
    grid = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) *
        triton.cdiv(headdim, META['BLOCK_SIZE_N']), batch * nchunks, nheads)
    z_strides = (z.stride(0), z.stride(1), z.stride(2), z.stride(3)
        ) if z is not None else (0, 0, 0, 0)
    _chunk_scan_fwd_kernel[grid](cb, x, z, out, out_x, dt, dA_cumsum,
        seq_idx, C, states, D, int(chunk_size), int(headdim), int(dstate),
        int(batch), int(seqlen), int(nheads // ngroups), cb.stride(0), cb.
        stride(1), cb.stride(2), cb.stride(3), cb.stride(4), x.stride(0), x
        .stride(1), x.stride(2), x.stride(3), z_strides[0], z_strides[1],
        z_strides[2], z_strides[3], out.stride(0), out.stride(1), out.
        stride(2), out.stride(3), dt.stride(0), dt.stride(2), dt.stride(1),
        dt.stride(3), dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.
        stride(1), dA_cumsum.stride(3), *((seq_idx.stride(0), seq_idx.
        stride(1)) if seq_idx is not None else (0, 0)), C.stride(0), C.
        stride(1), C.stride(2), C.stride(3), states.stride(0), states.
        stride(1), states.stride(2), states.stride(3), states.stride(4), D.
        stride(0) if D is not None else 0, True, D is not None, D.dim() == 
        2 if D is not None else True, BLOCK_SIZE_DSTATE=max(triton.
        next_power_of_2(int(dstate)), 16), HAS_Z=z is not None, HAS_SEQ_IDX
        =seq_idx is not None, IS_TRITON_22=TRITON_22)
    return out, out_x


def _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D=None, z=
    None, dt_bias=None, initial_states=None, seq_idx=None, dt_softplus=
    False, dt_limit=(0.0, float('inf'))):
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads,)
    assert C.shape == B.shape
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if x.stride(-1) != 1 and x.stride(1) != 1:
        x = x.contiguous()
    if z is not None and z.stride(-1) != 1 and z.stride(1) != 1:
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)
    dA_cumsum, dt = _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=dt_bias,
        dt_softplus=dt_softplus, dt_limit=dt_limit)
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx,
        states_in_fp32=True)
    states, final_states = _state_passing_fwd(rearrange(states,
        '... p n -> ... (p n)'), dA_cumsum[:, :, :, -1], initial_states=
        rearrange(initial_states, '... p n -> ... (p n)') if initial_states
         is not None else None, seq_idx=seq_idx, chunk_size=chunk_size,
        out_dtype=C.dtype)
    states, final_states = [rearrange(t, '... (p n) -> ... p n', n=dstate) for
        t in [states, final_states]]
    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=
        torch.float32)
    out, out_x = _chunk_scan_fwd(CB, x, dt, dA_cumsum, C, states, D=D, z=z,
        seq_idx=seq_idx)
    return out, out_x, dt, dA_cumsum, states, final_states


# Forward method (kernel launch code)
def _MambaChunkScanCombinedFn_forward(ctx, x, dt, A, B, C, chunk_size, D=
    None, z=None, dt_bias=None, initial_states=None, seq_idx=None,
    dt_softplus=False, dt_limit=(0.0, float('inf')), return_final_states=False
    ):
    ctx.dt_dtype = dt.dtype
    out, out_x, dt_out, dA_cumsum, states, final_states = (
        _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D=D, z=z,
        dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx,
        dt_softplus=dt_softplus, dt_limit=dt_limit))
    ctx.save_for_backward(out if z is None else out_x, x, dt, dA_cumsum, A,
        B, C, D, z, dt_bias, initial_states, seq_idx)
    ctx.dt_softplus = dt_softplus
    ctx.chunk_size = chunk_size
    ctx.dt_limit = dt_limit
    ctx.return_final_states = return_final_states
    return out if not return_final_states else (out, final_states)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_M': 128,
    'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_CS': 64}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_CS':
    32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128,
    'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_CS':
    32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_CS':
    32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_CS': 32}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_CS':
    32}, num_stages=5, num_warps=2), triton.Config({'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=2)],
    key=['chunk_size', 'K'])
@triton.jit
def _bmm_chunk_bwd_kernel(a_ptr, dout_ptr, db_ptr, res_ptr, seqlen,
    chunk_size, K, ngroups, stride_a_batch, stride_a_seqlen, stride_a_head,
    stride_ak, stride_dout_batch, stride_dout_chunk, stride_dout_head,
    stride_dout_csize_m, stride_dout_csize_n, stride_db_batch,
    stride_db_seqlen, stride_db_head, stride_db_k, stride_res_batch,
    stride_res_seqlen, stride_res_head, stride_res_k, dot_dtype: tl.
    constexpr, HAS_RESIDUAL: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_CS: tl.constexpr):
    pid_b = tl.program_id(axis=1)
    pid_ch = tl.program_id(axis=2)
    pid_c = pid_ch // ngroups
    pid_h = pid_ch - pid_c * ngroups
    num_pid_n = tl.cdiv(K, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    a_ptr += (pid_b * stride_a_batch + pid_c * chunk_size * stride_a_seqlen +
        pid_h * stride_a_head)
    dout_ptr += (pid_b * stride_dout_batch + pid_c * stride_dout_chunk + 
        pid_h * stride_dout_head)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cs = tl.arange(0, BLOCK_SIZE_CS)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_csize_n + offs_cs
        [None, :] * stride_dout_csize_m)
    a_ptrs = a_ptr + (offs_cs[:, None] * stride_a_seqlen + offs_n[None, :] *
        stride_ak)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for cs in range(0, tl.cdiv(chunk_size_limit, BLOCK_SIZE_CS)):
        dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size) & (
            offs_cs[None, :] < chunk_size_limit - cs * BLOCK_SIZE_CS),
            other=0.0).to(dot_dtype)
        a = tl.load(a_ptrs, mask=(offs_cs[:, None] < chunk_size_limit - cs *
            BLOCK_SIZE_CS) & (offs_n[None, :] < K), other=0.0).to(dot_dtype)
        acc += tl.dot(dout, a)
        dout_ptrs += BLOCK_SIZE_CS * stride_dout_csize_m
        a_ptrs += BLOCK_SIZE_CS * stride_a_seqlen
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_RESIDUAL:
        res_ptr += (pid_b * stride_res_batch + pid_c * chunk_size *
            stride_res_seqlen + pid_h * stride_res_head)
        res_ptrs = res_ptr + (offs_m[:, None] * stride_res_seqlen + offs_n[
            None, :] * stride_res_k)
        res = tl.load(res_ptrs, mask=(offs_m[:, None] < chunk_size_limit) &
            (offs_n[None, :] < K)).to(tl.float32)
        acc += res
    db = acc.to(db_ptr.dtype.element_ty)
    db_ptr += (pid_b * stride_db_batch + pid_c * chunk_size *
        stride_db_seqlen + pid_h * stride_db_head)
    db_ptrs = db_ptr + (offs_m[:, None] * stride_db_seqlen + offs_n[None, :
        ] * stride_db_k)
    tl.store(db_ptrs, db, mask=(offs_m[:, None] < chunk_size_limit) & (
        offs_n[None, :] < K))


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
def _chunk_scan_bwd_dc_kernel(dout_ptr, prev_states_ptr, C_ptr,
    dA_cumsum_ptr, seq_idx_ptr, dc_ptr, ddA_cumsum_ptr, chunk_size, dstate,
    hdim, batch, seqlen, nheads, nheads_per_program, ngroups,
    stride_dout_batch, stride_dout_seqlen, stride_dout_head,
    stride_dout_hdim, stride_prev_states_batch, stride_prev_states_chunk,
    stride_prev_states_head, stride_prev_states_hdim,
    stride_prev_states_dstate, stride_C_batch, stride_C_seqlen,
    stride_C_head, stride_C_dstate, stride_dA_cs_batch, stride_dA_cs_chunk,
    stride_dA_cs_head, stride_dA_cs_csize, stride_seq_idx_batch,
    stride_seq_idx_seqlen, stride_dc_batch, stride_dc_seqlen,
    stride_dc_split, stride_dc_group, stride_dc_dstate, stride_ddA_cs_batch,
    stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_csize,
    HAS_DDA_CS: tl.constexpr, HAS_SEQ_IDX: tl.constexpr, BLOCK_SIZE_M: tl.
    constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_sg = tl.program_id(axis=2)
    pid_s = pid_sg // ngroups
    pid_g = pid_sg - pid_s * ngroups
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    dout_ptr += (pid_b * stride_dout_batch + pid_c * chunk_size *
        stride_dout_seqlen + (pid_g * (nheads // ngroups) + pid_s *
        nheads_per_program) * stride_dout_head)
    dc_ptr += (pid_b * stride_dc_batch + pid_c * chunk_size *
        stride_dc_seqlen + pid_g * stride_dc_group + pid_s * stride_dc_split)
    prev_states_ptr += (pid_b * stride_prev_states_batch + pid_c *
        stride_prev_states_chunk + (pid_g * (nheads // ngroups) + pid_s *
        nheads_per_program) * stride_prev_states_head)
    dA_cumsum_ptr += (pid_b * stride_dA_cs_batch + pid_c *
        stride_dA_cs_chunk + (pid_g * (nheads // ngroups) + pid_s *
        nheads_per_program) * stride_dA_cs_head)
    if HAS_DDA_CS:
        C_ptr += (pid_b * stride_C_batch + pid_c * chunk_size *
            stride_C_seqlen + pid_g * stride_C_head)
        ddA_cumsum_ptr += (pid_b * stride_ddA_cs_batch + pid_c *
            stride_ddA_cs_chunk + (pid_g * (nheads // ngroups) + pid_s *
            nheads_per_program) * stride_ddA_cs_head)
    if HAS_SEQ_IDX:
        seq_idx_ptr += (pid_b * stride_seq_idx_batch + pid_c * chunk_size *
            stride_seq_idx_seqlen)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_k[
        None, :] * stride_dout_hdim)
    prev_states_ptrs = prev_states_ptr + (offs_n[None, :] *
        stride_prev_states_dstate + offs_k[:, None] * stride_prev_states_hdim)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_m * stride_dA_cs_csize
    if HAS_DDA_CS:
        C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_n[None,
            :] * stride_C_dstate)
        ddA_cumsum_ptrs = ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if HAS_DDA_CS:
        c = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (
            offs_n[None, :] < dstate), other=0.0).to(tl.float32)
    if HAS_SEQ_IDX:
        seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen, mask=
            pid_c >= 1, other=0)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen,
            mask=offs_m < chunk_size_limit, other=-1)
    nheads_iter = min(nheads_per_program, nheads // ngroups - pid_s *
        nheads_per_program)
    for h in range(nheads_iter):
        dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) &
            (offs_k[None, :] < hdim), other=0.0)
        prev_states = tl.load(prev_states_ptrs, mask=(offs_k[:, None] <
            hdim) & (offs_n[None, :] < dstate), other=0.0)
        prev_states = prev_states.to(dout_ptrs.dtype.element_ty)
        dc = tl.dot(dout, prev_states)
        dA_cs_m = tl.load(dA_cumsum_ptrs, mask=offs_m < chunk_size_limit,
            other=0.0).to(tl.float32)
        if not HAS_SEQ_IDX:
            scale = tl.exp(dA_cs_m)
        else:
            scale = tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)
        dc *= scale[:, None]
        if HAS_DDA_CS:
            ddA_cs = tl.sum(dc * c, axis=1)
            tl.atomic_add(ddA_cumsum_ptrs, ddA_cs, mask=offs_m < chunk_size)
        acc += dc
        dout_ptrs += stride_dout_head
        prev_states_ptrs += stride_prev_states_head
        dA_cumsum_ptrs += stride_dA_cs_head
        if HAS_DDA_CS:
            ddA_cumsum_ptrs += stride_ddA_cs_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dc_ptrs = dc_ptr + (offs_m[:, None] * stride_dc_seqlen + offs_n[None, :
        ] * stride_dc_dstate)
    tl.store(dc_ptrs, acc, mask=(offs_m[:, None] < chunk_size_limit) & (
        offs_n[None, :] < dstate))


@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N':
    128}, num_stages=3, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128,
    'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4), triton.Config({
    'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=3,
    num_warps=4), triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64},
    num_stages=3, num_warps=4), triton.Config({'BLOCK_SIZE_M': 32,
    'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4)], key=['chunk_size',
    'hdim'])
@triton.jit
def _chunk_scan_bwd_dcb_kernel(x_ptr, dout_ptr, cb_ptr, dt_ptr,
    dA_cumsum_ptr, seq_idx_ptr, dcb_ptr, ddA_cumsum_ptr, chunk_size, hdim,
    batch, seqlen, nheads, nheads_per_program, ngroups, stride_x_batch,
    stride_x_seqlen, stride_x_head, stride_x_hdim, stride_dout_batch,
    stride_dout_seqlen, stride_dout_head, stride_dout_hdim, stride_cb_batch,
    stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_n,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head,
    stride_dA_cs_csize, stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_dcb_batch, stride_dcb_chunk, stride_dcb_split, stride_dcb_group,
    stride_dcb_csize_m, stride_dcb_csize_n, stride_ddA_cs_batch,
    stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_csize_m,
    stride_ddA_cs_csize_n, HAS_DDA_CS: tl.constexpr, HAS_SEQ_IDX: tl.
    constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_sg = tl.program_id(axis=2)
    pid_s = pid_sg // ngroups
    pid_g = pid_sg - pid_s * ngroups
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + (
        pid_g * (nheads // ngroups) + pid_s * nheads_per_program
        ) * stride_x_head
    dout_ptr += (pid_b * stride_dout_batch + pid_c * chunk_size *
        stride_dout_seqlen + (pid_g * (nheads // ngroups) + pid_s *
        nheads_per_program) * stride_dout_head)
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + (pid_g *
        (nheads // ngroups) + pid_s * nheads_per_program) * stride_dt_head
    dA_cumsum_ptr += (pid_b * stride_dA_cs_batch + pid_c *
        stride_dA_cs_chunk + (pid_g * (nheads // ngroups) + pid_s *
        nheads_per_program) * stride_dA_cs_head)
    if HAS_DDA_CS:
        cb_ptr += (pid_b * stride_cb_batch + pid_c * stride_cb_chunk + 
            pid_g * stride_cb_head)
        ddA_cumsum_ptr += (pid_b * stride_ddA_cs_batch + pid_c *
            stride_ddA_cs_chunk + (pid_g * (nheads // ngroups) + pid_s *
            nheads_per_program) * stride_ddA_cs_head + pid_m *
            stride_ddA_cs_csize_m)
    if HAS_SEQ_IDX:
        seq_idx_ptr += (pid_b * stride_seq_idx_batch + pid_c * chunk_size *
            stride_seq_idx_seqlen)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_k[
        None, :] * stride_dout_hdim)
    x_ptrs = x_ptr + (offs_n[None, :] * stride_x_seqlen + offs_k[:, None] *
        stride_x_hdim)
    dt_ptrs = dt_ptr + offs_n * stride_dt_csize
    if HAS_DDA_CS:
        cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_n[
            None, :] * stride_cb_csize_n)
        ddA_cumsum_ptrs = ddA_cumsum_ptr + offs_n * stride_ddA_cs_csize_n
    if pid_n * BLOCK_SIZE_N >= (pid_m + 1) * BLOCK_SIZE_M:
        dcb_ptr += (pid_b * stride_dcb_batch + pid_c * stride_dcb_chunk + 
            pid_g * stride_dcb_group + pid_s * stride_dcb_split)
        dcb_ptrs = dcb_ptr + (offs_m[:, None] * stride_dcb_csize_m + offs_n
            [None, :] * stride_dcb_csize_n)
        tl.store(dcb_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=
            dcb_ptr.dtype.element_ty), mask=(offs_m[:, None] < chunk_size) &
            (offs_n[None, :] < chunk_size))
        return
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    chunk_size_limit_n = min(chunk_size_limit, (pid_m + 1) * BLOCK_SIZE_M)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if HAS_DDA_CS:
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_n
            [None, :] < chunk_size), other=0.0).to(tl.float32)
    nheads_iter = min(nheads_per_program, nheads // ngroups - pid_s *
        nheads_per_program)
    for h in range(nheads_iter):
        dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) &
            (offs_k[None, :] < hdim), other=0.0)
        x = tl.load(x_ptrs, mask=(offs_k[:, None] < hdim) & (offs_n[None, :
            ] < chunk_size_limit_n), other=0.0)
        dcb = tl.dot(dout, x)
        dt_n = tl.load(dt_ptrs, mask=offs_n < chunk_size, other=0.0).to(tl.
            float32)
        dcb *= dt_n
        dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask
            =offs_m < chunk_size_limit, other=0.0).to(tl.float32)
        dA_cs_n = tl.load(dA_cumsum_ptr + offs_n * stride_dA_cs_csize, mask
            =offs_n < chunk_size_limit, other=0.0).to(tl.float32)
        dcb *= tl.exp(dA_cs_m[:, None] - dA_cs_n[None, :])
        if HAS_DDA_CS:
            tl.static_assert(not HAS_SEQ_IDX,
                'HAS_SEQ_IDX not supported with HAS_DDA_CS yet')
            ddA_cs = dcb * cb
            mask = offs_m[:, None] >= offs_n[None, :] + 1
            ddA_cs = tl.where(mask, ddA_cs, 0.0)
            ddA_cs = tl.cumsum(ddA_cs, axis=1)
            ddA_cs = tl.where(mask, ddA_cs, 0.0)
            ddA_cs = tl.sum(ddA_cs, axis=0)
            tl.store(ddA_cumsum_ptrs + stride_ddA_cs_csize_n, ddA_cs, mask=
                offs_n < chunk_size - 1)
            tl.store(ddA_cumsum_ptr, 0.0)
        acc += dcb
        dout_ptrs += stride_dout_head
        x_ptrs += stride_x_head
        dt_ptrs += stride_dt_head
        dA_cumsum_ptr += stride_dA_cs_head
        if HAS_DDA_CS:
            ddA_cumsum_ptr += stride_ddA_cs_head
            ddA_cumsum_ptrs += stride_ddA_cs_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_SEQ_IDX:
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen,
            mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_n = tl.load(seq_idx_ptr + offs_n * stride_seq_idx_seqlen,
            mask=offs_n < chunk_size_limit, other=-2)
        acc = tl.where(seq_idx_m[:, None] == seq_idx_n[None, :], acc, 0.0)
    mask = offs_m[:, None] >= offs_n[None, :]
    acc = tl.where(mask, acc, 0.0)
    dcb_ptr += (pid_b * stride_dcb_batch + pid_c * stride_dcb_chunk + pid_g *
        stride_dcb_group + pid_s * stride_dcb_split)
    dcb_ptrs = dcb_ptr + (offs_m[:, None] * stride_dcb_csize_m + offs_n[
        None, :] * stride_dcb_csize_n)
    tl.store(dcb_ptrs, acc, mask=(offs_m[:, None] < chunk_size) & (offs_n[
        None, :] < chunk_size))


@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N':
    32}, num_stages=3, num_warps=4), triton.Config({'BLOCK_SIZE_M': 32,
    'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4), triton.Config({
    'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=3,
    num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64},
    num_stages=3, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4), triton.Config({
    'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=3,
    num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128},
    num_stages=3, num_warps=4)], key=['chunk_size', 'hdim'])
@triton.jit
def _chunk_scan_bwd_ddAcs_stable_kernel(x_ptr, dout_ptr, dt_ptr,
    dA_cumsum_ptr, cb_ptr, ddA_cumsum_ptr, chunk_size, hdim, batch, seqlen,
    nheads_ngroups_ratio, stride_x_batch, stride_x_seqlen, stride_x_head,
    stride_x_hdim, stride_dout_batch, stride_dout_seqlen, stride_dout_head,
    stride_dout_hdim, stride_dt_batch, stride_dt_chunk, stride_dt_head,
    stride_dt_csize, stride_dA_cs_batch, stride_dA_cs_chunk,
    stride_dA_cs_head, stride_dA_cs_csize, stride_cb_batch, stride_cb_chunk,
    stride_cb_head, stride_cb_csize_m, stride_cb_csize_n,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head,
    stride_ddA_cs_csize_m, stride_ddA_cs_csize_n, BLOCK_SIZE_M: tl.
    constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    x_ptr += (pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen +
        pid_h * stride_x_head)
    dout_ptr += (pid_b * stride_dout_batch + pid_c * chunk_size *
        stride_dout_seqlen + pid_h * stride_dout_head)
    dt_ptr += (pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h *
        stride_dt_head)
    dA_cumsum_ptr += (pid_b * stride_dA_cs_batch + pid_c *
        stride_dA_cs_chunk + pid_h * stride_dA_cs_head)
    cb_ptr += (pid_b * stride_cb_batch + pid_c * stride_cb_chunk + pid_h //
        nheads_ngroups_ratio * stride_cb_head)
    ddA_cumsum_ptr += (pid_b * stride_ddA_cs_batch + pid_c *
        stride_ddA_cs_chunk + pid_h * stride_ddA_cs_head + pid_m *
        stride_ddA_cs_csize_m)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_k[
        None, :] * stride_dout_hdim)
    x_ptrs = x_ptr + (offs_n[None, :] * stride_x_seqlen + offs_k[:, None] *
        stride_x_hdim)
    dt_ptrs = dt_ptr + offs_n * stride_dt_csize
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_n[None,
        :] * stride_cb_csize_n)
    ddAcs_ptrs = ddA_cumsum_ptr + offs_n * stride_ddA_cs_csize_n
    tl.store(ddA_cumsum_ptr, 0.0)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    rowsum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (
        offs_k[None, :] < hdim), other=0.0)
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=
        offs_m < chunk_size, other=0.0).to(tl.float32)
    lo, hi = 0, (pid_m + 1) * BLOCK_SIZE_M
    for start_n in range(lo, hi, BLOCK_SIZE_N):
        start_n = tl.multiple_of(start_n, BLOCK_SIZE_N)
        x = tl.load(x_ptrs, mask=(offs_k[:, None] < hdim) & (offs_n[None, :
            ] < chunk_size_limit - start_n), other=0.0)
        acc = tl.dot(dout, x)
        dt_n = tl.load(dt_ptrs, mask=offs_n < chunk_size - start_n, other=0.0
            ).to(tl.float32)
        acc *= dt_n
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_n
            [None, :] < chunk_size - start_n), other=0.0).to(tl.float32)
        acc *= cb
        dA_cs_n = tl.load(dA_cumsum_ptr + start_n + offs_n *
            stride_dA_cs_csize, mask=offs_n < chunk_size - start_n, other=0.0
            ).to(tl.float32)
        acc *= tl.exp(dA_cs_m[:, None] - dA_cs_n[None, :])
        mask = offs_m[:, None] >= start_n + offs_n[None, :] + 1
        acc = tl.where(mask, acc, 0.0)
        rowsum_new = rowsum + tl.sum(acc, axis=1)
        acc = rowsum[:, None] + tl.cumsum(acc, axis=1)
        rowsum = rowsum_new
        acc = tl.where(mask, acc, 0.0)
        ddA_cs = tl.sum(acc, axis=0)
        tl.store(ddAcs_ptrs + stride_ddA_cs_csize_n, ddA_cs, mask=offs_n < 
            chunk_size - start_n - 1)
        x_ptrs += BLOCK_SIZE_N * stride_x_seqlen
        dt_ptrs += BLOCK_SIZE_N * stride_dt_csize
        cb_ptrs += BLOCK_SIZE_N * stride_cb_csize_n
        ddAcs_ptrs += BLOCK_SIZE_N * stride_ddA_cs_csize_n
    for start_n in range(hi, chunk_size, BLOCK_SIZE_N):
        tl.store(ddAcs_ptrs + stride_ddA_cs_csize_n, tl.zeros((BLOCK_SIZE_N
            ,), dtype=tl.float32), mask=offs_n < chunk_size - start_n - 1)
        ddAcs_ptrs += BLOCK_SIZE_N * stride_ddA_cs_csize_n


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
def _chunk_scan_bwd_dstates_kernel(dout_ptr, c_ptr, dprev_states_ptr,
    dA_cumsum_ptr, seq_idx_ptr, hdim, dstate, chunk_size, batch, seqlen,
    nchunks, nheads_ngroups_ratio, stride_dout_batch, stride_dout_seqlen,
    stride_dout_head, stride_dout_hdim, stride_c_batch, stride_c_seqlen,
    stride_c_head, stride_c_dstate, stride_dprev_states_batch,
    stride_dprev_states_chunk, stride_dprev_states_head,
    stride_dprev_states_hdim, stride_dprev_states_dstate,
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
    c_ptr += (pid_b * stride_c_batch + pid_c * chunk_size * stride_c_seqlen +
        pid_h // nheads_ngroups_ratio * stride_c_head)
    dout_ptr += (pid_b * stride_dout_batch + pid_c * chunk_size *
        stride_dout_seqlen + pid_h * stride_dout_head)
    dA_cumsum_ptr += (pid_b * stride_dA_cs_batch + pid_c *
        stride_dA_cs_chunk + pid_h * stride_dA_cs_head)
    if HAS_SEQ_IDX:
        seq_idx_ptr += (pid_b * stride_seq_idx_batch + pid_c * chunk_size *
            stride_seq_idx_seqlen)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_hdim + offs_k[
        None, :] * stride_dout_seqlen)
    c_ptrs = c_ptr + (offs_n[None, :] * stride_c_dstate + offs_k[:, None] *
        stride_c_seqlen)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    if HAS_SEQ_IDX:
        seq_idx_ptrs = seq_idx_ptr + offs_k * stride_seq_idx_seqlen
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if HAS_SEQ_IDX:
        seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen, mask=
            pid_c >= 1, other=0)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < hdim) & (offs_k[
            None, :] < chunk_size_limit - k), other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size - k,
            other=0.0).to(tl.float32)
        if not HAS_SEQ_IDX:
            scale_k = tl.exp(dA_cs_k)
        else:
            seq_idx_k = tl.load(seq_idx_ptrs, mask=offs_k < 
                chunk_size_limit - k, other=-1)
            scale_k = tl.where(seq_idx_k == seq_idx_prev, tl.exp(dA_cs_k), 0.0)
        dout = (dout * scale_k).to(dout_ptr.dtype.element_ty)
        c = tl.load(c_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) &
            (offs_n[None, :] < dstate), other=0.0)
        acc += tl.dot(dout, c)
        dout_ptrs += BLOCK_SIZE_K * stride_dout_seqlen
        c_ptrs += BLOCK_SIZE_K * stride_c_seqlen
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
        if HAS_SEQ_IDX:
            seq_idx_ptrs += BLOCK_SIZE_K * stride_seq_idx_seqlen
    out = acc.to(dprev_states_ptr.dtype.element_ty)
    dprev_states_ptr += (pid_b * stride_dprev_states_batch + pid_c *
        stride_dprev_states_chunk + pid_h * stride_dprev_states_head)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dprev_states_ptrs = dprev_states_ptr + (offs_m[:, None] *
        stride_dprev_states_hdim + offs_n[None, :] * stride_dprev_states_dstate
        )
    tl.store(dprev_states_ptrs, out, mask=(offs_m[:, None] < hdim) & (
        offs_n[None, :] < dstate))


@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_M': 32}), triton.
    Config({'BLOCK_SIZE_M': 64}), triton.Config({'BLOCK_SIZE_M': 128}),
    triton.Config({'BLOCK_SIZE_M': 256})], key=['chunk_size', 'hdim'])
@triton.jit
def _chunk_scan_bwd_dz_kernel(dout_ptr, out_ptr, z_ptr, x_ptr, D_ptr,
    outz_ptr, dz_ptr, dout_x_ptr, dD_ptr, ddA_cumsum_ptr, chunk_size, hdim,
    batch, seqlen, stride_dout_batch, stride_dout_seqlen, stride_dout_head,
    stride_dout_hdim, stride_out_batch, stride_out_seqlen, stride_out_head,
    stride_out_hdim, stride_z_batch, stride_z_seqlen, stride_z_head,
    stride_z_hdim, stride_x_batch, stride_x_seqlen, stride_x_head,
    stride_x_hdim, stride_D_head, stride_outz_batch, stride_outz_seqlen,
    stride_outz_head, stride_outz_hdim, stride_dz_batch, stride_dz_seqlen,
    stride_dz_head, stride_dz_hdim, stride_doutx_batch, stride_doutx_seqlen,
    stride_doutx_head, stride_doutx_hdim, stride_dD_batch, stride_dD_chunk,
    stride_dD_head, stride_dD_csize, stride_dD_hdim, stride_ddA_cs_batch,
    stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_csize, HAS_D: tl
    .constexpr, D_HAS_HDIM: tl.constexpr, HAS_DDACS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    dout_ptr += (pid_b * stride_dout_batch + pid_c * chunk_size *
        stride_dout_seqlen + pid_h * stride_dout_head)
    dout_x_ptr += (pid_b * stride_doutx_batch + pid_c * chunk_size *
        stride_doutx_seqlen + pid_h * stride_doutx_head)
    out_ptr += (pid_b * stride_out_batch + pid_c * chunk_size *
        stride_out_seqlen + pid_h * stride_out_head)
    z_ptr += (pid_b * stride_z_batch + pid_c * chunk_size * stride_z_seqlen +
        pid_h * stride_z_head)
    dz_ptr += (pid_b * stride_dz_batch + pid_c * chunk_size *
        stride_dz_seqlen + pid_h * stride_dz_head)
    if RECOMPUTE_OUTPUT:
        outz_ptr += (pid_b * stride_outz_batch + pid_c * chunk_size *
            stride_outz_seqlen + pid_h * stride_outz_head)
    if HAS_DDACS:
        ddA_cumsum_ptr += (pid_b * stride_ddA_cs_batch + pid_c *
            stride_ddA_cs_chunk + pid_h * stride_ddA_cs_head)
    if HAS_D:
        x_ptr += (pid_b * stride_x_batch + pid_c * chunk_size *
            stride_x_seqlen + pid_h * stride_x_head)
        dD_ptr += (pid_b * stride_dD_batch + pid_c * stride_dD_chunk + 
            pid_h * stride_dD_head + pid_m * stride_dD_csize)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_n[
        None, :] * stride_dout_hdim)
    dout_x_ptrs = dout_x_ptr + (offs_m[:, None] * stride_doutx_seqlen + 
        offs_n[None, :] * stride_doutx_hdim)
    out_ptrs = out_ptr + (offs_m[:, None] * stride_out_seqlen + offs_n[None,
        :] * stride_out_hdim)
    z_ptrs = z_ptr + (offs_m[:, None] * stride_z_seqlen + offs_n[None, :] *
        stride_z_hdim)
    dz_ptrs = dz_ptr + (offs_m[:, None] * stride_dz_seqlen + offs_n[None, :
        ] * stride_dz_hdim)
    if RECOMPUTE_OUTPUT:
        outz_ptrs = outz_ptr + (offs_m[:, None] * stride_outz_seqlen + 
            offs_n[None, :] * stride_outz_hdim)
    if HAS_D:
        x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None,
            :] * stride_x_hdim)
        if D_HAS_HDIM:
            dD_ptrs = dD_ptr + offs_n * stride_dD_hdim
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (
        offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    out = tl.load(out_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (
        offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    z = tl.load(z_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n
        [None, :] < hdim), other=0.0).to(tl.float32)
    z_sigmoid = tl.sigmoid(z)
    if RECOMPUTE_OUTPUT:
        outz = out * z * z_sigmoid
        tl.store(outz_ptrs, outz, mask=(offs_m[:, None] < chunk_size_limit) &
            (offs_n[None, :] < hdim))
    dz = dout * out * z_sigmoid * (1 + z * (1 - z_sigmoid))
    tl.store(dz_ptrs, dz, mask=(offs_m[:, None] < chunk_size_limit) & (
        offs_n[None, :] < hdim))
    dout *= z * z_sigmoid
    tl.store(dout_x_ptrs, dout, mask=(offs_m[:, None] < chunk_size_limit) &
        (offs_n[None, :] < hdim))
    if HAS_D:
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (
            offs_n[None, :] < hdim), other=0.0).to(tl.float32)
        if D_HAS_HDIM:
            dD = tl.sum(dout * x, axis=0)
            tl.store(dD_ptrs, dD, mask=offs_n < hdim)
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n <
                hdim, other=0.0).to(tl.float32)
        else:
            dD = tl.sum(dout * x)
            tl.store(dD_ptr, dD)
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        out -= x * D
    if HAS_DDACS:
        ddA_cs = tl.sum(dout * out, axis=1)
        tl.store(ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize, ddA_cs,
            mask=offs_m < chunk_size)


@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_H': 1}, pre_hook=
    init_to_zero(['dA_ptr', 'ddt_bias_ptr'])), triton.Config({
    'BLOCK_SIZE_H': 2}, pre_hook=init_to_zero(['dA_ptr', 'ddt_bias_ptr'])),
    triton.Config({'BLOCK_SIZE_H': 4}, pre_hook=init_to_zero(['dA_ptr',
    'ddt_bias_ptr'])), triton.Config({'BLOCK_SIZE_H': 8}, pre_hook=
    init_to_zero(['dA_ptr', 'ddt_bias_ptr'])), triton.Config({
    'BLOCK_SIZE_H': 16}, pre_hook=init_to_zero(['dA_ptr', 'ddt_bias_ptr'])),
    triton.Config({'BLOCK_SIZE_H': 32}, pre_hook=init_to_zero(['dA_ptr',
    'ddt_bias_ptr'])), triton.Config({'BLOCK_SIZE_H': 64}, pre_hook=
    init_to_zero(['dA_ptr', 'ddt_bias_ptr']))], key=['chunk_size', 'nheads'])
@triton.jit
def _chunk_cumsum_bwd_kernel(ddA_ptr, ddt_out_ptr, dt_ptr, A_ptr,
    dt_bias_ptr, ddt_ptr, dA_ptr, ddt_bias_ptr, batch, seqlen, nheads,
    chunk_size, dt_min, dt_max, stride_ddA_batch, stride_ddA_chunk,
    stride_ddA_head, stride_ddA_csize, stride_ddt_out_batch,
    stride_ddt_out_chunk, stride_ddt_out_head, stride_ddt_out_csize,
    stride_dt_batch, stride_dt_seqlen, stride_dt_head, stride_A_head,
    stride_dt_bias_head, stride_ddt_batch, stride_ddt_seqlen,
    stride_ddt_head, stride_dA_head, stride_ddt_bias_head, DT_SOFTPLUS: tl.
    constexpr, HAS_DT_BIAS: tl.constexpr, BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_CHUNK: tl.constexpr):
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    ddt_out_ptr += pid_b * stride_ddt_out_batch + pid_c * stride_ddt_out_chunk
    ddA_ptr += pid_b * stride_ddA_batch + pid_c * stride_ddA_chunk
    dt_ptr += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    ddt_ptr += (pid_b * stride_ddt_batch + pid_c * chunk_size *
        stride_ddt_seqlen)
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    ddt_out_ptrs = ddt_out_ptr + (offs_h[:, None] * stride_ddt_out_head + 
        offs_c[None, :] * stride_ddt_out_csize)
    ddA_ptrs = ddA_ptr + (offs_h[:, None] * stride_ddA_head + offs_c[None,
        :] * stride_ddA_csize)
    dt_ptrs = dt_ptr + (offs_h[:, None] * stride_dt_head + offs_c[None, :] *
        stride_dt_seqlen)
    ddt_ptrs = ddt_ptr + (offs_h[:, None] * stride_ddt_head + offs_c[None,
        :] * stride_ddt_seqlen)
    A_ptrs = A_ptr + offs_h * stride_A_head
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    ddA = tl.load(ddA_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None,
        :] < chunk_size_limit), other=0.0).to(tl.float32)
    ddt_out = tl.load(ddt_out_ptrs, mask=(offs_h[:, None] < nheads) & (
        offs_c[None, :] < chunk_size_limit), other=0.0).to(tl.float32)
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    ddt = ddA * A[:, None] + ddt_out
    dt = tl.load(dt_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :
        ] < chunk_size_limit), other=0.0).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + offs_h * stride_dt_bias_head, mask=
            offs_h < nheads, other=0.0).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt_presoftplus = dt
        dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), ddt)
    clamp_mask = (dt < dt_min) | (dt > dt_max)
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    dt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] <
        chunk_size_limit), dt, 0.0)
    ddt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] <
        chunk_size_limit), ddt, 0.0)
    ddt = tl.where(clamp_mask, 0.0, ddt)
    if DT_SOFTPLUS:
        ddt = tl.where(dt_presoftplus <= 20.0, ddt * tl.sigmoid(
            dt_presoftplus), ddt)
    tl.store(ddt_ptrs, ddt, mask=(offs_h[:, None] < nheads) & (offs_c[None,
        :] < chunk_size_limit))
    dA = tl.sum(ddA * dt, axis=1)
    tl.atomic_add(dA_ptr + offs_h * stride_dA_head, dA, mask=offs_h < nheads)
    if HAS_DT_BIAS:
        ddt_bias = tl.sum(ddt, axis=1)
        tl.atomic_add(ddt_bias_ptr + offs_h * stride_ddt_bias_head,
            ddt_bias, mask=offs_h < nheads)


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
    pre_hook=init_to_zero(['ddt_ptr'])), triton.Config({'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4,
    pre_hook=init_to_zero(['ddt_ptr'])), triton.Config({'BLOCK_SIZE_M': 128,
    'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4,
    pre_hook=init_to_zero(['ddt_ptr'])), triton.Config({'BLOCK_SIZE_M': 128,
    'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4,
    pre_hook=init_to_zero(['ddt_ptr'])), triton.Config({'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4,
    pre_hook=init_to_zero(['ddt_ptr'])), triton.Config({'BLOCK_SIZE_M': 128,
    'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4,
    pre_hook=init_to_zero(['ddt_ptr'])), triton.Config({'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4,
    pre_hook=init_to_zero(['ddt_ptr'])), triton.Config({'BLOCK_SIZE_M': 32,
    'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4,
    pre_hook=init_to_zero(['ddt_ptr'])), triton.Config({'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4,
    pre_hook=init_to_zero(['ddt_ptr']))], key=['chunk_size', 'hdim', 'dstate'])
@triton.jit
def _chunk_scan_chunk_state_bwd_dx_kernel(x_ptr, cb_ptr, dout_ptr, dt_ptr,
    dA_cumsum_ptr, seq_idx_ptr, D_ptr, b_ptr, dstates_ptr, dx_ptr, ddt_ptr,
    dD_ptr, chunk_size, hdim, dstate, batch, seqlen, nheads_ngroups_ratio,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m,
    stride_cb_csize_k, stride_dout_batch, stride_dout_seqlen,
    stride_dout_head, stride_dout_hdim, stride_dt_batch, stride_dt_chunk,
    stride_dt_head, stride_dt_csize, stride_dA_cs_batch, stride_dA_cs_chunk,
    stride_dA_cs_head, stride_dA_cs_csize, stride_seq_idx_batch,
    stride_seq_idx_seqlen, stride_D_head, stride_b_batch, stride_b_seqlen,
    stride_b_head, stride_b_dstate, stride_dstates_batch,
    stride_dstates_chunk, stride_dstates_head, stride_dstates_hdim,
    stride_dstates_dstate, stride_dx_batch, stride_dx_seqlen,
    stride_dx_head, stride_dx_hdim, stride_ddt_batch, stride_ddt_chunk,
    stride_ddt_head, stride_ddt_csize, stride_dD_batch, stride_dD_chunk,
    stride_dD_head, stride_dD_csize, stride_dD_hdim, HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr, HAS_SEQ_IDX: tl.constexpr, BLOCK_SIZE_M: tl.
    constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr, IS_TRITON_22: tl.constexpr):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    x_ptr += (pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen +
        pid_h * stride_x_head)
    cb_ptr += (pid_b * stride_cb_batch + pid_c * stride_cb_chunk + pid_h //
        nheads_ngroups_ratio * stride_cb_head)
    dout_ptr += (pid_b * stride_dout_batch + pid_c * chunk_size *
        stride_dout_seqlen + pid_h * stride_dout_head)
    dt_ptr += (pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h *
        stride_dt_head)
    ddt_ptr += (pid_b * stride_ddt_batch + pid_c * stride_ddt_chunk + pid_h *
        stride_ddt_head)
    dA_cumsum_ptr += (pid_b * stride_dA_cs_batch + pid_c *
        stride_dA_cs_chunk + pid_h * stride_dA_cs_head)
    b_ptr += (pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen +
        pid_h // nheads_ngroups_ratio * stride_b_head)
    dstates_ptr += (pid_b * stride_dstates_batch + pid_c *
        stride_dstates_chunk + pid_h * stride_dstates_head)
    if HAS_SEQ_IDX:
        seq_idx_ptr += (pid_b * stride_seq_idx_batch + pid_c * chunk_size *
            stride_seq_idx_seqlen)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=
        offs_m < chunk_size_limit, other=0.0).to(tl.float32)
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize
        ).to(tl.float32)
    if not HAS_SEQ_IDX:
        scale = tl.exp(dA_cs_last - dA_cs_m)
    else:
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen,
            mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) *
            stride_seq_idx_seqlen)
        scale = tl.where(seq_idx_m == seq_idx_last, tl.exp(dA_cs_last -
            dA_cs_m), 0.0)
    offs_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if IS_TRITON_22 and 
        BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
    b_ptrs = b_ptr + (offs_m[:, None] * stride_b_seqlen + offs_dstate[None,
        :] * stride_b_dstate)
    dstates_ptrs = dstates_ptr + (offs_n[None, :] * stride_dstates_hdim + 
        offs_dstate[:, None] * stride_dstates_dstate)
    if IS_TRITON_22 and BLOCK_SIZE_DSTATE <= 128:
        b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (
            offs_dstate[None, :] < dstate), other=0.0)
        dstates = tl.load(dstates_ptrs, mask=(offs_dstate[:, None] < dstate
            ) & (offs_n[None, :] < hdim), other=0.0)
        dstates = dstates.to(b_ptr.dtype.element_ty)
        acc = tl.dot(b, dstates) * scale[:, None]
    else:
        for k in range(0, dstate, BLOCK_SIZE_K):
            b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) &
                (offs_dstate[None, :] < dstate - k), other=0.0)
            dstates = tl.load(dstates_ptrs, mask=(offs_dstate[:, None] < 
                dstate - k) & (offs_n[None, :] < hdim), other=0.0)
            dstates = dstates.to(b_ptr.dtype.element_ty)
            acc += tl.dot(b, dstates)
            b_ptrs += BLOCK_SIZE_K * stride_b_dstate
            dstates_ptrs += BLOCK_SIZE_K * stride_dstates_dstate
        acc *= scale[:, None]
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None,
        :] * stride_cb_csize_k)
    dout_ptrs = dout_ptr + (offs_k[:, None] * stride_dout_seqlen + offs_n[
        None, :] * stride_dout_hdim)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    K_MAX = chunk_size_limit
    K_MIN = pid_m * BLOCK_SIZE_M
    cb_ptrs += K_MIN * stride_cb_csize_k
    dout_ptrs += K_MIN * stride_dout_seqlen
    dA_cumsum_ptrs += K_MIN * stride_dA_cs_csize
    for k in range(K_MIN, K_MAX, BLOCK_SIZE_K):
        k = tl.multiple_of(k, BLOCK_SIZE_K)
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_k
            [None, :] < K_MAX - k), other=0.0)
        dout = tl.load(dout_ptrs, mask=(offs_k[:, None] < K_MAX - k) & (
            offs_n[None, :] < hdim), other=0.0)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < K_MAX - k, other=0.0
            ).to(tl.float32)
        cb *= tl.exp(dA_cs_k[None, :] - dA_cs_m[:, None])
        mask = k + offs_k[None, :] >= offs_m[:, None]
        cb = tl.where(mask, cb, 0.0)
        cb = cb.to(dout_ptr.dtype.element_ty)
        acc += tl.dot(cb, dout)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        dout_ptrs += BLOCK_SIZE_K * stride_dout_seqlen
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl
        .float32)
    dx = acc * dt_m[:, None]
    dx_ptr += (pid_b * stride_dx_batch + pid_c * chunk_size *
        stride_dx_seqlen + pid_h * stride_dx_head)
    dx_ptrs = dx_ptr + (offs_m[:, None] * stride_dx_seqlen + offs_n[None, :
        ] * stride_dx_hdim)
    if HAS_D:
        dout_res_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + 
            offs_n[None, :] * stride_dout_hdim)
        dout_res = tl.load(dout_res_ptrs, mask=(offs_m[:, None] <
            chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.
            float32)
        if D_HAS_HDIM:
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n <
                hdim, other=0.0).to(tl.float32)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        dx += dout_res * D
    tl.store(dx_ptrs, dx, mask=(offs_m[:, None] < chunk_size_limit) & (
        offs_n[None, :] < hdim))
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] *
        stride_x_hdim)
    x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n
        [None, :] < hdim), other=0.0).to(tl.float32)
    if HAS_D:
        dD_ptr += (pid_b * stride_dD_batch + pid_c * stride_dD_chunk + 
            pid_h * stride_dD_head + pid_m * stride_dD_csize)
        if D_HAS_HDIM:
            dD_ptrs = dD_ptr + offs_n * stride_dD_hdim
            dD = tl.sum(dout_res * x, axis=0)
            tl.store(dD_ptrs, dD, mask=offs_n < hdim)
        else:
            dD = tl.sum(dout_res * x)
            tl.store(dD_ptr, dD)
    ddt = tl.sum(acc * x, axis=1)
    ddt_ptrs = ddt_ptr + offs_m * stride_ddt_csize
    tl.atomic_add(ddt_ptrs, ddt, mask=offs_m < chunk_size)


@triton.autotune(configs=[triton.Config({'BLOCK_SIZE': 64}), triton.Config(
    {'BLOCK_SIZE': 128}), triton.Config({'BLOCK_SIZE': 256}), triton.Config
    ({'BLOCK_SIZE': 512}), triton.Config({'BLOCK_SIZE': 1024}), triton.
    Config({'BLOCK_SIZE': 2048})], key=['dim'])
@triton.jit
def _state_passing_bwd_kernel(dout_ptr, out_ptr, dA_cs_ptr,
    dfinal_states_ptr, seq_idx_ptr, dstates_ptr, ddA_cs_ptr,
    dinitstates_ptr, states_converted_ptr, dim, nchunks, seqlen, chunk_size,
    stride_dout_batch, stride_dout_chunk, stride_dout_head, stride_dout_dim,
    stride_out_batch, stride_out_chunk, stride_out_head, stride_out_dim,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head,
    stride_dfinal_states_batch, stride_dfinal_states_head,
    stride_dfinal_states_dim, stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_dstates_batch, stride_dstates_chunk, stride_dstates_head,
    stride_dstates_dim, stride_ddA_cs_batch, stride_ddA_cs_chunk,
    stride_ddA_cs_head, stride_dinitstates_batch, stride_dinitstates_head,
    stride_dinitstates_dim, CONVERT_STATES: tl.constexpr, HAS_DFINAL_STATES:
    tl.constexpr, HAS_DINITSTATES: tl.constexpr, HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    dstates_ptr += (pid_b * stride_dstates_batch + pid_h *
        stride_dstates_head + (nchunks - 1) * stride_dstates_chunk)
    dA_cs_ptr += pid_b * stride_dA_cs_batch + pid_h * stride_dA_cs_head + (
        nchunks - 1) * stride_dA_cs_chunk
    ddA_cs_ptr += pid_b * stride_ddA_cs_batch + pid_h * stride_ddA_cs_head + (
        nchunks - 1) * stride_ddA_cs_chunk + pid_m
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head + (nchunks -
        1) * stride_out_chunk
    dout_ptr += pid_b * stride_dout_batch + pid_h * stride_dout_head + (nchunks
         - 1) * stride_dout_chunk
    if CONVERT_STATES:
        states_converted_ptr += (pid_b * stride_out_batch + pid_h *
            stride_out_head + (nchunks - 1) * stride_out_chunk)
    if HAS_DFINAL_STATES:
        dfinal_states_ptr += (pid_b * stride_dfinal_states_batch + pid_h *
            stride_dfinal_states_head)
    if HAS_DINITSTATES:
        dinitstates_ptr += (pid_b * stride_dinitstates_batch + pid_h *
            stride_dinitstates_head)
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dstates_ptrs = dstates_ptr + offs_m * stride_dstates_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim
    dout_ptrs = dout_ptr + offs_m * stride_dout_dim
    if CONVERT_STATES:
        states_converted_ptrs = states_converted_ptr + offs_m * stride_out_dim
    if HAS_DFINAL_STATES:
        dstates = tl.load(dfinal_states_ptr + offs_m *
            stride_dfinal_states_dim, mask=offs_m < dim, other=0.0).to(tl.
            float32)
    else:
        dstates = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    tl.store(dstates_ptrs, dstates, mask=offs_m < dim)
    if HAS_SEQ_IDX:
        seq_idx = tl.load(seq_idx_ptr + (seqlen - 1) * stride_seq_idx_seqlen)
    dstates_ptrs -= stride_dstates_chunk
    for c in range(nchunks - 1):
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        scale = tl.exp(dA_cs)
        if HAS_SEQ_IDX:
            seq_idx_new = tl.load(seq_idx_ptr + ((nchunks - c - 1) *
                chunk_size - 1) * stride_seq_idx_seqlen)
            scale = tl.where(seq_idx_new == seq_idx, scale, 0.0)
            seq_idx = seq_idx_new
        out = tl.load(out_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if CONVERT_STATES:
            tl.store(states_converted_ptrs, out, mask=offs_m < dim)
        ddA = tl.sum(out * dstates) * scale
        tl.store(ddA_cs_ptr, ddA)
        dout = tl.load(dout_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        dstates = scale * dstates + dout
        tl.store(dstates_ptrs, dstates, mask=offs_m < dim)
        dout_ptrs -= stride_dout_chunk
        dstates_ptrs -= stride_dstates_chunk
        dA_cs_ptr -= stride_dA_cs_chunk
        ddA_cs_ptr -= stride_ddA_cs_chunk
        out_ptrs -= stride_out_chunk
        if CONVERT_STATES:
            states_converted_ptrs -= stride_out_chunk
    if CONVERT_STATES:
        out = tl.load(out_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        tl.store(states_converted_ptrs, out, mask=offs_m < dim)
    if not HAS_DINITSTATES:
        tl.store(ddA_cs_ptr, 0.0)
    else:
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        scale = tl.exp(dA_cs)
        if HAS_SEQ_IDX:
            scale = tl.where(seq_idx == 0, scale, 0.0)
        out = tl.load(out_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        ddA = tl.sum(out * dstates) * scale
        tl.store(ddA_cs_ptr, ddA)
        dout = tl.load(dout_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        dstates = scale * dstates + dout
        tl.store(dinitstates_ptr + offs_m * stride_dinitstates_dim, dstates,
            mask=offs_m < dim)


def _bmm_chunk_bwd(a, dout, residual=None, out=None):
    """
    Argument:
        a: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        dout: (batch, nchunks, chunk_size, chunk_size) or (batch, nchunks, ngroups, chunk_size, chunk_size)
        residual: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
    Return:
        out: (batch, seqlen, k) or (batch, seqlen, ngroups, k)

    If there was seq_idx in the fwd pass, then dout[i, j] for seq_idx[i] != seq_idx[j] should already be
    zeroed out before calling this function.
    """
    has_groups = a.dim() == 4
    if not has_groups:
        batch, seqlen, k = a.shape
    else:
        batch, seqlen, ngroups, k = a.shape
    nchunks, chunk_size = dout.shape[1], dout.shape[-1]
    if a.stride(-1) != 1 and a.stride(-2) != 1:
        a = a.contiguous()
    if dout.stride(-1) != 1 and dout.stride(-2) != 1:
        dout = dout.contiguous()
    if residual is not None:
        assert residual.shape == (batch, seqlen, k) if not has_groups else (
            batch, seqlen, ngroups, k)
        if residual.stride(-1) != 1 and residual.stride(1) != 1:
            residual = residual.contiguous()
    if out is not None:
        assert out.shape == a.shape
        assert out.stride(-1) == 1 or out.stride(1) == 1
    else:
        out = torch.empty_like(a)
    dot_dtype = (tl.bfloat16 if a.dtype == torch.bfloat16 or dout.dtype ==
        torch.bfloat16 else tl.float16 if a.dtype == torch.float16 or dout.
        dtype == torch.float16 else tl.float32)
    grid = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) *
        triton.cdiv(k, META['BLOCK_SIZE_N']), batch, nchunks if not
        has_groups else nchunks * ngroups)
    residual_strides = (residual.stride(0), residual.stride(1), 0 if not
        has_groups else residual.stride(2), residual.stride(-1)
        ) if residual is not None else (0, 0, 0, 0)
    with torch.cuda.device(a.device.index):
        _bmm_chunk_bwd_kernel[grid](a, dout, out, residual, int(seqlen),
            int(chunk_size), int(k), int(ngroups if has_groups else 1), a.
            stride(0), a.stride(1), 0 if not has_groups else a.stride(2), a
            .stride(-1), dout.stride(0), dout.stride(1), 0 if not
            has_groups else dout.stride(2), dout.stride(-2), dout.stride(-1
            ), out.stride(0), out.stride(1), 0 if not has_groups else out.
            stride(2), out.stride(-1), residual_strides[0],
            residual_strides[1], residual_strides[2], residual_strides[3],
            dot_dtype, HAS_RESIDUAL=residual is not None)
    return out


def _chunk_scan_bwd_dC(prev_states, dA_cumsum, dout, seq_idx=None, C=None,
    ngroups=1):
    batch, nchunks, nheads, headdim, dstate = prev_states.shape
    _, seqlen, _, _ = dout.shape
    _, _, _, chunk_size = dA_cumsum.shape
    assert prev_states.shape == (batch, nchunks, nheads, headdim, dstate)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert dout.shape == (batch, seqlen, nheads, headdim)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if C is not None:
        assert C.shape == (batch, seqlen, ngroups, dstate)
        C_strides = C.stride(0), C.stride(1), C.stride(2), C.stride(3)
        ddA_cumsum_prev = torch.empty(batch, nheads, nchunks, chunk_size,
            device=dout.device, dtype=torch.float32)
        ddA_cumsum_prev_strides = ddA_cumsum_prev.stride(0
            ), ddA_cumsum_prev.stride(2), ddA_cumsum_prev.stride(1
            ), ddA_cumsum_prev.stride(3)
    else:
        C_strides = 0, 0, 0, 0
        ddA_cumsum_prev = None
        ddA_cumsum_prev_strides = 0, 0, 0, 0
    nheads_ngroups_ratio = nheads // ngroups
    sm_count = torch.cuda.get_device_properties(dout.device
        ).multi_processor_count
    nheads_per_program = max(min(math.ceil(batch * nchunks * nheads /
        sm_count), nheads_ngroups_ratio), 1)
    nsplits = triton.cdiv(nheads_ngroups_ratio, nheads_per_program)
    dC = torch.empty(batch, seqlen, nsplits, ngroups, dstate, device=dout.
        device, dtype=torch.float32)
    grid_dc = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) *
        triton.cdiv(dstate, META['BLOCK_SIZE_N']), batch * nchunks, nsplits *
        ngroups)
    with torch.cuda.device(dout.device.index):
        _chunk_scan_bwd_dc_kernel[grid_dc](dout, prev_states, C, dA_cumsum,
            seq_idx, dC, ddA_cumsum_prev, int(chunk_size), int(dstate), int
            (headdim), int(batch), int(seqlen), int(nheads), int(
            nheads_per_program), int(ngroups), dout.stride(0), dout.stride(
            1), dout.stride(2), dout.stride(3), prev_states.stride(0),
            prev_states.stride(1), prev_states.stride(2), prev_states.
            stride(3), prev_states.stride(4), *C_strides, dA_cumsum.stride(
            0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(
            3), *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not
            None else (0, 0)), dC.stride(0), dC.stride(1), dC.stride(2), dC
            .stride(3), dC.stride(4), *ddA_cumsum_prev_strides, HAS_DDA_CS=
            ddA_cumsum_prev is not None, HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_SIZE_K=max(triton.next_power_of_2(headdim), 16))
    dC = dC.sum(2)
    return dC if C is None else (dC, ddA_cumsum_prev)


def _chunk_scan_bwd_dcb(x, dt, dA_cumsum, dout, seq_idx=None, CB=None,
    ngroups=1):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dout.shape == x.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if CB is not None:
        assert CB.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
        CB_strides = CB.stride(0), CB.stride(1), CB.stride(2), CB.stride(3
            ), CB.stride(4)
        BLOCK_SIZE_M_min = 16
        ddA_cumsum = torch.empty(batch, nheads, nchunks, triton.cdiv(
            chunk_size, BLOCK_SIZE_M_min), chunk_size, device=x.device,
            dtype=torch.float32)
        ddA_cumsum_strides = ddA_cumsum.stride(0), ddA_cumsum.stride(2
            ), ddA_cumsum.stride(1), ddA_cumsum.stride(3), ddA_cumsum.stride(4)
    else:
        CB_strides = 0, 0, 0, 0, 0
        ddA_cumsum = None
        ddA_cumsum_strides = 0, 0, 0, 0, 0
    nheads_ngroups_ratio = nheads // ngroups
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    nheads_per_program = max(min(math.ceil(batch * nchunks * nheads /
        sm_count), nheads_ngroups_ratio), 1)
    nsplits = triton.cdiv(nheads_ngroups_ratio, nheads_per_program)
    dcb = torch.empty(batch, nchunks, nsplits, ngroups, chunk_size,
        chunk_size, device=x.device, dtype=torch.float32)
    grid_dcb = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) *
        triton.cdiv(chunk_size, META['BLOCK_SIZE_N']), batch * nchunks, 
        nsplits * ngroups)
    with torch.cuda.device(x.device.index):
        _chunk_scan_bwd_dcb_kernel[grid_dcb](x, dout, CB, dt, dA_cumsum,
            seq_idx, dcb, ddA_cumsum, int(chunk_size), int(headdim), int(
            batch), int(seqlen), int(nheads), int(nheads_per_program), int(
            ngroups), x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            *CB_strides, dt.stride(0), dt.stride(2), dt.stride(1), dt.
            stride(3), dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.
            stride(1), dA_cumsum.stride(3), *((seq_idx.stride(0), seq_idx.
            stride(1)) if seq_idx is not None else (0, 0)), dcb.stride(0),
            dcb.stride(1), dcb.stride(2), dcb.stride(3), dcb.stride(4), dcb
            .stride(5), *ddA_cumsum_strides, HAS_DDA_CS=ddA_cumsum is not
            None, HAS_SEQ_IDX=seq_idx is not None, BLOCK_SIZE_K=max(triton.
            next_power_of_2(headdim), 16))
    dcb = dcb.sum(2)
    if ddA_cumsum is not None:
        BLOCK_SIZE_M_actual = _chunk_scan_bwd_dcb_kernel.best_config.kwargs[
            'BLOCK_SIZE_M']
        n_valid_blocks = (chunk_size + BLOCK_SIZE_M_actual - 1
            ) // BLOCK_SIZE_M_actual
        ddA_cumsum = ddA_cumsum[:, :, :, :n_valid_blocks].sum(dim=3)
    return dcb if CB is None else (dcb, ddA_cumsum)


def _chunk_scan_bwd_ddAcs_stable(x, dt, dA_cumsum, dout, cb):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dout.shape == x.shape
    assert dA_cumsum.shape == dt.shape
    ngroups = cb.shape[2]
    assert nheads % ngroups == 0
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    BLOCK_SIZE_M_min = 32
    ddA_cumsum = torch.empty(batch, nheads, nchunks, triton.cdiv(chunk_size,
        BLOCK_SIZE_M_min), chunk_size, device=x.device, dtype=torch.float32)
    grid_ddtcs = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']
        ), batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_scan_bwd_ddAcs_stable_kernel[grid_ddtcs](x, dout, dt,
            dA_cumsum, cb, ddA_cumsum, int(chunk_size), int(headdim), int(
            batch), int(seqlen), int(nheads // ngroups), x.stride(0), x.
            stride(1), x.stride(2), x.stride(3), dout.stride(0), dout.
            stride(1), dout.stride(2), dout.stride(3), dt.stride(0), dt.
            stride(2), dt.stride(1), dt.stride(3), dA_cumsum.stride(0),
            dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            cb.stride(0), cb.stride(1), cb.stride(2), cb.stride(3), cb.
            stride(4), ddA_cumsum.stride(0), ddA_cumsum.stride(2),
            ddA_cumsum.stride(1), ddA_cumsum.stride(3), ddA_cumsum.stride(4
            ), BLOCK_SIZE_K=max(triton.next_power_of_2(headdim), 16))
    BLOCK_SIZE_M_actual = (_chunk_scan_bwd_ddAcs_stable_kernel.best_config.
        kwargs['BLOCK_SIZE_M'])
    n_valid_blocks = (chunk_size + BLOCK_SIZE_M_actual - 1
        ) // BLOCK_SIZE_M_actual
    ddA_cumsum = ddA_cumsum[:, :, :, :n_valid_blocks].sum(dim=3)
    return ddA_cumsum


def _chunk_scan_bwd_dstates(C, dA_cumsum, dout, seq_idx=None, dtype=None):
    batch, seqlen, nheads, headdim = dout.shape
    _, _, nchunks, chunk_size = dA_cumsum.shape
    _, _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (batch, seqlen, ngroups, dstate)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    dtype = C.dtype if dtype is None else dtype
    dprev_states = torch.empty(batch, nchunks, nheads, headdim, dstate,
        device=C.device, dtype=dtype)
    grid_dstates = lambda META: (triton.cdiv(headdim, META['BLOCK_SIZE_M']) *
        triton.cdiv(dstate, META['BLOCK_SIZE_N']), batch * nchunks, nheads)
    with torch.cuda.device(C.device.index):
        _chunk_scan_bwd_dstates_kernel[grid_dstates](dout, C, dprev_states,
            dA_cumsum, seq_idx, int(headdim), int(dstate), int(chunk_size),
            int(batch), int(seqlen), int(nchunks), int(nheads // ngroups),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            dprev_states.stride(0), dprev_states.stride(1), dprev_states.
            stride(2), dprev_states.stride(3), dprev_states.stride(4),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1),
            dA_cumsum.stride(3), *((seq_idx.stride(0), seq_idx.stride(1)) if
            seq_idx is not None else (0, 0)), HAS_SEQ_IDX=seq_idx is not None)
    return dprev_states


def _chunk_scan_bwd_dz(x, z, out, dout, chunk_size, has_ddAcs=True, D=None,
    dz=None, recompute_output=False):
    batch, seqlen, nheads, headdim = x.shape
    assert z.shape == x.shape
    assert out.shape == x.shape
    assert dout.shape == out.shape
    nchunks = math.ceil(seqlen / chunk_size)
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
        assert D.stride(-1) == 1
    if has_ddAcs:
        ddA_cumsum = torch.empty(batch, nheads, nchunks, chunk_size, device
            =x.device, dtype=torch.float32)
    if D is not None:
        BLOCK_SIZE_min = 32
        dD = torch.empty(triton.cdiv(chunk_size, BLOCK_SIZE_min), batch,
            nchunks, nheads, headdim if D.dim() == 2 else 1, device=D.
            device, dtype=torch.float32)
    else:
        dD = None
    if dz is not None:
        assert dz.shape == z.shape
    else:
        dz = torch.empty_like(z)
    if recompute_output:
        outz = torch.empty_like(x)
    dout_x = torch.empty_like(dout)
    dD_strides = (dD.stride(0), dD.stride(1), dD.stride(2), dD.stride(3),
        dD.stride(4)) if D is not None else (0, 0, 0, 0, 0)
    grid_dz = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']), 
        batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_scan_bwd_dz_kernel[grid_dz](dout, out, z, x, D, outz if
            recompute_output else None, dz, dout_x, dD, ddA_cumsum if
            has_ddAcs else None, int(chunk_size), int(headdim), int(batch),
            int(seqlen), dout.stride(0), dout.stride(1), dout.stride(2),
            dout.stride(3), out.stride(0), out.stride(1), out.stride(2),
            out.stride(3), z.stride(0), z.stride(1), z.stride(2), z.stride(
            3), x.stride(0), x.stride(1), x.stride(2), x.stride(3), D.
            stride(0) if D is not None else 0, *((outz.stride(0), outz.
            stride(1), outz.stride(2), outz.stride(3)) if recompute_output else
            (0, 0, 0, 0)), dz.stride(0), dz.stride(1), dz.stride(2), dz.
            stride(3), dout_x.stride(0), dout_x.stride(1), dout_x.stride(2),
            dout_x.stride(3), dD_strides[1], dD_strides[2], dD_strides[3],
            dD_strides[0], dD_strides[4], *((ddA_cumsum.stride(0),
            ddA_cumsum.stride(2), ddA_cumsum.stride(1), ddA_cumsum.stride(3
            )) if has_ddAcs else (0, 0, 0, 0)), D is not None, D.dim() == 2 if
            D is not None else True, has_ddAcs, BLOCK_SIZE_N=max(triton.
            next_power_of_2(headdim), 16), RECOMPUTE_OUTPUT=recompute_output)
    if D is not None:
        BLOCK_SIZE_actual = _chunk_scan_bwd_dz_kernel.best_config.kwargs[
            'BLOCK_SIZE_M']
        n_valid_blocks = (chunk_size + BLOCK_SIZE_actual - 1
            ) // BLOCK_SIZE_actual
        dD = dD[:n_valid_blocks].sum(dim=(0, 1, 2)).to(dtype=D.dtype)
        if D.dim() == 1:
            dD = rearrange(dD, 'h 1 -> h')
    return_vals = (dz, dout_x, dD, ddA_cumsum) if has_ddAcs else (dz,
        dout_x, dD)
    return return_vals if not recompute_output else (*return_vals, outz)


def _chunk_cumsum_bwd(ddA, ddt_out, dt, A, dt_bias=None, dt_softplus=False,
    dt_limit=(0.0, float('inf')), ddt=None):
    batch, seqlen, nheads = dt.shape
    _, _, nchunks, chunk_size = ddA.shape
    assert ddA.shape == (batch, nheads, nchunks, chunk_size)
    assert ddt_out.shape == (batch, nheads, nchunks, chunk_size)
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
        ddt_bias = torch.empty_like(dt_bias, dtype=torch.float32)
    else:
        ddt_bias = None
    if ddt is not None:
        assert ddt.shape == dt.shape
    else:
        ddt = torch.empty_like(dt)
    dA = torch.empty_like(A, dtype=torch.float32)
    grid_chunk_cs = lambda META: (batch, nchunks, triton.cdiv(nheads, META[
        'BLOCK_SIZE_H']))
    with torch.cuda.device(dt.device.index):
        _chunk_cumsum_bwd_kernel[grid_chunk_cs](ddA, ddt_out, dt, A,
            dt_bias, ddt, dA, ddt_bias, int(batch), int(seqlen), int(nheads
            ), int(chunk_size), dt_limit[0], dt_limit[1], ddA.stride(0),
            ddA.stride(2), ddA.stride(1), ddA.stride(3), ddt_out.stride(0),
            ddt_out.stride(2), ddt_out.stride(1), ddt_out.stride(3), dt.
            stride(0), dt.stride(1), dt.stride(2), A.stride(0), dt_bias.
            stride(0) if dt_bias is not None else 0, ddt.stride(0), ddt.
            stride(1), ddt.stride(2), dA.stride(0), ddt_bias.stride(0) if 
            ddt_bias is not None else 0, dt_softplus, HAS_DT_BIAS=dt_bias
             is not None, BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size))
    return ddt, dA, ddt_bias


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


def _chunk_scan_chunk_state_bwd_dx(x, dt, dA_cumsum, B, CB, dout, dstates,
    D=None, seq_idx=None, dx=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert CB.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dout.shape == x.shape
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
        assert D.stride(-1) == 1
        BLOCK_SIZE_min = 32
        dD = torch.empty(triton.cdiv(chunk_size, BLOCK_SIZE_min), batch,
            nchunks, nheads, headdim if D.dim() == 2 else 1, device=D.
            device, dtype=torch.float32)
    else:
        dD = None
    dD_strides = (dD.stride(0), dD.stride(1), dD.stride(2), dD.stride(3),
        dD.stride(4)) if D is not None else (0, 0, 0, 0, 0)
    if dx is None:
        dx = torch.empty_like(x)
    else:
        assert dx.shape == x.shape
    ddt = torch.empty(batch, nheads, nchunks, chunk_size, device=dout.
        device, dtype=torch.float32)
    grid_dx = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) *
        triton.cdiv(headdim, META['BLOCK_SIZE_N']), batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_scan_chunk_state_bwd_dx_kernel[grid_dx](x, CB, dout, dt,
            dA_cumsum, seq_idx, D, B, dstates, dx, ddt, dD, int(chunk_size),
            int(headdim), int(dstate), int(batch), int(seqlen), int(nheads //
            ngroups), x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            CB.stride(0), CB.stride(1), CB.stride(2), CB.stride(-1), CB.
            stride(-2), dout.stride(0), dout.stride(1), dout.stride(2),
            dout.stride(3), dt.stride(0), dt.stride(2), dt.stride(1), dt.
            stride(3), dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.
            stride(1), dA_cumsum.stride(3), *((seq_idx.stride(0), seq_idx.
            stride(1)) if seq_idx is not None else (0, 0)), D.stride(0) if 
            D is not None else 0, B.stride(0), B.stride(1), B.stride(2), B.
            stride(3), dstates.stride(0), dstates.stride(1), dstates.stride
            (2), dstates.stride(3), dstates.stride(4), dx.stride(0), dx.
            stride(1), dx.stride(2), dx.stride(3), ddt.stride(0), ddt.
            stride(2), ddt.stride(1), ddt.stride(3), dD_strides[1],
            dD_strides[2], dD_strides[3], dD_strides[0], dD_strides[4], D
             is not None, D.dim() == 2 if D is not None else True,
            HAS_SEQ_IDX=seq_idx is not None, BLOCK_SIZE_DSTATE=max(triton.
            next_power_of_2(dstate), 16), IS_TRITON_22=TRITON_22)
    if D is not None:
        BLOCK_SIZE_actual = (_chunk_scan_chunk_state_bwd_dx_kernel.
            best_config.kwargs['BLOCK_SIZE_M'])
        n_valid_blocks = (chunk_size + BLOCK_SIZE_actual - 1
            ) // BLOCK_SIZE_actual
        dD = dD[:n_valid_blocks].sum(dim=(0, 1, 2)).to(dtype=D.dtype)
        if D.dim() == 1:
            dD = rearrange(dD, 'h 1 -> h')
    return dx, ddt.to(dtype=dt.dtype), dD


def _mamba_chunk_scan_combined_bwd(dout, x, dt, A, B, C, out, chunk_size, D
    =None, z=None, dt_bias=None, initial_states=None, dfinal_states=None,
    seq_idx=None, dt_softplus=False, dt_limit=(0.0, float('inf')), dx=None,
    ddt=None, dB=None, dC=None, dz=None, recompute_output=False):
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    batch, seqlen, nheads, headdim = x.shape
    nchunks = math.ceil(seqlen / chunk_size)
    _, _, ngroups, dstate = B.shape
    assert dout.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads,)
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert C.shape == B.shape
    assert out.shape == x.shape
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if dx is not None:
        assert dx.shape == x.shape
    if dB is not None:
        assert dB.shape == B.shape
        dB_given = dB
    else:
        dB_given = torch.empty_like(B)
    if dC is not None:
        assert dC.shape == C.shape
        dC_given = dC
    else:
        dC_given = torch.empty_like(C)
    if dz is not None:
        assert z is not None
        assert dz.shape == z.shape
    if ddt is not None:
        assert ddt.shape == dt.shape
        ddt_given = ddt
    else:
        ddt_given = torch.empty_like(dt)
    dt_in = dt.clone()
    dA_cumsum, dt = _chunk_cumsum_fwd(dt_in, A, chunk_size, dt_bias=dt_bias,
        dt_softplus=dt_softplus, dt_limit=dt_limit)
    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=
        torch.float32)
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx,
        states_in_fp32=True)
    states, _ = _state_passing_fwd(rearrange(states, '... p n -> ... (p n)'
        ), dA_cumsum[:, :, :, -1], initial_states=rearrange(initial_states,
        '... p n -> ... (p n)') if initial_states is not None else None,
        seq_idx=seq_idx, chunk_size=chunk_size)
    states = rearrange(states, '... (p n) -> ... p n', n=dstate)
    if z is not None:
        dz, dout, dD, *rest = _chunk_scan_bwd_dz(x, z, out, dout,
            chunk_size=chunk_size, has_ddAcs=False, D=D, dz=dz,
            recompute_output=recompute_output)
        outz = rest[0] if recompute_output else out
    else:
        dz = None
        outz = out
    dstates = _chunk_scan_bwd_dstates(C, dA_cumsum, dout, seq_idx=seq_idx,
        dtype=states.dtype)
    dstates, ddA_chunk_cumsum, dinitial_states, states = _state_passing_bwd(
        rearrange(states, '... p n -> ... (p n)'), dA_cumsum[:, :, :, -1],
        rearrange(dstates, '... p n -> ... (p n)'), dfinal_states=rearrange
        (dfinal_states, '... p n -> ... (p n)') if dfinal_states is not
        None else None, seq_idx=seq_idx, has_initial_states=initial_states
         is not None, dstates_dtype=x.dtype, states_dtype=x.dtype,
        chunk_size=chunk_size)
    states = rearrange(states, '... (p n) -> ... p n', n=dstate)
    dstates = rearrange(dstates, '... (p n) -> ... p n', n=dstate)
    dinitial_states = rearrange(dinitial_states, '... (p n) -> ... p n', n=
        dstate) if dinitial_states is not None else None
    dx, ddt, dD_from_x = _chunk_scan_chunk_state_bwd_dx(x, dt, dA_cumsum, B,
        CB, dout, dstates, D=D, seq_idx=seq_idx, dx=dx)
    dB, ddA_next = _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, seq_idx=
        seq_idx, B=B, ngroups=ngroups)
    dC, ddA_cumsum_prev = _chunk_scan_bwd_dC(states.to(x.dtype), dA_cumsum,
        dout, seq_idx=seq_idx, C=C, ngroups=ngroups)
    dCB = _chunk_scan_bwd_dcb(x, dt, dA_cumsum, dout, seq_idx=seq_idx,
        ngroups=ngroups)
    dCB = dCB.to(CB.dtype)
    _bmm_chunk_bwd(C, dCB, residual=dB, out=dB_given)
    _bmm_chunk_bwd(B, rearrange(dCB, '... l s -> ... s l'), residual=dC,
        out=dC_given)
    if z is None:
        dD = dD_from_x
    ddA_cumsum_prev[..., -1] += ddA_chunk_cumsum
    ddA_prev = ddA_cumsum_prev.flip([-1]).cumsum(dim=-1).flip([-1])
    ddA = _chunk_scan_bwd_ddAcs_stable(x, dt, dA_cumsum, dout, CB)
    ddA += ddA_next + ddA_prev
    ddt_given, dA, ddt_bias = _chunk_cumsum_bwd(ddA, ddt, dt_in, A, dt_bias
        =dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit, ddt=ddt_given)
    return_vals = (dx, ddt_given, dA, dB_given, dC_given, dD, dz, ddt_bias,
        dinitial_states)
    return return_vals if not recompute_output else (*return_vals, outz)


def _state_passing_bwd(states, dA_chunk_cumsum, dout, dfinal_states=None,
    seq_idx=None, has_initial_states=None, dstates_dtype=None, states_dtype
    =None, chunk_size=None):
    """
    states contains the initial_states at index 0. The final states are not included in states.
    """
    batch, nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)
    assert dout.shape == (batch, nchunks, nheads, dim)
    if seq_idx is not None:
        assert chunk_size is not None
        seqlen = seq_idx.shape[-1]
        assert seq_idx.shape == (batch, seqlen)
    dstates = torch.empty_like(dout, dtype=dstates_dtype if dstates_dtype
         is not None else dout.dtype)
    if states_dtype is not None and states_dtype != states.dtype:
        states_converted = torch.empty_like(states, dtype=dstates_dtype if 
            dstates_dtype is not None else dout.dtype)
        assert states_converted.stride() == states.stride()
    else:
        states_converted = None
    if has_initial_states:
        dinitstates = torch.empty_like(dstates[:, 0])
    else:
        dinitstates = None
    if dfinal_states is not None:
        assert dfinal_states.shape == (batch, nheads, dim)
    BLOCK_SIZE_min = 64
    n_blocks = (dim + BLOCK_SIZE_min - 1) // BLOCK_SIZE_min
    ddA_chunk_cumsum = torch.empty(batch, nheads, nchunks, n_blocks, dtype=
        torch.float32, device=dA_chunk_cumsum.device)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE']), batch, nheads)
    with torch.cuda.device(dout.device.index):
        _state_passing_bwd_kernel[grid](dout, states, dA_chunk_cumsum,
            dfinal_states, seq_idx, dstates, ddA_chunk_cumsum, dinitstates,
            states_converted, int(dim), int(nchunks), int(seqlen if seq_idx
             is not None else 0), int(chunk_size if seq_idx is not None else
            0), dout.stride(0), dout.stride(1), dout.stride(2), dout.stride
            (3), states.stride(0), states.stride(1), states.stride(2),
            states.stride(3), dA_chunk_cumsum.stride(0), dA_chunk_cumsum.
            stride(2), dA_chunk_cumsum.stride(1), *((dfinal_states.stride(0
            ), dfinal_states.stride(1), dfinal_states.stride(2)) if 
            dfinal_states is not None else (0, 0, 0)), *((seq_idx.stride(0),
            seq_idx.stride(1)) if seq_idx is not None else (0, 0)), dstates
            .stride(0), dstates.stride(1), dstates.stride(2), dstates.
            stride(3), ddA_chunk_cumsum.stride(0), ddA_chunk_cumsum.stride(
            2), ddA_chunk_cumsum.stride(1), *((dinitstates.stride(0),
            dinitstates.stride(1), dinitstates.stride(2)) if dinitstates is not
            None else (0, 0, 0)), CONVERT_STATES=states_converted is not
            None, HAS_DFINAL_STATES=dfinal_states is not None,
            HAS_DINITSTATES=dinitstates is not None, HAS_SEQ_IDX=seq_idx is not
            None)
    BLOCK_SIZE_actual = _state_passing_bwd_kernel.best_config.kwargs[
        'BLOCK_SIZE']
    n_valid_blocks = (dim + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
    ddA_chunk_cumsum = ddA_chunk_cumsum[..., :n_valid_blocks].sum(dim=-1).to(
        dtype=dA_chunk_cumsum.dtype)
    if states_dtype is not None and states_dtype == states.dtype:
        states_converted = states
    return (dstates, ddA_chunk_cumsum, dinitstates
        ) if states_dtype is None else (dstates, ddA_chunk_cumsum,
        dinitstates, states_converted)


# Backward method (kernel launch code)
def _MambaChunkScanCombinedFn_backward(ctx, dout, *args):
    (out, x, dt, dA_cumsum, A, B, C, D, z, dt_bias, initial_states, seq_idx
        ) = ctx.saved_tensors
    dfinal_states = args[0] if ctx.return_final_states else None
    dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states = (
        _mamba_chunk_scan_combined_bwd(dout, x, dt, A, B, C, out, ctx.
        chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=
        initial_states, dfinal_states=dfinal_states, seq_idx=seq_idx,
        dt_softplus=ctx.dt_softplus, dt_limit=ctx.dt_limit))
    return (dx, ddt, dA, dB, dC, None, dD, dz, ddt_bias, dinitial_states,
        None, None, None, None)


# ============================================================
# autograd.Function Class Definition
# ============================================================

class MambaChunkScanCombinedFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=
        None, initial_states=None, seq_idx=None, dt_softplus=False,
        dt_limit=(0.0, float('inf')), return_final_states=False):
        ctx.dt_dtype = dt.dtype
        out, out_x, dt_out, dA_cumsum, states, final_states = (
            _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D=D,
            z=z, dt_bias=dt_bias, initial_states=initial_states, seq_idx=
            seq_idx, dt_softplus=dt_softplus, dt_limit=dt_limit))
        ctx.save_for_backward(out if z is None else out_x, x, dt, dA_cumsum,
            A, B, C, D, z, dt_bias, initial_states, seq_idx)
        ctx.dt_softplus = dt_softplus
        ctx.chunk_size = chunk_size
        ctx.dt_limit = dt_limit
        ctx.return_final_states = return_final_states
        return out if not return_final_states else (out, final_states)

    @staticmethod
    def backward(ctx, dout, *args):
        (out, x, dt, dA_cumsum, A, B, C, D, z, dt_bias, initial_states, seq_idx
            ) = ctx.saved_tensors
        dfinal_states = args[0] if ctx.return_final_states else None
        dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states = (
            _mamba_chunk_scan_combined_bwd(dout, x, dt, A, B, C, out, ctx.
            chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=
            initial_states, dfinal_states=dfinal_states, seq_idx=seq_idx,
            dt_softplus=ctx.dt_softplus, dt_limit=ctx.dt_limit))
        return (dx, ddt, dA, dB, dC, None, dD, dz, ddt_bias,
            dinitial_states, None, None, None, None)
