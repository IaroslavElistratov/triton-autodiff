# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/thunlp/TritonBench
# Source-Files: data/TritonBench_G_v1/softmax_flaggems.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_chw4t45n/TritonBench-main/data/TritonBench_G_v1/softmax_flaggems.py
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
from math import log

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


@triton.heuristics({'TILE_N': heur_tile_n_inner, 'ONE_TILE_PER_CTA':
    heur_one_tile_per_cta, 'num_warps': heur_num_warps_inner})
@triton.jit
def softmax_kernel_inner(output_ptr, input_ptr, M, N, TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr):
    pid_m = tl.program_id(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=-float('inf')).to(output_ptr
            .dtype.element_ty)
        m = tl.max(inp, 0)
        e = tl.exp(inp - m)
        z = tl.sum(e, 0)
        out = e / z
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_N], value=float('-inf'), dtype=tl.float32)
        z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
        input_ptr += pid_m * N
        output_ptr += pid_m * N
        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets)
            m_new = tl.maximum(m, inp)
            z = z * tl.exp(m - m_new) + tl.exp(inp - m_new)
            m = m_new
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float('inf')
                )
            m_new = tl.maximum(m, inp)
            z = z * tl.exp(m - m_new) + tl.exp(inp - m_new)
            m = m_new
        m_reduced = tl.max(m, 0)
        z = tl.sum(z * tl.exp(m - m_reduced), 0)
        m = m_reduced
        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, TILE_N, TILE_N):
            n_offsets = previous_multiple - start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float(
                'inf'), eviction_policy='evict_first')
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o, mask=mask)
        for start_n in range(TILE_N, N, TILE_N):
            n_offsets = previous_multiple - start_n + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets, eviction_policy='evict_first')
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o)


@triton.heuristics({'TILE_K': heur_tile_k, 'TILE_N': heur_tile_n_non_inner,
    'ONE_TILE_PER_CTA': heur_one_tile_per_cta, 'num_warps':
    heur_num_warps_non_inner})
@triton.jit
def softmax_kernel_non_inner(output_ptr, input_ptr, M, N, K, TILE_N: tl.
    constexpr, TILE_K: tl.constexpr, ONE_TILE_PER_CTA: tl.constexpr):
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(0)
    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * N * K + n_offsets[:, None] * K + k_offsets
        mask = (n_offsets[:, None] < N) & (k_offsets < K)
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        m = tl.max(inp, 0)
        e = tl.exp(inp - m[None, :])
        z = tl.sum(e, 0)
        out = e / z
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_N, TILE_K], value=float('-inf'), dtype=tl.float32)
        z = tl.full([TILE_N, TILE_K], value=0.0, dtype=tl.float32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets
            mask = (n_offsets[:, None] < N) & (k_offsets < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
            m_new = tl.maximum(m, inp)
            alpha = tl.exp(m - m_new)
            z = z * alpha + tl.exp(inp - m_new)
            m = m_new
        m_reduced = tl.max(m, 0)
        z = tl.sum(z * tl.exp(m - m_reduced[None, :]), 0)
        m = m_reduced
        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, N, TILE_N):
            n_offsets = previous_multiple - start_n + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets
            mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
            o = tl.exp(inp - m[None, :]) / z[None, :]
            tl.store(output_ptr + offsets, o, mask=mask)


# Forward method (kernel launch code)
def _Softmax_forward(ctx, x, dim, dtype):
    logging.debug('GEMS SOFTMAX')
    assert dim >= -x.ndim and dim < x.ndim, 'Invalid dim'
    dim = dim % x.ndim
    M = 1
    N = x.shape[dim]
    for i in range(dim):
        M *= x.shape[i]
    inp = x.contiguous()
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(inp, dtype=dtype)
    K = inp.numel() // M // N
    with torch.cuda.device(inp.device):
        if K > 1:
            grid = lambda meta: (M, triton.cdiv(K, meta['TILE_K']), 1)
            softmax_kernel_non_inner[grid](out, inp, M, N, K)
        else:
            grid = M, 1, 1
            softmax_kernel_inner[grid](out, inp, M, N)
    ctx.save_for_backward(out)
    ctx.dim = dim
    return out


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=[triton.Config({'TILE_N': 32}), triton.Config({
    'TILE_N': 64}), triton.Config({'TILE_N': 128}), triton.Config({'TILE_N':
    256}), triton.Config({'TILE_N': 512}), triton.Config({'TILE_N': 1024})],
    key=['M', 'N'])
@triton.heuristics(values={'TILE_M': heru_tile_m, 'ONE_TILE_PER_CTA':
    heur_one_tile_per_cta})
@triton.jit
def softmax_backward_kernel_inner(out_ptr, out_grad_ptr, in_grad_ptr, M, N,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, ONE_TILE_PER_CTA: tl.constexpr
    ):
    pid_m = tl.program_id(0)
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        mask = (m_offsets[:, None] < M) & (n_offsets < N)
        out_tile = tl.load(out_ptr + offsets, mask=mask)
        out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
        scale = tl.sum(out_tile * out_grad_tile, 1)
        in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
        tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
    else:
        scale = tl.zeros([TILE_M, TILE_N], dtype=tl.float32)
        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_tile = tl.load(out_ptr + offsets, mask=mask,
                eviction_policy='evict_last')
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
            scale += out_tile * out_grad_tile
            n_offsets += TILE_N
            offsets += TILE_N
        scale = tl.sum(scale, 1)
        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_tile = tl.load(out_ptr + offsets, mask=mask,
                eviction_policy='evict_first')
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
            in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
            tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
            n_offsets += TILE_N
            offsets += TILE_N


@triton.autotune(configs=[triton.Config({'TILE_K': 32}), triton.Config({
    'TILE_K': 64}), triton.Config({'TILE_K': 128}), triton.Config({'TILE_K':
    256}), triton.Config({'TILE_K': 512}), triton.Config({'TILE_K': 1024})],
    key=['M', 'N', 'K'])
@triton.heuristics({'TILE_N': heur_tile_n_bwd_non_inner, 'ONE_TILE_PER_CTA':
    heur_one_tile_per_cta})
@triton.jit
def softmax_backward_kernel_non_inner(out_ptr, out_grad_ptr, in_grad_ptr, M,
    N, K, TILE_N: tl.constexpr, TILE_K: tl.constexpr, ONE_TILE_PER_CTA: tl.
    constexpr):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offsets_k = pid_k * TILE_K + tl.arange(0, TILE_K)
    if ONE_TILE_PER_CTA:
        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        mask = (offsets_n < N)[:, None] & (offsets_k < K)
        out_tile = tl.load(out_ptr + offsets, mask=mask)
        out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
        scale = tl.sum(out_tile * out_grad_tile, axis=0)
        in_grad_tile = out_tile * (out_grad_tile - scale[None, :])
        tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
    else:
        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        scale = tl.zeros([TILE_N, TILE_K], dtype=tl.float32)
        for _ in range(0, N, TILE_N):
            mask = (offsets_n < N)[:, None] & (offsets_k < K)
            out_tile = tl.load(out_ptr + offsets, mask=mask)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
            scale += out_tile * out_grad_tile
            offsets_n += TILE_N
            offsets += TILE_N * K
        scale = tl.sum(scale, axis=0)
        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        for _ in range(0, N, TILE_N):
            mask = (offsets_n < N)[:, None] & (offsets_k < K)
            out_tile = tl.load(out_ptr + offsets, mask=mask)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
            in_grad_tile = out_tile * (out_grad_tile - scale[None, :])
            tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
            offsets_n += TILE_N
            offsets += TILE_N * K


# Backward method (kernel launch code)
def _Softmax_backward(ctx, out_grad):
    logging.debug('GEMS SOFTMAX VJP')
    dim = ctx.dim
    out, = ctx.saved_tensors
    assert dim >= -out.ndim and dim < out.ndim, 'Invalid dim'
    dim = dim % out.ndim
    M = 1
    N = out.shape[dim]
    for i in range(dim):
        M *= out.shape[i]
    out_grad = out_grad.contiguous()
    in_grad = torch.empty_like(out)
    K = out.numel() // M // N
    with torch.cuda.device(in_grad.device):
        if K > 1:
            grid = lambda meta: (M, triton.cdiv(K, meta['TILE_K']), 1)
            softmax_backward_kernel_non_inner[grid](out, out_grad, in_grad,
                M, N, K)
        else:
            grid = lambda meta: (triton.cdiv(M, meta['TILE_M']), 1, 1)
            softmax_backward_kernel_inner[grid](out, out_grad, in_grad, M, N)
    return in_grad, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class Softmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, dim, dtype):
        logging.debug('GEMS SOFTMAX')
        assert dim >= -x.ndim and dim < x.ndim, 'Invalid dim'
        dim = dim % x.ndim
        M = 1
        N = x.shape[dim]
        for i in range(dim):
            M *= x.shape[i]
        inp = x.contiguous()
        if dtype is None:
            dtype = x.dtype
        out = torch.empty_like(inp, dtype=dtype)
        K = inp.numel() // M // N
        with torch.cuda.device(inp.device):
            if K > 1:
                grid = lambda meta: (M, triton.cdiv(K, meta['TILE_K']), 1)
                softmax_kernel_non_inner[grid](out, inp, M, N, K)
            else:
                grid = M, 1, 1
                softmax_kernel_inner[grid](out, inp, M, N)
        ctx.save_for_backward(out)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug('GEMS SOFTMAX VJP')
        dim = ctx.dim
        out, = ctx.saved_tensors
        assert dim >= -out.ndim and dim < out.ndim, 'Invalid dim'
        dim = dim % out.ndim
        M = 1
        N = out.shape[dim]
        for i in range(dim):
            M *= out.shape[i]
        out_grad = out_grad.contiguous()
        in_grad = torch.empty_like(out)
        K = out.numel() // M // N
        with torch.cuda.device(in_grad.device):
            if K > 1:
                grid = lambda meta: (M, triton.cdiv(K, meta['TILE_K']), 1)
                softmax_backward_kernel_non_inner[grid](out, out_grad,
                    in_grad, M, N, K)
            else:
                grid = lambda meta: (triton.cdiv(M, meta['TILE_M']), 1, 1)
                softmax_backward_kernel_inner[grid](out, out_grad, in_grad,
                    M, N)
        return in_grad, None, None
