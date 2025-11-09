# SPDX-License-Identifier: BSD-3-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/meta-pytorch/applied-ai
# Source-Files: kernels/triton/training/rms_norm/fused_rms_norm.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_6p51e771/applied-ai-main/kernels/triton/training/rms_norm/fused_rms_norm.py
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
from math import ceil

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8), triton.Config({}, num_warps=16), triton.Config({},
    num_warps=32)], key=['N'])
@triton.jit
def _rms_norm_fwd_kernel(X, stride_x, Y, stride_y, W, Rstd, eps, M, N,
    block_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, block_N)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    x_hat = x * rstd
    y = x_hat * w
    tl.store(Y + row * stride_y + cols, y, mask=mask)


# Forward method (kernel launch code)
def _ttt_RMSNorm_forward(ctx, x, weight, eps):
    x_shape_start = x.shape
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
    if weight.stride(-1) != 1:
        weight = weight.contiguous()
    M, N = x.shape
    y = torch.empty_like(x)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    max_size = 65536 // x.element_size()
    block_N = min(max_size, triton.next_power_of_2(N))
    if N > block_N:
        raise ValueError(f'N {N} must be <= block_N={block_N!r}')
    grid = lambda meta: (M,)
    _rms_norm_fwd_kernel[grid](x, x.stride(0), y, y.stride(0), weight, rstd,
        eps, M, N, block_N)
    ctx.eps = eps
    ctx.save_for_backward(x, weight, rstd)
    ctx.x_shape_start = x_shape_start
    y = y.reshape(x_shape_start)
    return y


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8), triton.Config({}, num_warps=16), triton.Config({},
    num_warps=32)], key=['N'])
@triton.jit
def _rms_norm_bwd_kernel_sm(X, stride_x, W, DY, stride_dy, DX, stride_dx,
    Rstd, DW, eps, M, N, rows_per_program, block_N: tl.constexpr):
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, block_N)
    mask = cols < N
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    dw = tl.zeros((block_N,), dtype=tl.float32)
    row_end = min(row_start + rows_per_program, M)
    for row in range(row_start, row_end):
        x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.
            float32)
        dy = tl.load(DY + row * stride_dy + cols, mask=mask, other=0.0).to(tl
            .float32)
        rstd = tl.load(Rstd + row)
        x_hat = x * rstd
        wdy = w * dy
        dw += dy * x_hat
        c1 = tl.sum(x_hat * wdy, axis=0) / N
        dx = (wdy - x_hat * c1) * rstd
        tl.store(DX + row * stride_dx + cols, dx, mask=mask)
    tl.store(DW + row_block_id * N + cols, dw, mask=mask)


# Backward method (kernel launch code)
def _ttt_RMSNorm_backward(ctx, dy):
    x, weight, rstd = ctx.saved_tensors
    eps = ctx.eps
    x_shape_start = ctx.x_shape_start
    dy = dy.reshape(-1, dy.shape[-1])
    if dy.stride(-1) != 1:
        dy = dy.contiguous()
    M, N = dy.shape
    dx = torch.empty_like(x)
    dw = torch.empty_like(weight)
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    _dw = torch.empty((sm_count, N), dtype=torch.float32, device=weight.device)
    max_size = 65536 // x.element_size()
    block_N = min(max_size, triton.next_power_of_2(N))
    rows_per_sm = math.ceil(M / sm_count)
    if N > block_N:
        raise ValueError(f'N {N} must be <= block_N={block_N!r}')
    grid = lambda meta: (sm_count,)
    _rms_norm_bwd_kernel_sm[grid](x, x.stride(0), weight, dy, dy.stride(0),
        dx, dx.stride(0), rstd, _dw, eps, M, N, rows_per_sm, block_N)
    dw = _dw.sum(0).to(weight.dtype)
    dx = dx.reshape(x_shape_start)
    return dx, dw, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class ttt_RMSNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, eps):
        x_shape_start = x.shape
        x = x.reshape(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if weight.stride(-1) != 1:
            weight = weight.contiguous()
        M, N = x.shape
        y = torch.empty_like(x)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
        max_size = 65536 // x.element_size()
        block_N = min(max_size, triton.next_power_of_2(N))
        if N > block_N:
            raise ValueError(f'N {N} must be <= block_N={block_N!r}')
        grid = lambda meta: (M,)
        _rms_norm_fwd_kernel[grid](x, x.stride(0), y, y.stride(0), weight,
            rstd, eps, M, N, block_N)
        ctx.eps = eps
        ctx.save_for_backward(x, weight, rstd)
        ctx.x_shape_start = x_shape_start
        y = y.reshape(x_shape_start)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        eps = ctx.eps
        x_shape_start = ctx.x_shape_start
        dy = dy.reshape(-1, dy.shape[-1])
        if dy.stride(-1) != 1:
            dy = dy.contiguous()
        M, N = dy.shape
        dx = torch.empty_like(x)
        dw = torch.empty_like(weight)
        sm_count = torch.cuda.get_device_properties(x.device
            ).multi_processor_count
        _dw = torch.empty((sm_count, N), dtype=torch.float32, device=weight
            .device)
        max_size = 65536 // x.element_size()
        block_N = min(max_size, triton.next_power_of_2(N))
        rows_per_sm = math.ceil(M / sm_count)
        if N > block_N:
            raise ValueError(f'N {N} must be <= block_N={block_N!r}')
        grid = lambda meta: (sm_count,)
        _rms_norm_bwd_kernel_sm[grid](x, x.stride(0), weight, dy, dy.stride
            (0), dx, dx.stride(0), rstd, _dw, eps, M, N, rows_per_sm, block_N)
        dw = _dw.sum(0).to(weight.dtype)
        dx = dx.reshape(x_shape_start)
        return dx, dw, None
