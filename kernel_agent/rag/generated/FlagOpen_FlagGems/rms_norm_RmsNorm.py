# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/FlagOpen/FlagGems
# Source-Files: src/flag_gems/ops/rms_norm.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_kcv6dce2/FlagGems-master/src/flag_gems/ops/rms_norm.py
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
from functools import partial

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@libentry()
@triton.jit(do_not_specialize=['eps'])
def rms_norm_kernel(out_ptr, INV_RMS, in_ptr, w_ptr, y_stride_r, y_stride_c,
    x_stride_r, x_stride_c, N, eps, BLOCK_SIZE: tl.constexpr):
    if tl.constexpr(in_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        in_ptr.dtype.element_ty == tl.bfloat16):
        cdtype = tl.float32
    else:
        cdtype = in_ptr.dtype.element_ty
    pid = tl.program_id(0)
    out_ptr += pid * y_stride_r
    in_ptr += pid * x_stride_r
    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(in_ptr + cols * x_stride_c, mask, other=0.0).to(cdtype)
    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms * w).to(cdtype)
    tl.store(out_ptr + cols * y_stride_c, y, mask=mask)
    tl.store(INV_RMS + pid, rrms)


# Forward method (kernel launch code)
def _RmsNorm_forward(ctx, x, normalized_shape, weight, eps=1e-05):
    logger.debug('GEMS LAYERNORM FORWARD')
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)
    BLOCK_SIZE = triton.next_power_of_2(N)
    x = x.contiguous()
    weight = weight.contiguous()
    y = torch.empty_like(x)
    inv_rms = torch.empty((M,), device=x.device, dtype=torch.float32)
    with torch_device_fn.device(x.device):
        rms_norm_kernel[M,](y, inv_rms, x, weight, N, 1, N, 1, N, eps,
            BLOCK_SIZE)
    ctx.save_for_backward(x, inv_rms, weight)
    ctx.normalized_shape = normalized_shape
    ctx.eps = eps
    return y


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@libentry()
@triton.jit
def rms_norm_grad_dw_kernel(X, DY, INV_RMS, DW, dx_stride_r, dx_stride_c,
    x_stride_r, x_stride_c, M, N, ROW_BLOCK_SIZE: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)
    row_start = row_pid * ROW_BLOCK_SIZE
    col_start = col_pid * COL_BLOCK_SIZE
    offset = row_start * x_stride_r + col_start * x_stride_c
    X += offset
    DY += offset
    INV_RMS += row_start
    rows = tl.arange(0, ROW_BLOCK_SIZE)
    cols = tl.arange(0, COL_BLOCK_SIZE)
    row_mask = row_start + rows < M
    col_mask = col_start + cols < N
    x = tl.load(X + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c,
        row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
    inv_rms = tl.load(INV_RMS + rows, row_mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + rows[:, None] * x_stride_r + cols[None, :] *
        x_stride_c, row_mask[:, None] & col_mask[None, :], other=0.0).to(tl
        .float32)
    d_weight = x * dy * inv_rms[:, None]
    partial_dweight_sum = tl.sum(d_weight, axis=0)
    tl.store(DW + row_pid * N + col_start + cols, partial_dweight_sum, mask
        =col_mask)


@libentry()
@triton.jit(do_not_specialize=['eps'])
def rms_norm_grad_dx_kernel(X, DY, INV_RMS, DX, W, dx_stride_r, dx_stride_c,
    x_stride_r, x_stride_c, N, eps, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0)
    DX += pid * dx_stride_r
    X += pid * x_stride_r
    DY += pid * x_stride_r
    INV_RMS += pid
    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    inv_rms = tl.load(INV_RMS).to(tl.float32)
    dy = tl.load(DY + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    dy = dy * w
    normalized_buf = x * inv_rms
    row_sum_stats = tl.sum(normalized_buf * dy, axis=0)
    norm_val = normalized_buf / N
    dx = (dy - norm_val * row_sum_stats) * inv_rms
    tl.store(DX + cols * dx_stride_c, dx, mask=mask)


@triton.jit
def program_id(axis: int) ->tl.tensor:
    return tl.program_id(axis).to(tl.int64)


# Backward method (kernel launch code)
def _RmsNorm_backward(ctx, dy):
    logger.debug('GEMS LAYERNORM BACKWARD')
    x, inv_rms, weight = ctx.saved_tensors
    normalized_shape = ctx.normalized_shape
    eps = ctx.eps
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)
    BLOCK_SIZE = triton.next_power_of_2(N)
    x = x.contiguous()
    weight = weight.contiguous()
    dx = torch.empty_like(x)
    with torch_device_fn.device(x.device):
        rms_norm_grad_dx_kernel[M,](x, dy, inv_rms, dx, weight, N, 1, N, 1,
            N, eps, BLOCK_SIZE)
    ROW_BLOCK_SIZE = 16
    COL_BLOCK_SIZE = 256
    row_block_num = triton.cdiv(M, ROW_BLOCK_SIZE)
    col_block_num = triton.cdiv(N, COL_BLOCK_SIZE)
    partial_buffer = torch.empty((row_block_num, N), dtype=torch.float32,
        device=x.device)
    with torch_device_fn.device(x.device):
        rms_norm_grad_dw_kernel[row_block_num, col_block_num](x, dy,
            inv_rms, partial_buffer, N, 1, N, 1, M, N, ROW_BLOCK_SIZE,
            COL_BLOCK_SIZE)
        dw = torch.sum(partial_buffer, dim=0, dtype=x.dtype).reshape(-1)
    return dx, None, dw, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class RmsNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps=1e-05):
        logger.debug('GEMS LAYERNORM FORWARD')
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)
        BLOCK_SIZE = triton.next_power_of_2(N)
        x = x.contiguous()
        weight = weight.contiguous()
        y = torch.empty_like(x)
        inv_rms = torch.empty((M,), device=x.device, dtype=torch.float32)
        with torch_device_fn.device(x.device):
            rms_norm_kernel[M,](y, inv_rms, x, weight, N, 1, N, 1, N, eps,
                BLOCK_SIZE)
        ctx.save_for_backward(x, inv_rms, weight)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        logger.debug('GEMS LAYERNORM BACKWARD')
        x, inv_rms, weight = ctx.saved_tensors
        normalized_shape = ctx.normalized_shape
        eps = ctx.eps
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)
        BLOCK_SIZE = triton.next_power_of_2(N)
        x = x.contiguous()
        weight = weight.contiguous()
        dx = torch.empty_like(x)
        with torch_device_fn.device(x.device):
            rms_norm_grad_dx_kernel[M,](x, dy, inv_rms, dx, weight, N, 1, N,
                1, N, eps, BLOCK_SIZE)
        ROW_BLOCK_SIZE = 16
        COL_BLOCK_SIZE = 256
        row_block_num = triton.cdiv(M, ROW_BLOCK_SIZE)
        col_block_num = triton.cdiv(N, COL_BLOCK_SIZE)
        partial_buffer = torch.empty((row_block_num, N), dtype=torch.
            float32, device=x.device)
        with torch_device_fn.device(x.device):
            rms_norm_grad_dw_kernel[row_block_num, col_block_num](x, dy,
                inv_rms, partial_buffer, N, 1, N, 1, M, N, ROW_BLOCK_SIZE,
                COL_BLOCK_SIZE)
            dw = torch.sum(partial_buffer, dim=0, dtype=x.dtype).reshape(-1)
        return dx, None, dw, None
