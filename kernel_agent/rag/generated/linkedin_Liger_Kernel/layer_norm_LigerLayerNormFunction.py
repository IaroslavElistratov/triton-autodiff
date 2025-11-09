# SPDX-License-Identifier: BSD-2-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/linkedin/Liger-Kernel
# Source-Files: src/liger_kernel/ops/layer_norm.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_d1vk6zhm/Liger-Kernel-main/src/liger_kernel/ops/layer_norm.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

def calculate_settings(n):
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f'Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}.'
            )
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32 if not is_hip() else 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


def is_hip() ->bool:
    return torch.version.hip is not None


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def _layer_norm_forward_kernel(Y_ptr, Y_row_stride, X_ptr, X_row_stride,
    W_ptr, W_row_stride, B_ptr, B_row_stride, Mean_ptr, Mean_row_stride,
    RSTD_ptr, RSTD_row_stride, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    """
    References:
    https://arxiv.org/abs/1607.06450
    https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0.0)
    W_f32 = W_row.to(tl.float32)
    B_f32 = B_row.to(tl.float32)
    row_X_ptr = X_ptr + row_idx * X_row_stride
    row_Y_ptr = Y_ptr + row_idx * Y_row_stride
    row_Mean_ptr = Mean_ptr + row_idx * Mean_row_stride
    row_RSTD_ptr = RSTD_ptr + row_idx * RSTD_row_stride
    X_row = tl.load(row_X_ptr + col_offsets, mask=mask, other=0.0)
    X_f32 = X_row.to(tl.float32)
    mean = tl.sum(X_f32, axis=0) / n_cols
    X_centered = X_f32 - mean
    X_centered_masked = tl.where(mask, X_centered, 0.0)
    var = tl.sum(X_centered_masked * X_centered_masked, axis=0) / n_cols
    rstd = rsqrt(var + eps)
    tl.store(row_Mean_ptr, mean.to(X_row.dtype))
    tl.store(row_RSTD_ptr, rstd.to(X_row.dtype))
    Y_f32 = X_centered * rstd * W_f32 + B_f32
    tl.store(row_Y_ptr + col_offsets, Y_f32.to(X_row.dtype), mask=mask)


def layer_norm_forward(X, W, B, eps):
    """
    Args:
        X: Input tensor of shape (..., hidden_size)
        W: Weight tensor of shape (hidden_size,)
        B: Bias tensor of shape (hidden_size,)
        eps: Small constant for numerical stability

    Returns:
        Tuple of (output, input, mean, rstd, block_size, num_warps)
    """
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    Mean = torch.empty(n_rows, dtype=X.dtype, device=X.device)
    RSTD = torch.empty(n_rows, dtype=X.dtype, device=X.device)
    if X.shape[1] != W.shape[0]:
        raise ValueError(
            f'Incompatible dimensions: input feature size (X.shape[1]={X.shape[1]}) must match weight size (W.shape[0]={W.shape[0]})'
            )
    kernel_args = {}
    if X.device.type == 'xpu':
        kernel_args['grf_mode'] = 'large'
    grid = n_rows,
    _layer_norm_forward_kernel[grid](Y, Y.stride(0), X, X.stride(0), W, W.
        stride(0), B, B.stride(0), Mean, Mean.stride(0), RSTD, RSTD.stride(
        0), n_cols, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, **
        kernel_args)
    return Y.view(*shape), X, Mean, RSTD, BLOCK_SIZE, num_warps


# Forward method (kernel launch code)
@ensure_contiguous
def _LigerLayerNormFunction_forward(ctx, X, W, B, eps):
    Y, X, Mean, RSTD, BLOCK_SIZE, num_warps = layer_norm_forward(X, W, B, eps)
    ctx.save_for_backward(X, W, B, Mean, RSTD)
    return Y


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _layer_norm_backward_kernel(X_ptr, W_ptr, Mean_ptr, RSTD_ptr, DX_ptr,
    DW_ptr, DB_ptr, DY_ptr, stride_x, stride_dx, stride_dy, n_cols,
    BLOCK_SIZE: tl.constexpr, dtype: tl.constexpr, atomic_dtype: tl.constexpr):
    """
    References:
    https://arxiv.org/abs/1607.06450
    https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
    """
    row_idx = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    w_f32 = w.to(tl.float32)
    row_X_ptr = X_ptr + row_idx * stride_x
    row_DX_ptr = DX_ptr + row_idx * stride_dx
    row_DY_ptr = DY_ptr + row_idx * stride_dy
    row_Mean_ptr = Mean_ptr + row_idx
    row_RSTD_ptr = RSTD_ptr + row_idx
    x = tl.load(row_X_ptr + cols, mask=mask, other=0.0)
    dy = tl.load(row_DY_ptr + cols, mask=mask, other=0.0)
    mean = tl.load(row_Mean_ptr)
    rstd = tl.load(row_RSTD_ptr)
    x_f32 = x.to(tl.float32)
    dy_f32 = dy.to(tl.float32)
    mean_f32 = mean.to(tl.float32)
    rstd_f32 = rstd.to(tl.float32)
    x_hat = (x_f32 - mean_f32) * rstd_f32
    wdy = w_f32 * dy_f32
    c1 = tl.sum(x_hat * wdy, axis=0) / n_cols
    c2 = tl.sum(wdy, axis=0) / n_cols
    dx = (wdy - (x_hat * c1 + c2)) * rstd_f32
    tl.store(row_DX_ptr + cols, dx.to(dtype), mask=mask)
    dw = dy_f32 * x_hat
    db = dy_f32
    tl.atomic_add(DW_ptr + cols, dw.to(atomic_dtype), mask=mask)
    tl.atomic_add(DB_ptr + cols, db.to(atomic_dtype), mask=mask)


def layer_norm_backward(dY, X, W, B, Mean, RSTD):
    """
    Args:
        dY: Gradient of output
        X: Input tensor
        W: Weight tensor
        B: Bias tensor
        Mean: Pre-computed mean
        RSTD: Pre-computed reciprocal standard deviation

    Returns:
        Tuple of (input_grad, weight_grad, bias_grad)
    """
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape
    DX = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    grad_dtype = torch.float32 if W.dtype == torch.bfloat16 else W.dtype
    DW = torch.zeros(n_cols, dtype=grad_dtype, device=W.device)
    DB = torch.zeros(n_cols, dtype=grad_dtype, device=W.device)
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    if n_cols > BLOCK_SIZE:
        raise RuntimeError(
            f'Feature dimension {n_cols} exceeds maximum supported size of {BLOCK_SIZE}.'
            )
    triton_dtype = (tl.float32 if X.dtype == torch.float32 else tl.bfloat16 if
        X.dtype == torch.bfloat16 else tl.float16 if X.dtype == torch.
        float16 else tl.float32)
    atomic_dtype = tl.float32 if triton_dtype == tl.bfloat16 else triton_dtype
    kernel_args = {'num_warps': num_warps}
    if X.device.type == 'xpu':
        kernel_args.update({'grf_mode': 'large', 'num_warps': 32,
            'num_stages': 4})
    grid = n_rows,
    _layer_norm_backward_kernel[grid](X, W, Mean, RSTD, DX, DW, DB, dY, X.
        stride(0), DX.stride(0), dY.stride(0), n_cols, BLOCK_SIZE=
        BLOCK_SIZE, dtype=triton_dtype, atomic_dtype=atomic_dtype, **
        kernel_args)
    DX = DX.view(*shape)
    return DX, DW.to(W.dtype), DB.to(W.dtype)


# Backward method (kernel launch code)
@ensure_contiguous
def _LigerLayerNormFunction_backward(ctx, dY):
    X, W, B, Mean, RSTD = ctx.saved_tensors
    DX, DW, DB = layer_norm_backward(dY, X, W, B, Mean, RSTD)
    return DX, DW, DB, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LigerLayerNormFunction(torch.autograd.Function):

    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, B, eps):
        Y, X, Mean, RSTD, BLOCK_SIZE, num_warps = layer_norm_forward(X, W,
            B, eps)
        ctx.save_for_backward(X, W, B, Mean, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        X, W, B, Mean, RSTD = ctx.saved_tensors
        DX, DW, DB = layer_norm_backward(dY, X, W, B, Mean, RSTD)
        return DX, DW, DB, None
