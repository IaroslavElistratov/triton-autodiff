# SPDX-License-Identifier: BSD-2-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/linkedin/Liger-Kernel
# Source-Files: src/liger_kernel/ops/poly_norm.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_d1vk6zhm/Liger-Kernel-main/src/liger_kernel/ops/poly_norm.py
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
def _poly_norm_forward_kernel(Y_ptr, Y_row_stride, X_ptr, X_row_stride,
    W_ptr, B_ptr, RSTD_ptr, RSTD_row_stride, n_cols, eps, BLOCK_SIZE: tl.
    constexpr):
    """
    PolyNorm formula:
        y = w₀·norm(x³) + w₁·norm(x²) + w₂·norm(x) + b
        where norm(u) = u / sqrt(mean(u²) + ε)

    Reference:
    1. https://github.com/BryceZhuo/PolyCom/
    2. https://arxiv.org/pdf/2411.03884

    Cache rstd values for backward pass
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride
    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0)
    w0 = tl.load(W_ptr + 0)
    w1 = tl.load(W_ptr + 1)
    w2 = tl.load(W_ptr + 2)
    b = tl.load(B_ptr)
    X_pow3 = X_row * X_row * X_row
    X_pow2 = X_row * X_row
    X_pow1 = X_row
    mean_square_3 = tl.sum(X_pow3 * X_pow3, axis=0) / n_cols
    rstd_3 = rsqrt(mean_square_3 + eps)
    norm_x3 = X_pow3 * rstd_3
    mean_square_2 = tl.sum(X_pow2 * X_pow2, axis=0) / n_cols
    rstd_2 = rsqrt(mean_square_2 + eps)
    norm_x2 = X_pow2 * rstd_2
    mean_square_1 = tl.sum(X_pow1 * X_pow1, axis=0) / n_cols
    rstd_1 = rsqrt(mean_square_1 + eps)
    norm_x1 = X_pow1 * rstd_1
    tl.store(RSTD_ptr + 0, rstd_3)
    tl.store(RSTD_ptr + 1, rstd_2)
    tl.store(RSTD_ptr + 2, rstd_1)
    Y_row = w0 * norm_x3 + w1 * norm_x2 + w2 * norm_x1 + b
    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)


def poly_norm_forward(X, W, B, eps=1e-06):
    """
    PolyNorm Forward Pass

    Args:
        X: input tensor of shape (*, H) where H is hidden dimension
        W: weight tensor of shape (3,) for [w0, w1, w2]
        B: bias scalar tensor
        eps: epsilon for numerical stability

    Returns:
        Y: output tensor of same shape as X
        X: reshaped input (for backward)
        RSTD: cached rstd values (for backward)
        BLOCK_SIZE: block size used
        num_warps: number of warps used
    """
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    RSTD = torch.empty((n_rows, 3), dtype=torch.float32, device=X.device)
    assert W.shape[0] == 3, 'Weight tensor must have shape (3,)'
    assert B.numel() == 1, 'Bias must be a scalar'
    kernel_args = {}
    if X.device.type == 'xpu':
        kernel_args['grf_mode'] = 'large'
    _poly_norm_forward_kernel[n_rows,](Y, Y.stride(0), X, X.stride(0), W, B,
        RSTD, RSTD.stride(0), n_cols, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps
        =num_warps, **kernel_args)
    return Y.view(*shape), X, RSTD, BLOCK_SIZE, num_warps


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


# Forward method (kernel launch code)
@ensure_contiguous
def _LigerPolyNormFunction_forward(ctx, X, W, B, eps=1e-06, in_place=True):
    """
        Args:
            X: input tensor of shape (B, T, H) or (BxT, H)
            W: weight tensor of shape (3,) for [w0, w1, w2]
            B: bias scalar
            eps: epsilon for numerical stability
            in_place: whether to in-place modify grad_output in backward (saves memory)

        Returns:
            Y: output tensor of same shape as X
        """
    Y, X, RSTD, BLOCK_SIZE, num_warps = poly_norm_forward(X, W, B, eps)
    ctx.BLOCK_SIZE = BLOCK_SIZE
    ctx.num_warps = num_warps
    ctx.in_place = in_place
    ctx.save_for_backward(X, W, RSTD)
    return Y


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _poly_norm_backward_kernel(dY_ptr, dY_row_stride, dX_ptr, dX_row_stride,
    X_ptr, X_row_stride, W_ptr, RSTD_ptr, RSTD_row_stride, dW_ptr,
    dW_row_stride, dB_ptr, n_rows, n_cols, rows_per_program: tl.constexpr,
    BLOCK_SIZE: tl.constexpr):
    """
    PolyNorm Backward Kernel Gradient:
        ∂L/∂x_i = Σ_p w_p * [p*x_i^(p-1) * grad_i/D_p - (p/d)*x_i^(2p-1) * S_p/(D_p³)]

    where:
        - D_p = RMS(x^p) = 1/rstd_p
        - S_p = sum(grad * x^p) over the row
        - d = n_cols
        - p ∈ {3, 2, 1}
    """
    row_block_id = tl.program_id(0).to(tl.int64)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dW0_acc = 0.0
    dW1_acc = 0.0
    dW2_acc = 0.0
    dB_acc = 0.0
    w0 = tl.load(W_ptr + 0).to(tl.float32)
    w1 = tl.load(W_ptr + 1).to(tl.float32)
    w2 = tl.load(W_ptr + 2).to(tl.float32)
    dY_ptr += row_start * dY_row_stride
    dX_ptr += row_start * dX_row_stride
    X_ptr += row_start * X_row_stride
    RSTD_ptr += row_start * RSTD_row_stride
    for _ in range(row_start, row_end):
        dY_row = tl.load(dY_ptr + col_offsets, mask=mask, other=0.0).to(tl.
            float32)
        X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0).to(tl.
            float32)
        rstd_3 = tl.load(RSTD_ptr + 0).to(tl.float32)
        rstd_2 = tl.load(RSTD_ptr + 1).to(tl.float32)
        rstd_1 = tl.load(RSTD_ptr + 2).to(tl.float32)
        X_pow3 = X_row * X_row * X_row
        X_pow2 = X_row * X_row
        X_pow1 = X_row
        dB_acc += tl.sum(dY_row, axis=0)
        S_3 = tl.sum(dY_row * X_pow3, axis=0)
        grad_x_3 = w0 * (3.0 * X_pow2 * rstd_3 * dY_row - 3.0 / n_cols *
            X_row * X_row * X_row * X_row * X_row * (rstd_3 * rstd_3 *
            rstd_3) * S_3)
        S_2 = tl.sum(dY_row * X_pow2, axis=0)
        grad_x_2 = w1 * (2.0 * X_row * rstd_2 * dY_row - 2.0 / n_cols *
            X_row * X_row * X_row * (rstd_2 * rstd_2 * rstd_2) * S_2)
        S_1 = tl.sum(dY_row * X_pow1, axis=0)
        grad_x_1 = w2 * (1.0 * rstd_1 * dY_row - 1.0 / n_cols * X_row * (
            rstd_1 * rstd_1 * rstd_1) * S_1)
        dW0_acc += rstd_3 * S_3
        dW1_acc += rstd_2 * S_2
        dW2_acc += rstd_1 * S_1
        dX_row = grad_x_3 + grad_x_2 + grad_x_1
        tl.store(dX_ptr + col_offsets, dX_row, mask=mask)
        dY_ptr += dY_row_stride
        dX_ptr += dX_row_stride
        X_ptr += X_row_stride
        RSTD_ptr += RSTD_row_stride
    tl.store(dW_ptr + row_block_id * dW_row_stride + 0, dW0_acc)
    tl.store(dW_ptr + row_block_id * dW_row_stride + 1, dW1_acc)
    tl.store(dW_ptr + row_block_id * dW_row_stride + 2, dW2_acc)
    tl.store(dB_ptr + row_block_id, dB_acc)


def poly_norm_backward(dY, X, W, RSTD, BLOCK_SIZE, num_warps, in_place):
    """
    PolyNorm Backward Pass

    Args:
        dY: gradient of output
        X: input tensor (already reshaped to 2D)
        W: weight tensor
        RSTD: cached rstd values from forward
        BLOCK_SIZE: block size from forward
        num_warps: number of warps from forward
        in_place: whether to in-place modify dY to store dX (saves memory)

    Returns:
        dX: gradient w.r.t. input
        dW: gradient w.r.t. weight
        dB: gradient w.r.t. bias
    """
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape
    import math
    sm_count = 1
    if X.device.type == 'cuda':
        sm_count = torch.cuda.get_device_properties(X.device
            ).multi_processor_count
    elif X.device.type == 'xpu':
        sm_count = torch.xpu.get_device_properties(X.device).gpu_eu_count
    if in_place is True:
        dX = dY
    else:
        dX = torch.zeros_like(dY)
    _dW = torch.empty((sm_count, 3), dtype=torch.float32, device=W.device)
    _dB = torch.empty((sm_count,), dtype=torch.float32, device=W.device)
    rows_per_program = math.ceil(n_rows / sm_count)
    grid = sm_count,
    kernel_args = {}
    if X.device.type == 'xpu':
        kernel_args['grf_mode'] = 'large'
    _poly_norm_backward_kernel[grid](dY, dY.stride(0), dX, dX.stride(0), X,
        X.stride(0), W, RSTD, RSTD.stride(0), _dW, _dW.stride(0), _dB,
        n_rows, n_cols, rows_per_program, BLOCK_SIZE=BLOCK_SIZE, num_warps=
        num_warps, **kernel_args)
    dX = dX.view(*shape)
    dW = _dW.sum(dim=0).to(W.dtype)
    dB = _dB.sum().to(W.dtype)
    return dX, dW, dB


# Backward method (kernel launch code)
@ensure_contiguous
def _LigerPolyNormFunction_backward(ctx, grad_output):
    """
        Args:
            grad_output: gradient of output

        Returns:
            dX, dW, dB: gradients w.r.t. X, W, B
        """
    X, W, RSTD = ctx.saved_tensors
    dX, dW, dB = poly_norm_backward(grad_output, X, W, RSTD, ctx.BLOCK_SIZE,
        ctx.num_warps, ctx.in_place)
    return dX, dW, dB, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LigerPolyNormFunction(torch.autograd.Function):
    """
    PolyNorm Function with forward and backward pass

    PolyNorm formula:
        y = w₀·norm(x³) + w₁·norm(x²) + w₂·norm(x) + b
        where norm(u) = u / sqrt(mean(u²) + ε)

    Backward uses closed-form gradient:
        ∂L/∂x_i = Σ_p w_p * [p*x_i^(p-1) * grad_i/D_p - (p/d)*x_i^(2p-1) * S_p/(D_p³)]
    """

    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, B, eps=1e-06, in_place=True):
        """
        Args:
            X: input tensor of shape (B, T, H) or (BxT, H)
            W: weight tensor of shape (3,) for [w0, w1, w2]
            B: bias scalar
            eps: epsilon for numerical stability
            in_place: whether to in-place modify grad_output in backward (saves memory)

        Returns:
            Y: output tensor of same shape as X
        """
        Y, X, RSTD, BLOCK_SIZE, num_warps = poly_norm_forward(X, W, B, eps)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.in_place = in_place
        ctx.save_for_backward(X, W, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output):
        """
        Args:
            grad_output: gradient of output

        Returns:
            dX, dW, dB: gradients w.r.t. X, W, B
        """
        X, W, RSTD = ctx.saved_tensors
        dX, dW, dB = poly_norm_backward(grad_output, X, W, RSTD, ctx.
            BLOCK_SIZE, ctx.num_warps, ctx.in_place)
        return dX, dW, dB, None, None
