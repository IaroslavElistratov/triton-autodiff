# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/ROCm/aiter
# Source-Files: aiter/ops/triton/rmsnorm.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_ciw0cknm/aiter-main/aiter/ops/triton/rmsnorm.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

def block_size(x):
    return min(65536 // x.element_size(), triton.next_power_of_2(x.shape[1]))


def num_programs(x):
    return min(x.shape[0], get_num_sms())


def use_blocked(x):
    return x.shape[1] > block_size(x)


def get_num_sms():
    current_device_index = torch.cuda.current_device()
    current_device = torch.cuda.get_device_properties(current_device_index)
    num_sms = current_device.multi_processor_count
    return num_sms


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def _rms_norm_kernel(input_ptr, output_ptr, g_ptr, rsigma_ptr,
    input_row_stride, output_row_stride, n_rows, n_cols, epsilon,
    BLOCK_SIZE: tl.constexpr, USE_BLOCKED: tl.constexpr, NUM_PRGMS: tl.
    constexpr):
    """
    Note: this is Triton jited function and not meant to be called directly. Call rms_norm function
    below.

    Applies Root Mean Square Layer Normalization over a mini-batch of inputs.

    Key parameters:
    - Input: The input tensor to be normalized with shape (n_rows, n_cols).
    - Output: The output tensor with shape (n_rows, n_cols).
    - G: The learnable weights tensor with shape (n_cols, ).
    """
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    if USE_BLOCKED:
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=1):
            row_input_ptr = input_ptr + row_idx * input_row_stride
            row_output_ptr = output_ptr + row_idx * output_row_stride
            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            sum_squares = 0.0
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                x = tl.load(input_ptrs).to(tl.float32)
                sum_squares += tl.sum(x * x, axis=0)
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier='.cg'
                ).to(tl.float32)
            sum_squares += tl.sum(x * x, axis=0)
            mean_square = sum_squares / n_cols
            norm_factor = tl.rsqrt(mean_square + epsilon)
            tl.store(rsigma_ptr + row_idx, norm_factor)
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                x = tl.load(input_ptrs).to(tl.float32)
                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                rms_norm = x * norm_factor * g
                output_ptrs = row_output_ptr + cols
                tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty))
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier='.cg'
                ).to(tl.float32)
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            rms_norm = x * norm_factor * g
            output_ptrs = row_output_ptr + cols
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty),
                mask=mask)
    else:
        mask = col_offsets < n_cols
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
            input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            row = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=
                '.cg').to(tl.float32)
            g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.
                float32)
            row_norm = row * row
            row_norm = tl.sum(row_norm, axis=-1)
            norm_factor = tl.math.rsqrt(row_norm / n_cols + epsilon)
            tl.store(rsigma_ptr + row_idx, norm_factor)
            rms_norm = row * norm_factor * g
            output_ptrs = (output_ptr + row_idx * output_row_stride +
                col_offsets)
            output_ptrs = tl.multiple_of(output_ptrs, (16,))
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty),
                mask=mask)


def _rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, epsilon: float):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    rsigma = torch.empty((n_rows,), dtype=torch.float32, device=x.device)
    blk_size = block_size(x)
    USE_BLOCKED = use_blocked(x)
    NUM_PRGMS = num_programs(x)
    grid = lambda meta: (NUM_PRGMS,)
    _rms_norm_kernel[grid](x, y, weight, rsigma, x.stride(0), y.stride(0),
        n_rows, n_cols, epsilon, blk_size, USE_BLOCKED, NUM_PRGMS)
    return y, rsigma


# Forward method (kernel launch code)
def __RMSNorm_forward(ctx, x, weight, epsilon, is_grad_enabled):
    is_grad = is_grad_enabled and any(tensor.requires_grad for tensor in [x,
        weight])
    y, rsigma = _rmsnorm_forward(x, weight, epsilon)
    if is_grad:
        ctx.save_for_backward(x, weight, rsigma)
    return y


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _rmsnorm_bwd_dg_reduce_triton(dg_in_ptr, dg_out_ptr, dg_in_stride,
    n_rows, n_cols, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, n_rows, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)
        offs = rows[:, None] * n_cols + cols[None, :]
        acc += tl.load(dg_in_ptr + offs, mask=mask, other=0.0,
            cache_modifier='.cg').to(tl.float32)
    sum_dg = tl.sum(acc, axis=0)
    tl.store(dg_out_ptr + cols, sum_dg.to(dg_out_ptr.type.element_ty), mask
        =cols < n_cols)


@triton.jit
def _rmsnorm_bwd_triton(grad_output_ptr, input_ptr, g_ptr, rsigma_ptr,
    dx_ptr, dg_ptr, input_row_stride, output_row_stride, n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr, USE_BLOCKED: tl.constexpr, NUM_PRGMS: tl.
    constexpr):
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    if USE_BLOCKED:
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=1):
            row_input_ptr = input_ptr + row_idx * input_row_stride
            row_grad_output_ptr = grad_output_ptr + row_idx * output_row_stride
            row_dx_ptr = dx_ptr + row_idx * input_row_stride
            row_dg_ptr = dg_ptr + row_idx * input_row_stride
            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            grad_sum = 0.0
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                grad_output_ptrs = row_grad_output_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                grad_output_ptrs = tl.multiple_of(grad_output_ptrs, (16,))
                x = tl.load(input_ptrs).to(tl.float32)
                grad_output = tl.load(grad_output_ptrs).to(tl.float32)
                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                grad_sum += tl.sum(grad_output * x * g, axis=0)
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
            grad_output_ptrs = row_grad_output_ptr + cols
            grad_output = tl.load(grad_output_ptrs, mask=mask, other=0.0).to(tl
                .float32)
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            grad_sum += tl.sum(grad_output * x * g, axis=0)
            norm_factor = tl.load(rsigma_ptr + row_idx).to(tl.float32)
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                grad_output_ptrs = row_grad_output_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                grad_output_ptrs = tl.multiple_of(grad_output_ptrs, (16,))
                x = tl.load(input_ptrs).to(tl.float32)
                grad_output = tl.load(grad_output_ptrs).to(tl.float32)
                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                grad_input = (grad_output * norm_factor * g - norm_factor *
                    norm_factor * norm_factor * x * (grad_sum / n_cols))
                dx_ptrs = row_dx_ptr + cols
                tl.store(dx_ptrs, grad_input.to(dx_ptr.type.element_ty))
                dg = grad_output * x * norm_factor
                dg_ptrs = row_dg_ptr + cols
                tl.store(dg_ptrs, dg.to(tl.float32))
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
            grad_output_ptrs = row_grad_output_ptr + cols
            grad_output = tl.load(grad_output_ptrs, mask=mask, other=0.0).to(tl
                .float32)
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            grad_input = (grad_output * norm_factor * g - norm_factor *
                norm_factor * norm_factor * x * (grad_sum / n_cols))
            dx_ptrs = row_dx_ptr + cols
            tl.store(dx_ptrs, grad_input.to(dx_ptr.type.element_ty), mask=mask)
            dg = grad_output * x * norm_factor
            dg_ptrs = row_dg_ptr + cols
            tl.store(dg_ptrs, dg.to(tl.float32), mask=mask)
    else:
        mask = col_offsets < n_cols
        dg_col_redux = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
            input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
            grad_output_ptrs = (grad_output_ptr + row_idx *
                output_row_stride + col_offsets)
            dx_ptrs = dx_ptr + row_idx * input_row_stride + col_offsets
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            grad_output_ptrs = tl.multiple_of(grad_output_ptrs, (16,))
            dx_ptrs = tl.multiple_of(dx_ptrs, (16,))
            x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
            grad_output = tl.load(grad_output_ptrs, mask=mask, other=0.0).to(tl
                .float32)
            g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.
                float32)
            norm_factor = tl.load(rsigma_ptr + row_idx).to(tl.float32)
            grad_sum = tl.sum(grad_output * x * g, axis=0)
            grad_input = (grad_output * norm_factor * g - norm_factor *
                norm_factor * norm_factor * x * (grad_sum / n_cols))
            tl.store(dx_ptrs, grad_input.to(dx_ptr.type.element_ty), mask=mask)
            dg = grad_output * x * norm_factor
            dg_col_redux += dg.to(tl.float32)
        tl.store(dg_ptr + tl.program_id(0) * input_row_stride + col_offsets,
            dg_col_redux, mask=mask)


def _rmsnorm_backward(dz, x, gamma, rsigma):
    dz_ = dz.contiguous()
    x_ = x.contiguous()
    gamma_ = gamma.contiguous()
    rsigma_ = rsigma.contiguous()
    dx = torch.empty_like(x_)
    dgamma = torch.empty_like(gamma_)
    M, N = x_.shape
    blk_size = block_size(x_)
    USE_BLOCKED = use_blocked(x_)
    NUM_PRGMS = num_programs(x_)
    need_reduction = N > 1
    dg_tmp = torch.empty(dg_tmp_rows(x_), N, device='cuda', dtype=torch.
        float32, requires_grad=False) if need_reduction else None
    grid_bwd = lambda meta: (NUM_PRGMS,)
    _rmsnorm_bwd_triton[grid_bwd](dz_, x_, gamma_, rsigma_, dx, dg_tmp if
        need_reduction else dgamma, x_.stride(0), dz_.stride(0), M, N,
        blk_size, USE_BLOCKED, NUM_PRGMS, num_warps=8)
    if need_reduction:
        grid_reduce = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
        _rmsnorm_bwd_dg_reduce_triton[grid_reduce](dg_tmp, dgamma, dg_tmp.
            stride(0), dg_tmp.shape[0], dg_tmp.shape[1], BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=64)
    return dx, dgamma


def dg_tmp_rows(x):
    return x.shape[0] if use_blocked(x) else num_programs(x)


# Backward method (kernel launch code)
def __RMSNorm_backward(ctx, grad_output):
    x, weight, rsigma = ctx.saved_tensors
    dx, dg = _rmsnorm_backward(grad_output, x, weight, rsigma)
    return dx, dg, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _RMSNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, epsilon, is_grad_enabled):
        is_grad = is_grad_enabled and any(tensor.requires_grad for tensor in
            [x, weight])
        y, rsigma = _rmsnorm_forward(x, weight, epsilon)
        if is_grad:
            ctx.save_for_backward(x, weight, rsigma)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, rsigma = ctx.saved_tensors
        dx, dg = _rmsnorm_backward(grad_output, x, weight, rsigma)
        return dx, dg, None, None
