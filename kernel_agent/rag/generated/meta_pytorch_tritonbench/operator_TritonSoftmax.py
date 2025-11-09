# SPDX-License-Identifier: BSD-3-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/meta-pytorch/tritonbench
# Source-Files: tritonbench/operators/softmax/operator.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_ct7v_342/tritonbench-main/tritonbench/operators/softmax/operator.py
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
def softmax_kernel(output_ptr, input_ptr, input_row_stride,
    output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


# Forward method (kernel launch code)
def _TritonSoftmax_forward(ctx, x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    y = torch.empty_like(x)
    Operator.softmax_kernel[n_rows,](y, x, x.stride(0), y.stride(0), n_cols,
        num_warps=num_warps, BLOCK_SIZE=BLOCK_SIZE)
    ctx.save_for_backward(y)
    return y


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def softmax_bwd_kernel(softmax_output, grad_output, grad_input,
    grad_input_stride_0, grad_input_stride_1, grad_output_stride_0,
    grad_output_stride_1, softmax_output_stride_0, softmax_output_stride_1,
    m, n, BLOCK_SIZE_0: tl.constexpr, BLOCK_SIZE_1: tl.constexpr,
    BLOCK_SIZE_2: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < m
    sum_per_row = tl.full([BLOCK_SIZE_0], 0.0, tl.float32)
    for offset_1 in tl.range(0, n.to(tl.int32), BLOCK_SIZE_1):
        indices_1 = offset_1 + tl.arange(0, BLOCK_SIZE_1).to(tl.int32)
        mask_1 = indices_1 < n
        sum_per_row_copy = sum_per_row
        sum_per_row_copy_0 = sum_per_row_copy
        load = tl.load(softmax_output + (indices_0[:, None] *
            softmax_output_stride_0 + indices_1[None, :] *
            softmax_output_stride_1), mask_0[:, None] & mask_1[None, :],
            other=0)
        load_1 = tl.load(grad_output + (indices_0[:, None] *
            grad_output_stride_0 + indices_1[None, :] *
            grad_output_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
        v_0 = load * load_1
        sum_1 = tl.cast(tl.sum(v_0, 1), tl.float16)
        v_1 = tl.cast(sum_1, tl.float32)
        sum_per_row = sum_per_row_copy_0 + v_1
    for offset_2 in tl.range(0, n.to(tl.int32), BLOCK_SIZE_2):
        indices_2 = offset_2 + tl.arange(0, BLOCK_SIZE_2).to(tl.int32)
        mask_2 = indices_2 < n
        sum_per_row_copy_1 = sum_per_row
        sum_per_row_copy_1_0 = sum_per_row_copy_1
        load_2 = tl.load(softmax_output + (indices_0[:, None] *
            softmax_output_stride_0 + indices_2[None, :] *
            softmax_output_stride_1), mask_0[:, None] & mask_2[None, :],
            other=0)
        load_3 = tl.load(grad_output + (indices_0[:, None] *
            grad_output_stride_0 + indices_2[None, :] *
            grad_output_stride_1), mask_0[:, None] & mask_2[None, :], other=0)
        subscript = sum_per_row_copy_1_0[:, None]
        v_3 = tl.cast(load_3, tl.float32)
        v_4 = v_3 - subscript
        v_5 = tl.cast(load_2, tl.float32)
        v_6 = v_5 * v_4
        v_7 = tl.cast(v_6, tl.float16)
        tl.store(grad_input + (indices_0[:, None] * grad_input_stride_0 + 
            indices_2[None, :] * grad_input_stride_1), v_7, mask_0[:, None] &
            mask_2[None, :])


@staticmethod
def softmax_bwd_triton(grad_output, softmax_output):
    """
        Helion generated triton kernel for softmax backward pass
        PR: https://github.com/pytorch/helion/pull/744
        """
    m, n = grad_output.size()
    grad_input = torch.empty_like(grad_output)
    BLOCK_SIZE_0 = min(32, triton.next_power_of_2(m))
    BLOCK_SIZE_1 = triton.next_power_of_2(n)
    BLOCK_SIZE_2 = BLOCK_SIZE_1
    Operator.softmax_bwd_kernel[triton.cdiv(m, BLOCK_SIZE_0),](softmax_output,
        grad_output, grad_input, grad_input.stride(0), grad_input.stride(1),
        grad_output.stride(0), grad_output.stride(1), softmax_output.stride
        (0), softmax_output.stride(1), m, n, BLOCK_SIZE_0, BLOCK_SIZE_1,
        BLOCK_SIZE_2)
    return grad_input


# Backward method (kernel launch code)
def _TritonSoftmax_backward(ctx, grad_output):
    y, = ctx.saved_tensors
    return Operator.softmax_bwd_triton(grad_output, y)


# ============================================================
# autograd.Function Class Definition
# ============================================================

class TritonSoftmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        n_rows, n_cols = x.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK_SIZE >= 2048:
            num_warps = 8
        if BLOCK_SIZE >= 4096:
            num_warps = 16
        y = torch.empty_like(x)
        Operator.softmax_kernel[n_rows,](y, x, x.stride(0), y.stride(0),
            n_cols, num_warps=num_warps, BLOCK_SIZE=BLOCK_SIZE)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        return Operator.softmax_bwd_triton(grad_output, y)
