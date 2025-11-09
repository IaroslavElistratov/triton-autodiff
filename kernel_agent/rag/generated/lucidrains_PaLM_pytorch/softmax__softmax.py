# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/lucidrains/PaLM-pytorch
# Source-Files: palm_pytorch/triton/softmax.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_9ct1uewg/PaLM-pytorch-main/palm_pytorch/triton/softmax.py
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
def softmax_kernel_forward(output_ptr, input_ptr, input_row_stride,
    output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    causal_mask = col_offsets > row_idx % n_cols
    row = row + tl.where(causal_mask, -float('inf'), 0.0)
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


# Forward method (kernel launch code)
def __softmax_forward(self, ctx, x):
    shape = x.shape
    x = x.view(-1, shape[-1])
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = calc_num_warps(BLOCK_SIZE)
    y = torch.empty_like(x)
    softmax_kernel_forward[n_rows,](y, x, x.stride(0), y.stride(0), n_cols,
        num_warps=num_warps, BLOCK_SIZE=BLOCK_SIZE)
    if x.requires_grad:
        ctx.save_for_backward(y)
    return y.view(*shape)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def softmax_kernel_backward(output_ptr, input_ptr, grad_ptr,
    grad_row_stride, input_row_stride, output_row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    grad_row_start_ptr = grad_ptr + row_idx * grad_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    grad_ptrs = grad_row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    probs_row = tl.load(input_ptrs, mask=mask, other=0.0)
    grad_row = tl.load(grad_ptrs, mask=mask, other=0.0)
    dxhat = probs_row * grad_row
    softmax_grad_output = dxhat - probs_row * tl.sum(dxhat, axis=0)
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_grad_output, mask=mask)


# Backward method (kernel launch code)
def __softmax_backward(self, ctx, grad_probs):
    shape = grad_probs.shape
    probs, = ctx.saved_tensors
    grad_probs = grad_probs.view(-1, grad_probs.shape[-1])
    n_rows, n_cols = grad_probs.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = calc_num_warps(BLOCK_SIZE)
    dx = torch.empty_like(probs)
    softmax_kernel_backward[n_rows,](dx, probs, grad_probs, grad_probs.
        stride(0), probs.stride(0), dx.stride(0), n_cols, num_warps=
        num_warps, BLOCK_SIZE=BLOCK_SIZE)
    return dx.view(*shape), None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _softmax(autograd.Function):

    @classmethod
    def forward(self, ctx, x):
        shape = x.shape
        x = x.view(-1, shape[-1])
        n_rows, n_cols = x.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = calc_num_warps(BLOCK_SIZE)
        y = torch.empty_like(x)
        softmax_kernel_forward[n_rows,](y, x, x.stride(0), y.stride(0),
            n_cols, num_warps=num_warps, BLOCK_SIZE=BLOCK_SIZE)
        if x.requires_grad:
            ctx.save_for_backward(y)
        return y.view(*shape)

    @classmethod
    def backward(self, ctx, grad_probs):
        shape = grad_probs.shape
        probs, = ctx.saved_tensors
        grad_probs = grad_probs.view(-1, grad_probs.shape[-1])
        n_rows, n_cols = grad_probs.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = calc_num_warps(BLOCK_SIZE)
        dx = torch.empty_like(probs)
        softmax_kernel_backward[n_rows,](dx, probs, grad_probs, grad_probs.
            stride(0), probs.stride(0), dx.stride(0), n_cols, num_warps=
            num_warps, BLOCK_SIZE=BLOCK_SIZE)
        return dx.view(*shape), None
