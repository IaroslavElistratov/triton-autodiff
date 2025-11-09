# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/FlagOpen/FlagGems
# Source-Files: src/flag_gems/runtime/backend/_metax/ops/tanh.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_kcv6dce2/FlagGems-master/src/flag_gems/runtime/backend/_metax/ops/tanh.py
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
from math import log

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@pointwise_dynamic(promotion_methods=[(0, 'INT_TO_FLOAT')])
@triton.jit
def tanh_forward(x):
    return _tanh(x.to(tl.float32))


# Forward method (kernel launch code)
def _Tanh_forward(ctx, A):
    logger.debug('METAX GEMS TANH FORWARD')
    if A.requires_grad is True:
        out = tanh_forward(A.to(torch.float32))
        ctx.save_for_backward(out)
        return out.to(A.dtype)
    else:
        out = tanh_forward(A)
        return out


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@pointwise_dynamic(promotion_methods=[(0, 'INT_TO_FLOAT')])
@triton.jit
def tanh_backward(y, dy):
    return dy * (1.0 - y * y)


@triton.jit
def tanh_backward_custom_kernel(x_ptr: tl.tensor, y_ptr: tl.tensor,
    output_ptr: tl.tensor, n_elements: int, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr)
    output = y * (1 - x * x)
    tl.store(output_ptr + offsets, output, mask=mask)


def tanh_backward_custom(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    tanh_backward_custom_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024
        )
    return output


# Backward method (kernel launch code)
def _Tanh_backward(ctx, out_grad):
    logger.debug('METAX GEMS TANH BACKWARD')
    out, = ctx.saved_tensors
    is_grad_stride_0 = True
    for i in range(len(out_grad.stride())):
        if out_grad.stride()[i] != 0:
            is_grad_stride_0 = False
            break
    if is_grad_stride_0 and out_grad.numel() % 1024 == 0:
        in_grad = tanh_backward_custom(out, out_grad)
        return in_grad
    in_grad = tanh_backward(out, out_grad)
    return in_grad


# ============================================================
# autograd.Function Class Definition
# ============================================================

class Tanh(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A):
        logger.debug('METAX GEMS TANH FORWARD')
        if A.requires_grad is True:
            out = tanh_forward(A.to(torch.float32))
            ctx.save_for_backward(out)
            return out.to(A.dtype)
        else:
            out = tanh_forward(A)
            return out

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug('METAX GEMS TANH BACKWARD')
        out, = ctx.saved_tensors
        is_grad_stride_0 = True
        for i in range(len(out_grad.stride())):
            if out_grad.stride()[i] != 0:
                is_grad_stride_0 = False
                break
        if is_grad_stride_0 and out_grad.numel() % 1024 == 0:
            in_grad = tanh_backward_custom(out, out_grad)
            return in_grad
        in_grad = tanh_backward(out, out_grad)
        return in_grad
