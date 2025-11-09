# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/kakaobrain/trident
# Source-Files: trident/operation/max.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rl_ay0ux/trident-main/trident/operation/max.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

def pop_trace():
    if config.use_trace:
        nvtx.pop_range(domain='Trident')


def push_trace(message: str):
    if config.use_trace:
        nvtx.push_range(message, color='green', domain='Trident')


def size_and_stride(input: torch.Tensor, dim: int):
    if input.dim() == 2:
        if dim == 0:
            x_size, y_size = input.shape
            y_stride = input.stride(1)
            x_stride = input.stride(0)
        else:
            y_size, x_size = input.shape
            y_stride = input.stride(0)
            x_stride = input.stride(1)
        return y_size, x_size, y_stride, x_stride
    elif input.dim() == 3:
        if dim == 0:
            z_size, y_size, x_size = input.shape[0], input.shape[1
                ], input.shape[2]
            z_stride, y_stride, x_stride = input.stride(0), input.stride(1
                ), input.stride(2)
        elif dim == 1:
            z_size, y_size, x_size = input.shape[1], input.shape[0
                ], input.shape[2]
            z_stride, y_stride, x_stride = input.stride(1), input.stride(0
                ), input.stride(2)
        else:
            z_size, y_size, x_size = input.shape[2], input.shape[0
                ], input.shape[1]
            z_stride, y_stride, x_stride = input.stride(2), input.stride(0
                ), input.stride(1)
        return z_size, y_size, x_size, z_stride, y_stride, x_stride
    else:
        raise ValueError(f'{dim} is not supported.')


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@staticmethod
@triton.jit
def forward(output_ptr: tl.tensor, argmax_ptr: tl.tensor, input_ptr: tl.
    tensor, y_size: tl.int32, x_size: tl.int32, y_stride: tl.int32,
    x_stride: tl.int32, x_block_size: tl.constexpr):
    y_offset = tl.program_id(0)
    output_block_ptr = tl.make_block_ptr(output_ptr, shape=(y_size,),
        strides=(1,), offsets=(y_offset,), block_shape=(1,), order=(0,))
    argmax_block_ptr = tl.make_block_ptr(argmax_ptr, shape=(y_size,),
        strides=(1,), offsets=(y_offset,), block_shape=(1,), order=(0,))
    input_block_ptr = tl.make_block_ptr(input_ptr, shape=(y_size, x_size),
        strides=(y_stride, x_stride), offsets=(y_offset, 0), block_shape=(1,
        x_block_size), order=(1, 0))
    input = tl.load(input_block_ptr, boundary_check=(1,))
    argmax = tl.argmax(input, 1)
    output = tl.load(input_ptr + y_offset * y_stride + argmax * x_stride)
    tl.store(output_block_ptr, output)
    tl.store(argmax_block_ptr, argmax.to(tl.int64))


# Forward method (kernel launch code)
def _Max_forward(ctx: Any, *args: Any, **kwargs: Any):
    input, dim = args
    util.push_trace('Max.__forward')
    output, argmax = Max.__forward(input, dim)
    util.pop_trace()
    ctx.save_for_backward(input, output, argmax)
    ctx.dim = dim
    return output, argmax


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@staticmethod
@triton.jit
def backward(grad_input_ptr: tl.tensor, grad_output_ptr: tl.tensor,
    argmax_ptr: tl.tensor, y_size: tl.constexpr, x_size: tl.constexpr,
    y_stride: tl.constexpr, x_stride: tl.constexpr, x_block_size: tl.constexpr
    ):
    y_offset = tl.program_id(0)
    grad_input_block_ptr = tl.make_block_ptr(grad_input_ptr, shape=(y_size,
        x_size), strides=(y_stride, x_stride), offsets=(y_offset, 0),
        block_shape=(1, x_block_size), order=(1, 0))
    grad_output_block_ptr = tl.make_block_ptr(grad_output_ptr, shape=(
        y_size,), strides=(1,), offsets=(y_offset,), block_shape=(1,),
        order=(0,))
    argmax_block_ptr = tl.make_block_ptr(argmax_ptr, shape=(1, y_size),
        strides=(y_size, 1), offsets=(0, y_offset), block_shape=(1, 1),
        order=(1, 0))
    grad_output = tl.load(grad_output_block_ptr)
    argmax = tl.load(argmax_block_ptr)
    condition = tl.arange(0, x_block_size) == argmax
    grad_input = tl.where(condition, grad_output, 0)
    tl.store(grad_input_block_ptr, grad_input, boundary_check=(1,))


# Backward method (kernel launch code)
def _Max_backward(ctx: Any, *grad_outputs: Any):
    grad_output, grad_argmax = grad_outputs
    input, output, argmax = ctx.saved_tensors
    util.push_trace('Max.__forward')
    grad_input = Max.__backward(grad_output, input, argmax, ctx.dim)
    util.pop_trace()
    return grad_input, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class Max(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, dim = args
        util.push_trace('Max.__forward')
        output, argmax = Max.__forward(input, dim)
        util.pop_trace()
        ctx.save_for_backward(input, output, argmax)
        ctx.dim = dim
        return output, argmax

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        grad_output, grad_argmax = grad_outputs
        input, output, argmax = ctx.saved_tensors
        util.push_trace('Max.__forward')
        grad_input = Max.__backward(grad_output, input, argmax, ctx.dim)
        util.pop_trace()
        return grad_input, None, None

    @staticmethod
    def __forward(input: torch.Tensor, dim: torch.int32):
        factory_kwargs = {'device': input.device}
        y_size, x_size, y_stride, x_stride = util.size_and_stride(input, dim)
        output = torch.empty(y_size, **factory_kwargs, dtype=input.dtype)
        argmax = torch.empty(y_size, **factory_kwargs, dtype=torch.int64)

        def grid(meta):
            return y_size,
        util.push_trace('kernel.Max.forward')
        kernel.Max.forward[grid](output, argmax, input, y_size, x_size,
            y_stride, x_stride, triton.next_power_of_2(x_size))
        util.pop_trace()
        return output, argmax

    @staticmethod
    def __backward(grad_output: torch.Tensor, input: torch.Tensor, argmax:
        torch.Tensor, dim: torch.int32):
        y_size, x_size, y_stride, x_stride = util.size_and_stride(input, dim)
        grad_input = torch.zeros_like(input)

        def grid(meta):
            return y_size,
        util.push_trace('kernel.Max.backward')
        kernel.Max.backward[grid](grad_input, grad_output, argmax, y_size,
            x_size, y_stride, x_stride, triton.next_power_of_2(x_size))
        util.pop_trace()
        return grad_input
