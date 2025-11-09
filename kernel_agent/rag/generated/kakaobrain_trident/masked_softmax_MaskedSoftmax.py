# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/kakaobrain/trident
# Source-Files: trident/operation/masked_softmax.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rl_ay0ux/trident-main/trident/operation/masked_softmax.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

def dtype(input):
    if input == torch.float32:
        return tl.float32
    elif input == torch.float16:
        return tl.float16
    elif input == torch.bfloat16:
        return tl.bfloat16
    elif input == torch.int64:
        return tl.int64
    else:
        raise ValueError(f"Unable to convert the given input: '{input}'.")


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
@triton.heuristics({'require_x_boundary_check': lambda args: args['x_size'] %
    args['x_block_size']})
@triton.jit
def forward(output_ptr: tl.tensor, input_ptr: tl.tensor, y_size: tl.int32,
    x_size: tl.int32, y_stride: tl.int32, x_stride: tl.int32, mask_ptr: tl.
    tensor, dtype: tl.constexpr, x_block_size: tl.constexpr,
    require_x_boundary_check: tl.constexpr):
    y_offset = tl.program_id(0)
    output_block_ptr = tl.make_block_ptr(output_ptr, shape=(y_size, x_size),
        strides=(y_stride, x_stride), offsets=(y_offset, 0), block_shape=(1,
        x_block_size), order=(1, 0))
    input_block_ptr = tl.make_block_ptr(input_ptr, shape=(y_size, x_size),
        strides=(y_stride, x_stride), offsets=(y_offset, 0), block_shape=(1,
        x_block_size), order=(1, 0))
    mask_block_ptr = tl.make_block_ptr(mask_ptr, shape=(y_size, x_size),
        strides=(y_stride, x_stride), offsets=(y_offset, 0), block_shape=(1,
        x_block_size), order=(1, 0))
    if require_x_boundary_check:
        input = tl.load(input_block_ptr, boundary_check=(1,))
        mask = tl.load(mask_block_ptr, boundary_check=(1,))
        condition = tl.arange(0, x_block_size) < x_size
        mask = tl.where(condition, mask, 1)
    else:
        input = tl.load(input_block_ptr)
        mask = tl.load(mask_block_ptr)
    input = tl.where(mask > language.eps, float('-inf'), input)
    max = tl.max(input, 1)
    numerator = tl.math.fast_expf(input - max)
    output = numerator / tl.sum(numerator)
    if require_x_boundary_check:
        tl.store(output_block_ptr, output.to(dtype), boundary_check=(1,))
    else:
        tl.store(output_block_ptr, output.to(dtype))


# Forward method (kernel launch code)
def _MaskedSoftmax_forward(ctx: Any, *args: Any, **kwargs: Any):
    input, mask, dim = args
    util.push_trace('MaskedSoftmax.__forward')
    output = MaskedSoftmax.__forward(input, mask, dim)
    util.pop_trace()
    ctx.save_for_backward(output)
    ctx.dim = dim
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@staticmethod
@triton.heuristics({'require_x_boundary_check': lambda args: args['x_size'] %
    args['x_block_size']})
@triton.jit
def backward(grad_input_ptr: tl.tensor, grad_output_ptr: tl.tensor,
    output_ptr: tl.tensor, delta_ptr: tl.tensor, y_size: tl.int32, x_size:
    tl.int32, y_stride: tl.int32, x_stride: tl.int32, dtype: tl.constexpr,
    x_block_size: tl.constexpr, require_x_boundary_check: tl.constexpr):
    y_offset = tl.program_id(0)
    grad_input_block_ptr = tl.make_block_ptr(grad_input_ptr, shape=(y_size,
        x_size), strides=(y_stride, x_stride), offsets=(y_offset, 0),
        block_shape=(1, x_block_size), order=(1, 0))
    grad_output_block_ptr = tl.make_block_ptr(grad_output_ptr, shape=(
        y_size, x_size), strides=(y_stride, x_stride), offsets=(y_offset, 0
        ), block_shape=(1, x_block_size), order=(1, 0))
    output_block_ptr = tl.make_block_ptr(output_ptr, shape=(y_size, x_size),
        strides=(y_stride, x_stride), offsets=(y_offset, 0), block_shape=(1,
        x_block_size), order=(1, 0))
    delta_block_ptr = tl.make_block_ptr(delta_ptr, shape=(y_size,), strides
        =(1,), offsets=(y_offset,), block_shape=(1,), order=(0,))
    if require_x_boundary_check:
        output = tl.load(output_block_ptr, boundary_check=(1,))
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(1,))
    else:
        output = tl.load(output_block_ptr)
        grad_output = tl.load(grad_output_block_ptr)
    delta = tl.load(delta_block_ptr)
    grad_input = output * (grad_output - delta)
    if require_x_boundary_check:
        tl.store(grad_input_block_ptr, grad_input.to(dtype), boundary_check
            =(1,))
    else:
        tl.store(grad_input_block_ptr, grad_input.to(dtype))


@staticmethod
@triton.heuristics({'require_x_boundary_check': lambda args: args['x_size'] %
    args['x_block_size']})
@triton.jit
def backward_delta(delta_ptr: tl.tensor, grad_output_ptr: tl.tensor,
    output_ptr: tl.tensor, y_size: tl.int32, x_size: tl.int32, y_stride: tl
    .int32, x_stride: tl.int32, dtype: tl.constexpr, x_block_size: tl.
    constexpr, require_x_boundary_check: tl.constexpr):
    y_offset = tl.program_id(0)
    delta_block_ptr = tl.make_block_ptr(delta_ptr, shape=(y_size,), strides
        =(1,), offsets=(y_offset,), block_shape=(1,), order=(0,))
    grad_output_block_ptr = tl.make_block_ptr(grad_output_ptr, shape=(
        y_size, x_size), strides=(y_stride, x_stride), offsets=(y_offset, 0
        ), block_shape=(1, x_block_size), order=(1, 0))
    output_block_ptr = tl.make_block_ptr(output_ptr, shape=(y_size, x_size),
        strides=(y_stride, x_stride), offsets=(y_offset, 0), block_shape=(1,
        x_block_size), order=(1, 0))
    if require_x_boundary_check:
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(1,),
            padding_option='zero')
        output = tl.load(output_block_ptr, boundary_check=(1,))
    else:
        grad_output = tl.load(grad_output_block_ptr)
        output = tl.load(output_block_ptr)
    delta = tl.sum(grad_output * output, 1)
    tl.store(delta_block_ptr, delta.to(dtype))


# Backward method (kernel launch code)
def _MaskedSoftmax_backward(ctx: Any, *grad_outputs: Any):
    grad_output, = grad_outputs
    output, = ctx.saved_tensors
    util.push_trace('MaskedSoftmax.__backward')
    grad_input = MaskedSoftmax.__backward(grad_output, output, ctx.dim)
    util.pop_trace()
    return grad_input, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class MaskedSoftmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, mask, dim = args
        util.push_trace('MaskedSoftmax.__forward')
        output = MaskedSoftmax.__forward(input, mask, dim)
        util.pop_trace()
        ctx.save_for_backward(output)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        grad_output, = grad_outputs
        output, = ctx.saved_tensors
        util.push_trace('MaskedSoftmax.__backward')
        grad_input = MaskedSoftmax.__backward(grad_output, output, ctx.dim)
        util.pop_trace()
        return grad_input, None, None

    @staticmethod
    def __forward(input: torch.Tensor, mask: torch.Tensor, dim: torch.int32):
        y_size, x_size, y_stride, x_stride = util.size_and_stride(input, dim)
        output = torch.empty_like(input)

        def grid(meta):
            return y_size,
        util.push_trace('kernel.MaskedSoftmax.forward')
        kernel.MaskedSoftmax.forward[grid](output, input, y_size, x_size,
            y_stride, x_stride, mask, util.dtype(output.dtype), triton.
            next_power_of_2(x_size))
        util.pop_trace()
        return output

    @staticmethod
    def __backward(grad_output: torch.Tensor, output: torch.Tensor, dim:
        torch.int32):
        factory_kwargs = {'device': output.device, 'dtype': output.dtype}
        y_size, x_size, y_stride, x_stride = util.size_and_stride(output, dim)
        delta = torch.empty(y_size, **factory_kwargs)
        grad_input = torch.empty_like(output)

        def grid(meta):
            return y_size,
        util.push_trace('kernel.MaskedSoftmax.backward_delta')
        kernel.MaskedSoftmax.backward_delta[grid](delta, grad_output,
            output, y_size, x_size, y_stride, x_stride, util.dtype(delta.
            dtype), triton.next_power_of_2(x_size))
        util.pop_trace()

        def grid(meta):
            return y_size,
        util.push_trace('kernel.MaskedSoftmax.backward')
        kernel.MaskedSoftmax.backward[grid](grad_input, grad_output, output,
            delta, y_size, x_size, y_stride, x_stride, util.dtype(output.
            dtype), triton.next_power_of_2(x_size))
        util.pop_trace()
        return grad_input
