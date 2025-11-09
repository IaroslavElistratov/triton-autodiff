# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/kakaobrain/trident
# Source-Files: trident/operation/mean.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rl_ay0ux/trident-main/trident/operation/mean.py
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
@util.autotune(mean_configs(), ['x_size'])
@triton.heuristics({'require_x_boundary_check': lambda args: args['x_size'] %
    args['x_block_size']})
@triton.jit
def forward(output_ptr: tl.tensor, input_ptr: tl.tensor, y_size: tl.int32,
    x_size: tl.int32, y_stride: tl.int32, x_stride: tl.int32, dtype: tl.
    constexpr, x_block_size: tl.constexpr, require_x_boundary_check: tl.
    constexpr):
    y_offset = tl.program_id(0)
    output_block_ptr = tl.make_block_ptr(output_ptr, shape=(y_size,),
        strides=(1,), offsets=(y_offset,), block_shape=(1,), order=(0,))
    output = language.Mean.forward(input_ptr, y_size, x_size, y_stride,
        x_stride, y_offset, dtype, x_block_size, require_x_boundary_check)
    tl.store(output_block_ptr, output)


# Forward method (kernel launch code)
def _Mean_forward(ctx: Any, *args: Any, **kwargs: Any):
    input, dim = args
    util.push_trace('Mean.__forward')
    output = Mean.__forward(input, dim)
    util.pop_trace()
    ctx.save_for_backward(input)
    ctx.dim = dim
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@staticmethod
@util.autotune(mean_configs(), ['x_size'])
@triton.heuristics({'require_x_boundary_check': lambda args: args['x_size'] %
    args['x_block_size']})
@triton.jit
def backward(grad_input_ptr: tl.tensor, grad_output_ptr: tl.tensor, y_size:
    tl.int32, x_size: tl.int32, y_stride: tl.int32, x_stride: tl.int32,
    dtype: tl.constexpr, x_block_size: tl.constexpr,
    require_x_boundary_check: tl.constexpr):
    pid = tl.program_id(0)
    num_x_blocks = tl.cdiv(x_size, x_block_size)
    y_offset = pid // num_x_blocks
    x = pid % num_x_blocks
    x_offset = x * x_block_size
    grad_input_block_ptr = tl.make_block_ptr(grad_input_ptr, shape=(y_size,
        x_size), strides=(y_stride, x_stride), offsets=(y_offset, x_offset),
        block_shape=(1, x_block_size), order=(0, 1))
    grad_input = language.Mean.backward(grad_output_ptr, y_size, x_size,
        y_offset, dtype, x_block_size)
    if require_x_boundary_check:
        tl.store(grad_input_block_ptr, grad_input, boundary_check=(1,))
    else:
        tl.store(grad_input_block_ptr, grad_input)


# Backward method (kernel launch code)
def _Mean_backward(ctx: Any, *grad_outputs: Any):
    input, = ctx.saved_tensors
    grad_output, = grad_outputs
    util.push_trace('Mean.__backward')
    grad_input = Mean.__backward(grad_output, input, ctx.dim)
    util.pop_trace()
    return grad_input, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class Mean(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, dim = args
        util.push_trace('Mean.__forward')
        output = Mean.__forward(input, dim)
        util.pop_trace()
        ctx.save_for_backward(input)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        input, = ctx.saved_tensors
        grad_output, = grad_outputs
        util.push_trace('Mean.__backward')
        grad_input = Mean.__backward(grad_output, input, ctx.dim)
        util.pop_trace()
        return grad_input, None

    @staticmethod
    def __forward(input: torch.Tensor, dim: torch.int32):
        factory_kwargs = {'device': input.device, 'dtype': input.dtype}
        y_size, x_size, y_stride, x_stride = util.size_and_stride(input, dim)
        output = torch.empty(y_size, **factory_kwargs)

        def grid(meta):
            return y_size,
        util.push_trace('kernel.Mean.forward')
        kernel.Mean.forward[grid](output, input, y_size, x_size, y_stride,
            x_stride, util.dtype(input.dtype))
        util.pop_trace()
        return output

    @staticmethod
    def __backward(grad_output: torch.Tensor, input: torch.Tensor, dim:
        torch.int32):
        y_size, x_size, y_stride, x_stride = util.size_and_stride(input, dim)
        grad_input = torch.zeros_like(input)

        def grid(meta):
            return y_size * triton.cdiv(x_size, meta['x_block_size']),
        util.push_trace('kernel.Mean.backward')
        kernel.Mean.backward[grid](grad_input, grad_output, y_size, x_size,
            y_stride, x_stride, util.dtype(grad_input.dtype))
        util.pop_trace()
        return grad_input
