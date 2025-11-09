# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/kakaobrain/trident
# Source-Files: trident/operation/leaky_relu.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rl_ay0ux/trident-main/trident/operation/leaky_relu.py
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


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@staticmethod
@util.autotune(leaky_relu_configs(), ['x_size'])
@triton.jit
def forward(output_ptr: tl.tensor, input_ptr: tl.tensor, x_size: tl.int32,
    negative_slope: tl.float32, dtype: tl.constexpr, x_block_size: tl.constexpr
    ):
    x_offset = tl.program_id(0) * x_block_size
    output_block_ptr = tl.make_block_ptr(output_ptr, shape=(x_size,),
        strides=(1,), offsets=(x_offset,), block_shape=(x_block_size,),
        order=(0,))
    input_block_ptr = tl.make_block_ptr(input_ptr, shape=(x_size,), strides
        =(1,), offsets=(x_offset,), block_shape=(x_block_size,), order=(0,))
    input = tl.load(input_block_ptr, boundary_check=(0,))
    output = language.math.LeakyReLU.forward(input, negative_slope)
    tl.store(output_block_ptr, output.to(dtype), boundary_check=(0,))


# Forward method (kernel launch code)
def _LeakyReLU_forward(ctx: Any, *args: Any, **kwargs: Any):
    input, negative_slope = args
    util.push_trace('LeakyReLU.__forward')
    output = LeakyReLU.__forward(input, negative_slope)
    util.pop_trace()
    ctx.save_for_backward(input)
    ctx.negative_slope = negative_slope
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@staticmethod
@util.autotune(leaky_relu_configs(), ['x_size'])
@triton.jit
def backward(grad_input_ptr: tl.tensor, grad_output_ptr: tl.tensor,
    input_ptr: tl.tensor, x_size: tl.int32, negative_slope: tl.float32,
    dtype: tl.constexpr, x_block_size: tl.constexpr):
    x_offset = tl.program_id(0) * x_block_size
    grad_input_block_ptr = tl.make_block_ptr(grad_input_ptr, shape=(x_size,
        ), strides=(1,), offsets=(x_offset,), block_shape=(x_block_size,),
        order=(0,))
    grad_output_block_ptr = tl.make_block_ptr(grad_output_ptr, shape=(
        x_size,), strides=(1,), offsets=(x_offset,), block_shape=(
        x_block_size,), order=(0,))
    input_block_ptr = tl.make_block_ptr(input_ptr, shape=(x_size,), strides
        =(1,), offsets=(x_offset,), block_shape=(x_block_size,), order=(0,))
    grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,))
    input = tl.load(input_block_ptr, boundary_check=(0,))
    grad_input = language.math.LeakyReLU.backward(grad_output, input,
        negative_slope)
    tl.store(grad_input_block_ptr, grad_input.to(dtype), boundary_check=(0,))


# Backward method (kernel launch code)
def _LeakyReLU_backward(ctx: Any, *grad_outputs: Any):
    grad_output = grad_outputs[0]
    input, = ctx.saved_tensors
    util.push_trace('LeakyReLU.__backward')
    grad_input = LeakyReLU.__backward(grad_output, input, ctx.negative_slope)
    util.pop_trace()
    return grad_input, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LeakyReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, negative_slope = args
        util.push_trace('LeakyReLU.__forward')
        output = LeakyReLU.__forward(input, negative_slope)
        util.pop_trace()
        ctx.save_for_backward(input)
        ctx.negative_slope = negative_slope
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        grad_output = grad_outputs[0]
        input, = ctx.saved_tensors
        util.push_trace('LeakyReLU.__backward')
        grad_input = LeakyReLU.__backward(grad_output, input, ctx.
            negative_slope)
        util.pop_trace()
        return grad_input, None

    @staticmethod
    def __forward(input: torch.Tensor, negative_slope: torch.float32):
        x_size = input.numel()
        output = torch.empty_like(input)

        def grid(meta):
            return triton.cdiv(x_size, meta['x_block_size']),
        util.push_trace('kernel.LeakyReLU.forward')
        kernel.LeakyReLU.forward[grid](output, input, x_size,
            negative_slope, util.dtype(input.dtype))
        util.pop_trace()
        return output

    @staticmethod
    def __backward(grad_output: torch.Tensor, input: torch.Tensor,
        negative_slope: torch.float32):
        x_size = input.numel()
        grad_input = torch.empty_like(input)

        def grid(meta):
            return [triton.cdiv(x_size, meta['x_block_size'])]
        util.push_trace('kernel.LeakyReLU.backward')
        kernel.LeakyReLU.backward[grid](grad_input, grad_output, input,
            x_size, negative_slope, util.dtype(grad_input.dtype))
        util.pop_trace()
        return grad_input
