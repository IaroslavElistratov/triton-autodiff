# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/kakaobrain/trident
# Source-Files: trident/operation/relu.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rl_ay0ux/trident-main/trident/operation/relu.py
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
@util.autotune(relu_configs(), ['x_size'])
@triton.jit
def forward(output_ptr: tl.tensor, input_ptr: tl.tensor, x_size: tl.int32,
    dtype: tl.constexpr, x_block_size: tl.constexpr):
    x_offset = tl.program_id(0) * x_block_size
    output_block_ptr = tl.make_block_ptr(output_ptr, shape=(x_size,),
        strides=(1,), offsets=(x_offset,), block_shape=(x_block_size,),
        order=(0,))
    input_block_ptr = tl.make_block_ptr(input_ptr, shape=(x_size,), strides
        =(1,), offsets=(x_offset,), block_shape=(x_block_size,), order=(0,))
    input = tl.load(input_block_ptr, boundary_check=(0,))
    output = language.math.ReLU.forward(input)
    tl.store(output_block_ptr, output.to(dtype), boundary_check=(0,))


# Forward method (kernel launch code)
def _ReLU_forward(ctx: Any, *args: Any, **kwargs: Any):
    input, = args
    util.push_trace('ReLU.__forward')
    output = ReLU.__forward(input)
    util.pop_trace()
    ctx.save_for_backward(input)
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@staticmethod
@util.autotune(relu_configs(), ['x_size'])
@triton.jit
def backward(grad_input_ptr: tl.tensor, grad_output_ptr: tl.tensor,
    input_ptr: tl.tensor, x_size: tl.int32, dtype: tl.constexpr,
    x_block_size: tl.constexpr):
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
    grad_input = language.math.ReLU.backward(grad_output, input)
    tl.store(grad_input_block_ptr, grad_input.to(dtype), boundary_check=(0,))


# Backward method (kernel launch code)
def _ReLU_backward(ctx: Any, *grad_outputs: Any):
    grad_output = grad_outputs[0]
    input, = ctx.saved_tensors
    util.push_trace('ReLU.__forward')
    grad_input = ReLU.__backward(grad_output, input)
    util.pop_trace()
    return grad_input


# ============================================================
# autograd.Function Class Definition
# ============================================================

class ReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, = args
        util.push_trace('ReLU.__forward')
        output = ReLU.__forward(input)
        util.pop_trace()
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        grad_output = grad_outputs[0]
        input, = ctx.saved_tensors
        util.push_trace('ReLU.__forward')
        grad_input = ReLU.__backward(grad_output, input)
        util.pop_trace()
        return grad_input

    @staticmethod
    def __forward(input: torch.Tensor):
        x_size = input.numel()
        output = torch.empty_like(input)

        def grid(meta):
            return triton.cdiv(x_size, meta['x_block_size']),
        util.push_trace('kernel.ReLU.forward')
        kernel.ReLU.forward[grid](output, input, x_size, util.dtype(output.
            dtype))
        util.pop_trace()
        return output

    @staticmethod
    def __backward(grad_output: torch.Tensor, input: torch.Tensor):
        x_size = input.numel()
        grad_input = torch.empty_like(input)

        def grid(meta):
            return [triton.cdiv(x_size, meta['x_block_size'])]
        util.push_trace('kernel.ReLU.backward')
        kernel.ReLU.backward[grid](grad_input, grad_output, input, x_size,
            util.dtype(grad_input.dtype))
        util.pop_trace()
        return grad_input
