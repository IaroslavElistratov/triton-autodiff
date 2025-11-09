# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/kakaobrain/trident
# Source-Files: trident/operation/cosine_similarity.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rl_ay0ux/trident-main/trident/operation/cosine_similarity.py
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
@util.autotune(cosine_similarity_configs(), ['y_size', 'x_size'])
@triton.heuristics({'require_boundary_check': lambda args: args[
    'size_along_dim'] % args['block_size']})
@triton.jit
def forward(output_ptr: tl.tensor, denominator_ptr: tl.tensor,
    numerator_ptr: tl.tensor, x1_ptr: tl.tensor, x2_ptr: tl.tensor, z_size:
    tl.int32, y_size: tl.int32, x_size: tl.int32, z_stride: tl.int32,
    y_stride: tl.int32, x_stride: tl.int32, eps: tl.float32, size_along_dim:
    tl.int32, output_y_size: tl.int32, output_x_size: tl.int32, dtype: tl.
    constexpr, block_size: tl.constexpr, require_boundary_check: tl.constexpr):
    pid = tl.program_id(0)
    num_output_y = pid // output_x_size
    num_output_x = pid % output_x_size
    x1_block_ptr = tl.make_block_ptr(x1_ptr, shape=(z_size, y_size, x_size),
        strides=(z_stride, y_stride, x_stride), offsets=(0, num_output_y,
        num_output_x), block_shape=(block_size, 1, 1), order=(2, 1, 0))
    x2_block_ptr = tl.make_block_ptr(x2_ptr, shape=(z_size, y_size, x_size),
        strides=(z_stride, y_stride, x_stride), offsets=(0, num_output_y,
        num_output_x), block_shape=(block_size, 1, 1), order=(2, 1, 0))
    output_block_ptr = tl.make_block_ptr(output_ptr, shape=(output_y_size,
        output_x_size), strides=(output_x_size, 1), offsets=(num_output_y,
        num_output_x), block_shape=(1, 1), order=(1, 0))
    denominator_block_ptr = tl.make_block_ptr(denominator_ptr, shape=(
        output_y_size, output_x_size), strides=(output_x_size, 1), offsets=
        (num_output_y, num_output_x), block_shape=(1, 1), order=(1, 0))
    numerator_block_ptr = tl.make_block_ptr(numerator_ptr, shape=(
        output_y_size, output_x_size), strides=(output_x_size, 1), offsets=
        (num_output_y, num_output_x), block_shape=(1, 1), order=(1, 0))
    denominator_accumulation1 = tl.zeros((block_size, 1, 1), tl.float32)
    denominator_accumulation2 = tl.zeros((block_size, 1, 1), tl.float32)
    numerator_accumulation = tl.zeros((block_size, 1, 1), tl.float32)
    for _ in range(0, size_along_dim, block_size):
        if require_boundary_check:
            x1 = tl.load(x1_block_ptr, boundary_check=(0,), padding_option=
                'zero')
            x2 = tl.load(x2_block_ptr, boundary_check=(0,), padding_option=
                'zero')
        else:
            x1 = tl.load(x1_block_ptr)
            x2 = tl.load(x2_block_ptr)
        denominator_accumulation1 += x1 * x1
        denominator_accumulation2 += x2 * x2
        numerator_accumulation += x1 * x2
        x1_block_ptr = tl.advance(x1_block_ptr, (block_size, 0, 0))
        x2_block_ptr = tl.advance(x2_block_ptr, (block_size, 0, 0))
    denominator1 = tl.sum(denominator_accumulation1, 0)
    denominator2 = tl.sum(denominator_accumulation2, 0)
    denominator = tl.sqrt(denominator1) * tl.sqrt(denominator2)
    numerator = tl.sum(numerator_accumulation, 0)
    output = numerator / tl.math.max(denominator, eps)
    tl.store(output_block_ptr, output.to(dtype))
    tl.store(denominator_block_ptr, denominator.to(dtype))
    tl.store(numerator_block_ptr, numerator.to(dtype))


# Forward method (kernel launch code)
def _CosineSimilarity_forward(ctx: Any, *args: Any, **kwargs: Any):
    x1, x2, dim, eps = args
    if dim >= x1.dim():
        raise ValueError(f"Unable to process the given dim: '{dim}'.")
    util.push_trace('CosineSimilarity.__forward')
    output, denominator, numerator = CosineSimilarity.__forward(x1.view(
        CosineSimilarity.__input_shape(x1)), x2.view(CosineSimilarity.
        __input_shape(x2)), CosineSimilarity.__dim(x1, dim), eps)
    util.pop_trace()
    ctx.save_for_backward(x1, x2, denominator, numerator)
    ctx.dim = dim
    return output.view(CosineSimilarity.__output_shape(x1, output))


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@staticmethod
@util.autotune(cosine_similarity_configs(), ['y_size', 'x_size'])
@triton.heuristics({'require_boundary_check': lambda args: args[
    'size_along_dim'] % args['block_size']})
@triton.jit
def backward(grad_x1_ptr: tl.tensor, grad_x2_ptr: tl.tensor,
    grad_output_ptr: tl.tensor, denominator_ptr: tl.tensor, numerator_ptr:
    tl.tensor, x1_ptr: tl.tensor, x2_ptr: tl.tensor, z_size: tl.int32,
    y_size: tl.int32, x_size: tl.int32, z_stride: tl.int32, y_stride: tl.
    int32, x_stride: tl.int32, size_along_dim: tl.int32, output_y_size: tl.
    int32, output_x_size: tl.int32, dtype: tl.constexpr, block_size: tl.
    constexpr, require_boundary_check: tl.constexpr):
    pid = tl.program_id(0)
    num_output_y = pid // output_x_size
    num_output_x = pid % output_x_size
    grad_x1_block_ptr = tl.make_block_ptr(grad_x1_ptr, shape=(z_size,
        y_size, x_size), strides=(z_stride, y_stride, x_stride), offsets=(0,
        num_output_y, num_output_x), block_shape=(block_size, 1, 1), order=
        (2, 1, 0))
    grad_x2_block_ptr = tl.make_block_ptr(grad_x2_ptr, shape=(z_size,
        y_size, x_size), strides=(z_stride, y_stride, x_stride), offsets=(0,
        num_output_y, num_output_x), block_shape=(block_size, 1, 1), order=
        (2, 1, 0))
    grad_output_block_ptr = tl.make_block_ptr(grad_output_ptr, shape=(
        output_y_size, output_x_size), strides=(output_x_size, 1), offsets=
        (num_output_y, num_output_x), block_shape=(1, 1), order=(1, 0))
    x1_block_ptr = tl.make_block_ptr(x1_ptr, shape=(z_size, y_size, x_size),
        strides=(z_stride, y_stride, x_stride), offsets=(0, num_output_y,
        num_output_x), block_shape=(block_size, 1, 1), order=(2, 1, 0))
    x2_block_ptr = tl.make_block_ptr(x2_ptr, shape=(z_size, y_size, x_size),
        strides=(z_stride, y_stride, x_stride), offsets=(0, num_output_y,
        num_output_x), block_shape=(block_size, 1, 1), order=(2, 1, 0))
    denominator_block_ptr = tl.make_block_ptr(denominator_ptr, shape=(
        output_y_size, output_x_size), strides=(output_x_size, 1), offsets=
        (num_output_y, num_output_x), block_shape=(1, 1), order=(1, 0))
    numerator_block_ptr = tl.make_block_ptr(numerator_ptr, shape=(
        output_y_size, output_x_size), strides=(output_x_size, 1), offsets=
        (num_output_y, num_output_x), block_shape=(1, 1), order=(1, 0))
    for _ in range(0, size_along_dim, block_size):
        if require_boundary_check:
            x1 = tl.load(x1_block_ptr, boundary_check=(0,), padding_option=
                'zero').to(tl.float32)
            x2 = tl.load(x2_block_ptr, boundary_check=(0,), padding_option=
                'zero').to(tl.float32)
        else:
            x1 = tl.load(x1_block_ptr)
            x2 = tl.load(x2_block_ptr)
        denominator = tl.load(denominator_block_ptr)
        numerator = tl.load(numerator_block_ptr)
        grad_output = tl.load(grad_output_block_ptr)
        squared_x1 = x1 * x1
        squared_x2 = x2 * x2
        squared_x1_sum = tl.sum(squared_x1, 0)
        squared_x2_sum = tl.sum(squared_x2, 0)
        grad_denominator = grad_output * numerator * (-1 / (denominator *
            denominator))
        grad_mul1 = grad_denominator * tl.sqrt(tl.sum(squared_x2, 0))
        grad_mul2 = grad_denominator * tl.sqrt(tl.sum(squared_x1, 0))
        grad_sqrt1 = grad_mul1 / (2 * tl.sqrt(squared_x1_sum))
        grad_sqrt2 = grad_mul2 / (2 * tl.sqrt(squared_x2_sum))
        grad_to_dot = grad_output / denominator
        grad_x1 = grad_sqrt1 * 2 * x1 + grad_to_dot * x2
        grad_x2 = grad_sqrt2 * 2 * x2 + grad_to_dot * x1
        if require_boundary_check:
            tl.store(grad_x1_block_ptr, grad_x1.to(dtype), boundary_check=(0,))
            tl.store(grad_x2_block_ptr, grad_x2.to(dtype), boundary_check=(0,))
        else:
            tl.store(grad_x1_block_ptr, grad_x1.to(dtype))
            tl.store(grad_x2_block_ptr, grad_x2.to(dtype))
        x1_block_ptr = tl.advance(x1_block_ptr, (block_size, 0, 0))
        x2_block_ptr = tl.advance(x2_block_ptr, (block_size, 0, 0))
        grad_x1_block_ptr = tl.advance(grad_x1_block_ptr, (block_size, 0, 0))
        grad_x2_block_ptr = tl.advance(grad_x2_block_ptr, (block_size, 0, 0))


# Backward method (kernel launch code)
def _CosineSimilarity_backward(ctx: Any, *grad_outputs: Any):
    grad_output = grad_outputs[0]
    x1, x2, denominator, numerator = ctx.saved_tensors
    util.push_trace('CosineSimilarity.__backward')
    grad_x1, grad_x2 = CosineSimilarity.__backward(grad_output, x1.view(
        CosineSimilarity.__input_shape(x1)), x2.view(CosineSimilarity.
        __input_shape(x2)), denominator, numerator, CosineSimilarity.__dim(
        x1, ctx.dim))
    util.pop_trace()
    return grad_x1.view(x1.shape), grad_x2.view(x2.shape), None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class CosineSimilarity(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        x1, x2, dim, eps = args
        if dim >= x1.dim():
            raise ValueError(f"Unable to process the given dim: '{dim}'.")
        util.push_trace('CosineSimilarity.__forward')
        output, denominator, numerator = CosineSimilarity.__forward(x1.view
            (CosineSimilarity.__input_shape(x1)), x2.view(CosineSimilarity.
            __input_shape(x2)), CosineSimilarity.__dim(x1, dim), eps)
        util.pop_trace()
        ctx.save_for_backward(x1, x2, denominator, numerator)
        ctx.dim = dim
        return output.view(CosineSimilarity.__output_shape(x1, output))

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        grad_output = grad_outputs[0]
        x1, x2, denominator, numerator = ctx.saved_tensors
        util.push_trace('CosineSimilarity.__backward')
        grad_x1, grad_x2 = CosineSimilarity.__backward(grad_output, x1.view
            (CosineSimilarity.__input_shape(x1)), x2.view(CosineSimilarity.
            __input_shape(x2)), denominator, numerator, CosineSimilarity.
            __dim(x1, ctx.dim))
        util.pop_trace()
        return grad_x1.view(x1.shape), grad_x2.view(x2.shape), None, None

    @staticmethod
    def __forward(x1: torch.Tensor, x2: torch.Tensor, dim: torch.int32, eps:
        torch.float32):
        assert x1.is_contiguous() and x2.is_contiguous(
            ) and x1.shape == x2.shape
        factory_kwargs = {'device': x1.device, 'dtype': x1.dtype}
        z_size, y_size, x_size, z_stride, y_stride, x_stride = (util.
            size_and_stride(x1, dim))
        output_y_size, output_x_size, size_along_dim = (CosineSimilarity.
            __output_size_and_size_along_dim(x1, dim))
        output = torch.empty(output_y_size, output_x_size, **factory_kwargs)
        denominator = torch.empty_like(output)
        numerator = torch.empty_like(output)

        def grid(meta):
            return output_y_size * output_x_size,
        util.push_trace('kernel.CosineSimilarity.forward')
        kernel.CosineSimilarity.forward[grid](output, denominator,
            numerator, x1, x2, z_size, y_size, x_size, z_stride, y_stride,
            x_stride, eps, size_along_dim, output_y_size, output_x_size,
            util.dtype(x1.dtype))
        util.pop_trace()
        return output, denominator, numerator

    @staticmethod
    def __backward(grad_output, x1, x2, denominator, numerator, dim):
        grad_x1 = torch.empty_like(x1)
        grad_x2 = torch.empty_like(x2)
        z_size, y_size, x_size, z_stride, y_stride, x_stride = (util.
            size_and_stride(x1, dim))
        output_y_size, output_x_size, size_along_dim = (CosineSimilarity.
            __output_size_and_size_along_dim(x1, dim))

        def grid(meta):
            return output_y_size * output_x_size,
        util.push_trace('kernel.CosineSimilarity.backward')
        kernel.CosineSimilarity.backward[grid](grad_x1, grad_x2,
            grad_output, denominator, numerator, x1, x2, z_size, y_size,
            x_size, z_stride, y_stride, x_stride, size_along_dim,
            output_y_size, output_x_size, util.dtype(x1.dtype))
        util.pop_trace()
        return grad_x1, grad_x2

    @staticmethod
    def __output_size_and_size_along_dim(input: torch.Tensor, dim: int):
        z_size, y_size, x_size = input.shape
        if dim == 0:
            output_y_size, output_x_size = y_size, x_size
            size_along_dim = z_size
        elif dim == 1:
            output_y_size, output_x_size = z_size, x_size
            size_along_dim = y_size
        else:
            output_y_size, output_x_size = z_size, y_size
            size_along_dim = x_size
        return output_y_size, output_x_size, size_along_dim

    @staticmethod
    def __input_shape(input: torch.Tensor):
        if input.dim() == 1:
            return 1, 1, *input.shape
        elif input.dim() == 2:
            return 1, *input.shape
        elif input.dim() == 3:
            return input.shape
        else:
            raise ValueError(f"Unable to convert the given input: '{input}'.")

    @staticmethod
    def __output_shape(input: torch.Tensor, output: torch.Tensor):
        if input.dim() == 1:
            return ()
        if input.dim() == 2:
            return output.shape[1]
        elif input.dim() == 3:
            return output.shape
        else:
            raise ValueError(f"Unable to convert the given x: '{input}'.")

    @staticmethod
    def __dim(input: torch.Tensor, dim: torch.int32):
        return dim + 3 - input.dim()
