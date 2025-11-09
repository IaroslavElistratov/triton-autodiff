# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/kakaobrain/trident
# Source-Files: trident/operation/linear.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rl_ay0ux/trident-main/trident/operation/linear.py
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
@util.autotune(linear_configs([16, 64, 128], [32, 64, 128], [32, 64]), [
    'm_size', 'n_size', 'k_size'])
@triton.heuristics({'require_m_boundary_check': lambda args: args['m_size'] %
    args['m_block_size'], 'require_n_boundary_check': lambda args: args[
    'n_size'] % args['n_block_size'], 'require_k_boundary_check': lambda
    args: args['k_size'] % args['k_block_size']})
@triton.jit
def forward(output_ptr: tl.tensor, input_ptr: tl.tensor, weight_ptr: tl.
    tensor, bias_ptr: tl.tensor, m_size: tl.int32, n_size: tl.int32, k_size:
    tl.int32, input_batch_stride: tl.int32, input_m_stride: tl.int32,
    input_k_stride: tl.int32, weight_n_stride: tl.int32, weight_k_stride:
    tl.int32, use_accelerator: tl.constexpr, dtype: tl.constexpr,
    m_block_size: tl.constexpr, n_block_size: tl.constexpr, k_block_size:
    tl.constexpr, require_m_boundary_check: tl.constexpr,
    require_n_boundary_check: tl.constexpr, require_k_boundary_check: tl.
    constexpr):
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(m_size, m_block_size)
    num_n_blocks = tl.cdiv(n_size, n_block_size)
    num_blocks = num_m_blocks * num_n_blocks
    batch = pid // num_blocks
    block = pid % num_blocks
    m_block = block // num_n_blocks
    n_block = block % num_n_blocks
    m_offset = m_block * m_block_size
    n_offset = n_block * n_block_size
    output = language.Linear.forward(input_ptr + batch * input_batch_stride,
        weight_ptr, bias_ptr, m_size, n_size, k_size, input_m_stride,
        input_k_stride, weight_n_stride, weight_k_stride, m_offset,
        n_offset, use_accelerator, m_block_size, n_block_size, k_block_size,
        require_m_boundary_check, require_n_boundary_check,
        require_k_boundary_check, dtype)
    output_block_ptr = tl.make_block_ptr(output_ptr + batch * m_size *
        n_size, shape=(m_size, n_size), strides=(n_size, 1), offsets=(
        m_offset, n_offset), block_shape=(m_block_size, n_block_size),
        order=(1, 0))
    if require_m_boundary_check | require_n_boundary_check:
        tl.store(output_block_ptr, output, boundary_check=(0, 1))
    else:
        tl.store(output_block_ptr, output)


# Forward method (kernel launch code)
def _Linear_forward(ctx: Any, *args: Any, **kwargs: Any):
    input, weight, bias, use_accelerator = args
    util.push_trace('Linear.__forward')
    output = Linear.__forward(input, weight, bias, use_accelerator)
    util.pop_trace()
    ctx.save_for_backward(input, weight, bias)
    ctx.use_accelerator = use_accelerator
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@staticmethod
@util.autotune(linear_backward_configs([64, 128], [32, 64], [32, 64, 128]),
    ['m_size', 'n_size', 'k_size'])
@triton.heuristics({'require_m_boundary_check': lambda args: args['m_size'] %
    args['m_block_size'], 'require_n_boundary_check': lambda args: args[
    'n_size'] % args['n_block_size'], 'require_k_boundary_check': lambda
    args: args['k_size'] % args['k_block_size']})
@triton.jit
def backward(grad_input_ptr: tl.tensor, grad_output_ptr: tl.tensor,
    weight_ptr: tl.tensor, m_size: tl.int32, n_size: tl.int32, k_size: tl.
    int32, input_m_stride: tl.int32, input_k_stride: tl.int32,
    weight_n_stride: tl.int32, weight_k_stride: tl.int32, use_accelerator:
    tl.constexpr, dtype: tl.constexpr, m_block_size: tl.constexpr,
    n_block_size: tl.constexpr, k_block_size: tl.constexpr,
    require_m_boundary_check: tl.constexpr, require_n_boundary_check: tl.
    constexpr, require_k_boundary_check: tl.constexpr):
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(m_size, m_block_size)
    num_k_blocks = tl.cdiv(k_size, k_block_size)
    num_blocks = num_m_blocks * num_k_blocks
    batch = pid // num_blocks
    block = pid % num_blocks
    m_block = block // num_k_blocks
    k_block = block % num_k_blocks
    m_offset = m_block * m_block_size
    k_offset = k_block * k_block_size
    grad_input = language.Linear.backward(grad_output_ptr + batch * m_size *
        n_size, weight_ptr, m_size, n_size, k_size, weight_n_stride,
        weight_k_stride, m_offset, k_offset, use_accelerator, m_block_size,
        n_block_size, k_block_size, require_m_boundary_check,
        require_n_boundary_check, require_k_boundary_check, dtype)
    grad_input_block_ptr = tl.make_block_ptr(grad_input_ptr + batch *
        m_size * k_size, shape=(m_size, k_size), strides=(input_m_stride,
        input_k_stride), offsets=(m_offset, k_offset), block_shape=(
        m_block_size, k_block_size), order=(1, 0))
    if require_m_boundary_check | require_k_boundary_check:
        tl.store(grad_input_block_ptr, grad_input, boundary_check=(0, 1))
    else:
        tl.store(grad_input_block_ptr, grad_input)


@staticmethod
@util.autotune(linear_configs_for_backward_bias(), ['m_size', 'n_size'])
@triton.heuristics({'require_m_boundary_check': lambda args: args['m_size'] %
    args['m_block_size']})
@triton.jit
def backward_bias(grad_bias_staging_ptr: tl.tensor, grad_output_ptr: tl.
    tensor, m_size: tl.int32, n_size: tl.int32, dtype: tl.constexpr,
    m_block_size: tl.constexpr, require_m_boundary_check: tl.constexpr):
    pid = tl.program_id(0)
    batch = pid // n_size
    n_offset = pid % n_size
    grad_bias = language.Linear.backward_bias(grad_output_ptr + batch *
        m_size * n_size, m_size, n_size, n_offset, m_block_size,
        require_m_boundary_check, dtype)
    grad_bias_staging_block_ptr = tl.make_block_ptr(grad_bias_staging_ptr +
        batch * n_size, shape=(n_size,), strides=(1,), offsets=(n_offset,),
        block_shape=(1,), order=(0,))
    tl.store(grad_bias_staging_block_ptr, grad_bias)


@staticmethod
@util.autotune(linear_backward_weight_configs([32, 64], [64, 128], [32, 64,
    128]), ['m_size', 'n_size', 'k_size'])
@triton.heuristics({'require_m_boundary_check': lambda args: args['m_size'] %
    args['m_block_size'], 'require_n_boundary_check': lambda args: args[
    'n_size'] % args['n_block_size'], 'require_k_boundary_check': lambda
    args: args['k_size'] % args['k_block_size']})
@triton.jit
def backward_weight(grad_weight_staging_ptr: tl.tensor, grad_output_ptr: tl
    .tensor, input_ptr: tl.tensor, m_size: tl.int32, n_size: tl.int32,
    k_size: tl.int32, input_batch_stride: tl.int32, input_m_stride: tl.
    int32, input_k_stride: tl.int32, use_accelerator: tl.constexpr, dtype:
    tl.constexpr, m_block_size: tl.constexpr, n_block_size: tl.constexpr,
    k_block_size: tl.constexpr, require_m_boundary_check: tl.constexpr,
    require_n_boundary_check: tl.constexpr, require_k_boundary_check: tl.
    constexpr):
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(n_size, n_block_size)
    num_k_blocks = tl.cdiv(k_size, k_block_size)
    num_blocks = num_n_blocks * num_k_blocks
    batch = pid // num_blocks
    block = pid % num_blocks
    n_block = block // num_k_blocks
    k_block = block % num_k_blocks
    n_offset = n_block * n_block_size
    k_offset = k_block * k_block_size
    grad_weight = language.Linear.backward_weight(grad_output_ptr + batch *
        m_size * n_size, input_ptr + batch * input_batch_stride, m_size,
        n_size, k_size, input_m_stride, input_k_stride, n_offset, k_offset,
        use_accelerator, m_block_size, n_block_size, k_block_size,
        require_m_boundary_check, require_n_boundary_check,
        require_k_boundary_check, dtype)
    grad_weight_staging_block_ptr = tl.make_block_ptr(
        grad_weight_staging_ptr + batch * n_size * k_size, shape=(n_size,
        k_size), strides=(k_size, 1), offsets=(n_offset, k_offset),
        block_shape=(n_block_size, k_block_size), order=(1, 0))
    if require_n_boundary_check | require_k_boundary_check:
        tl.store(grad_weight_staging_block_ptr, grad_weight, boundary_check
            =(0, 1))
    else:
        tl.store(grad_weight_staging_block_ptr, grad_weight)


@staticmethod
@triton.jit
def backward_bias(grad_output_ptr: tl.tensor, m_size: tl.int32, n_size: tl.
    int32, n_offset: tl.int32, m_block_size: tl.constexpr,
    require_m_boundary_check: tl.constexpr, dtype: tl.constexpr):
    grad_output_block_ptr = tl.make_block_ptr(grad_output_ptr, shape=(
        n_size, m_size), strides=(1, n_size), offsets=(n_offset, 0),
        block_shape=(1, m_block_size), order=(0, 1))
    grad_bias = tl.zeros((1, m_block_size), dtype)
    for m_offset in range(0, m_size, m_block_size):
        if require_m_boundary_check:
            grad_output = tl.load(grad_output_block_ptr, boundary_check=(1,
                ), padding_option='zero')
        else:
            grad_output = tl.load(grad_output_block_ptr)
        grad_bias += grad_output
        grad_output_block_ptr = tl.advance(grad_output_block_ptr, (0,
            m_block_size))
    return tl.sum(grad_bias, 1).to(dtype)


@staticmethod
@triton.jit
def backward_weight(grad_output_ptr: tl.tensor, input_ptr: tl.tensor,
    m_size: tl.int32, n_size: tl.int32, k_size: tl.int32, input_m_stride:
    tl.int32, input_k_stride: tl.int32, n_offset: tl.int32, k_offset: tl.
    int32, use_accelerator: tl.constexpr, m_block_size: tl.constexpr,
    n_block_size: tl.constexpr, k_block_size: tl.constexpr,
    require_m_boundary_check: tl.constexpr, require_n_boundary_check: tl.
    constexpr, require_k_boundary_check: tl.constexpr, dtype: tl.constexpr):
    grad_output_block_ptr = tl.make_block_ptr(grad_output_ptr, shape=(
        n_size, m_size), strides=(1, n_size), offsets=(n_offset, 0),
        block_shape=(n_block_size, m_block_size), order=(0, 1))
    input_block_ptr = tl.make_block_ptr(input_ptr, shape=(m_size, k_size),
        strides=(input_m_stride, input_k_stride), offsets=(0, k_offset),
        block_shape=(m_block_size, k_block_size), order=(1, 0))
    grad_weight = tl.zeros((n_block_size, k_block_size), dtype)
    for _ in range(0, m_size, m_block_size):
        if require_m_boundary_check | require_n_boundary_check:
            grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,
                1), padding_option='zero')
        else:
            grad_output = tl.load(grad_output_block_ptr)
        if require_m_boundary_check | require_k_boundary_check:
            input = tl.load(input_block_ptr, boundary_check=(0, 1),
                padding_option='zero')
        else:
            input = tl.load(input_block_ptr)
        grad_weight += language.dot(grad_output, input, use_accelerator, dtype)
        grad_output_block_ptr = tl.advance(grad_output_block_ptr, (0,
            m_block_size))
        input_block_ptr = tl.advance(input_block_ptr, (m_block_size, 0))
    return grad_weight


# Backward method (kernel launch code)
def _Linear_backward(ctx: Any, *grad_outputs: Any):
    grad_output, = grad_outputs
    input, weight, bias = ctx.saved_tensors
    util.push_trace('Linear.__backward')
    grad_input, grad_weight, grad_bias = Linear.__backward(grad_output,
        input, weight, bias, ctx.use_accelerator)
    util.pop_trace()
    return grad_input, grad_weight, grad_bias, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class Linear(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, weight, bias, use_accelerator = args
        util.push_trace('Linear.__forward')
        output = Linear.__forward(input, weight, bias, use_accelerator)
        util.pop_trace()
        ctx.save_for_backward(input, weight, bias)
        ctx.use_accelerator = use_accelerator
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        grad_output, = grad_outputs
        input, weight, bias = ctx.saved_tensors
        util.push_trace('Linear.__backward')
        grad_input, grad_weight, grad_bias = Linear.__backward(grad_output,
            input, weight, bias, ctx.use_accelerator)
        util.pop_trace()
        return grad_input, grad_weight, grad_bias, None, None

    @staticmethod
    def __forward(input, weight, bias, use_accelerator):
        factory_kwargs = {'device': input.device, 'dtype': input.dtype}
        num_batches, m_size, k_size = input.shape
        n_size, _ = weight.shape
        output = torch.empty(num_batches, m_size, n_size, **factory_kwargs)

        def grid(meta):
            num_m_blocks = triton.cdiv(m_size, meta['m_block_size'])
            num_n_blocks = triton.cdiv(n_size, meta['n_block_size'])
            return num_batches * num_m_blocks * num_n_blocks,
        util.push_trace('kernel.Linear.forward')
        kernel.Linear.forward[grid](output, input, weight, bias, m_size,
            n_size, k_size, input.stride(0), input.stride(1), input.stride(
            2), weight.stride(0), weight.stride(1), use_accelerator, util.
            dtype(input.dtype))
        util.pop_trace()
        return output

    @staticmethod
    def __backward(grad_output, input, weight, bias, use_accelerator):
        factory_kwargs = {'device': input.device, 'dtype': input.dtype}
        num_batches, m_size, k_size = input.shape
        n_size, _ = weight.shape
        grad_input = torch.empty_like(input)
        grad_weight_staging = torch.empty(num_batches, n_size, k_size, **
            factory_kwargs)

        def grid(meta):
            num_m_blocks = triton.cdiv(m_size, meta['m_block_size'])
            num_k_blocks = triton.cdiv(k_size, meta['k_block_size'])
            return num_batches * num_m_blocks * num_k_blocks,
        util.push_trace('kernel.Linear.backward')
        kernel.Linear.backward[grid](grad_input, grad_output, weight,
            m_size, n_size, k_size, input.stride(1), input.stride(2),
            weight.stride(0), weight.stride(1), use_accelerator, util.dtype
            (grad_input.dtype))
        util.pop_trace()

        def grid(meta):
            num_n_blocks = triton.cdiv(n_size, meta['n_block_size'])
            num_k_blocks = triton.cdiv(k_size, meta['k_block_size'])
            return num_batches * num_n_blocks * num_k_blocks,
        util.push_trace('kernel.Linear.backward_weight')
        kernel.Linear.backward_weight[grid](grad_weight_staging,
            grad_output, input, m_size, n_size, k_size, input.stride(0),
            input.stride(1), input.stride(2), use_accelerator, util.dtype(
            grad_weight_staging.dtype))
        util.pop_trace()
        util.push_trace('torch.sum')
        grad_weight = torch.sum(grad_weight_staging, 0)
        util.pop_trace()
        if bias is not None:
            grad_bias_staging = torch.empty(num_batches, n_size, **
                factory_kwargs)

            def grid(meta):
                return num_batches * n_size,
            util.push_trace('kernel.Linear.backward_bias')
            kernel.Linear.backward_bias[grid](grad_bias_staging,
                grad_output, m_size, n_size, util.dtype(grad_bias_staging.
                dtype))
            util.pop_trace()
            util.push_trace('torch.sum')
            grad_bias = torch.sum(grad_bias_staging, 0)
            util.pop_trace()
        else:
            grad_bias = None
        return grad_input, grad_weight, grad_bias
