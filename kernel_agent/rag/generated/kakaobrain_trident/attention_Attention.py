# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/kakaobrain/trident
# Source-Files: trident/operation/attention.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rl_ay0ux/trident-main/trident/operation/attention.py
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
from math import exp
from math import log

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
@util.autotune(attention_configs(), ['y_size'])
@triton.jit
def forward(output_ptr: tl.tensor, log2sum_ptr: tl.tensor, query_ptr: tl.
    tensor, key_ptr: tl.tensor, value_ptr: tl.tensor, y_size: tl.int32,
    x_size: tl.int32, head_stride: tl.int32, y_stride: tl.int32, x_stride:
    tl.int32, mask_ptr: tl.tensor, mask_head_stride: tl.int32,
    mask_y_stride: tl.int32, mask_x_stride: tl.int32, dropout_p: tl.float32,
    seed: tl.int32, is_causal: tl.constexpr, softmax_scale: tl.float32,
    use_accelerator: tl.constexpr, dtype: tl.constexpr, y_block_size: tl.
    constexpr, x_block_size: tl.constexpr):
    pid = tl.program_id(0)
    num_y_blocks = tl.cdiv(y_size, y_block_size)
    head = pid // num_y_blocks
    y_block = pid % num_y_blocks
    head_offset = head * head_stride
    y_offset = y_block * y_block_size
    output_block_ptr = tl.make_block_ptr(output_ptr + head_offset, shape=(
        y_size, x_size), strides=(y_stride, x_stride), offsets=(y_offset, 0
        ), block_shape=(y_block_size, x_block_size), order=(1, 0))
    log2sum_block_ptr = tl.make_block_ptr(log2sum_ptr + head * y_size,
        shape=(y_size,), strides=(1,), offsets=(y_offset,), block_shape=(
        y_block_size,), order=(0,))
    query_block_ptr = tl.make_block_ptr(query_ptr + head_offset, shape=(
        y_size, x_size), strides=(y_stride, x_stride), offsets=(y_offset, 0
        ), block_shape=(y_block_size, x_block_size), order=(1, 0))
    key_block_ptr = tl.make_block_ptr(key_ptr + head_offset, shape=(x_size,
        y_size), strides=(x_stride, y_stride), offsets=(0, 0), block_shape=
        (x_block_size, y_block_size), order=(0, 1))
    value_block_ptr = tl.make_block_ptr(value_ptr + head_offset, shape=(
        y_size, x_size), strides=(y_stride, x_stride), offsets=(0, 0),
        block_shape=(y_block_size, x_block_size), order=(1, 0))
    if mask_ptr is not None:
        mask_block_ptr = tl.make_block_ptr(mask_ptr + head *
            mask_head_stride, shape=(y_size, y_size), strides=(
            mask_y_stride, mask_x_stride), offsets=(y_offset, 0),
            block_shape=(y_block_size, y_block_size), order=(1, 0))
    query = tl.load(query_block_ptr)
    score_scale = (softmax_scale * language.log2e).to(dtype)
    query *= score_scale
    max = tl.full((y_block_size,), float('-inf'), tl.float32)
    sum = tl.zeros((y_block_size,), tl.float32)
    output = tl.zeros((y_block_size, x_block_size), dtype)
    m_offsets = tl.arange(0, y_block_size) + y_offset
    if is_causal:
        n_size = y_offset + y_block_size
    else:
        n_size = y_size
    for n_offset in range(0, n_size, y_block_size):
        score = tl.zeros((y_block_size, y_block_size), dtype)
        if is_causal:
            n_offsets = tl.arange(0, y_block_size) + n_offset
            condition = m_offsets[:, None] >= n_offsets[None, :]
            score = tl.where(condition, score, float('-inf'))
        elif mask_ptr is not None:
            mask = tl.load(mask_block_ptr)
            mask *= language.log2e
            score += mask
        key = tl.load(key_block_ptr)
        score += language.dot(query, key, use_accelerator, dtype)
        peak = tl.maximum(max, tl.max(score, 1))
        alpha = tl.math.exp2(max - peak)
        beta = tl.math.exp2(score - peak[:, None])
        sum = sum * alpha + tl.sum(beta, 1)
        max = peak
        output *= alpha[:, None].to(dtype)
        value = tl.load(value_block_ptr)
        output += language.dot(beta.to(dtype), value, use_accelerator, dtype)
        key_block_ptr = tl.advance(key_block_ptr, (0, y_block_size))
        value_block_ptr = tl.advance(value_block_ptr, (y_block_size, 0))
        if mask_ptr is not None:
            mask_block_ptr = tl.advance(mask_block_ptr, (0, y_block_size))
    output /= sum[:, None].to(dtype)
    if dropout_p > language.eps:
        dropout_mask = tl.rand(seed, tl.arange(0, x_block_size) + y_offset
            ) > dropout_p
        dropout_scale = 1.0 - dropout_p + language.eps
        output = tl.where(dropout_mask, output / dropout_scale, 0.0)
    tl.store(output_block_ptr, output.to(dtype))
    log2sum = max + tl.math.log2(sum)
    tl.store(log2sum_block_ptr, log2sum.to(dtype))


def is_pow2(value):
    return False if value == 0 else value & value - 1 == 0


# Forward method (kernel launch code)
def _Attention_forward(ctx: Any, *args: Any, **kwargs: Any):
    (query, key, value, mask, dropout_p, is_causal, softmax_scale,
        use_accelerator) = args
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError('The dimension of query, key and value should be 4.')
    if not util.is_pow2(query.shape[-2]) or not util.is_pow2(query.shape[-1]
        ) or not util.is_pow2(key.shape[-2]) or not util.is_pow2(key.shape[-1]
        ) or not util.is_pow2(value.shape[-2]) or not util.is_pow2(value.
        shape[-1]):
        raise ValueError('Attention supports only for power of 2 size tensors.'
            )
    if mask is not None:
        if is_causal:
            raise ValueError(
                'Error because both attn_mask and is_causal are set.')
        if mask.dtype == torch.bool:
            raise ValueError('Boolean mask is not supported yet.')
    util.push_trace('Attention.__forward')
    output, log_sum_exp = Attention.__forward(query, key, value, mask,
        dropout_p, is_causal, softmax_scale, use_accelerator)
    util.pop_trace()
    ctx.save_for_backward(query, key, value, output, log_sum_exp)
    ctx.mask = mask
    ctx.dropout_p = dropout_p
    ctx.is_causal = is_causal
    ctx.softmax_scale = softmax_scale
    ctx.use_accelerator = use_accelerator
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@staticmethod
@triton.jit
def backward(grad_query_ptr: tl.tensor, grad_key_ptr: tl.tensor,
    grad_value_ptr: tl.tensor, grad_mask_ptr: tl.tensor, grad_output_ptr:
    tl.tensor, query_ptr: tl.tensor, key_ptr: tl.tensor, value_ptr: tl.
    tensor, y_size: tl.int32, x_size: tl.int32, head_stride: tl.int32,
    y_stride: tl.int32, x_stride: tl.int32, mask_ptr: tl.tensor,
    mask_head_stride: tl.int32, mask_y_stride: tl.int32, mask_x_stride: tl.
    int32, output_ptr: tl.tensor, log2sum_ptr: tl.tensor, delta_ptr: tl.
    tensor, dropout_p: tl.float32, is_causal: tl.constexpr, softmax_scale:
    tl.float32, use_accelerator: tl.constexpr, dtype: tl.constexpr,
    y_block_size: tl.constexpr, x_block_size: tl.constexpr):
    pid = tl.program_id(0)
    num_y_blocks = tl.cdiv(y_size, y_block_size)
    grad_key_ptr += pid * head_stride
    grad_value_ptr += pid * head_stride
    key_ptr += pid * head_stride
    value_ptr += pid * head_stride
    score_scale = softmax_scale * language.log2e
    n_strides = tl.arange(0, y_block_size) * y_stride
    x_strides = tl.arange(0, x_block_size) * x_stride
    for n_block in range(0, num_y_blocks):
        if is_causal:
            n_offset = n_block * y_block_size
        else:
            n_offset = 0
        grad_query_block_ptr = tl.make_block_ptr(grad_query_ptr + pid *
            head_stride, shape=(y_size, x_size), strides=(y_stride,
            x_stride), offsets=(n_offset, 0), block_shape=(y_block_size,
            x_block_size), order=(1, 0))
        grad_output_block_ptr = tl.make_block_ptr(grad_output_ptr + pid *
            head_stride, shape=(y_size, x_size), strides=(y_stride,
            x_stride), offsets=(n_offset, 0), block_shape=(y_block_size,
            x_block_size), order=(1, 0))
        query_block_ptr = tl.make_block_ptr(query_ptr + pid * head_stride,
            shape=(y_size, x_size), strides=(y_stride, x_stride), offsets=(
            n_offset, 0), block_shape=(y_block_size, x_block_size), order=(
            1, 0))
        output_block_ptr = tl.make_block_ptr(output_ptr + pid * head_stride,
            shape=(y_size, x_size), strides=(y_stride, x_stride), offsets=(
            n_offset, 0), block_shape=(y_block_size, x_block_size), order=(
            1, 0))
        log2sum_block_ptr = tl.make_block_ptr(log2sum_ptr + pid * y_size,
            shape=(y_size,), strides=(1,), offsets=(n_offset,), block_shape
            =(y_block_size,), order=(0,))
        delta_block_ptr = tl.make_block_ptr(delta_ptr + pid * y_size, shape
            =(y_size,), strides=(1,), offsets=(n_offset,), block_shape=(
            y_block_size,), order=(0,))
        if mask_ptr is not None:
            grad_mask_block_ptr = tl.make_block_ptr(grad_mask_ptr + pid *
                mask_head_stride, shape=(y_size, y_size), strides=(
                mask_y_stride, mask_x_stride), offsets=(0, n_block *
                y_block_size), block_shape=(y_block_size, y_block_size),
                order=(1, 0))
            mask_block_ptr = tl.make_block_ptr(mask_ptr + pid *
                mask_head_stride, shape=(y_size, y_size), strides=(
                mask_y_stride, mask_x_stride), offsets=(0, n_block *
                y_block_size), block_shape=(y_block_size, y_block_size),
                order=(1, 0))
        grad_value = tl.zeros((y_block_size, x_block_size), dtype)
        grad_key = tl.zeros((y_block_size, x_block_size), dtype)
        ptr_offsets = n_strides[:, None] + x_strides[None, :]
        key = tl.load(key_ptr + ptr_offsets)
        value = tl.load(value_ptr + ptr_offsets)
        n_offsets = n_offset + tl.arange(0, y_block_size)
        for m_offset in range(n_offset, y_size, y_block_size):
            query = tl.load(query_block_ptr)
            m_offsets = tl.arange(0, y_block_size) + m_offset
            if is_causal:
                condition = m_offsets[:, None] >= n_offsets[None, :]
                score = tl.where(condition, 0.0, float('-inf'))
            elif mask_ptr is not None:
                mask = tl.load(mask_block_ptr)
                mask *= language.log2e
                score = mask
            else:
                score = tl.zeros((y_block_size, y_block_size), dtype)
            score += language.dot(query, tl.trans(key), use_accelerator, dtype
                ) * score_scale
            log2sum = tl.load(log2sum_block_ptr)
            alpha = tl.math.exp2(score - log2sum[:, None]).to(dtype)
            grad_output = tl.load(grad_output_block_ptr)
            if dropout_p > language.eps:
                output = tl.load(output_block_ptr)
                dropout_scale = 1.0 - dropout_p + language.eps
                grad_dropout = tl.where(output > 0.0, dropout_scale, 0.0).to(
                    dtype)
                grad_output *= grad_dropout
            grad_value += language.dot(tl.trans(alpha), grad_output,
                use_accelerator, dtype)
            delta = tl.load(delta_block_ptr)
            grad_alpha = tl.zeros((y_block_size, y_block_size), dtype) - delta[
                :, None]
            grad_alpha += language.dot(grad_output, tl.trans(value),
                use_accelerator, dtype)
            grad_softmax = (alpha * grad_alpha * softmax_scale).to(dtype)
            grad_key += language.dot(tl.trans(grad_softmax), query,
                use_accelerator, dtype)
            grad_query = tl.load(grad_query_block_ptr)
            grad_query += language.dot(grad_softmax, key, use_accelerator,
                dtype)
            tl.store(grad_query_block_ptr, grad_query)
            grad_query_block_ptr = tl.advance(grad_query_block_ptr, (
                y_block_size, 0))
            grad_output_block_ptr = tl.advance(grad_output_block_ptr, (
                y_block_size, 0))
            query_block_ptr = tl.advance(query_block_ptr, (y_block_size, 0))
            output_block_ptr = tl.advance(output_block_ptr, (y_block_size, 0))
            delta_block_ptr = tl.advance(delta_block_ptr, (y_block_size,))
            log2sum_block_ptr = tl.advance(log2sum_block_ptr, (y_block_size,))
            if mask_ptr is not None:
                tl.store(grad_mask_block_ptr, (grad_softmax / softmax_scale
                    ).to(dtype))
                mask_block_ptr = tl.advance(mask_block_ptr, (y_block_size, 0))
                grad_mask_block_ptr = tl.advance(grad_mask_block_ptr, (
                    y_block_size, 0))
        tl.store(grad_key_ptr + ptr_offsets, grad_key)
        tl.store(grad_value_ptr + ptr_offsets, grad_value)
        n_strides += y_block_size * y_stride


@staticmethod
@util.autotune(softmax_configs(), ['x_size'])
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
    delta = tl.zeros((1, x_block_size), dtype)
    for _ in range(0, x_size, x_block_size):
        if require_x_boundary_check:
            grad_output = tl.load(grad_output_block_ptr, boundary_check=(1,
                ), padding_option='zero')
            output = tl.load(output_block_ptr, boundary_check=(1,))
        else:
            grad_output = tl.load(grad_output_block_ptr)
            output = tl.load(output_block_ptr)
        delta += grad_output * output
        output_block_ptr = tl.advance(output_block_ptr, (0, x_block_size))
        grad_output_block_ptr = tl.advance(grad_output_block_ptr, (0,
            x_block_size))
    delta = tl.sum(delta, 1)
    tl.store(delta_block_ptr, delta.to(dtype))


# Backward method (kernel launch code)
def _Attention_backward(ctx: Any, *grad_outputs: Any):
    grad_output, = grad_outputs
    query, key, value, output, log_sum_exp = ctx.saved_tensors
    util.push_trace('Attention.__backward')
    grad_query, grad_key, grad_value, grad_mask = Attention.__backward(
        grad_output, query, key, value, output, log_sum_exp, ctx.mask, ctx.
        softmax_scale, ctx.dropout_p, ctx.is_causal, ctx.use_accelerator)
    util.pop_trace()
    return grad_query, grad_key, grad_value, grad_mask, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class Attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        (query, key, value, mask, dropout_p, is_causal, softmax_scale,
            use_accelerator) = args
        if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
            raise ValueError(
                'The dimension of query, key and value should be 4.')
        if not util.is_pow2(query.shape[-2]) or not util.is_pow2(query.
            shape[-1]) or not util.is_pow2(key.shape[-2]) or not util.is_pow2(
            key.shape[-1]) or not util.is_pow2(value.shape[-2]
            ) or not util.is_pow2(value.shape[-1]):
            raise ValueError(
                'Attention supports only for power of 2 size tensors.')
        if mask is not None:
            if is_causal:
                raise ValueError(
                    'Error because both attn_mask and is_causal are set.')
            if mask.dtype == torch.bool:
                raise ValueError('Boolean mask is not supported yet.')
        util.push_trace('Attention.__forward')
        output, log_sum_exp = Attention.__forward(query, key, value, mask,
            dropout_p, is_causal, softmax_scale, use_accelerator)
        util.pop_trace()
        ctx.save_for_backward(query, key, value, output, log_sum_exp)
        ctx.mask = mask
        ctx.dropout_p = dropout_p
        ctx.is_causal = is_causal
        ctx.softmax_scale = softmax_scale
        ctx.use_accelerator = use_accelerator
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        grad_output, = grad_outputs
        query, key, value, output, log_sum_exp = ctx.saved_tensors
        util.push_trace('Attention.__backward')
        grad_query, grad_key, grad_value, grad_mask = Attention.__backward(
            grad_output, query, key, value, output, log_sum_exp, ctx.mask,
            ctx.softmax_scale, ctx.dropout_p, ctx.is_causal, ctx.
            use_accelerator)
        util.pop_trace()
        return (grad_query, grad_key, grad_value, grad_mask, None, None,
            None, None)

    @staticmethod
    def __forward(query: torch.Tensor, key: torch.Tensor, value: torch.
        Tensor, mask: torch.Tensor, dropout_p: torch.float32, is_causal:
        torch.bool, softmax_scale: torch.float32, use_accelerator: torch.bool):
        assert query.shape[-1] == key.shape[-1] and key.shape[-1
            ] == value.shape[-1]
        assert key.shape[-1] in {16, 32, 64, 128}
        factory_kwargs = {'device': query.device, 'dtype': query.dtype}
        num_batches, num_heads, y_size, x_size = query.shape
        output = torch.empty_like(query)
        log2sum = torch.empty(num_batches, num_heads, y_size, **factory_kwargs)

        def grid(meta):
            num_m_blocks = triton.cdiv(y_size, meta['y_block_size'])
            return num_batches * num_heads * num_m_blocks,
        util.push_trace('kernel.Attention.forward')
        kernel.Attention.forward[grid](output, log2sum, query, key, value,
            y_size, x_size, query.stride(1), query.stride(2), query.stride(
            3), mask, mask.stride(1) if mask is not None else 0, mask.
            stride(2) if mask is not None else 0, mask.stride(3) if mask is not
            None else 0, dropout_p, torch.random.seed(), is_causal,
            softmax_scale, use_accelerator, util.dtype(output.dtype),
            x_block_size=triton.next_power_of_2(x_size))
        util.pop_trace()
        return output, log2sum

    @staticmethod
    def __backward(grad_output: torch.Tensor, query: torch.Tensor, key:
        torch.Tensor, value: torch.Tensor, output: torch.Tensor, log2sum:
        torch.Tensor, mask: torch.Tensor, softmax_scale: torch.float32,
        dropout_p: torch.float32, is_causal: torch.bool, use_accelerator:
        torch.bool):
        num_batches, num_heads, y_size, x_size = output.shape
        grad_query = torch.zeros_like(query)
        grad_key = torch.empty_like(key)
        grad_value = torch.empty_like(value)
        grad_mask = torch.empty_like(mask) if mask is not None else None
        delta = torch.empty_like(log2sum)

        def grid(meta):
            return num_batches * num_heads * y_size,
        util.push_trace('kernel.Softmax.backward_delta')
        kernel.Softmax.backward_delta[grid](delta, output, grad_output, 
            num_batches * num_heads * y_size, x_size, x_size, 1, util.dtype
            (delta.dtype))
        util.pop_trace()

        def grid(meta):
            return num_batches * num_heads,
        util.push_trace('kernel.Attention.backward')
        kernel.Attention.backward[grid](grad_query, grad_key, grad_value,
            grad_mask, grad_output, query, key, value, y_size, x_size,
            query.stride(1), query.stride(2), query.stride(3), mask, mask.
            stride(1) if mask is not None else 0, mask.stride(2) if mask is not
            None else 0, mask.stride(3) if mask is not None else 0, output,
            log2sum, delta, dropout_p, is_causal, softmax_scale,
            use_accelerator, util.dtype(grad_query.dtype), 64, triton.
            next_power_of_2(x_size), num_warps=2)
        util.pop_trace()
        return grad_query, grad_key, grad_value, grad_mask
