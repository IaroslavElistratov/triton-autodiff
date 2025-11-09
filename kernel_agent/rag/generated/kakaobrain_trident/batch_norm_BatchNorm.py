# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/kakaobrain/trident
# Source-Files: trident/operation/batch_norm.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rl_ay0ux/trident-main/trident/operation/batch_norm.py
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
@triton.jit
def forward(output_ptr: tl.tensor, mean_ptr: tl.tensor, var_ptr: tl.tensor,
    input_ptr: tl.tensor, num_batches: tl.int32, y_size: tl.int32, x_size:
    tl.int32, weight_ptr: tl.tensor, bias_ptr: tl.tensor, running_mean_ptr:
    tl.tensor, running_var_ptr: tl.tensor, momentum: tl.float32, eps: tl.
    float32, dtype: tl.constexpr, batch_block_size: tl.constexpr,
    x_block_size: tl.constexpr):
    pid = tl.program_id(0)
    output_block_ptr = tl.make_block_ptr(output_ptr + pid * x_size, shape=(
        num_batches, x_size), strides=(y_size * x_size, 1), offsets=(0, 0),
        block_shape=(batch_block_size, x_block_size), order=(1, 0))
    input_block_ptr = tl.make_block_ptr(input_ptr + pid * x_size, shape=(
        num_batches, x_size), strides=(y_size * x_size, 1), offsets=(0, 0),
        block_shape=(batch_block_size, x_block_size), order=(1, 0))
    batch_condition = tl.arange(0, batch_block_size) < num_batches
    x_condition = tl.arange(0, x_block_size) < x_size
    condition = batch_condition[:, None] & x_condition[None, :]
    denominator = num_batches * x_size
    input = tl.load(input_block_ptr, boundary_check=(0, 1), padding_option=
        'zero')
    mean = tl.sum(input / denominator)
    deviation = tl.where(condition, input - mean, 0)
    var = tl.sum(deviation * deviation / denominator)
    std = tl.sqrt(var + eps)
    output = (input - mean) / std
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + pid)
        output = output * weight
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + pid)
        output = output + bias
    tl.store(output_block_ptr, output.to(dtype), boundary_check=(0, 1))
    tl.store(mean_ptr + pid, mean)
    tl.store(var_ptr + pid, var)
    if running_mean_ptr is not None:
        running_mean = tl.load(running_mean_ptr + pid)
        tl.store(running_mean_ptr + pid, running_mean * (1 - momentum) + 
            mean * momentum)
    if running_var_ptr is not None:
        running_var = tl.load(running_var_ptr + pid)
        tl.store(running_var_ptr + pid, running_var * (1 - momentum) + var *
            (denominator / (denominator - 1)) * momentum)


# Forward method (kernel launch code)
def _BatchNorm_forward(ctx: Any, *args: Any, **kwargs: Any):
    input, running_mean, running_var, weight, bias, momentum, eps = args
    util.push_trace('BatchNorm.__forward')
    output, mean, var = BatchNorm.__forward(input, running_mean,
        running_var, weight, bias, momentum, eps)
    util.pop_trace()
    ctx.save_for_backward(input, weight, bias, mean, var)
    ctx.eps = eps
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@staticmethod
@triton.jit
def backward(grad_input_ptr: tl.tensor, grad_weight_ptr: tl.tensor,
    grad_bias_ptr: tl.tensor, grad_output_ptr: tl.tensor, input_ptr: tl.
    tensor, weight_ptr: tl.tensor, mean_ptr: tl.tensor, var_ptr: tl.tensor,
    num_batches: tl.int32, y_size: tl.int32, x_size: tl.int32, eps: tl.
    float32, dtype: tl.constexpr, batch_block_size: tl.constexpr,
    x_block_size: tl.constexpr):
    pid = tl.program_id(0)
    grad_input_block_ptr = tl.make_block_ptr(grad_input_ptr + pid * x_size,
        shape=(num_batches, x_size), strides=(y_size * x_size, 1), offsets=
        (0, 0), block_shape=(batch_block_size, x_block_size), order=(1, 0))
    grad_output_block_ptr = tl.make_block_ptr(grad_output_ptr + pid *
        x_size, shape=(num_batches, x_size), strides=(y_size * x_size, 1),
        offsets=(0, 0), block_shape=(batch_block_size, x_block_size), order
        =(1, 0))
    input_block_ptr = tl.make_block_ptr(input_ptr + pid * x_size, shape=(
        num_batches, x_size), strides=(y_size * x_size, 1), offsets=(0, 0),
        block_shape=(batch_block_size, x_block_size), order=(1, 0))
    batch_condition = tl.arange(0, batch_block_size) < num_batches
    x_condition = tl.arange(0, x_block_size) < x_size
    condition = batch_condition[:, None] & x_condition[None, :]
    denominator = num_batches * x_size
    grad_output = tl.load(grad_output_block_ptr, boundary_check=(0, 1),
        padding_option='zero')
    input = tl.load(input_block_ptr, boundary_check=(0, 1), padding_option=
        'zero')
    weight = tl.load(weight_ptr + pid) if weight_ptr is not None else 1
    mean = tl.load(mean_ptr + pid)
    var = tl.load(var_ptr + pid)
    std = tl.sqrt(var + eps)
    centered_mean = tl.where(condition, input - mean, 0)
    grad_norm = weight * grad_output
    grad_std = -tl.sum(grad_norm * centered_mean / (std * std))
    grad_var = 0.5 * grad_std / std
    grad_centered_mean = (grad_norm / std + 2.0 / denominator *
        centered_mean * grad_var)
    grad_mean = tl.sum(tl.where(condition, grad_centered_mean, 0.0) /
        denominator)
    grad_input = grad_centered_mean - grad_mean
    tl.store(grad_input_block_ptr, grad_input.to(dtype), boundary_check=(0, 1))
    if grad_weight_ptr:
        input_norm = centered_mean / std
        grad_weight = tl.sum(input_norm * grad_output)
        tl.store(grad_weight_ptr + pid, grad_weight.to(dtype))
    if grad_bias_ptr:
        grad_bias = tl.sum(grad_output)
        tl.store(grad_bias_ptr + pid, grad_bias)


# Backward method (kernel launch code)
def _BatchNorm_backward(ctx: Any, *grad_outputs: Any):
    util.push_trace('BatchNorm.__backward')
    grad_input, grad_weight, grad_bias = BatchNorm.__backward(*grad_outputs,
        *ctx.saved_tensors, ctx.eps)
    util.pop_trace()
    return grad_input, None, None, grad_weight, grad_bias, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class BatchNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, running_mean, running_var, weight, bias, momentum, eps = args
        util.push_trace('BatchNorm.__forward')
        output, mean, var = BatchNorm.__forward(input, running_mean,
            running_var, weight, bias, momentum, eps)
        util.pop_trace()
        ctx.save_for_backward(input, weight, bias, mean, var)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        util.push_trace('BatchNorm.__backward')
        grad_input, grad_weight, grad_bias = BatchNorm.__backward(*
            grad_outputs, *ctx.saved_tensors, ctx.eps)
        util.pop_trace()
        return grad_input, None, None, grad_weight, grad_bias, None, None

    @staticmethod
    def __forward(input: torch.Tensor, running_mean: torch.Tensor,
        running_var: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
        momentum: torch.float32, eps: torch.float32):
        num_batches, y_size, x_size = input.shape
        factory_kwargs = {'device': input.device, 'dtype': input.dtype}

        def grid(meta):
            return y_size,
        output = torch.empty_like(input)
        mean = torch.empty(y_size, **factory_kwargs)
        var = torch.empty(y_size, **factory_kwargs)
        util.push_trace('kernel.BatchNorm.forward')
        kernel.BatchNorm.forward[grid](output, mean, var, input,
            num_batches, y_size, x_size, weight, bias, running_mean,
            running_var, momentum, eps, util.dtype(input.dtype),
            batch_block_size=triton.next_power_of_2(num_batches),
            x_block_size=triton.next_power_of_2(x_size))
        util.pop_trace()
        return output, mean, var

    @staticmethod
    def __backward(grad_output: torch.Tensor, input: torch.Tensor, weight:
        torch.Tensor, bias: torch.Tensor, mean: torch.Tensor, var: torch.
        Tensor, eps: torch.float32):
        num_batches, y_size, x_size = input.shape

        def grid(meta):
            return y_size,
        grad_input = torch.empty_like(input)
        grad_weight = torch.empty_like(weight) if weight is not None else None
        grad_bias = torch.empty_like(bias) if bias is not None else None
        util.push_trace('kernel.BatchNorm.backward')
        kernel.BatchNorm.backward[grid](grad_input, grad_weight, grad_bias,
            grad_output, input, weight, mean, var, num_batches, y_size,
            x_size, eps, util.dtype(input.dtype), batch_block_size=triton.
            next_power_of_2(num_batches), x_block_size=triton.
            next_power_of_2(x_size))
        util.pop_trace()
        return grad_input, grad_weight, grad_bias
