# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/kakaobrain/trident
# Source-Files: trident/operation/instance_norm.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rl_ay0ux/trident-main/trident/operation/instance_norm.py
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
    input_ptr: tl.tensor, y_size: tl.int32, x_size: tl.int32,
    running_mean_ptr: tl.tensor, running_var_ptr: tl.tensor, weight_ptr: tl
    .tensor, bias_ptr: tl.tensor, use_input_stats: tl.int32, eps: tl.
    float32, dtype: tl.constexpr, x_block_size: tl.constexpr):
    pid = tl.program_id(0)
    batch = pid // y_size
    y_offset = pid % y_size
    output_block_ptr = tl.make_block_ptr(output_ptr + batch * y_size *
        x_size, shape=(y_size, x_size), strides=(x_size, 1), offsets=(
        y_offset, 0), block_shape=(1, x_block_size), order=(1, 0))
    input_block_ptr = tl.make_block_ptr(input_ptr + batch * y_size * x_size,
        shape=(y_size, x_size), strides=(x_size, 1), offsets=(y_offset, 0),
        block_shape=(1, x_block_size), order=(1, 0))
    input = tl.load(input_block_ptr, boundary_check=(1,), padding_option='zero'
        )
    if use_input_stats == 1:
        mean_block_ptr = tl.make_block_ptr(mean_ptr + batch * y_size, shape
            =(y_size,), strides=(1,), offsets=(y_offset,), block_shape=(1,),
            order=(0,))
        mean = tl.sum(input / x_size, 1).to(dtype)
        tl.store(mean_block_ptr, mean)
    else:
        running_mean_block_ptr = tl.make_block_ptr(running_mean_ptr, shape=
            (y_size,), strides=(1,), offsets=(y_offset,), block_shape=(1,),
            order=(0,))
        mean = tl.load(running_mean_block_ptr)
    condition = tl.arange(0, x_block_size) < x_size
    centered_mean = tl.where(condition, input - mean, 0)
    if use_input_stats == 1:
        var_block_ptr = tl.make_block_ptr(var_ptr + batch * y_size, shape=(
            y_size,), strides=(1,), offsets=(y_offset,), block_shape=(1,),
            order=(0,))
        var = tl.sum(centered_mean * centered_mean / x_size, 1).to(dtype)
        tl.store(var_block_ptr, var)
    else:
        running_var_block_ptr = tl.make_block_ptr(running_var_ptr, shape=(
            y_size,), strides=(1,), offsets=(y_offset,), block_shape=(1,),
            order=(0,))
        var = tl.load(running_var_block_ptr)
    rstd = tl.math.rsqrt(var + eps)
    output = centered_mean * rstd
    if weight_ptr is not None:
        weight_block_ptr = tl.make_block_ptr(weight_ptr, shape=(y_size,),
            strides=(1,), offsets=(y_offset,), block_shape=(1,), order=(0,))
        weight = tl.load(weight_block_ptr)
        output *= weight
    if bias_ptr is not None:
        bias_block_ptr = tl.make_block_ptr(bias_ptr, shape=(y_size,),
            strides=(1,), offsets=(y_offset,), block_shape=(1,), order=(0,))
        bias = tl.load(bias_block_ptr)
        output += bias
    tl.store(output_block_ptr, output.to(dtype), boundary_check=(1,))


@staticmethod
@triton.jit
def forward_running_mean_running_var(mean_ptr: tl.tensor, var_ptr: tl.
    tensor, running_mean_ptr: tl.tensor, running_var_ptr: tl.tensor,
    num_batches: tl.int32, y_size: tl.int32, x_size: tl.int32, momentum: tl
    .constexpr, batch_block_size: tl.constexpr):
    y_offset = tl.program_id(0)
    mean_block_ptr = tl.make_block_ptr(mean_ptr, shape=(num_batches, y_size
        ), strides=(y_size, 1), offsets=(0, y_offset), block_shape=(
        batch_block_size, 1), order=(0, 1))
    var_block_ptr = tl.make_block_ptr(var_ptr, shape=(num_batches, y_size),
        strides=(y_size, 1), offsets=(0, y_offset), block_shape=(
        batch_block_size, 1), order=(0, 1))
    running_mean_block_ptr = tl.make_block_ptr(running_mean_ptr, shape=(
        y_size,), strides=(1,), offsets=(y_offset,), block_shape=(1,),
        order=(0,))
    running_var_block_ptr = tl.make_block_ptr(running_var_ptr, shape=(
        y_size,), strides=(1,), offsets=(y_offset,), block_shape=(1,),
        order=(0,))
    mean = tl.load(mean_block_ptr, boundary_check=(0,))
    mean = tl.sum(mean, 0) / num_batches
    var = tl.load(var_block_ptr, boundary_check=(0,))
    var = tl.sum(var, 0) / num_batches
    running_mean = tl.load(running_mean_block_ptr, boundary_check=(0,))
    running_mean = mean * momentum + running_mean * (1 - momentum)
    running_var = tl.load(running_var_block_ptr, boundary_check=(0,))
    running_var = var * (x_size / (x_size - 1)) * momentum + running_var * (
        1 - momentum)
    tl.store(running_mean_block_ptr, running_mean, boundary_check=(0,))
    tl.store(running_var_block_ptr, running_var, boundary_check=(0,))


# Forward method (kernel launch code)
def _InstanceNorm_forward(ctx: Any, *args: Any, **kwargs: Any):
    (input, running_mean, running_var, weight, bias, use_input_stats,
        momentum, eps) = args
    util.push_trace('InstanceNorm.__forward')
    output, mean, var = InstanceNorm.__forward(input, running_mean,
        running_var, weight, bias, use_input_stats, momentum, eps)
    util.pop_trace()
    ctx.save_for_backward(input, running_mean, running_var, weight, mean,
        var, weight, bias)
    ctx.use_input_stats = use_input_stats
    ctx.eps = eps
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@staticmethod
@triton.jit
def backward(grad_input_ptr: tl.tensor, grad_weight_staging_ptr: tl.tensor,
    grad_bias_staging_ptr: tl.tensor, grad_output_ptr: tl.tensor, input_ptr:
    tl.tensor, y_size: tl.int32, x_size: tl.int32, running_mean_ptr: tl.
    tensor, running_var_ptr: tl.tensor, mean_ptr: tl.tensor, var_ptr: tl.
    tensor, weight_ptr: tl.tensor, use_input_stats: tl.int32, eps: tl.
    float32, dtype: tl.constexpr, x_block_size: tl.constexpr):
    pid = tl.program_id(0)
    batch = pid // y_size
    y_offset = pid % y_size
    grad_input_block_ptr = tl.make_block_ptr(grad_input_ptr + batch *
        y_size * x_size, shape=(y_size, x_size), strides=(x_size, 1),
        offsets=(y_offset, 0), block_shape=(1, x_block_size), order=(1, 0))
    grad_output_block_ptr = tl.make_block_ptr(grad_output_ptr + batch *
        y_size * x_size, shape=(y_size, x_size), strides=(x_size, 1),
        offsets=(y_offset, 0), block_shape=(1, x_block_size), order=(1, 0))
    input_block_ptr = tl.make_block_ptr(input_ptr + batch * y_size * x_size,
        shape=(y_size, x_size), strides=(x_size, 1), offsets=(y_offset, 0),
        block_shape=(1, x_block_size), order=(1, 0))
    grad_output = tl.load(grad_output_block_ptr, boundary_check=(1,))
    input = tl.load(input_block_ptr, boundary_check=(1,))
    if use_input_stats == 1:
        mean_block_ptr = tl.make_block_ptr(mean_ptr + batch * y_size, shape
            =(y_size,), strides=(1,), offsets=(y_offset,), block_shape=(1,),
            order=(0,))
        var_block_ptr = tl.make_block_ptr(var_ptr + batch * y_size, shape=(
            y_size,), strides=(1,), offsets=(y_offset,), block_shape=(1,),
            order=(0,))
        mean = tl.load(mean_block_ptr)
        var = tl.load(var_block_ptr)
    else:
        running_mean_block_ptr = tl.make_block_ptr(running_mean_ptr, shape=
            (y_size,), strides=(1,), offsets=(y_offset,), block_shape=(1,),
            order=(0,))
        running_var_block_ptr = tl.make_block_ptr(running_var_ptr, shape=(
            y_size,), strides=(1,), offsets=(y_offset,), block_shape=(1,),
            order=(0,))
        mean = tl.load(running_mean_block_ptr)
        var = tl.load(running_var_block_ptr)
    condition = tl.arange(0, x_block_size) < x_size
    centered_mean = tl.where(condition, input - mean, 0)
    if weight_ptr is not None:
        weight_block_ptr = tl.make_block_ptr(weight_ptr, shape=(y_size,),
            strides=(1,), offsets=(y_offset,), block_shape=(1,), order=(0,))
        weight = tl.load(weight_block_ptr)
        grad_norm = weight * grad_output
    else:
        grad_norm = grad_output
    rstd = tl.math.rsqrt(var + eps)
    grad_std = tl.sum(grad_norm * centered_mean, 1)
    grad_var = grad_std * -(0.5 * rstd * rstd * rstd) / x_size
    grad_distance = 2 * centered_mean * grad_var
    grad_centered_mean = tl.where(condition, grad_norm * rstd +
        grad_distance, 0)
    grad_mean = -tl.sum(grad_centered_mean, 1) / x_size
    grad_input = grad_centered_mean + grad_mean
    tl.store(grad_input_block_ptr, grad_input.to(dtype), boundary_check=(1,))
    if grad_weight_staging_ptr is not None:
        norm = centered_mean * rstd
        grad_weight = norm * grad_output
        grad_weight_staging_block_ptr = tl.make_block_ptr(
            grad_weight_staging_ptr + batch * y_size, shape=(y_size,),
            strides=(1,), offsets=(y_offset,), block_shape=(1,), order=(0,))
        grad_weight = tl.sum(grad_weight, 1)
        tl.store(grad_weight_staging_block_ptr, grad_weight.to(dtype))
    if grad_bias_staging_ptr is not None:
        grad_bias_staging_block_ptr = tl.make_block_ptr(
            grad_bias_staging_ptr + batch * y_size, shape=(y_size,),
            strides=(1,), offsets=(y_offset,), block_shape=(1,), order=(0,))
        grad_bias = tl.sum(grad_output, 1)
        tl.store(grad_bias_staging_block_ptr, grad_bias.to(dtype))


# Backward method (kernel launch code)
def _InstanceNorm_backward(ctx: Any, *grad_outputs: Any):
    input, running_mean, running_var, weight, mean, var, weight, bias = (ctx
        .saved_tensors)
    grad_output, = grad_outputs
    util.push_trace('InstanceNorm.__backward')
    grad_input, grad_weight, grad_bias = InstanceNorm.__backward(grad_output,
        input, running_mean, running_var, mean, var, weight, bias, ctx.
        use_input_stats, ctx.eps)
    util.pop_trace()
    return (grad_input, None, None, grad_weight, grad_bias, None, None,
        None, None)


# ============================================================
# autograd.Function Class Definition
# ============================================================

class InstanceNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        (input, running_mean, running_var, weight, bias, use_input_stats,
            momentum, eps) = args
        util.push_trace('InstanceNorm.__forward')
        output, mean, var = InstanceNorm.__forward(input, running_mean,
            running_var, weight, bias, use_input_stats, momentum, eps)
        util.pop_trace()
        ctx.save_for_backward(input, running_mean, running_var, weight,
            mean, var, weight, bias)
        ctx.use_input_stats = use_input_stats
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        (input, running_mean, running_var, weight, mean, var, weight, bias
            ) = ctx.saved_tensors
        grad_output, = grad_outputs
        util.push_trace('InstanceNorm.__backward')
        grad_input, grad_weight, grad_bias = InstanceNorm.__backward(
            grad_output, input, running_mean, running_var, mean, var,
            weight, bias, ctx.use_input_stats, ctx.eps)
        util.pop_trace()
        return (grad_input, None, None, grad_weight, grad_bias, None, None,
            None, None)

    @staticmethod
    def __forward(input: torch.Tensor, running_mean: torch.Tensor,
        running_var: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
        use_input_stats: torch.bool, momentum: torch.float32, eps: torch.
        float32):
        factory_kwargs = {'device': input.device, 'dtype': input.dtype}
        num_batches, y_size, x_size = input.shape
        output = torch.empty_like(input)
        mean = torch.empty(num_batches, y_size, **factory_kwargs)
        var = torch.empty(num_batches, y_size, **factory_kwargs)

        def grid(meta):
            return num_batches * y_size,
        util.push_trace('kernel.InstanceNorm.forward')
        kernel.InstanceNorm.forward[grid](output, mean, var, input, y_size,
            x_size, running_mean, running_var, weight, bias, 1 if
            use_input_stats else 0, eps, util.dtype(input.dtype), triton.
            next_power_of_2(x_size))
        util.pop_trace()
        if (use_input_stats and running_mean is not None and running_var is not
            None):

            def grid(meta):
                return y_size,
            util.push_trace(
                'kernel.InstanceNorm.forward_running_mean_running_var')
            kernel.InstanceNorm.forward_running_mean_running_var[grid](mean,
                var, running_mean, running_var, num_batches, y_size, x_size,
                momentum, triton.next_power_of_2(num_batches))
            util.pop_trace()
        return output, mean, var

    @staticmethod
    def __backward(grad_output: torch.Tensor, input: torch.Tensor,
        running_mean: torch.Tensor, running_var: torch.Tensor, mean: torch.
        Tensor, var: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
        use_input_stats: torch.bool, eps: torch.float):
        factory_kwargs = {'device': input.device, 'dtype': input.dtype}
        num_batches, y_size, x_size = input.shape
        grad_input = torch.empty_like(input)
        if weight is not None:
            grad_weight_staging = torch.zeros(num_batches, y_size, **
                factory_kwargs)
        else:
            grad_weight_staging = None
        if bias is not None:
            grad_bias_staging = torch.zeros(num_batches, y_size, **
                factory_kwargs)
        else:
            grad_bias_staging = None

        def grid(meta):
            return num_batches * y_size,
        util.push_trace('kernel.InstanceNorm.backward')
        kernel.InstanceNorm.backward[grid](grad_input, grad_weight_staging,
            grad_bias_staging, grad_output, input, y_size, x_size,
            running_mean, running_var, mean, var, weight, 1 if
            use_input_stats else 0, eps, util.dtype(grad_input.dtype),
            triton.next_power_of_2(x_size))
        util.pop_trace()
        if grad_weight_staging is not None:
            util.push_trace('torch.sum')
            grad_weight = torch.sum(grad_weight_staging, 0)
            util.pop_trace()
        else:
            grad_weight = None
        if grad_bias_staging is not None:
            util.push_trace('torch.sum')
            grad_bias = torch.sum(grad_bias_staging, 0)
            util.pop_trace()
        else:
            grad_bias = None
        return grad_input, grad_weight, grad_bias
