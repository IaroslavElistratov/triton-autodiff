# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/BobMcDear/attorch
# Source-Files: attorch/linear_layer.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_leort3lx/attorch-main/attorch/linear_layer.py
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
from triton import cdiv
from math import log

@triton.jit
def relu(input):
    """
    Applies ReLU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by ReLU.
    """
    return tl.maximum(0, input)


@triton.jit
def relu6(input):
    """
    Applies ReLU6 to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by ReLU6.
    """
    return tl.minimum(relu(input), 6)


@triton.jit
def sigmoid(input):
    """
    Applies sigmoid to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by sigmoid.
    """
    return 1 / (1 + tl.exp(-input))


@triton.jit
def tanh(input):
    """
    Applies tanh to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by tanh.
    """
    return 2 * sigmoid(2 * input) - 1


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def apply_act_func(input, drop_p, seed, offset, param, act_func: tl.
    constexpr, dropout: tl.constexpr):
    """
    Applies an activation function to the input, optionally fusing dropout.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        drop_p: Probability of dropping an element if dropout is True.
        seed: Seed for generating the dropout mask if dropout is True.
        offset: Offset to generate the dropout mask for if dropout is True.
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'logsigmoid', 'tanh', 'relu', 'gelu', 'geluapprox', 'silu',
            'relu6', 'hardsigmoid', 'hardtanh', 'hardswish', 'selu', 'mish',
            'softplus', 'softsign', 'tanhshrink', 'leaky_relu', 'elu', 'celu', 'hardshrink',
            and 'softshrink'.
        dropout: Flag for performing dropout on the activation output.

    Returns:
        Input transformed by the desired activation function,
        potentially with fused dropout.
    """
    if act_func == 'sigmoid':
        input = input.to(tl.float32)
        output = sigmoid(input)
    if act_func == 'logsigmoid':
        input = input.to(tl.float32)
        output = logsigmoid(input)
    elif act_func == 'tanh':
        input = input.to(tl.float32)
        output = tanh(input)
    elif act_func == 'relu':
        output = relu(input)
    elif act_func == 'gelu':
        input = input.to(tl.float32)
        output = gelu(input)
    elif act_func == 'geluapprox':
        input = input.to(tl.float32)
        output = geluapprox(input)
    elif act_func == 'silu':
        input = input.to(tl.float32)
        output = silu(input)
    elif act_func == 'relu6':
        output = relu6(input)
    elif act_func == 'hardsigmoid':
        output = hardsigmoid(input)
    elif act_func == 'hardtanh':
        output = hardtanh(input)
    elif act_func == 'hardswish':
        output = hardswish(input)
    elif act_func == 'selu':
        input = input.to(tl.float32)
        output = selu(input)
    elif act_func == 'mish':
        input = input.to(tl.float32)
        output = mish(input)
    elif act_func == 'softplus':
        input = input.to(tl.float32)
        output = softplus(input)
    elif act_func == 'softsign':
        output = softsign(input)
    elif act_func == 'tanhshrink':
        input = input.to(tl.float32)
        output = tanhshrink(input)
    elif act_func == 'leaky_relu':
        output = leaky_relu(input, param)
    elif act_func == 'elu':
        input = input.to(tl.float32)
        output = elu(input, param)
    elif act_func == 'celu':
        input = input.to(tl.float32)
        output = celu(input, param)
    elif act_func == 'hardshrink':
        output = hardshrink(input, param)
    elif act_func == 'softshrink':
        output = softshrink(input, param)
    if dropout:
        output = apply_dropout(output, drop_p, seed, offset)
    return output


@triton.jit
def celu(input, alpha):
    """
    Applies CELU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        alpha: Alpha value.

    Returns:
        Input transformed by CELU.
    """
    return relu(input) + tl.minimum(0, alpha * (tl.exp(input / alpha) - 1))


@triton.jit
def elu(input, alpha):
    """
    Applies ELU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        alpha: Alpha value.

    Returns:
        Input transformed by ELU.
    """
    return tl.where(input <= 0, alpha * (tl.exp(input) - 1), input)


@triton.jit
def gelu(input):
    """
    Applies GELU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by GELU.
    """
    cdf = 0.5 * (1 + tl.math.erf(0.707106781 * input))
    return cdf * input


@triton.jit
def geluapprox(input):
    """
    Applies the tanh approximation of GELU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by the tanh approximation of GELU.
    """
    cdf = 0.5 * (1 + tanh(0.7978845608 * input * (1 + 0.044715 * input *
        input)))
    return cdf * input


@triton.jit
def hardshrink(input, lambd):
    """
    Applies hard shrink to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        lambd: Lambda value.

    Returns:
        Input transformed by hard shrink.
    """
    return tl.where(tl.abs(input) < lambd, 0, input)


@triton.jit
def hardsigmoid(input):
    """
    Applies hard sigmoid to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by hard sigmoid.
    """
    return tl.maximum(0, tl.minimum(1, input / 6 + 0.5))


@triton.jit
def hardswish(input):
    """
    Applies hard Swish to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by hard Swish.
    """
    return input * relu6(input + 3) / 6


@triton.jit
def hardtanh(input):
    """
    Applies hard tanh to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by hard tanh.
    """
    return tl.maximum(-1, tl.minimum(1, input))


@triton.jit
def leaky_relu(input, negative_slope):
    """
    Applies leaky ReLU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        negative_slope: Slope of the negative component.

    Returns:
        Input transformed by leaky ReLU.
    """
    return relu(input) + negative_slope * tl.minimum(0, input)


@triton.jit
def logsigmoid(input):
    """
    Applies the log of sigmoid to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by the log of sigmoid.
    """
    return tl.log(sigmoid(input))


@triton.jit
def mish(input):
    """
    Applies Mish to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by Mish.
    """
    return input * tanh(tl.log(1 + tl.exp(input)))


@triton.jit
def selu(input):
    """
    Applies SELU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by SELU.
    """
    scale = 1.0507009873554805
    alpha = 1.6732632423543772
    return scale * (tl.maximum(0, input) + tl.minimum(0, alpha * (tl.exp(
        input) - 1)))


@triton.jit
def silu(input):
    """
    Applies SiLU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by SiLU.
    """
    return input * sigmoid(input)


@triton.jit
def softplus(input):
    """
    Applies softplus to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by softplus.
    """
    return tl.log(1 + tl.exp(input))


@triton.jit
def softshrink(input, lambd):
    """
    Applies softshrink to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        lambd: Lambda value.

    Returns:
        Input transformed by softshrink.
    """
    return tl.where(input > lambd, input - lambd, tl.where(input < -lambd, 
        input + lambd, 0))


@triton.jit
def softsign(input):
    """
    Applies softsign to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by softsign.
    """
    return input / (1 + tl.abs(input))


@triton.jit
def tanhshrink(input):
    """
    Applies tanh shrink to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by tanh shrink.
    """
    return input - tanh(input)


@triton.jit
def apply_dropout(input, drop_p, seed, offset):
    """
    Randomly zeroes elements in the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        drop_p: Probability of dropping an element.
        seed: Seed for generating the dropout mask.
        offset: Offset to generate the mask for.

    Returns:
        Input with elements randomly zeroed out.
    """
    random = tl.rand(seed, offset)
    return tl.where(random < drop_p, 0, input / (1 - drop_p))


@triton.autotune(configs=[linear_forward_config(32, 32, 32, n_warps=2,
    n_stages=2), linear_forward_config(64, 32, 32, n_warps=2, n_stages=5),
    linear_forward_config(64, 32, 128, n_warps=4, n_stages=4),
    linear_forward_config(64, 32, 256, n_warps=4, n_stages=4),
    linear_forward_config(128, 32, 32, n_warps=4, n_stages=4),
    linear_forward_config(128, 32, 64, n_warps=4, n_stages=4),
    linear_forward_config(128, 32, 128, n_warps=4, n_stages=4),
    linear_forward_config(128, 64, 256, n_warps=8, n_stages=3)], key=[
    'batch_dim', 'in_feat_dim', 'out_feat_dim', 'fp16'])
@triton.heuristics({'tf32': lambda _: allow_tf32()})
@triton.jit
def linear_forward_kernel(input_pointer, weight_pointer, bias_pointer,
    pre_act_pointer, output_pointer, batch_dim, in_feat_dim, out_feat_dim,
    input_batch_stride, input_in_feat_stride, weight_in_feat_stride,
    weight_out_feat_stride, pre_act_batch_stride, pre_act_out_feat_stride,
    output_batch_stride, output_out_feat_stride, param, add_bias: tl.
    constexpr, act_func: tl.constexpr, save_pre_act: tl.constexpr, fp16: tl
    .constexpr, tf32: tl.constexpr, BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_IN_FEAT: tl.constexpr, BLOCK_SIZE_OUT_FEAT: tl.constexpr,
    GROUP_SIZE_BATCH: tl.constexpr):
    """
    Linearly transforms the input using weights, optionally adding bias
    and fusing an activation function.

    Args:
        input_pointer: Pointer to the input to transform.
            The input must be of shape [batch_dim, in_feat_dim].
        weight_pointer: Pointer to the weights input is transformed by.
            The weights must be of shape [in_feat_dim, out_feat_dim].
        bias_pointer: Pointer to an optional additive bias vector.
            The bias vector, if provided, must be of shape [out_feat_dim].
        pre_act_pointer: Pointer to an optional container the pre-activation input
            is written to if act_func is not None and save_pre_act is True.
            The container, if provided, must be of shape [batch_dim, out_feat_dim].
        output_pointer: Pointer to a container the result is written to.
            The container must be of shape [batch_dim, out_feat_dim].
        batch_dim: Batch dimension of the input and output.
        in_feat_dim: Dimensionality of the input features.
        out_feat_dim: Dimensionality of the output features.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_in_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        weight_in_feat_stride: Stride necessary to jump one element along the
            weights' input feature dimension.
        weight_out_feat_stride: Stride necessary to jump one element along the
            weights' output feature dimension.
        pre_act_batch_stride: Stride necessary to jump one element along the
            pre-activation input container's batch dimension.
        pre_act_out_feat_stride: Stride necessary to jump one element along the
            pre-activation input container's feature dimension.
        output_batch_stride: Stride necessary to jump one element along the
            output container's batch dimension.
        output_out_feat_stride: Stride necessary to jump one element along the
            output container's feature dimension.
        param: Parameter in the case of parameterized activation functions.
        add_bias: Flag for adding a bias vector.
        act_func: Name of activation function to apply, with None for identity.
            Options are 'sigmoid', 'logsigmoid', 'tanh', 'relu', 'gelu', 'geluapprox', 'silu',
            'softplus', 'softsign', 'tanhshrink', 'leaky_relu', 'elu', 'celu', 'hardshrink',
            and 'softshrink'.
        save_pre_act: Flag for saving the pre-activation input.
        fp16: Flag for loading the input, weights, and bias in FP16.
        tf32: Flag for performing matrix products in TF32.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_IN_FEAT: Block size across the input feature dimension.
        BLOCK_SIZE_OUT_FEAT: Block size across the output feature dimension.
        GROUP_SIZE_BATCH: Group size across the batch dimension.
    """
    pid = tl.program_id(axis=0)
    n_batch_pids = tl.cdiv(batch_dim, BLOCK_SIZE_BATCH)
    n_out_feat_pids = tl.cdiv(out_feat_dim, BLOCK_SIZE_OUT_FEAT)
    pids_per_group = GROUP_SIZE_BATCH * n_out_feat_pids
    group_id = pid // pids_per_group
    first_batch_pid = group_id * GROUP_SIZE_BATCH
    GROUP_SIZE_BATCH = min(n_batch_pids - first_batch_pid, GROUP_SIZE_BATCH)
    batch_pid = first_batch_pid + pid % GROUP_SIZE_BATCH
    out_feat_pid = pid % pids_per_group // GROUP_SIZE_BATCH
    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH
        )
    out_feat_offset = out_feat_pid * BLOCK_SIZE_OUT_FEAT + tl.arange(0,
        BLOCK_SIZE_OUT_FEAT)
    batch_mask = batch_offset < batch_dim
    out_feat_mask = out_feat_offset < out_feat_dim
    input_pointer += input_batch_stride * batch_offset[:, None]
    weight_pointer += weight_out_feat_stride * out_feat_offset[None, :]
    accum = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_OUT_FEAT), dtype=tl.float32)
    for block_ind in range(0, tl.cdiv(in_feat_dim, BLOCK_SIZE_IN_FEAT)):
        in_feat_offset = block_ind * BLOCK_SIZE_IN_FEAT + tl.arange(0,
            BLOCK_SIZE_IN_FEAT)
        in_feat_mask = in_feat_offset < in_feat_dim
        curr_input_pointer = (input_pointer + input_in_feat_stride *
            in_feat_offset[None, :])
        curr_weight_pointer = (weight_pointer + weight_in_feat_stride *
            in_feat_offset[:, None])
        input_block = tl.load(curr_input_pointer, mask=batch_mask[:, None] &
            in_feat_mask[None, :])
        weight_block = tl.load(curr_weight_pointer, mask=out_feat_mask[None,
            :] & in_feat_mask[:, None])
        if fp16:
            input_block = input_block.to(tl.float16)
            weight_block = weight_block.to(tl.float16)
        accum += tl.dot(input_block, weight_block, allow_tf32=tf32)
    if add_bias:
        bias = tl.load(bias_pointer + out_feat_offset, mask=out_feat_mask)
        if fp16:
            bias = bias.to(tl.float16)
        accum += bias[None, :]
    if act_func is not None:
        if save_pre_act:
            pre_act_pointer += pre_act_batch_stride * batch_offset[:, None
                ] + pre_act_out_feat_stride * out_feat_offset[None, :]
            tl.store(pre_act_pointer, accum, mask=batch_mask[:, None] &
                out_feat_mask[None, :])
        accum = apply_act_func(accum, None, None, None, param, act_func, False)
    output_pointer += output_batch_stride * batch_offset[:, None
        ] + output_out_feat_stride * out_feat_offset[None, :]
    tl.store(output_pointer, accum, mask=batch_mask[:, None] &
        out_feat_mask[None, :])


def get_output_dtype(input_dtype: torch.dtype=torch.float32, autocast:
    Optional[str]=None) ->torch.dtype:
    """
    Returns the appropriate output dtype for automatic mixed precision
    given the input dtype and the operation's autocast behaviour.

    Args:
        input_dtype: Input dtype.
        autocast: The relevent operation's autocast behaviour.
            None signifies the input dtype should flow through,
            'fp16' signifies autocasting to FP16 when AMP is enabled,
            and 'fp32' signifies autocasting to FP32 when AMP is enabled.
    """
    dtype = torch.get_autocast_dtype('cuda')
    assert dtype, f'Only autocast to float16 is supported, received {dtype}'
    if torch.is_autocast_enabled():
        if autocast is None:
            return input_dtype
        elif autocast == 'fp16':
            return torch.float16
        elif autocast == 'fp32':
            return torch.float32
        else:
            raise RuntimeError(
                f'Autocast type {autocast} is invalid. Options are None, fp16, and fp32'
                )
    else:
        return input_dtype


# Forward method (kernel launch code)
def _LinearAutoGrad_forward(ctx: Context, input: Tensor, weight: Tensor,
    bias: Optional[Tensor]=None, act_func: Optional[str]=None) ->Tensor:
    """
        Linearly transforms the input using weights, optionally adding bias
        and fusing an activation function.

        Args:
            input: Input to transform.
                Must be of shape [..., in_feat_dim].
            weight: Weights input is transformed by.
                Must be of shape [in_feat_dim, out_feat_dim].
            bias: Optional additive bias vector, with None for no bias.
                If provided, must be of shape [out_feat_dim].
            act_func: Name of activation function to apply, with None for identity.
                Options are 'sigmoid', 'logsigmoid', 'tanh', 'relu', 'gelu', 'geluapprox', 'silu',
                'relu6', 'hardsigmoid', 'hardtanh', 'hardswish', 'selu', 'mish',
                'softplus', 'softsign', 'tanhshrink', 'leaky_relu_PARAM',
                'elu_PARAM', 'celu_PARAM', 'hardshrink_PARAM', and 'softshrink_PARAM'
                where PARAM stands for the parameter in the case of parameterized
                activation functions (e.g., 'leaky_relu_0.01' for leaky ReLU with a
                negative slope of 0.01).

        Returns:
            Input linearly transformed, potentially with added biased and
            fused activation.
        """
    assert weight.ndim == 2, f'Weights must be 2D, received shape {weight.shape}'
    assert bias is None or bias.ndim == 1, f'Bias must be 1D, received shape {bias.shape}'
    assert input.shape[-1] == weight.shape[0
        ], f'Incompatible input ({input.shape}) and weights ({weight.shape}) shape'
    assert bias is None or weight.shape[1] == bias.shape[0
        ], f'Incompatible weights ({weight.shape}) and bias ({bias.shape}) shape'
    param = None
    if act_func is not None and '_' in act_func:
        comps = act_func.split('_')
        act_func = '_'.join(comps[:-1])
        param = float(comps[-1])
    flattened_input = input.flatten(0, -2)
    batch_dim, in_feat_dim = flattened_input.shape
    _, out_feat_dim = weight.shape
    requires_grad = (input.requires_grad or weight.requires_grad or bias is not
        None and bias.requires_grad)
    save_pre_act = requires_grad and act_func is not None
    output_dtype = get_output_dtype(input.dtype, autocast='fp16')
    output = torch.empty((batch_dim, out_feat_dim), device=input.device,
        dtype=output_dtype)
    pre_act = torch.empty_like(output) if save_pre_act else output
    grid = lambda META: (cdiv(batch_dim, META['BLOCK_SIZE_BATCH']) * cdiv(
        out_feat_dim, META['BLOCK_SIZE_OUT_FEAT']),)
    linear_forward_kernel[grid](flattened_input, weight, input if bias is
        None else bias, pre_act, output, batch_dim, in_feat_dim,
        out_feat_dim, *flattened_input.stride(), *weight.stride(), *pre_act
        .stride(), *output.stride(), param, add_bias=bias is not None,
        act_func=act_func, save_pre_act=save_pre_act, fp16=output_dtype is
        torch.float16)
    ctx.param = param
    ctx.act_func = act_func
    ctx.bias_requires_grad = False if bias is None else bias.requires_grad
    ctx.output_dtype = output_dtype
    if requires_grad:
        ctx.save_for_backward(input, pre_act if save_pre_act else None, weight)
    return output.view(*input.shape[:-1], out_feat_dim)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=element_wise_kernel_configs(), key=['size'])
@triton.jit
def act_func_backward_kernel(output_grad_pointer, input_pointer,
    input_grad_pointer, size, drop_p, seed, param, act_func: tl.constexpr,
    dropout: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Calculates the input gradient of an activation function.

    Args:
        output_grad_pointer: Pointer to the activation's output gradients.
            The output gradients must be of shape [size].
        input_pointer: Pointer to the activation's input.
            The input must be of shape [size].
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [size].
        size: Number of elements in the input.
        drop_p: Probability of dropping an element if dropout is True.
        seed: Seed for generating the dropout mask if dropout is True.
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'logsigmoid', 'tanh', 'relu', 'gelu', 'geluapprox', 'silu',
            'relu6', 'hardsigmoid', 'hardtanh', 'hardswish', 'selu', 'mish',
            'softplus', 'softsign', 'tanhshrink', 'leaky_relu', 'elu', 'celu', 'hardshrink',
            and 'softshrink'.
        dropout: Flag for performing dropout on the activation output.
        BLOCK_SIZE: Block size.
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    output_grad = tl.load(output_grad_pointer + offset, mask=mask)
    input = tl.load(input_pointer + offset, mask=mask)
    tl.store(input_grad_pointer + offset, apply_act_func_grad(output_grad,
        input, drop_p, seed, offset, param, act_func, dropout), mask=mask)


@triton.jit
def apply_act_func_grad(output_grad, input, drop_p, seed, offset, param,
    act_func: tl.constexpr, dropout: tl.constexpr):
    """
    Calculates the gradient of an activation function.

    Args:
        output_grad: Output gradients. The output gradients must be
            loaded and cannot be a pointer.
        input: Input. The input must be loaded and cannot be a pointer.
        drop_p: Probability of dropping an element if dropout is True.
        seed: Seed for generating the dropout mask if dropout is True.
        offset: Offset to generate the dropout mask for if dropout is True.
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'logsigmoid', 'tanh', 'relu', 'gelu', 'geluapprox', 'silu',
            'relu6', 'hardsigmoid', 'hardtanh', 'hardswish', 'selu', 'mish',
            'softplus', 'softsign', 'tanhshrink', 'leaky_relu', 'elu', 'celu', 'hardshrink',
            and 'softshrink'.
        dropout: Flag for performing dropout on the activation output.

    Returns:
        Gradient of the desired activation function.
    """
    if act_func == 'sigmoid':
        input = input.to(tl.float32)
        output = sigmoid_grad(input)
    if act_func == 'logsigmoid':
        input = input.to(tl.float32)
        output = logsigmoid_grad(input)
    elif act_func == 'tanh':
        input = input.to(tl.float32)
        output = tanh_grad(input)
    elif act_func == 'relu':
        output = relu_grad(input)
    elif act_func == 'gelu':
        input = input.to(tl.float32)
        output = gelu_grad(input)
    elif act_func == 'geluapprox':
        input = input.to(tl.float32)
        output = geluapprox_grad(input)
    elif act_func == 'silu':
        input = input.to(tl.float32)
        output = silu_grad(input)
    elif act_func == 'relu6':
        output = relu6_grad(input)
    elif act_func == 'hardsigmoid':
        output = hardsigmoid_grad(input)
    elif act_func == 'hardtanh':
        output = hardtanh_grad(input)
    elif act_func == 'hardswish':
        output = hardswish_grad(input)
    elif act_func == 'selu':
        input = input.to(tl.float32)
        output = selu_grad(input)
    elif act_func == 'mish':
        input = input.to(tl.float32)
        output = mish_grad(input)
    elif act_func == 'softplus':
        input = input.to(tl.float32)
        output = softplus_grad(input)
    elif act_func == 'softsign':
        output = softsign_grad(input)
    elif act_func == 'tanhshrink':
        input = input.to(tl.float32)
        output = tanhshrink_grad(input)
    elif act_func == 'leaky_relu':
        output = leaky_relu_grad(input, param)
    elif act_func == 'elu':
        input = input.to(tl.float32)
        output = elu_grad(input, param)
    elif act_func == 'celu':
        input = input.to(tl.float32)
        output = celu_grad(input, param)
    elif act_func == 'hardshrink':
        output = hardshrink_grad(input, param)
    elif act_func == 'softshrink':
        output = softshrink_grad(input, param)
    if dropout:
        output_grad = apply_dropout_grad(output_grad, drop_p, seed, offset)
    return output_grad * output


@triton.jit
def celu_grad(input, alpha):
    """
    Calculates the gradient of CELU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        alpha: Alpha value.

    Returns:
        Gradient of CELU.
    """
    return tl.where(input <= 0, tl.exp(input / alpha), 1)


@triton.jit
def elu_grad(input, alpha):
    """
    Calculates the gradient of ELU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        alpha: Alpha value.

    Returns:
        Gradient of ELU.
    """
    return tl.where(input <= 0, alpha * tl.exp(input), 1)


@triton.jit
def gelu_grad(input):
    """
    Calculates the gradient of GELU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of GELU.
    """
    cdf = 0.5 * (1 + tl.math.erf(0.707106781 * input))
    cdf_grad = 0.39894228 * tl.exp(-0.5 * input * input)
    return cdf_grad * input + cdf


@triton.jit
def geluapprox_grad(input):
    """
    Calculates the gradient of the tanh approximation of GELU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of the tanh approximation of GELU.
    """
    tanh_res = tanh(0.7978845608 * input * (1 + 0.044715 * input * input))
    sech_sq = 1 - tanh_res * tanh_res
    return 0.5 * (1 + tanh_res + input * sech_sq * 0.7978845608 * (1 + 3 * 
        0.044715 * input * input))


@triton.jit
def hardshrink_grad(input, lambd):
    """
    Calculates the gradient of hard shrink.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        lambd: Lambda value.

    Returns:
        Gradient of hard shrink.
    """
    return tl.where(tl.abs(input) < lambd, 0, 1)


@triton.jit
def hardsigmoid_grad(input):
    """
    Calculates the gradient of hard sigmoid.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of hard sigmoid.
    """
    return tl.where((-3 < input) & (input < 3), 1 / 6, 0)


@triton.jit
def hardswish_grad(input):
    """
    Calculates the gradient of hard Swish.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of hard Swish.
    """
    return (relu6(input + 3) + input * relu6_grad(input + 3)) / 6


@triton.jit
def hardtanh_grad(input):
    """
    Calculates the gradient of hard tanh.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of hard tanh.
    """
    return tl.where((-1 < input) & (input < 1), 1, 0)


@triton.jit
def leaky_relu_grad(input, negative_slope):
    """
    Calculates the gradient of leaky ReLU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        negative_slope: Slope of the negative component.

    Returns:
        Gradient of leaky ReLU.
    """
    return tl.where(input <= 0, negative_slope, 1)


@triton.jit
def logsigmoid_grad(input):
    """
    Calculates the gradient of the log of sigmoid.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of the log of sigmoid.
    """
    return 1 / (1 + tl.exp(input))


@triton.jit
def mish_grad(input):
    """
    Calculates the gradient of Mish.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of Mish.
    """
    exp = tl.exp(input)
    delta = exp * (exp + 2) + 2
    return exp * (exp * (4 * input + 6 + exp * (exp + 4)) + 4 * (input + 1)
        ) / (delta * delta)


@triton.jit
def relu6_grad(input):
    """
    Calculates the gradient of ReLU6.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of ReLU6.
    """
    return tl.where((0 < input) & (input < 6), 1, 0)


@triton.jit
def relu_grad(input):
    """
    Calculates the gradient of ReLU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of ReLU.
    """
    return tl.where(input <= 0, 0, 1)


@triton.jit
def selu_grad(input):
    """
    Calculates the gradient of SELU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of SELU.
    """
    scale = 1.0507009873554805
    alpha = 1.6732632423543772
    return scale * tl.where(input <= 0, alpha * tl.exp(input), 1)


@triton.jit
def sigmoid_grad(input):
    """
    Calculates the gradient of sigmoid.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of sigmoid.
    """
    output_sigmoid = sigmoid(input)
    return output_sigmoid * (1 - output_sigmoid)


@triton.jit
def silu_grad(input):
    """
    Calculates the gradient of SiLU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of SiLU.
    """
    output_sigmoid = sigmoid(input)
    return output_sigmoid * (input * (1 - output_sigmoid) + 1)


@triton.jit
def softplus_grad(input):
    """
    Calculates the gradient of softplus.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of softplus.
    """
    return sigmoid(input)


@triton.jit
def softshrink_grad(input, lambd):
    """
    Calculates the gradient of softshrink.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        lambd: Lambda value.

    Returns:
        Gradient of softshrink.
    """
    return tl.where(tl.abs(input) < lambd, 0, 1)


@triton.jit
def softsign_grad(input):
    """
    Calculates the gradient of softsign.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of softsign.
    """
    denom = 1 + tl.abs(input)
    return 1 / (denom * denom)


@triton.jit
def tanh_grad(input):
    """
    Calculates the gradient of tanh.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of tanh.
    """
    output_tanh = tanh(input)
    return 1 - output_tanh * output_tanh


@triton.jit
def tanhshrink_grad(input):
    """
    Calculates the gradient of tanh shrink.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of tanh shrink.
    """
    return 1 - tanh_grad(input)


@triton.jit
def apply_dropout_grad(output_grad, drop_p, seed, offset):
    """
    Calculates the input gradient of dropout.

    Args:
        output_grad: Output gradients. The output gradients must be
            loaded and cannot be a pointer.
        drop_p: Probability of dropping an element.
        seed: Seed for generating the dropout mask.
        offset: Offset to generate the mask for.

    Returns:
        Gradient of dropout.
    """
    random = tl.rand(seed, offset)
    return tl.where(random < drop_p, 0, output_grad / (1 - drop_p))


# Backward method (kernel launch code)
def _LinearAutoGrad_backward(ctx: Context, output_grad: Tensor) ->Tuple[
    Optional[Tensor], ...]:
    """
        Calculates the input gradient of the linear layer.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of the linear layer.
        """
    input, pre_act, weight = ctx.saved_tensors
    output_grad = output_grad.flatten(0, -2)
    flattened_input = input.flatten(0, -2)
    batch_dim, _ = flattened_input.shape
    _, out_feat_dim = weight.shape
    if ctx.act_func is None:
        pre_act_grad = output_grad
    else:
        size = batch_dim * out_feat_dim
        pre_act_grad = torch.empty(size, dtype=pre_act.dtype, device=
            pre_act.device)
        grid = lambda META: (cdiv(size, META['BLOCK_SIZE']),)
        act_func_backward_kernel[grid](output_grad, pre_act, pre_act_grad,
            size, None, None, ctx.param, ctx.act_func, False)
        pre_act_grad = pre_act_grad.view_as(pre_act)
    with torch.autocast('cuda', dtype=ctx.output_dtype):
        input_grad = pre_act_grad @ weight.T if input.requires_grad else None
        weight_grad = (flattened_input.T @ pre_act_grad if weight.
            requires_grad else None)
    bias_grad = pre_act_grad.sum(dim=0) if ctx.bias_requires_grad else None
    return input_grad.view_as(input
        ) if input_grad is not None else None, weight_grad, bias_grad, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LinearAutoGrad(torch.autograd.Function):
    """
    Autodiff for linear layer.
    """

    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor, bias: Optional
        [Tensor]=None, act_func: Optional[str]=None) ->Tensor:
        """
        Linearly transforms the input using weights, optionally adding bias
        and fusing an activation function.

        Args:
            input: Input to transform.
                Must be of shape [..., in_feat_dim].
            weight: Weights input is transformed by.
                Must be of shape [in_feat_dim, out_feat_dim].
            bias: Optional additive bias vector, with None for no bias.
                If provided, must be of shape [out_feat_dim].
            act_func: Name of activation function to apply, with None for identity.
                Options are 'sigmoid', 'logsigmoid', 'tanh', 'relu', 'gelu', 'geluapprox', 'silu',
                'relu6', 'hardsigmoid', 'hardtanh', 'hardswish', 'selu', 'mish',
                'softplus', 'softsign', 'tanhshrink', 'leaky_relu_PARAM',
                'elu_PARAM', 'celu_PARAM', 'hardshrink_PARAM', and 'softshrink_PARAM'
                where PARAM stands for the parameter in the case of parameterized
                activation functions (e.g., 'leaky_relu_0.01' for leaky ReLU with a
                negative slope of 0.01).

        Returns:
            Input linearly transformed, potentially with added biased and
            fused activation.
        """
        assert weight.ndim == 2, f'Weights must be 2D, received shape {weight.shape}'
        assert bias is None or bias.ndim == 1, f'Bias must be 1D, received shape {bias.shape}'
        assert input.shape[-1] == weight.shape[0
            ], f'Incompatible input ({input.shape}) and weights ({weight.shape}) shape'
        assert bias is None or weight.shape[1] == bias.shape[0
            ], f'Incompatible weights ({weight.shape}) and bias ({bias.shape}) shape'
        param = None
        if act_func is not None and '_' in act_func:
            comps = act_func.split('_')
            act_func = '_'.join(comps[:-1])
            param = float(comps[-1])
        flattened_input = input.flatten(0, -2)
        batch_dim, in_feat_dim = flattened_input.shape
        _, out_feat_dim = weight.shape
        requires_grad = (input.requires_grad or weight.requires_grad or 
            bias is not None and bias.requires_grad)
        save_pre_act = requires_grad and act_func is not None
        output_dtype = get_output_dtype(input.dtype, autocast='fp16')
        output = torch.empty((batch_dim, out_feat_dim), device=input.device,
            dtype=output_dtype)
        pre_act = torch.empty_like(output) if save_pre_act else output
        grid = lambda META: (cdiv(batch_dim, META['BLOCK_SIZE_BATCH']) *
            cdiv(out_feat_dim, META['BLOCK_SIZE_OUT_FEAT']),)
        linear_forward_kernel[grid](flattened_input, weight, input if bias is
            None else bias, pre_act, output, batch_dim, in_feat_dim,
            out_feat_dim, *flattened_input.stride(), *weight.stride(), *
            pre_act.stride(), *output.stride(), param, add_bias=bias is not
            None, act_func=act_func, save_pre_act=save_pre_act, fp16=
            output_dtype is torch.float16)
        ctx.param = param
        ctx.act_func = act_func
        ctx.bias_requires_grad = False if bias is None else bias.requires_grad
        ctx.output_dtype = output_dtype
        if requires_grad:
            ctx.save_for_backward(input, pre_act if save_pre_act else None,
                weight)
        return output.view(*input.shape[:-1], out_feat_dim)

    @staticmethod
    def backward(ctx: Context, output_grad: Tensor) ->Tuple[Optional[Tensor
        ], ...]:
        """
        Calculates the input gradient of the linear layer.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of the linear layer.
        """
        input, pre_act, weight = ctx.saved_tensors
        output_grad = output_grad.flatten(0, -2)
        flattened_input = input.flatten(0, -2)
        batch_dim, _ = flattened_input.shape
        _, out_feat_dim = weight.shape
        if ctx.act_func is None:
            pre_act_grad = output_grad
        else:
            size = batch_dim * out_feat_dim
            pre_act_grad = torch.empty(size, dtype=pre_act.dtype, device=
                pre_act.device)
            grid = lambda META: (cdiv(size, META['BLOCK_SIZE']),)
            act_func_backward_kernel[grid](output_grad, pre_act,
                pre_act_grad, size, None, None, ctx.param, ctx.act_func, False)
            pre_act_grad = pre_act_grad.view_as(pre_act)
        with torch.autocast('cuda', dtype=ctx.output_dtype):
            input_grad = (pre_act_grad @ weight.T if input.requires_grad else
                None)
            weight_grad = (flattened_input.T @ pre_act_grad if weight.
                requires_grad else None)
        bias_grad = pre_act_grad.sum(dim=0) if ctx.bias_requires_grad else None
        return input_grad.view_as(input
            ) if input_grad is not None else None, weight_grad, bias_grad, None
