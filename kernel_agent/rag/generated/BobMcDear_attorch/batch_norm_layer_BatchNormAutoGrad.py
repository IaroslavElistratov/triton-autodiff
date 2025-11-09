# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/BobMcDear/attorch
# Source-Files: attorch/batch_norm_layer.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_leort3lx/attorch-main/attorch/batch_norm_layer.py
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
from torch.cuda.amp import custom_bwd
from torch.cuda.amp import custom_fwd
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


def make_3d_for_bn(input: Tensor) ->Tensor:
    """
    Converts the input to a 3D view for batch normalization.

    Args:
        input: Input to render 3D.

    Returns:
        Input's 3D view.
    """
    if input.ndim == 2:
        input = input.unsqueeze(-1)
    elif input.ndim == 4:
        input = input.flatten(2, -1)
    return input


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


@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim',
    'spatial_dim'], restore_value=['running_mean_pointer',
    'running_var_pointer'])
@triton.heuristics({'BLOCK_SIZE_BATCH': lambda args: next_power_of_2(args[
    'batch_dim']), 'BLOCK_SIZE_SPATIAL': BLOCK_SIZE_SPATIAL_heuristic})
@triton.jit
def batch_norm_forward_kernel(input_pointer, weight_pointer, bias_pointer,
    mean_pointer, inv_std_pointer, pre_act_add_pointer, pre_act_pointer,
    output_pointer, running_mean_pointer, running_var_pointer, batch_dim,
    spatial_dim, input_batch_stride, input_feat_stride,
    input_spatial_stride, pre_act_add_batch_stride, pre_act_add_feat_stride,
    pre_act_add_spatial_stride, pre_act_batch_stride, pre_act_feat_stride,
    pre_act_spatial_stride, output_batch_stride, output_feat_stride,
    output_spatial_stride, momentum, eps, param, affine: tl.constexpr,
    save_stats: tl.constexpr, track_running_stats: tl.constexpr, is_train:
    tl.constexpr, add_pre_act: tl.constexpr, act_func: tl.constexpr,
    save_pre_act: tl.constexpr, BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_SPATIAL: tl.constexpr):
    """
    Batch-normalizes the input, optionally adding a residual and fusing an activation function.

    Args:
        input_pointer: Pointer to the input to layer-normalize.
            The input must be of shape [batch_dim, feat_dim, spatial_dim].
        weight_pointer: Pointer to optional weights for affine transform.
            The weights, if provided, must be of shape [feat_dim].
        bias_pointer: Pointer to an optional bias vector for affine transform.
            The bias vector, if provided, must be of shape [feat_dim].
        mean_pointer: Pointer to an optional container the input's mean
            is written to if save_stats is True.
            The container, if provided, must be of shape [feat_dim].
        inv_std_pointer: Pointer to an optional container the input's inverse
            standard deviation is written to if save_stats is True.
            The container, if provided, must be of shape [feat_dim].
        pre_act_add_pointer: Pointer to an optional residual added to the pre-activation result.
            The residual, if provided, must be of shape [batch_dim, feat_dim, spatial_dim].
        pre_act_pointer: Pointer to an optional container the pre-activation input
            is written to if act_func is not None and save_pre_act is True.
            The container, if provided, must be of shape [batch_dim, feat_dim, spatial_dim].
        output_pointer: Pointer to a container the result is written to.
            The container must be of shape [batch_dim, feat_dim, spatial_dim].
        running_mean_pointer: Pointer to an optional container the input's running
            mean is written to if track_running_stats and is_train are True.
            The container, if provided, must be of shape [feat_dim].
        running_var_pointer: Pointer to an optional container the input's running
            variance is written to if track_running_stats and is_train are True.
            The container, if provided, must be of shape [feat_dim].
        batch_dim: Batch dimension.
        spatial_dim: Spatial dimension.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        input_spatial_stride: Stride necessary to jump one element along the
            input's spatial dimension.
        pre_act_add_batch_stride: Stride necessary to jump one element along the
            residual's batch dimension.
        pre_act_add_out_feat_stride: Stride necessary to jump one element along the
            residual's feature dimension.
        pre_act_add_spatial_stride: Stride necessary to jump one element along the
            residual's spatial dimension.
        pre_act_batch_stride: Stride necessary to jump one element along the
            pre-activation input container's batch dimension.
        pre_act_out_feat_stride: Stride necessary to jump one element along the
            pre-activation input container's feature dimension.
        pre_act_spatial_stride: Stride necessary to jump one element along the
            pre-activation input container's spatial dimension.
        output_batch_stride: Stride necessary to jump one element along the
            output container's batch dimension.
        output_feat_stride: Stride necessary to jump one element along the
            output container's feature dimension.
        output_spatial_stride: Stride necessary to jump one element along the
            output container's spatial dimension.
        momentum: Momentum for the running mean and variance.
        eps: Epsilon added in the square root in the denominator
            to avoid division by zero.
        param: Parameter in the case of parameterized activation functions.
        affine: Flag for performing an affine transformation on the normalized output.
        save_stats: Flag for saving the mean and standard deviation.
        track_running_stats: Flag for tracking running mean and variance if
            is_train is also True.
        is_train: Flag indicating if the model is in training mode.
        add_pre_act: Flag for adding the residual to the pre-activation result.
        act_func: Name of activation function to apply, with None for identity.
            Options are 'sigmoid', 'logsigmoid', 'tanh', 'relu', 'gelu', 'geluapprox', 'silu',
            'softplus', 'softsign', 'tanhshrink', 'leaky_relu', 'elu', 'celu', 'hardshrink',
            and 'softshrink'.
        save_pre_act: Flag for saving the pre-activation input.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_SPATIAL: Block size across the spatial dimension.
    """
    feat_pid = tl.program_id(axis=0)
    batch_offset = tl.arange(0, BLOCK_SIZE_BATCH)
    batch_mask = batch_offset < batch_dim
    if is_train or not track_running_stats:
        count = 0
        mean = 0.0
        var = 0.0
        for block_ind in range(0, tl.cdiv(spatial_dim, BLOCK_SIZE_SPATIAL)):
            spatial_offset = block_ind * BLOCK_SIZE_SPATIAL + tl.arange(0,
                BLOCK_SIZE_SPATIAL)
            spatial_mask = spatial_offset < spatial_dim
            curr_input_pointer = (input_pointer + input_feat_stride *
                feat_pid + input_batch_stride * batch_offset[:, None] + 
                input_spatial_stride * spatial_offset[None, :])
            curr_input = tl.load(curr_input_pointer, mask=batch_mask[:,
                None] & spatial_mask[None, :]).to(tl.float32)
            spatial_count = min(BLOCK_SIZE_SPATIAL, spatial_dim - block_ind *
                BLOCK_SIZE_SPATIAL)
            curr_count = spatial_count * batch_dim
            count += curr_count
            prev_mean = mean
            mean += (tl.sum(curr_input) - curr_count * mean) / count
            deltas = tl.where(batch_mask[:, None] & spatial_mask[None, :], 
                (curr_input - mean) * (curr_input - prev_mean), 0.0)
            var += tl.sum(deltas)
        var /= count
        inv_std = tl.rsqrt(var + eps)
        if save_stats:
            tl.store(feat_pid + mean_pointer, mean)
            tl.store(feat_pid + inv_std_pointer, inv_std)
        if track_running_stats:
            running_mean_pointer += feat_pid
            running_var_pointer += feat_pid
            running_mean = tl.load(running_mean_pointer)
            running_var = tl.load(running_var_pointer)
            n = batch_dim * spatial_dim
            tl.store(running_mean_pointer, (1 - momentum) * running_mean + 
                momentum * mean)
            tl.store(running_var_pointer, (1 - momentum) * running_var + 
                momentum * var * n / (n - 1))
    else:
        mean = tl.load(feat_pid + running_mean_pointer)
        inv_std = tl.rsqrt(tl.load(feat_pid + running_var_pointer) + eps)
    if affine:
        weight = tl.load(feat_pid + weight_pointer)
        bias = tl.load(feat_pid + bias_pointer)
    else:
        weight = 1.0
        bias = 0.0
    for block_ind in range(0, tl.cdiv(spatial_dim, BLOCK_SIZE_SPATIAL)):
        spatial_offset = block_ind * BLOCK_SIZE_SPATIAL + tl.arange(0,
            BLOCK_SIZE_SPATIAL)
        spatial_mask = spatial_offset < spatial_dim
        curr_input_pointer = (input_pointer + input_feat_stride * feat_pid +
            input_batch_stride * batch_offset[:, None] + 
            input_spatial_stride * spatial_offset[None, :])
        curr_output_pointer = (output_pointer + output_feat_stride *
            feat_pid + output_batch_stride * batch_offset[:, None] + 
            output_spatial_stride * spatial_offset[None, :])
        curr_input = tl.load(curr_input_pointer, mask=batch_mask[:, None] &
            spatial_mask[None, :]).to(tl.float32)
        output = weight * (curr_input - mean) * inv_std + bias
        if add_pre_act:
            curr_pre_act_add_pointer = (pre_act_add_pointer + 
                pre_act_add_feat_stride * feat_pid + 
                pre_act_add_batch_stride * batch_offset[:, None] + 
                pre_act_add_spatial_stride * spatial_offset[None, :])
            curr_pre_act_add = tl.load(curr_pre_act_add_pointer, mask=
                batch_mask[:, None] & spatial_mask[None, :])
            output += curr_pre_act_add
        if act_func is not None:
            if save_pre_act:
                curr_pre_act_pointer = (pre_act_pointer + 
                    pre_act_feat_stride * feat_pid + pre_act_batch_stride *
                    batch_offset[:, None] + pre_act_spatial_stride *
                    spatial_offset[None, :])
                tl.store(curr_pre_act_pointer, output, mask=batch_mask[:,
                    None] & spatial_mask[None, :])
            output = apply_act_func(output, None, None, None, param,
                act_func, False)
        tl.store(curr_output_pointer, output, mask=batch_mask[:, None] &
            spatial_mask[None, :])


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


# Forward method (kernel launch code)
@custom_fwd(device_type='cuda')
def _BatchNormAutoGrad_forward(ctx: Context, input: Tensor, training: bool,
    weight: Optional[Tensor]=None, bias: Optional[Tensor]=None,
    running_mean: Optional[Tensor]=None, running_var: Optional[Tensor]=None,
    momentum: float=0.1, eps: float=1e-05, track_running_stats: bool=True,
    pre_act_add: Optional[Tensor]=None, act_func: Optional[str]=None) ->Tensor:
    """
        Batch-normalizes the input, optionally adding a residual and
        fusing an activation function.

        Args:
            ctx: Context for variable storage.
            input: Input to layer-normalize.
                Must be of shape [batch_dim, feat_dim] or [batch_dim, feat_dim, spatial_dim].
            training: Flag indicating if the model is in training mode.
            weight: Optional weights for affine transform when bias is provided.
                If provided, must be of shape [feat_dim].
            bias: Optional bias vector for affine transform when weight is provided.
                If provided, must be of shape [feat_dim].
            running_mean: Optional container for storing the input's running mean
                if training and track_running_stats are True.
            running_var: Optional container for storing the input's running variance
                if training and track_running_stats are True.
            momentum: Momentum for the running mean and variance.
            eps: Epsilon added in the square root in the denominator
                to avoid division by zero.
            track_running_stats: Flag for tracking running mean and variance if
                is_train is also True.
            pre_act_add: Optional residual added to the pre-activation result.
            act_func: Name of activation function to apply, with None for identity.
                Options are 'sigmoid', 'logsigmoid', 'tanh', 'relu', 'gelu', 'geluapprox', 'silu',
                'relu6', 'hardsigmoid', 'hardtanh', 'hardswish', 'selu', 'mish',
                'softplus', 'softsign', 'tanhshrink', 'leaky_relu_PARAM',
                'elu_PARAM', 'celu_PARAM', 'hardshrink_PARAM', and 'softshrink_PARAM'
                where PARAM stands for the parameter in the case of parameterized
                activation functions (e.g., 'leaky_relu_0.01' for leaky ReLU with a
                negative slope of 0.01).

        Returns:
            Batch-normalized input, potentially with fused activation and added residual.
        """
    param = None
    if act_func is not None and '_' in act_func:
        comps = act_func.split('_')
        act_func = '_'.join(comps[:-1])
        param = float(comps[-1])
    ctx.param = param
    ctx.act_func = act_func
    add_pre_act = pre_act_add is not None
    pre_act_add = pre_act_add if add_pre_act else torch.empty((1, 1, 1),
        device='cuda')
    input_3d = make_3d_for_bn(input)
    pre_act_add = make_3d_for_bn(pre_act_add)
    transpose = False
    if input_3d.shape[-1] > 1:
        input_3d = input_3d.transpose(0, -1)
        pre_act_add = pre_act_add.transpose(0, -1)
        transpose = True
    affine = weight is not None and bias is not None
    requires_grad = (input.requires_grad or pre_act_add.requires_grad or 
        affine and weight.requires_grad or affine and bias.requires_grad)
    save_pre_act = requires_grad and act_func is not None
    batch_dim, feat_dim, spatial_dim = input_3d.shape
    output = torch.empty_like(input_3d)
    pre_act = torch.empty_like(input_3d) if save_pre_act else output
    if requires_grad:
        mean = torch.empty(feat_dim, device=input.device, dtype=torch.float32)
        inv_std = torch.empty(feat_dim, device=input.device, dtype=torch.
            float32)
    else:
        mean = inv_std = None
    running_mean = input if running_mean is None else running_mean
    running_var = input if running_var is None else running_var
    grid = lambda _: (feat_dim,)
    batch_norm_forward_kernel[grid](input_3d, weight, bias, mean, inv_std,
        pre_act_add, pre_act, output, running_mean, running_var, batch_dim,
        spatial_dim, *input_3d.stride(), *pre_act_add.stride(), *pre_act.
        stride(), *output.stride(), momentum, eps, param, affine=affine,
        save_stats=requires_grad, track_running_stats=track_running_stats,
        is_train=training, add_pre_act=add_pre_act, act_func=act_func,
        save_pre_act=save_pre_act)
    if transpose:
        output = output.transpose(0, -1)
        if save_pre_act:
            pre_act = pre_act.transpose(0, -1)
    ctx.affine = affine
    ctx.act_func = act_func
    ctx.add_pre_act = add_pre_act
    if requires_grad:
        ctx.save_for_backward(input, mean, inv_std, weight, pre_act if
            save_pre_act else None)
    return output.view_as(input)


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


@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim',
    'spatial_dim'])
@triton.heuristics({'BLOCK_SIZE_BATCH': lambda args: next_power_of_2(args[
    'batch_dim']), 'BLOCK_SIZE_SPATIAL': BLOCK_SIZE_SPATIAL_heuristic})
@triton.jit
def batch_norm_backward_kernel(output_grad_pointer, input_pointer,
    mean_pointer, inv_std_pointer, weight_pointer, input_grad_pointer,
    weight_grad_pointer, bias_grad_pointer, batch_dim, spatial_dim,
    output_grad_batch_stride, output_grad_feat_stride,
    output_grad_spatial_stride, input_batch_stride, input_feat_stride,
    input_spatial_stride, input_grad_batch_stride, input_grad_feat_stride,
    input_grad_spatial_stride, affine: tl.constexpr, BLOCK_SIZE_BATCH: tl.
    constexpr, BLOCK_SIZE_SPATIAL: tl.constexpr):
    """
    Calculates the input gradient of batch normalization.

    Args:
        output_grad_pointer: Pointer to layer normalization's output gradients.
            The output gradients must be of shape [batch_dim, feat_dim, spatial_dim].
        input_pointer: Pointer to the input.
            The input must be of shape [batch_dim, feat_dim, spatial_dim].
        mean_pointer: Pointer to the input's mean.
            The mean should be of shape [feat_dim].
        inv_std_pointer: Pointer to the input's inverse standard deviation.
            The inverse standard deviation should be of shape [feat_dim].
        weight_pointer: Pointer to optional weights if affine transform occurred.
            The weights, if provided, must be of shape [feat_dim].
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [batch_dim, feat_dim, spatial_dim].
        weight_grad_pointer: Pointer to an optional container the weights' gradients
            are written to if scale_by_weight is True.
            The container, if provided, must be of shape [feat_dim].
        bias_grad_pointer: Pointer to an optional container the bias vector's gradients
            are written to if scale_by_weight is True.
            The container, if provided, must be of shape [feat_dim].
        batch_dim: Batch dimension.
        spatial_dim: Spatial dimension.
        output_grad_batch_stride: Stride necessary to jump one element along the
            output gradients' batch dimension.
        output_grad_feat_stride: Stride necessary to jump one element along the
            output gradients' feature dimension.
        output_grad_spatial_stride: Stride necessary to jump one element along the
            output gradients' spatial dimension.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        input_spatial_stride: Stride necessary to jump one element along the
            input's spatial dimension.
        input_grad_batch_stride: Stride necessary to jump one element along the
            input gradient container's batch dimension.
        input_grad_feat_stride: Stride necessary to jump one element along the
            input gradient container's feature dimension.
        input_grad_spatial_stride: Stride necessary to jump one element along the
            input gradient container's spatial dimension.
        affine: Flag for performing an affine transformation on the normalized output.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_SPATIAL: Block size across the spatial dimension.
    """
    feat_pid = tl.program_id(axis=0)
    batch_offset = tl.arange(0, BLOCK_SIZE_BATCH)
    batch_mask = batch_offset < batch_dim
    mean = tl.load(feat_pid + mean_pointer)
    inv_std = tl.load(feat_pid + inv_std_pointer)
    term1 = 0.0
    term2 = 0.0
    for block_ind in range(0, tl.cdiv(spatial_dim, BLOCK_SIZE_SPATIAL)):
        spatial_offset = block_ind * BLOCK_SIZE_SPATIAL + tl.arange(0,
            BLOCK_SIZE_SPATIAL)
        spatial_mask = spatial_offset < spatial_dim
        curr_output_grad_pointer = (output_grad_pointer + 
            output_grad_feat_stride * feat_pid + output_grad_batch_stride *
            batch_offset[:, None] + output_grad_spatial_stride *
            spatial_offset[None, :])
        curr_input_pointer = (input_pointer + input_feat_stride * feat_pid +
            input_batch_stride * batch_offset[:, None] + 
            input_spatial_stride * spatial_offset[None, :])
        curr_input = tl.load(curr_input_pointer, mask=batch_mask[:, None] &
            spatial_mask[None, :]).to(tl.float32)
        curr_pre_lin = (curr_input - mean) * inv_std
        curr_output_grad = tl.load(curr_output_grad_pointer, mask=
            batch_mask[:, None] & spatial_mask[None, :]).to(tl.float32)
        term1 += tl.sum(curr_pre_lin * curr_output_grad)
        term2 += tl.sum(curr_output_grad)
    if affine:
        weight = tl.load(feat_pid + weight_pointer)
        weight_grad = 0.0
        bias_grad = 0.0
    else:
        weight = 1.0
    count = batch_dim * spatial_dim
    term1 *= weight / count
    term2 *= weight / count
    for block_ind in range(0, tl.cdiv(spatial_dim, BLOCK_SIZE_SPATIAL)):
        spatial_offset = block_ind * BLOCK_SIZE_SPATIAL + tl.arange(0,
            BLOCK_SIZE_SPATIAL)
        spatial_mask = spatial_offset < spatial_dim
        curr_output_grad_pointer = (output_grad_pointer + 
            output_grad_feat_stride * feat_pid + output_grad_batch_stride *
            batch_offset[:, None] + output_grad_spatial_stride *
            spatial_offset[None, :])
        curr_input_pointer = (input_pointer + input_feat_stride * feat_pid +
            input_batch_stride * batch_offset[:, None] + 
            input_spatial_stride * spatial_offset[None, :])
        curr_input_grad_pointer = (input_grad_pointer + 
            input_grad_feat_stride * feat_pid + input_grad_batch_stride *
            batch_offset[:, None] + input_grad_spatial_stride *
            spatial_offset[None, :])
        curr_input = tl.load(curr_input_pointer, mask=batch_mask[:, None] &
            spatial_mask[None, :]).to(tl.float32)
        curr_pre_lin = (curr_input - mean) * inv_std
        curr_output_grad = tl.load(curr_output_grad_pointer, mask=
            batch_mask[:, None] & spatial_mask[None, :]).to(tl.float32)
        curr_input_grad = inv_std * (weight * curr_output_grad - (term1 *
            curr_pre_lin + term2))
        tl.store(curr_input_grad_pointer, curr_input_grad, mask=batch_mask[
            :, None] & spatial_mask[None, :])
        if affine:
            weight_grad += tl.sum(curr_pre_lin * curr_output_grad)
            bias_grad += tl.sum(curr_output_grad)
    if affine:
        tl.store(feat_pid + weight_grad_pointer, weight_grad)
        tl.store(feat_pid + bias_grad_pointer, bias_grad)


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
@custom_bwd(device_type='cuda')
def _BatchNormAutoGrad_backward(ctx: Context, output_grad: Tensor) ->Tuple[
    Optional[Tensor], ...]:
    """
        Calculates the input gradient of batch normalization.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of batch normalization.
        """
    input, mean, inv_std, weight, pre_act = ctx.saved_tensors
    input_3d = make_3d_for_bn(input)
    if ctx.act_func is None:
        pre_act_grad = make_3d_for_bn(output_grad)
    else:
        size = output_grad.numel()
        pre_act_grad = torch.empty(size, dtype=pre_act.dtype, device=
            pre_act.device)
        grid = lambda META: (cdiv(size, META['BLOCK_SIZE']),)
        act_func_backward_kernel[grid](output_grad.flatten(), pre_act,
            pre_act_grad, size, None, None, ctx.param, ctx.act_func, False)
        pre_act_grad = pre_act_grad.view_as(pre_act)
    transpose = False
    if input_3d.shape[-1] > 1:
        input_3d = input_3d.transpose(0, -1)
        pre_act_grad = pre_act_grad.transpose(0, -1)
        transpose = True
    batch_dim, feat_dim, spatial_dim = input_3d.shape
    input_grad = torch.empty_like(input_3d)
    if ctx.affine:
        weight_grad = torch.empty((feat_dim,), device=input.device)
        bias_grad = torch.empty_like(weight_grad)
    else:
        weight_grad = bias_grad = None
    grid = lambda _: (feat_dim,)
    batch_norm_backward_kernel[grid](pre_act_grad, input_3d, mean, inv_std,
        weight, input_grad, weight_grad, bias_grad, batch_dim, spatial_dim,
        *pre_act_grad.stride(), *input_3d.stride(), *input_grad.stride(),
        affine=ctx.affine)
    if transpose:
        input_grad = input_grad.transpose(0, -1)
        pre_act_grad = pre_act_grad.transpose(0, -1)
    return (input_grad.view_as(input), None, weight_grad, bias_grad, None,
        None, None, None, None, pre_act_grad.view_as(input) if ctx.
        add_pre_act else None, None)


# ============================================================
# autograd.Function Class Definition
# ============================================================

class BatchNormAutoGrad(torch.autograd.Function):
    """
    Autodiff for batch normalization.
    """

    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx: Context, input: Tensor, training: bool, weight:
        Optional[Tensor]=None, bias: Optional[Tensor]=None, running_mean:
        Optional[Tensor]=None, running_var: Optional[Tensor]=None, momentum:
        float=0.1, eps: float=1e-05, track_running_stats: bool=True,
        pre_act_add: Optional[Tensor]=None, act_func: Optional[str]=None
        ) ->Tensor:
        """
        Batch-normalizes the input, optionally adding a residual and
        fusing an activation function.

        Args:
            ctx: Context for variable storage.
            input: Input to layer-normalize.
                Must be of shape [batch_dim, feat_dim] or [batch_dim, feat_dim, spatial_dim].
            training: Flag indicating if the model is in training mode.
            weight: Optional weights for affine transform when bias is provided.
                If provided, must be of shape [feat_dim].
            bias: Optional bias vector for affine transform when weight is provided.
                If provided, must be of shape [feat_dim].
            running_mean: Optional container for storing the input's running mean
                if training and track_running_stats are True.
            running_var: Optional container for storing the input's running variance
                if training and track_running_stats are True.
            momentum: Momentum for the running mean and variance.
            eps: Epsilon added in the square root in the denominator
                to avoid division by zero.
            track_running_stats: Flag for tracking running mean and variance if
                is_train is also True.
            pre_act_add: Optional residual added to the pre-activation result.
            act_func: Name of activation function to apply, with None for identity.
                Options are 'sigmoid', 'logsigmoid', 'tanh', 'relu', 'gelu', 'geluapprox', 'silu',
                'relu6', 'hardsigmoid', 'hardtanh', 'hardswish', 'selu', 'mish',
                'softplus', 'softsign', 'tanhshrink', 'leaky_relu_PARAM',
                'elu_PARAM', 'celu_PARAM', 'hardshrink_PARAM', and 'softshrink_PARAM'
                where PARAM stands for the parameter in the case of parameterized
                activation functions (e.g., 'leaky_relu_0.01' for leaky ReLU with a
                negative slope of 0.01).

        Returns:
            Batch-normalized input, potentially with fused activation and added residual.
        """
        param = None
        if act_func is not None and '_' in act_func:
            comps = act_func.split('_')
            act_func = '_'.join(comps[:-1])
            param = float(comps[-1])
        ctx.param = param
        ctx.act_func = act_func
        add_pre_act = pre_act_add is not None
        pre_act_add = pre_act_add if add_pre_act else torch.empty((1, 1, 1),
            device='cuda')
        input_3d = make_3d_for_bn(input)
        pre_act_add = make_3d_for_bn(pre_act_add)
        transpose = False
        if input_3d.shape[-1] > 1:
            input_3d = input_3d.transpose(0, -1)
            pre_act_add = pre_act_add.transpose(0, -1)
            transpose = True
        affine = weight is not None and bias is not None
        requires_grad = (input.requires_grad or pre_act_add.requires_grad or
            affine and weight.requires_grad or affine and bias.requires_grad)
        save_pre_act = requires_grad and act_func is not None
        batch_dim, feat_dim, spatial_dim = input_3d.shape
        output = torch.empty_like(input_3d)
        pre_act = torch.empty_like(input_3d) if save_pre_act else output
        if requires_grad:
            mean = torch.empty(feat_dim, device=input.device, dtype=torch.
                float32)
            inv_std = torch.empty(feat_dim, device=input.device, dtype=
                torch.float32)
        else:
            mean = inv_std = None
        running_mean = input if running_mean is None else running_mean
        running_var = input if running_var is None else running_var
        grid = lambda _: (feat_dim,)
        batch_norm_forward_kernel[grid](input_3d, weight, bias, mean,
            inv_std, pre_act_add, pre_act, output, running_mean,
            running_var, batch_dim, spatial_dim, *input_3d.stride(), *
            pre_act_add.stride(), *pre_act.stride(), *output.stride(),
            momentum, eps, param, affine=affine, save_stats=requires_grad,
            track_running_stats=track_running_stats, is_train=training,
            add_pre_act=add_pre_act, act_func=act_func, save_pre_act=
            save_pre_act)
        if transpose:
            output = output.transpose(0, -1)
            if save_pre_act:
                pre_act = pre_act.transpose(0, -1)
        ctx.affine = affine
        ctx.act_func = act_func
        ctx.add_pre_act = add_pre_act
        if requires_grad:
            ctx.save_for_backward(input, mean, inv_std, weight, pre_act if
                save_pre_act else None)
        return output.view_as(input)

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx: Context, output_grad: Tensor) ->Tuple[Optional[Tensor
        ], ...]:
        """
        Calculates the input gradient of batch normalization.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of batch normalization.
        """
        input, mean, inv_std, weight, pre_act = ctx.saved_tensors
        input_3d = make_3d_for_bn(input)
        if ctx.act_func is None:
            pre_act_grad = make_3d_for_bn(output_grad)
        else:
            size = output_grad.numel()
            pre_act_grad = torch.empty(size, dtype=pre_act.dtype, device=
                pre_act.device)
            grid = lambda META: (cdiv(size, META['BLOCK_SIZE']),)
            act_func_backward_kernel[grid](output_grad.flatten(), pre_act,
                pre_act_grad, size, None, None, ctx.param, ctx.act_func, False)
            pre_act_grad = pre_act_grad.view_as(pre_act)
        transpose = False
        if input_3d.shape[-1] > 1:
            input_3d = input_3d.transpose(0, -1)
            pre_act_grad = pre_act_grad.transpose(0, -1)
            transpose = True
        batch_dim, feat_dim, spatial_dim = input_3d.shape
        input_grad = torch.empty_like(input_3d)
        if ctx.affine:
            weight_grad = torch.empty((feat_dim,), device=input.device)
            bias_grad = torch.empty_like(weight_grad)
        else:
            weight_grad = bias_grad = None
        grid = lambda _: (feat_dim,)
        batch_norm_backward_kernel[grid](pre_act_grad, input_3d, mean,
            inv_std, weight, input_grad, weight_grad, bias_grad, batch_dim,
            spatial_dim, *pre_act_grad.stride(), *input_3d.stride(), *
            input_grad.stride(), affine=ctx.affine)
        if transpose:
            input_grad = input_grad.transpose(0, -1)
            pre_act_grad = pre_act_grad.transpose(0, -1)
        return (input_grad.view_as(input), None, weight_grad, bias_grad,
            None, None, None, None, None, pre_act_grad.view_as(input) if
            ctx.add_pre_act else None, None)
