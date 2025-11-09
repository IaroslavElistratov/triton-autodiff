# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/BobMcDear/attorch
# Source-Files: attorch/layer_norm_layer.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_leort3lx/attorch-main/attorch/layer_norm_layer.py
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

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim', 'feat_dim'])
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic,
    'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim'])})
@triton.jit
def layer_norm_forward_kernel(input_pointer, weight_pointer, bias_pointer,
    mean_pointer, inv_std_pointer, output_pointer, batch_dim, feat_dim,
    input_batch_stride, input_feat_stride, output_batch_stride,
    output_feat_stride, eps, scale_by_weight: tl.constexpr, add_bias: tl.
    constexpr, save_stats: tl.constexpr, BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_FEAT: tl.constexpr):
    """
    Layer-normalizes the input.

    Args:
        input_pointer: Pointer to the input to layer-normalize.
            The input must be of shape [batch_dim, feat_dim].
        weight_pointer: Pointer to optional weights for affine transform.
            The weights, if provided, must be of shape [feat_dim].
        bias_pointer: Pointer to an optional bias vector for affine transform.
            The bias vector, if provided, must be of shape [feat_dim].
        mean_pointer: Pointer to an optional container the input's mean
            is written to if save_stats is True.
            The container, if provided, must be of shape [batch_dim].
        inv_std_pointer: Pointer to an optional container the input's inverse
            standard deviation is written to if save_stats is True.
            The container, if provided, must be of shape [batch_dim].
        output_pointer: Pointer to a container the result is written to.
            The container must be of shape [batch_dim, feat_dim].
        batch_dim: Batch dimension.
        feat_dim: Dimensionality of the features.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        output_batch_stride: Stride necessary to jump one element along the
            output container's batch dimension.
        output_feat_stride: Stride necessary to jump one element along the
            output container's feature dimension.
        eps: Epsilon added in the square root in the denominator
            to avoid division by zero.
        scale_by_weight: Flag for scaling the normalized output by weights.
        add_bias: Flag for adding a bias vector to the normalized output
            if scale_by_weight is True.
        save_stats: Flag for saving the mean and standard deviation.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_FEAT: Block size across the feature dimension.
    """
    batch_pid = tl.program_id(axis=0)
    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH
        )
    feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)
    batch_mask = batch_offset < batch_dim
    feat_mask = feat_offset < feat_dim
    input_pointer += input_batch_stride * batch_offset[:, None
        ] + input_feat_stride * feat_offset[None, :]
    output_pointer += output_batch_stride * batch_offset[:, None
        ] + output_feat_stride * feat_offset[None, :]
    input = tl.load(input_pointer, mask=batch_mask[:, None] & feat_mask[
        None, :]).to(tl.float32)
    mean = tl.sum(input, axis=1) / feat_dim
    diff = tl.where(feat_mask[None, :], input - mean[:, None], 0)
    inv_std = tl.rsqrt(tl.sum(diff * diff, axis=1) / feat_dim + eps)
    if save_stats:
        tl.store(mean_pointer + batch_offset, mean, mask=batch_mask)
        tl.store(inv_std_pointer + batch_offset, inv_std, mask=batch_mask)
    output = diff * inv_std[:, None]
    if scale_by_weight:
        weight = tl.load(weight_pointer + feat_offset, mask=feat_mask)
        output *= weight
        if add_bias:
            bias = tl.load(bias_pointer + feat_offset, mask=feat_mask)
            output += bias
    tl.store(output_pointer, output, mask=batch_mask[:, None] & feat_mask[
        None, :])


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
def _LayerNormAutoGrad_forward(ctx: Context, input: Tensor, weight:
    Optional[Tensor]=None, bias: Optional[Tensor]=None, eps: float=1e-05,
    autocast_to_fp32: bool=True) ->Tensor:
    """
        Layer-normalizes the input.

        Args:
            ctx: Context for variable storage.
            input: Input to layer-normalize.
                Can have arbitrary shape.
            weight: Optional weights for affine transform.
                If provided, must be of shape [feat_dim].
            bias: Optional bias vector for affine transform when weight is provided.
                If provided, must be of shape [feat_dim].
            eps: Epsilon added in the square root in the denominator
                to avoid division by zero.
            autocast_to_fp32: Flag for autocasting the output dtype to fp32.
                If False, the input dtype flows through.

        Returns:
            Layer-normalized input.
        """
    flattened_input = input.unsqueeze(0) if input.ndim == 1 else input
    flattened_input = flattened_input.flatten(0, -2)
    batch_dim, feat_dim = flattened_input.shape
    output_dtype = get_output_dtype(input.dtype, autocast='fp32' if
        autocast_to_fp32 else None)
    output = torch.empty_like(flattened_input, dtype=output_dtype)
    scale_by_weight = weight is not None
    add_bias = scale_by_weight and bias is not None
    requires_grad = (input.requires_grad or scale_by_weight and weight.
        requires_grad or add_bias and bias.requires_grad)
    if requires_grad:
        mean = torch.empty(batch_dim, device=input.device, dtype=torch.float32)
        inv_std = torch.empty(batch_dim, device=input.device, dtype=torch.
            float32)
    else:
        mean = inv_std = None
    grid = lambda META: (cdiv(batch_dim, META['BLOCK_SIZE_BATCH']),)
    layer_norm_forward_kernel[grid](flattened_input, weight, bias, mean,
        inv_std, output, batch_dim, feat_dim, *flattened_input.stride(), *
        output.stride(), eps, scale_by_weight=scale_by_weight, add_bias=
        add_bias, save_stats=requires_grad)
    ctx.scale_by_weight = scale_by_weight
    ctx.add_bias = add_bias
    ctx.output_dtype = output_dtype
    if requires_grad:
        ctx.save_for_backward(flattened_input, mean, inv_std, weight)
    return output.view_as(input)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim', 'feat_dim'])
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic,
    'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim'])})
@triton.jit
def layer_norm_backward_kernel(output_grad_pointer, input_pointer,
    mean_pointer, inv_std_pointer, weight_pointer, input_grad_pointer,
    weight_grad_pointer, bias_grad_pointer, batch_dim, feat_dim,
    output_grad_batch_stride, output_grad_feat_stride, input_batch_stride,
    input_feat_stride, input_grad_batch_stride, input_grad_feat_stride,
    weight_grad_batch_stride, weight_grad_feat_stride,
    bias_grad_batch_stride, bias_grad_feat_stride, scale_by_weight: tl.
    constexpr, add_bias: tl.constexpr, BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_FEAT: tl.constexpr):
    """
    Calculates the input gradient of layer normalization.

    Args:
        output_grad_pointer: Pointer to layer normalization's output gradients.
            The output gradients must be of shape [batch_dim, feat_dim].
        input_pointer: Pointer to the input.
            The input must be of shape [batch_dim, feat_dim].
        mean_pointer: Pointer to the input's mean.
            The mean should be of shape [batch_dim].
        inv_std_pointer: Pointer to the input's inverse standard deviation.
            The inverse standard deviation should be of shape [batch_dim].
        weight_pointer: Pointer to optional weights if affine transform occurred.
            The weights, if provided, must be of shape [feat_dim].
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [batch_dim, feat_dim].
        weight_grad_pointer: Pointer to an optional container the weights' row-wise gradients
            are written to if scale_by_weight is True, which should later be summed.
            The container, if provided, must be of shape [batch_dim/BLOCK_SIZE_BATCH, feat_dim].
        bias_grad_pointer: Pointer to an optional container the bias vector's row-wise gradients
            are written to if scale_by_weight and add_bias are True, which should later be summed.
            The container, if provided, must be of shape [batch_dim/BLOCK_SIZE_BATCH, feat_dim].
        batch_dim: Batch dimension.
        feat_dim: Dimensionality of the features.
        output_grad_batch_stride: Stride necessary to jump one element along the
            output gradients' batch dimension.
        output_grad_feat_stride: Stride necessary to jump one element along the
            output gradients' feature dimension.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        input_grad_batch_stride: Stride necessary to jump one element along the
            input gradient container's batch dimension.
        input_grad_feat_stride: Stride necessary to jump one element along the
            input gradient container's feature dimension.
        weight_grad_batch_stride: Stride necessary to jump one element along the
            weight gradient container's batch dimension.
        weight_grad_feat_stride: Stride necessary to jump one element along the
            weight gradient container's feature dimension.
        bias_grad_batch_stride: Stride necessary to jump one element along the
            weight gradient container's batch dimension.
        bias_grad_feat_stride: Stride necessary to jump one element along the
            weight gradient container's feature dimension.
        scale_by_weight: Flag for scaling the normalized output by weights.
        add_bias: Flag for adding a bias vector to the normalized output
            if scale_by_weight is True.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_FEAT: Block size across the feature dimension.
    """
    batch_pid = tl.program_id(axis=0)
    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH
        )
    feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)
    batch_mask = batch_offset < batch_dim
    feat_mask = feat_offset < feat_dim
    output_grad_pointer += output_grad_batch_stride * batch_offset[:, None
        ] + output_grad_feat_stride * feat_offset[None, :]
    input_pointer += input_batch_stride * batch_offset[:, None
        ] + input_feat_stride * feat_offset[None, :]
    input_grad_pointer += input_grad_batch_stride * batch_offset[:, None
        ] + input_grad_feat_stride * feat_offset[None, :]
    output_grad = tl.load(output_grad_pointer, mask=batch_mask[:, None] &
        feat_mask[None, :]).to(tl.float32)
    input = tl.load(input_pointer, mask=batch_mask[:, None] & feat_mask[
        None, :]).to(tl.float32)
    mean = tl.load(mean_pointer + batch_offset, mask=batch_mask)
    inv_std = tl.load(inv_std_pointer + batch_offset, mask=batch_mask)
    pre_lin = (input - mean[:, None]) * inv_std[:, None]
    if scale_by_weight:
        weight = tl.load(weight_pointer + feat_offset, mask=feat_mask)
        weight_output_grad_prod = weight * output_grad
    else:
        weight_output_grad_prod = output_grad
    term1 = tl.sum(pre_lin * weight_output_grad_prod, axis=1) / feat_dim
    term1 = pre_lin * term1[:, None]
    term2 = tl.sum(weight_output_grad_prod, axis=1) / feat_dim
    input_grad = inv_std[:, None] * (weight_output_grad_prod - (term1 +
        term2[:, None]))
    tl.store(input_grad_pointer, input_grad, mask=batch_mask[:, None] &
        feat_mask[None, :])
    if scale_by_weight:
        weight_grad_pointer += (weight_grad_batch_stride * batch_pid + 
            weight_grad_feat_stride * feat_offset)
        tl.store(weight_grad_pointer, tl.sum(output_grad * pre_lin, axis=0),
            mask=feat_mask)
        if add_bias:
            bias_grad_pointer += (bias_grad_batch_stride * batch_pid + 
                bias_grad_feat_stride * feat_offset)
            tl.store(bias_grad_pointer, tl.sum(output_grad, axis=0), mask=
                feat_mask)


def BLOCK_SIZE_BATCH_heuristic(args: Dict) ->int:
    """
    Approximates an appropriate batch block size for softmax using a heuristic.

    Args:
        args: Arguments to softmax kernel.

    Returns:
        Appropriate batch block size.
    """
    return min(max(1, next_power_of_2(args['batch_dim'] // 2 ** 10)), 128
        ) if args['feat_dim'] < 64 else 1


# Backward method (kernel launch code)
def _LayerNormAutoGrad_backward(ctx: Context, output_grad: Tensor) ->Tuple[
    Optional[Tensor], ...]:
    """
        Calculates the input gradient of layer normalization.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of layer normalization.
        """
    scale_by_weight, add_bias = ctx.scale_by_weight, ctx.add_bias
    flattened_input, mean, inv_std, weight = ctx.saved_tensors
    flattened_output_grad = output_grad.view_as(flattened_input)
    batch_dim, feat_dim = flattened_output_grad.shape
    input_grad = torch.empty_like(flattened_output_grad, dtype=ctx.output_dtype
        )
    if scale_by_weight:
        BLOCK_SIZE_BATCH = BLOCK_SIZE_BATCH_heuristic({'batch_dim':
            batch_dim, 'feat_dim': feat_dim})
        out_batch_dim = batch_dim // BLOCK_SIZE_BATCH
        weight_grad = torch.empty((out_batch_dim, feat_dim), device=
            flattened_input.device)
        if add_bias:
            bias_grad = torch.empty((out_batch_dim, feat_dim), device=
                flattened_input.device)
        else:
            bias_grad = None
    else:
        weight_grad = bias_grad = None
    grid = lambda META: (cdiv(batch_dim, META['BLOCK_SIZE_BATCH']),)
    layer_norm_backward_kernel[grid](flattened_output_grad, flattened_input,
        mean, inv_std, weight, input_grad, weight_grad, bias_grad,
        batch_dim, feat_dim, *flattened_output_grad.stride(), *
        flattened_input.stride(), *input_grad.stride(), *(weight_grad.
        stride() if scale_by_weight else (1, 1)), *(bias_grad.stride() if 
        scale_by_weight and add_bias else (1, 1)), scale_by_weight=
        scale_by_weight, add_bias=add_bias)
    if scale_by_weight:
        weight_grad = weight_grad.sum(dim=0)
        if add_bias:
            bias_grad = bias_grad.sum(dim=0)
    return input_grad.view_as(output_grad), weight_grad, bias_grad, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LayerNormAutoGrad(torch.autograd.Function):
    """
    Autodiff for layer normalization.
    """

    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Optional[Tensor]=None,
        bias: Optional[Tensor]=None, eps: float=1e-05, autocast_to_fp32:
        bool=True) ->Tensor:
        """
        Layer-normalizes the input.

        Args:
            ctx: Context for variable storage.
            input: Input to layer-normalize.
                Can have arbitrary shape.
            weight: Optional weights for affine transform.
                If provided, must be of shape [feat_dim].
            bias: Optional bias vector for affine transform when weight is provided.
                If provided, must be of shape [feat_dim].
            eps: Epsilon added in the square root in the denominator
                to avoid division by zero.
            autocast_to_fp32: Flag for autocasting the output dtype to fp32.
                If False, the input dtype flows through.

        Returns:
            Layer-normalized input.
        """
        flattened_input = input.unsqueeze(0) if input.ndim == 1 else input
        flattened_input = flattened_input.flatten(0, -2)
        batch_dim, feat_dim = flattened_input.shape
        output_dtype = get_output_dtype(input.dtype, autocast='fp32' if
            autocast_to_fp32 else None)
        output = torch.empty_like(flattened_input, dtype=output_dtype)
        scale_by_weight = weight is not None
        add_bias = scale_by_weight and bias is not None
        requires_grad = (input.requires_grad or scale_by_weight and weight.
            requires_grad or add_bias and bias.requires_grad)
        if requires_grad:
            mean = torch.empty(batch_dim, device=input.device, dtype=torch.
                float32)
            inv_std = torch.empty(batch_dim, device=input.device, dtype=
                torch.float32)
        else:
            mean = inv_std = None
        grid = lambda META: (cdiv(batch_dim, META['BLOCK_SIZE_BATCH']),)
        layer_norm_forward_kernel[grid](flattened_input, weight, bias, mean,
            inv_std, output, batch_dim, feat_dim, *flattened_input.stride(),
            *output.stride(), eps, scale_by_weight=scale_by_weight,
            add_bias=add_bias, save_stats=requires_grad)
        ctx.scale_by_weight = scale_by_weight
        ctx.add_bias = add_bias
        ctx.output_dtype = output_dtype
        if requires_grad:
            ctx.save_for_backward(flattened_input, mean, inv_std, weight)
        return output.view_as(input)

    @staticmethod
    def backward(ctx: Context, output_grad: Tensor) ->Tuple[Optional[Tensor
        ], ...]:
        """
        Calculates the input gradient of layer normalization.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of layer normalization.
        """
        scale_by_weight, add_bias = ctx.scale_by_weight, ctx.add_bias
        flattened_input, mean, inv_std, weight = ctx.saved_tensors
        flattened_output_grad = output_grad.view_as(flattened_input)
        batch_dim, feat_dim = flattened_output_grad.shape
        input_grad = torch.empty_like(flattened_output_grad, dtype=ctx.
            output_dtype)
        if scale_by_weight:
            BLOCK_SIZE_BATCH = BLOCK_SIZE_BATCH_heuristic({'batch_dim':
                batch_dim, 'feat_dim': feat_dim})
            out_batch_dim = batch_dim // BLOCK_SIZE_BATCH
            weight_grad = torch.empty((out_batch_dim, feat_dim), device=
                flattened_input.device)
            if add_bias:
                bias_grad = torch.empty((out_batch_dim, feat_dim), device=
                    flattened_input.device)
            else:
                bias_grad = None
        else:
            weight_grad = bias_grad = None
        grid = lambda META: (cdiv(batch_dim, META['BLOCK_SIZE_BATCH']),)
        layer_norm_backward_kernel[grid](flattened_output_grad,
            flattened_input, mean, inv_std, weight, input_grad, weight_grad,
            bias_grad, batch_dim, feat_dim, *flattened_output_grad.stride(),
            *flattened_input.stride(), *input_grad.stride(), *(weight_grad.
            stride() if scale_by_weight else (1, 1)), *(bias_grad.stride() if
            scale_by_weight and add_bias else (1, 1)), scale_by_weight=
            scale_by_weight, add_bias=add_bias)
        if scale_by_weight:
            weight_grad = weight_grad.sum(dim=0)
            if add_bias:
                bias_grad = bias_grad.sum(dim=0)
        return input_grad.view_as(output_grad
            ), weight_grad, bias_grad, None, None
