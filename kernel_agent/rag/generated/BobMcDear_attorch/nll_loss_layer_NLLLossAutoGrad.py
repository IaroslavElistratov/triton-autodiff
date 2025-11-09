# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/BobMcDear/attorch
# Source-Files: attorch/nll_loss_layer.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_leort3lx/attorch-main/attorch/nll_loss_layer.py
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

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim',
    'spatial_dim'])
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic,
    'BLOCK_SIZE_SPATIAL': lambda args: next_power_of_2(args['spatial_dim'])})
@triton.jit
def nll_loss_forward_kernel(input_pointer, target_pointer, weight_pointer,
    sum_weights_pointer, output_pointer, batch_dim, spatial_dim,
    input_batch_stride, input_feat_stride, input_spatial_stride,
    target_batch_stride, target_spatial_stride, output_batch_stride,
    output_spatial_stride, reduction: tl.constexpr, weighted: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_SPATIAL: tl.constexpr):
    """
    Measures the negative log likelihood loss between the input and target,
    with optional reweighing of each class.

    Args:
        input_pointer: Pointer to the input.
            The input must be of shape [batch_dim, feat_dim, spatial_dim].
        target_pointer: Pointer to the target.
            The target must be of shape [batch_dim, spatial_dim].
        weight_pointer: Pointer to an optional class weight vector.
            The class weight vector, if provided, must be of shape [feat_dim].
        sum_weights_pointer: Pointer to a container the sum of the class weights is written to.
            The container must be of shape [batch_dim/BLOCK_SIZE_BATCH].
        output_pointer: Pointer to a container the loss is written to.
            The container must be of shape [batch_dim, spatial_dim] if reduction is 'none',
            and otherwise of shape [batch_dim/BLOCK_SIZE].
        batch_dim: Batch dimension.
        spatial_dim: Spatial dimension.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        input_spatial_stride: Stride necessary to jump one element along the
            input's spatial dimension.
        target_batch_stride: Stride necessary to jump one element along the
            target's batch dimension.
        target_spatial_stride: Stride necessary to jump one element along the
            target's spatial dimension.
        output_batch_stride: Stride necessary to jump one element along the
            output container's batch dimension.
        output_spatial_stride: Stride necessary to jump one element along the
            output container's spatial dimension.
        reduction: Reduction strategy for the output.
            Options are 'none' for no reduction, 'mean' for averaging the loss
            across all entries, and 'sum' for summing the loss across all entries.
            If a reduction method is specified, the reduced result of each
            program is written to a separate index in the summed weights and
            output container, which should later be summed.
        weighted: Flag for weighing each class.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_SPATIAL: Block size across the spatial dimension.
    """
    batch_pid = tl.program_id(axis=0)
    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH
        )
    spatial_offset = tl.arange(0, BLOCK_SIZE_SPATIAL)
    batch_mask = batch_offset < batch_dim
    spatial_mask = spatial_offset < spatial_dim
    target_pointer += target_batch_stride * batch_offset[:, None
        ] + target_spatial_stride * spatial_offset[None, :]
    target = tl.load(target_pointer, mask=batch_mask[:, None] &
        spatial_mask[None, :])
    input_pointer += (input_feat_stride * target + input_batch_stride *
        batch_offset[:, None] + input_spatial_stride * spatial_offset[None, :])
    input = tl.load(input_pointer, mask=batch_mask[:, None] & spatial_mask[
        None, :]).to(tl.float32)
    output = -input
    if weighted:
        weight = tl.load(weight_pointer + target, mask=batch_mask[:, None] &
            spatial_mask[None, :]).to(tl.float32)
        output *= weight
    if reduction == 'none':
        output_pointer += output_batch_stride * batch_offset[:, None
            ] + output_spatial_stride * spatial_offset[None, :]
        tl.store(output_pointer, output, mask=batch_mask[:, None] &
            spatial_mask[None, :])
    elif reduction == 'mean':
        if weighted:
            tl.store(sum_weights_pointer + batch_pid, tl.sum(weight))
            tl.store(output_pointer + batch_pid, tl.sum(output))
        else:
            tl.store(output_pointer + batch_pid, tl.sum(output) / (
                batch_dim * spatial_dim))
    elif reduction == 'sum':
        tl.store(output_pointer + batch_pid, tl.sum(output))


def BLOCK_SIZE_BATCH_heuristic(args) ->int:
    """
    Approximates an appropriate batch block size for NLL loss using a heuristic.

    Args:
        args: Arguments to NLL loss kernel.

    Returns:
        Appropriate batch block size.
    """
    return min(max(1, next_power_of_2(args['batch_dim'] // 2 ** 10)), 128
        ) if args['spatial_dim'] < 64 else 1


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
def _NLLLossAutoGrad_forward(ctx: Context, input: Tensor, target: Tensor,
    reduction: str, weight: Optional[Tensor]=None) ->Tensor:
    """
        Measures the negative log likelihood loss between the input and target,
        with optional reweighing of each class.

        Args:
            ctx: Context for variable storage.
            input: Input.
                Must be of shape [batch_dim, feat_dim, ...],
                where ... denotes an arbitrary number of spatial dimensions.
            target: Target.
                Must be of shape [batch_dim, ...],
                where ... denotes the same spatial dimensions as the input.
            reduction: Reduction strategy for the output.
                Options are 'none' for no reduction, 'mean' for averaging the loss
                across all entries, and 'sum' for summing the loss across all entries.
            weight: Optional class weight vector, with None for no reweighing.
                If provided, must be of shape [feat_dim].

        Returns:
            Loss.
        """
    assert len(input) == len(target) and input.shape[2:] == target.shape[1:
        ], f'Incompatible input shape ({input.shape}) and target shape ({target.shape})'
    assert weight is None or len(weight) == input.shape[1
        ], f'Dimensionality of weight vector ({len(weight)}) and input features ({input.shape[1]}) not equal'
    flattened_input = input.unsqueeze(-1) if input.ndim == 2 else input
    flattened_input = flattened_input.flatten(2, -1)
    flattened_target = target.unsqueeze(-1) if target.ndim == 1 else target
    flattened_target = flattened_target.flatten(1, -1)
    batch_dim, _, spatial_dim = flattened_input.shape
    BLOCK_SIZE_BATCH = BLOCK_SIZE_BATCH_heuristic({'batch_dim': batch_dim,
        'spatial_dim': spatial_dim})
    out_batch_dim = batch_dim // BLOCK_SIZE_BATCH
    output_dtype = get_output_dtype(input.dtype, autocast='fp32')
    sum_weights = torch.empty(out_batch_dim, dtype=torch.float32, device=
        input.device) if reduction == 'mean' else None
    output = torch.empty_like(flattened_target, dtype=output_dtype
        ) if reduction == 'none' else torch.empty(out_batch_dim, dtype=
        output_dtype, device=input.device)
    grid = lambda META: (cdiv(len(input), META['BLOCK_SIZE_BATCH']),)
    nll_loss_forward_kernel[grid](input, target, weight, sum_weights,
        output, batch_dim, spatial_dim, *flattened_input.stride(), *
        flattened_target.stride(), *(output.stride() if reduction == 'none'
         else (1, 1)), reduction=reduction, weighted=weight is not None)
    if reduction != 'none':
        output = output.sum()
        if reduction == 'mean' and weight is not None:
            sum_weights = sum_weights.sum()
            output /= sum_weights
    else:
        output = output.view_as(target)
    ctx.sum_weights = sum_weights
    ctx.reduction = reduction
    ctx.weight = weight
    ctx.output_dtype = output_dtype
    if input.requires_grad:
        ctx.save_for_backward(input, flattened_target)
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim',
    'spatial_dim'])
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic,
    'BLOCK_SIZE_SPATIAL': lambda args: next_power_of_2(args['spatial_dim'])})
@triton.jit
def nll_loss_backward_kernel(output_grad_pointer, target_pointer,
    weight_pointer, sum_weights_pointer, input_grad_pointer, batch_dim,
    spatial_dim, output_grad_batch_stride, output_grad_feat_stride,
    target_batch_stride, target_spatial_stride, input_grad_batch_stride,
    input_grad_feat_stride, input_grad_spatial_stride, reduction: tl.
    constexpr, weighted: tl.constexpr, BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_SPATIAL: tl.constexpr):
    """
    Calculates the input gradient of negative log likelihood loss.

    Args:
        output_grad_pointer: Pointer to the loss's output gradients.
            The output gradients must be of shape [batch_dim, spatial_dim]
            if reduction is 'none', and otherwise [batch_dim/BLOCK_SIZE_BATCH].
        target_pointer: Pointer to the target.
            The target must be of shape [batch_dim, spatial_dim].
        weight_pointer: Pointer to an optional class weight vector.
            The class weight vector, if provided, must be of shape [feat_dim].
        sum_weights_pointer: Pointer to the sum of the class weights if the classes were weighed.
            The sum of weights must be a scalar.
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [batch_dim, feat_dim, spatial_dim] and zeroed.
        batch_dim: Batch dimension.
        spatial_dim: Spatial dimension.
        output_grad_batch_stride: Stride necessary to jump one element along the
            output gradients' batch dimension.
        output_grad_feat_stride: Stride necessary to jump one element along the
            output gradients' feature dimension.
        input_spatial_stride: Stride necessary to jump one element along the
            input's spatial dimension.
        target_batch_stride: Stride necessary to jump one element along the
            target's batch dimension.
        target_spatial_stride: Stride necessary to jump one element along the
            target's spatial dimension.
        input_grad_batch_stride: Stride necessary to jump one element along the
            input gradient container's batch dimension.
        input_grad_feat_stride: Stride necessary to jump one element along the
            input gradient container's feature dimension.
        input_grad_spatial_stride: Stride necessary to jump one element along the
            input gradient container's spatial dimension.
        reduction: Reduction strategy for the output whose gradient is calculated.
            Options are 'none' for no reduction, 'mean' for averaging the loss
            across all entries, and 'sum' for summing the loss across all entries.
        weighted: Flag for weighing each class.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_SPATIAL: Block size across the spatial dimension.
    """
    batch_pid = tl.program_id(axis=0)
    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH
        )
    spatial_offset = tl.arange(0, BLOCK_SIZE_SPATIAL)
    batch_mask = batch_offset < batch_dim
    spatial_mask = spatial_offset < spatial_dim
    output_grad_mask = None
    if reduction == 'none':
        output_grad_pointer += output_grad_batch_stride * batch_offset[:, None
            ] + output_grad_feat_stride * spatial_offset[None, :]
        output_grad_mask = batch_mask[:, None] & spatial_mask[None, :]
    output_grad = tl.load(output_grad_pointer, mask=output_grad_mask).to(tl
        .float32)
    input_grad = -output_grad
    target_pointer += target_batch_stride * batch_offset[:, None
        ] + target_spatial_stride * spatial_offset[None, :]
    target = tl.load(target_pointer, mask=batch_mask[:, None] &
        spatial_mask[None, :])
    if weighted:
        weight = tl.load(weight_pointer + target, mask=batch_mask[:, None] &
            spatial_mask[None, :]).to(tl.float32)
        input_grad *= weight
        if reduction == 'mean':
            input_grad /= tl.load(sum_weights_pointer)
    elif reduction == 'mean':
        input_grad /= batch_dim * spatial_dim
    input_grad_pointer += (input_grad_feat_stride * target + 
        input_grad_batch_stride * batch_offset[:, None] + 
        input_grad_spatial_stride * spatial_offset[None, :])
    tl.store(input_grad_pointer, input_grad, mask=batch_mask[:, None] &
        spatial_mask[None, :])


# Backward method (kernel launch code)
def _NLLLossAutoGrad_backward(ctx: Context, output_grad: Tensor) ->Tuple[
    Optional[Tensor], ...]:
    """
        Calculates the input gradient of the loss.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of the loss.
        """
    input, flattened_target = ctx.saved_tensors
    flattened_input = input.view(len(flattened_target), -1,
        flattened_target.shape[-1])
    output_grad = output_grad.view_as(flattened_target
        ) if output_grad.ndim > 0 else output_grad
    batch_dim, _, spatial_dim = flattened_input.shape
    input_grad = torch.zeros_like(flattened_input, dtype=ctx.output_dtype)
    grid = lambda META: (cdiv(len(input), META['BLOCK_SIZE_BATCH']),)
    nll_loss_backward_kernel[grid](output_grad, flattened_target, ctx.
        weight, ctx.sum_weights, input_grad, batch_dim, spatial_dim, *(
        output_grad.stride() if ctx.reduction == 'none' else (1, 1)), *
        flattened_target.stride(), *input_grad.stride(), reduction=ctx.
        reduction, weighted=ctx.weight is not None)
    return input_grad.view_as(input), None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class NLLLossAutoGrad(torch.autograd.Function):
    """
    Autodiff for negative log likelihood loss.
    """

    @staticmethod
    def forward(ctx: Context, input: Tensor, target: Tensor, reduction: str,
        weight: Optional[Tensor]=None) ->Tensor:
        """
        Measures the negative log likelihood loss between the input and target,
        with optional reweighing of each class.

        Args:
            ctx: Context for variable storage.
            input: Input.
                Must be of shape [batch_dim, feat_dim, ...],
                where ... denotes an arbitrary number of spatial dimensions.
            target: Target.
                Must be of shape [batch_dim, ...],
                where ... denotes the same spatial dimensions as the input.
            reduction: Reduction strategy for the output.
                Options are 'none' for no reduction, 'mean' for averaging the loss
                across all entries, and 'sum' for summing the loss across all entries.
            weight: Optional class weight vector, with None for no reweighing.
                If provided, must be of shape [feat_dim].

        Returns:
            Loss.
        """
        assert len(input) == len(target) and input.shape[2:] == target.shape[1:
            ], f'Incompatible input shape ({input.shape}) and target shape ({target.shape})'
        assert weight is None or len(weight) == input.shape[1
            ], f'Dimensionality of weight vector ({len(weight)}) and input features ({input.shape[1]}) not equal'
        flattened_input = input.unsqueeze(-1) if input.ndim == 2 else input
        flattened_input = flattened_input.flatten(2, -1)
        flattened_target = target.unsqueeze(-1) if target.ndim == 1 else target
        flattened_target = flattened_target.flatten(1, -1)
        batch_dim, _, spatial_dim = flattened_input.shape
        BLOCK_SIZE_BATCH = BLOCK_SIZE_BATCH_heuristic({'batch_dim':
            batch_dim, 'spatial_dim': spatial_dim})
        out_batch_dim = batch_dim // BLOCK_SIZE_BATCH
        output_dtype = get_output_dtype(input.dtype, autocast='fp32')
        sum_weights = torch.empty(out_batch_dim, dtype=torch.float32,
            device=input.device) if reduction == 'mean' else None
        output = torch.empty_like(flattened_target, dtype=output_dtype
            ) if reduction == 'none' else torch.empty(out_batch_dim, dtype=
            output_dtype, device=input.device)
        grid = lambda META: (cdiv(len(input), META['BLOCK_SIZE_BATCH']),)
        nll_loss_forward_kernel[grid](input, target, weight, sum_weights,
            output, batch_dim, spatial_dim, *flattened_input.stride(), *
            flattened_target.stride(), *(output.stride() if reduction ==
            'none' else (1, 1)), reduction=reduction, weighted=weight is not
            None)
        if reduction != 'none':
            output = output.sum()
            if reduction == 'mean' and weight is not None:
                sum_weights = sum_weights.sum()
                output /= sum_weights
        else:
            output = output.view_as(target)
        ctx.sum_weights = sum_weights
        ctx.reduction = reduction
        ctx.weight = weight
        ctx.output_dtype = output_dtype
        if input.requires_grad:
            ctx.save_for_backward(input, flattened_target)
        return output

    @staticmethod
    def backward(ctx: Context, output_grad: Tensor) ->Tuple[Optional[Tensor
        ], ...]:
        """
        Calculates the input gradient of the loss.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of the loss.
        """
        input, flattened_target = ctx.saved_tensors
        flattened_input = input.view(len(flattened_target), -1,
            flattened_target.shape[-1])
        output_grad = output_grad.view_as(flattened_target
            ) if output_grad.ndim > 0 else output_grad
        batch_dim, _, spatial_dim = flattened_input.shape
        input_grad = torch.zeros_like(flattened_input, dtype=ctx.output_dtype)
        grid = lambda META: (cdiv(len(input), META['BLOCK_SIZE_BATCH']),)
        nll_loss_backward_kernel[grid](output_grad, flattened_target, ctx.
            weight, ctx.sum_weights, input_grad, batch_dim, spatial_dim, *(
            output_grad.stride() if ctx.reduction == 'none' else (1, 1)), *
            flattened_target.stride(), *input_grad.stride(), reduction=ctx.
            reduction, weighted=ctx.weight is not None)
        return input_grad.view_as(input), None, None, None
