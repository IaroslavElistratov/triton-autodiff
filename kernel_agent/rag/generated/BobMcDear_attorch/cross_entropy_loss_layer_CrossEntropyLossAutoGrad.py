# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/BobMcDear/attorch
# Source-Files: attorch/cross_entropy_loss_layer.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_leort3lx/attorch-main/attorch/cross_entropy_loss_layer.py
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
def cross_entropy_loss_forward_kernel(input_pointer, target_pointer,
    weight_pointer, sum_weights_pointer, output_pointer, batch_dim,
    feat_dim, input_batch_stride, input_feat_stride, weighted: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_FEAT: tl.constexpr):
    """
    Measures the mean cross entropy loss between the input and target,
    with optional reweighing of each class.

    Args:
        input_pointer: Pointer to the input.
            The input must be of shape [batch_dim, feat_dim].
        target_pointer: Pointer to the target.
            The target must be of shape [batch_dim].
        weight_pointer: Pointer to an optional class weight vector.
            The class weight vector, if provided, must be of shape [feat_dim].
        sum_weights_pointer: Pointer to a container the sum of the class weights is written to.
            The container must be of shape [batch_dim/BLOCK_SIZE_BATCH].
        output_pointer: Pointer to a container the loss is written to.
            The container must be of shape [batch_dim/BLOCK_SIZE_BATCH].
        batch_dim: Batch dimension.
        feat_dim: Dimensionality of the features.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        weighted: Flag for weighing each class.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_FEAT: Block size across the feature dimension.
    """
    batch_pid = tl.program_id(axis=0)
    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH
        )
    feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)
    batch_mask = batch_offset < batch_dim
    feat_mask = feat_offset < feat_dim
    target = tl.load(target_pointer + batch_offset, mask=batch_mask)
    pred_pointer = (input_pointer + input_feat_stride * target + 
        input_batch_stride * batch_offset)
    input_pointer += input_batch_stride * batch_offset[:, None
        ] + input_feat_stride * feat_offset[None, :]
    input = tl.load(input_pointer, mask=batch_mask[:, None] & feat_mask[
        None, :], other=-float('inf')).to(tl.float32)
    pred = tl.load(pred_pointer, mask=batch_mask).to(tl.float32)
    mx = tl.max(input, axis=1)
    input -= mx[:, None]
    loss = tl.log(tl.sum(tl.exp(input), axis=1)) - pred + mx
    if weighted:
        weight = tl.load(weight_pointer + target, mask=batch_mask).to(tl.
            float32)
        loss *= weight
        tl.store(sum_weights_pointer + batch_pid, tl.sum(weight))
    else:
        loss /= batch_dim
    tl.store(output_pointer + batch_pid, tl.sum(loss))


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
def _CrossEntropyLossAutoGrad_forward(ctx: Context, input: Tensor, target:
    Tensor, weight: Optional[Tensor]=None) ->Tensor:
    """
        Measures the mean cross entropy loss between the input and target,
        with optional reweighing of each class.

        Args:
            ctx: Context for variable storage.
            input: Input.
                Must be of shape [batch_dim, feat_dim].
            target: Target.
                Must be of shape [batch_dim].
            weight: Optional class weight vector, with None for no reweighing.
                If provided, must be of shape [feat_dim].

        Returns:
            Loss.
        """
    assert input.ndim == 2, f'Inputs of rank other than 2 not valid'
    assert len(input) == len(target
        ), f'Incompatible input shape ({input.shape}) and target shape ({target.shape})'
    assert weight is None or len(weight) == input.shape[1
        ], f'Dimensionality of weight vector ({len(weight)}) and input features ({input.shape[1]}) not equal'
    batch_dim, feat_dim = input.shape
    BLOCK_SIZE_BATCH = BLOCK_SIZE_BATCH_heuristic({'batch_dim': batch_dim,
        'feat_dim': feat_dim})
    out_batch_dim = batch_dim // BLOCK_SIZE_BATCH
    weighted = weight is not None
    output_dtype = get_output_dtype(input.dtype, autocast='fp32')
    output = torch.empty(out_batch_dim, dtype=output_dtype, device=input.device
        )
    if weighted:
        sum_weights = torch.empty_like(output, dtype=torch.float32)
    else:
        sum_weights = None
    grid = lambda META: (cdiv(len(input), META['BLOCK_SIZE_BATCH']),)
    cross_entropy_loss_forward_kernel[grid](input, target, weight,
        sum_weights, output, batch_dim, feat_dim, *input.stride(), weighted
        =weighted)
    output = output.sum()
    if weighted:
        sum_weights = sum_weights.sum()
        output /= sum_weights
    ctx.sum_weights = sum_weights
    ctx.weight = weight
    ctx.output_dtype = output_dtype
    if input.requires_grad:
        ctx.save_for_backward(input, target)
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim', 'feat_dim'])
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic,
    'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim'])})
@triton.jit
def cross_entropy_loss_backward_kernel(output_grad_pointer, target_pointer,
    input_pointer, weight_pointer, sum_weights_pointer, input_grad_pointer,
    batch_dim, feat_dim, input_batch_stride, input_feat_stride,
    input_grad_batch_stride, input_grad_feat_stride, weighted: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_FEAT: tl.constexpr):
    """
    Calculates the input gradient of cross entropy loss.

    Args:
        output_grad_pointer: Pointer to the loss's output gradients.
            The output gradient must be a scalar.
        target_pointer: Pointer to the target.
            The target must be of shape [batch_dim].
        input_pointer: Pointer to the input.
            The input must be of shape [batch_dim, feat_dim].
        weight_pointer: Pointer to an optional class weight vector.
            The class weight vector, if provided, must be of shape [feat_dim].
        sum_weights_pointer: Pointer to the sum of the class weights if the classes were weighed.
            The sum of weights must be a scalar.
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [batch_dim, feat_dim].
        batch_dim: Batch dimension.
        feat_dim: Dimensionality of the features.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        input_grad_batch_stride: Stride necessary to jump one element along the
            input gradient container's batch dimension.
        input_grad_feat_stride: Stride necessary to jump one element along the
            input gradient container's feature dimension.
        weighted: Flag for weighing each class.
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
    input_grad_pointer += input_grad_batch_stride * batch_offset[:, None
        ] + input_grad_feat_stride * feat_offset[None, :]
    input = tl.load(input_pointer, mask=batch_mask[:, None] & feat_mask[
        None, :], other=-float('inf')).to(tl.float32)
    input -= tl.max(input, axis=1)[:, None]
    numerator = tl.exp(input)
    softmax = numerator / tl.sum(numerator, axis=1)[:, None]
    output_grad = tl.load(output_grad_pointer).to(tl.float32)
    target = tl.load(target_pointer + batch_offset, mask=batch_mask)
    broadcasted_feat_offset = tl.broadcast_to(feat_offset[None, :], (
        BLOCK_SIZE_BATCH, BLOCK_SIZE_FEAT))
    broadcasted_target = tl.broadcast_to(target[:, None], (BLOCK_SIZE_BATCH,
        BLOCK_SIZE_FEAT))
    input_grad = output_grad * (softmax - (broadcasted_feat_offset ==
        broadcasted_target))
    if weighted:
        weight = tl.load(weight_pointer + target, mask=batch_mask).to(tl.
            float32)
        sum_weights = tl.load(sum_weights_pointer)
        input_grad *= weight[:, None] / sum_weights
    else:
        input_grad /= batch_dim
    tl.store(input_grad_pointer, input_grad, mask=batch_mask[:, None] &
        feat_mask[None, :])


# Backward method (kernel launch code)
def _CrossEntropyLossAutoGrad_backward(ctx: Context, output_grad: Tensor
    ) ->Tuple[Optional[Tensor], ...]:
    """
        Calculates the input gradient of the loss.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be a scalar.

        Returns:
            Input gradient of the loss.
        """
    input, target = ctx.saved_tensors
    batch_dim, feat_dim = input.shape
    input_grad = torch.empty_like(input, dtype=ctx.output_dtype)
    grid = lambda META: (cdiv(len(input), META['BLOCK_SIZE_BATCH']),)
    cross_entropy_loss_backward_kernel[grid](output_grad, target, input,
        ctx.weight, ctx.sum_weights, input_grad, batch_dim, feat_dim, *
        input.stride(), *input_grad.stride(), weighted=ctx.weight is not None)
    return input_grad, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class CrossEntropyLossAutoGrad(torch.autograd.Function):
    """
    Autodiff for cross entropy loss.
    """

    @staticmethod
    def forward(ctx: Context, input: Tensor, target: Tensor, weight:
        Optional[Tensor]=None) ->Tensor:
        """
        Measures the mean cross entropy loss between the input and target,
        with optional reweighing of each class.

        Args:
            ctx: Context for variable storage.
            input: Input.
                Must be of shape [batch_dim, feat_dim].
            target: Target.
                Must be of shape [batch_dim].
            weight: Optional class weight vector, with None for no reweighing.
                If provided, must be of shape [feat_dim].

        Returns:
            Loss.
        """
        assert input.ndim == 2, f'Inputs of rank other than 2 not valid'
        assert len(input) == len(target
            ), f'Incompatible input shape ({input.shape}) and target shape ({target.shape})'
        assert weight is None or len(weight) == input.shape[1
            ], f'Dimensionality of weight vector ({len(weight)}) and input features ({input.shape[1]}) not equal'
        batch_dim, feat_dim = input.shape
        BLOCK_SIZE_BATCH = BLOCK_SIZE_BATCH_heuristic({'batch_dim':
            batch_dim, 'feat_dim': feat_dim})
        out_batch_dim = batch_dim // BLOCK_SIZE_BATCH
        weighted = weight is not None
        output_dtype = get_output_dtype(input.dtype, autocast='fp32')
        output = torch.empty(out_batch_dim, dtype=output_dtype, device=
            input.device)
        if weighted:
            sum_weights = torch.empty_like(output, dtype=torch.float32)
        else:
            sum_weights = None
        grid = lambda META: (cdiv(len(input), META['BLOCK_SIZE_BATCH']),)
        cross_entropy_loss_forward_kernel[grid](input, target, weight,
            sum_weights, output, batch_dim, feat_dim, *input.stride(),
            weighted=weighted)
        output = output.sum()
        if weighted:
            sum_weights = sum_weights.sum()
            output /= sum_weights
        ctx.sum_weights = sum_weights
        ctx.weight = weight
        ctx.output_dtype = output_dtype
        if input.requires_grad:
            ctx.save_for_backward(input, target)
        return output

    @staticmethod
    def backward(ctx: Context, output_grad: Tensor) ->Tuple[Optional[Tensor
        ], ...]:
        """
        Calculates the input gradient of the loss.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be a scalar.

        Returns:
            Input gradient of the loss.
        """
        input, target = ctx.saved_tensors
        batch_dim, feat_dim = input.shape
        input_grad = torch.empty_like(input, dtype=ctx.output_dtype)
        grid = lambda META: (cdiv(len(input), META['BLOCK_SIZE_BATCH']),)
        cross_entropy_loss_backward_kernel[grid](output_grad, target, input,
            ctx.weight, ctx.sum_weights, input_grad, batch_dim, feat_dim, *
            input.stride(), *input_grad.stride(), weighted=ctx.weight is not
            None)
        return input_grad, None, None
