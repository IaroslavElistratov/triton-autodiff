# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/BobMcDear/attorch
# Source-Files: attorch/softmax_layers.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_leort3lx/attorch-main/attorch/softmax_layers.py
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

@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim', 'feat_dim'])
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic,
    'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim'])})
@triton.jit
def softmax_forward_kernel(input_pointer, output_pointer, batch_dim,
    feat_dim, input_batch_stride, input_feat_stride, output_batch_stride,
    output_feat_stride, neg: tl.constexpr, log: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_FEAT: tl.constexpr):
    """
    Normalizes the input using softmax.

    Args:
        input_pointer: Pointer to the input to normalize.
            The input must be of shape [batch_dim, feat_dim].
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
        neg: Flag indicating if the input should be negated to get softmin.
        log: Flag indicating if the log of softmax should be taken.
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
        None, :], other=float('inf') if neg else -float('inf')).to(tl.float32)
    if neg:
        input = -input
    input -= tl.max(input, axis=1)[:, None]
    numerator = tl.exp(input)
    denominator = tl.sum(numerator, axis=1)[:, None]
    if log:
        output = input - tl.log(denominator)
    else:
        output = numerator / denominator
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
def _SoftmaxAutoGrad_forward(ctx: Context, input: Tensor, neg: bool, log: bool
    ) ->Tensor:
    """
        Normalizes the input using softmax.

        Args:
            ctx: Context for variable storage.
            input: Input to normalize.
                Can have arbitrary shape.
            neg: Flag indicating if the input should be negated to get softmin.
            log: Flag indicating if the log of softmax should be taken.

        Returns:
            Input normalized by softmax.
        """
    flattened_input = input.unsqueeze(0) if input.ndim == 1 else input
    flattened_input = flattened_input.flatten(0, -2)
    batch_dim, feat_dim = flattened_input.shape
    output_dtype = get_output_dtype(input.dtype, autocast='fp32')
    output = torch.empty_like(flattened_input, dtype=output_dtype)
    grid = lambda META: (cdiv(batch_dim, META['BLOCK_SIZE_BATCH']),)
    softmax_forward_kernel[grid](flattened_input, output, batch_dim,
        feat_dim, *flattened_input.stride(), *output.stride(), neg=neg, log=log
        )
    ctx.neg = neg
    ctx.log = log
    if input.requires_grad:
        ctx.save_for_backward(output)
    return output.view_as(input)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim', 'feat_dim'])
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic,
    'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim'])})
@triton.jit
def softmax_backward_kernel(output_grad_pointer, output_pointer,
    input_grad_pointer, batch_dim, feat_dim, output_grad_batch_stride,
    output_grad_feat_stride, output_batch_stride, output_feat_stride,
    input_grad_batch_stride, input_grad_feat_stride, neg: tl.constexpr, log:
    tl.constexpr, BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_FEAT: tl.constexpr
    ):
    """
    Calculates the input gradient of softmax.

    Args:
        output_grad_pointer: Pointer to softmax's output gradients.
            The output gradients must be of shape [batch_dim, feat_dim].
        output_pointer: Pointer to softmax's output.
            The output must be of shape [batch_dim, feat_dim].
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [batch_dim, feat_dim].
        batch_dim: Batch dimension.
        feat_dim: Dimensionality of the features.
        output_grad_batch_stride: Stride necessary to jump one element along the
            output gradients' batch dimension.
        output_grad_feat_stride: Stride necessary to jump one element along the
            output gradients' feature dimension.
        output_batch_stride: Stride necessary to jump one element along the
            output's batch dimension.
        output_feat_stride: Stride necessary to jump one element along the
            output's feature dimension.
        input_grad_batch_stride: Stride necessary to jump one element along the
            input gradient container's batch dimension.
        input_grad_feat_stride: Stride necessary to jump one element along the
            input gradient container's feature dimension.
        neg: Flag indicating if the input was negated to get softmin.
        log: Flag indicating if log of softmax was taken.
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
    output_pointer += output_batch_stride * batch_offset[:, None
        ] + output_feat_stride * feat_offset[None, :]
    input_grad_pointer += input_grad_batch_stride * batch_offset[:, None
        ] + input_grad_feat_stride * feat_offset[None, :]
    output_grad = tl.load(output_grad_pointer, mask=batch_mask[:, None] &
        feat_mask[None, :]).to(tl.float32)
    output = tl.load(output_pointer, mask=batch_mask[:, None] & feat_mask[
        None, :]).to(tl.float32)
    if log:
        input_grad = output_grad - tl.exp(output) * tl.sum(output_grad, axis=1
            )[:, None]
    else:
        input_grad = output * (output_grad - tl.sum(output_grad * output,
            axis=1)[:, None])
    tl.store(input_grad_pointer, -input_grad if neg else input_grad, mask=
        batch_mask[:, None] & feat_mask[None, :])


# Backward method (kernel launch code)
def _SoftmaxAutoGrad_backward(ctx: Context, output_grad: Tensor) ->Tuple[
    Optional[Tensor], ...]:
    """
        Calculates the input gradient of softmax.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of softmax.
        """
    output, = ctx.saved_tensors
    flattened_output_grad = output_grad.view_as(output)
    batch_dim, feat_dim = output.shape
    input_grad = torch.empty_like(output)
    grid = lambda META: (cdiv(batch_dim, META['BLOCK_SIZE_BATCH']),)
    softmax_backward_kernel[grid](flattened_output_grad, output, input_grad,
        batch_dim, feat_dim, *flattened_output_grad.stride(), *output.
        stride(), *input_grad.stride(), neg=ctx.neg, log=ctx.log)
    return input_grad.view_as(output_grad), None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class SoftmaxAutoGrad(torch.autograd.Function):
    """
    Autodiff for softmax and related functions.
    """

    @staticmethod
    def forward(ctx: Context, input: Tensor, neg: bool, log: bool) ->Tensor:
        """
        Normalizes the input using softmax.

        Args:
            ctx: Context for variable storage.
            input: Input to normalize.
                Can have arbitrary shape.
            neg: Flag indicating if the input should be negated to get softmin.
            log: Flag indicating if the log of softmax should be taken.

        Returns:
            Input normalized by softmax.
        """
        flattened_input = input.unsqueeze(0) if input.ndim == 1 else input
        flattened_input = flattened_input.flatten(0, -2)
        batch_dim, feat_dim = flattened_input.shape
        output_dtype = get_output_dtype(input.dtype, autocast='fp32')
        output = torch.empty_like(flattened_input, dtype=output_dtype)
        grid = lambda META: (cdiv(batch_dim, META['BLOCK_SIZE_BATCH']),)
        softmax_forward_kernel[grid](flattened_input, output, batch_dim,
            feat_dim, *flattened_input.stride(), *output.stride(), neg=neg,
            log=log)
        ctx.neg = neg
        ctx.log = log
        if input.requires_grad:
            ctx.save_for_backward(output)
        return output.view_as(input)

    @staticmethod
    def backward(ctx: Context, output_grad: Tensor) ->Tuple[Optional[Tensor
        ], ...]:
        """
        Calculates the input gradient of softmax.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of softmax.
        """
        output, = ctx.saved_tensors
        flattened_output_grad = output_grad.view_as(output)
        batch_dim, feat_dim = output.shape
        input_grad = torch.empty_like(output)
        grid = lambda META: (cdiv(batch_dim, META['BLOCK_SIZE_BATCH']),)
        softmax_backward_kernel[grid](flattened_output_grad, output,
            input_grad, batch_dim, feat_dim, *flattened_output_grad.stride(
            ), *output.stride(), *input_grad.stride(), neg=ctx.neg, log=ctx.log
            )
        return input_grad.view_as(output_grad), None, None
