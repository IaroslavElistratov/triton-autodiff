# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/BobMcDear/attorch
# Source-Files: attorch/p_loss_layers.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_leort3lx/attorch-main/attorch/p_loss_layers.py
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

@triton.autotune(configs=element_wise_kernel_configs(), key=['size'])
@triton.jit
def p_loss_forward_kernel(input_pointer, target_pointer, output_pointer,
    param, size, p_loss: tl.constexpr, reduction: tl.constexpr, BLOCK_SIZE:
    tl.constexpr):
    """
    Measures the smooth L1, L1, squared L2, or Huber loss of the difference
    between the input and target.

    Args:
        input_pointer: Pointer to the input.
            The input must be of shape [size].
        target_pointer: Pointer to the target.
            The target must be of shape [size].
        output_pointer: Pointer to a container the error is written to.
            The container must be of shape [size] if reduction is 'none',
            and otherwise of shape [size/BLOCK_SIZE].
        param: Parameter of loss function (i.e., beta or delta for smooth L1 and Huber).
        size: Number of elements in the input and target.
        p_loss: p-norm used to compute the error.
            Options are 0 for smooth L1, 1 for L1, 2 for squared L2, and 3 for Huber loss.
        reduction: Reduction strategy for the output.
            Options are 'none' for no reduction, 'mean' for averaging the error
            across all entries, and 'sum' for summing the error across all entries.
            If a reduction method is specified, the reduced result of each
            program is written to a separate index in the output container,
            which should later be summed.
        BLOCK_SIZE: Block size.
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    input = tl.load(input_pointer + offset, mask=mask).to(tl.float32)
    target = tl.load(target_pointer + offset, mask=mask).to(tl.float32)
    diff = input - target
    if p_loss == 0:
        error = tl.where(diff < param, 0.5 * diff * diff / param, tl.abs(
            diff) - 0.5 * param)
    elif p_loss == 1:
        error = tl.abs(diff)
    elif p_loss == 2:
        error = diff * diff
    elif p_loss == 3:
        error = tl.where(diff < param, 0.5 * diff * diff, param * (tl.abs(
            diff) - 0.5 * param))
    if reduction == 'none':
        tl.store(output_pointer + offset, error, mask=mask)
    elif reduction == 'mean':
        tl.store(output_pointer + pid, tl.sum(error) / size)
    elif reduction == 'sum':
        tl.store(output_pointer + pid, tl.sum(error))


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
def _PLossAutoGrad_forward(ctx: Context, input: Tensor, target: Tensor,
    p_loss: int, reduction: str, param: float=1.0) ->Tensor:
    """
        Measures the smooth L1, L1, squared L2, or Huber loss of the difference
        between the input and target.

        Args:
            ctx: Context for variable storage.
            input: Input.
                Can have arbitrary shape.
            target: Target.
                Must be the same shape as input.
            p_loss: p-norm used to compute the error.
                Options are 0 for smooth L1, 1 for L1, 2 for squared L2, and 3 for Huber loss.
            reduction: Reduction strategy for the output.
                Options are 'none' for no reduction, 'mean' for averaging the error
                across all entries, and 'sum' for summing the error across all entries.
            param: Parameter of loss function (i.e., beta or delta for smooth L1 and Huber).

        Returns:
            Error.
        """
    assert input.shape == target.shape, f'Input shape {input.shape} and target shape {target.shape} not equal'
    output_dtype = get_output_dtype(input.dtype, autocast='fp32')
    ctx.p_loss = p_loss
    ctx.reduction = reduction
    ctx.param = param
    ctx.output_dtype = output_dtype
    if input.requires_grad or target.requires_grad:
        ctx.save_for_backward(input, target)
    flattened_input = input.flatten()
    flattened_target = target.flatten()
    size = len(flattened_input)
    output = torch.empty_like(flattened_input, dtype=output_dtype
        ) if reduction == 'none' else torch.empty(cdiv(size, 32), dtype=
        output_dtype, device=input.device)
    grid = lambda META: (cdiv(size, META['BLOCK_SIZE']),)
    p_loss_forward_kernel[grid](flattened_input, flattened_target, output,
        param, size, p_loss=p_loss, reduction=reduction)
    if reduction != 'none':
        BLOCK_SIZE = p_loss_forward_kernel.best_config.kwargs['BLOCK_SIZE']
        output = output[:cdiv(size, BLOCK_SIZE)].sum()
    else:
        output = output.view_as(input)
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=element_wise_kernel_configs(), key=['size'])
@triton.jit
def p_loss_backward_kernel(output_grad_pointer, input_pointer,
    target_pointer, input_grad_pointer, target_grad_pointer, param, size,
    p_loss: tl.constexpr, reduction: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Calculates the input gradient of the smooth L1 norm, L1 norm, L2 norm, or Huber loss.

    Args:
        output_grad_pointer: Pointer to the error's output gradients.
            The output gradients must be a scalar or of shape [size].
        input_pointer: Pointer to the input.
            The input must be of shape [size].
        target_pointer: Pointer to the target.
            The target must be of shape [size].
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [size].
        target_grad_pointer: Pointer to a container the target's gradients are written to.
            The container must be of shape [size].
        param: Parameter of loss function (i.e., beta or delta for smooth L1 and Huber).
        size: Number of elements in the input and target.
        p_loss: p-norm used to compute the error whose gradient is calculated.
            Options are 0 for smooth L1, 1 for L1, 2 for squared L2, and 3 for Huber loss.
        reduction: Reduction strategy for the output whose gradient is calculated.
            Options are 'none' for no reduction, 'mean' for averaging the error
            across all entries, and 'sum' for summing the error across all entries.
        BLOCK_SIZE: Block size.
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    output_grad_mask = None
    if reduction == 'none':
        output_grad_pointer += offset
        output_grad_mask = mask
    input = tl.load(input_pointer + offset, mask=mask).to(tl.float32)
    target = tl.load(target_pointer + offset, mask=mask).to(tl.float32)
    diff = input - target
    output_grad = tl.load(output_grad_pointer, mask=output_grad_mask).to(tl
        .float32)
    if p_loss == 0:
        input_grad = tl.where(diff < param, diff / param, tl.where(0 <=
            diff, 1, -1))
    elif p_loss == 1:
        input_grad = tl.where(0 <= diff, 1, -1)
    elif p_loss == 2:
        input_grad = 2 * diff
    elif p_loss == 3:
        input_grad = tl.where(diff < param, diff, param * tl.where(0 <=
            diff, 1, -1))
    if reduction == 'mean':
        input_grad /= size
    input_grad *= output_grad
    tl.store(input_grad_pointer + offset, input_grad, mask=mask)
    tl.store(target_grad_pointer + offset, -input_grad, mask=mask)


# Backward method (kernel launch code)
def _PLossAutoGrad_backward(ctx: Context, output_grad: Tensor) ->Tuple[
    Optional[Tensor], ...]:
    """
        Calculates the input gradient of the error.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of the error.
        """
    input, target = ctx.saved_tensors
    flattened_input = input.flatten()
    flattened_target = target.flatten()
    output_grad = output_grad.flatten()
    size = len(flattened_input)
    input_grad = torch.empty_like(flattened_input, dtype=ctx.output_dtype)
    target_grad = torch.empty_like(flattened_target, dtype=ctx.output_dtype)
    grid = lambda META: (cdiv(size, META['BLOCK_SIZE']),)
    p_loss_backward_kernel[grid](output_grad, flattened_input,
        flattened_target, input_grad, target_grad, ctx.param, size, p_loss=
        ctx.p_loss, reduction=ctx.reduction)
    return input_grad.view_as(input), target_grad.view_as(input), None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class PLossAutoGrad(torch.autograd.Function):
    """
    Autodiff for p-losses.
    """

    @staticmethod
    def forward(ctx: Context, input: Tensor, target: Tensor, p_loss: int,
        reduction: str, param: float=1.0) ->Tensor:
        """
        Measures the smooth L1, L1, squared L2, or Huber loss of the difference
        between the input and target.

        Args:
            ctx: Context for variable storage.
            input: Input.
                Can have arbitrary shape.
            target: Target.
                Must be the same shape as input.
            p_loss: p-norm used to compute the error.
                Options are 0 for smooth L1, 1 for L1, 2 for squared L2, and 3 for Huber loss.
            reduction: Reduction strategy for the output.
                Options are 'none' for no reduction, 'mean' for averaging the error
                across all entries, and 'sum' for summing the error across all entries.
            param: Parameter of loss function (i.e., beta or delta for smooth L1 and Huber).

        Returns:
            Error.
        """
        assert input.shape == target.shape, f'Input shape {input.shape} and target shape {target.shape} not equal'
        output_dtype = get_output_dtype(input.dtype, autocast='fp32')
        ctx.p_loss = p_loss
        ctx.reduction = reduction
        ctx.param = param
        ctx.output_dtype = output_dtype
        if input.requires_grad or target.requires_grad:
            ctx.save_for_backward(input, target)
        flattened_input = input.flatten()
        flattened_target = target.flatten()
        size = len(flattened_input)
        output = torch.empty_like(flattened_input, dtype=output_dtype
            ) if reduction == 'none' else torch.empty(cdiv(size, 32), dtype
            =output_dtype, device=input.device)
        grid = lambda META: (cdiv(size, META['BLOCK_SIZE']),)
        p_loss_forward_kernel[grid](flattened_input, flattened_target,
            output, param, size, p_loss=p_loss, reduction=reduction)
        if reduction != 'none':
            BLOCK_SIZE = p_loss_forward_kernel.best_config.kwargs['BLOCK_SIZE']
            output = output[:cdiv(size, BLOCK_SIZE)].sum()
        else:
            output = output.view_as(input)
        return output

    @staticmethod
    def backward(ctx: Context, output_grad: Tensor) ->Tuple[Optional[Tensor
        ], ...]:
        """
        Calculates the input gradient of the error.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of the error.
        """
        input, target = ctx.saved_tensors
        flattened_input = input.flatten()
        flattened_target = target.flatten()
        output_grad = output_grad.flatten()
        size = len(flattened_input)
        input_grad = torch.empty_like(flattened_input, dtype=ctx.output_dtype)
        target_grad = torch.empty_like(flattened_target, dtype=ctx.output_dtype
            )
        grid = lambda META: (cdiv(size, META['BLOCK_SIZE']),)
        p_loss_backward_kernel[grid](output_grad, flattened_input,
            flattened_target, input_grad, target_grad, ctx.param, size,
            p_loss=ctx.p_loss, reduction=ctx.reduction)
        return input_grad.view_as(input), target_grad.view_as(input
            ), None, None
