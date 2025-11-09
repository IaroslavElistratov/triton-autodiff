# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/BobMcDear/attorch
# Source-Files: attorch/dropout_layer.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_leort3lx/attorch-main/attorch/dropout_layer.py
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
from random import randint

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

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


@triton.autotune(configs=element_wise_kernel_configs(), key=['size'])
@triton.jit
def dropout_forward_kernel(input_pointer, output_pointer, size, drop_p,
    seed, BLOCK_SIZE: tl.constexpr):
    """
    Randomly zeroes elements in the input.

    Args:
        input_pointer: Pointer to the input to perform dropout on.
            The input must be of shape [size].
        output_pointer: Pointer to a container the result is written to.
            The container must be of shape [size].
        size: Number of elements in the input.
        drop_p: Probability of dropping an element.
        seed: Seed for generating the dropout mask.
        BLOCK_SIZE: Block size.
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    input = tl.load(input_pointer + offset, mask=mask)
    output = apply_dropout(input, drop_p, seed, offset)
    tl.store(output_pointer + offset, output, mask=mask)


# Forward method (kernel launch code)
@custom_fwd(device_type='cuda')
def _DropoutAutoGrad_forward(ctx: Context, input: Tensor, drop_p: float,
    training: bool) ->Tensor:
    """
        Randomly zeroes elements in the input.

        Args:
            ctx: Context for variable storage.
            input: Input to perform dropout on.
                Can have arbitrary shape.
            drop_p: Probability of dropping an element.
            training: Flag indicating if the model is in training mode.
                If False, no dropout is applied.

        Returns:
            Input with some elements zeroed out.
        """
    ctx.do_dropout = True
    if not training or drop_p == 0:
        ctx.do_dropout = False
        return input
    ctx.drop_all = False
    if drop_p == 1:
        ctx.drop_all = True
        return torch.zeros_like(input)
    flattened_input = input.flatten()
    size = len(flattened_input)
    output = torch.empty_like(flattened_input)
    seed = randint(0, 65535)
    ctx.seed = seed
    ctx.drop_p = drop_p
    grid = lambda META: (cdiv(size, META['BLOCK_SIZE']),)
    dropout_forward_kernel[grid](flattened_input, output, size, drop_p, seed)
    return output.view_as(input)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

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


@triton.autotune(configs=element_wise_kernel_configs(), key=['size'])
@triton.jit
def dropout_backward_kernel(output_grad_pointer, input_grad_pointer, size,
    drop_p, seed, BLOCK_SIZE: tl.constexpr):
    """
    Calculates the input gradient of dropout.

    Args:
        output_grad_pointer: Pointer to dropout's output gradients.
            The output gradients must be of shape [size].
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [size].
        size: Number of elements in the input.
        drop_p: Probability of dropping an element used in dropout.
        seed: Seed for generating the dropout mask.
        BLOCK_SIZE: Block size.
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    output_grad = tl.load(output_grad_pointer + offset, mask=mask)
    input_grad = apply_dropout_grad(output_grad, drop_p, seed, offset)
    tl.store(input_grad_pointer + offset, input_grad, mask=mask)


# Backward method (kernel launch code)
@custom_bwd(device_type='cuda')
def _DropoutAutoGrad_backward(ctx: Context, output_grad: Tensor) ->Tuple[
    Optional[Tensor], ...]:
    """
        Calculates the input gradient of dropout.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of dropout.
        """
    if not ctx.do_dropout:
        return output_grad, None, None
    if ctx.drop_all:
        return torch.zeros_like(output_grad), None, None
    orig_shape = output_grad.shape
    output_grad = output_grad.flatten()
    size = len(output_grad)
    input_grad = torch.empty_like(output_grad)
    grid = lambda META: (cdiv(size, META['BLOCK_SIZE']),)
    dropout_backward_kernel[grid](output_grad, input_grad, size, ctx.drop_p,
        ctx.seed)
    return input_grad.view(orig_shape), None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class DropoutAutoGrad(torch.autograd.Function):
    """
    Autodiff for dropout.
    """

    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx: Context, input: Tensor, drop_p: float, training: bool
        ) ->Tensor:
        """
        Randomly zeroes elements in the input.

        Args:
            ctx: Context for variable storage.
            input: Input to perform dropout on.
                Can have arbitrary shape.
            drop_p: Probability of dropping an element.
            training: Flag indicating if the model is in training mode.
                If False, no dropout is applied.

        Returns:
            Input with some elements zeroed out.
        """
        ctx.do_dropout = True
        if not training or drop_p == 0:
            ctx.do_dropout = False
            return input
        ctx.drop_all = False
        if drop_p == 1:
            ctx.drop_all = True
            return torch.zeros_like(input)
        flattened_input = input.flatten()
        size = len(flattened_input)
        output = torch.empty_like(flattened_input)
        seed = randint(0, 65535)
        ctx.seed = seed
        ctx.drop_p = drop_p
        grid = lambda META: (cdiv(size, META['BLOCK_SIZE']),)
        dropout_forward_kernel[grid](flattened_input, output, size, drop_p,
            seed)
        return output.view_as(input)

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx: Context, output_grad: Tensor) ->Tuple[Optional[Tensor
        ], ...]:
        """
        Calculates the input gradient of dropout.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of dropout.
        """
        if not ctx.do_dropout:
            return output_grad, None, None
        if ctx.drop_all:
            return torch.zeros_like(output_grad), None, None
        orig_shape = output_grad.shape
        output_grad = output_grad.flatten()
        size = len(output_grad)
        input_grad = torch.empty_like(output_grad)
        grid = lambda META: (cdiv(size, META['BLOCK_SIZE']),)
        dropout_backward_kernel[grid](output_grad, input_grad, size, ctx.
            drop_p, ctx.seed)
        return input_grad.view(orig_shape), None, None
