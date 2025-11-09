# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/BobMcDear/attorch
# Source-Files: attorch/conv_layer.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_leort3lx/attorch-main/attorch/conv_layer.py
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

@triton.autotune(configs=[conv2d_forward_config(128, 32, 128, n_warps=8,
    n_stages=2), conv2d_forward_config(256, 32, 64, n_warps=8, n_stages=2),
    conv2d_forward_config(256, 32, 32, n_warps=4, n_stages=4),
    conv2d_forward_config(256, 64, 32, n_warps=4, n_stages=4),
    conv2d_forward_config(256, 32, 16, n_warps=2, n_stages=4),
    conv2d_forward_config(64, 32, 128, n_warps=8, n_stages=4),
    conv2d_forward_config(128, 32, 64, n_warps=4, n_stages=4),
    conv2d_forward_config(64, 32, 64, n_warps=4, n_stages=4),
    conv2d_forward_config(128, 32, 16, n_warps=4, n_stages=4),
    conv2d_forward_config(128, 128, 128, n_warps=8, n_stages=3),
    conv2d_forward_config(256, 128, 64, n_warps=8, n_stages=3),
    conv2d_forward_config(256, 128, 32, n_warps=4, n_stages=4),
    conv2d_forward_config(64, 128, 128, n_warps=4, n_stages=4),
    conv2d_forward_config(128, 128, 64, n_warps=4, n_stages=4),
    conv2d_forward_config(128, 64, 32, n_warps=2, n_stages=4),
    conv2d_forward_config(64, 64, 64, n_warps=2, n_stages=4)], key=[
    'batch_dim', 'in_feat_dim', 'in_height', 'in_width', 'out_feat_dim',
    'out_height', 'out_width', 'kernel_height', 'kernel_width',
    'stride_height', 'stride_width', 'padding_height', 'padding_width',
    'groups', 'fp16'])
@triton.heuristics({'tf32': lambda _: allow_tf32()})
@triton.jit
def conv2d_forward_kernel(input_pointer, weight_pointer, output_pointer,
    batch_dim, in_feat_dim, in_height, in_width, out_feat_dim, out_height,
    out_width, input_batch_stride, input_in_feat_stride,
    input_height_stride, input_width_stride, weight_out_feat_stride,
    weight_in_feat_stride, weight_height_stride, weight_width_stride,
    output_batch_stride, output_out_feat_stride, output_height_stride,
    output_width_stride, kernel_height: tl.constexpr, kernel_width: tl.
    constexpr, stride_height: tl.constexpr, stride_width: tl.constexpr,
    padding_height: tl.constexpr, padding_width: tl.constexpr, groups: tl.
    constexpr, fp16: tl.constexpr, tf32: tl.constexpr,
    BLOCK_SIZE_BATCH_HEIGHT_WIDTH: tl.constexpr, BLOCK_SIZE_IN_FEAT: tl.
    constexpr, BLOCK_SIZE_OUT_FEAT: tl.constexpr):
    """
    2D-convolves over the input using weights.

    Args:
        input_pointer: Pointer to the input to convolve over.
            The input must be of shape [batch_dim, in_feat_dim, in_height, in_width].
        weight_pointer: Pointer to the weights input is convolved over by.
            The weights must be of shape [out_feat_dim, in_feat_dim, kernel_height, kernel_width].
        output_pointer: Pointer to a container the result is written to.
            The container must be of shape [batch_dim, out_feat_dim, out_height, out_width].
        batch_dim: Batch dimension of the input and output.
        in_feat_dim: Dimensionality of the input features.
        in_height: Input height.
        in_width: Input width.
        out_feat_dim: Dimensionality of the output features.
        out_height: Output height.
        out_width: Output width.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_in_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        input_height_stride: Stride necessary to jump one element along the
            input's height dimension.
        input_width_stride: Stride necessary to jump one element along the
            input's width dimension.
        weight_out_feat_stride: Stride necessary to jump one element along the
            weights' output feature dimension.
        weight_in_feat_stride: Stride necessary to jump one element along the
            weights' input feature dimension.
        weight_height_stride: Stride necessary to jump one element along the
            weights' height dimension.
        weight_width_stride: Stride necessary to jump one element along the
            weights' width dimension.
        output_batch_stride: Stride necessary to jump one element along the
            output's batch dimension.
        output_out_feat_stride: Stride necessary to jump one element along the
            output's feature dimension.
        output_height_stride: Stride necessary to jump one element along the
            output's height dimension.
        output_width_stride: Stride necessary to jump one element along the
            output's width dimension.
        kernel_height: Kernel height.
        kernel_width: Kernel width.
        stride_height: Stride of kernel across the height dimension.
        stride_width: Stride of kernel across the width dimension.
        padding_height: Padding applied to the input across the height dimension.
        padding_width: Padding applied to the input across the width dimension.
        groups: Number of groups for the convolution.
        fp16: Flag for loading the input and weights in FP16.
        tf32: Flag for performing matrix products in TF32.
        BLOCK_SIZE_BATCH_HEIGHT_WIDTH: Block size across the batch, height, and
            width dimensions.
        BLOCK_SIZE_IN_FEAT: Block size across the input feature dimension.
        BLOCK_SIZE_OUT_FEAT: Block size across the output feature dimension.
    """
    batch_height_width_pid = tl.program_id(0)
    out_feat_pid = tl.program_id(1)
    group_pid = tl.program_id(2)
    in_group_dim = in_feat_dim // groups
    out_group_dim = out_feat_dim // groups
    batch_height_width_offset = (batch_height_width_pid *
        BLOCK_SIZE_BATCH_HEIGHT_WIDTH + tl.arange(0,
        BLOCK_SIZE_BATCH_HEIGHT_WIDTH))
    batch_height_offset = batch_height_width_offset // out_width
    batch_offset = batch_height_offset // out_height
    output_feat_offset = out_feat_pid * BLOCK_SIZE_OUT_FEAT + tl.arange(0,
        BLOCK_SIZE_OUT_FEAT)
    output_height_offset = batch_height_offset % out_height
    output_width_offset = batch_height_width_offset % out_width
    input_pointer += (input_batch_stride * batch_offset + 
        input_in_feat_stride * group_pid * in_group_dim)[:, None]
    weight_pointer += (weight_out_feat_stride * output_feat_offset + 
        weight_out_feat_stride * group_pid * out_group_dim)[None, :]
    accum = tl.zeros((BLOCK_SIZE_BATCH_HEIGHT_WIDTH, BLOCK_SIZE_OUT_FEAT),
        dtype=tl.float32)
    for h in range(kernel_height):
        for w in range(kernel_width):
            for c in range(0, in_group_dim, BLOCK_SIZE_IN_FEAT):
                input_feat_offset = c + tl.arange(0, BLOCK_SIZE_IN_FEAT)
                input_height_offset = (h - padding_height + stride_height *
                    output_height_offset)
                input_width_offset = (w - padding_width + stride_width *
                    output_width_offset)
                curr_input_pointer = input_pointer + (input_in_feat_stride *
                    input_feat_offset)[None, :] + (input_height_stride *
                    input_height_offset)[:, None] + (input_width_stride *
                    input_width_offset)[:, None]
                curr_weight_pointer = weight_pointer + (weight_in_feat_stride *
                    input_feat_offset)[:, None
                    ] + weight_height_stride * h + weight_width_stride * w
                input_mask = (batch_offset < batch_dim)[:, None] & (
                    input_feat_offset < in_group_dim)[None, :] & (0 <=
                    input_height_offset)[:, None] & (input_height_offset <
                    in_height)[:, None] & (0 <= input_width_offset)[:, None
                    ] & (input_width_offset < in_width)[:, None]
                weight_mask = (input_feat_offset < in_group_dim)[:, None] & (
                    output_feat_offset < out_group_dim)[None, :]
                input_block = tl.load(curr_input_pointer, mask=input_mask)
                weight_block = tl.load(curr_weight_pointer, mask=weight_mask)
                if fp16:
                    input_block = input_block.to(tl.float16)
                    weight_block = weight_block.to(tl.float16)
                accum += tl.dot(input_block, weight_block, allow_tf32=tf32)
    output_pointer += (output_batch_stride * batch_offset)[:, None] + (
        output_out_feat_stride * (group_pid * out_group_dim +
        output_feat_offset))[None, :] + (output_height_stride *
        output_height_offset)[:, None] + (output_width_stride *
        output_width_offset)[:, None]
    output_mask = (batch_offset < batch_dim)[:, None] & (output_feat_offset <
        out_group_dim)[None, :] & (output_height_offset < out_height)[:, None
        ] & (output_width_offset < out_width)[:, None]
    tl.store(output_pointer, accum, mask=output_mask)


def conv2d_output_size(in_size: int, kernel_size: int, stride: int, padding:
    int) ->int:
    """
    Determines the output size of a 2D convolution operation.

    Args:
        in_size: Input size.
        kernel_size: Kernel size.
        stride: Stride.
        padding: Padding.

    Returns:
        Output size of 2D convolution.
    """
    return (in_size + 2 * padding - kernel_size) // stride + 1


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
def _Conv2dAutoGrad_forward(ctx: Context, input: Tensor, weight: Tensor,
    bias: Optional[Tensor]=None, stride_height: int=1, stride_width: int=1,
    padding_height: int=1, padding_width: int=1, groups: int=1) ->Tensor:
    """
        2D-convolves over the input using weights, optionally adding bias.

        Args:
            input: Input to convolve over.
                Must be of shape [batch_dim, in_feat_dim, in_height, in_width].
            weight: Weights input is convolved over by.
                Must be of shape [out_feat_dim, in_feat_dim, kernel_height, kernel_width].
            bias: Optional additive bias vector, with None for no bias.
                If provided, must be of shape [out_feat_dim].
            stride_height: Stride of kernel across the height dimension.
            stride_width: Stride of kernel across the width dimension.
            padding_height: Padding applied to the input across the height dimension.
            padding_width: Padding applied to the input across the width dimension.
            groups: Number of groups for the convolution.

        Returns:
            Input 2D-convolved over, potentially with added biased.
        """
    assert weight.ndim == 4, f'Weights must be 4D, received shape {weight.shape}'
    assert bias is None or bias.ndim == 1, f'Bias must be 1D, received shape {bias.shape}'
    assert input.shape[1] == groups * weight.shape[1
        ], f'Incompatible input ({input.shape}) and weights ({weight.shape}) shape with {groups} groups'
    assert bias is None or weight.shape[0] == bias.shape[0
        ], f'Incompatible weights ({weight.shape}) and bias ({bias.shape}) shape'
    batch_dim, in_feat_dim, in_height, in_width = input.shape
    out_feat_dim, _, kernel_height, kernel_width = weight.shape
    out_height = conv2d_output_size(in_height, kernel_height, stride_height,
        padding_height)
    out_width = conv2d_output_size(in_width, kernel_width, stride_width,
        padding_width)
    output_dtype = get_output_dtype(input.dtype, autocast='fp16')
    output = torch.empty((batch_dim, out_feat_dim, out_height, out_width),
        device=input.device, dtype=output_dtype)
    grid = lambda META: (cdiv(batch_dim * out_height * out_width, META[
        'BLOCK_SIZE_BATCH_HEIGHT_WIDTH']), cdiv(out_feat_dim, META[
        'BLOCK_SIZE_OUT_FEAT']), groups)
    conv2d_forward_kernel[grid](input, weight, output, batch_dim,
        in_feat_dim, in_height, in_width, out_feat_dim, out_height,
        out_width, *input.stride(), *weight.stride(), *output.stride(),
        kernel_height, kernel_width, stride_height, stride_width,
        padding_height, padding_width, groups=groups, fp16=output_dtype is
        torch.float16)
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    requires_grad = (input.requires_grad or weight.requires_grad or bias is not
        None and bias.requires_grad)
    ctx.stride = stride_height, stride_width
    ctx.padding = padding_height, padding_width
    ctx.groups = groups
    ctx.bias_requires_grad = False if bias is None else bias.requires_grad
    ctx.output_dtype = output_dtype
    if requires_grad:
        ctx.save_for_backward(input, weight)
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

# Backward method (kernel launch code)
def _Conv2dAutoGrad_backward(ctx: Context, output_grad: Tensor) ->Tuple[
    Optional[Tensor], ...]:
    """
        Calculates the input gradient of the 2D convolutional layer.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of the 2D convolutional layer.
        """
    input, weight = ctx.saved_tensors
    input = input.to(ctx.output_dtype)
    weight = weight.to(ctx.output_dtype)
    input_grad = nn.grad.conv2d_input(input.shape, weight, output_grad, ctx
        .stride, ctx.padding, groups=ctx.groups)
    weight_grad = nn.grad.conv2d_weight(input, weight.shape, output_grad,
        ctx.stride, ctx.padding, groups=ctx.groups)
    bias_grad = output_grad.sum(dim=(0, 2, 3)).to(ctx.output_dtype
        ) if ctx.bias_requires_grad else None
    return input_grad, weight_grad, bias_grad, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class Conv2dAutoGrad(torch.autograd.Function):
    """
    Autodiff for 2D convolutional layer.
    """

    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor, bias: Optional
        [Tensor]=None, stride_height: int=1, stride_width: int=1,
        padding_height: int=1, padding_width: int=1, groups: int=1) ->Tensor:
        """
        2D-convolves over the input using weights, optionally adding bias.

        Args:
            input: Input to convolve over.
                Must be of shape [batch_dim, in_feat_dim, in_height, in_width].
            weight: Weights input is convolved over by.
                Must be of shape [out_feat_dim, in_feat_dim, kernel_height, kernel_width].
            bias: Optional additive bias vector, with None for no bias.
                If provided, must be of shape [out_feat_dim].
            stride_height: Stride of kernel across the height dimension.
            stride_width: Stride of kernel across the width dimension.
            padding_height: Padding applied to the input across the height dimension.
            padding_width: Padding applied to the input across the width dimension.
            groups: Number of groups for the convolution.

        Returns:
            Input 2D-convolved over, potentially with added biased.
        """
        assert weight.ndim == 4, f'Weights must be 4D, received shape {weight.shape}'
        assert bias is None or bias.ndim == 1, f'Bias must be 1D, received shape {bias.shape}'
        assert input.shape[1] == groups * weight.shape[1
            ], f'Incompatible input ({input.shape}) and weights ({weight.shape}) shape with {groups} groups'
        assert bias is None or weight.shape[0] == bias.shape[0
            ], f'Incompatible weights ({weight.shape}) and bias ({bias.shape}) shape'
        batch_dim, in_feat_dim, in_height, in_width = input.shape
        out_feat_dim, _, kernel_height, kernel_width = weight.shape
        out_height = conv2d_output_size(in_height, kernel_height,
            stride_height, padding_height)
        out_width = conv2d_output_size(in_width, kernel_width, stride_width,
            padding_width)
        output_dtype = get_output_dtype(input.dtype, autocast='fp16')
        output = torch.empty((batch_dim, out_feat_dim, out_height,
            out_width), device=input.device, dtype=output_dtype)
        grid = lambda META: (cdiv(batch_dim * out_height * out_width, META[
            'BLOCK_SIZE_BATCH_HEIGHT_WIDTH']), cdiv(out_feat_dim, META[
            'BLOCK_SIZE_OUT_FEAT']), groups)
        conv2d_forward_kernel[grid](input, weight, output, batch_dim,
            in_feat_dim, in_height, in_width, out_feat_dim, out_height,
            out_width, *input.stride(), *weight.stride(), *output.stride(),
            kernel_height, kernel_width, stride_height, stride_width,
            padding_height, padding_width, groups=groups, fp16=output_dtype is
            torch.float16)
        if bias is not None:
            output += bias.view(1, -1, 1, 1)
        requires_grad = (input.requires_grad or weight.requires_grad or 
            bias is not None and bias.requires_grad)
        ctx.stride = stride_height, stride_width
        ctx.padding = padding_height, padding_width
        ctx.groups = groups
        ctx.bias_requires_grad = False if bias is None else bias.requires_grad
        ctx.output_dtype = output_dtype
        if requires_grad:
            ctx.save_for_backward(input, weight)
        return output

    @staticmethod
    def backward(ctx: Context, output_grad: Tensor) ->Tuple[Optional[Tensor
        ], ...]:
        """
        Calculates the input gradient of the 2D convolutional layer.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of the 2D convolutional layer.
        """
        input, weight = ctx.saved_tensors
        input = input.to(ctx.output_dtype)
        weight = weight.to(ctx.output_dtype)
        input_grad = nn.grad.conv2d_input(input.shape, weight, output_grad,
            ctx.stride, ctx.padding, groups=ctx.groups)
        weight_grad = nn.grad.conv2d_weight(input, weight.shape,
            output_grad, ctx.stride, ctx.padding, groups=ctx.groups)
        bias_grad = output_grad.sum(dim=(0, 2, 3)).to(ctx.output_dtype
            ) if ctx.bias_requires_grad else None
        return input_grad, weight_grad, bias_grad, None, None, None, None, None
