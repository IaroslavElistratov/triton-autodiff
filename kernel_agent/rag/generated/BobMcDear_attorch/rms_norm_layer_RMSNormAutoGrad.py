# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/BobMcDear/attorch
# Source-Files: attorch/rms_norm_layer.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_leort3lx/attorch-main/attorch/rms_norm_layer.py
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

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim', 'feat_dim'])
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic,
    'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim'])})
@triton.jit
def rms_norm_forward_kernel(input_pointer, weight_pointer, inv_rms_pointer,
    output_pointer, batch_dim, feat_dim, input_batch_stride,
    input_feat_stride, output_batch_stride, output_feat_stride, eps,
    scale_by_weight: tl.constexpr, save_stats: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_FEAT: tl.constexpr):
    """
    Root-mean-square-normalizes the input.

    Args:
        input_pointer: Pointer to the input to root-mean-square-normalize.
            The input must be of shape [batch_dim, feat_dim].
        weight_pointer: Pointer to optional weights for linear transform.
            The weights, if provided, must be of shape [feat_dim].
        inv_rms_pointer: Pointer to an optional container the input's inverse
            root mean square is written to if save_stats is True.
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
        save_stats: Flag for saving the root mean square.
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
    inv_rms = tl.rsqrt(tl.sum(input * input, axis=1) / feat_dim + eps)
    output = input * inv_rms[:, None]
    if save_stats:
        tl.store(inv_rms_pointer + batch_offset, inv_rms, mask=batch_mask)
    if scale_by_weight:
        weight = tl.load(weight_pointer + feat_offset, mask=feat_mask)
        output *= weight
    tl.store(output_pointer, output, mask=batch_mask[:, None] & feat_mask[
        None, :])


# Forward method (kernel launch code)
@custom_fwd(device_type='cuda')
def _RMSNormAutoGrad_forward(ctx: Context, input: Tensor, weight: Optional[
    Tensor]=None, eps: Optional[float]=None) ->Tensor:
    """
        Root-mean-square-normalizes the input.

        Args:
            ctx: Context for variable storage.
            input: Input to root-mean-square-normalize.
                Can have arbitrary shape.
            weight: Optional weights for linear transform.
                If provided, must be of shape [feat_dim].
            eps: Epsilon added in the square root in the denominator
                to avoid division by zero. If None, it defaults to
                torch.finfo(input.dtype).eps.

        Returns:
            Root-mean-square-normalized input.
        """
    flattened_input = input.unsqueeze(0) if input.ndim == 1 else input
    flattened_input = flattened_input.flatten(0, -2)
    batch_dim, feat_dim = flattened_input.shape
    eps = torch.finfo(input.dtype).eps if eps is None else eps
    output = torch.empty_like(flattened_input)
    scale_by_weight = weight is not None
    requires_grad = (input.requires_grad or scale_by_weight and weight.
        requires_grad)
    if requires_grad:
        inv_rms = torch.empty(batch_dim, device=input.device, dtype=torch.
            float32)
    else:
        inv_rms = None
    grid = lambda META: (cdiv(batch_dim, META['BLOCK_SIZE_BATCH']),)
    rms_norm_forward_kernel[grid](flattened_input, weight, inv_rms, output,
        batch_dim, feat_dim, *flattened_input.stride(), *output.stride(),
        eps, scale_by_weight=scale_by_weight, save_stats=requires_grad)
    ctx.scale_by_weight = scale_by_weight
    if requires_grad:
        ctx.save_for_backward(flattened_input, inv_rms, weight)
    return output.view_as(input)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim', 'feat_dim'])
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic,
    'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim'])})
@triton.jit
def rms_norm_backward_kernel(output_grad_pointer, input_pointer,
    inv_rms_pointer, weight_pointer, input_grad_pointer,
    weight_grad_pointer, batch_dim, feat_dim, output_grad_batch_stride,
    output_grad_feat_stride, input_batch_stride, input_feat_stride,
    input_grad_batch_stride, input_grad_feat_stride,
    weight_grad_batch_stride, weight_grad_feat_stride, scale_by_weight: tl.
    constexpr, BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_FEAT: tl.constexpr):
    """
    Calculates the input gradient of root mean square normalization.

    Args:
        output_grad_pointer: Pointer to root mean square normalization's output gradients.
            The output gradients must be of shape [batch_dim, feat_dim].
        input_pointer: Pointer to the input.
            The input must be of shape [batch_dim, feat_dim].
        inv_rms_pointer: Pointer to the input's inverse root mean square.
            The inverse root mean square should be of shape [batch_dim].
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
        scale_by_weight: Flag for scaling the normalized output by weights.
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
    inv_rms = tl.load(inv_rms_pointer + batch_offset, mask=batch_mask)
    pre_lin = input * inv_rms[:, None]
    if scale_by_weight:
        weight = tl.load(weight_pointer + feat_offset, mask=feat_mask)
        weight_output_grad_prod = weight * output_grad
    else:
        weight_output_grad_prod = output_grad
    term1 = input * tl.sum(input * weight_output_grad_prod, axis=1)
    term2 = inv_rms[:, None] * inv_rms[:, None]
    input_grad = inv_rms[:, None] * (weight_output_grad_prod - term1 *
        term2 / feat_dim)
    tl.store(input_grad_pointer, input_grad, mask=batch_mask[:, None] &
        feat_mask[None, :])
    if scale_by_weight:
        weight_grad_pointer += (weight_grad_batch_stride * batch_pid + 
            weight_grad_feat_stride * feat_offset)
        tl.store(weight_grad_pointer, tl.sum(output_grad * pre_lin, axis=0),
            mask=feat_mask)


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
@custom_bwd(device_type='cuda')
def _RMSNormAutoGrad_backward(ctx: Context, output_grad: Tensor) ->Tuple[
    Optional[Tensor], ...]:
    """
        Calculates the input gradient of root mean square normalization.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of root mean square normalization.
        """
    scale_by_weight = ctx.scale_by_weight
    flattened_input, inv_rms, weight = ctx.saved_tensors
    flattened_output_grad = output_grad.view_as(flattened_input)
    batch_dim, feat_dim = flattened_output_grad.shape
    input_grad = torch.empty_like(flattened_output_grad)
    if scale_by_weight:
        BLOCK_SIZE_BATCH = BLOCK_SIZE_BATCH_heuristic({'batch_dim':
            batch_dim, 'feat_dim': feat_dim})
        out_batch_dim = batch_dim // BLOCK_SIZE_BATCH
        weight_grad = torch.empty((out_batch_dim, feat_dim), device=
            flattened_input.device)
    else:
        weight_grad = None
    grid = lambda META: (cdiv(batch_dim, META['BLOCK_SIZE_BATCH']),)
    rms_norm_backward_kernel[grid](flattened_output_grad, flattened_input,
        inv_rms, weight, input_grad, weight_grad, batch_dim, feat_dim, *
        flattened_output_grad.stride(), *flattened_input.stride(), *
        input_grad.stride(), *(weight_grad.stride() if scale_by_weight else
        (1, 1)), scale_by_weight=scale_by_weight)
    if scale_by_weight:
        weight_grad = weight_grad.sum(dim=0)
    return input_grad.view_as(output_grad), weight_grad, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class RMSNormAutoGrad(torch.autograd.Function):
    """
    Autodiff for root mean square normalization.
    """

    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx: Context, input: Tensor, weight: Optional[Tensor]=None,
        eps: Optional[float]=None) ->Tensor:
        """
        Root-mean-square-normalizes the input.

        Args:
            ctx: Context for variable storage.
            input: Input to root-mean-square-normalize.
                Can have arbitrary shape.
            weight: Optional weights for linear transform.
                If provided, must be of shape [feat_dim].
            eps: Epsilon added in the square root in the denominator
                to avoid division by zero. If None, it defaults to
                torch.finfo(input.dtype).eps.

        Returns:
            Root-mean-square-normalized input.
        """
        flattened_input = input.unsqueeze(0) if input.ndim == 1 else input
        flattened_input = flattened_input.flatten(0, -2)
        batch_dim, feat_dim = flattened_input.shape
        eps = torch.finfo(input.dtype).eps if eps is None else eps
        output = torch.empty_like(flattened_input)
        scale_by_weight = weight is not None
        requires_grad = (input.requires_grad or scale_by_weight and weight.
            requires_grad)
        if requires_grad:
            inv_rms = torch.empty(batch_dim, device=input.device, dtype=
                torch.float32)
        else:
            inv_rms = None
        grid = lambda META: (cdiv(batch_dim, META['BLOCK_SIZE_BATCH']),)
        rms_norm_forward_kernel[grid](flattened_input, weight, inv_rms,
            output, batch_dim, feat_dim, *flattened_input.stride(), *output
            .stride(), eps, scale_by_weight=scale_by_weight, save_stats=
            requires_grad)
        ctx.scale_by_weight = scale_by_weight
        if requires_grad:
            ctx.save_for_backward(flattened_input, inv_rms, weight)
        return output.view_as(input)

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx: Context, output_grad: Tensor) ->Tuple[Optional[Tensor
        ], ...]:
        """
        Calculates the input gradient of root mean square normalization.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of root mean square normalization.
        """
        scale_by_weight = ctx.scale_by_weight
        flattened_input, inv_rms, weight = ctx.saved_tensors
        flattened_output_grad = output_grad.view_as(flattened_input)
        batch_dim, feat_dim = flattened_output_grad.shape
        input_grad = torch.empty_like(flattened_output_grad)
        if scale_by_weight:
            BLOCK_SIZE_BATCH = BLOCK_SIZE_BATCH_heuristic({'batch_dim':
                batch_dim, 'feat_dim': feat_dim})
            out_batch_dim = batch_dim // BLOCK_SIZE_BATCH
            weight_grad = torch.empty((out_batch_dim, feat_dim), device=
                flattened_input.device)
        else:
            weight_grad = None
        grid = lambda META: (cdiv(batch_dim, META['BLOCK_SIZE_BATCH']),)
        rms_norm_backward_kernel[grid](flattened_output_grad,
            flattened_input, inv_rms, weight, input_grad, weight_grad,
            batch_dim, feat_dim, *flattened_output_grad.stride(), *
            flattened_input.stride(), *input_grad.stride(), *(weight_grad.
            stride() if scale_by_weight else (1, 1)), scale_by_weight=
            scale_by_weight)
        if scale_by_weight:
            weight_grad = weight_grad.sum(dim=0)
        return input_grad.view_as(output_grad), weight_grad, None
