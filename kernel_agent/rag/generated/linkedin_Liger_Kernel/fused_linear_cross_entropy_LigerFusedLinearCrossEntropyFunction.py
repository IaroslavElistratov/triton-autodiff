# SPDX-License-Identifier: BSD-2-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/linkedin/Liger-Kernel
# Source-Files: src/liger_kernel/ops/fused_linear_cross_entropy.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_d1vk6zhm/Liger-Kernel-main/src/liger_kernel/ops/fused_linear_cross_entropy.py
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
from torch.cuda.amp import custom_bwd
from torch.cuda.amp import custom_fwd
from math import log

def is_hip() ->bool:
    return torch.version.hip is not None


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def liger_cross_entropy_kernel(X_ptr, X_stride, Y_ptr, Y_stride, weight_ptr,
    loss_ptr, z_loss_ptr, loss_stride, n_cols, n_non_ignore,
    sum_non_ignore_weight, weight_sum, ignore_index, lse_square_scale: tl.
    constexpr, label_smoothing: tl.constexpr, reduction: tl.constexpr,
    softcap, RETURN_Z_LOSS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr, HAS_SOFTCAPPING: tl.constexpr, HAS_GRADIENTS:
    tl.constexpr):
    """
    This kernel computes both cross entropy loss and the gradient of the input.
    We only consider hard label + mean reduction for now. Please refer to https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for the math.

    Parameters:
    X_ptr: Pointer to input tensor.
    X_stride (int): The stride of the input tensor.
    Y_ptr: Pointer to target tensor.
    Y_stride (int): The stride of the target tensor.
    weight_ptr: Pointer to weight tensor.
    loss_ptr: Pointer to tensor to store the loss.
    z_loss_ptr: Pointer to tensor to store the z loss. No operation if RETURN_Z_LOSS is 0.
    loss_stride (int): The stride of the loss tensor.
    n_cols (int): The number of columns in the input tensor.
    n_non_ignore (float): The number of non-ignored elements in the batch.
    sum_non_ignore_weight (float): The sum of non-ignored target's weights in the batch.
    weight_sum (float): The sum of weight tensor.
    ignore_index (int): The index to ignore in the target.
    label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
    lse_square_scale (float): The scaler of (logsumexp(_input)) ^ 2 adding to the loss for the stability of training.
    reduction (str): The string for the reduction to apply
    softcap (float): The upper threshold for scaling logits to the range (-softcap, +softcap).
    RETURN_Z_LOSS (int): The boolean value to decide whether storing z loss to z_loss_ptr or not. It must be 0 or 1.
    BLOCK_SIZE (int): The block size for Triton operations.
    HAS_WEIGHT (bool): The boolean value to determine whether assigning weight to each of the classes.
    HAS_SOFTCAPPING (bool): The boolean value to determine whether applying soft-capping or not.
    HAS_GRADIENTS (bool): The boolean value to determine whether calculating gradients in forward pass.
    """
    program_id = tl.program_id(0).to(tl.int64)
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)
    X_ptr += program_id * X_stride
    if y == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return
    loss_ptr += program_id * loss_stride
    if RETURN_Z_LOSS:
        z_loss_ptr += program_id * loss_stride
    if HAS_WEIGHT:
        weight_y = tl.load(weight_ptr + y).cast(tl.float32)
    m = float('-inf')
    d = 0.0
    ori_X_y = tl.load(X_ptr + y).cast(tl.float32)
    if HAS_SOFTCAPPING:
        ori_X_y = softcap * tanh(ori_X_y / softcap)
    scaled_x_sum = 0.0
    eps = label_smoothing / n_cols
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other
            =float('-inf')).cast(tl.float32)
        if HAS_SOFTCAPPING:
            X_block = softcap * tanh(X_block / softcap)
        block_max = tl.max(X_block)
        if label_smoothing > 0:
            if HAS_WEIGHT:
                weight_block = tl.load(weight_ptr + X_offsets, mask=
                    X_offsets < n_cols)
                scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps *
                    X_block * weight_block, 0.0))
            else:
                scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps *
                    X_block, 0.0))
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new
    lse = m + tl.log(d)
    if HAS_GRADIENTS:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols,
                other=float('-inf')).cast(tl.float32)
            if HAS_SOFTCAPPING:
                intermediate = tanh(X_block / softcap)
                X_block = softcap * intermediate
            if not HAS_WEIGHT:
                X_block = tl.exp(X_block - m) / d
                X_block += 2 * lse_square_scale * lse * X_block
                X_block += -eps
                X_block = tl.where(X_offsets != y, X_block, X_block - (1 -
                    label_smoothing))
                if reduction == 'mean':
                    X_block = X_block / n_non_ignore
            else:
                weight_block = tl.load(weight_ptr + X_offsets, mask=
                    X_offsets < n_cols)
                softmax_X = tl.exp(X_block - m) / d
                dloss_ori = (1 - label_smoothing) * softmax_X
                dloss_ori = tl.where(X_offsets != y, dloss_ori, dloss_ori -
                    (1 - label_smoothing))
                dloss_ori = dloss_ori * weight_y
                dloss_smooth = eps * (-weight_block + softmax_X * weight_sum)
                dz_loss = 2 * lse_square_scale * lse * softmax_X
                if reduction == 'mean':
                    dloss_ori = dloss_ori / sum_non_ignore_weight
                    dloss_smooth = dloss_smooth / sum_non_ignore_weight
                    dz_loss = dz_loss / n_non_ignore
                X_block = dloss_ori + dloss_smooth + dz_loss
            if HAS_SOFTCAPPING:
                X_block = X_block * (1 - intermediate * intermediate)
            tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)
    tl.debug_barrier()
    loss = lse - ori_X_y
    if HAS_WEIGHT:
        loss = weight_y * loss
    if label_smoothing > 0:
        if HAS_WEIGHT:
            smooth_loss = scaled_x_sum + eps * lse * weight_sum
        else:
            smooth_loss = scaled_x_sum + label_smoothing * lse
        loss = loss * (1 - label_smoothing) + smooth_loss
    z_loss = lse_square_scale * lse * lse
    if reduction == 'mean':
        if HAS_WEIGHT:
            loss = loss / sum_non_ignore_weight
        else:
            loss = loss / n_non_ignore
        z_loss = z_loss / n_non_ignore
    loss += z_loss
    tl.store(loss_ptr, loss)
    if RETURN_Z_LOSS:
        tl.store(z_loss_ptr, z_loss)


def fused_linear_cross_entropy_forward(_input, weight, target, ce_weight=
    None, bias=None, ignore_index=-100, lse_square_scale=0.0,
    label_smoothing=0.0, reduction='mean', softcap=None, return_z_loss=
    False, accum_dtype=None, use_token_scaling=False):
    assert isinstance(return_z_loss, bool
        ), f'return_z_loss must be True or False. Got: {return_z_loss}'
    device = _input.device
    input_requires_grad = _input.requires_grad
    BT, H = _input.shape
    V = weight.shape[0]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    inc_factor = triton.cdiv(V, H)
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))
    num_chunks = triton.cdiv(BT, chunk_size)
    grad_input = torch.zeros_like(_input, device=device)
    if input_requires_grad:
        if accum_dtype is None:
            grad_weight = torch.zeros_like(weight, device=device
                ) if weight.requires_grad else None
            grad_bias = torch.zeros_like(bias, device=device
                ) if bias is not None else None
        else:
            grad_weight = torch.zeros_like(weight, dtype=accum_dtype,
                device=device) if weight.requires_grad else None
            grad_bias = torch.zeros_like(bias, dtype=accum_dtype, device=device
                ) if bias is not None else None
    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)
    z_loss_1d = torch.zeros(BT, dtype=_input.dtype, device=_input.device
        ) if return_z_loss else None
    target_mask = target != ignore_index
    total_n_non_ignore = target_mask.sum().item()
    total_sum_non_ignore_ce_weight = total_n_non_ignore
    ce_weight_sum = 0.0
    if ce_weight is not None:
        assert ce_weight.shape[0
            ] == V, f'If given, weight has to be a Tensor of size V. Got: {ce_weight.shape}'
        assert torch.is_floating_point(ce_weight
            ), f'If given, weight has to be a Tensor of floating point dtype. Got: {ce_weight.dtype}'
        total_sum_non_ignore_ce_weight = torch.gather(ce_weight, dim=0,
            index=target.masked_select(target_mask)).sum().item()
        ce_weight_sum = ce_weight.sum().item()
        if ce_weight.stride(-1) != 1:
            ce_weight = ce_weight.contiguous()
    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _input_chunk = _input[start_idx:end_idx]
        logits_chunk = _input_chunk @ weight.t()
        if bias is not None:
            logits_chunk = logits_chunk + bias
        target_chunk = target[start_idx:end_idx]
        n_rows = logits_chunk.shape[0]
        if use_token_scaling:
            logits_for_softmax = logits_chunk.detach().clone()
            if softcap is not None:
                logits_for_softmax = softcap * torch.tanh(
                    logits_for_softmax / softcap)
            probs = torch.softmax(logits_for_softmax, dim=-1)
            valid_target_mask = target_chunk != ignore_index
            valid_targets = target_chunk[valid_target_mask]
            if len(valid_targets) > 0:
                valid_probs = probs[valid_target_mask]
                pred_probs_valid = torch.gather(valid_probs, -1,
                    valid_targets.unsqueeze(-1)).squeeze(-1)
                pred_probs = torch.zeros_like(target_chunk, dtype=probs.
                    dtype, device=probs.device)
                pred_probs[valid_target_mask] = pred_probs_valid
            else:
                pred_probs = torch.zeros_like(target_chunk, dtype=probs.
                    dtype, device=probs.device)
            scaling_factors = pred_probs.detach()
        loss_1d_slice = loss_1d[start_idx:end_idx]
        z_loss_1d_slice = z_loss_1d[start_idx:end_idx
            ] if return_z_loss else None
        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()
        liger_cross_entropy_kernel[n_rows,](X_ptr=logits_chunk, X_stride=
            logits_chunk.stride(-2), Y_ptr=target_chunk, Y_stride=
            target_chunk.stride(-1), weight_ptr=ce_weight, loss_ptr=
            loss_1d_slice, z_loss_ptr=z_loss_1d_slice, loss_stride=
            loss_1d_slice.stride(-1), n_cols=V, n_non_ignore=
            total_n_non_ignore, sum_non_ignore_weight=
            total_sum_non_ignore_ce_weight, weight_sum=ce_weight_sum,
            ignore_index=ignore_index, lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing, reduction=reduction, softcap=
            softcap, RETURN_Z_LOSS=return_z_loss, HAS_WEIGHT=True if 
            ce_weight is not None else False, HAS_SOFTCAPPING=True if 
            softcap is not None else False, HAS_GRADIENTS=
            input_requires_grad, BLOCK_SIZE=BLOCK_SIZE, num_warps=32 if not
            is_hip() else 16)
        if use_token_scaling:
            loss_1d_slice = loss_1d_slice * scaling_factors
            if return_z_loss:
                z_loss_1d_slice = z_loss_1d_slice * scaling_factors
        loss_1d[start_idx:end_idx] = loss_1d_slice
        if return_z_loss:
            z_loss_1d[start_idx:end_idx] = z_loss_1d_slice
        grad_logits_chunk = logits_chunk
        if use_token_scaling:
            scaling_factors_expanded = scaling_factors.unsqueeze(-1)
            grad_logits_chunk = grad_logits_chunk * scaling_factors_expanded
        if input_requires_grad:
            grad_input[start_idx:end_idx] = grad_logits_chunk @ weight
        if grad_weight is not None and input_requires_grad:
            grad_weight += torch.mm(grad_logits_chunk.t(), _input_chunk).float(
                )
        if bias is not None and input_requires_grad:
            torch.add(input=grad_bias, other=grad_logits_chunk.sum(dim=0),
                out=grad_bias, alpha=1.0)
    if reduction == 'none':
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None
    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None
    grad_weight = grad_weight.to(weight.dtype
        ) if grad_weight is not None else None
    grad_bias = grad_bias.to(bias.dtype) if grad_bias is not None else None
    return loss, z_loss, grad_input, grad_weight, grad_bias


# Forward method (kernel launch code)
@amp_custom_fwd
def _LigerFusedLinearCrossEntropyFunction_forward(ctx, _input, weight,
    target, bias=None, ce_weight=None, ignore_index=-100, lse_square_scale=
    0.0, label_smoothing=0.0, reduction='mean', softcap=None, return_z_loss:
    bool=False, accum_dtype=None, use_token_scaling: bool=False):
    """
        Fusing the last linear layer with cross-entropy loss
            Reference: https://github.com/mgmalek/efficient_cross_entropy

        Handle the forward and backward pass of the final linear layer via cross-entropy loss by avoiding
        the materialization of the large logits tensor. Since Cross Entropy Loss is the last layer, we can
        compute the gradient at the forward pass. By doing so, we don't have to store the _input and target
        for the backward pass.

        _input: (B*T, H) where B is batch size, T is sequence length, H is hidden dimension.
        target: (B*T) where each value is in [0, V-1]
        weight: (V, H) where V is the number of classes
        bias: (V) where V is the number of classes
        ce_weight: a manual rescaling weight given to each class. If given, has to be a Tensor of size V and floating point dtype
        ignore_index: the index to ignore in the target
        label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
        reduction: reduction to apply
        accum_dtype (torch.dtype): the dtype of intermediate result buffers for weight and bias gradient accumulations.
            Recommended to set `accum_dtype` to higher precision, e.g. `torch.float32`, if the training is unstable with original dtype. Default: `None`, performing accumulations in original dtype
        use_token_scaling (bool): whether to scale each token's loss by its predicted probability (detached).
            When True, each token's loss is multiplied by the model's predicted probability for that token's true class.
            Default: False.
        """
    loss, z_loss, grad_input, grad_weight, grad_bias = (
        fused_linear_cross_entropy_forward(_input=_input, weight=weight,
        target=target, bias=bias, ce_weight=ce_weight, ignore_index=
        ignore_index, lse_square_scale=lse_square_scale, label_smoothing=
        label_smoothing, reduction=reduction, softcap=softcap,
        return_z_loss=return_z_loss, accum_dtype=accum_dtype,
        use_token_scaling=use_token_scaling))
    ctx.save_for_backward(grad_input.detach(), grad_weight.detach() if 
        grad_weight is not None else None, grad_bias.detach() if bias is not
        None else None)
    ctx.return_z_loss = return_z_loss
    return loss, z_loss


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def element_mul_kernel(X_ptr, X_stride, grad_output_ptr, n_cols, BLOCK_SIZE:
    tl.constexpr):
    """
    This function multiplies each element of the tensor pointed by X_ptr with the value pointed by grad_output_ptr.
    The multiplication is performed in-place on the tensor pointed by X_ptr.

    Parameters:
    X_ptr: Pointer to the input tensor.
    X_stride (int): The stride of the input tensor.
    grad_output_ptr: Pointer to the gradient output value.
    n_cols (int): The number of columns in the input tensor.
    BLOCK_SIZE (int): The block size for Triton operations.
    """
    program_id = tl.program_id(0).to(tl.int64)
    X_ptr += program_id * X_stride
    grad_output = tl.load(grad_output_ptr)
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols)
        tl.store(X_ptr + X_offsets, X_block * grad_output, mask=X_offsets <
            n_cols)


def fused_linear_cross_entropy_backward(grad_output, grad_input,
    grad_weight, grad_bias):
    if not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.
        device)):
        BT, H = grad_input.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))
        element_mul_kernel[n_rows,](grad_input, grad_input.stride(-2),
            grad_output, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=32 if not
            is_hip() else 16)
        if grad_weight is not None:
            V, H = grad_weight.shape
            n_rows = V
            element_mul_kernel[n_rows,](grad_weight, grad_weight.stride(-2),
                grad_output, H, BLOCK_SIZE=BLOCK_SIZE, num_warps=32 if not
                is_hip() else 16)
        if grad_bias is not None:
            V = grad_bias.shape[0]
            n_rows = V
            element_mul_kernel[n_rows,](grad_bias, grad_bias.stride(-1),
                grad_output, 1, BLOCK_SIZE=BLOCK_SIZE, num_warps=32 if not
                is_hip() else 16)
    return grad_input, grad_weight, grad_bias


# Backward method (kernel launch code)
@amp_custom_bwd
def _LigerFusedLinearCrossEntropyFunction_backward(ctx, grad_output,
    grad_output2):
    if ctx.return_z_loss:
        del grad_output2
    grad_input, grad_weight, grad_bias = ctx.saved_tensors
    grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_backward(
        grad_output, grad_input, grad_weight, grad_bias)
    return (grad_input, grad_weight, None, grad_bias, None, None, None,
        None, None, None, None, None, None)


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LigerFusedLinearCrossEntropyFunction(torch.autograd.Function):

    @staticmethod
    @amp_custom_fwd
    def forward(ctx, _input, weight, target, bias=None, ce_weight=None,
        ignore_index=-100, lse_square_scale=0.0, label_smoothing=0.0,
        reduction='mean', softcap=None, return_z_loss: bool=False,
        accum_dtype=None, use_token_scaling: bool=False):
        """
        Fusing the last linear layer with cross-entropy loss
            Reference: https://github.com/mgmalek/efficient_cross_entropy

        Handle the forward and backward pass of the final linear layer via cross-entropy loss by avoiding
        the materialization of the large logits tensor. Since Cross Entropy Loss is the last layer, we can
        compute the gradient at the forward pass. By doing so, we don't have to store the _input and target
        for the backward pass.

        _input: (B*T, H) where B is batch size, T is sequence length, H is hidden dimension.
        target: (B*T) where each value is in [0, V-1]
        weight: (V, H) where V is the number of classes
        bias: (V) where V is the number of classes
        ce_weight: a manual rescaling weight given to each class. If given, has to be a Tensor of size V and floating point dtype
        ignore_index: the index to ignore in the target
        label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
        reduction: reduction to apply
        accum_dtype (torch.dtype): the dtype of intermediate result buffers for weight and bias gradient accumulations.
            Recommended to set `accum_dtype` to higher precision, e.g. `torch.float32`, if the training is unstable with original dtype. Default: `None`, performing accumulations in original dtype
        use_token_scaling (bool): whether to scale each token's loss by its predicted probability (detached).
            When True, each token's loss is multiplied by the model's predicted probability for that token's true class.
            Default: False.
        """
        loss, z_loss, grad_input, grad_weight, grad_bias = (
            fused_linear_cross_entropy_forward(_input=_input, weight=weight,
            target=target, bias=bias, ce_weight=ce_weight, ignore_index=
            ignore_index, lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing, reduction=reduction, softcap=
            softcap, return_z_loss=return_z_loss, accum_dtype=accum_dtype,
            use_token_scaling=use_token_scaling))
        ctx.save_for_backward(grad_input.detach(), grad_weight.detach() if 
            grad_weight is not None else None, grad_bias.detach() if bias
             is not None else None)
        ctx.return_z_loss = return_z_loss
        return loss, z_loss

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output, grad_output2):
        if ctx.return_z_loss:
            del grad_output2
        grad_input, grad_weight, grad_bias = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = (
            fused_linear_cross_entropy_backward(grad_output, grad_input,
            grad_weight, grad_bias))
        return (grad_input, grad_weight, None, grad_bias, None, None, None,
            None, None, None, None, None, None)
