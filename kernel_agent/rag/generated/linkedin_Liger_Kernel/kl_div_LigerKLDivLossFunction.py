# SPDX-License-Identifier: BSD-2-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/linkedin/Liger-Kernel
# Source-Files: src/liger_kernel/ops/kl_div.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_d1vk6zhm/Liger-Kernel-main/src/liger_kernel/ops/kl_div.py
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
from math import exp
from math import log

def get_num_warps(BLOCK_SIZE):
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32 if not is_hip() else 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return num_warps


def is_hip() ->bool:
    return torch.version.hip is not None


def infer_device():
    """
    Get current device name based on available devices
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.xpu.is_available():
        return 'xpu'
    else:
        return 'cpu'


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def _kldiv_kernel_forward(y_ptr, y_stride, gt_ptr, gt_stride, loss_ptr,
    loss_stride, n_cols, eps, BLOCK_SIZE: tl.constexpr, log_target: tl.
    constexpr=False, reduction: tl.constexpr=_REDUCTION_MODE_BATCHMEAN):
    pid = tl.program_id(0).to(tl.int64)
    y_ptr += pid * y_stride
    gt_ptr += pid * gt_stride
    loss_ptr += pid * loss_stride
    base_offsets = tl.arange(0, BLOCK_SIZE)
    loss_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)
        if not log_target:
            loss = y_true * (tl.log(tl.maximum(y_true, eps)) - y)
        else:
            loss = tl.exp(y_true) * (y_true - y)
        if reduction == _REDUCTION_MODE_NONE:
            tl.store(loss_ptr + offsets, loss, mask=mask)
        else:
            loss_sum += tl.sum(loss, axis=0)
    if reduction != _REDUCTION_MODE_NONE:
        tl.store(loss_ptr, loss_sum)


def kldiv_forward_triton(y_pred, y_true, log_target, reduction, eps):
    BT, V = y_pred.shape
    BLOCK_SIZE = min(8192, triton.next_power_of_2(V)) if infer_device(
        ) == 'xpu' else min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    num_warps = 32 if infer_device() == 'xpu' else get_num_warps(BLOCK_SIZE)
    grid = BT,
    reduction = _str_to_reduction_mode[reduction]
    out_size = (BT, V) if reduction == _REDUCTION_MODE_NONE.value else (BT,)
    output_tensor = torch.zeros(out_size, device=y_pred.device, dtype=torch
        .float32)
    _kldiv_kernel_forward[grid](y_pred, y_pred.stride(0), y_true, y_true.
        stride(0), output_tensor, output_tensor.stride(0), V, eps=eps,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, log_target=log_target,
        reduction=reduction)
    if reduction == _REDUCTION_MODE_BATCHMEAN.value:
        return output_tensor.sum() / BT
    elif reduction == _REDUCTION_MODE_SUM.value:
        return output_tensor.sum(dim=0)
    elif reduction == _REDUCTION_MODE_MEAN.value:
        return output_tensor.sum() / (BT * V)
    else:
        return output_tensor


# Forward method (kernel launch code)
@ensure_contiguous
def _LigerKLDivLossFunction_forward(ctx, y_pred: torch.Tensor, y_true:
    torch.Tensor, reduction: REDUCTION_LITERAL='batchmean', log_target:
    bool=False, eps: float=1e-10) ->torch.Tensor:
    """A forward pass for the KL Divergence Loss.

        Args:
            ctx: Torch autograd context
            y_pred (torch.Tensor): A tensor of shape (BT, V) containing the predicted values, expected to be log-probabilities.
            y_true (torch.Tensor): A tensor of shape (BT, V) containing the target values, expected to be either probabilities or log-probabilities, depending on the value of `log_target`.
            reduction (REDUCTION_LITERAL, optional): Reduction to be used. Defaults to "batchmean".
            log_target (bool, optional): If set to true, expects the ground truth to already be log-probabilities. Defaults to False.
            eps: (float, optional): A small value to avoid division by zero. Defaults to 1e-10.

        Returns:
            torch.Tensor: The computed KL Divergence Loss, with shape (BT, V) if `reduction` is "none", else a scalar.
        """
    ctx.save_for_backward(y_true)
    ctx.reduction = reduction
    ctx.log_target = log_target
    return kldiv_forward_triton(y_pred, y_true, log_target=log_target,
        reduction=reduction, eps=eps)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _kldiv_kernel_backward(target_ptr, target_stride, new_grads_ptr,
    new_grads_stride, n_cols, BLOCK_SIZE: tl.constexpr, log_target: tl.
    constexpr=False):
    pid = tl.program_id(0).to(tl.int64)
    target_ptr += pid * target_stride
    new_grads_ptr += pid * new_grads_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
        if not log_target:
            res = target * -1
        else:
            res = -tl.exp(target)
        tl.store(new_grads_ptr + offsets, res, mask=mask)


def kldiv_backward_triton(target, grad_output, new_grads, log_target):
    BT, V = target.shape
    BLOCK_SIZE = min(8192, triton.next_power_of_2(V)) if infer_device(
        ) == 'xpu' else min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    num_warps = 32 if infer_device() == 'xpu' else get_num_warps(BLOCK_SIZE)
    grid = BT,
    _kldiv_kernel_backward[grid](target, target.stride(0), new_grads,
        new_grads.stride(0), V, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
        log_target=log_target)
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        return new_grads
    return new_grads * grad_output


# Backward method (kernel launch code)
@ensure_contiguous
def _LigerKLDivLossFunction_backward(ctx, grad_output: torch.Tensor
    ) ->torch.Tensor:
    """A backward pass for the KL Divergence Loss.

        Args:
            ctx: Torch autograd context
            grad_output (torch.Tensor): The gradient of the loss with respect to the output.

        Returns:
            tuple[torch.Tensor, None, None, None, None]: The gradient of the loss with respect to the inputs and None for the other arguments of the forward method.
        """
    y_true, = ctx.saved_tensors
    new_grads = torch.empty_like(y_true)
    derivative = kldiv_backward_triton(y_true, grad_output, new_grads, ctx.
        log_target)
    if ctx.reduction == 'batchmean':
        derivative = derivative / y_true.shape[0]
    elif ctx.reduction == 'sum' or ctx.reduction == 'none':
        pass
    elif ctx.reduction == 'mean':
        derivative = derivative / (y_true.shape[0] * y_true.shape[1])
    return derivative, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LigerKLDivLossFunction(torch.autograd.Function):
    """
    Class implementing the forward and backward pass for the KL Divergence Loss using Triton, as defined by the following formula:
    ```python
    if log_target:
        loss = target.exp() * (target - input)
    else:
        loss = target * (target.log() - input)
    ```,
    then the loss is reduced according to the `reduction` parameter.
    as defined in the PyTorch documentation: https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
    """

    @staticmethod
    @ensure_contiguous
    def forward(ctx, y_pred: torch.Tensor, y_true: torch.Tensor, reduction:
        REDUCTION_LITERAL='batchmean', log_target: bool=False, eps: float=1e-10
        ) ->torch.Tensor:
        """A forward pass for the KL Divergence Loss.

        Args:
            ctx: Torch autograd context
            y_pred (torch.Tensor): A tensor of shape (BT, V) containing the predicted values, expected to be log-probabilities.
            y_true (torch.Tensor): A tensor of shape (BT, V) containing the target values, expected to be either probabilities or log-probabilities, depending on the value of `log_target`.
            reduction (REDUCTION_LITERAL, optional): Reduction to be used. Defaults to "batchmean".
            log_target (bool, optional): If set to true, expects the ground truth to already be log-probabilities. Defaults to False.
            eps: (float, optional): A small value to avoid division by zero. Defaults to 1e-10.

        Returns:
            torch.Tensor: The computed KL Divergence Loss, with shape (BT, V) if `reduction` is "none", else a scalar.
        """
        ctx.save_for_backward(y_true)
        ctx.reduction = reduction
        ctx.log_target = log_target
        return kldiv_forward_triton(y_pred, y_true, log_target=log_target,
            reduction=reduction, eps=eps)

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor) ->torch.Tensor:
        """A backward pass for the KL Divergence Loss.

        Args:
            ctx: Torch autograd context
            grad_output (torch.Tensor): The gradient of the loss with respect to the output.

        Returns:
            tuple[torch.Tensor, None, None, None, None]: The gradient of the loss with respect to the inputs and None for the other arguments of the forward method.
        """
        y_true, = ctx.saved_tensors
        new_grads = torch.empty_like(y_true)
        derivative = kldiv_backward_triton(y_true, grad_output, new_grads,
            ctx.log_target)
        if ctx.reduction == 'batchmean':
            derivative = derivative / y_true.shape[0]
        elif ctx.reduction == 'sum' or ctx.reduction == 'none':
            pass
        elif ctx.reduction == 'mean':
            derivative = derivative / (y_true.shape[0] * y_true.shape[1])
        return derivative, None, None, None, None
