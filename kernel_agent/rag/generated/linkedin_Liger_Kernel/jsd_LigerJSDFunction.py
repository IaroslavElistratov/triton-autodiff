# SPDX-License-Identifier: BSD-2-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/linkedin/Liger-Kernel
# Source-Files: src/liger_kernel/ops/jsd.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_d1vk6zhm/Liger-Kernel-main/src/liger_kernel/ops/jsd.py
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
from math import log

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def _jsd_kernel(X_ptr, X_stride, Y_ptr, Y_stride, loss_ptr, loss_stride,
    dX_ptr, dX_stride, label_ptr, beta: tl.constexpr, n_non_ignore: int,
    ignore_index: tl.constexpr, n_cols, BLOCK_SIZE: tl.constexpr, HAS_LABEL:
    tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    X_ptr += pid * X_stride
    dX_ptr += pid * dX_stride
    Y_ptr += pid * Y_stride
    loss_ptr += pid * loss_stride
    label_ptr += pid
    if HAS_LABEL:
        label = tl.load(label_ptr)
        if label == ignore_index:
            for i in range(0, n_cols, BLOCK_SIZE):
                offsets = i + tl.arange(0, BLOCK_SIZE)
                tl.store(dX_ptr + offsets, 0.0, mask=offsets < n_cols)
            return
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        X = tl.load(X_ptr + offsets, mask=mask, other=float('-inf')).to(tl.
            float32)
        Y = tl.load(Y_ptr + offsets, mask=mask, other=float('-inf')).to(tl.
            float32)
        if beta == 0.0:
            Y_max = tl.max(Y, axis=0)
            Y_shifted = Y - Y_max
            Y_prob = tl.exp(Y_shifted) * tl.exp(Y_max)
            loss = Y_prob * (Y - X)
            dX = -Y_prob
        elif beta == 1.0:
            X_max = tl.max(X, axis=0)
            X_shifted = X - X_max
            X_prob = tl.exp(X_shifted) * tl.exp(X_max)
            loss = X_prob * (X - Y)
            dX = loss + X_prob
        else:
            max_val = tl.maximum(tl.max(X, axis=0), tl.max(Y, axis=0))
            X_shifted = X - max_val
            Y_shifted = Y - max_val
            exp_max = tl.exp(max_val)
            Q = tl.exp(X_shifted) * exp_max
            P = tl.exp(Y_shifted) * exp_max
            beta_P = beta * P
            one_minus_beta_Q = (1 - beta) * Q
            M = beta_P + one_minus_beta_Q
            log_M = tl.log(M)
            loss = beta_P * Y + one_minus_beta_Q * X - M * log_M
            dX = one_minus_beta_Q * (X - log_M)
        scale = 1.0 / n_non_ignore
        loss = loss * scale
        dX = dX * scale
        tl.store(loss_ptr + offsets, loss, mask=mask)
        tl.store(dX_ptr + offsets, dX, mask=mask)


def jsd_forward(_input, target, shift_labels, beta, ignore_index, has_label):
    BT, V = _input.shape
    n_rows = BT
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    loss = torch.zeros(_input.shape, dtype=torch.float32, device=_input.device)
    dX = torch.empty_like(_input)
    if has_label:
        n_non_ignore = (shift_labels != ignore_index).sum().item()
    else:
        n_non_ignore = BT
    _jsd_kernel[n_rows,](X_ptr=_input, X_stride=_input.stride(-2), Y_ptr=
        target, Y_stride=target.stride(-2), loss_ptr=loss, loss_stride=loss
        .stride(-2), dX_ptr=dX, dX_stride=dX.stride(-2), label_ptr=
        shift_labels if has_label else torch.empty(1, device=_input.device),
        beta=beta, n_non_ignore=n_non_ignore, ignore_index=ignore_index,
        n_cols=V, BLOCK_SIZE=BLOCK_SIZE, HAS_LABEL=has_label)
    loss = torch.sum(loss)
    return loss.to(_input.dtype), dX


# Forward method (kernel launch code)
@ensure_contiguous
def _LigerJSDFunction_forward(ctx, _input: torch.Tensor, target: torch.
    Tensor, shift_labels: Optional[torch.Tensor]=None, beta: float=0.5,
    ignore_index: int=-100) ->torch.Tensor:
    """
        Args:
            _input (torch.Tensor): predict values with shape (BT, V) in logspace
            target (torch.Tensor): ground truth values with shape (BT, V) in logspace
            shift_labels (Optional[torch.LongTensor]): indicator of next predicted vocab with shape (BT) where each value is in [0, V-1].
            beta (float): coefficient beta of generalized JSD in the interval [0, 1]. It implements forward/reverse KL when beta equals 0 and 1 respectively. Default: `0.5`
            ignore_index (int): the index to ignore. Default: -100

        Returns:
            loss (torch.Tensor): generalized JSD
        """
    has_label = False
    if shift_labels is not None:
        assert shift_labels.shape == (_input.shape[0],
            ), f'the shape of shift_labels must be (BT,). Got: {shift_labels.shape}'
        shift_labels = shift_labels.contiguous()
        has_label = True
    loss, dX = jsd_forward(_input, target, shift_labels, beta, ignore_index,
        has_label)
    ctx.save_for_backward(dX)
    return loss


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

def jsd_backward(dX, grad_output):
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        return dX
    else:
        return grad_output * dX


# Backward method (kernel launch code)
@ensure_contiguous
def _LigerJSDFunction_backward(ctx, grad_output: torch.Tensor) ->torch.Tensor:
    dX, = ctx.saved_tensors
    dX = jsd_backward(dX, grad_output)
    return dX, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LigerJSDFunction(torch.autograd.Function):
    """
    This class implements the forward and backward pass for the generalized Jensen-Shannon Divergence.
    .. math::
        JSD(\\beta)(P || Q)
            = \\beta * KLDiv(P || (\\beta * P + (1 - \\beta) * Q)) + (1 - \\beta) * KLDiv(Q || (\\beta * P + (1 - \\beta) * Q))

    .. note::
        As all the other losses in PyTorch, this function expects the first argument,
        :attr:`_input`, to be the predictions, the output of the student model, in log-space
        and the second, :attr:`target`, to be the observations, the output of the teacher model, in log-space.
        This differs from the standard mathematical notation :math:`JSD(P || Q)` where
        :math:`P` denotes the teacher model and :math:`Q` denotes the student model.
    """

    @staticmethod
    @ensure_contiguous
    def forward(ctx, _input: torch.Tensor, target: torch.Tensor,
        shift_labels: Optional[torch.Tensor]=None, beta: float=0.5,
        ignore_index: int=-100) ->torch.Tensor:
        """
        Args:
            _input (torch.Tensor): predict values with shape (BT, V) in logspace
            target (torch.Tensor): ground truth values with shape (BT, V) in logspace
            shift_labels (Optional[torch.LongTensor]): indicator of next predicted vocab with shape (BT) where each value is in [0, V-1].
            beta (float): coefficient beta of generalized JSD in the interval [0, 1]. It implements forward/reverse KL when beta equals 0 and 1 respectively. Default: `0.5`
            ignore_index (int): the index to ignore. Default: -100

        Returns:
            loss (torch.Tensor): generalized JSD
        """
        has_label = False
        if shift_labels is not None:
            assert shift_labels.shape == (_input.shape[0],
                ), f'the shape of shift_labels must be (BT,). Got: {shift_labels.shape}'
            shift_labels = shift_labels.contiguous()
            has_label = True
        loss, dX = jsd_forward(_input, target, shift_labels, beta,
            ignore_index, has_label)
        ctx.save_for_backward(dX)
        return loss

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor) ->torch.Tensor:
        dX, = ctx.saved_tensors
        dX = jsd_backward(dX, grad_output)
        return dX, None, None, None, None
