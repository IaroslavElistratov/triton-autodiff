# SPDX-License-Identifier: BSD-2-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/linkedin/Liger-Kernel
# Source-Files: src/liger_kernel/ops/cross_entropy.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_d1vk6zhm/Liger-Kernel-main/src/liger_kernel/ops/cross_entropy.py
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


def cross_entropy_forward(_input, target, weight, ignore_index,
    lse_square_scale, label_smoothing, reduction, softcap, return_z_loss):
    assert isinstance(return_z_loss, bool
        ), f'return_z_loss must be True or False. Got: {return_z_loss}'
    BT, V = _input.shape
    n_rows = BT
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    loss_1d = torch.zeros(n_rows, dtype=_input.dtype, device=_input.device)
    z_loss_1d = torch.zeros(n_rows, dtype=_input.dtype, device=_input.device
        ) if return_z_loss else None
    target_mask = target != ignore_index
    n_non_ignore = target_mask.sum().item()
    assert (target * target_mask).max() < _input.shape[-1
        ], f'Target {target.max()} is out of bounds. Expected < {_input.shape[-1]}'
    assert (target * target_mask).min(
        ) >= 0, f'Target {target.min()} is out of bounds. Expected >= 0'
    sum_non_ignore_weight = n_non_ignore
    weight_sum = 0.0
    if weight is not None:
        assert weight.shape[0
            ] == V, f'If given, weight has to be a Tensor of size V. Got: {weight.shape}'
        assert torch.is_floating_point(weight
            ), f'If given, weight has to be a Tensor of floating point dtype. Got: {weight.dtype}'
        sum_non_ignore_weight = torch.gather(weight, dim=0, index=target.
            masked_select(target_mask)).sum().item()
        weight_sum = weight.sum().item()
        if weight.stride(-1) != 1:
            weight = weight.contiguous()
    if _input.stride(-1) != 1:
        _input = _input.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()
    liger_cross_entropy_kernel[n_rows,](X_ptr=_input, X_stride=_input.
        stride(-2), Y_ptr=target, Y_stride=target.stride(-1), weight_ptr=
        weight, loss_ptr=loss_1d, z_loss_ptr=z_loss_1d, loss_stride=loss_1d
        .stride(-1), n_cols=V, n_non_ignore=n_non_ignore,
        sum_non_ignore_weight=sum_non_ignore_weight, ignore_index=
        ignore_index, weight_sum=weight_sum, lse_square_scale=
        lse_square_scale, label_smoothing=label_smoothing, reduction=
        reduction, softcap=softcap, RETURN_Z_LOSS=return_z_loss, BLOCK_SIZE
        =BLOCK_SIZE, HAS_WEIGHT=True if weight is not None else False,
        HAS_SOFTCAPPING=True if softcap is not None else False,
        HAS_GRADIENTS=_input.requires_grad, num_warps=32 if not is_hip() else
        16)
    if reduction == 'none':
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None
    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None
    return loss, z_loss, _input


# Forward method (kernel launch code)
def _LigerCrossEntropyFunction_forward(ctx, _input: torch.Tensor, target:
    torch.Tensor, weight: Optional[torch.FloatTensor], ignore_index: int=-
    100, lse_square_scale: float=0.0, label_smoothing: float=0.0, reduction:
    str='mean', softcap: Optional[float]=None, return_z_loss: bool=False):
    """
        The forward pass of the Liger Cross Entropy loss.

        Parameters:
        ctx : The context object.
        _input (tensor): The input tensor of shape (BT, V) where B is batch size, T is sequence length, V is vocab size.
        target (tensor): The target tensor of shape (BT) where each value is in [0, V-1].
        weight(Tensor, optional): a manual rescaling weight given to each class. If given, has to be a Tensor of size V and floating point dtype
        ignore_index (int): The index to ignore in the target.
        lse_square_scale (float): The scaler of (logsumexp(_input)) ^ 2 adding to the loss for the stability of training.
        label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
        reduction (str): The reduction to apply to the output: "none" | "mean | "sum".
        softcap (Optional[float]): The upper threshold for scaling logits to the range (-softcap, +softcap).
        return_z_loss (bool): When `return_z_loss` is `True`, returns (loss, z_loss) instead of (loss, None). Default: `False`

        Returns:
        tuple: A tuple with the compouted losses with respect to loss and z loss. The elements are tensors or None.
        """
    input_requires_grad = _input.requires_grad
    loss, z_loss, _input = cross_entropy_forward(_input, target, weight,
        ignore_index, lse_square_scale, label_smoothing, reduction, softcap,
        return_z_loss)
    if input_requires_grad:
        ctx.save_for_backward(_input.detach())
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


def cross_entropy_backward(_input, grad_output):
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        pass
    elif grad_output.ndim > 0:
        _input = _input * grad_output.unsqueeze(dim=1)
    else:
        BT, V = _input.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
        element_mul_kernel[n_rows,](_input, _input.stride(-2), grad_output,
            V, BLOCK_SIZE=BLOCK_SIZE, num_warps=32 if not is_hip() else 16)
    return _input


# Backward method (kernel launch code)
def _LigerCrossEntropyFunction_backward(ctx, grad_output, grad_ouput2):
    """
        The backward pass of the Liger Cross Entropy loss.

        Parameters:
        ctx : The context object with saved tensors.
        grad_output (tensor): The tensor containing the gradient of the loss with respect to the output.
        grad_output2 (tenosr): No use.
        Returns:
        tuple: A tuple with the gradients with respect to the inputs. The elements are tensors or None.
        """
    if ctx.return_z_loss:
        del grad_ouput2
    _input, = ctx.saved_tensors
    _input = cross_entropy_backward(_input, grad_output)
    return _input, None, None, None, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LigerCrossEntropyFunction(torch.autograd.Function):
    """
    This class implements a custom autograd function for the Liger Cross Entropy loss.
    It overrides the forward and backward methods of the torch.autograd.Function class.
    """

    @staticmethod
    def forward(ctx, _input: torch.Tensor, target: torch.Tensor, weight:
        Optional[torch.FloatTensor], ignore_index: int=-100,
        lse_square_scale: float=0.0, label_smoothing: float=0.0, reduction:
        str='mean', softcap: Optional[float]=None, return_z_loss: bool=False):
        """
        The forward pass of the Liger Cross Entropy loss.

        Parameters:
        ctx : The context object.
        _input (tensor): The input tensor of shape (BT, V) where B is batch size, T is sequence length, V is vocab size.
        target (tensor): The target tensor of shape (BT) where each value is in [0, V-1].
        weight(Tensor, optional): a manual rescaling weight given to each class. If given, has to be a Tensor of size V and floating point dtype
        ignore_index (int): The index to ignore in the target.
        lse_square_scale (float): The scaler of (logsumexp(_input)) ^ 2 adding to the loss for the stability of training.
        label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
        reduction (str): The reduction to apply to the output: "none" | "mean | "sum".
        softcap (Optional[float]): The upper threshold for scaling logits to the range (-softcap, +softcap).
        return_z_loss (bool): When `return_z_loss` is `True`, returns (loss, z_loss) instead of (loss, None). Default: `False`

        Returns:
        tuple: A tuple with the compouted losses with respect to loss and z loss. The elements are tensors or None.
        """
        input_requires_grad = _input.requires_grad
        loss, z_loss, _input = cross_entropy_forward(_input, target, weight,
            ignore_index, lse_square_scale, label_smoothing, reduction,
            softcap, return_z_loss)
        if input_requires_grad:
            ctx.save_for_backward(_input.detach())
        ctx.return_z_loss = return_z_loss
        return loss, z_loss

    @staticmethod
    def backward(ctx, grad_output, grad_ouput2):
        """
        The backward pass of the Liger Cross Entropy loss.

        Parameters:
        ctx : The context object with saved tensors.
        grad_output (tensor): The tensor containing the gradient of the loss with respect to the output.
        grad_output2 (tenosr): No use.
        Returns:
        tuple: A tuple with the gradients with respect to the inputs. The elements are tensors or None.
        """
        if ctx.return_z_loss:
            del grad_ouput2
        _input, = ctx.saved_tensors
        _input = cross_entropy_backward(_input, grad_output)
        return _input, None, None, None, None, None, None, None, None
