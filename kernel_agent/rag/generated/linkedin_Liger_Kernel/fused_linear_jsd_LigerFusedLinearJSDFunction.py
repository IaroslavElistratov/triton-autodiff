# SPDX-License-Identifier: BSD-2-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/linkedin/Liger-Kernel
# Source-Files: src/liger_kernel/ops/fused_linear_jsd.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_d1vk6zhm/Liger-Kernel-main/src/liger_kernel/ops/fused_linear_jsd.py
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


def fused_linear_jsd_forward(student_input, student_weight, teacher_input,
    teacher_weight, shift_labels, jsd_beta, ignore_index, has_label,
    temperature):
    device = student_input.device
    dtype = student_input.dtype
    BT, H = student_input.shape
    V = student_weight.shape[0]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    inc_factor = triton.cdiv(V, H)
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))
    num_chunks = triton.cdiv(BT, chunk_size)
    grad_weight = torch.zeros_like(student_weight, device=device
        ) if student_weight.requires_grad else None
    grad_input = torch.zeros_like(student_input)
    loss_1d = torch.zeros((BT, V), dtype=torch.float32, device=device)
    if has_label:
        n_non_ignore = (shift_labels != ignore_index).sum().item()
    else:
        n_non_ignore = BT
    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        student_input_chunk = student_input[start_idx:end_idx]
        teacher_input_chunk = teacher_input[start_idx:end_idx]
        student_logits_chunk = (student_input_chunk @ student_weight.t()).to(
            torch.float32)
        teacher_logits_chunk = (teacher_input_chunk @ teacher_weight.t()).to(
            torch.float32)
        chunk_n_rows = student_logits_chunk.shape[0]
        loss_1d_slice = loss_1d[start_idx:end_idx]
        student_logits_chunk = student_logits_chunk / temperature
        teacher_logits_chunk = teacher_logits_chunk / temperature
        student_prob_chunk = torch.log_softmax(student_logits_chunk, dim=-1)
        teacher_prob_chunk = torch.log_softmax(teacher_logits_chunk, dim=-1)
        student_prob_chunk = student_prob_chunk.contiguous()
        teacher_prob_chunk = teacher_prob_chunk.contiguous()
        _jsd_kernel[chunk_n_rows,](X_ptr=student_prob_chunk, X_stride=
            student_prob_chunk.stride(-2), Y_ptr=teacher_prob_chunk,
            Y_stride=teacher_prob_chunk.stride(-2), loss_ptr=loss_1d_slice,
            loss_stride=loss_1d_slice.stride(-2), dX_ptr=student_prob_chunk,
            dX_stride=student_prob_chunk.stride(-2), label_ptr=shift_labels
            [start_idx:end_idx] if has_label else torch.empty(1, device=
            device), beta=jsd_beta, n_non_ignore=n_non_ignore, ignore_index
            =ignore_index, n_cols=V, BLOCK_SIZE=BLOCK_SIZE, HAS_LABEL=has_label
            )
        loss_1d[start_idx:end_idx] = loss_1d_slice
        student_logits_chunk = (student_prob_chunk - torch.softmax(
            student_logits_chunk, dim=-1) * student_prob_chunk.sum(dim=-1,
            keepdim=True).broadcast_to(student_prob_chunk.shape)) / temperature
        student_logits_chunk = student_logits_chunk.to(dtype)
        grad_input[start_idx:end_idx] = student_logits_chunk @ student_weight
        if grad_weight is not None:
            grad_weight.add_(student_logits_chunk.t() @ student_input_chunk)
    loss = torch.sum(loss_1d)
    return loss, grad_input, grad_weight


# Forward method (kernel launch code)
@amp_custom_fwd
def _LigerFusedLinearJSDFunction_forward(ctx, student_input: torch.Tensor,
    student_weight: torch.Tensor, teacher_input: torch.Tensor,
    teacher_weight: torch.Tensor, shift_labels: Optional[torch.Tensor]=None,
    jsd_beta: float=0.5, ignore_index: int=-100, temperature: float=1.0):
    """
        Args:

            student_input (torch.tensor): input of the last projection layer in student model, with shape (B*T, H), where B is batch size, T is sequence length, H is hidden dimension.
            student_weight (torch.tensor): the last projection layer in student model, with shape (V, H), where V is vocab size
            teacher_input (torch.tensor): input of the last projection layer in teacher model, with shape (B*T, H), where B is batch size, T is sequence length, H is hidden dimension.
            teacher_weight (torch.tensor): the last projection layer in teacher model, with shape (V, H), where V is vocab size
            shift_labels (Optional[torch.LongTensor]): indicator of next predicted vocab with shape (BT) where each value is in [0, V-1].
            jsd_beta (float): coefficient beta of generalized JSD in the interval [0, 1]. It implements forward/reverse KL when beta equals 0 and 1 respectively. Default: `0.5`
            ignore_index (int): the index to ignore. Default: -100
            temperature (float): temperature in softmax function to control the output probability distribution. Default: `1.0`

        Returns:
            loss (torch.Tensor): generalized JSD
        """
    has_label = False
    if shift_labels is not None:
        assert shift_labels.shape == (teacher_input.shape[0],
            ), f'the shape of shift_labels must be (BT,). Got: {shift_labels.shape}'
        shift_labels = shift_labels.contiguous()
        has_label = True
    loss, grad_input, grad_weight = fused_linear_jsd_forward(student_input,
        student_weight, teacher_input, teacher_weight, shift_labels,
        jsd_beta, ignore_index, has_label, temperature)
    ctx.save_for_backward(grad_input.detach(), grad_weight.detach() if 
        grad_weight is not None else None)
    return loss


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


def fused_linear_jsd_backward(grad_output, grad_input, grad_weight):
    if torch.ne(grad_output, torch.tensor(1.0, device=grad_output.device)):
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
    return grad_input, grad_weight


def is_hip() ->bool:
    return torch.version.hip is not None


# Backward method (kernel launch code)
@amp_custom_bwd
def _LigerFusedLinearJSDFunction_backward(ctx, grad_output):
    grad_input, grad_weight = ctx.saved_tensors
    grad_input, grad_weight = fused_linear_jsd_backward(grad_output,
        grad_input, grad_weight)
    return grad_input, grad_weight, None, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LigerFusedLinearJSDFunction(torch.autograd.Function):
    """
    Fusing the last linear layer with generalized JSD

    Handle the forward and backward pass of the final linear layer via JSD by avoiding
    the materialization of the large logits tensor. Since JSD is the last layer, we can
    compute the gradient at the forward pass.
    """

    @staticmethod
    @amp_custom_fwd
    def forward(ctx, student_input: torch.Tensor, student_weight: torch.
        Tensor, teacher_input: torch.Tensor, teacher_weight: torch.Tensor,
        shift_labels: Optional[torch.Tensor]=None, jsd_beta: float=0.5,
        ignore_index: int=-100, temperature: float=1.0):
        """
        Args:

            student_input (torch.tensor): input of the last projection layer in student model, with shape (B*T, H), where B is batch size, T is sequence length, H is hidden dimension.
            student_weight (torch.tensor): the last projection layer in student model, with shape (V, H), where V is vocab size
            teacher_input (torch.tensor): input of the last projection layer in teacher model, with shape (B*T, H), where B is batch size, T is sequence length, H is hidden dimension.
            teacher_weight (torch.tensor): the last projection layer in teacher model, with shape (V, H), where V is vocab size
            shift_labels (Optional[torch.LongTensor]): indicator of next predicted vocab with shape (BT) where each value is in [0, V-1].
            jsd_beta (float): coefficient beta of generalized JSD in the interval [0, 1]. It implements forward/reverse KL when beta equals 0 and 1 respectively. Default: `0.5`
            ignore_index (int): the index to ignore. Default: -100
            temperature (float): temperature in softmax function to control the output probability distribution. Default: `1.0`

        Returns:
            loss (torch.Tensor): generalized JSD
        """
        has_label = False
        if shift_labels is not None:
            assert shift_labels.shape == (teacher_input.shape[0],
                ), f'the shape of shift_labels must be (BT,). Got: {shift_labels.shape}'
            shift_labels = shift_labels.contiguous()
            has_label = True
        loss, grad_input, grad_weight = fused_linear_jsd_forward(student_input,
            student_weight, teacher_input, teacher_weight, shift_labels,
            jsd_beta, ignore_index, has_label, temperature)
        ctx.save_for_backward(grad_input.detach(), grad_weight.detach() if 
            grad_weight is not None else None)
        return loss

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output):
        grad_input, grad_weight = ctx.saved_tensors
        grad_input, grad_weight = fused_linear_jsd_backward(grad_output,
            grad_input, grad_weight)
        return grad_input, grad_weight, None, None, None, None, None, None
