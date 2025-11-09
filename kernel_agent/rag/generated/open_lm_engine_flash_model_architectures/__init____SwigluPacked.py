# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/open-lm-engine/flash-model-architectures
# Source-Files: fma/functional/swiglu/__init__.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_r3l3p8y1/flash-model-architectures-main/fma/functional/swiglu/__init__.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

def ceil_divide(x: int, y: int) ->int:
    return (x + y - 1) // y


@triton.jit
def sigmoid(x, output_dtype: tl.constexpr=None):
    if output_dtype is None:
        output_dtype = x.dtype
    x = x.to(tl.float32)
    x = tanh(0.5 * x, output_dtype=tl.float32)
    x = 0.5 * x + 0.5
    x = x.to(output_dtype)
    return x


@triton.jit
def tanh(x, output_dtype: tl.constexpr=None):
    if output_dtype is None:
        output_dtype = x.dtype
    x = x.to(tl.float32)
    x = tl.inline_asm_elementwise('tanh.approx.f32 $0, $1;', '=f,f', [x],
        dtype=tl.float32, is_pure=True, pack=1)
    x = x.to(output_dtype)
    return x


def get_num_elements_and_hidden_size(x: torch.Tensor) ->tuple[int]:
    hidden_size = x.size(-1)
    num_elements = x.numel() // hidden_size
    return num_elements, hidden_size


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def swiglu_forward_triton_kernel(gate_ptr, gate_stride_b, up_ptr,
    output_ptr, output_stride_b, B, H, BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr):
    BLOCK_ID_B = tl.program_id(axis=0)
    BLOCK_ID_H = tl.program_id(axis=1)
    indices_b = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = BLOCK_ID_H * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    mask_b = indices_b < B
    mask_h = indices_h < H
    mask = mask_b[:, None] & mask_h[None, :]
    indices = indices_b[:, None] * gate_stride_b + indices_h[None, :]
    gate = tl.load(gate_ptr + indices, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + indices, mask=mask)
    output = up * gate * sigmoid(gate)
    indices = indices_b[:, None] * output_stride_b + indices_h[None, :]
    tl.store(output_ptr + indices, output, mask=mask)


def divide_if_divisible(dividend: int, divisor: int, msg: str='') ->int:
    assert dividend % divisor == 0, msg
    return dividend // divisor


@custom_op(f'{LIBRARY_NAME}::swiglu_forward_triton', mutates_args={'output'})
def swiglu_forward_triton(gate: torch.Tensor, up: torch.Tensor, output:
    torch.Tensor) ->None:
    B, H = get_num_elements_and_hidden_size(gate)
    BLOCK_SIZE_B = 64
    BLOCK_SIZE_H = 64
    with torch.device(gate.device):
        swiglu_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B),
            ceil_divide(H, BLOCK_SIZE_H)](gate_ptr=gate, gate_stride_b=gate
            .stride(-2), up_ptr=up, output_ptr=output, output_stride_b=
            output.stride(-2), B=B, H=H, BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H)


# Forward method (kernel launch code)
@ensure_contiguous
def __SwigluPacked_forward(ctx, x: torch.Tensor) ->torch.Tensor:
    ctx.save_for_backward(x)
    output = torch.empty(*x.size()[:-1], divide_if_divisible(x.size(-1), 2),
        device=x.device, dtype=x.dtype)
    up, gate = x.chunk(2, dim=-1)
    swiglu_forward_triton(gate=gate, up=up, output=output)
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def swiglu_backward_triton_kernel(gate_ptr, gate_stride_b, up_ptr,
    output_grad_ptr, output_grad_stride_b, gate_grad_ptr, up_grad_ptr, B, H,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_H: tl.constexpr):
    BLOCK_ID_B = tl.program_id(axis=0)
    BLOCK_ID_H = tl.program_id(axis=1)
    indices_b = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = BLOCK_ID_H * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    mask_b = indices_b < B
    mask_h = indices_h < H
    mask = mask_b[:, None] & mask_h[None, :]
    indices_gate = indices_b[:, None] * gate_stride_b + indices_h[None, :]
    indices_output = indices_b[:, None] * output_grad_stride_b + indices_h[
        None, :]
    gate = tl.load(gate_ptr + indices_gate, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + indices_gate, mask=mask)
    output_grad = tl.load(output_grad_ptr + indices_output, mask=mask)
    gate_sigmoid = sigmoid(gate)
    gate_silu = gate * gate_sigmoid
    gate_grad = output_grad * up * (gate_sigmoid + gate_silu * (1 -
        gate_sigmoid))
    up_grad = output_grad * gate_silu
    tl.store(gate_grad_ptr + indices_gate, gate_grad, mask=mask)
    tl.store(up_grad_ptr + indices_gate, up_grad, mask=mask)


@custom_op(f'{LIBRARY_NAME}::swiglu_backward_triton', mutates_args={
    'gate_grad', 'up_grad'})
def swiglu_backward_triton(gate: torch.Tensor, up: torch.Tensor,
    output_grad: torch.Tensor, gate_grad: torch.Tensor, up_grad: torch.Tensor
    ) ->None:
    B, H = get_num_elements_and_hidden_size(gate)
    BLOCK_SIZE_B = 64
    BLOCK_SIZE_H = 64
    with torch.device(gate.device):
        swiglu_backward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B),
            ceil_divide(H, BLOCK_SIZE_H)](gate_ptr=gate, gate_stride_b=gate
            .stride(-2), up_ptr=up, output_grad_ptr=output_grad,
            output_grad_stride_b=output_grad.stride(-2), gate_grad_ptr=
            gate_grad, up_grad_ptr=up_grad, B=B, H=H, BLOCK_SIZE_B=
            BLOCK_SIZE_B, BLOCK_SIZE_H=BLOCK_SIZE_H)


def empty_like_contiguous(x: torch.Tensor, dtype: (torch.dtype | None)=None
    ) ->torch.Tensor:
    return torch.empty_like(x, dtype=dtype, memory_format=torch.
        contiguous_format)


# Backward method (kernel launch code)
@ensure_contiguous
def __SwigluPacked_backward(ctx, output_grad: torch.Tensor) ->tuple[torch.
    Tensor | None]:
    x: torch.Tensor = ctx.saved_tensors[0]
    x_grad = empty_like_contiguous(x)
    up, gate = x.chunk(2, dim=-1)
    up_grad, gate_grad = x_grad.chunk(2, dim=-1)
    swiglu_backward_triton(gate=gate, up=up, output_grad=output_grad,
        gate_grad=gate_grad, up_grad=up_grad)
    return x_grad


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _SwigluPacked(torch.autograd.Function):

    @staticmethod
    @ensure_contiguous
    def forward(ctx, x: torch.Tensor) ->torch.Tensor:
        ctx.save_for_backward(x)
        output = torch.empty(*x.size()[:-1], divide_if_divisible(x.size(-1),
            2), device=x.device, dtype=x.dtype)
        up, gate = x.chunk(2, dim=-1)
        swiglu_forward_triton(gate=gate, up=up, output=output)
        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) ->tuple[torch.Tensor | None]:
        x: torch.Tensor = ctx.saved_tensors[0]
        x_grad = empty_like_contiguous(x)
        up, gate = x.chunk(2, dim=-1)
        up_grad, gate_grad = x_grad.chunk(2, dim=-1)
        swiglu_backward_triton(gate=gate, up=up, output_grad=output_grad,
            gate_grad=gate_grad, up_grad=up_grad)
        return x_grad
