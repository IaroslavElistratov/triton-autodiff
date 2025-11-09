# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/open-lm-engine/flash-model-architectures
# Source-Files: fma/functional/softmax/__init__.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_r3l3p8y1/flash-model-architectures-main/fma/functional/softmax/__init__.py
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

def ceil_divide(x: int, y: int) ->int:
    return (x + y - 1) // y


def get_next_power_of_2(x: int) ->int:
    for p in _POWERS_OF_2:
        if p >= x:
            return p
    raise ValueError(
        f'x ({x}) is bigger than the max allowable power of 2 ({p})')


def empty_like_contiguous(x: torch.Tensor, dtype: (torch.dtype | None)=None
    ) ->torch.Tensor:
    return torch.empty_like(x, dtype=dtype, memory_format=torch.
        contiguous_format)


def get_num_elements_and_hidden_size(x: torch.Tensor) ->tuple[int]:
    hidden_size = x.size(-1)
    num_elements = x.numel() // hidden_size
    return num_elements, hidden_size


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def softmax_forward_triton_kernel(x_ptr, x_stride, y_ptr, y_stride,
    logits_multiplier, B, H, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_H: tl.
    constexpr):
    BLOCK_ID = tl.program_id(axis=0)
    BLOCK_B = BLOCK_ID * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    MASK_B = BLOCK_B < B
    Z = tl.zeros((BLOCK_SIZE_B, 1), dtype=tl.float32)
    M = tl.full((BLOCK_SIZE_B, 1), -float('inf'), dtype=tl.float32)
    NUM_BLOCKS_H = tl.cdiv(H, BLOCK_SIZE_H)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)
    x_ptrs = x_ptr + BLOCK_B[:, None] * x_stride[0] + BLOCK_H[None, :
        ] * x_stride[1]
    for _ in range(NUM_BLOCKS_H):
        MASK_H = BLOCK_H < H
        MASK_BH = MASK_B[:, None] & MASK_H[None, :]
        x = tl.load(x_ptrs, mask=MASK_BH, other=-float('inf'))
        x = x.to(tl.float32)
        if logits_multiplier is not None:
            x *= logits_multiplier
        prev_m = M
        m = tl.max(x, axis=1, keep_dims=True)
        M = max(M, m)
        x -= M
        x = tl.exp(x)
        Z = Z * tl.exp(prev_m - M) + tl.sum(x, axis=1, keep_dims=True)
        BLOCK_H += BLOCK_SIZE_H
        x_ptrs += BLOCK_SIZE_H * x_stride[1]
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)
    x_ptrs = x_ptr + BLOCK_B[:, None] * x_stride[0] + BLOCK_H[None, :
        ] * x_stride[1]
    y_ptrs = y_ptr + BLOCK_B[:, None] * y_stride[0] + BLOCK_H[None, :
        ] * y_stride[1]
    for _ in range(NUM_BLOCKS_H):
        MASK_H = BLOCK_H < H
        MASK_BH = MASK_B[:, None] & MASK_H[None, :]
        x = tl.load(x_ptrs, mask=MASK_BH)
        x = x.to(tl.float32)
        if logits_multiplier is not None:
            x *= logits_multiplier
        x -= M
        x = tl.exp(x)
        x /= Z
        tl.store(y_ptrs, x, mask=MASK_BH)
        BLOCK_H += BLOCK_SIZE_H
        x_ptrs += BLOCK_SIZE_H * x_stride[1]
        y_ptrs += BLOCK_SIZE_H * y_stride[1]


@custom_op(f'{LIBRARY_NAME}::softmax_forward_triton', mutates_args={'output'})
def softmax_forward_triton(x: torch.Tensor, output: torch.Tensor,
    logits_multiplier: (float | None)) ->None:
    if x.dim() == 1:
        B = 1
        H = x.size(-1)
    else:
        B, H = get_num_elements_and_hidden_size(x)
    BLOCK_SIZE_B = 1
    BLOCK_SIZE_H = min(get_next_power_of_2(H), 4096 if x.dtype == torch.
        float32 else 8192)
    with torch.device(x.device):
        softmax_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B),](x_ptr=
            x, x_stride=x.stride(), y_ptr=output, y_stride=output.stride(),
            logits_multiplier=logits_multiplier, B=B, H=H, BLOCK_SIZE_B=
            BLOCK_SIZE_B, BLOCK_SIZE_H=BLOCK_SIZE_H)


# Forward method (kernel launch code)
def __Softmax_forward(ctx, x: torch.Tensor, logits_multiplier: (float | None)
    ) ->torch.Tensor:
    output = empty_like_contiguous(x)
    softmax_forward_triton(x=x, output=output, logits_multiplier=
        logits_multiplier)
    ctx.save_for_backward(output)
    ctx.logits_multiplier = logits_multiplier
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def softmax_backward_triton_kernel(y_ptr, y_stride, dy_ptr, dy_stride,
    dx_ptr, dx_stride, logits_multiplier, B, H, BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr):
    BLOCK_ID = tl.program_id(axis=0)
    BLOCK_B = BLOCK_ID * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    MASK_B = BLOCK_B < B
    accumulator = tl.zeros((BLOCK_SIZE_B, 1), dtype=tl.float32)
    NUM_BLOCKS_H = tl.cdiv(H, BLOCK_SIZE_H)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)
    y_ptrs = y_ptr + BLOCK_B[:, None] * y_stride[0] + BLOCK_H[None, :
        ] * y_stride[1]
    dy_ptrs = dy_ptr + BLOCK_B[:, None] * dy_stride[0] + BLOCK_H[None, :
        ] * dy_stride[1]
    for _ in range(NUM_BLOCKS_H):
        MASK_H = BLOCK_H < H
        MASK_BH = MASK_B[:, None] & MASK_H[None, :]
        y = tl.load(y_ptrs, mask=MASK_BH)
        dy = tl.load(dy_ptrs, mask=MASK_BH)
        acc = dy * y
        acc = acc.to(tl.float32)
        accumulator += tl.sum(acc, axis=1, keep_dims=True)
        BLOCK_H += BLOCK_SIZE_H
        y_ptrs += BLOCK_SIZE_H * y_stride[1]
        dy_ptrs += BLOCK_SIZE_H * dy_stride[1]
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)
    y_ptrs = y_ptr + BLOCK_B[:, None] * y_stride[0] + BLOCK_H[None, :
        ] * y_stride[1]
    dy_ptrs = dy_ptr + BLOCK_B[:, None] * dy_stride[0] + BLOCK_H[None, :
        ] * dy_stride[1]
    dx_ptrs = dx_ptr + BLOCK_B[:, None] * dx_stride[0] + BLOCK_H[None, :
        ] * dx_stride[1]
    for _ in range(NUM_BLOCKS_H):
        MASK_H = BLOCK_H < H
        MASK_BH = MASK_B[:, None] & MASK_H[None, :]
        y = tl.load(y_ptrs, mask=MASK_BH)
        dy = tl.load(dy_ptrs, mask=MASK_BH)
        dy -= accumulator
        y *= dy
        if logits_multiplier is not None:
            y *= logits_multiplier
        tl.store(dx_ptrs, y, mask=MASK_BH)
        BLOCK_H += BLOCK_SIZE_H
        y_ptrs += BLOCK_SIZE_H * y_stride[1]
        dy_ptrs += BLOCK_SIZE_H * dy_stride[1]
        dx_ptrs += BLOCK_SIZE_H * dx_stride[1]


@custom_op(f'{LIBRARY_NAME}::softmax_backward_triton', mutates_args={'x_grad'})
def softmax_backward_triton(output: torch.Tensor, output_grad: torch.Tensor,
    x_grad: torch.Tensor, logits_multiplier: (float | None)) ->None:
    B, H = get_num_elements_and_hidden_size(x_grad)
    BLOCK_SIZE_B = 1
    BLOCK_SIZE_H = min(get_next_power_of_2(H), 4096 if output.dtype ==
        torch.float32 else 8192)
    with torch.device(x_grad.device):
        softmax_backward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B),](y_ptr
            =output, y_stride=output.stride(), dy_ptr=output_grad,
            dy_stride=output_grad.stride(), dx_ptr=x_grad, dx_stride=x_grad
            .stride(), logits_multiplier=logits_multiplier, B=B, H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B, BLOCK_SIZE_H=BLOCK_SIZE_H)


# Backward method (kernel launch code)
def __Softmax_backward(ctx, output_grad: torch.Tensor) ->tuple[torch.Tensor |
    None]:
    output = ctx.saved_tensors[0]
    x_grad = empty_like_contiguous(output)
    softmax_backward_triton(output=output, output_grad=output_grad, x_grad=
        x_grad, logits_multiplier=ctx.logits_multiplier)
    return x_grad, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _Softmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, logits_multiplier: (float | None)
        ) ->torch.Tensor:
        output = empty_like_contiguous(x)
        softmax_forward_triton(x=x, output=output, logits_multiplier=
            logits_multiplier)
        ctx.save_for_backward(output)
        ctx.logits_multiplier = logits_multiplier
        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) ->tuple[torch.Tensor | None]:
        output = ctx.saved_tensors[0]
        x_grad = empty_like_contiguous(output)
        softmax_backward_triton(output=output, output_grad=output_grad,
            x_grad=x_grad, logits_multiplier=ctx.logits_multiplier)
        return x_grad, None
