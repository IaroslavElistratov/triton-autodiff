# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/open-lm-engine/flash-model-architectures
# Source-Files: fma/functional/fused_residual_add_rmsnorm/__init__.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_r3l3p8y1/flash-model-architectures-main/fma/functional/fused_residual_add_rmsnorm/__init__.py
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
def fused_residual_add_rmsnorm_forward_triton_kernel(x_ptr, x_stride, r_ptr,
    r_stride, W_ptr, W_stride, y_ptr, y_stride, xr_ptr, xr_stride, s_ptr,
    s_stride, eps, multiplier, B, H, BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr):
    BLOCK_ID_B = tl.program_id(axis=0)
    BLOCK_B = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)
    MASK_B = BLOCK_B < B
    MASK_H = BLOCK_H < H
    MASK_BH = MASK_B[:, None] & MASK_H[None, :]
    x = tl.load(x_ptr + BLOCK_B[:, None] * x_stride[0] + BLOCK_H[None, :] *
        x_stride[1], mask=MASK_BH).to(tl.float32)
    if multiplier is not None:
        x *= multiplier
    if r_ptr is not None:
        r = tl.load(r_ptr + BLOCK_B[:, None] * r_stride[0] + BLOCK_H[None,
            :] * r_stride[1], mask=MASK_BH)
        x += r
    if xr_ptr is not None:
        tl.store(xr_ptr + BLOCK_B[:, None] * xr_stride[0] + BLOCK_H[None, :
            ] * xr_stride[1], x, mask=MASK_BH)
    r = tl.sum(x * x, axis=1)
    r = tl.rsqrt(r / H + eps)
    if s_ptr is not None:
        tl.store(s_ptr + BLOCK_B * s_stride[0], r, mask=MASK_B)
    x *= r[:, None]
    if W_ptr is not None:
        W = tl.load(W_ptr + BLOCK_H * W_stride[0], mask=MASK_H)
        x = x.to(x_ptr.dtype.element_ty) * W[None, :]
    tl.store(y_ptr + BLOCK_B[:, None] * y_stride[0] + BLOCK_H[None, :] *
        y_stride[1], x, mask=MASK_BH)


@custom_op(f'{LIBRARY_NAME}::fused_residual_add_rmsnorm_forward_triton',
    mutates_args={'output', 'added_x_residual', 'rmsnorm_denominator'})
def fused_residual_add_rmsnorm_forward_triton(x: torch.Tensor, residual: (
    torch.Tensor | None), weight: (torch.Tensor | None), output: torch.
    Tensor, eps: float, multiplier: (float | None), added_x_residual: (
    torch.Tensor | None), rmsnorm_denominator: (torch.Tensor | None)) ->None:
    B, H = get_num_elements_and_hidden_size(x)
    BLOCK_SIZE_B = 1
    BLOCK_SIZE_H = get_next_power_of_2(H)
    assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE
    NUM_WARPS = 8
    with torch.device(x.device):
        fused_residual_add_rmsnorm_forward_triton_kernel[ceil_divide(B,
            BLOCK_SIZE_B),](x_ptr=x, x_stride=x.stride(), r_ptr=residual,
            r_stride=None if residual is None else residual.stride(), W_ptr
            =weight, W_stride=None if weight is None else weight.stride(),
            y_ptr=output, y_stride=output.stride(), xr_ptr=added_x_residual,
            xr_stride=None if added_x_residual is None else
            added_x_residual.stride(), s_ptr=rmsnorm_denominator, s_stride=
            None if rmsnorm_denominator is None else rmsnorm_denominator.
            stride(), eps=eps, multiplier=multiplier, B=B, H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B, BLOCK_SIZE_H=BLOCK_SIZE_H, num_warps
            =NUM_WARPS)


# Forward method (kernel launch code)
def __FusedResidualAddRMSNorm_forward(ctx, x: torch.Tensor, residual: (
    torch.Tensor | None), weight: (torch.Tensor | None), eps: (float | None
    ), multiplier: (float | None), memory_efficient: bool, deterministic: bool
    ) ->tuple[torch.Tensor, torch.Tensor | None]:
    if eps is None:
        eps = torch.finfo(x.dtype).eps
    B, _ = get_num_elements_and_hidden_size(x)
    has_residual = residual is not None
    output = empty_like_contiguous(x)
    added_x_residual = empty_like_contiguous(x) if has_residual else None
    rmsnorm_denominator = None if memory_efficient else torch.empty(B,
        device=x.device, dtype=torch.float32)
    fused_residual_add_rmsnorm_forward_triton(x=x, residual=residual,
        weight=weight, output=output, eps=eps, multiplier=multiplier,
        added_x_residual=added_x_residual, rmsnorm_denominator=
        rmsnorm_denominator)
    ctx.save_for_backward(added_x_residual if has_residual else x, weight,
        rmsnorm_denominator)
    ctx.eps = eps
    ctx.has_residual = has_residual
    ctx.multiplier = multiplier
    ctx.deterministic = deterministic
    return output, added_x_residual


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def fused_residual_add_rmsnorm_backward_triton_kernel(xr_ptr, xr_stride,
    W_ptr, W_stride, dy_ptr, dy_stride, dxr_ptr, dxr_stride, dx_ptr,
    dx_stride, dr_ptr, dr_stride, dW_ptr, dW_stride, s_ptr, s_stride, eps,
    multiplier, B, H, ATOMIC_ADD: tl.constexpr, BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr):
    BLOCK_ID = tl.program_id(axis=0)
    NUM_BLOCKS = tl.num_programs(axis=0)
    NUM_ELEMENTS_PER_BLOCK = tl.cdiv(B, NUM_BLOCKS)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)
    MASK_H = BLOCK_H < H
    start = BLOCK_ID * NUM_ELEMENTS_PER_BLOCK
    end = min(start + NUM_ELEMENTS_PER_BLOCK, B)
    NUM_ELEMENTS_IN_CURRENT_BLOCK = end - start
    NUM_LOOPS = tl.cdiv(NUM_ELEMENTS_IN_CURRENT_BLOCK, BLOCK_SIZE_B)
    x_dtype = xr_ptr.dtype.element_ty
    if W_ptr is not None:
        W = tl.load(W_ptr + BLOCK_H * W_stride[0], mask=MASK_H)[None, :]
        dW = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)
    for i in range(NUM_LOOPS):
        BLOCK_B = start + i * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        MASK_B = BLOCK_B < end
        MASK_BH = MASK_B[:, None] & MASK_H[None, :]
        xr = tl.load(xr_ptr + BLOCK_B[:, None] * xr_stride[0] + BLOCK_H[
            None, :] * xr_stride[1], mask=MASK_BH).to(tl.float32)
        if s_ptr is None:
            r = tl.sum(xr * xr, axis=1)
            r = tl.rsqrt(r / H + eps)
        else:
            r = tl.load(s_ptr + BLOCK_B * s_stride[0], mask=MASK_B)
        dy = tl.load(dy_ptr + BLOCK_B[:, None] * dy_stride[0] + BLOCK_H[
            None, :] * dy_stride[1], mask=MASK_BH)
        dyW = dy
        if W_ptr is not None:
            dyW *= W
        dyW = dyW.to(tl.float32)
        dx = r[:, None] * dyW
        dx -= 1 / H * r[:, None] * r[:, None] * r[:, None] * xr * tl.sum(
            dyW * xr, axis=1, keep_dims=True)
        dx = dx.to(x_dtype)
        if dxr_ptr is not None:
            dx += tl.load(dxr_ptr + BLOCK_B[:, None] * dxr_stride[0] + 
                BLOCK_H[None, :] * dxr_stride[1], mask=MASK_BH)
        if dr_ptr is not None:
            tl.store(dr_ptr + BLOCK_B[:, None] * dr_stride[0] + BLOCK_H[
                None, :] * dr_stride[1], dx, mask=MASK_BH)
        if multiplier is not None:
            dx *= multiplier
        tl.store(dx_ptr + BLOCK_B[:, None] * dx_stride[0] + BLOCK_H[None, :
            ] * dx_stride[1], dx, mask=MASK_BH)
        if W_ptr is not None:
            dW += tl.sum(dy * (xr * r[:, None]).to(x_dtype), axis=0)
    if W_ptr is not None:
        if ATOMIC_ADD:
            tl.atomic_add(dW_ptr + BLOCK_H * dW_stride[0], dW, mask=MASK_H,
                sem='relaxed')
        else:
            tl.store(dW_ptr + BLOCK_ID * dW_stride[0] + BLOCK_H * dW_stride
                [1], dW, mask=MASK_H)


@custom_op(f'{LIBRARY_NAME}::fused_residual_add_rmsnorm_backward_triton',
    mutates_args={'x_grad', 'residual_grad', 'weight_grad'})
def fused_residual_add_rmsnorm_backward_triton(added_x_residual: torch.
    Tensor, weight: (torch.Tensor | None), output_grad: torch.Tensor,
    added_x_residual_grad: (torch.Tensor | None), rmsnorm_denominator: (
    torch.Tensor | None), x_grad: torch.Tensor, residual_grad: (torch.
    Tensor | None), weight_grad: (torch.Tensor | None), eps: float,
    multiplier: (float | None), deterministic: bool) ->None:
    B, H = get_num_elements_and_hidden_size(added_x_residual)
    BLOCK_SIZE_B = 1
    BLOCK_SIZE_H = get_next_power_of_2(H)
    assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE
    NUM_WARPS = 8
    sm_count = get_sm_count(added_x_residual.device)
    NUM_BLOCKS = min(sm_count, ceil_divide(B, BLOCK_SIZE_B))
    with torch.device(added_x_residual.device):
        fused_residual_add_rmsnorm_backward_triton_kernel[NUM_BLOCKS,](xr_ptr
            =added_x_residual, xr_stride=None if added_x_residual is None else
            added_x_residual.stride(), W_ptr=weight, W_stride=None if 
            weight is None else weight.stride(), dy_ptr=output_grad,
            dy_stride=output_grad.stride(), dxr_ptr=added_x_residual_grad,
            dxr_stride=None if added_x_residual_grad is None else
            added_x_residual_grad.stride(), dx_ptr=x_grad, dx_stride=x_grad
            .stride(), dr_ptr=residual_grad, dr_stride=None if 
            residual_grad is None else residual_grad.stride(), dW_ptr=
            weight_grad, dW_stride=None if weight_grad is None else
            weight_grad.stride(), s_ptr=rmsnorm_denominator, s_stride=None if
            rmsnorm_denominator is None else rmsnorm_denominator.stride(),
            eps=eps, multiplier=multiplier, B=B, H=H, ATOMIC_ADD=not
            deterministic, BLOCK_SIZE_B=BLOCK_SIZE_B, BLOCK_SIZE_H=
            BLOCK_SIZE_H, num_warps=NUM_WARPS)


def get_sm_count(device: torch.device) ->int:
    if device.type == 'cuda':
        sm_count = torch.cuda.get_device_properties(device
            ).multi_processor_count
    elif device.type == 'xpu':
        sm_count = torch.xpu.get_device_properties(device).gpu_subslice_count
    return sm_count


def zeros_like_contiguous(x: torch.Tensor, dtype: (torch.dtype | None)=None
    ) ->torch.Tensor:
    return torch.zeros_like(x, dtype=dtype, memory_format=torch.
        contiguous_format)


# Backward method (kernel launch code)
def __FusedResidualAddRMSNorm_backward(ctx, output_grad: torch.Tensor,
    added_x_residual_grad: torch.Tensor) ->tuple[torch.Tensor | None]:
    has_residual = ctx.has_residual
    deterministic = ctx.deterministic
    added_x_residual, weight, rmsnorm_denominator = ctx.saved_tensors
    x_grad = empty_like_contiguous(added_x_residual)
    residual_grad = empty_like_contiguous(added_x_residual
        ) if has_residual else None
    if weight is not None:
        if deterministic:
            weight_grad = torch.empty(get_sm_count(x_grad.device), *weight.
                size(), dtype=weight.dtype, device=weight.device)
        else:
            weight_grad = zeros_like_contiguous(weight, dtype=torch.float32)
    else:
        weight_grad = None
    if not has_residual:
        assert added_x_residual_grad is None
    fused_residual_add_rmsnorm_backward_triton(added_x_residual=
        added_x_residual, weight=weight, output_grad=output_grad,
        added_x_residual_grad=added_x_residual_grad, rmsnorm_denominator=
        rmsnorm_denominator, x_grad=x_grad, residual_grad=residual_grad,
        weight_grad=weight_grad, eps=ctx.eps, multiplier=ctx.multiplier,
        deterministic=deterministic)
    if weight_grad is not None:
        if deterministic:
            weight_grad = weight_grad.sum(0)
        else:
            weight_grad = weight_grad.type_as(weight)
    return x_grad, residual_grad, weight_grad, *([None] * 4)


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _FusedResidualAddRMSNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, residual: (torch.Tensor | None),
        weight: (torch.Tensor | None), eps: (float | None), multiplier: (
        float | None), memory_efficient: bool, deterministic: bool) ->tuple[
        torch.Tensor, torch.Tensor | None]:
        if eps is None:
            eps = torch.finfo(x.dtype).eps
        B, _ = get_num_elements_and_hidden_size(x)
        has_residual = residual is not None
        output = empty_like_contiguous(x)
        added_x_residual = empty_like_contiguous(x) if has_residual else None
        rmsnorm_denominator = None if memory_efficient else torch.empty(B,
            device=x.device, dtype=torch.float32)
        fused_residual_add_rmsnorm_forward_triton(x=x, residual=residual,
            weight=weight, output=output, eps=eps, multiplier=multiplier,
            added_x_residual=added_x_residual, rmsnorm_denominator=
            rmsnorm_denominator)
        ctx.save_for_backward(added_x_residual if has_residual else x,
            weight, rmsnorm_denominator)
        ctx.eps = eps
        ctx.has_residual = has_residual
        ctx.multiplier = multiplier
        ctx.deterministic = deterministic
        return output, added_x_residual

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor, added_x_residual_grad:
        torch.Tensor) ->tuple[torch.Tensor | None]:
        has_residual = ctx.has_residual
        deterministic = ctx.deterministic
        added_x_residual, weight, rmsnorm_denominator = ctx.saved_tensors
        x_grad = empty_like_contiguous(added_x_residual)
        residual_grad = empty_like_contiguous(added_x_residual
            ) if has_residual else None
        if weight is not None:
            if deterministic:
                weight_grad = torch.empty(get_sm_count(x_grad.device), *
                    weight.size(), dtype=weight.dtype, device=weight.device)
            else:
                weight_grad = zeros_like_contiguous(weight, dtype=torch.float32
                    )
        else:
            weight_grad = None
        if not has_residual:
            assert added_x_residual_grad is None
        fused_residual_add_rmsnorm_backward_triton(added_x_residual=
            added_x_residual, weight=weight, output_grad=output_grad,
            added_x_residual_grad=added_x_residual_grad,
            rmsnorm_denominator=rmsnorm_denominator, x_grad=x_grad,
            residual_grad=residual_grad, weight_grad=weight_grad, eps=ctx.
            eps, multiplier=ctx.multiplier, deterministic=deterministic)
        if weight_grad is not None:
            if deterministic:
                weight_grad = weight_grad.sum(0)
            else:
                weight_grad = weight_grad.type_as(weight)
        return x_grad, residual_grad, weight_grad, *([None] * 4)
