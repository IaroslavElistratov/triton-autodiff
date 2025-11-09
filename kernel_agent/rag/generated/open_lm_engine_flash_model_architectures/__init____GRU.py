# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/open-lm-engine/flash-model-architectures
# Source-Files: fma/functional/gru/__init__.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_r3l3p8y1/flash-model-architectures-main/fma/functional/gru/__init__.py
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


@triton.jit
def matmul(A, B, C, output_dtype: tl.constexpr):
    if A.shape[0] == 1:
        x = tl.sum(A.T * B, axis=0, keep_dims=True)
        if C is not None:
            x += C
        x = x.to(output_dtype)
    elif A.shape[1] == 1:
        x = A * B
        if C is not None:
            x += C
        x = x.to(output_dtype)
    elif B.shape[1] == 1:
        x = tl.sum(A * B.T, axis=1, keep_dims=True)
        if C is not None:
            x += C
        x = x.to(output_dtype)
    elif C is None:
        if output_dtype == tl.bfloat16:
            x = tl.dot(A, B, out_dtype=tl.float32).to(output_dtype)
        else:
            x = tl.dot(A, B, out_dtype=output_dtype)
    elif C.shape[0] == 1 or C.shape[1] == 1:
        x = tl.dot(A, B, out_dtype=tl.float32)
        x += C
        x = x.to(output_dtype)
    else:
        x = tl.dot(A, B, C.to(tl.float32), out_dtype=tl.float32).to(
            output_dtype)
    return x


def empty_like_contiguous(x: torch.Tensor, dtype: (torch.dtype | None)=None
    ) ->torch.Tensor:
    return torch.empty_like(x, dtype=dtype, memory_format=torch.
        contiguous_format)


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

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


@triton.autotune(configs=_get_autotune_configs(), key=['BLOCK_SIZE_H'])
@triton.jit
def gru_forward_triton_kernel(x_ptr, x_stride, xf_ptr, xf_stride, xr_ptr,
    xr_stride, W_ptr, W_stride, Wf_ptr, Wf_stride, Wr_ptr, Wr_stride, z_ptr,
    z_stride, f_ptr, f_stride, r_ptr, r_stride, h0_ptr, h0_stride, y_ptr,
    y_stride, cu_seqlens_ptr, cu_seqlens_stride, IS_MAX_SEQLEN_TENSOR: tl.
    constexpr, max_seqlen_ptr, B, S, H, BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr):
    BLOCK_ID_B = tl.program_id(axis=0)
    BLOCK_ID_N = tl.program_id(axis=1)
    BLOCK_B = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)
    MASK_B = BLOCK_B < B
    MASK_H = BLOCK_H < H
    MASK_BH = MASK_B[:, None] & MASK_H[None, :]
    MASK_HH = MASK_H[:, None] & MASK_H[None, :]
    W = tl.load(W_ptr + BLOCK_ID_N * W_stride[0] + BLOCK_H[:, None] *
        W_stride[1] + BLOCK_H[None, :] * W_stride[2], mask=MASK_HH)
    Wf = tl.load(Wf_ptr + BLOCK_ID_N * Wf_stride[0] + BLOCK_H[:, None] *
        Wf_stride[1] + BLOCK_H[None, :] * Wf_stride[2], mask=MASK_HH)
    Wr = tl.load(Wr_ptr + BLOCK_ID_N * Wr_stride[0] + BLOCK_H[:, None] *
        Wr_stride[1] + BLOCK_H[None, :] * Wr_stride[2], mask=MASK_HH)
    if h0_ptr is None:
        h = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=x_ptr.dtype.element_ty
            )
    else:
        h = tl.load(h0_ptr + BLOCK_B[:, None] * h0_stride[0] + BLOCK_ID_N *
            h0_stride[1] + BLOCK_H[None, :] * h0_stride[2], mask=MASK_BH)
    IS_VARLEN: tl.constexpr = cu_seqlens_ptr is not None
    if IS_VARLEN:
        cu_seqlens_ptrs = cu_seqlens_ptr + BLOCK_B[:, None
            ] * cu_seqlens_stride[0]
        start = tl.load(cu_seqlens_ptrs, mask=MASK_B[:, None])
        end = tl.load(cu_seqlens_ptrs + cu_seqlens_stride[0], mask=MASK_B[:,
            None])
        if IS_MAX_SEQLEN_TENSOR:
            S = tl.load(max_seqlen_ptr)
        else:
            S = max_seqlen_ptr
        x_ptrs = x_ptr + start * x_stride[0] + BLOCK_ID_N * x_stride[1
            ] + BLOCK_H[None, :] * x_stride[2]
        xr_ptrs = xr_ptr + start * xr_stride[0] + BLOCK_ID_N * xr_stride[1
            ] + BLOCK_H[None, :] * xr_stride[2]
        xf_ptrs = xf_ptr + start * xf_stride[0] + BLOCK_ID_N * xf_stride[1
            ] + BLOCK_H[None, :] * xf_stride[2]
        z_ptrs = z_ptr + start * z_stride[0] + BLOCK_ID_N * z_stride[1
            ] + BLOCK_H[None, :] * z_stride[2]
        r_ptrs = r_ptr + start * r_stride[0] + BLOCK_ID_N * r_stride[1
            ] + BLOCK_H[None, :] * r_stride[2]
        f_ptrs = f_ptr + start * f_stride[0] + BLOCK_ID_N * f_stride[1
            ] + BLOCK_H[None, :] * f_stride[2]
        y_ptrs = y_ptr + start * y_stride[0] + BLOCK_ID_N * y_stride[1
            ] + BLOCK_H[None, :] * y_stride[2]
    else:
        x_ptrs = x_ptr + BLOCK_B[:, None] * x_stride[0
            ] + BLOCK_ID_N * x_stride[2] + BLOCK_H[None, :] * x_stride[3]
        xr_ptrs = xr_ptr + BLOCK_B[:, None] * xr_stride[0
            ] + BLOCK_ID_N * xr_stride[2] + BLOCK_H[None, :] * xr_stride[3]
        xf_ptrs = xf_ptr + BLOCK_B[:, None] * xf_stride[0
            ] + BLOCK_ID_N * xf_stride[2] + BLOCK_H[None, :] * xf_stride[3]
        z_ptrs = z_ptr + BLOCK_B[:, None] * z_stride[0
            ] + BLOCK_ID_N * z_stride[2] + BLOCK_H[None, :] * z_stride[3]
        r_ptrs = r_ptr + BLOCK_B[:, None] * r_stride[0
            ] + BLOCK_ID_N * r_stride[2] + BLOCK_H[None, :] * r_stride[3]
        f_ptrs = f_ptr + BLOCK_B[:, None] * f_stride[0
            ] + BLOCK_ID_N * f_stride[2] + BLOCK_H[None, :] * f_stride[3]
        y_ptrs = y_ptr + BLOCK_B[:, None] * y_stride[0
            ] + BLOCK_ID_N * y_stride[2] + BLOCK_H[None, :] * y_stride[3]
    for _ in range(S):
        if IS_VARLEN:
            MASK = (start < end) & MASK_H[None, :]
        else:
            MASK = MASK_BH
        x = tl.load(xr_ptrs, mask=MASK)
        r = matmul(A=h, B=Wr, C=x, output_dtype=tl.float32)
        r = sigmoid(r, output_dtype=x.dtype)
        tl.store(r_ptrs, r, mask=MASK)
        x = tl.load(x_ptrs, mask=MASK)
        z = matmul(A=h * r, B=W, C=x, output_dtype=tl.float32)
        z = tanh(z, output_dtype=x.dtype)
        tl.store(z_ptrs, z, mask=MASK)
        x = tl.load(xf_ptrs, mask=MASK)
        f = matmul(A=h, B=Wf, C=x, output_dtype=tl.float32)
        f = sigmoid(f, output_dtype=x.dtype)
        tl.store(f_ptrs, f, mask=MASK)
        h = f * h + (1 - f) * z
        tl.store(y_ptrs, h, mask=MASK)
        x_ptrs += x_stride[1 - IS_VARLEN]
        xr_ptrs += xr_stride[1 - IS_VARLEN]
        xf_ptrs += xf_stride[1 - IS_VARLEN]
        z_ptrs += z_stride[1 - IS_VARLEN]
        r_ptrs += r_stride[1 - IS_VARLEN]
        f_ptrs += f_stride[1 - IS_VARLEN]
        y_ptrs += y_stride[1 - IS_VARLEN]
        if IS_VARLEN:
            start += 1


@custom_op(f'{LIBRARY_NAME}::gru_forward_triton', mutates_args={
    'forget_gate', 'reset_gate', 'output_update', 'output'})
def gru_forward_triton(input: torch.Tensor, weight: torch.Tensor,
    forget_input: torch.Tensor, forget_weight: torch.Tensor, forget_gate:
    torch.Tensor, reset_input: torch.Tensor, reset_weight: torch.Tensor,
    reset_gate: torch.Tensor, output_update: torch.Tensor, input_state: (
    torch.Tensor | None), output: torch.Tensor, cu_seqlens: (torch.Tensor |
    None), max_seqlen_tensor: (torch.Tensor | None), max_seqlen: (int | None)
    ) ->None:
    if cu_seqlens is None:
        assert max_seqlen is None
        assert max_seqlen_tensor is None
        B, S, N, H = input.size()
    else:
        B = cu_seqlens.size(0) - 1
        S = None
        _, N, H = input.size()
    is_max_seqlen_tensor = max_seqlen_tensor is not None
    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)
    GRID = lambda meta: (ceil_divide(B, meta['BLOCK_SIZE_B']), N)
    with torch.device(input.device):
        gru_forward_triton_kernel[GRID](x_ptr=input, x_stride=input.stride(
            ), xf_ptr=forget_input, xf_stride=forget_input.stride(), xr_ptr
            =reset_input, xr_stride=reset_input.stride(), W_ptr=weight,
            W_stride=weight.stride(), Wf_ptr=forget_weight, Wf_stride=
            forget_weight.stride(), Wr_ptr=reset_weight, Wr_stride=
            reset_weight.stride(), z_ptr=output_update, z_stride=
            output_update.stride(), f_ptr=forget_gate, f_stride=forget_gate
            .stride(), r_ptr=reset_gate, r_stride=reset_gate.stride(),
            h0_ptr=input_state, h0_stride=None if input_state is None else
            input_state.stride(), y_ptr=output, y_stride=output.stride(),
            cu_seqlens_ptr=cu_seqlens, cu_seqlens_stride=None if cu_seqlens is
            None else cu_seqlens.stride(), IS_MAX_SEQLEN_TENSOR=
            is_max_seqlen_tensor, max_seqlen_ptr=max_seqlen_tensor if
            is_max_seqlen_tensor else max_seqlen, B=B, S=S, H=H,
            BLOCK_SIZE_H=BLOCK_SIZE_H)


def get_max_seqlen_and_max_seqlen_tensor(max_seqlen: (torch.Tensor | int |
    None)) ->tuple[torch.Tensor | None, int | None]:
    if isinstance(max_seqlen, torch.Tensor):
        return max_seqlen, None
    else:
        return None, max_seqlen


# Forward method (kernel launch code)
def __GRU_forward(ctx, input: torch.Tensor, weight: torch.Tensor,
    forget_input: torch.Tensor, forget_weight: torch.Tensor, reset_input:
    torch.Tensor, reset_weight: torch.Tensor, input_state: (torch.Tensor |
    None), gradient_clipping: (float | None), cu_seqlens: (torch.Tensor |
    None), max_seqlen: (torch.Tensor | int | None)) ->torch.Tensor:
    output = empty_like_contiguous(input)
    forget_gate = empty_like_contiguous(input)
    reset_gate = empty_like_contiguous(input)
    output_update = empty_like_contiguous(input)
    max_seqlen_tensor, max_seqlen = get_max_seqlen_and_max_seqlen_tensor(
        max_seqlen)
    gru_forward_triton(input=input, weight=weight, forget_input=
        forget_input, forget_weight=forget_weight, forget_gate=forget_gate,
        reset_input=reset_input, reset_weight=reset_weight, reset_gate=
        reset_gate, output_update=output_update, input_state=input_state,
        output=output, cu_seqlens=cu_seqlens, max_seqlen_tensor=
        max_seqlen_tensor, max_seqlen=max_seqlen)
    ctx.save_for_backward(weight, forget_weight, forget_gate, reset_weight,
        reset_gate, output_update, output, input_state, cu_seqlens,
        max_seqlen_tensor)
    ctx.max_seqlen = max_seqlen
    ctx.gradient_clipping = gradient_clipping
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def clamp(x, min_value, max_value):
    dtype = x.dtype
    x = max(min_value, x)
    x = min(max_value, x)
    x = x.to(dtype)
    return x


@triton.jit
def sigmoid_backward(y):
    dtype = y.dtype
    y = y.to(tl.float32)
    y = y * (1 - y)
    y = y.to(dtype)
    return y


@triton.jit
def tanh_backward(y):
    dtype = y.dtype
    y = y.to(tl.float32)
    y = 1 - y * y
    y = y.to(dtype)
    return y


@triton.autotune(configs=_get_autotune_configs(), key=['BLOCK_SIZE_H'],
    reset_to_zero=['dW_ptr', 'dWf_ptr', 'dWr_ptr'])
@triton.jit
def gru_backward_triton_kernel(W_ptr, W_stride, Wf_ptr, Wf_stride, Wr_ptr,
    Wr_stride, z_ptr, z_stride, f_ptr, f_stride, r_ptr, r_stride, h0_ptr,
    h0_stride, y_ptr, y_stride, dx_ptr, dx_stride, dxf_ptr, dxf_stride,
    dxr_ptr, dxr_stride, dW_ptr, dW_stride, dWf_ptr, dWf_stride, dWr_ptr,
    dWr_stride, dy_ptr, dy_stride, cu_seqlens_ptr, cu_seqlens_stride,
    IS_MAX_SEQLEN_TENSOR: tl.constexpr, max_seqlen_ptr, B, S, H,
    gradient_clipping, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_H: tl.constexpr):
    BLOCK_ID_B = tl.program_id(axis=0)
    BLOCK_ID_N = tl.program_id(axis=1)
    BLOCK_B = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)
    MASK_B = BLOCK_B < B
    MASK_H = BLOCK_H < H
    MASK_BH = MASK_B[:, None] & MASK_H[None, :]
    MASK_HH = MASK_H[:, None] & MASK_H[None, :]
    dh = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=W_ptr.dtype.element_ty)
    dW = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_H), dtype=tl.float32)
    dWf = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_H), dtype=tl.float32)
    dWr = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_H), dtype=tl.float32)
    W = tl.load(W_ptr + BLOCK_ID_N * W_stride[0] + BLOCK_H[:, None] *
        W_stride[1] + BLOCK_H[None, :] * W_stride[2], mask=MASK_HH)
    Wf = tl.load(Wf_ptr + BLOCK_ID_N * Wf_stride[0] + BLOCK_H[:, None] *
        Wf_stride[1] + BLOCK_H[None, :] * Wf_stride[2], mask=MASK_HH)
    Wr = tl.load(Wr_ptr + BLOCK_ID_N * Wr_stride[0] + BLOCK_H[:, None] *
        Wr_stride[1] + BLOCK_H[None, :] * Wr_stride[2], mask=MASK_HH)
    IS_VARLEN: tl.constexpr = cu_seqlens_ptr is not None
    if IS_VARLEN:
        cu_seqlens_ptrs = cu_seqlens_ptr + BLOCK_B[:, None
            ] * cu_seqlens_stride[0]
        start = tl.load(cu_seqlens_ptrs, mask=MASK_B[:, None])
        end = tl.load(cu_seqlens_ptrs + cu_seqlens_stride[0], mask=MASK_B[:,
            None])
        if IS_MAX_SEQLEN_TENSOR:
            S = tl.load(max_seqlen_ptr)
        else:
            S = max_seqlen_ptr
        end -= 1
        z_ptrs = z_ptr + end * z_stride[0] + BLOCK_ID_N * z_stride[1
            ] + BLOCK_H[None, :] * z_stride[2]
        f_ptrs = f_ptr + end * f_stride[0] + BLOCK_ID_N * f_stride[1
            ] + BLOCK_H[None, :] * f_stride[2]
        r_ptrs = r_ptr + end * r_stride[0] + BLOCK_ID_N * r_stride[1
            ] + BLOCK_H[None, :] * r_stride[2]
        y_ptrs = y_ptr + end * y_stride[0] + BLOCK_ID_N * y_stride[1
            ] + BLOCK_H[None, :] * y_stride[2]
        dx_ptrs = dx_ptr + end * dx_stride[0] + BLOCK_ID_N * dx_stride[1
            ] + BLOCK_H[None, :] * dx_stride[2]
        dxf_ptrs = dxf_ptr + end * dxf_stride[0] + BLOCK_ID_N * dxf_stride[1
            ] + BLOCK_H[None, :] * dxf_stride[2]
        dxr_ptrs = dxr_ptr + end * dxr_stride[0] + BLOCK_ID_N * dxr_stride[1
            ] + BLOCK_H[None, :] * dxr_stride[2]
        dy_ptrs = dy_ptr + end * dy_stride[0] + BLOCK_ID_N * dy_stride[1
            ] + BLOCK_H[None, :] * dy_stride[2]
    else:
        z_ptrs = z_ptr + BLOCK_B[:, None] * z_stride[0] + (S - 1) * z_stride[1
            ] + BLOCK_ID_N * z_stride[2] + BLOCK_H[None, :] * z_stride[3]
        f_ptrs = f_ptr + BLOCK_B[:, None] * f_stride[0] + (S - 1) * f_stride[1
            ] + BLOCK_ID_N * f_stride[2] + BLOCK_H[None, :] * f_stride[3]
        r_ptrs = r_ptr + BLOCK_B[:, None] * r_stride[0] + (S - 1) * r_stride[1
            ] + BLOCK_ID_N * r_stride[2] + BLOCK_H[None, :] * r_stride[3]
        y_ptrs = y_ptr + BLOCK_B[:, None] * y_stride[0] + (S - 1) * y_stride[1
            ] + BLOCK_ID_N * y_stride[2] + BLOCK_H[None, :] * y_stride[3]
        dx_ptrs = dx_ptr + BLOCK_B[:, None] * dx_stride[0] + (S - 1
            ) * dx_stride[1] + BLOCK_ID_N * dx_stride[2] + BLOCK_H[None, :
            ] * dx_stride[3]
        dxf_ptrs = dxf_ptr + BLOCK_B[:, None] * dxf_stride[0] + (S - 1
            ) * dxf_stride[1] + BLOCK_ID_N * dxf_stride[2] + BLOCK_H[None, :
            ] * dxf_stride[3]
        dxr_ptrs = dxr_ptr + BLOCK_B[:, None] * dxr_stride[0] + (S - 1
            ) * dxr_stride[1] + BLOCK_ID_N * dxr_stride[2] + BLOCK_H[None, :
            ] * dxr_stride[3]
        dy_ptrs = dy_ptr + BLOCK_B[:, None] * dy_stride[0] + (S - 1
            ) * dy_stride[1] + BLOCK_ID_N * dy_stride[2] + BLOCK_H[None, :
            ] * dy_stride[3]
    for s in range(S - 1, -1, -1):
        if gradient_clipping is not None:
            dh = clamp(dh, min_value=-gradient_clipping, max_value=
                gradient_clipping)
        if IS_VARLEN:
            MASK = (end >= start) & MASK_H[None, :]
        else:
            MASK = MASK_BH
        dy = tl.load(dy_ptrs, mask=MASK) + dh
        z = tl.load(z_ptrs, mask=MASK)
        f = tl.load(f_ptrs, mask=MASK)
        r = tl.load(r_ptrs, mask=MASK)
        y_ptrs -= y_stride[1 - IS_VARLEN]
        if IS_VARLEN:
            y_prev = tl.where(start == end, _load_input_state(h0_ptr=h0_ptr,
                h0_stride=h0_stride, BLOCK_ID_N=BLOCK_ID_N, BLOCK_B=BLOCK_B,
                BLOCK_H=BLOCK_H, MASK_BH=MASK_BH, BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H, dtype=W.dtype), tl.load(y_ptrs,
                mask=MASK))
        elif s == 0:
            if h0_ptr is None:
                y_prev = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=W.dtype)
            else:
                y_prev = tl.load(h0_ptr + BLOCK_B[:, None] * h0_stride[0] +
                    BLOCK_ID_N * h0_stride[1] + BLOCK_H[None, :] *
                    h0_stride[2], mask=MASK)
        else:
            y_prev = tl.load(y_ptrs, mask=MASK)
        dh = f * dy
        dz = dy * (1 - f)
        df = dy * (y_prev - z)
        dx = dz * tanh_backward(z)
        drh = matmul(A=dx, B=W.T, C=None, output_dtype=dx.dtype)
        dW = matmul(A=(r * y_prev).T, B=dx, C=dW, output_dtype=dW.dtype)
        tl.store(dx_ptrs, dx, mask=MASK)
        dh += drh * r
        dxf = df * sigmoid_backward(f)
        dh = matmul(A=dxf, B=Wf.T, C=dh, output_dtype=dx.dtype)
        dWf = matmul(A=y_prev.T, B=dxf, C=dWf, output_dtype=dW.dtype)
        tl.store(dxf_ptrs, dxf, mask=MASK)
        dxr = drh * y_prev * sigmoid_backward(r)
        dh = matmul(A=dxr, B=Wr.T, C=dh, output_dtype=dx.dtype)
        dWr = matmul(A=y_prev.T, B=dxr, C=dWr, output_dtype=dW.dtype)
        tl.store(dxr_ptrs, dxr, mask=MASK)
        z_ptrs -= z_stride[1 - IS_VARLEN]
        f_ptrs -= f_stride[1 - IS_VARLEN]
        r_ptrs -= r_stride[1 - IS_VARLEN]
        dx_ptrs -= dx_stride[1 - IS_VARLEN]
        dxf_ptrs -= dxf_stride[1 - IS_VARLEN]
        dxr_ptrs -= dxr_stride[1 - IS_VARLEN]
        dy_ptrs -= dy_stride[1 - IS_VARLEN]
        if IS_VARLEN:
            end -= 1
    tl.atomic_add(dW_ptr + BLOCK_ID_N * dW_stride[0] + BLOCK_H[:, None] *
        dW_stride[1] + BLOCK_H[None, :] * dW_stride[2], dW, mask=MASK_HH,
        sem='relaxed')
    tl.atomic_add(dWf_ptr + BLOCK_ID_N * dWf_stride[0] + BLOCK_H[:, None] *
        dWf_stride[1] + BLOCK_H[None, :] * dWf_stride[2], dWf, mask=MASK_HH,
        sem='relaxed')
    tl.atomic_add(dWr_ptr + BLOCK_ID_N * dWr_stride[0] + BLOCK_H[:, None] *
        dWr_stride[1] + BLOCK_H[None, :] * dWr_stride[2], dWr, mask=MASK_HH,
        sem='relaxed')


@triton.jit
def _load_input_state(h0_ptr, h0_stride, BLOCK_ID_N, BLOCK_B, BLOCK_H,
    MASK_BH, BLOCK_SIZE_B, BLOCK_SIZE_H, dtype):
    if h0_ptr is None:
        y_prev = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=dtype)
    else:
        y_ptrs = h0_ptr + BLOCK_B[:, None] * h0_stride[0
            ] + BLOCK_ID_N * h0_stride[1] + BLOCK_H[None, :] * h0_stride[2]
        y_prev = tl.load(y_ptrs, mask=MASK_BH)
    return y_prev


@custom_op(f'{LIBRARY_NAME}::gru_backward_triton', mutates_args={
    'forget_input_grad', 'forget_weight_grad', 'reset_input_grad',
    'reset_weight_grad', 'input_grad', 'weight_grad'})
def gru_backward_triton(weight: torch.Tensor, output: torch.Tensor,
    forget_weight: torch.Tensor, forget_gate: torch.Tensor,
    forget_input_grad: torch.Tensor, forget_weight_grad: torch.Tensor,
    reset_weight: torch.Tensor, reset_gate: torch.Tensor, reset_input_grad:
    torch.Tensor, reset_weight_grad: torch.Tensor, output_update: torch.
    Tensor, input_state: (torch.Tensor | None), output_grad: torch.Tensor,
    input_grad: torch.Tensor, weight_grad: torch.Tensor, cu_seqlens: (torch
    .Tensor | None), max_seqlen_tensor: (torch.Tensor | None), max_seqlen:
    (int | None), gradient_clipping: (float | None)) ->None:
    if cu_seqlens is None:
        assert max_seqlen is None
        assert max_seqlen_tensor is None
        B, S, N, H = output.size()
    else:
        B = cu_seqlens.size(0) - 1
        S = None
        _, N, H = output.size()
    is_max_seqlen_tensor = max_seqlen_tensor is not None
    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)
    GRID = lambda meta: (ceil_divide(B, meta['BLOCK_SIZE_B']), N)
    with torch.device(output.device):
        gru_backward_triton_kernel[GRID](W_ptr=weight, W_stride=weight.
            stride(), Wf_ptr=forget_weight, Wf_stride=forget_weight.stride(
            ), Wr_ptr=reset_weight, Wr_stride=reset_weight.stride(), z_ptr=
            output_update, z_stride=output_update.stride(), f_ptr=
            forget_gate, f_stride=forget_gate.stride(), r_ptr=reset_gate,
            r_stride=reset_gate.stride(), h0_ptr=input_state, h0_stride=
            None if input_state is None else input_state.stride(), y_ptr=
            output, y_stride=output.stride(), dx_ptr=input_grad, dx_stride=
            input_grad.stride(), dxf_ptr=forget_input_grad, dxf_stride=
            forget_input_grad.stride(), dxr_ptr=reset_input_grad,
            dxr_stride=reset_input_grad.stride(), dW_ptr=weight_grad,
            dW_stride=weight_grad.stride(), dWf_ptr=forget_weight_grad,
            dWf_stride=forget_weight_grad.stride(), dWr_ptr=
            reset_weight_grad, dWr_stride=reset_weight_grad.stride(),
            dy_ptr=output_grad, dy_stride=output_grad.stride(),
            cu_seqlens_ptr=cu_seqlens, cu_seqlens_stride=None if cu_seqlens is
            None else cu_seqlens.stride(), IS_MAX_SEQLEN_TENSOR=
            is_max_seqlen_tensor, max_seqlen_ptr=max_seqlen_tensor if
            is_max_seqlen_tensor else max_seqlen, B=B, S=S, H=H,
            gradient_clipping=gradient_clipping, BLOCK_SIZE_H=BLOCK_SIZE_H)


def zeros_like_contiguous(x: torch.Tensor, dtype: (torch.dtype | None)=None
    ) ->torch.Tensor:
    return torch.zeros_like(x, dtype=dtype, memory_format=torch.
        contiguous_format)


# Backward method (kernel launch code)
def __GRU_backward(ctx, output_grad: torch.Tensor) ->tuple[torch.Tensor | None
    ]:
    (weight, forget_weight, forget_gate, reset_weight, reset_gate,
        output_update, output, input_state, cu_seqlens, max_seqlen_tensor
        ) = ctx.saved_tensors
    input_grad = empty_like_contiguous(output)
    forget_input_grad = empty_like_contiguous(output)
    reset_input_grad = empty_like_contiguous(output)
    weight_grad = zeros_like_contiguous(weight, dtype=torch.float32)
    forget_weight_grad = zeros_like_contiguous(weight, dtype=torch.float32)
    reset_weight_grad = zeros_like_contiguous(weight, dtype=torch.float32)
    gru_backward_triton(weight=weight, output=output, forget_weight=
        forget_weight, forget_gate=forget_gate, forget_input_grad=
        forget_input_grad, forget_weight_grad=forget_weight_grad,
        reset_weight=reset_weight, reset_gate=reset_gate, reset_input_grad=
        reset_input_grad, reset_weight_grad=reset_weight_grad,
        output_update=output_update, input_state=input_state, output_grad=
        output_grad, input_grad=input_grad, weight_grad=weight_grad,
        cu_seqlens=cu_seqlens, max_seqlen_tensor=max_seqlen_tensor,
        max_seqlen=ctx.max_seqlen, gradient_clipping=ctx.gradient_clipping)
    weight_grad = weight_grad.type_as(weight)
    forget_weight_grad = forget_weight_grad.type_as(forget_weight)
    reset_weight_grad = reset_weight_grad.type_as(reset_weight)
    return (input_grad, weight_grad, forget_input_grad, forget_weight_grad,
        reset_input_grad, reset_weight_grad, *([None] * 5))


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _GRU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor,
        forget_input: torch.Tensor, forget_weight: torch.Tensor,
        reset_input: torch.Tensor, reset_weight: torch.Tensor, input_state:
        (torch.Tensor | None), gradient_clipping: (float | None),
        cu_seqlens: (torch.Tensor | None), max_seqlen: (torch.Tensor | int |
        None)) ->torch.Tensor:
        output = empty_like_contiguous(input)
        forget_gate = empty_like_contiguous(input)
        reset_gate = empty_like_contiguous(input)
        output_update = empty_like_contiguous(input)
        max_seqlen_tensor, max_seqlen = get_max_seqlen_and_max_seqlen_tensor(
            max_seqlen)
        gru_forward_triton(input=input, weight=weight, forget_input=
            forget_input, forget_weight=forget_weight, forget_gate=
            forget_gate, reset_input=reset_input, reset_weight=reset_weight,
            reset_gate=reset_gate, output_update=output_update, input_state
            =input_state, output=output, cu_seqlens=cu_seqlens,
            max_seqlen_tensor=max_seqlen_tensor, max_seqlen=max_seqlen)
        ctx.save_for_backward(weight, forget_weight, forget_gate,
            reset_weight, reset_gate, output_update, output, input_state,
            cu_seqlens, max_seqlen_tensor)
        ctx.max_seqlen = max_seqlen
        ctx.gradient_clipping = gradient_clipping
        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) ->tuple[torch.Tensor | None]:
        (weight, forget_weight, forget_gate, reset_weight, reset_gate,
            output_update, output, input_state, cu_seqlens, max_seqlen_tensor
            ) = ctx.saved_tensors
        input_grad = empty_like_contiguous(output)
        forget_input_grad = empty_like_contiguous(output)
        reset_input_grad = empty_like_contiguous(output)
        weight_grad = zeros_like_contiguous(weight, dtype=torch.float32)
        forget_weight_grad = zeros_like_contiguous(weight, dtype=torch.float32)
        reset_weight_grad = zeros_like_contiguous(weight, dtype=torch.float32)
        gru_backward_triton(weight=weight, output=output, forget_weight=
            forget_weight, forget_gate=forget_gate, forget_input_grad=
            forget_input_grad, forget_weight_grad=forget_weight_grad,
            reset_weight=reset_weight, reset_gate=reset_gate,
            reset_input_grad=reset_input_grad, reset_weight_grad=
            reset_weight_grad, output_update=output_update, input_state=
            input_state, output_grad=output_grad, input_grad=input_grad,
            weight_grad=weight_grad, cu_seqlens=cu_seqlens,
            max_seqlen_tensor=max_seqlen_tensor, max_seqlen=ctx.max_seqlen,
            gradient_clipping=ctx.gradient_clipping)
        weight_grad = weight_grad.type_as(weight)
        forget_weight_grad = forget_weight_grad.type_as(forget_weight)
        reset_weight_grad = reset_weight_grad.type_as(reset_weight)
        return (input_grad, weight_grad, forget_input_grad,
            forget_weight_grad, reset_input_grad, reset_weight_grad, *([
            None] * 5))
