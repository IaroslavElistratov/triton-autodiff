# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/unslothai/unsloth
# Source-Files: unsloth/kernels/fp8.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_cfgjhluw/unsloth-main/unsloth/kernels/fp8.py
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

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    s = 1.0 if s == 0 else s
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(x: torch.Tensor, block_size: int=128) ->tuple[torch.Tensor,
    torch.Tensor]:
    if not x.is_contiguous():
        x = x.contiguous()
    assert x.shape[-1] % block_size == 0
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.
        float32)

    def grid(meta):
        return triton.cdiv(x.numel(), meta['BLOCK_SIZE']),
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


# Forward method (kernel launch code)
def _FP8BlockQuantLinear_forward(ctx, X, weight, weight_scale):
    m, n = weight.shape
    p, q = weight_scale.shape
    block_size = getattr(weight, 'block_size', None) or getattr(weight_scale,
        'block_size', None)
    assert block_size is not None, 'block_size is not set'
    if triton.cdiv(m, block_size[0]) != p or triton.cdiv(n, block_size[1]
        ) != q:
        if triton.cdiv(m, block_size[0]) == q and triton.cdiv(n, block_size[1]
            ) == p:
            weight_scale = weight_scale.T
        else:
            raise ValueError(
                f'Weight shape {weight.shape} and scales shape {weight_scale.shape} is not compatible with block size {block_size}'
                )
    if not weight.is_contiguous():
        weight = weight.contiguous()
    qinput, scale = act_quant(X, block_size[1])
    output = fp8_block_matmul(qinput, weight, scale, weight_scale,
        block_size, output_dtype=X.dtype)
    ctx.weight = weight
    ctx.weight_scale = weight_scale
    ctx.block_size = block_size
    return output.to(X.dtype)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, dtype=torch.bfloat16):
    if s.shape[1] == 1:
        if x.shape[0] == s.shape[0]:
            y = x.to(dtype) * s.to(dtype)
        elif x.shape[1] == s.shape[0]:
            y = x.t().to(dtype) * s.to(dtype)
            y = y.t()
        else:
            raise ValueError(
                f'Incompatible shapes x.shape={x.shape!r}, s.shape={s.shape!r}'
                )
        return y
    else:
        return weight_dequant_block(x, s, dtype=dtype)


def weight_dequant_block(x: torch.Tensor, s: torch.Tensor, block_size: int=
    128, dtype=torch.bfloat16) ->torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()
    if not s.is_contiguous():
        s = s.contiguous()
    assert x.dim() == 2 and s.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype=dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N,
        meta['BLOCK_SIZE']))
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


# Backward method (kernel launch code)
def _FP8BlockQuantLinear_backward(ctx, grad_output):
    W_deq = weight_dequant(ctx.weight, ctx.weight_scale)
    grad_X = torch_matmul(grad_output, W_deq.t())
    del W_deq
    return grad_X, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class FP8BlockQuantLinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, weight, weight_scale):
        m, n = weight.shape
        p, q = weight_scale.shape
        block_size = getattr(weight, 'block_size', None) or getattr(
            weight_scale, 'block_size', None)
        assert block_size is not None, 'block_size is not set'
        if triton.cdiv(m, block_size[0]) != p or triton.cdiv(n, block_size[1]
            ) != q:
            if triton.cdiv(m, block_size[0]) == q and triton.cdiv(n,
                block_size[1]) == p:
                weight_scale = weight_scale.T
            else:
                raise ValueError(
                    f'Weight shape {weight.shape} and scales shape {weight_scale.shape} is not compatible with block size {block_size}'
                    )
        if not weight.is_contiguous():
            weight = weight.contiguous()
        qinput, scale = act_quant(X, block_size[1])
        output = fp8_block_matmul(qinput, weight, scale, weight_scale,
            block_size, output_dtype=X.dtype)
        ctx.weight = weight
        ctx.weight_scale = weight_scale
        ctx.block_size = block_size
        return output.to(X.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        W_deq = weight_dequant(ctx.weight, ctx.weight_scale)
        grad_X = torch_matmul(grad_output, W_deq.t())
        del W_deq
        return grad_X, None, None
