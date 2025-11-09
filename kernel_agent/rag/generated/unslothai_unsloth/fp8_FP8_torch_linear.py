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

# Forward method (kernel launch code)
def _FP8_torch_linear_forward(ctx, X, weight, weight_scale, bias=None):
    orig_shape = X.shape
    X = X.view(-1, X.shape[-1])
    bs_n, bs_k = getattr(weight, 'block_size', None) or getattr(weight_scale,
        'block_size', [128, 128])
    bs_m = bs_n
    m, n = weight.shape
    p, q = weight_scale.shape
    if triton.cdiv(m, bs_n) != p or triton.cdiv(n, bs_k) != q:
        if triton.cdiv(m, bs_n) == q and triton.cdiv(n, bs_k) == p:
            weight_scale = weight_scale.T
        else:
            raise ValueError(
                f'Weight shape {weight.shape} and scales shape {weight_scale.shape} is not compatible with block size {block_size}'
                )
    xq, xs = triton_quantize_fp8_block(X, bs_m, bs_n, None)
    output = torch.ops.fbgemm.f8f8bf16_blockwise(xq, weight.contiguous(),
        xs, weight_scale.contiguous(), bs_m, bs_n, bs_k)
    output = output + bias if bias is not None else output
    output = output.view(*orig_shape[:-1], -1)
    del xq
    del xs
    ctx.weight = weight
    ctx.weight_scale = weight_scale
    ctx.block_size = [bs_m, bs_n, bs_k]
    return output


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
def _FP8_torch_linear_backward(ctx, grad_output):
    W_deq = weight_dequant(ctx.weight, ctx.weight_scale)
    grad_X = torch_matmul(grad_output, W_deq.t())
    del W_deq
    return grad_X, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class FP8_torch_linear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, weight, weight_scale, bias=None):
        orig_shape = X.shape
        X = X.view(-1, X.shape[-1])
        bs_n, bs_k = getattr(weight, 'block_size', None) or getattr(
            weight_scale, 'block_size', [128, 128])
        bs_m = bs_n
        m, n = weight.shape
        p, q = weight_scale.shape
        if triton.cdiv(m, bs_n) != p or triton.cdiv(n, bs_k) != q:
            if triton.cdiv(m, bs_n) == q and triton.cdiv(n, bs_k) == p:
                weight_scale = weight_scale.T
            else:
                raise ValueError(
                    f'Weight shape {weight.shape} and scales shape {weight_scale.shape} is not compatible with block size {block_size}'
                    )
        xq, xs = triton_quantize_fp8_block(X, bs_m, bs_n, None)
        output = torch.ops.fbgemm.f8f8bf16_blockwise(xq, weight.contiguous(
            ), xs, weight_scale.contiguous(), bs_m, bs_n, bs_k)
        output = output + bias if bias is not None else output
        output = output.view(*orig_shape[:-1], -1)
        del xq
        del xs
        ctx.weight = weight
        ctx.weight_scale = weight_scale
        ctx.block_size = [bs_m, bs_n, bs_k]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        W_deq = weight_dequant(ctx.weight, ctx.weight_scale)
        grad_X = torch_matmul(grad_output, W_deq.t())
        del W_deq
        return grad_X, None, None, None, None
