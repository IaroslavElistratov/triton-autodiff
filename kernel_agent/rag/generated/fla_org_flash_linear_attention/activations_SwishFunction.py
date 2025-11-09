# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/fla-org/flash-linear-attention
# Source-Files: fla/modules/activations.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rh2l8zei/flash-linear-attention-main/fla/modules/activations.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=[triton.Config({'B': bs}, num_warps=num_warps) for
    bs in [512, 1024, 2048, 4096, 8192] for num_warps in NUM_WARPS_AUTOTUNE
    ], key=['D'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def swish_fwd_kernel(x, y, T, B: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < T
    x_val = tl.load(x + offs, mask=mask, other=0.0).to(tl.float32)
    s = 1.0 / (1.0 + exp(-x_val))
    y_val = x_val * s
    tl.store(y + offs, y_val.to(y.dtype.element_ty), mask=mask)


def swish_fwd(x: torch.Tensor) ->torch.Tensor:
    T, D = x.numel(), x.shape[-1]
    y = torch.empty_like(x)
    swish_fwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](x, y, T=T, D=D)
    return y


# Forward method (kernel launch code)
def _SwishFunction_forward(ctx, x):
    ctx.save_for_backward(x)
    return swish_fwd(x)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=[triton.Config({'B': bs}, num_warps=num_warps) for
    bs in [512, 1024, 2048, 4096, 8192] for num_warps in NUM_WARPS_AUTOTUNE
    ], key=['D'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def swish_bwd_kernel(x, dy, dx, T, B: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < T
    x_val = tl.load(x + offs, mask=mask, other=0.0).to(tl.float32)
    g_val = tl.load(dy + offs, mask=mask, other=0.0).to(tl.float32)
    s = 1.0 / (1.0 + exp(-x_val))
    dx_val = g_val * s * (1.0 + x_val * (1.0 - s))
    tl.store(dx + offs, dx_val.to(dx.dtype.element_ty), mask=mask)


def swish_bwd(x: torch.Tensor, dy: torch.Tensor) ->torch.Tensor:
    T, D = x.numel(), x.shape[-1]
    dx = torch.empty_like(x)
    swish_bwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](x, dy, dx,
        T=T, D=D)
    return dx


# Backward method (kernel launch code)
def _SwishFunction_backward(ctx, dout):
    x, = ctx.saved_tensors
    return swish_bwd(x, dout)


# ============================================================
# autograd.Function Class Definition
# ============================================================

class SwishFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_fwd(x)

    @staticmethod
    def backward(ctx, dout):
        x, = ctx.saved_tensors
        return swish_bwd(x, dout)
