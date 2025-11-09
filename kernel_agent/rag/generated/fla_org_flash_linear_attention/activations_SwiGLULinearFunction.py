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

@triton.autotune(configs=[triton.Config({'B': bs}, num_warps=num_warps) for
    bs in [512, 1024, 2048, 4096, 8192] for num_warps in NUM_WARPS_AUTOTUNE
    ], key=['D'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def swiglu_fwd_kernel(x, y, z, T, B: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < T
    x_val = tl.load(x + offs, mask=mask, other=0.0).to(tl.float32)
    y_val = tl.load(y + offs, mask=mask, other=0.0).to(tl.float32)
    s = 1.0 / (1.0 + exp(-x_val))
    z_val = x_val * s * y_val
    tl.store(z + offs, z_val.to(z.dtype.element_ty), mask=mask)


def swiglu_fwd(x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
    T, D = x.numel(), x.shape[-1]
    z = torch.empty_like(x)
    swiglu_fwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](x, y, z, T
        =T, D=D)
    return z


# Forward method (kernel launch code)
@autocast_custom_fwd
def _SwiGLULinearFunction_forward(ctx, x, y, weight, bias):
    z = swiglu_fwd(x, y)
    out = F.linear(z, weight, bias)
    ctx.save_for_backward(x, y, weight)
    ctx.linear_bias_is_none = bias is None
    return out


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.heuristics({'HAS_WEIGHT': lambda args: args['z'] is not None})
@triton.autotune(configs=[triton.Config({'B': bs}, num_warps=num_warps) for
    bs in [512, 1024, 2048, 4096, 8192] for num_warps in NUM_WARPS_AUTOTUNE
    ], key=['D'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def swiglu_fwdbwd_kernel(x, y, g, dx, dy, z, T, B: tl.constexpr, D: tl.
    constexpr, HAS_WEIGHT: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < T
    x_val = tl.load(x + offs, mask=mask, other=0.0).to(tl.float32)
    y_val = tl.load(y + offs, mask=mask, other=0.0).to(tl.float32)
    g_val = tl.load(g + offs, mask=mask, other=0.0).to(tl.float32)
    s = 1.0 / (1.0 + exp(-x_val))
    x_s = x_val * s
    dx_val = g_val * s * (1.0 + x_val * (1.0 - s)) * y_val
    dy_val = g_val * x_s
    tl.store(dx + offs, dx_val.to(dx.dtype.element_ty), mask=mask)
    tl.store(dy + offs, dy_val.to(dy.dtype.element_ty), mask=mask)
    if HAS_WEIGHT:
        z_val = x_s * y_val
        tl.store(z + offs, z_val.to(z.dtype.element_ty), mask=mask)


def swiglu_fwdbwd(x: torch.Tensor, y: torch.Tensor, g: torch.Tensor,
    use_weight: bool=False):
    T, D = x.numel(), x.shape[-1]
    dx = torch.empty_like(x)
    dy = torch.empty_like(x)
    if use_weight:
        z = torch.empty_like(x)
    else:
        z = None
    swiglu_fwdbwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](x, y, g,
        dx, dy, z, T=T, D=D)
    if use_weight:
        return dx, dy, z
    return dx, dy


# Backward method (kernel launch code)
@autocast_custom_bwd
def _SwiGLULinearFunction_backward(ctx, dout, *args):
    x, y, weight = ctx.saved_tensors
    dout = dout.reshape(-1, dout.shape[-1])
    dz = F.linear(dout, weight.t()).view_as(x)
    dx, dy, z = swiglu_fwdbwd(x, y, dz, use_weight=True)
    dlinear_weight = torch.einsum('bo,bi->oi', dout, z.reshape(-1, z.shape[-1])
        )
    dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
    return dx, dy, dlinear_weight, dlinear_bias


# ============================================================
# autograd.Function Class Definition
# ============================================================

class SwiGLULinearFunction(torch.autograd.Function):
    """
    Swish-Gated Linear Unit (SwiGLU) function followed by a linear transformation.

    .. math::
        \\text{SwiGLULinear}(x, y, W, b) = (swish(x) * y) W + b

    This simple wrap discards the intermediate results of SwiGLU(x, y) to save memory.
    """

    @staticmethod
    @autocast_custom_fwd
    def forward(ctx, x, y, weight, bias):
        z = swiglu_fwd(x, y)
        out = F.linear(z, weight, bias)
        ctx.save_for_backward(x, y, weight)
        ctx.linear_bias_is_none = bias is None
        return out

    @staticmethod
    @autocast_custom_bwd
    def backward(ctx, dout, *args):
        x, y, weight = ctx.saved_tensors
        dout = dout.reshape(-1, dout.shape[-1])
        dz = F.linear(dout, weight.t()).view_as(x)
        dx, dy, z = swiglu_fwdbwd(x, y, dz, use_weight=True)
        dlinear_weight = torch.einsum('bo,bi->oi', dout, z.reshape(-1, z.
            shape[-1]))
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        return dx, dy, dlinear_weight, dlinear_bias
