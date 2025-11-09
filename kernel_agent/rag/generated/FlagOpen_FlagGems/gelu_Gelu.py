# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/FlagOpen/FlagGems
# Source-Files: src/flag_gems/runtime/backend/_hygon/ops/gelu.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_kcv6dce2/FlagGems-master/src/flag_gems/runtime/backend/_hygon/ops/gelu.py
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

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@pointwise_dynamic(promotion_methods=[(0, 'DEFAULT')])
@triton.jit
def gelu_none(x):
    x_fp32 = x.to(tl.float32)
    scale: tl.constexpr = 0.7071067811
    output = 0.5 * x_fp32 * (1 + erf(x_fp32 * scale))
    return output


@pointwise_dynamic(promotion_methods=[(0, 'DEFAULT')])
@triton.jit
def gelu_tanh(x):
    x_fp32 = x.to(tl.float32)
    output = 0.5 * x_fp32 * (1 + tanh(x_fp32 * 0.79788456 * (1 + 0.044715 *
        pow(x_fp32.to(tl.float32), 2))))
    return output


# Forward method (kernel launch code)
def _Gelu_forward(ctx, A, approximate):
    logger.debug('GEMS GELU FORWARD')
    if approximate == 'tanh':
        out = gelu_tanh(A)
    else:
        out = gelu_none(A)
    ctx.save_for_backward(A)
    ctx.approximate = approximate
    return out


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@pointwise_dynamic(promotion_methods=[(0, 1, 'DEFAULT')])
@triton.jit
def gelu_backward_none(x, dy):
    scale1: tl.constexpr = 0.7071067811
    scale2: tl.constexpr = 0.3989422803
    x_fp32 = x.to(tl.float32)
    dydx = scale2 * x_fp32 * exp(-pow(scale1 * x_fp32, 2)) + 0.5 * erf(
        scale1 * x_fp32) + 0.5
    dx = dydx * dy
    return dx


@pointwise_dynamic(promotion_methods=[(0, 1, 'DEFAULT')])
@triton.jit
def gelu_backward_tanh(x, dy):
    x_fp32 = x.to(tl.float32)
    tanh_out = tanh(0.79788456 * x_fp32 * (1 + 0.044715 * pow(x_fp32, 2)))
    dydx = 0.5 * x_fp32 * ((1 - pow(tanh_out, 2)) * (0.79788456 + 
        0.1070322243 * pow(x_fp32, 2))) + 0.5 * (1 + tanh_out)
    dx = dydx * dy
    return dx


# Backward method (kernel launch code)
def _Gelu_backward(ctx, out_grad):
    logger.debug('GEMS GELU BACKWARD')
    inp, = ctx.saved_tensors
    approximate = ctx.approximate
    if approximate == 'tanh':
        in_grad = gelu_backward_tanh(inp, out_grad)
    else:
        in_grad = gelu_backward_none(inp, out_grad)
    return in_grad, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class Gelu(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, approximate):
        logger.debug('GEMS GELU FORWARD')
        if approximate == 'tanh':
            out = gelu_tanh(A)
        else:
            out = gelu_none(A)
        ctx.save_for_backward(A)
        ctx.approximate = approximate
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug('GEMS GELU BACKWARD')
        inp, = ctx.saved_tensors
        approximate = ctx.approximate
        if approximate == 'tanh':
            in_grad = gelu_backward_tanh(inp, out_grad)
        else:
            in_grad = gelu_backward_none(inp, out_grad)
        return in_grad, None
