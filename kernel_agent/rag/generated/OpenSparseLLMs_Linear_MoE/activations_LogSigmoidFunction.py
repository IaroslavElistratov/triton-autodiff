# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/OpenSparseLLMs/Linear-MoE
# Source-Files: linear_moe/model/common_modules/activations.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_kyohjmxq/Linear-MoE-main/linear_moe/model/common_modules/activations.py
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
from math import log

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=[triton.Config({'BT': 16}, num_warps=2), triton.
    Config({'BT': 16}, num_warps=4), triton.Config({'BT': 16}, num_warps=8),
    triton.Config({'BT': 32}, num_warps=2), triton.Config({'BT': 32},
    num_warps=4), triton.Config({'BT': 32}, num_warps=8), triton.Config({
    'BT': 64}, num_warps=2), triton.Config({'BT': 64}, num_warps=4), triton
    .Config({'BT': 64}, num_warps=8), triton.Config({'BT': 128}, num_warps=
    2), triton.Config({'BT': 128}, num_warps=4), triton.Config({'BT': 128},
    num_warps=8), triton.Config({'BT': 256}, num_warps=2), triton.Config({
    'BT': 256}, num_warps=4), triton.Config({'BT': 256}, num_warps=8)], key
    =['D'])
@triton.jit
def logsigmoid_fwd_kernel(x, y, T: tl.constexpr, D: tl.constexpr, BT: tl.
    constexpr):
    i = tl.program_id(0)
    o_i = i * BT + tl.arange(0, BT)
    p_x = x + o_i
    p_y = y + o_i
    mask = o_i < T
    b_x = tl.load(p_x, mask=mask, other=0.0).to(tl.float32)
    b_m = tl.minimum(0.0, b_x)
    b_z = 1.0 + tl.exp(-tl.abs(b_x))
    b_y = b_m - tl.log(b_z)
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), mask=mask)


# Forward method (kernel launch code)
@contiguous
def _LogSigmoidFunction_forward(ctx, x):
    T, D = x.numel(), x.shape[-1]
    y = torch.empty_like(x)
    logsigmoid_fwd_kernel[lambda meta: (triton.cdiv(meta['T'], meta['D']),)](x,
        y, T=T, D=D)
    ctx.save_for_backward(x)
    return y


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=[triton.Config({'BT': 16}, num_warps=2), triton.
    Config({'BT': 16}, num_warps=4), triton.Config({'BT': 16}, num_warps=8),
    triton.Config({'BT': 32}, num_warps=2), triton.Config({'BT': 32},
    num_warps=4), triton.Config({'BT': 32}, num_warps=8), triton.Config({
    'BT': 64}, num_warps=2), triton.Config({'BT': 64}, num_warps=4), triton
    .Config({'BT': 64}, num_warps=8), triton.Config({'BT': 128}, num_warps=
    2), triton.Config({'BT': 128}, num_warps=4), triton.Config({'BT': 128},
    num_warps=8), triton.Config({'BT': 256}, num_warps=2), triton.Config({
    'BT': 256}, num_warps=4), triton.Config({'BT': 256}, num_warps=8)], key
    =['D'])
@triton.jit
def logsigmoid_bwd_kernel(x, dx, dy, T: tl.constexpr, D: tl.constexpr, BT:
    tl.constexpr):
    i = tl.program_id(0)
    o_i = i * BT + tl.arange(0, BT)
    p_x = x + o_i
    p_dx = dx + o_i
    p_dy = dy + o_i
    mask = o_i < T
    b_x = tl.load(p_x, mask=mask, other=0.0).to(tl.float32)
    b_dy = tl.load(p_dy, mask=mask, other=0.0).to(tl.float32)
    b_dx = b_dy * (1.0 - tl.sigmoid(b_x))
    tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), mask=mask)


# Backward method (kernel launch code)
@contiguous
def _LogSigmoidFunction_backward(ctx, dy):
    x, = ctx.saved_tensors
    T, D = x.numel(), x.shape[-1]
    dx = torch.empty_like(x)
    logsigmoid_bwd_kernel[lambda meta: (triton.cdiv(meta['T'], meta['D']),)](x,
        dx, dy, T=T, D=D)
    return dx


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LogSigmoidFunction(torch.autograd.Function):

    @contiguous
    @staticmethod
    def forward(ctx, x):
        T, D = x.numel(), x.shape[-1]
        y = torch.empty_like(x)
        logsigmoid_fwd_kernel[lambda meta: (triton.cdiv(meta['T'], meta['D']),)
            ](x, y, T=T, D=D)
        ctx.save_for_backward(x)
        return y

    @contiguous
    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        T, D = x.numel(), x.shape[-1]
        dx = torch.empty_like(x)
        logsigmoid_bwd_kernel[lambda meta: (triton.cdiv(meta['T'], meta['D']),)
            ](x, dx, dy, T=T, D=D)
        return dx
