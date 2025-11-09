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
from math import log

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=[triton.Config({'B': bs}, num_warps=num_warps) for
    bs in [512, 1024, 2048, 4096, 8192] for num_warps in NUM_WARPS_AUTOTUNE
    ], key=['D'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def logsigmoid_fwd_kernel(x, y, temperature, T, B: tl.constexpr, D: tl.
    constexpr):
    i = tl.program_id(0)
    o_i = i * B + tl.arange(0, B)
    m_i = o_i < T
    b_x = tl.load(x + o_i, mask=m_i, other=0.0).to(tl.float32)
    b_m = tl.minimum(0.0, b_x)
    b_z = 1.0 + exp(-tl.abs(b_x))
    b_y = (b_m - log(b_z)) / temperature
    tl.store(y + o_i, b_y.to(y.dtype.element_ty), mask=m_i)


def logsigmoid_fwd(x: torch.Tensor, temperature: float=1.0) ->torch.Tensor:
    T, D = x.numel(), x.shape[-1]
    y = torch.empty_like(x)
    logsigmoid_fwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](x=x, y
        =y, temperature=temperature, T=T, D=D)
    return y


# Forward method (kernel launch code)
@input_guard
def _LogSigmoidFunction_forward(ctx, x, temperature):
    ctx.save_for_backward(x)
    ctx.temperature = temperature
    return logsigmoid_fwd(x, temperature)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=[triton.Config({'B': bs}, num_warps=num_warps) for
    bs in [512, 1024, 2048, 4096, 8192] for num_warps in NUM_WARPS_AUTOTUNE
    ], key=['D'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def logsigmoid_bwd_kernel(x, dx, dy, temperature, T, B: tl.constexpr, D: tl
    .constexpr):
    i = tl.program_id(0)
    o_i = i * B + tl.arange(0, B)
    m_i = o_i < T
    b_x = tl.load(x + o_i, mask=m_i, other=0.0).to(tl.float32)
    b_dy = tl.load(dy + o_i, mask=m_i, other=0.0).to(tl.float32)
    b_dx = b_dy * ((1.0 - tl.sigmoid(b_x)) / temperature)
    tl.store(dx + o_i, b_dx.to(dx.dtype.element_ty), mask=m_i)


def logsigmoid_bwd(x: torch.Tensor, dy: torch.Tensor, temperature: float=1.0
    ) ->torch.Tensor:
    T, D = x.numel(), x.shape[-1]
    dx = torch.empty_like(x)
    logsigmoid_bwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](x=x,
        dx=dx, dy=dy, temperature=temperature, T=T, D=D)
    return dx


# Backward method (kernel launch code)
@input_guard
def _LogSigmoidFunction_backward(ctx, dy):
    x, = ctx.saved_tensors
    return logsigmoid_bwd(x, dy, ctx.temperature), None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LogSigmoidFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(ctx, x, temperature):
        ctx.save_for_backward(x)
        ctx.temperature = temperature
        return logsigmoid_fwd(x, temperature)

    @staticmethod
    @input_guard
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        return logsigmoid_bwd(x, dy, ctx.temperature), None
