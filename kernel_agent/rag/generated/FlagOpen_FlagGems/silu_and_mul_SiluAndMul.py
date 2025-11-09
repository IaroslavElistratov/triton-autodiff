# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/FlagOpen/FlagGems
# Source-Files: src/flag_gems/runtime/backend/_cambricon/fused/silu_and_mul.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_kcv6dce2/FlagGems-master/src/flag_gems/runtime/backend/_cambricon/fused/silu_and_mul.py
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

@pointwise_dynamic(promotion_methods=[(0, 1, 'DEFAULT')])
@triton.jit
def silu_and_mul_kernel(x, y):
    x_fp32 = x.to(tl.float32)
    x_silu = tl.fdiv(x_fp32, 1.0 + tl.exp(-x_fp32))
    return x_silu * y


# Forward method (kernel launch code)
def _SiluAndMul_forward(ctx, A, B):
    ctx.save_for_backward(A, B)
    logger.debug('GEMS_CAMBRICON SILU AND MUL FORWARD')
    return silu_and_mul_kernel(A, B)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@pointwise_dynamic(promotion_methods=[(0, 1, 2, 'DEFAULT'), (0, 1, 2,
    'DEFAULT')], num_outputs=2)
@triton.jit
def silu_and_mul_grad_kernel(x, y, dgrad):
    x_fp32 = x.to(tl.float32)
    sig = tl.extra.mlu.libdevice.fast_sigmoid(x_fp32)
    x_silu = x_fp32 * sig
    d_x_silu = sig * (1 + x_fp32 * (1 - sig))
    dx = d_x_silu * dgrad * y
    dy = dgrad * x_silu
    return dx, dy


# Backward method (kernel launch code)
def _SiluAndMul_backward(ctx, grad_output):
    A, B = ctx.saved_tensors
    grad_A, grad_B = silu_and_mul_grad_kernel(A, B, grad_output)
    return grad_A, grad_B


# ============================================================
# autograd.Function Class Definition
# ============================================================

class SiluAndMul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        logger.debug('GEMS_CAMBRICON SILU AND MUL FORWARD')
        return silu_and_mul_kernel(A, B)

    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A, grad_B = silu_and_mul_grad_kernel(A, B, grad_output)
        return grad_A, grad_B
