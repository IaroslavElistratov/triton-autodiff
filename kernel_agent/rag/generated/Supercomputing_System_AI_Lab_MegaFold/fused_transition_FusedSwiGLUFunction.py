# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/Supercomputing-System-AI-Lab/MegaFold
# Source-Files: megafold/model/FusedTransition/fused_transition.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_qk7vt93p/MegaFold-main/megafold/model/FusedTransition/fused_transition.py
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

@triton.jit
def _swiglu_forward_kernel(x_ptr, y_ptr, stride_y, D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(axis=0).to(tl.int64)
    a_ptr = x_ptr + row_idx * stride_y * 2
    b_ptr = x_ptr + row_idx * stride_y * 2 + stride_y
    y_ptr += row_idx * stride_y
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < stride_y
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    c_row = b_row * tl.sigmoid(b_row) * a_row
    tl.store(y_ptr + col_offsets, c_row, mask=mask)


def swiglu_forward(x):
    """ 
    Input: x = [left, right] -- left half and right half
    Output: y = F.silu(right) * left
    """
    ori_shape = x.shape
    D = ori_shape[-1]
    dim_out = D // 2
    x = x.view(-1, D)
    M = x.shape[0]
    y = torch.empty((M, dim_out), device=x.device, dtype=x.dtype)
    BLOCK_SIZE, num_warps = calculate_settings(dim_out)
    _swiglu_forward_kernel[M,](x, y, y.stride(0), D=D, BLOCK_SIZE=
        BLOCK_SIZE, num_warps=num_warps)
    return x, y.view(ori_shape[:-1] + (dim_out,))


# Forward method (kernel launch code)
@ensure_contiguous
@torch.amp.custom_fwd(device_type=infer_device(), cast_inputs=torch.bfloat16)
def _FusedSwiGLUFunction_forward(ctx, input):
    input, output = swiglu_forward(input)
    ctx.save_for_backward(input)
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _swiglu_backward_kernel(dy_ptr, x_ptr, stride, D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(axis=0).to(tl.int64)
    dy_ptr += row_idx * stride
    a_ptr = x_ptr + row_idx * stride * 2
    b_ptr = x_ptr + row_idx * stride * 2 + stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < D
    dy_row = tl.load(dy_ptr + col_offsets, mask=mask, other=0.0)
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0.0)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    sig_b = tl.sigmoid(b_row)
    silu_b = b_row * sig_b
    da_row = dy_row * silu_b
    db_row = dy_row * (silu_b * (1 - sig_b) + sig_b) * a_row
    tl.store(a_ptr + col_offsets, da_row, mask=mask)
    tl.store(b_ptr + col_offsets, db_row, mask=mask)


def swiglu_backward(x, dy):
    """ 
    Input: dy
    Output: dx = [dLEFT, dRIGHT] -- left half and right half
    """
    ori_shape = dy.shape
    D = ori_shape[-1]
    dy = dy.view(-1, D)
    M = dy.shape[0]
    BLOCK_SIZE, num_warps = calculate_settings(D)
    _swiglu_backward_kernel[M,](dy, x, dy.stride(0), D=D, BLOCK_SIZE=
        BLOCK_SIZE, num_warps=num_warps)
    return x.view(ori_shape[:-1] + (D * 2,))


# Backward method (kernel launch code)
@ensure_contiguous
@torch.amp.custom_bwd(device_type=infer_device())
def _FusedSwiGLUFunction_backward(ctx, grad_output):
    input, = ctx.saved_tensors
    grad_input = swiglu_backward(input, grad_output)
    return grad_input


# ============================================================
# autograd.Function Class Definition
# ============================================================

class FusedSwiGLUFunction(torch.autograd.Function):

    @staticmethod
    @ensure_contiguous
    @torch.amp.custom_fwd(device_type=infer_device(), cast_inputs=torch.
        bfloat16)
    def forward(ctx, input):
        input, output = swiglu_forward(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    @ensure_contiguous
    @torch.amp.custom_bwd(device_type=infer_device())
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = swiglu_backward(input, grad_output)
        return grad_input
