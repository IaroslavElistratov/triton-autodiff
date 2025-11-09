# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/intel/auto-round
# Source-Files: auto_round_extension/triton/triton_utils_zp/dequant.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_us85855o/auto-round-main/auto_round_extension/triton/triton_utils_zp/dequant.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

def dequant248(qweight, scales, qzeros, g_idx, bits, maxq=None, input_dtype
    =torch.float16):
    """
    Launcher for triton dequant kernel. Only valid for bits = 2, 4, 8
    """
    device_type = qweight.device.type
    if device_type in {'cuda', 'xpu'}:
        with getattr(torch, device_type).device(qweight.device):
            return dequant248_core(qweight, scales, qzeros, g_idx, bits,
                maxq=maxq, input_dtype=input_dtype)
    else:
        raise ValueError(f'Unsupported device type: {device_type}')


def dequant248_core(qweight, scales, qzeros, g_idx, bits, maxq=None,
    input_dtype=torch.float16):
    """
    Launcher for triton dequant kernel.  Only valid for bits = 2, 4, 8
    """
    num_groups = scales.shape[0]
    outfeatures = scales.shape[1]
    infeatures = g_idx.shape[0]
    out = torch.empty((infeatures, outfeatures), device=qweight.device,
        dtype=input_dtype)
    numels = out.numel()
    maxq = 2 ** bits - 1 if maxq is None else maxq
    grid = lambda meta: (triton.cdiv(numels, meta['X_BLOCK']),)
    dequant_kernel_248[grid](g_idx, scales, qweight, qzeros, out, numels,
        maxq=maxq, bits=bits, outfeatures=outfeatures, num_groups=num_groups)
    return out


@triton.autotune(DEFAULT_DEQUANT_CONFIGS, key=['numels'])
@triton.jit
def dequant_kernel_248(g_idx_ptr, scales_ptr, qweight_ptr, qzeros_ptr,
    out_ptr, numels, maxq: tl.constexpr, bits: tl.constexpr, outfeatures:
    tl.constexpr, num_groups: tl.constexpr, X_BLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * X_BLOCK
    x_index = xoffset + tl.arange(0, X_BLOCK)
    xmask = x_index < numels
    row_idx = x_index // outfeatures
    col_idx = x_index % outfeatures
    elements_per_feature: tl.constexpr = 32 // bits
    g_idx = tl.load(g_idx_ptr + row_idx, None, eviction_policy='evict_last')
    qweights = tl.load(qweight_ptr + (col_idx + outfeatures * (row_idx //
        elements_per_feature)), None)
    wf_weights = row_idx % elements_per_feature * bits
    wf_zeros = col_idx % elements_per_feature * bits
    tmp1 = g_idx + num_groups
    tmp2 = g_idx < 0
    tl.device_assert(g_idx >= 0, 'index out of bounds: 0 <= tmp0 < 0')
    groups = tl.where(tmp2, tmp1, g_idx)
    scales = tl.load(scales_ptr + (col_idx + outfeatures * groups), None).to(tl
        .float32)
    weights = qweights >> wf_weights
    weights = weights & maxq
    qzero_ncols: tl.constexpr = outfeatures // elements_per_feature
    qzeros = tl.load(qzeros_ptr + (qzero_ncols * groups + col_idx //
        elements_per_feature), None, eviction_policy='evict_last')
    zeros = qzeros >> wf_zeros
    zeros = zeros & maxq
    zeros = zeros + 1
    weights = weights - zeros
    weights = weights.to(tl.float32)
    weights = scales * weights
    tl.store(out_ptr + x_index, weights, mask=xmask)


def quant_matmul_248(input, qweight, scales, qzeros, g_idx, bits, maxq=None,
    transpose=False):
    input_dtype = input.dtype
    W = dequant248(qweight, scales, qzeros, g_idx, bits, maxq=maxq,
        input_dtype=input_dtype)
    orig_device = input.device
    if transpose:
        return (input.to(W.device) @ W.t()).to(orig_device)
    return (input.to(W.device) @ W).to(orig_device)


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

# Forward method (kernel launch code)
def _QuantLinearFunction_forward(ctx, input, qweight, scales, qzeros, g_idx,
    bits, maxq):
    output = quant_matmul_248(input, qweight, scales, qzeros, g_idx, bits, maxq
        )
    ctx.save_for_backward(qweight, scales, qzeros, g_idx)
    ctx.bits, ctx.maxq = bits, maxq
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

# Backward method (kernel launch code)
def _QuantLinearFunction_backward(ctx, grad_output):
    qweight, scales, qzeros, g_idx = ctx.saved_tensors
    bits, maxq = ctx.bits, ctx.maxq
    grad_input = None
    if ctx.needs_input_grad[0]:
        grad_input = quant_matmul_248(grad_output, qweight, scales, qzeros,
            g_idx, bits, maxq, transpose=True)
    return grad_input, None, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class QuantLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, qweight, scales, qzeros, g_idx, bits, maxq):
        output = quant_matmul_248(input, qweight, scales, qzeros, g_idx,
            bits, maxq)
        ctx.save_for_backward(qweight, scales, qzeros, g_idx)
        ctx.bits, ctx.maxq = bits, maxq
        return output

    @staticmethod
    def backward(ctx, grad_output):
        qweight, scales, qzeros, g_idx = ctx.saved_tensors
        bits, maxq = ctx.bits, ctx.maxq
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = quant_matmul_248(grad_output, qweight, scales,
                qzeros, g_idx, bits, maxq, transpose=True)
        return grad_input, None, None, None, None, None, None
