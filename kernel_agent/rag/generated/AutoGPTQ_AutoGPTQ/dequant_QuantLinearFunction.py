# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/AutoGPTQ/AutoGPTQ
# Source-Files: auto_gptq/nn_modules/triton_utils/dequant.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_s7c2nfmu/AutoGPTQ-main/auto_gptq/nn_modules/triton_utils/dequant.py
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

def dequant248(qweight, scales, qzeros, g_idx, bits, maxq=None):
    """
    Launcher for triton dequant kernel.  Only valid for bits = 2, 4, 8
    """
    num_groups = scales.shape[0]
    outfeatures = scales.shape[1]
    infeatures = g_idx.shape[0]
    out = torch.empty((infeatures, outfeatures), device='cuda', dtype=torch
        .float16)
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
    W = dequant248(qweight, scales, qzeros, g_idx, bits, maxq=maxq)
    if transpose:
        return input @ W.t()
    return input @ W


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

# Forward method (kernel launch code)
@custom_fwd
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
@custom_bwd
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
    @custom_fwd
    def forward(ctx, input, qweight, scales, qzeros, g_idx, bits, maxq):
        output = quant_matmul_248(input, qweight, scales, qzeros, g_idx,
            bits, maxq)
        ctx.save_for_backward(qweight, scales, qzeros, g_idx)
        ctx.bits, ctx.maxq = bits, maxq
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        qweight, scales, qzeros, g_idx = ctx.saved_tensors
        bits, maxq = ctx.bits, ctx.maxq
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = quant_matmul_248(grad_output, qweight, scales,
                qzeros, g_idx, bits, maxq, transpose=True)
        return grad_input, None, None, None, None, None, None
