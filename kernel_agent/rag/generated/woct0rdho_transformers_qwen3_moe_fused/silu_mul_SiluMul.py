# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/woct0rdho/transformers-qwen3-moe-fused
# Source-Files: qwen3_moe_fused/kernels/silu_mul.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_n8hgmd0z/transformers-qwen3-moe-fused-master/qwen3_moe_fused/kernels/silu_mul.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=_autotune_configs, key=[])
@triton.jit
def _silu_mul_forward_kernel(e_ptr, g_ptr, h_ptr, n_elements: int,
    BLOCK_SIZE: tl.constexpr=128) ->None:
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    e = tl.load(e_ptr + offsets, mask=mask).to(tl.float32)
    g = tl.load(g_ptr + offsets, mask=mask)
    f = e * tl.sigmoid(e)
    f = f.to(e_ptr.dtype.element_ty)
    h = f * g
    tl.store(h_ptr + offsets, h, mask=mask)


def silu_mul_forward(e: torch.Tensor, g: torch.Tensor) ->torch.Tensor:
    assert e.is_cuda
    assert g.device == e.device
    assert e.is_contiguous()
    assert g.is_contiguous()
    assert g.numel() == e.numel()
    n_elements = e.numel()
    h = torch.empty_like(e)
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)
    _silu_mul_forward_kernel[grid](e, g, h, n_elements)
    return h


# Forward method (kernel launch code)
def _SiluMul_forward(ctx, e, g):
    ctx.save_for_backward(e, g)
    return silu_mul_forward(e, g)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=_autotune_configs, key=[])
@triton.jit
def _silu_mul_backward_kernel(dh_ptr, e_ptr, g_ptr, de_ptr, dg_ptr,
    n_elements: int, BLOCK_SIZE: tl.constexpr=128) ->None:
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    dh = tl.load(dh_ptr + offsets, mask=mask)
    e = tl.load(e_ptr + offsets, mask=mask).to(tl.float32)
    g = tl.load(g_ptr + offsets, mask=mask)
    se = tl.sigmoid(e)
    ese = e * se
    f = ese.to(e_ptr.dtype.element_ty)
    df = dh * g
    dg = f * dh
    de = df.to(tl.float32) * se * (1.0 + e - ese)
    de = de.to(de_ptr.dtype.element_ty)
    tl.store(de_ptr + offsets, de, mask=mask)
    tl.store(dg_ptr + offsets, dg, mask=mask)


def silu_mul_backward(dh: torch.Tensor, e: torch.Tensor, g: torch.Tensor
    ) ->tuple[torch.Tensor, torch.Tensor]:
    assert e.is_cuda
    assert g.device == e.device
    assert dh.device == e.device
    assert e.is_contiguous()
    assert g.is_contiguous()
    assert dh.is_contiguous()
    assert g.numel() == e.numel()
    assert dh.numel() == e.numel()
    n_elements = e.numel()
    de = torch.empty_like(e)
    dg = torch.empty_like(g)
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)
    _silu_mul_backward_kernel[grid](dh, e, g, de, dg, n_elements)
    return de, dg


# Backward method (kernel launch code)
def _SiluMul_backward(ctx, dh):
    e, g = ctx.saved_tensors
    return silu_mul_backward(dh, e, g)


# ============================================================
# autograd.Function Class Definition
# ============================================================

class SiluMul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, e, g):
        ctx.save_for_backward(e, g)
        return silu_mul_forward(e, g)

    @staticmethod
    def backward(ctx, dh):
        e, g = ctx.saved_tensors
        return silu_mul_backward(dh, e, g)
