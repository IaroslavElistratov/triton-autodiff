# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/fla-org/flash-linear-attention
# Source-Files: fla/ops/rwkv7/gate_output_correction.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rh2l8zei/flash-linear-attention-main/fla/ops/rwkv7/gate_output_correction.py
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
from torch.cuda.amp import custom_bwd
from torch.cuda.amp import custom_fwd

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=[triton.Config({'BT': BT}, num_warps=num_warps) for
    num_warps in [2, 4, 8] for BT in [2, 4, 8]], key=['num_heads',
    'head_dim', 'BLOCK_SIZE_D'], **autotune_cache_kwargs)
@triton.jit
def gate_output_correction_fwd_kernel(o_ptr, r_ptr, k_ptr, r_k_ptr, v_ptr,
    g_ptr, output_ptr, o_b_stride, o_t_stride, r_b_stride, r_t_stride,
    r_h_stride, v_b_stride, v_t_stride, v_h_stride, r_k_h_stride, T,
    T_OFFSET, num_heads: tl.constexpr, head_dim: tl.constexpr, BLOCK_SIZE_D:
    tl.constexpr, BT: tl.constexpr):
    pid_b, pid_t_block = tl.program_id(0), tl.program_id(1)
    pid_h = tl.program_id(2)
    t_start = pid_t_block * BT + T_OFFSET
    t_idx = t_start + tl.arange(0, BT)[:, None]
    mask_t = t_idx < T
    d_idx = tl.arange(0, BLOCK_SIZE_D)[None, :]
    mask_d = d_idx < head_dim
    mask = mask_t & mask_d
    offset_rk_h = pid_h * r_k_h_stride
    vec_r_k = tl.load(r_k_ptr + offset_rk_h + d_idx, mask=mask_d, other=0.0
        ).to(tl.float32)
    offset_rh = pid_b * r_b_stride + t_idx * r_t_stride + pid_h * r_h_stride
    offset_vh = pid_b * v_b_stride + t_idx * v_t_stride + pid_h * v_h_stride
    vec_r = tl.load(r_ptr + offset_rh + d_idx, mask=mask, other=0.0).to(tl.
        float32)
    vec_k = tl.load(k_ptr + offset_rh + d_idx, mask=mask, other=0.0).to(tl.
        float32)
    vec_v = tl.load(v_ptr + offset_vh + d_idx, mask=mask, other=0.0).to(tl.
        float32)
    correction = tl.sum(vec_r * vec_k * vec_r_k, axis=1)[:, None] * vec_v
    offset_o = pid_b * o_b_stride + t_idx * o_t_stride + pid_h * head_dim
    vec_o = tl.load(o_ptr + offset_o + d_idx, mask=mask, other=0.0).to(tl.
        float32)
    vec_g = tl.load(g_ptr + offset_o + d_idx, mask=mask, other=0.0).to(tl.
        float32)
    final_output = (vec_o + correction) * vec_g
    tl.store(output_ptr + offset_o + d_idx, final_output.to(output_ptr.
        dtype.element_ty), mask=mask)


# Forward method (kernel launch code)
@autocast_custom_fwd
@input_guard
def _GateOutputCorrection_forward(ctx, o, r, k, r_k, v, g):
    assert r_k.dim() == 2 and r_k.shape[0] == r.shape[-2] and r_k.shape[1
        ] == r.shape[-1]
    batch_size, seq_len, _ = o.shape
    num_heads, head_dim = r.shape[-2], r.shape[-1]
    output = torch.empty_like(o)
    ctx.save_for_backward(o, r, k, r_k, v, g)
    for t in range(0, seq_len, 65536):
        T_OFFSET = t
        T_SIZE = min(65536, seq_len - t)

        def grid(meta):
            return batch_size, triton.cdiv(T_SIZE, meta['BT']), num_heads
        gate_output_correction_fwd_kernel[grid](o, r, k, r_k, v, g, output,
            o.stride(0), o.stride(1), r.stride(0), r.stride(1), r.stride(2),
            v.stride(0), v.stride(1), v.stride(2), r_k.stride(0), T_SIZE,
            T_OFFSET, num_heads, head_dim, BLOCK_SIZE_D=triton.
            next_power_of_2(head_dim))
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=[triton.Config({'BT': BT}, num_warps=num_warps,
    num_stages=num_stages) for num_warps in [2, 4, 8] for num_stages in [1,
    2, 4] for BT in [2, 4, 8]], key=['num_heads', 'head_dim',
    'BLOCK_SIZE_D'], **autotune_cache_kwargs)
@triton.jit
def gate_output_correction_bwd_kernel(grad_output_ptr, o_ptr, r_ptr, k_ptr,
    r_k_ptr, v_ptr, g_ptr, grad_o_ptr, grad_r_ptr, grad_k_ptr,
    grad_r_k_intermediate_ptr, grad_v_ptr, grad_g_ptr, r_b_stride,
    r_t_stride, r_h_stride, o_b_stride, o_t_stride, r_k_h_stride, T,
    T_OFFSET, num_heads: tl.constexpr, head_dim: tl.constexpr, BLOCK_SIZE_D:
    tl.constexpr, BT: tl.constexpr):
    pid_b, pid_t_block = tl.program_id(0), tl.program_id(1)
    pid_h = tl.program_id(2)
    t_idx = pid_t_block * BT + T_OFFSET + tl.arange(0, BT)[:, None]
    mask_t = t_idx < T
    d_idx = tl.arange(0, BLOCK_SIZE_D)[None, :]
    mask_d = d_idx < head_dim
    mask = mask_t & mask_d
    rkv_offset = pid_b * r_b_stride + t_idx * r_t_stride + pid_h * r_h_stride
    vec_r = tl.load(r_ptr + rkv_offset + d_idx, mask=mask, other=0.0).to(tl
        .float32)
    vec_k = tl.load(k_ptr + rkv_offset + d_idx, mask=mask, other=0.0).to(tl
        .float32)
    vec_v = tl.load(v_ptr + rkv_offset + d_idx, mask=mask, other=0.0).to(tl
        .float32)
    og_offset = pid_b * o_b_stride + t_idx * o_t_stride + pid_h * head_dim
    vec_o = tl.load(o_ptr + og_offset + d_idx, mask=mask, other=0.0).to(tl.
        float32)
    vec_g = tl.load(g_ptr + og_offset + d_idx, mask=mask, other=0.0).to(tl.
        float32)
    vec_grad_output = tl.load(grad_output_ptr + og_offset + d_idx, mask=
        mask, other=0.0).to(tl.float32)
    offset_rk_h = pid_h * r_k_h_stride
    vec_r_k = tl.load(r_k_ptr + offset_rk_h + d_idx, mask=mask_d, other=0.0
        ).to(tl.float32)
    prod_r_k_rk = vec_r * vec_k * vec_r_k
    corr_scalar = tl.sum(prod_r_k_rk, axis=1)
    corr_vec = corr_scalar[:, None] * vec_v
    gated_input = vec_o + corr_vec
    vec_grad_g = vec_grad_output * gated_input
    vec_grad_gate = vec_grad_output * vec_g
    vec_grad_o = vec_grad_gate
    vec_grad_corr = vec_grad_gate
    vec_grad_v = vec_grad_corr * corr_scalar[:, None]
    grad_corr_s = tl.sum(vec_grad_corr * vec_v, axis=1)[:, None]
    vec_grad_r = grad_corr_s * vec_k * vec_r_k
    vec_grad_k = grad_corr_s * vec_r * vec_r_k
    local_grad_rk = grad_corr_s * vec_r * vec_k
    tl.store(grad_o_ptr + og_offset + d_idx, vec_grad_o.to(grad_o_ptr.dtype
        .element_ty), mask=mask)
    tl.store(grad_g_ptr + og_offset + d_idx, vec_grad_g.to(grad_g_ptr.dtype
        .element_ty), mask=mask)
    tl.store(grad_r_ptr + rkv_offset + d_idx, vec_grad_r.to(grad_r_ptr.
        dtype.element_ty), mask=mask)
    tl.store(grad_k_ptr + rkv_offset + d_idx, vec_grad_k.to(grad_k_ptr.
        dtype.element_ty), mask=mask)
    tl.store(grad_v_ptr + rkv_offset + d_idx, vec_grad_v.to(grad_v_ptr.
        dtype.element_ty), mask=mask)
    tl.store(grad_r_k_intermediate_ptr + rkv_offset + d_idx, local_grad_rk.
        to(grad_r_k_intermediate_ptr.dtype.element_ty), mask=mask)


def gate_output_correction_backward_triton(grad_output, o, r, k, r_k, v, g):
    batch_size, seq_len, _ = o.shape
    num_heads, head_dim = r.shape[-2], r.shape[-1]
    grad_o = torch.empty_like(o)
    grad_r = torch.empty_like(r)
    grad_k = torch.empty_like(k)
    grad_v = torch.empty_like(v)
    grad_g = torch.empty_like(g)
    grad_r_k = torch.empty_like(r, dtype=torch.float32)
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    for t_offset in range(0, seq_len, 65536):
        T_SIZE = min(65536, seq_len - t_offset)

        def grid(meta):
            return batch_size, triton.cdiv(T_SIZE, meta['BT']), num_heads
        gate_output_correction_bwd_kernel[grid](grad_output, o, r, k, r_k,
            v, g, grad_o, grad_r, grad_k, grad_r_k, grad_v, grad_g, r.
            stride(0), r.stride(1), r.stride(2), o.stride(0), o.stride(1),
            r_k.stride(0), T_SIZE, t_offset, num_heads=num_heads, head_dim=
            head_dim, BLOCK_SIZE_D=BLOCK_SIZE_D)
    grad_r_k = grad_r_k.sum(dim=(0, 1)).type_as(r_k)
    return grad_o, grad_r, grad_k, grad_r_k, grad_v, grad_g


# Backward method (kernel launch code)
@autocast_custom_bwd
@input_guard
def _GateOutputCorrection_backward(ctx, grad_output):
    o, r, k, r_k, v, g = ctx.saved_tensors
    return gate_output_correction_backward_triton(grad_output, o, r, k, r_k,
        v, g)


# ============================================================
# autograd.Function Class Definition
# ============================================================

class GateOutputCorrection(torch.autograd.Function):

    @staticmethod
    @autocast_custom_fwd
    @input_guard
    def forward(ctx, o, r, k, r_k, v, g):
        assert r_k.dim() == 2 and r_k.shape[0] == r.shape[-2] and r_k.shape[1
            ] == r.shape[-1]
        batch_size, seq_len, _ = o.shape
        num_heads, head_dim = r.shape[-2], r.shape[-1]
        output = torch.empty_like(o)
        ctx.save_for_backward(o, r, k, r_k, v, g)
        for t in range(0, seq_len, 65536):
            T_OFFSET = t
            T_SIZE = min(65536, seq_len - t)

            def grid(meta):
                return batch_size, triton.cdiv(T_SIZE, meta['BT']), num_heads
            gate_output_correction_fwd_kernel[grid](o, r, k, r_k, v, g,
                output, o.stride(0), o.stride(1), r.stride(0), r.stride(1),
                r.stride(2), v.stride(0), v.stride(1), v.stride(2), r_k.
                stride(0), T_SIZE, T_OFFSET, num_heads, head_dim,
                BLOCK_SIZE_D=triton.next_power_of_2(head_dim))
        return output

    @staticmethod
    @autocast_custom_bwd
    @input_guard
    def backward(ctx, grad_output):
        o, r, k, r_k, v, g = ctx.saved_tensors
        return gate_output_correction_backward_triton(grad_output, o, r, k,
            r_k, v, g)
