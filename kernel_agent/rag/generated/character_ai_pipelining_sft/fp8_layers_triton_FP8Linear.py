# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/character-ai/pipelining-sft
# Source-Files: models/deepseek_v3/fp8_layers_triton.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_eyyroz7s/pipelining-sft-main/models/deepseek_v3/fp8_layers_triton.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

@triton.jit
def _fp8_cast_kernel(x_ptr, out_ptr, scale_ptr, M, N, N_padded, stride_x_m,
    stride_x_n, stride_out_m, stride_out_n, stride_scale_m, stride_scale_n,
    BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    num_chunks = (N_padded + BLOCK_SIZE - 1) // BLOCK_SIZE
    for chunk_idx in range(num_chunks):
        col_start = chunk_idx * BLOCK_SIZE
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x_ptrs = x_ptr + row_idx * stride_x_m + cols * stride_x_n
        out_ptrs = out_ptr + row_idx * stride_out_m + cols * stride_out_n
        x_chunk = tl.load(x_ptrs, mask=mask, other=0.0)
        x_abs = tl.abs(x_chunk)
        amax = tl.max(x_abs, axis=0)
        amax = tl.maximum(amax, 0.0001)
        scale = amax / 448.0
        scale_ptr_loc = (scale_ptr + row_idx * stride_scale_m + chunk_idx *
            stride_scale_n)
        tl.store(scale_ptr_loc, scale)
        scale_factor = 448.0 / amax
        x_scaled = x_chunk * scale_factor
        x_fp8 = x_scaled.to(tl.float8e4nv)
        tl.store(out_ptrs, x_fp8, mask=mask)


@triton.jit
def _fp8_per_block_cast_kernel(x_ptr, out_ptr, scale_ptr, M, N, M_padded,
    N_padded, stride_x_m, stride_x_n, stride_out_m, stride_out_n,
    stride_scale_m, stride_scale_n, BLOCK_M: tl.constexpr=128, BLOCK_N: tl.
    constexpr=128):
    block_id_m = tl.program_id(0)
    block_id_n = tl.program_id(1)
    m_start = block_id_m * BLOCK_M
    n_start = block_id_n * BLOCK_N
    block_max = 0.0
    TILE_M: tl.constexpr = 32
    TILE_N: tl.constexpr = 32
    for tile_m in range(0, BLOCK_M, TILE_M):
        for tile_n in range(0, BLOCK_N, TILE_N):
            rm = tl.arange(0, TILE_M)
            rn = tl.arange(0, TILE_N)
            rows = m_start + tile_m + rm[:, None]
            cols = n_start + tile_n + rn[None, :]
            mask = (rows < M) & (cols < N)
            ptrs = x_ptr + rows * stride_x_m + cols * stride_x_n
            tile_data = tl.load(ptrs, mask=mask, other=0.0)
            tile_abs = tl.abs(tile_data)
            tile_max = tl.max(tile_abs)
            block_max = tl.maximum(block_max, tile_max)
    block_max = tl.maximum(block_max, 0.0001)
    scale = block_max / 448.0
    scale_factor = 448.0 / block_max
    scale_ptr_loc = (scale_ptr + block_id_m * stride_scale_m + block_id_n *
        stride_scale_n)
    tl.store(scale_ptr_loc, scale)
    for tile_m in range(0, BLOCK_M, TILE_M):
        for tile_n in range(0, BLOCK_N, TILE_N):
            rm = tl.arange(0, TILE_M)
            rn = tl.arange(0, TILE_N)
            rows = m_start + tile_m + rm[:, None]
            cols = n_start + tile_n + rn[None, :]
            mask = (rows < M) & (cols < N)
            in_ptrs = x_ptr + rows * stride_x_m + cols * stride_x_n
            out_ptrs = out_ptr + rows * stride_out_m + cols * stride_out_n
            tile_data = tl.load(in_ptrs, mask=mask, other=0.0)
            tile_scaled = tile_data * scale_factor
            tile_fp8 = tile_scaled.to(tl.float8e4nv)
            tl.store(out_ptrs, tile_fp8, mask=mask)


def per_block_cast_to_fp8_triton(x: torch.Tensor) ->Tuple[torch.Tensor,
    torch.Tensor]:
    """
    Triton implementation of per-block FP8 casting that matches the original PyTorch version.
    
    Each 128x128 block gets its own scale factor based on the maximum absolute value
    in that block.
    
    Args:
        x: Input tensor of shape (M, N)
    
    Returns:
        x_fp8: FP8 quantized tensor of shape (M, N)
        scales: Scale factors of shape (num_block_rows, num_block_cols)
                where num_block_rows = ceil(M/128), num_block_cols = ceil(N/128)
    """
    assert x.dim() == 2, f'Expected 2D tensor, got {x.dim()}D'
    M, N = x.shape
    BLOCK_SIZE = 128
    M_padded = ceil_div(M, BLOCK_SIZE) * BLOCK_SIZE
    N_padded = ceil_div(N, BLOCK_SIZE) * BLOCK_SIZE
    num_block_rows = M_padded // BLOCK_SIZE
    num_block_cols = N_padded // BLOCK_SIZE
    x = x.contiguous()
    x_fp8 = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((num_block_rows, num_block_cols), dtype=torch.
        float32, device=x.device)
    grid = num_block_rows, num_block_cols
    _fp8_per_block_cast_kernel[grid](x, x_fp8, scales, M, N, M_padded,
        N_padded, x.stride(0), x.stride(1), x_fp8.stride(0), x_fp8.stride(1
        ), scales.stride(0), scales.stride(1), num_warps=8)
    return x_fp8, scales


def per_token_cast_to_fp8_triton(x: torch.Tensor) ->Tuple[torch.Tensor,
    torch.Tensor]:
    """
    Correct implementation of per-token FP8 casting that matches the original PyTorch version.
    
    Args:
        x: Input tensor of shape (M, N)
    
    Returns:
        x_fp8: FP8 quantized tensor of shape (M, N)
        scales: Scale factors of shape (M, num_chunks) where num_chunks = ceil(N/128)
    """
    assert x.dim() == 2, f'Expected 2D tensor, got {x.dim()}D'
    M, N = x.shape
    BLOCK_SIZE = 128
    pad_size = (BLOCK_SIZE - N % BLOCK_SIZE) % BLOCK_SIZE
    N_padded = N + pad_size
    num_chunks = N_padded // BLOCK_SIZE
    x = x.contiguous()
    x_fp8_padded = torch.empty((M, N_padded), dtype=torch.float8_e4m3fn,
        device=x.device)
    scales = torch.empty((M, num_chunks), dtype=torch.float32, device=x.device)
    grid = M,
    _fp8_cast_kernel[grid](x, x_fp8_padded, scales, M, N, N_padded, x.
        stride(0), x.stride(1), x_fp8_padded.stride(0), x_fp8_padded.stride
        (1), scales.stride(0), scales.stride(1), BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4)
    return x_fp8_padded[:, :N], scales


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

# Forward method (kernel launch code)
def _FP8Linear_forward(ctx, x, weight, bias=None):
    assert x.dtype == torch.bfloat16, f'only allow bf16 inputs to fp8 linear'
    shape = x.shape
    x = x.view(-1, shape[-1])
    x_fp8 = per_token_cast_to_fp8_triton(x)
    x_fp8 = x_fp8[0].contiguous(), get_col_major_tma_aligned_tensor(x_fp8[1
        ].contiguous())
    weight_fp8 = per_block_cast_to_fp8_triton(weight)
    ctx.save_for_backward(x, weight)
    out_dim = weight.shape[0]
    out = torch.zeros((x.shape[0], out_dim), device=x.device, dtype=x.dtype)
    deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, weight_fp8, out)
    if len(shape) == 3:
        out = out.view(shape[0], shape[1], out_dim)
    return out


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

# Backward method (kernel launch code)
def _FP8Linear_backward(ctx, grad_output):
    grad_input = grad_weight = grad_bias = None
    shape = grad_output.shape
    grad_output = grad_output.view(-1, shape[-1])
    x, weight = ctx.saved_tensors
    if ctx.needs_input_grad[1]:
        dy_fp8 = per_token_cast_to_fp8_triton(grad_output.t().contiguous())
        x_fp8 = per_token_cast_to_fp8_triton(x.t().contiguous())
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        deep_gemm.wgrad_gemm_fp8_fp8_fp32_nt(dy_fp8, x_fp8, grad_weight)
    if ctx.needs_input_grad[0]:
        dy_fp8 = per_token_cast_to_fp8_triton(grad_output.contiguous())
        dy_fp8 = dy_fp8[0].contiguous(), get_col_major_tma_aligned_tensor(
            dy_fp8[1].contiguous())
        weight_fp8 = per_block_cast_to_fp8_triton(weight.t().contiguous())
        grad_input = torch.zeros_like(x)
        deep_gemm.gemm_fp8_fp8_bf16_nt(dy_fp8, weight_fp8, grad_input)
        if len(shape) == 3:
            in_dim = weight.shape[1]
            grad_input = grad_input.view(shape[0], shape[1], in_dim)
    return grad_input, grad_weight, grad_bias


# ============================================================
# autograd.Function Class Definition
# ============================================================

class FP8Linear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias=None):
        assert x.dtype == torch.bfloat16, f'only allow bf16 inputs to fp8 linear'
        shape = x.shape
        x = x.view(-1, shape[-1])
        x_fp8 = per_token_cast_to_fp8_triton(x)
        x_fp8 = x_fp8[0].contiguous(), get_col_major_tma_aligned_tensor(x_fp8
            [1].contiguous())
        weight_fp8 = per_block_cast_to_fp8_triton(weight)
        ctx.save_for_backward(x, weight)
        out_dim = weight.shape[0]
        out = torch.zeros((x.shape[0], out_dim), device=x.device, dtype=x.dtype
            )
        deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, weight_fp8, out)
        if len(shape) == 3:
            out = out.view(shape[0], shape[1], out_dim)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_weight = grad_bias = None
        shape = grad_output.shape
        grad_output = grad_output.view(-1, shape[-1])
        x, weight = ctx.saved_tensors
        if ctx.needs_input_grad[1]:
            dy_fp8 = per_token_cast_to_fp8_triton(grad_output.t().contiguous())
            x_fp8 = per_token_cast_to_fp8_triton(x.t().contiguous())
            grad_weight = torch.zeros_like(weight, dtype=torch.float32)
            deep_gemm.wgrad_gemm_fp8_fp8_fp32_nt(dy_fp8, x_fp8, grad_weight)
        if ctx.needs_input_grad[0]:
            dy_fp8 = per_token_cast_to_fp8_triton(grad_output.contiguous())
            dy_fp8 = dy_fp8[0].contiguous(), get_col_major_tma_aligned_tensor(
                dy_fp8[1].contiguous())
            weight_fp8 = per_block_cast_to_fp8_triton(weight.t().contiguous())
            grad_input = torch.zeros_like(x)
            deep_gemm.gemm_fp8_fp8_bf16_nt(dy_fp8, weight_fp8, grad_input)
            if len(shape) == 3:
                in_dim = weight.shape[1]
                grad_input = grad_input.view(shape[0], shape[1], in_dim)
        return grad_input, grad_weight, grad_bias
