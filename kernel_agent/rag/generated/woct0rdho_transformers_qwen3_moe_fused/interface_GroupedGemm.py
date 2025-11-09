# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/woct0rdho/transformers-qwen3-moe-fused
# Source-Files: qwen3_moe_fused/grouped_gemm/interface.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_n8hgmd0z/transformers-qwen3-moe-fused-master/qwen3_moe_fused/grouped_gemm/interface.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

def get_num_sms() ->int:
    return torch.cuda.get_device_properties('cuda').multi_processor_count


def is_int_tensor(x: torch.Tensor) ->bool:
    return x.dtype in {torch.uint8, torch.int8, torch.int16, torch.int32,
        torch.int64}


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=get_autotune_configs(), prune_configs_by={
    'early_config_prune': prune_configs}, key=get_autotune_keys())
@triton.jit
def _grouped_gemm_forward_kernel(x_ptr, w_ptr, m_sizes_ptr, y_ptr, M: int,
    N: tl.constexpr, K: tl.constexpr, NUM_EXPERTS: tl.constexpr, NUM_SMS:
    tl.constexpr, stride_xm: tl.constexpr, stride_xk: tl.constexpr,
    stride_we: tl.constexpr, stride_wn: tl.constexpr, stride_wk: tl.
    constexpr, stride_ym: tl.constexpr, stride_yn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr=64, BLOCK_SIZE_N: tl.constexpr=64,
    BLOCK_SIZE_K: tl.constexpr=64) ->None:
    tidx = tl.program_id(0)
    m_end = 0
    processed_tiles = 0
    for expert_idx in range(NUM_EXPERTS):
        m_start = m_end
        m_size = tl.load(m_sizes_ptr + expert_idx).to(tl.int32)
        m_end = m_start + m_size
        if m_size > 0:
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
            num_tiles_per_expert = num_m_tiles * num_n_tiles
            while (tidx >= processed_tiles and tidx < processed_tiles +
                num_tiles_per_expert):
                tile_idx = tidx - processed_tiles
                tile_m_idx = tile_idx % num_m_tiles
                tile_n_idx = tile_idx // num_m_tiles
                offs_k = tl.arange(0, BLOCK_SIZE_K)
                offs_m = m_start + tile_m_idx * BLOCK_SIZE_M + tl.arange(0,
                    BLOCK_SIZE_M)
                x_ptrs = x_ptr + stride_xm * offs_m[:, None
                    ] + stride_xk * offs_k[None, :]
                mask_m = offs_m < m_end
                offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                w_ptrs = w_ptr + stride_we * expert_idx + stride_wn * offs_n[
                    :, None] + stride_wk * offs_k[None, :]
                mask_n = offs_n < N
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=
                    tl.float32)
                for _ in range(tl.cdiv(K, BLOCK_SIZE_K)):
                    mask_k = offs_k < K
                    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :])
                    w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :])
                    accumulator += tl.dot(x, w.T)
                    offs_k += BLOCK_SIZE_K
                    x_ptrs += stride_xk * BLOCK_SIZE_K
                    w_ptrs += stride_wk * BLOCK_SIZE_K
                y = accumulator.to(y_ptr.dtype.element_ty)
                y_ptrs = y_ptr + stride_ym * offs_m[:, None
                    ] + stride_yn * offs_n[None, :]
                tl.store(y_ptrs, y, mask=mask_m[:, None] & mask_n[None, :])
                tidx += NUM_SMS
            processed_tiles += num_tiles_per_expert


def grouped_gemm_forward(x: torch.Tensor, w: torch.Tensor, m_sizes: torch.
    Tensor, dtype: Optional[torch.dtype]=None) ->torch.Tensor:
    assert x.is_cuda
    assert w.device == x.device
    assert m_sizes.device == x.device
    assert is_int_tensor(m_sizes)
    assert x.is_contiguous()
    assert w.is_contiguous()
    assert m_sizes.is_contiguous()
    assert x.ndim == 2
    assert w.ndim == 3
    assert m_sizes.ndim == 1
    M, K = x.shape
    E, N, _ = w.shape
    assert w.shape[2] == K
    assert m_sizes.numel() == E
    if dtype is None:
        dtype = x.dtype
    y = torch.empty((M, N), device=x.device, dtype=dtype)
    NUM_SMS = get_num_sms()
    grid = lambda META: (NUM_SMS,)
    _grouped_gemm_forward_kernel[grid](x, w, m_sizes, y, M, N, K, E,
        NUM_SMS, x.stride(0), x.stride(1), w.stride(0), w.stride(1), w.
        stride(2), y.stride(0), y.stride(1))
    return y


# Forward method (kernel launch code)
def _GroupedGemm_forward(ctx, x, w, m_sizes):
    ctx.save_for_backward(x, w, m_sizes)
    return grouped_gemm_forward(x, w, m_sizes)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=get_autotune_configs(), prune_configs_by={
    'early_config_prune': prune_configs}, key=get_autotune_keys())
@triton.jit
def _grouped_gemm_backward_dw_kernel(x_ptr, y_ptr, m_sizes_ptr, w_ptr, M:
    int, N: tl.constexpr, K: tl.constexpr, NUM_EXPERTS: tl.constexpr,
    NUM_SMS: tl.constexpr, stride_xm: tl.constexpr, stride_xk: tl.constexpr,
    stride_ym: tl.constexpr, stride_yn: tl.constexpr, stride_we: tl.
    constexpr, stride_wn: tl.constexpr, stride_wk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr=64, BLOCK_SIZE_N: tl.constexpr=64,
    BLOCK_SIZE_K: tl.constexpr=64) ->None:
    tidx = tl.program_id(0)
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles_per_expert = num_n_tiles * num_k_tiles
    for tile_idx in range(tidx, num_tiles_per_expert, NUM_SMS):
        tile_n_idx = tile_idx % num_n_tiles
        tile_k_idx = tile_idx // num_n_tiles
        offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tile_k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_n = offs_n < N
        mask_k = offs_k < K
        m_end = 0
        for expert_idx in range(NUM_EXPERTS):
            m_start = m_end
            m_size = tl.load(m_sizes_ptr + expert_idx).to(tl.int32)
            m_end = m_start + m_size
            if m_size > 0:
                offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
                x_ptrs = x_ptr + stride_xm * offs_m[:, None
                    ] + stride_xk * offs_k[None, :]
                y_ptrs = y_ptr + stride_ym * offs_m[:, None
                    ] + stride_yn * offs_n[None, :]
                accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=
                    tl.float32)
                for _ in range(tl.cdiv(m_size, BLOCK_SIZE_M)):
                    mask_m = offs_m < m_end
                    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :])
                    y = tl.load(y_ptrs, mask=mask_m[:, None] & mask_n[None, :])
                    accumulator += tl.dot(y.T, x)
                    offs_m += BLOCK_SIZE_M
                    x_ptrs += stride_xm * BLOCK_SIZE_M
                    y_ptrs += stride_ym * BLOCK_SIZE_M
                w = accumulator.to(w_ptr.dtype.element_ty)
                w_ptrs = w_ptr + stride_we * expert_idx + stride_wn * offs_n[
                    :, None] + stride_wk * offs_k[None, :]
                tl.store(w_ptrs, w, mask=mask_n[:, None] & mask_k[None, :])


@triton.autotune(configs=get_autotune_configs(), prune_configs_by={
    'early_config_prune': prune_configs}, key=get_autotune_keys())
@triton.jit
def _grouped_gemm_forward_transposed_kernel(x_ptr, w_ptr, m_sizes_ptr,
    y_ptr, M: int, N: tl.constexpr, K: tl.constexpr, NUM_EXPERTS: tl.
    constexpr, NUM_SMS: tl.constexpr, stride_xm: tl.constexpr, stride_xk:
    tl.constexpr, stride_we: tl.constexpr, stride_wk: tl.constexpr,
    stride_wn: tl.constexpr, stride_ym: tl.constexpr, stride_yn: tl.
    constexpr, BLOCK_SIZE_M: tl.constexpr=64, BLOCK_SIZE_N: tl.constexpr=64,
    BLOCK_SIZE_K: tl.constexpr=64) ->None:
    tidx = tl.program_id(0)
    m_end = 0
    processed_tiles = 0
    for expert_idx in range(NUM_EXPERTS):
        m_start = m_end
        m_size = tl.load(m_sizes_ptr + expert_idx).to(tl.int32)
        m_end = m_start + m_size
        if m_size > 0:
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
            num_tiles_per_expert = num_m_tiles * num_n_tiles
            while (tidx >= processed_tiles and tidx < processed_tiles +
                num_tiles_per_expert):
                tile_idx = tidx - processed_tiles
                tile_m_idx = tile_idx % num_m_tiles
                tile_n_idx = tile_idx // num_m_tiles
                offs_k = tl.arange(0, BLOCK_SIZE_K)
                offs_m = m_start + tile_m_idx * BLOCK_SIZE_M + tl.arange(0,
                    BLOCK_SIZE_M)
                x_ptrs = x_ptr + stride_xm * offs_m[:, None
                    ] + stride_xk * offs_k[None, :]
                mask_m = offs_m < m_start + m_size
                offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                w_ptrs = w_ptr + stride_we * expert_idx + stride_wn * offs_n[
                    :, None] + stride_wk * offs_k[None, :]
                mask_n = offs_n < N
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=
                    tl.float32)
                for _ in range(tl.cdiv(K, BLOCK_SIZE_K)):
                    mask_k = offs_k < K
                    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :])
                    w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :])
                    accumulator += tl.dot(x, w.T)
                    offs_k += BLOCK_SIZE_K
                    x_ptrs += stride_xk * BLOCK_SIZE_K
                    w_ptrs += stride_wk * BLOCK_SIZE_K
                y = accumulator.to(y_ptr.dtype.element_ty)
                y_ptrs = y_ptr + stride_ym * offs_m[:, None
                    ] + stride_yn * offs_n[None, :]
                tl.store(y_ptrs, y, mask=mask_m[:, None] & mask_n[None, :])
                tidx += NUM_SMS
            processed_tiles += num_tiles_per_expert


def grouped_gemm_backward_dw(x: torch.Tensor, y: torch.Tensor, m_sizes:
    torch.Tensor, dtype: torch.dtype) ->torch.Tensor:
    assert x.is_cuda
    assert y.device == x.device
    assert m_sizes.device == x.device
    assert is_int_tensor(m_sizes)
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert m_sizes.is_contiguous()
    assert x.ndim == 2
    assert y.ndim == 2
    assert m_sizes.ndim == 1
    M, K = x.shape
    _, N = y.shape
    assert y.shape[0] == M
    E = m_sizes.numel()
    w = torch.zeros((E, N, K), device=x.device, dtype=dtype)
    NUM_SMS = get_num_sms()
    grid = lambda META: (NUM_SMS,)
    _grouped_gemm_backward_dw_kernel[grid](x, y, m_sizes, w, M, N, K, E,
        NUM_SMS, x.stride(0), x.stride(1), y.stride(0), y.stride(1), w.
        stride(0), w.stride(1), w.stride(2))
    return w


def grouped_gemm_forward_transposed(x: torch.Tensor, w: torch.Tensor,
    m_sizes: torch.Tensor, dtype: Optional[torch.dtype]=None) ->torch.Tensor:
    assert x.is_cuda
    assert w.device == x.device
    assert m_sizes.device == x.device
    assert is_int_tensor(m_sizes)
    assert x.is_contiguous()
    assert w.is_contiguous()
    assert m_sizes.is_contiguous()
    assert x.ndim == 2
    assert w.ndim == 3
    assert m_sizes.ndim == 1
    M, K = x.shape
    E, _, N = w.shape
    assert w.shape[1] == K
    assert m_sizes.numel() == E
    if dtype is None:
        dtype = x.dtype
    y = torch.empty((M, N), device=x.device, dtype=dtype)
    NUM_SMS = get_num_sms()
    grid = lambda META: (NUM_SMS,)
    _grouped_gemm_forward_transposed_kernel[grid](x, w, m_sizes, y, M, N, K,
        E, NUM_SMS, x.stride(0), x.stride(1), w.stride(0), w.stride(1), w.
        stride(2), y.stride(0), y.stride(1))
    return y


# Backward method (kernel launch code)
def _GroupedGemm_backward(ctx, dy):
    x, w, m_sizes = ctx.saved_tensors
    if x.requires_grad:
        dx = grouped_gemm_forward_transposed(dy, w, m_sizes, x.dtype)
    else:
        dx = None
    if w.requires_grad:
        dw = grouped_gemm_backward_dw(x, dy, m_sizes, w.dtype)
    else:
        dw = None
    return dx, dw, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class GroupedGemm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, w, m_sizes):
        ctx.save_for_backward(x, w, m_sizes)
        return grouped_gemm_forward(x, w, m_sizes)

    @staticmethod
    def backward(ctx, dy):
        x, w, m_sizes = ctx.saved_tensors
        if x.requires_grad:
            dx = grouped_gemm_forward_transposed(dy, w, m_sizes, x.dtype)
        else:
            dx = None
        if w.requires_grad:
            dw = grouped_gemm_backward_dw(x, dy, m_sizes, w.dtype)
        else:
            dw = None
        return dx, dw, None
