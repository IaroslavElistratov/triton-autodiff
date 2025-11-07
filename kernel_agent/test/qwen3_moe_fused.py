# CREDIT: /woct0rdho_transformers_qwen3_moe_fused/transformers-qwen3-moe-fused-master/qwen3_moe_fused/grouped_gemm/interface.py

import math
import os
from itertools import product
from typing import Any, Optional

import torch
import triton
import triton.language as tl

from kernel_agent.autodiff import autodiff


######## utils

DEFAULT_M_BLOCK_SIZES = [16, 32, 64, 128, 256]
DEFAULT_N_BLOCK_SIZES = [16, 32, 64, 128, 256]
DEFAULT_K_BLOCK_SIZES = [16, 32, 64, 128, 256]
DEFAULT_NUM_WARPS = [4, 8]
DEFAULT_NUM_STAGES = [3, 4, 5, 6]


def get_num_sms() -> int:
    return torch.cuda.get_device_properties("cuda").multi_processor_count


def get_autotune_configs() -> list[triton.Config]:
    configs = []
    for m, n, k, w, s in product(
        DEFAULT_M_BLOCK_SIZES,
        DEFAULT_N_BLOCK_SIZES,
        DEFAULT_K_BLOCK_SIZES,
        DEFAULT_NUM_WARPS,
        DEFAULT_NUM_STAGES,
    ):
        configs.append(
            triton.Config({"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n, "BLOCK_SIZE_K": k}, num_warps=w, num_stages=s)
        )
    return configs


def _get_device_properties() -> dict[str, Any]:
    return triton.runtime.driver.active.utils.get_device_properties(torch.cuda.current_device())


def _exceeds_smem_capacity(
    num_stages: int,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    dtype: torch.dtype,
    smem_size: int,
    slack: int = 0,
) -> bool:
    return (
        num_stages * BLOCK_SIZE_K * (BLOCK_SIZE_M + BLOCK_SIZE_N) + BLOCK_SIZE_M * BLOCK_SIZE_N
    ) * dtype.itemsize > smem_size + slack


def _common_prune_criteria(config: triton.Config, kwargs: dict[str, Any]) -> bool:
    num_stages = config.num_stages
    BLOCK_SIZE_M = config.kwargs["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = config.kwargs["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = config.kwargs["BLOCK_SIZE_K"]
    dtype = kwargs["x_ptr"].dtype
    device_properties = _get_device_properties()
    smem_size = device_properties["max_shared_mem"]
    if _exceeds_smem_capacity(num_stages, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype, smem_size):
        return True

    M = kwargs["M"]
    N = kwargs["N"]
    K = kwargs["K"]
    num_experts = kwargs["NUM_EXPERTS"]
    tokens_per_expert = M // num_experts
    max_block_size_M = max(tokens_per_expert * 2, DEFAULT_M_BLOCK_SIZES[0])
    max_block_size_N = max(N, DEFAULT_N_BLOCK_SIZES[0])
    max_block_size_K = max(K, DEFAULT_K_BLOCK_SIZES[0])
    if BLOCK_SIZE_M > max_block_size_M:
        return True
    if BLOCK_SIZE_N > max_block_size_N:
        return True
    if BLOCK_SIZE_K > max_block_size_K:
        return True

    min_block_size_M = min(triton.next_power_of_2(tokens_per_expert // 2 + 1), 64)
    min_block_size_N = min(triton.next_power_of_2(N // 2 + 1), 64)
    min_block_size_K = min(triton.next_power_of_2(K // 2 + 1), 64)
    if BLOCK_SIZE_M * BLOCK_SIZE_N < min_block_size_M * min_block_size_N:
        return True
    if BLOCK_SIZE_M * BLOCK_SIZE_K < min_block_size_M * min_block_size_K:
        return True
    if BLOCK_SIZE_N * BLOCK_SIZE_K < min_block_size_N * min_block_size_K:
        return True

    return False


def prune_configs(configs: list[triton.Config], args, **kwargs) -> list[triton.Config]:
    pruned_configs = []
    for config in configs:
        if _common_prune_criteria(config, args):
            continue
        pruned_configs.append(config)
    return pruned_configs


# We need to autotune on batch size only when benchmarking with a large range of batch sizes
def get_autotune_keys() -> list[str]:
    if os.getenv("AUTOTUNE_BATCH_SIZE", "0") == "1":
        return ["M", "N", "K", "NUM_EXPERTS"]
    else:
        return ["N", "K", "NUM_EXPERTS"]

########




# y[m, n] = sum_k w[s[m], n, k] * x[m, k]


@triton.autotune(
    configs=get_autotune_configs(),
    prune_configs_by={"early_config_prune": prune_configs},
    key=get_autotune_keys(),
)
@triton.jit
def _grouped_gemm_forward_kernel(
    # Pointers
    x_ptr,
    w_ptr,
    m_sizes_ptr,
    y_ptr,
    # Dimensions
    M: int,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    # Strides
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_we: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    # Metadata
    BLOCK_SIZE_M: tl.constexpr = 64,
    BLOCK_SIZE_N: tl.constexpr = 64,
    BLOCK_SIZE_K: tl.constexpr = 64,
) -> None:
    tidx = tl.program_id(0)
    m_end = 0
    processed_tiles = 0
    for expert_idx in range(NUM_EXPERTS):
        m_start = m_end
        m_size = tl.load(m_sizes_ptr + expert_idx).to(tl.int32)
        m_end = m_start + m_size
        if m_size > 0:
            # tiles for this group's GEMM
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
            num_tiles_per_expert = num_m_tiles * num_n_tiles

            # Lower bound and upper bound are defined relative to the total tiles processed so far
            # This ensures that we are only processing tiles for the current expert group AND
            # we never exceed the total number of tiles for all expert groups
            while tidx >= processed_tiles and tidx < processed_tiles + num_tiles_per_expert:
                tile_idx = tidx - processed_tiles

                # Output tile for this thread block for this expert group
                # TODO: Check if L2 cache re-use for this order is optimal
                tile_m_idx = tile_idx % num_m_tiles
                tile_n_idx = tile_idx // num_m_tiles

                offs_k = tl.arange(0, BLOCK_SIZE_K)

                offs_m = m_start + tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                x_ptrs = x_ptr + stride_xm * offs_m[:, None] + stride_xk * offs_k[None, :]
                mask_m = offs_m < m_end

                offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                w_ptrs = w_ptr + stride_we * expert_idx + stride_wn * offs_n[:, None] + stride_wk * offs_k[None, :]
                mask_n = offs_n < N

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                # GEMM main loop
                for _ in range(tl.cdiv(K, BLOCK_SIZE_K)):
                    mask_k = offs_k < K
                    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :])
                    w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :])

                    accumulator += tl.dot(x, w.T)

                    offs_k += BLOCK_SIZE_K
                    x_ptrs += stride_xk * BLOCK_SIZE_K
                    w_ptrs += stride_wk * BLOCK_SIZE_K
                y = accumulator.to(y_ptr.dtype.element_ty)

                y_ptrs = y_ptr + stride_ym * offs_m[:, None] + stride_yn * offs_n[None, :]
                tl.store(y_ptrs, y, mask=mask_m[:, None] & mask_n[None, :])

                # Move to the next tile within this expert group
                tidx += NUM_SMS

            # Update the total tiles count for the next expert group
            processed_tiles += num_tiles_per_expert


def is_int_tensor(x: torch.Tensor) -> bool:
    return x.dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }


@autodiff(idxs_buffers=(0, 1))
def grouped_gemm_forward(
    x: torch.Tensor, w: torch.Tensor, m_sizes: torch.Tensor, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
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
    _grouped_gemm_forward_kernel[grid](
        # Pointers
        x,
        w,
        m_sizes,
        y,
        # Dimensions
        M,
        N,
        K,
        E,
        NUM_SMS,
        # Strides
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        y.stride(0),
        y.stride(1),
    )
    return y



SWEEP = [
    {"M": 1024, "N": 768, "K": 2048, "NUM_EXPERTS": 128, "required": True},
    {"M": 4096, "N": 768, "K": 2048, "NUM_EXPERTS": 128, "required": True},
    {"M": 8192, "N": 768, "K": 2048, "NUM_EXPERTS": 128, "required": True},
    {"M": 16384, "N": 768, "K": 2048, "NUM_EXPERTS": 128, "required": False},
]


def make_args(dims, device="cuda", dtype=torch.bfloat16):
    dims = dict(dims)
    M = dims["M"]
    N = dims["N"]
    K = dims["K"]
    num_experts = dims["NUM_EXPERTS"]
    x = torch.randn((M, K), device=device, dtype=dtype)
    weight_scale = 1.0 / math.sqrt(K)
    w = weight_scale * torch.randn((num_experts, N, K), device=device, dtype=dtype)
    # kernel expects contiguous token spans per expert instead of explicit routing
    tokens_per_expert = M // num_experts
    m_sizes = torch.full((num_experts,), tokens_per_expert, device=device, dtype=torch.int32)
    # distributing the remainder avoids starving tail experts when M % NUM_EXPERTS != 0
    remainder = M - tokens_per_expert * num_experts
    if remainder > 0:
        m_sizes[:remainder] += 1
    return (x, w, m_sizes), {"dtype": dtype}


def flops(dims):
    M = dims["M"]
    N = dims["N"]
    K = dims["K"]
    # downstream perf chart already reports M*N*K as the flop proxy
    return M * N * K


def setup():
    (x, w, m_sizes), kwargs = make_args(SWEEP[0])
    grouped_gemm_forward(x, w, m_sizes, **kwargs)


if __name__ == "__main__":
    (x, w, m_sizes), kwargs = make_args(SWEEP[0])
    grouped_gemm_forward(x, w, m_sizes, **kwargs)
