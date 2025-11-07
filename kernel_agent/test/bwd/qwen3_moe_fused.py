# CREDIT: /woct0rdho_transformers_qwen3_moe_fused/transformers-qwen3-moe-fused-master/qwen3_moe_fused/grouped_gemm/interface.py

# python -m kernel_agent.test.bwd.qwen3_moe_fused
# moe_fused_linear_all:
#           N  grouped_gemm_dw  grouped_gemm_dx  grouped_gemm_fwd
# 0    1024.0       640.498435      1216.945480       1213.688199
# 1    2048.0      1254.558894      2367.322799       2389.321176
# 2    3072.0      1813.818477      3334.804546       3533.533248
# 3    4096.0      2343.432009      4072.139789       4641.107123
# 4    5120.0      2840.963575      4517.736226       5718.920576
# 5    6144.0      3312.600225      4855.377093       6763.794437
# 6    7168.0      3702.956872      5059.474177       7721.185132
# 7    8192.0      3961.815570      5239.332450       8573.290722
# 8    9216.0      4090.119591      5399.104018       9144.188684
# 9   10240.0      4226.105378      5498.172701       9414.207451
# 10  11264.0      4455.633785      5491.914389       9574.794525
# 11  12288.0      4695.280543      5551.846170       9674.666885
# 12  13312.0      4919.079733      5650.162468       9812.414547
# 13  14336.0      5125.161662      5706.281974       9811.510947
# 14  15360.0      5319.757817      5755.740184       9804.612294
# 15  16384.0      5518.650822      5884.505820       9917.138441


from typing import Optional

import torch
import triton
import triton.language as tl

from ..qwen3_moe_fused import grouped_gemm_forward, get_autotune_configs, prune_configs, get_autotune_keys, get_num_sms



# w[e, n, k] = sum_m if(s[m] == e) y[m, n] * x[m, k]


def is_int_tensor(x: torch.Tensor) -> bool:
    return x.dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }


@triton.autotune(
    configs=get_autotune_configs(),
    prune_configs_by={"early_config_prune": prune_configs},
    key=get_autotune_keys(),
)
@triton.jit
def _grouped_gemm_backward_dw_kernel(
    # Pointers
    x_ptr,
    y_ptr,
    m_sizes_ptr,
    w_ptr,
    # Dimensions
    M: int,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    # Strides
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    stride_we: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    # Metadata
    BLOCK_SIZE_M: tl.constexpr = 64,
    BLOCK_SIZE_N: tl.constexpr = 64,
    BLOCK_SIZE_K: tl.constexpr = 64,
) -> None:
    tidx = tl.program_id(0)

    # Output tiles per expert, since each expert weight matrix is [N, K]
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles_per_expert = num_n_tiles * num_k_tiles

    for tile_idx in range(tidx, num_tiles_per_expert, NUM_SMS):
        # Output tile index
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

                x_ptrs = x_ptr + stride_xm * offs_m[:, None] + stride_xk * offs_k[None, :]
                y_ptrs = y_ptr + stride_ym * offs_m[:, None] + stride_yn * offs_n[None, :]

                accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
                for _ in range(tl.cdiv(m_size, BLOCK_SIZE_M)):
                    mask_m = offs_m < m_end
                    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :])
                    y = tl.load(y_ptrs, mask=mask_m[:, None] & mask_n[None, :])

                    accumulator += tl.dot(y.T, x)

                    offs_m += BLOCK_SIZE_M
                    x_ptrs += stride_xm * BLOCK_SIZE_M
                    y_ptrs += stride_ym * BLOCK_SIZE_M
                w = accumulator.to(w_ptr.dtype.element_ty)

                w_ptrs = w_ptr + stride_we * expert_idx + stride_wn * offs_n[:, None] + stride_wk * offs_k[None, :]
                tl.store(w_ptrs, w, mask=mask_n[:, None] & mask_k[None, :])


def grouped_gemm_backward_dw(
    x: torch.Tensor, y: torch.Tensor, m_sizes: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
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
    _grouped_gemm_backward_dw_kernel[grid](
        # Pointers
        x,
        y,
        m_sizes,
        w,
        # Dimensions
        M,
        N,
        K,
        E,
        NUM_SMS,
        # Strides
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        w.stride(0),
        w.stride(1),
        w.stride(2),
    )
    return w



## y[m, n] = sum_k w[s[m], k, n] * x[m, k]

@triton.autotune(
    configs=get_autotune_configs(),
    prune_configs_by={"early_config_prune": prune_configs},
    key=get_autotune_keys(),
)
@triton.jit
def _grouped_gemm_forward_transposed_kernel(
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
    stride_wk: tl.constexpr,
    stride_wn: tl.constexpr,
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
                mask_m = offs_m < m_start + m_size

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


def grouped_gemm_forward_transposed(
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
    E, _, N = w.shape
    assert w.shape[1] == K
    assert m_sizes.numel() == E

    if dtype is None:
        dtype = x.dtype
    y = torch.empty((M, N), device=x.device, dtype=dtype)
    NUM_SMS = get_num_sms()
    grid = lambda META: (NUM_SMS,)
    _grouped_gemm_forward_transposed_kernel[grid](
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


grouped_gemm = GroupedGemm.apply





import gc
import os
from functools import partial
from math import sqrt


os.environ["AUTOTUNE_BATCH_SIZE"] = "1"
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

# Faster than torch.histc when each element of s is an int in [0, E)
@partial(torch.compile, fullgraph=True, mode="max-autotune-no-cudagraphs")
@torch.no_grad()
def get_expert_counts(s: torch.Tensor, E: int) -> torch.Tensor:
    arange = torch.arange(E, device=s.device, dtype=s.dtype)
    counts = (arange[:, None] == s[None, :]).sum(dim=1, dtype=torch.int32)
    return counts


providers = {
    "grouped_gemm_dw": partial(grouped_gemm_backward_dw, dtype=torch.bfloat16),
    "grouped_gemm_dx": grouped_gemm_forward_transposed,
    "grouped_gemm_fwd": grouped_gemm_forward,
}
provider_names = list(providers)


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=range(1024, 16384 + 1, 1024),
            line_arg="provider",
            line_vals=provider_names,
            line_names=provider_names,
            ylabel="GFLOPS",
            plot_name="moe_fused_linear_all",
            args={},
        )
    ]
)
def benchmark(N, provider):
    print("N", N, "provider", provider, "begin")
    gc.collect()
    torch.cuda.empty_cache()

    in_features = 2048
    out_features = 768
    num_experts = 128
    device = "cuda"
    dtype = torch.bfloat16

    input = torch.randn(N, in_features, device=device, dtype=dtype)
    selected_experts = torch.randint(0, num_experts, (N,), device=device, dtype=torch.int32)
    # Assume selected_experts is sorted
    selected_experts, _ = torch.sort(selected_experts)
    m_sizes = get_expert_counts(selected_experts, num_experts)
    grad_output = torch.randn(N, out_features, device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    weight = 1 / sqrt(in_features) * torch.randn(num_experts, out_features, in_features, device=device, dtype=dtype)

    if provider == "grouped_gemm_dw":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: providers[provider](input, grad_output, m_sizes), quantiles=quantiles
        )

    elif provider == "grouped_gemm_dx":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: providers[provider](grad_output, weight, m_sizes), quantiles=quantiles
        )

    elif provider == "grouped_gemm_fwd":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: providers[provider](input, weight, m_sizes), quantiles=quantiles
        )

    else:
        raise ValueError("unreachable")

    perf = lambda ms: N * out_features * in_features / ms * 1e-6
    print("N", N, "provider", provider, "end", perf(ms))
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    with torch.inference_mode():
        benchmark.run(print_data=True)

