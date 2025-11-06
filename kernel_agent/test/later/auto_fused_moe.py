# pip install hf_transfer
# >>> python kernel_agent/test/auto_fused_moe.py --model qwen3 --seqlen 1024 --dtype bfloat16 --autotune
# config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 963/963 [00:00<00:00, 5.18MB/s]
# Benchmarking Qwen/Qwen3-30B-A3B forward: seqlen=1024, dtype=torch.bfloat16, permute_x=False, permute_y=False, autotune
# Forward: ref 45.1674, fused 4.9462, speedup 9.1x
# forward (128, 1024, 1536, 2048, False, False, False, 'torch.bfloat16', 'torch.bfloat16', 'torch.int64', 'torch.int64', 'torch.bfloat16'): {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'USE_TMA_LOAD_X': False, 'USE_TMA_LOAD_W': False, 'USE_TMA_STORE': False, 'num_warps': 8, 'num_ctas': 1, 'num_stages': 3}
# forward (128, 1024, 2048, 768, False, False, False, 'torch.bfloat16', 'torch.bfloat16', 'torch.int64', 'torch.int64', 'torch.bfloat16'): {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'USE_TMA_LOAD_X': False, 'USE_TMA_LOAD_W': False, 'USE_TMA_STORE': False, 'num_warps': 8, 'num_ctas': 1, 'num_stages': 3}
# Saving autotune results to benchmark_results/forward/autotune/20251106_1145/NVIDIA_RTX_4000_Ada_Generation/128_1024_1536_2048_False_False_False_bfloat16_bfloat16_int64_int64_bfloat16.json
# Saving autotune results to benchmark_results/forward/autotune/20251106_1145/NVIDIA_RTX_4000_Ada_Generation/128_1024_2048_768_False_False_False_bfloat16_bfloat16_int64_int64_bfloat16.json
# Total time: 40.0750 seconds


import sys
import types


def _ensure_package(name: str):
    if not name:
        return None
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__dict__.setdefault("__package__", name.rpartition(".")[0])
        module.__dict__.setdefault("__path__", [])
        sys.modules[name] = module
    parent_name, _, attr = name.rpartition('.')
    if parent_name:
        parent = _ensure_package(parent_name)
        if getattr(parent, attr, None) is not module:
            setattr(parent, attr, module)
    return module


def _register_module(name: str, namespace: dict):
    module = types.ModuleType(name)
    module.__dict__.update(namespace)
    module.__dict__.setdefault("__package__", name.rpartition(".")[0])
    module.__dict__.setdefault("__file__", name.replace('.', '/') + ".py")
    sys.modules[name] = module
    parent_name, _, attr = name.rpartition('.')
    if parent_name:
        parent = _ensure_package(parent_name)
        setattr(parent, attr, module)
    else:
        globals()[attr] = module
    return module

# ===== Module: grouped_gemm.kernels.autotuning =====
_prev_name = __name__
_module_name = 'grouped_gemm.kernels.autotuning'
__name__ = _module_name
_ensure_package(_module_name.rpartition('.')[0])
_module_before = set(globals())

# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""
Autotuning utils
"""

import logging
from itertools import product
from typing import List

import torch
import triton

logger = logging.getLogger(__name__)

DEFAULT_M_BLOCK_SIZES = [64, 128]
DEFAULT_N_BLOCK_SIZES = [64, 128, 256]
DEFAULT_K_BLOCK_SIZES = [64, 128, 256]
DEFAULT_NUM_CTAS = 1
DEFAULT_NUM_WARPS = [4, 8]
DEFAULT_NUM_STAGES = [3, 4, 5]
BOOLS = [True, False]


def val_to_list(val):
    if val is None:
        return None
    elif isinstance(val, list):
        return val
    else:
        return [val]


def convert_args_to_list(args):
    return [val_to_list(arg) for arg in args]


def get_forward_configs(
    BLOCK_M=DEFAULT_M_BLOCK_SIZES,
    BLOCK_N=DEFAULT_N_BLOCK_SIZES,
    BLOCK_K=DEFAULT_K_BLOCK_SIZES,
    TMA_LOAD_X=True,
    TMA_LOAD_W=True,
    TMA_STORE=False,  # NOTE: TMA_STORE is disabled for now
    num_warps=DEFAULT_NUM_WARPS,
    num_stages=DEFAULT_NUM_STAGES,
    num_ctas=DEFAULT_NUM_CTAS,
):
    (
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        TMA_LOAD_X,
        TMA_LOAD_W,
        TMA_STORE,
        num_warps,
        num_stages,
        num_ctas,
    ) = convert_args_to_list(
        [
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            TMA_LOAD_X,
            TMA_LOAD_W,
            TMA_STORE,
            num_warps,
            num_stages,
            num_ctas,
        ]
    )
    kernel_configs = []
    for (
        block_m,
        block_n,
        block_k,
        w,
        s,
        tma_load_x,
        tma_load_w,
        tma_store,
        num_ctas,
    ) in product(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps,
        num_stages,
        TMA_LOAD_X,
        TMA_LOAD_W,
        TMA_STORE,
        num_ctas,
    ):
        kernel_configs.append(
            triton.Config(
                dict(
                    BLOCK_SIZE_M=block_m,
                    BLOCK_SIZE_N=block_n,
                    BLOCK_SIZE_K=block_k,
                    USE_TMA_LOAD_X=tma_load_x,
                    USE_TMA_LOAD_W=tma_load_w,
                    USE_TMA_STORE=tma_store,
                ),
                num_warps=w,
                num_stages=s,
                num_ctas=num_ctas,
            )
        )

    return kernel_configs


def get_dX_kernel_configs(
    BLOCK_M=DEFAULT_M_BLOCK_SIZES,
    BLOCK_N=DEFAULT_N_BLOCK_SIZES,
    BLOCK_K=DEFAULT_K_BLOCK_SIZES,
    TMA_LOAD_dY=True,
    TMA_LOAD_W=True,
    TMA_STORE=False,  # NOTE: TMA_STORE is disabled for now
    num_warps=DEFAULT_NUM_WARPS,
    num_stages=DEFAULT_NUM_STAGES,
    num_ctas=DEFAULT_NUM_CTAS,
):
    (
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        TMA_LOAD_dY,
        TMA_LOAD_W,
        TMA_STORE,
        num_warps,
        num_stages,
        num_ctas,
    ) = convert_args_to_list(
        [
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            TMA_LOAD_dY,
            TMA_LOAD_W,
            TMA_STORE,
            num_warps,
            num_stages,
            num_ctas,
        ]
    )
    kernel_configs = []
    for (
        block_m,
        block_n,
        block_k,
        w,
        s,
        tma_load_dy,
        tma_load_w,
        tma_store,
        num_ctas,
    ) in product(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps,
        num_stages,
        TMA_LOAD_dY,
        TMA_LOAD_W,
        TMA_STORE,
        num_ctas,
    ):
        kernel_configs.append(
            triton.Config(
                dict(
                    BLOCK_SIZE_M=block_m,
                    BLOCK_SIZE_N=block_n,
                    BLOCK_SIZE_K=block_k,
                    USE_TMA_LOAD_dY=tma_load_dy,
                    USE_TMA_LOAD_W=tma_load_w,
                    USE_TMA_STORE=tma_store,
                ),
                num_warps=w,
                num_stages=s,
                num_ctas=num_ctas,
            )
        )

    return kernel_configs


def get_dW_kernel_configs(
    BLOCK_M=DEFAULT_M_BLOCK_SIZES,
    BLOCK_N=DEFAULT_N_BLOCK_SIZES,
    BLOCK_K=DEFAULT_K_BLOCK_SIZES,
    num_warps=DEFAULT_NUM_WARPS,
    num_stages=DEFAULT_NUM_STAGES,
    num_ctas=DEFAULT_NUM_CTAS,
    TMA_LOAD_dY=True,
    TMA_LOAD_X=True,
    TMA_STORE=False,
):
    (
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps,
        num_stages,
        num_ctas,
        TMA_LOAD_dY,
        TMA_LOAD_X,
        TMA_STORE,
    ) = convert_args_to_list(
        [
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            num_warps,
            num_stages,
            num_ctas,
            TMA_LOAD_dY,
            TMA_LOAD_X,
            TMA_STORE,
        ]
    )
    kernel_configs = []
    for (
        block_m,
        block_n,
        block_k,
        w,
        s,
        tma_load_dy,
        tma_load_x,
        tma_store,
        num_ctas,
    ) in product(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps,
        num_stages,
        TMA_LOAD_dY,
        TMA_LOAD_X,
        TMA_STORE,
        num_ctas,
    ):
        kernel_configs.append(
            triton.Config(
                dict(
                    BLOCK_SIZE_M=block_m,
                    BLOCK_SIZE_N=block_n,
                    BLOCK_SIZE_K=block_k,
                    USE_TMA_LOAD_dY=tma_load_dy,
                    USE_TMA_LOAD_X=tma_load_x,
                    USE_TMA_STORE=tma_store,
                ),
                num_warps=w,
                num_stages=s,
                num_ctas=num_ctas,
            )
        )

    return kernel_configs


def estimate_smem_reqs(
    num_stages: int,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    dtype: torch.dtype,
):
    num_bytes = dtype.itemsize
    return (
        num_stages * BLOCK_SIZE_K * (BLOCK_SIZE_M + BLOCK_SIZE_N)
        + BLOCK_SIZE_M * BLOCK_SIZE_N
    ) * num_bytes


def exceeds_smem_capacity(
    num_stages: int,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    dtype: torch.dtype,
    smem_size: int,
    slack: float = 50000,
):
    smem_reqs = estimate_smem_reqs(
        num_stages, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype
    )
    return smem_reqs > smem_size + slack


def common_prune_criteria(config: triton.Config, kwargs: dict, dtype):
    from grouped_gemm.interface import supports_tma
    from grouped_gemm.kernels.tuning import get_device_properties

    smem_size = get_device_properties().SIZE_SMEM

    num_stages = config.num_stages
    BLOCK_SIZE_M = config.kwargs["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = config.kwargs["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = config.kwargs["BLOCK_SIZE_K"]

    num_tokens = kwargs["NUM_TOKENS"]
    num_experts = kwargs["NUM_EXPERTS"]
    permute_x = kwargs["PERMUTE_X"]
    permute_y = kwargs["PERMUTE_Y"]
    tokens_per_expert = num_tokens // num_experts

    # use_tma = [k for k in config.kwargs.keys() if k.startswith("USE_TMA_")]
    MIN_BLOCK_SIZE_M = DEFAULT_M_BLOCK_SIZES[0]
    if exceeds_smem_capacity(
        num_stages, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype, smem_size
    ):
        return True
    if BLOCK_SIZE_M > tokens_per_expert * 2 and tokens_per_expert > MIN_BLOCK_SIZE_M:
        return True
    if permute_x and permute_y:
        return True
    # if not supports_tma() and any(use_tma):
    #     return True
    return False


def maybe_disable_tma(config: triton.Config):
    from grouped_gemm.interface import supports_tma

    tma_keys = [k for k in config.kwargs.keys() if k.startswith("USE_TMA_")]
    if not supports_tma():
        logger.info("Disabling TMA")
        for k in tma_keys:
            config.kwargs[k] = False


def prune_kernel_configs_fwd(configs: list[triton.Config], args, **kwargs):
    x = kwargs["x_ptr"]
    dtype = x.dtype

    logger.debug(f"Pruning configs: {len(configs)}")

    pruned_configs = []
    for config in configs:
        # disable TMA if gpu does not support it
        maybe_disable_tma(config)

        if common_prune_criteria(config, kwargs, dtype):
            continue
        if config.kwargs["USE_TMA_LOAD_X"] and kwargs["PERMUTE_X"]:
            # Dynamically disable TMA_LOAD_X for permuted X
            config.kwargs["USE_TMA_LOAD_X"] = False
        if config.kwargs["USE_TMA_STORE"] and kwargs["PERMUTE_Y"]:
            continue

        pruned_configs.append(config)

    logger.debug(f"Pruned configs: {len(pruned_configs)}")
    return pruned_configs


def prune_dX_configs(configs: List[triton.Config], args, **kwargs):
    dtype = kwargs["w_ptr"].dtype

    logger.debug(f"Pruning configs: {len(configs)}")
    pruned_configs = []

    for config in configs:
        if common_prune_criteria(config, kwargs, dtype):
            continue
        if config.kwargs["USE_TMA_LOAD_dY"] and kwargs["PERMUTE_Y"]:
            # dynamically disable TMA_LOAD_dY for permuted Y
            config.kwargs["USE_TMA_LOAD_dY"] = False
        if config.kwargs["USE_TMA_STORE"] and kwargs["PERMUTE_X"]:
            continue
        pruned_configs.append(config)

    logger.debug(f"Pruned configs: {len(pruned_configs)}")
    return pruned_configs


def prune_kernel_configs_backward_dW(configs: list[triton.Config], args, **kwargs):
    dtype = kwargs["x_ptr"].dtype

    pruned_configs = []
    logger.debug(f"Pruning configs: {len(configs)}")

    for config in configs:
        if common_prune_criteria(config, kwargs, dtype):
            continue
        if config.kwargs["USE_TMA_LOAD_dY"] and kwargs["PERMUTE_Y"]:
            config.kwargs["USE_TMA_LOAD_dY"] = False
        if config.kwargs["USE_TMA_LOAD_X"] and kwargs["PERMUTE_X"]:
            config.kwargs["USE_TMA_LOAD_X"] = False
        pruned_configs.append(config)

    logger.debug(f"Pruned configs: {len(pruned_configs)}")
    return pruned_configs

__name__ = _prev_name
_module_exports = {k: globals()[k] for k in set(globals()) - _module_before}
_register_module(_module_name, _module_exports)
del _module_exports
del _module_before
del _module_name
del _prev_name

# ===== Module: grouped_gemm.kernels.forward =====
_prev_name = __name__
_module_name = 'grouped_gemm.kernels.forward'
__name__ = _module_name
_ensure_package(_module_name.rpartition('.')[0])
_module_before = set(globals())

# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

import torch
import triton
import triton.language as tl

from grouped_gemm.kernels.autotuning import (
    get_forward_configs,
    prune_kernel_configs_fwd,
)


#
# PERMUTE_X -> permute tokens so that they are ordered by expert
# PERMUTE_Y -> permute output so that they are ordered by token
# These are effectively the same thing: the former loads in permuted order, the latter stores in permuted order => we only need to define the permutation indices once
# In the former, we use these row indices when loading X
# For the latter, we use these row indices when storing Y
# FUSE_MUL -> multiply routed outputs by their respective weights
# topk_weights are in token order
# Only account for the case when X is in expert order and we are permuting Y when fusing mul -- this precondition is checked in the interface
@triton.jit
def _grouped_gemm_forward_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    # Variable depending on routed probs
    m_sizes_ptr,
    gather_indices_ptr,
    topk_weights_ptr,
    # Constant problem shapes
    NUM_EXPERTS: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    # Tuning params
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    PERMUTE_X: tl.constexpr = False,
    PERMUTE_Y: tl.constexpr = False,
    FUSE_MUL_PRE: tl.constexpr = False,
    FUSE_MUL_POST: tl.constexpr = False,
    USE_FAST_ACCUM: tl.constexpr = False,
    USE_TMA_LOAD_W: tl.constexpr = False,
    USE_TMA_LOAD_X: tl.constexpr = False,
    USE_TMA_STORE: tl.constexpr = False,
    acc_dtype: tl.constexpr = tl.float32,
    FLATTEN: tl.constexpr = True,
) -> None:
    tl.static_assert(K % BLOCK_SIZE_K == 0)

    TOTAL_TOKENS: tl.constexpr = NUM_TOKENS * TOPK
    SHOULD_PERMUTE: tl.constexpr = PERMUTE_X or PERMUTE_Y
    SHOULD_FUSE_MUL: tl.constexpr = FUSE_MUL_PRE or FUSE_MUL_POST
    SHOULD_PERMUTE_OR_FUSE: tl.constexpr = SHOULD_PERMUTE or SHOULD_FUSE_MUL
    # tl.static_print("SHOULD_PERMUTE", PERMUTE_X, PERMUTE_Y, FUSE_MUL_PRE, FUSE_MUL_POST, SHOULD_PERMUTE, SHOULD_FUSE, SHOULD_PERMUTE_OR_FUSE)
    tidx = tl.program_id(0)
    output_dtype: tl.dtype = y_ptr.dtype.element_ty

    # Create TMA descriptors for loading sorted tokens
    # When using TMA load, we don't permute_x, so shape should be [TOTAL_TOKENS, K]
    # Also, we are defining a single global descriptor with single block shape
    # Need to check that this does not result in errors when crossing expert boundaries
    if USE_TMA_LOAD_X:
        x_desc = tl._experimental_make_tensor_descriptor(
            x_ptr,
            shape=[TOTAL_TOKENS, K],
            strides=[K, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )

    if USE_TMA_LOAD_W:
        expert_stride = N * K
        w_desc = tl._experimental_make_tensor_descriptor(
            w_ptr,
            shape=[NUM_EXPERTS, N, K],
            strides=[expert_stride, K, 1],
            block_shape=[1, BLOCK_SIZE_N, BLOCK_SIZE_K],
        )

    m_end = 0
    processed_tiles = 0
    m_block_range = tl.arange(0, BLOCK_SIZE_M)

    for expert_idx in tl.range(NUM_EXPERTS, flatten=FLATTEN):
        m_start = m_end
        m_size = tl.load(m_sizes_ptr + expert_idx).to(tl.int32)
        m_end = m_start + m_size

        if m_size > 0:
            n_start = expert_idx * N

            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
            num_tiles_per_expert = num_m_tiles * num_n_tiles

            # Need to create tma_store within loop since we need to predicate stores based on m_size
            if USE_TMA_STORE:
                y_desc = tl._experimental_make_tensor_descriptor(
                    y_ptr,  # + m_start * N,
                    shape=[m_end, N],
                    strides=[N, 1],
                    block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                )

            # Process tiles for this expert
            while (
                tidx >= processed_tiles
                and tidx < processed_tiles + num_tiles_per_expert
            ):
                tile_idx = tidx - processed_tiles

                # Check if L2 cache re-use for this order is optimal
                tile_m_idx = tile_idx % num_m_tiles
                tile_n_idx = tile_idx // num_m_tiles

                if SHOULD_PERMUTE_OR_FUSE:
                    # These will be used for loading and storing in permuted order
                    gather_offsets = tile_m_idx * BLOCK_SIZE_M + m_block_range
                    indices_to_gather = m_start + tl.max_contiguous(
                        tl.multiple_of(gather_offsets % m_size, BLOCK_SIZE_M),
                        BLOCK_SIZE_M,
                    )
                    expert_token_idx = tl.load(
                        gather_indices_ptr + indices_to_gather,
                        mask=indices_to_gather < TOTAL_TOKENS,
                    )
                    expert_token_offsets = expert_token_idx[:, None]

                    # Masks for permuted load and store

                    row_mask = gather_offsets < m_size
                    row_mask = row_mask[:, None]

                    # row_mask = indices_to_gather < m_end
                    # row_mask = row_mask[:, None]

                # We only take into account the following two cases: (PERMUTE_X and NOT PERMUTE_Y) and (NOT PERMUTE_X and PERMUTE_Y)
                # Hence, we can make the following simplifying assumptions when loading and storing
                # Note the different strides between the two cases: the offsets for loading and storing are flipped and the strides must also be adjusted
                if PERMUTE_X:
                    load_idx = (
                        (expert_token_offsets // TOPK) * K
                    )  # Permute on load from token -> expert order, divide by TOPK to index from original number of tokens
                    store_idx = (
                        indices_to_gather[:, None] * N
                    )  # Store in contiguous order
                else:
                    off_am = tile_m_idx * BLOCK_SIZE_M
                    if not PERMUTE_Y:
                        # These will already be computed if permuting y
                        offs_am = off_am + m_block_range
                        row_mask = offs_am[:, None] < m_size
                        row_idx = m_start + offs_am[:, None]
                        store_idx = row_idx * N
                        if not USE_TMA_LOAD_X:
                            load_idx = row_idx * K

                if PERMUTE_Y:
                    if not USE_TMA_LOAD_X:
                        load_idx = (
                            indices_to_gather[:, None] * K
                        )  # Load in contiguous order (no permutation on load)
                    # offs_am = off_am + m_block_range
                    # row_mask = offs_am[:, None] < m_size
                    store_idx = (
                        expert_token_offsets * N
                    )  # Permute on store from expert -> token order

                # We always load topk weights in expert order
                # In the pre-multiplication case, we multiply permuted hidden states by weights before the first gemm
                # In the post-multiplication case, we multiply permuted hidden states by weights after the second gemm
                # In either case, the hidden states are grouped by expert, so we always permute on load of topk weights
                if SHOULD_FUSE_MUL:
                    topk_load_idx = expert_token_offsets

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

                offs_k = tl.arange(0, BLOCK_SIZE_K)

                if not USE_TMA_LOAD_X:
                    x_ptrs = x_ptr + load_idx + offs_k[None, :]

                if not USE_TMA_LOAD_W:
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    offs_bn = tl.max_contiguous(
                        tl.multiple_of(offs_bn % N, BLOCK_SIZE_N), BLOCK_SIZE_N
                    )
                    w_ptrs = w_ptr + (n_start + offs_bn[:, None]) * K + offs_k[None, :]

                for k_offset in range(0, K, BLOCK_SIZE_K):
                    if not USE_TMA_LOAD_X:
                        x = tl.load(x_ptrs, mask=row_mask)
                    else:
                        x = x_desc.load([m_start + off_am, k_offset])

                    if FUSE_MUL_PRE:
                        # Check for correct broadcasting
                        topk_weights = tl.load(
                            topk_weights_ptr + topk_load_idx, mask=row_mask
                        )
                        x *= topk_weights.to(x.dtype)

                    if not USE_TMA_LOAD_W:
                        w = tl.load(w_ptrs, mask=offs_bn[:, None] < N)
                    else:
                        w = w_desc.load(
                            [expert_idx, tile_n_idx * BLOCK_SIZE_N, k_offset]
                        )
                        w = tl.reshape(w, (BLOCK_SIZE_N, BLOCK_SIZE_K))

                    accumulator += tl.dot(x, w.T)

                    if not USE_TMA_LOAD_X:
                        x_ptrs += BLOCK_SIZE_K

                    if not USE_TMA_LOAD_W:
                        w_ptrs += BLOCK_SIZE_K

                y = accumulator.to(output_dtype)

                # NOTE: order of fusing multiplication is important
                # Fusing before accumulator dtype conversion results in numerical diffs
                if FUSE_MUL_POST:
                    # Check for correct broadcasting
                    topk_weights = tl.load(
                        topk_weights_ptr + topk_load_idx, mask=row_mask
                    )
                    y *= topk_weights.to(output_dtype)

                offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                store_mask = row_mask & (offs_bn[None, :] < N)

                if USE_TMA_STORE:
                    offset_m = tile_m_idx * BLOCK_SIZE_M  # .to(tl.int32)
                    offset_n = tile_n_idx * BLOCK_SIZE_N  # .to(tl.int32)
                    y_desc.store([m_start + offset_m, offset_n], y)
                else:
                    tl.store(
                        y_ptr + store_idx + offs_bn[None, :],
                        y,
                        mask=store_mask,
                    )
                tidx += NUM_SMS

            processed_tiles += num_tiles_per_expert


_autotuned_grouped_gemm_forward_kernel = triton.autotune(
    configs=get_forward_configs(),
    prune_configs_by={"early_config_prune": prune_kernel_configs_fwd},
    key=[
        "NUM_EXPERTS",
        "NUM_TOKENS",
        "N",
        "K",
        "PERMUTE_X",
        "PERMUTE_Y",
        "FUSE_MUL_POST",
    ],
)(_grouped_gemm_forward_kernel)

__name__ = _prev_name
_module_exports = {k: globals()[k] for k in set(globals()) - _module_before}
_register_module(_module_name, _module_exports)
del _module_exports
del _module_before
del _module_name
del _prev_name

# ===== Module: grouped_gemm.kernels.backward =====
_prev_name = __name__
_module_name = 'grouped_gemm.kernels.backward'
__name__ = _module_name
_ensure_package(_module_name.rpartition('.')[0])
_module_before = set(globals())

# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

import torch
import triton
import triton.language as tl

from grouped_gemm.kernels.autotuning import (
    get_dW_kernel_configs,
    get_dX_kernel_configs,
    prune_dX_configs,
    prune_kernel_configs_backward_dW,
)

"""
dX backward kernel

- Shapes
    - the forward pass input X shape is [NUM_TOKENS, K] if permute_x else [NUM_TOKENS * TOPK, K]; output y is [NUM_TOKENS * TOPK, N]
    - the backward pass input dy shape is [NUM_TOKENS * TOPK, N], reduce across N, output dX is [NUM_TOKENS * TOPK, K]
- Note that in the backward pass, the output size is still [NUM_TOKENS * TOPK, K] since we still need to accumulate gradients for each expert chosen by the token in a post-processing step.

`permute_x` notes:
- In the forward pass, if we permute X on load, we need to permute on store in the backward pass to restore to original token order
- the output dX with have shape [NUM_TOKENS * TOPK, K] and we need to perform an additional reduction across topk to accumulate gradients
- This is done as a post-processing step in autograd.Function.
- If not `permute_x`, this postprocessing step should take place outside autograd.Function such that the gradient shape matches the input X shape.

`permute_y` notes:
- In the forward pass, if we permuted output on store (e.g., in the second grouped GEMM in fused MoE MLP), we need to permute on load to get from token order to expert grouped order
- We still store in contiguous order since we are writing out dX which will be the input to the backwards pass of the first grouped GEMM

`fused_mul` notes:
- In the forward pass, if we used the multiplication of topk weights (e.g., in the second grouped GEMM in fused MoE MLP), we need to make a few additional changes:
    1) We load topk_weights in natural (token) order.  Since we only enable `fuse_mul` when permuting on store (`permute_y`), we multiply grad_output by topk_weights before backpropagating
    2) We need to calculate the gradient of the topk_weights.  This gets messy since we need do an additional elementwise multiplication in the GEMM main loop and then write out in unpermuted order.  For now, we do not fuse this step but calculate as a simple

Invalid combinations:
- permute_y and use_tma_load: permuting y on store in forward -> load in permuted order in backward, therefore can't use TMA load (unless Blackwell which supports gather / scatter TMA)
- permute_x and use_tma_store: permuting x on load in forward -> store in permuted order in backward, therefore can't use TMA store (unless Blackwell which supports gather / scatter TMA)

TODO:
- We define indices for all conditions and expect that unused indices will be DCE'd during compilation.  Check that this is the case otherwise will result in unnecessary register usage.
"""


@triton.jit
def _grouped_gemm_dX_kernel(
    dY_ptr,  # [M_total, N]
    w_ptr,  # [E, N, K]
    dX_ptr,  # [M_total, K]
    gather_indices_ptr,
    m_sizes_ptr,
    # problem sizes
    NUM_EXPERTS: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    # Tuning parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    PERMUTE_X: tl.constexpr = False,
    PERMUTE_Y: tl.constexpr = False,
    USE_TMA_LOAD_W: tl.constexpr = False,
    USE_TMA_LOAD_dY: tl.constexpr = False,
    USE_TMA_STORE: tl.constexpr = False,
    FLATTEN: tl.constexpr = True,
) -> None:
    TOTAL_TOKENS: tl.constexpr = NUM_TOKENS * TOPK
    output_dtype = dX_ptr.dtype.element_ty

    tidx = tl.program_id(0)
    # This removes the need for predication along N in the GEMM main loop
    tl.static_assert(N % BLOCK_SIZE_N == 0, "N must be divisible by BLOCK_SIZE_N")
    tl.static_assert(K % BLOCK_SIZE_K == 0, "K must be divisible by BLOCK_SIZE_K")

    # Create TMA descriptors for loading sorted tokens
    # When using TMA load, we don't permute_x, so shape should be [TOTAL_TOKENS, K]
    # Also, we are defining a single global descriptor with single block shape
    # Need to check that this does not result in errors when crossing expert boundaries
    if USE_TMA_LOAD_dY:
        dY_desc = tl._experimental_make_tensor_descriptor(
            dY_ptr,
            shape=[TOTAL_TOKENS, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )

    if USE_TMA_LOAD_W:
        expert_stride = N * K
        w_desc = tl._experimental_make_tensor_descriptor(
            w_ptr,
            shape=[NUM_EXPERTS, N, K],
            strides=[expert_stride, K, 1],
            block_shape=[1, BLOCK_SIZE_N, BLOCK_SIZE_K],
        )

    m_end = 0
    processed_tiles = 0
    m_block_range = tl.arange(0, BLOCK_SIZE_M)
    n_block_range = tl.arange(0, BLOCK_SIZE_N)
    k_block_range = tl.arange(0, BLOCK_SIZE_K)

    for expert_idx in range(NUM_EXPERTS, flatten=FLATTEN):
        m_start = m_end
        m_size = tl.load(m_sizes_ptr + expert_idx).to(tl.int32)
        m_end = m_start + m_size

        if m_size > 0:
            # Advance n offset to the weights for that respective expert
            n_start = expert_idx * N
            # N_start_offset = g.to(tl.int64) * N
            # tiles for this group's GEMM
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
            num_tiles_per_expert = num_m_tiles * num_k_tiles

            if USE_TMA_STORE:
                # Need to define descript within loop to predicate store along M
                tl.static_assert(
                    K % BLOCK_SIZE_K == 0, "K must be divisible by BLOCK_SIZE_K"
                )
                dX_desc = tl._experimental_make_tensor_descriptor(
                    dX_ptr,
                    shape=[m_end, K],
                    strides=[K, 1],
                    block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                )

            # Lower bound and upper bound are defined relative to the total tiles processed so far
            # This ensures that we are only processing tiles for the current expert group AND
            # we never exceed the total number of tiles for all expert groups
            while tidx >= processed_tiles and tidx < (
                processed_tiles + num_tiles_per_expert
            ):
                group_index = tidx - processed_tiles

                # Output tile for this thread block for this expert group
                tile_m_idx = group_index % num_m_tiles
                tile_k_idx = group_index // num_m_tiles

                if PERMUTE_X or PERMUTE_Y:
                    # These will be used for loading and storing in permuted order
                    gather_offsets = tile_m_idx * BLOCK_SIZE_M + m_block_range
                    # indices_to_gather = m_start + gather_offsets
                    indices_to_gather = m_start + tl.max_contiguous(
                        tl.multiple_of(gather_offsets % m_size, BLOCK_SIZE_M),
                        BLOCK_SIZE_M,
                    )
                    expert_token_idx = tl.load(
                        gather_indices_ptr + indices_to_gather,
                        mask=indices_to_gather < TOTAL_TOKENS,
                    )
                    expert_token_offsets = expert_token_idx[:, None]

                    # Masks for permuted load and store
                    row_mask = gather_offsets < m_size
                    row_mask = row_mask[:, None]

                    # We only take into account the following two cases: (PERMUTE_X and NOT PERMUTE_Y) and (NOT PERMUTE_X and PERMUTE_Y)
                    # Hence, we can make the following simplifying assumptions when loading and storing
                    # Note the different strides between the two cases: the offsets for loading and storing are flipped and the strides must also be adjusted

                    if PERMUTE_X:
                        # Case where we permuted on load in the forward pass (typically first grouped GEMM in MoE MLP)
                        load_a_idx = (
                            indices_to_gather[:, None] * N
                        )  # Load in contiguous (expert grouped) order
                        store_idx = (
                            expert_token_offsets * K
                        )  # Permute on store from expert -> token order
                    else:
                        # Case where we permuted on store in the forward pass (typically second grouped GEMM in MoE MLP)
                        load_a_idx = (
                            expert_token_offsets * N
                        )  # Permute on load from token -> expert order
                        store_idx = (
                            indices_to_gather[:, None] * K
                        )  # Store in contiguous order
                else:
                    # # Position in full matrix - needed for TMA
                    # m_offset = (M_start + (tile_m_idx * BLOCK_SIZE_M)).to(tl.int32)
                    # k_offset = (tile_k_idx * BLOCK_SIZE_K).to(tl.int32)
                    # Offsets *relative* to the *current* expert -- m_start will then advance to this expert's start token
                    offs_am = tile_m_idx * BLOCK_SIZE_M + m_block_range

                    # [M, N] @ [N, K] -> [M, K] => Stride for A is N, stride for B is K
                    # We need two additional offsets:
                    # 1. For A, m_start to advance to this expert's start token
                    # 2. For B, n_start to advance to this expert's weights since we are passing in an [E, N, K] weight matrix
                    row_offsets_a = m_start + offs_am[:, None]
                    load_a_idx = row_offsets_a * N
                    store_idx = row_offsets_a * K
                    row_mask = offs_am[:, None] < m_size

                if not USE_TMA_LOAD_dY:
                    dY_ptrs = dY_ptr + load_a_idx + n_block_range[None, :]

                offs_bk = tile_k_idx * BLOCK_SIZE_K + k_block_range
                if not USE_TMA_LOAD_W:
                    row_offsets_b = n_start + n_block_range
                    # offs_bn = n_start + n_block_range
                    # row_offsets_b = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
                    w_ptrs = w_ptr + row_offsets_b[:, None] * K + offs_bk[None, :]

                # TODO: check whether predication along K is needed since we checked that K is divisible by BLOCK_SIZE_K in the forward kernel
                # col_mask = offs_bk[None, :] < K
                store_mask = row_mask  # & col_mask

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

                # GEMM main loop
                for n_offset in range(0, N, BLOCK_SIZE_N):
                    # dY block [M, N]
                    if not USE_TMA_LOAD_dY:
                        dY = tl.load(dY_ptrs, mask=row_mask)
                    else:
                        dY = dY_desc.load(
                            [m_start + tile_m_idx * BLOCK_SIZE_M, n_offset]
                        )

                    if not USE_TMA_LOAD_W:
                        w = tl.load(w_ptrs)  # , mask=col_mask)
                    else:
                        w = w_desc.load(
                            [expert_idx, n_offset, tile_k_idx * BLOCK_SIZE_K]
                        )
                        w = tl.reshape(w, (BLOCK_SIZE_N, BLOCK_SIZE_K))
                    # TODO: check if predication along K is needed since we checked that K is divisible by BLOCK_SIZE_K in the forward kernel

                    # [M, N] @ [N, K] -> [M, K]
                    accumulator += tl.dot(dY, w)  # NOTE: no transpose of b

                    # Advance A along contiguous dimension
                    if not USE_TMA_LOAD_dY:
                        dY_ptrs += BLOCK_SIZE_N
                    # Note we are no longer advancing B along contiguous dimension since weights are arranged as [N, K]
                    # Instead, we need to stride by K to advance to the [N_BLOCK_SIZE, K_BLOCK_SIZE] tile
                    if not USE_TMA_LOAD_W:
                        w_ptrs += BLOCK_SIZE_N * K

                dX = accumulator.to(output_dtype)

                # Writing out a BLOCK_M x BLOCK_K tile, so we need to stride by K
                if USE_TMA_STORE:
                    offset_m = tile_m_idx * BLOCK_SIZE_M  # .to(tl.int32)
                    offset_k = tile_k_idx * BLOCK_SIZE_K  # .to(tl.int32)
                    dX_desc.store([m_start + offset_m, offset_k], dX)
                else:
                    tl.store(
                        dX_ptr + store_idx + offs_bk[None, :],
                        dX,
                        mask=store_mask,
                    )

                # Move to the next tile within this expert group
                tidx += NUM_SMS

            # Update the total tiles count for the next expert group
            processed_tiles += num_tiles_per_expert


_autotuned_grouped_gemm_dX_kernel = triton.autotune(
    configs=get_dX_kernel_configs(),
    prune_configs_by={"early_config_prune": prune_dX_configs},
    key=["NUM_EXPERTS", "NUM_TOKENS", "N", "K", "PERMUTE_X", "PERMUTE_Y"],
)(_grouped_gemm_dX_kernel)

"""
notes on permute_x:
- for the first grouped GEMM, we permuted on load -> X was [num_tokens, K] and stored y in expert grouped order [num_tokens * topk, K]
- in the backwards pass, we need to permute on load of X while loading dy in contiguous (expert grouped) order
- since we are writing out dW, there is no need to permute on store

notes on permute_y:
- for the second grouped GEMM, we permuted on store -> y was permuted from expert grouped order to token order, x was loaded in expert grouped order since it was the output of the first grouped GEMM
- in the backwards pass, we need to permute on load of dy to get from token order to expert grouped order to match the order of X
- since we are writing out dW, there is no need to permute on store

notes on TMA loading:
- if we're TMA loading both X and dY, then we need to mask along the M dimension
to account for expert boundaries
- we can either
    - define TMA descriptors within the outer for loop to predicate loads
    or
    - mask along M after loading
"""


@triton.jit
def _grouped_gemm_dW_kernel(
    x_ptr,
    dY_ptr,
    dW_ptr,
    m_sizes_ptr,
    gather_indices_ptr,
    # problem sizes
    NUM_TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    PERMUTE_X: tl.constexpr = False,
    PERMUTE_Y: tl.constexpr = False,
    USE_TMA_LOAD_dY: tl.constexpr = False,
    USE_TMA_LOAD_X: tl.constexpr = False,
    USE_TMA_STORE: tl.constexpr = False,
    FLATTEN: tl.constexpr = True,
    acc_dtype: tl.constexpr = tl.float32,
) -> None:
    TOTAL_TOKENS: tl.constexpr = NUM_TOKENS * TOPK
    TMA_LOAD_BOTH: tl.constexpr = USE_TMA_LOAD_X and USE_TMA_LOAD_dY

    tidx = tl.program_id(0)
    output_dtype = dW_ptr.dtype.element_ty

    if USE_TMA_LOAD_dY and not TMA_LOAD_BOTH:
        dY_desc = tl._experimental_make_tensor_descriptor(
            dY_ptr,
            shape=[TOTAL_TOKENS, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )

    if USE_TMA_LOAD_X and not TMA_LOAD_BOTH:
        x_desc = tl._experimental_make_tensor_descriptor(
            x_ptr,
            shape=[TOTAL_TOKENS, K],
            strides=[K, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )
    # Output tiles per expert, since each expert weight matrix is [N, K]
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    output_tiles_per_expert = num_n_tiles * num_k_tiles

    block_range_m = tl.arange(0, BLOCK_SIZE_M)
    block_range_n = tl.arange(0, BLOCK_SIZE_N)
    block_range_k = tl.arange(0, BLOCK_SIZE_K)

    # NOTE: Important that N % BLOCK_SIZE_N == 0 and K % BLOCK_SIZE_K == 0 when using TMA store
    if USE_TMA_STORE:
        tl.static_assert(N % BLOCK_SIZE_N == 0, "N must be divisible by BLOCK_SIZE_N")
        tl.static_assert(K % BLOCK_SIZE_K == 0, "K must be divisible by BLOCK_SIZE_K")
        dW_desc = tl._experimental_make_tensor_descriptor(
            dW_ptr,
            shape=[NUM_EXPERTS, N, K],
            strides=[N * K, K, 1],
            block_shape=[1, BLOCK_SIZE_N, BLOCK_SIZE_K],
        )

    for tile_idx in range(
        tidx, output_tiles_per_expert, NUM_SMS
    ):  # , flatten=FLATTEN):
        # Output tile index
        tile_n_idx = tile_idx % num_n_tiles
        tile_k_idx = tile_idx // num_n_tiles

        # Output tile offsets
        n_offset = tile_n_idx * BLOCK_SIZE_N
        k_offset = tile_k_idx * BLOCK_SIZE_K

        # For storing
        # TODO: Check whether the k mask is needed since we statically check that K is divisible by BLOCK_SIZE_K in the forward kernel
        # ditto for n_mask
        n_mask = block_range_n + n_offset < N
        k_mask = block_range_k + k_offset < K
        nk_mask = n_mask[:, None] & k_mask[None, :]

        m_end = 0
        for expert_idx in range(NUM_EXPERTS):
            # We need to instantiate a fresh accumulator for each expert
            accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=acc_dtype)

            m_start = m_end
            # Need to figure out why this cast is needed, otherwise compiler complains about mismatching types
            m_size = tl.load(m_sizes_ptr + expert_idx).to(tl.int32)
            m_end = m_start + m_size

            # NOTE: when storing the result, we need to offset by n_start since we are storing the result for this expert to the global [E, N, K] weight matrix
            n_start = expert_idx * N
            store_row_offs = n_start + n_offset + block_range_n

            if m_size > 0:
                if TMA_LOAD_BOTH:
                    dY_desc = tl._experimental_make_tensor_descriptor(
                        dY_ptr,
                        shape=[m_end, N],
                        strides=[N, 1],
                        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                    )

                    x_desc = tl._experimental_make_tensor_descriptor(
                        x_ptr,
                        shape=[m_end, K],
                        strides=[K, 1],
                        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                    )

                for tile_m_idx in range(0, m_size, BLOCK_SIZE_M):
                    m_block_size = tl.minimum(BLOCK_SIZE_M, m_size - tile_m_idx)

                    if m_block_size > 0:
                        # Global offset for this chunk
                        m_global_offset = m_start + tile_m_idx
                        m_offsets = m_global_offset + block_range_m

                        if PERMUTE_X or PERMUTE_Y:
                            # These will be used for loading and storing in permuted order
                            gather_offsets = (
                                tile_m_idx + block_range_m
                            )  # NOTE: tile_m_idx is already strided by BLOCK_SIZE_M

                            indices_to_gather = m_start + tl.max_contiguous(
                                tl.multiple_of(gather_offsets % m_size, BLOCK_SIZE_M),
                                BLOCK_SIZE_M,
                            )
                            # indices_to_gather = m_start + gather_offsets
                            expert_token_idx = tl.load(
                                gather_indices_ptr + indices_to_gather,
                                mask=indices_to_gather < TOTAL_TOKENS,
                            )
                            expert_token_offsets = expert_token_idx[:, None]

                            # Masks for permuted load and store
                            row_load_mask = gather_offsets < m_size

                            # We only take into account the following two cases: (PERMUTE_X and NOT PERMUTE_Y) and (NOT PERMUTE_X and PERMUTE_Y)
                            # Hence, we can make the following simplifying assumptions when loading and storing
                            # Note the different strides between the two cases: the offsets for loading and storing are flipped and the strides must also be adjusted
                            if PERMUTE_X:
                                x_row_load_idx = (
                                    (expert_token_offsets // TOPK) * K
                                )  # Permute on load from token -> expert order, divide by TOPK to index from original number of tokens
                                dY_row_load_idx = m_offsets[:, None] * N
                            else:
                                x_row_load_idx = (
                                    indices_to_gather[:, None] * K
                                )  # Load in contiguous order (no permutation on load)
                                dY_row_load_idx = expert_token_offsets * N

                        else:
                            x_row_load_idx = m_offsets[:, None] * K
                            dY_row_load_idx = m_offsets[:, None] * N
                            row_load_mask = block_range_m < m_block_size

                        mk_mask = row_load_mask[:, None] & k_mask[None, :]
                        mn_mask = row_load_mask[:, None] & n_mask[None, :]

                        if USE_TMA_LOAD_X:
                            x = x_desc.load([m_global_offset, k_offset])
                        else:
                            x = tl.load(
                                x_ptr
                                + x_row_load_idx
                                + (k_offset + block_range_k)[None, :],
                                mask=mk_mask,
                            )

                        if USE_TMA_LOAD_dY:
                            dY = dY_desc.load([m_global_offset, n_offset])
                        else:
                            dY = tl.load(
                                dY_ptr
                                + dY_row_load_idx
                                + (n_offset + block_range_n)[None, :],
                                mask=mn_mask,
                            )

                        accumulator += tl.dot(
                            dY.T,  # [BLOCK_N, BLOCK_M]
                            x,  # [BLOCK_M, BLOCK_K]
                        )

                y = accumulator.to(output_dtype)
                if USE_TMA_STORE:
                    # Need to expand dims to match [E, N, K] shape
                    y = tl.expand_dims(y, 0)
                    dW_desc.store([expert_idx, n_offset, k_offset], y)
                else:
                    tl.store(
                        dW_ptr
                        # + (n_offset + offs_n)[:, None] * K
                        + store_row_offs[:, None] * K
                        + (k_offset + block_range_k)[None, :],
                        y,
                        mask=nk_mask,
                    )


_autotuned_grouped_gemm_dW_kernel = triton.autotune(
    configs=get_dW_kernel_configs(),
    prune_configs_by={"early_config_prune": prune_kernel_configs_backward_dW},
    key=["NUM_EXPERTS", "NUM_TOKENS", "N", "K", "PERMUTE_X", "PERMUTE_Y"],
)(_grouped_gemm_dW_kernel)

__name__ = _prev_name
_module_exports = {k: globals()[k] for k in set(globals()) - _module_before}
_register_module(_module_name, _module_exports)
del _module_exports
del _module_before
del _module_name
del _prev_name

# ===== Module: grouped_gemm.kernels.tuning =====
_prev_name = __name__
_module_name = 'grouped_gemm.kernels.tuning'
__name__ = _module_name
_ensure_package(_module_name.rpartition('.')[0])
_module_before = set(globals())

# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""
Manual tuning utils
"""

from collections import OrderedDict
from dataclasses import asdict, dataclass, fields
from itertools import product
from typing import Optional

import pandas as pd
import torch
import triton
from triton.runtime.errors import OutOfResources

from grouped_gemm.kernels.autotuning import (
    BOOLS,
    DEFAULT_K_BLOCK_SIZES,
    DEFAULT_M_BLOCK_SIZES,
    DEFAULT_N_BLOCK_SIZES,
    DEFAULT_NUM_STAGES,
    DEFAULT_NUM_WARPS,
)


@dataclass
class DeviceProperties:
    NUM_SM: int
    NUM_REGS: int
    SIZE_SMEM: int
    WARP_SIZE: int


_DEVICE_PROPERTIES: Optional[DeviceProperties] = None


def get_device_properties():
    global _DEVICE_PROPERTIES
    if _DEVICE_PROPERTIES is None:
        properties = triton.runtime.driver.active.utils.get_device_properties(
            torch.cuda.current_device()
        )
        NUM_SM = properties["multiprocessor_count"]
        NUM_REGS = properties["max_num_regs"]
        SIZE_SMEM = properties["max_shared_mem"]
        WARP_SIZE = properties["warpSize"]
        _DEVICE_PROPERTIES = DeviceProperties(NUM_SM, NUM_REGS, SIZE_SMEM, WARP_SIZE)
    return _DEVICE_PROPERTIES


@dataclass
class KernelConfig:
    BLOCK_SIZE_M: int = 32
    BLOCK_SIZE_N: int = 32
    BLOCK_SIZE_K: int = 32
    num_warps: int = 4
    num_stages: int = 2
    flatten: bool = True
    permute_x: bool = False
    permute_y: bool = False
    fuse_mul_post: bool = False
    use_tma_store: bool = False

    def to_string(self, include_tuning_params: bool = False, include_tma: bool = False):
        s = []
        if self.permute_x:
            s.append("permute_x")
        if self.permute_y:
            s.append("permute_y")
        if include_tuning_params:
            s.append(
                f"BLOCK_SIZE_M={self.BLOCK_SIZE_M},BLOCK_SIZE_N={self.BLOCK_SIZE_N},BLOCK_SIZE_K={self.BLOCK_SIZE_K},num_warps={self.num_warps},num_stages={self.num_stages},flatten={self.flatten}"
            )
        if include_tma:
            for f in fields(self):
                if f.name.startswith("use_tma_"):
                    if getattr(self, f.name):
                        s.append(f.name)
        return ",".join(s)


@dataclass
class KernelConfigForward(KernelConfig):
    use_tma_load_w: bool = False
    use_tma_load_x: bool = False


@dataclass
class KernelConfigBackward_dW(KernelConfig):
    use_tma_load_dy: bool = False
    use_tma_load_x: bool = False


@dataclass
class KernelConfigBackward_dX(KernelConfig):
    use_tma_load_dy: bool = False
    use_tma_load_w: bool = False


@dataclass
class KernelResult:
    torch_time: float
    triton_time: float
    speedup: float
    kernel_config: KernelConfig

    def to_dict(self):
        return OrderedDict(
            **asdict(self.kernel_config),
            torch_time=self.torch_time,
            triton_time=self.triton_time,
            speedup=self.speedup,
        )

    @staticmethod
    def to_dataframe(
        results: list["KernelResult"], sort_by: str = "speedup", ascending: bool = False
    ):
        df = pd.DataFrame([result.to_dict() for result in results])
        df = df.sort_values(by=sort_by, ascending=ascending)
        return df

    @staticmethod
    def to_csv(
        results: list["KernelResult"],
        sort_by: str = "speedup",
        ascending: bool = False,
        filename: str = "results.csv",
    ):
        df = KernelResult.to_dataframe(results, sort_by, ascending)
        df.to_csv(filename, index=False)

    @staticmethod
    def print_table(
        results: list["KernelResult"],
        sort_by: str = "speedup",
        ascending: bool = False,
        num_results: int = 10,
    ):
        df = KernelResult.to_dataframe(results, sort_by, ascending)
        print(df.head(num_results).to_string(index=False))


def get_kernel_configs(
    BLOCK_M=DEFAULT_M_BLOCK_SIZES,
    BLOCK_N=DEFAULT_N_BLOCK_SIZES,
    BLOCK_K=DEFAULT_K_BLOCK_SIZES,
    num_warps=DEFAULT_NUM_WARPS,
    num_stages=DEFAULT_NUM_STAGES,
    use_tma_loads=BOOLS,
    fuse_permute=BOOLS,
):
    kernel_configs_fwd = []
    kernel_configs_backward_dW = []
    kernel_configs_backward_dX = []
    for block_m, block_n, block_k, w, s, use_tma_load, permute in product(
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, use_tma_loads, fuse_permute
    ):
        kernel_configs_fwd.append(
            KernelConfigForward(
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_N=block_n,
                BLOCK_SIZE_K=block_k,
                num_warps=w,
                num_stages=s,
                use_tma_load_x=use_tma_load,
                use_tma_load_w=use_tma_load,
                use_tma_store=False,
                permute_x=permute,
                permute_y=permute,
            )
        )
        kernel_configs_backward_dW.append(
            KernelConfigBackward_dW(
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_N=block_n,
                BLOCK_SIZE_K=block_k,
                num_warps=w,
                num_stages=s,
                use_tma_load_dy=use_tma_load,
                use_tma_load_x=use_tma_load,
                use_tma_store=False,
                permute_x=permute,
                permute_y=permute,
            )
        )
        kernel_configs_backward_dX.append(
            KernelConfigBackward_dX(
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_N=block_n,
                BLOCK_SIZE_K=block_k,
                num_warps=w,
                num_stages=s,
                use_tma_load_dy=use_tma_load,
                use_tma_load_w=use_tma_load,
                use_tma_store=False,
                permute_x=permute,
                permute_y=permute,
            )
        )

    kernel_configs_fwd = prune_kernel_configs_fwd(kernel_configs_fwd)
    kernel_configs_backward_dW = prune_kernel_configs_backward_dW(
        kernel_configs_backward_dW
    )
    kernel_configs_backward_dX = prune_kernel_configs_backward_dX(
        kernel_configs_backward_dX
    )
    return kernel_configs_fwd, kernel_configs_backward_dW, kernel_configs_backward_dX


def prune_kernel_configs_fwd(configs: list[KernelConfigForward]):
    pruned_configs = []
    for config in configs:
        if config.use_tma_load_x and config.permute_x:
            continue
        if config.permute_x and config.permute_y:
            continue
        if config.use_tma_store and config.permute_y:
            continue
        pruned_configs.append(config)
    return pruned_configs


def prune_kernel_configs_backward_dX(configs: list[KernelConfigBackward_dX]):
    pruned_configs = []
    for config in configs:
        if config.use_tma_load_dy and config.permute_y:
            continue
        if config.permute_x and config.permute_y:
            continue
        if config.use_tma_store and config.permute_x:
            continue
        pruned_configs.append(config)
    return pruned_configs


def prune_kernel_configs_backward_dW(configs: list[KernelConfigBackward_dW]):
    pruned_configs = []
    for config in configs:
        if config.use_tma_load_dy and config.permute_y:
            continue
        if config.use_tma_load_x and config.permute_x:
            continue
        if config.permute_x and config.permute_y:
            continue
        pruned_configs.append(config)
    return pruned_configs


class TritonTuningContext:
    def __init__(self, kernel_config: KernelConfig):
        self.kernel_config = kernel_config
        self.success = True

    def __enter__(self):
        # Setup code can be added here if needed
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is OutOfResources:
            name = exc_value.name
            required = exc_value.required
            limit = exc_value.limit
            print(
                f"Kernel config {self.kernel_config} failed: {name}, required: {required}, limit: {limit}"
            )
            self.success = False
        elif exc_type is not None:
            print(
                f"Error running Triton grouped GEMM for kernel config: {self.kernel_config}: {exc_value}"
            )
            self.success = False
        # Return False to propagate exceptions, True to suppress them
        return True

__name__ = _prev_name
_module_exports = {k: globals()[k] for k in set(globals()) - _module_before}
_register_module(_module_name, _module_exports)
del _module_exports
del _module_before
del _module_name
del _prev_name

# ===== Module: grouped_gemm.interface =====
_prev_name = __name__
_module_name = 'grouped_gemm.interface'
__name__ = _module_name
_ensure_package(_module_name.rpartition('.')[0])
_module_before = set(globals())

# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

import logging
import warnings
from dataclasses import asdict

import torch
import triton

from grouped_gemm.kernels.backward import (
    _autotuned_grouped_gemm_dW_kernel,
    _autotuned_grouped_gemm_dX_kernel,
    _grouped_gemm_dW_kernel,
    _grouped_gemm_dX_kernel,
)
from grouped_gemm.kernels.forward import (
    _autotuned_grouped_gemm_forward_kernel,
    _grouped_gemm_forward_kernel,
)
from grouped_gemm.kernels.tuning import (
    KernelConfigBackward_dW,
    KernelConfigBackward_dX,
    KernelConfigForward,
)

logger = logging.getLogger(__name__)
# Set formatter to include timestamp, pathname and lineno
formatter = logging.Formatter(
    "%(asctime)s::%(levelname)s,%(pathname)s:%(lineno)d:: %(message)s"
)

# Add console handler
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

_FUSED_MUL_WARN = False
_SUPPORTS_TMA = None


def supports_tma():
    global _SUPPORTS_TMA
    if _SUPPORTS_TMA is None:
        _SUPPORTS_TMA = torch.cuda.get_device_capability()[0] >= 9
    return _SUPPORTS_TMA


_per_device_alloc_fns = {}


def get_per_device_per_stream_alloc_fn(device):
    if device not in _per_device_alloc_fns:
        _per_stream_tensors = {}

        def alloc_fn(size: int, alignment: int, stream):
            assert alignment == 128
            if (
                stream not in _per_stream_tensors
                or _per_stream_tensors[stream].numel() < size
            ):
                _per_stream_tensors[stream] = torch.empty(
                    size, device=device, dtype=torch.int8
                )
                _per_stream_tensors[stream].__hibernate__ = {"type": "ignore"}
            return _per_stream_tensors[stream]

        _per_device_alloc_fns[device] = alloc_fn
    return _per_device_alloc_fns[device]


def log_kernel_info(
    compiled_kernel: triton.compiler.CompiledKernel, best_config: triton.Config = None
):
    kernel_name = compiled_kernel.name
    nregs = compiled_kernel.n_regs
    nspills = compiled_kernel.n_spills
    metadata = compiled_kernel.metadata
    logger.debug(
        f"{kernel_name}: n_regs={nregs} n_spills={nspills} metadata={metadata}"
    )
    if best_config is not None:
        logger.debug(f"{kernel_name} autotuned best_config: {best_config}")


def grouped_gemm_forward(
    X: torch.Tensor,
    W: torch.Tensor,
    topk: int,
    m_sizes: torch.Tensor,
    gather_indices: torch.Tensor = None,
    topk_weights: torch.Tensor = None,
    # Fusions
    permute_x: bool = False,
    permute_y: bool = False,
    fuse_mul_post: bool = False,
    # Autotuning - manual kernel params will be ignored if autotune is True
    autotune: bool = False,
    # Kernel tuning params if not autotuning -- NOTE: these params need to be tuned, otherwise performance will be poor
    BLOCK_SIZE_M: int = 32,
    BLOCK_SIZE_N: int = 32,
    BLOCK_SIZE_K: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
    use_tma_load_w: bool = False,
    use_tma_load_x: bool = False,
    use_tma_store: bool = False,
    # software pipelining -- set to True for now, won't impact until loop is re-written
    flatten: bool = True,
    # debugging
    debug: bool = False,
) -> torch.Tensor:
    """
    Grouped GEMM forward pass for MoE MLPs.

    The implementation offers a number of fusions specific to MoE:
    - `permute_x`: fuse the permutation of hidden states from token order (original order) to grouped expert order, typically only needed for the first grouped GEMM in an MoE MLP.
        - When `permute_x` is True, `X` is expected to be of shape (num_tokens, K).
        - When `permute_x` is False, `X` is expected to be of shape (total_tokens, K) where `total_tokens = num_tokens * topk` AND already permuted to grouped expert order, i.e., hidden states are sorted such that tokens assigned to each expert are contiguous.
    - `permute_y`: fused the permutation of the output from expert grouped order back to original token order, typically only needed for the second grouped GEMM in an MoE MLP.
    - `fuse_mul_pre`: fuse the multiplication of the routed input with topk_weights, only done in the first grouped GEMM in an MoE MLP as for Llama4.  Do not use, since results in performance regression as it interrupts the GEMM mainloop.
    - `fuse_mul_post`: fuse the multiplication of the routed output with topk_weights, used only when `permute_y` is True. NOTE: this should only be used when using this kernel for inference, not for training.

    X: (M, K) hidden states where M is the num_tokens if `permute_x` is True, otherwise `total_tokens` where `total_tokens = num_tokens * topk`.
    W: (E, N, K) expert weights, where E is number of experts, N in the intermediate (output) dim, and K is the reduction dim
    m_sizes: tokens assigned to each expert which correspond to the size of M in the respective GEMMs in the grouped GEMM.
    gather_indices: (total_tokens,) indices of tokens assigned to each expert.  E.g., slicing gather_indices by cumsum of m_sizes gives the indices of tokens assigned to each expert.
    topk_weights: (total_tokens,) weights to multiply routed output by in expert MLP calculation, used only when `fuse_mul_post` is True (see note on `fuse_mul_post`).
    use_fast_accum: currently unused; trade off faster accumulation dtype in GEMM for less precision.
    use_tma_load_x: use TMA for loading activations, incompatible with permute_x.  TODO: add TMA gather / scatter support for Blackwell+.
    use_tma_load_w: use TMA for loading weights.  If TMA supported, this should always be enabled as it is faster than global memory load.
    use_tma_store: use TMA for storing output, incompatible with permute_y.  TODO: add TMA scatter support for Blackwell+.

    Returns:
        y: (total_tokens, N) output of grouped GEMM
    """

    assert X.device.type == "cuda", "X and W must be on CUDA"
    assert m_sizes.device.type == "cuda", "m_sizes must be on CUDA"

    X = X.contiguous()
    W = W.contiguous()
    m_sizes = m_sizes.contiguous()

    # Preconditions
    assert not (permute_x and permute_y), "Cannot permute both X and Y"
    assert not (permute_y and use_tma_store), "Cannot use both TMA store and permute_y"

    if use_tma_load_x:
        # TMA load for activations, TMA gather only supported on Blackwell+
        assert not permute_x, "Cannot use both use_tma_load_x and permute_x"

    use_tma = use_tma_load_w or use_tma_load_x or use_tma_store
    if not supports_tma() and use_tma:
        warnings.warn("TMA not supported, tma_load will be set to False")
        use_tma_load_w = False
        use_tma_load_x = False
        use_tma_store = False

    if use_tma or autotune:

        def alloc_fn(size: int, alignment: int, stream: int):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

    X = X.view(-1, X.shape[-1])
    W = W.view(-1, W.shape[-1])

    if permute_x or permute_y:
        assert gather_indices is not None, (
            "gather_indices must be provided when permute_x or permute_y is True"
        )
        assert gather_indices.is_contiguous()
        assert gather_indices.device.type == "cuda"
        assert gather_indices.ndim == 1
        total_tokens = gather_indices.shape[0]
        num_tokens = total_tokens // topk
        if permute_x:
            assert X.shape[0] == num_tokens, (
                f"X.shape[0] ({X.shape[0]}) must match num_tokens ({num_tokens})"
            )
        else:
            assert X.shape[0] == total_tokens, (
                f"X.shape[0] ({X.shape[0]}) must match total_tokens ({total_tokens})"
            )
    else:
        total_tokens = X.shape[0]
        num_tokens = total_tokens // topk

    num_experts = m_sizes.shape[0]
    _, K = X.shape
    N = W.shape[0] // num_experts
    assert K == W.shape[1], f"K ({K}) must match W.shape[1] ({W.shape[1]})"

    if fuse_mul_post:
        global _FUSED_MUL_WARN
        if not _FUSED_MUL_WARN:
            warnings.warn(
                "fused_mul should only be used for inference, not for training"
            )
            _FUSED_MUL_WARN = True
        assert permute_y, "FUSE_MUL requires PERMUTE_Y"
        assert topk_weights is not None
        assert topk_weights.numel() == total_tokens
        assert topk_weights.device.type == "cuda"
        assert topk_weights.is_contiguous()
        topk_weights = topk_weights.view(-1)
        if debug:
            print(
                f"DEBUG::GROUPED_GEMM {topk_weights.tolist()} {gather_indices.tolist()}"
            )

    y = torch.empty((total_tokens, N), device=X.device, dtype=X.dtype)
    if total_tokens == 0 or N == 0:
        return y

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    def grid(META):
        return (NUM_SMS,)

    if not autotune:
        BLOCK_SIZE_K = min(K, BLOCK_SIZE_K)
        BLOCK_SIZE_N = min(N, BLOCK_SIZE_N)
        BLOCK_SIZE_M = min(total_tokens, BLOCK_SIZE_M)

    if debug:
        print(
            f"DEBUG::GROUPED_GEMM {num_tokens=} {topk=} {num_experts=} {N=} {K=} {BLOCK_SIZE_M=} {BLOCK_SIZE_N=} {BLOCK_SIZE_K=} {permute_x=}"
        )
        print(
            f"DEBUG::GROUPED_GEMM {m_sizes.tolist()} {(gather_indices // topk).tolist()}"
        )

    kernel_args = {
        # Inputs
        "x_ptr": X,
        "w_ptr": W,
        "m_sizes_ptr": m_sizes,
        "gather_indices_ptr": gather_indices,
        "topk_weights_ptr": topk_weights,
        # Output
        "y_ptr": y,
        # Problem shapes
        "NUM_TOKENS": num_tokens,
        "NUM_EXPERTS": num_experts,
        "TOPK": topk,
        "N": N,
        "K": K,
        "NUM_SMS": NUM_SMS,
        # Gather / Scatter
        "PERMUTE_X": permute_x,
        "PERMUTE_Y": permute_y,
        # TopK weight merging
        "FUSE_MUL_POST": fuse_mul_post,
        # Loop pipelining
        "FLATTEN": flatten,
    }
    if not autotune:
        kernel_args.update(
            {
                "USE_TMA_LOAD_W": use_tma_load_w,
                "USE_TMA_LOAD_X": use_tma_load_x,
                "USE_TMA_STORE": use_tma_store,
                "BLOCK_SIZE_M": BLOCK_SIZE_M,
                "BLOCK_SIZE_N": BLOCK_SIZE_N,
                "BLOCK_SIZE_K": BLOCK_SIZE_K,
                "num_warps": num_warps,
                "num_stages": num_stages,
            }
        )

    kernel = (
        _autotuned_grouped_gemm_forward_kernel
        if autotune
        else _grouped_gemm_forward_kernel
    )
    compiled_kernel: triton.compiler.CompiledKernel = kernel[grid](**kernel_args)

    if autotune:
        log_kernel_info(compiled_kernel, kernel.best_config)
    else:
        log_kernel_info(compiled_kernel)

    return y


def grouped_gemm_dX(
    dY: torch.Tensor,
    W: torch.Tensor,
    gather_indices: torch.Tensor,
    m_sizes: torch.Tensor,
    topk: int,
    BLOCK_SIZE_M: int = 32,
    BLOCK_SIZE_N: int = 32,
    BLOCK_SIZE_K: int = 32,
    debug: bool = False,
    permute_x: bool = False,
    permute_y: bool = False,
    use_tma_load_w: bool = False,
    use_tma_load_dy: bool = False,
    use_tma_store: bool = False,
    num_warps: int = 4,
    num_stages: int = 2,
    flatten: bool = True,
    fuse_mul_pre: bool = False,
    fuse_mul_post: bool = False,
    autotune: bool = False,
) -> torch.Tensor:
    """
    dX backward kernel
    grad_output: (M, N)
    gather_indices: (total_tokens,), indices of tokens assigned to each expert.  E.g., slicing gather_indices by cumsum of m_sizes gives the indices of tokens assigned to each expert.
    m_sizes: tokens assigned to each expert which correspond to the size of M in the respective GEMMs in the grouped GEMM.
    topk: number of experts chosen per token.
    `permute_x`: whether X was permuted on load in the forward pass, typically only used for the first grouped GEMM in an MoE MLP to group tokens by expert.
    - In the forward pass, if we permuted X on load, we need to permute store in the backward pass
    - Shapes
        - the forward pass input X shape is [NUM_TOKENS, K], reduce across K, output y is [NUM_TOKENS * TOPK, K]
        - the backward pass input dy shape is [NUM_TOKENS * TOPK, N], reduce across N, output dX is [NUM_TOKENS * TOPK, K]
    - Note that in the backward pass, the output size is still [NUM_TOKENS * TOPK, K] since we still need to accumulate gradients for each expert chosen by the token in a post-processing step.
    `permute_y`: whether the output was permuted on store in the forward pass, typically only used for the second grouped GEMM in an MoE MLP to restore to the original token order.
    - In the forward pass, if we permuted output on store (e.g., in the second grouped GEMM in fused MoE MLP), we need to permute on load to get from token order to expert grouped order
    - We still store in contiguous order since we are writing out dX which will be the input to the backwards pass of the first grouped GEMM
    `fuse_mul_{pre,post}`: always set to False since this should only be used for inference.
    use_tma_load_dy: use TMA for loading dy. use_tma_load_dy is incompatible with permute_y.  TODO: add TMA gather / scatter support for Blackwell+ which will enable permute_y and use_tma_load_dy.
    use_tma_load_w: use TMA for loading weights.  If TMA supported, this should always be enabled as it is faster than global memory load.
    use_tma_store: use TMA for storing dX.  Incompatible with permute_x.  TODO: add TMA gather / scatter support for Blackwell+ which will enable permute_x and use_tma_store.
    """
    assert not fuse_mul_pre, (
        "fuse_mul_pre should only be used for inference, not for training"
    )
    assert not fuse_mul_post, (
        "fuse_mul_post should only be used for inference, not for training"
    )
    assert dY.is_contiguous()
    assert W.is_contiguous()
    assert m_sizes.is_contiguous()
    assert m_sizes.ndim == 1

    # Preconditions
    assert not (permute_x and permute_y), "Cannot permute both X and Y"
    # Note that this is flipped from the forward pass
    # If we permuted y in the forward, we need to permute on load in the backward
    assert not (permute_y and use_tma_load_dy), "Cannot use both TMA load and permute_y"
    assert not (permute_x and use_tma_store), "Cannot use both TMA store and permute_x"

    use_tma = use_tma_load_dy or use_tma_load_w or use_tma_store
    if not supports_tma() and use_tma:
        warnings.warn("TMA not supported, tma_load will be set to False")
        use_tma_load_w = False
        use_tma_load_dy = False
        use_tma_store = False

    if use_tma or autotune:

        def alloc_fn(size: int, alignment: int, stream: int):
            # print(f"DEBUG::GROUPED_GEMM alloc_fn {size=} {alignment=} {stream=}")
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

    num_experts = m_sizes.shape[0]
    dY = dY.view(-1, dY.shape[-1])
    W = W.view(-1, W.shape[-1])

    M_total, N_grad = dY.shape
    N_total, K = W.shape
    N = N_total // num_experts
    assert N_grad == N, f"Grad_output N ({N_grad}) must match weight N ({N})"

    assert M_total % topk == 0, (
        f"M_total ({M_total}) must be divisible by topk ({topk})"
    )
    num_tokens = M_total // topk

    total_tokens = gather_indices.shape[0]
    assert total_tokens == M_total, (
        f"Total tokens ({total_tokens}) must match M_total ({M_total})"
    )

    # Note that the output shape is [NUM_TOKENS * TOPK, K] even when `permute_x` is True since we need to accumulate gradients across all experts chosen by the token.
    # This will be done in a post-processing step reduction step.
    output_shape = (total_tokens, K)
    dX = torch.zeros(output_shape, device=dY.device, dtype=dY.dtype)

    NUM_SMS = torch.cuda.get_device_properties(
        "cuda"
    ).multi_processor_count  # if not debug else 1

    def grid(META):
        return (NUM_SMS,)

    if not autotune:
        BLOCK_SIZE_M = min(M_total, BLOCK_SIZE_M)
        BLOCK_SIZE_N = min(N_grad, BLOCK_SIZE_N)
        BLOCK_SIZE_K = min(K, BLOCK_SIZE_K)

    if debug:
        print(
            f"DEBUG::GROUPED_GEMM {num_tokens=} {topk=} {output_shape=} {num_experts=} {N=} {K=} {BLOCK_SIZE_M=} {BLOCK_SIZE_N=} {BLOCK_SIZE_K=} {NUM_SMS=}"
        )
        print(f"DEBUG::GROUPED_GEMM {m_sizes.tolist()}")

    kernel_args = {
        # Inputs
        "dY_ptr": dY,
        "w_ptr": W,
        "gather_indices_ptr": gather_indices,
        "m_sizes_ptr": m_sizes,
        # Output
        "dX_ptr": dX,
        # Problem sizes
        "NUM_EXPERTS": num_experts,
        "NUM_TOKENS": num_tokens,
        "TOPK": topk,
        "N": N,
        "K": K,
        "NUM_SMS": NUM_SMS,
        # Gather / Scatter
        "PERMUTE_X": permute_x,
        "PERMUTE_Y": permute_y,
        "FLATTEN": flatten,
    }
    if not autotune:
        kernel_args.update(
            {
                "BLOCK_SIZE_M": BLOCK_SIZE_M,
                "BLOCK_SIZE_N": BLOCK_SIZE_N,
                "BLOCK_SIZE_K": BLOCK_SIZE_K,
                "num_warps": num_warps,
                "num_stages": num_stages,
                "USE_TMA_LOAD_dY": use_tma_load_dy,
                "USE_TMA_LOAD_W": use_tma_load_w,
                "USE_TMA_STORE": use_tma_store,
            }
        )
    kernel = _autotuned_grouped_gemm_dX_kernel if autotune else _grouped_gemm_dX_kernel
    compiled_kernel: triton.compiler.CompiledKernel = kernel[grid](**kernel_args)

    if autotune:
        log_kernel_info(compiled_kernel, kernel.best_config)
    else:
        log_kernel_info(compiled_kernel)
    return dX


def grouped_gemm_dW(
    X: torch.Tensor,
    dY: torch.Tensor,
    m_sizes: torch.Tensor,
    gather_indices: torch.Tensor,
    topk: int,
    BLOCK_SIZE_M: int = 32,
    BLOCK_SIZE_N: int = 32,
    BLOCK_SIZE_K: int = 32,
    permute_x: bool = False,
    permute_y: bool = False,
    use_tma_load_dy: bool = False,
    use_tma_load_x: bool = False,
    use_tma_store: bool = False,
    fuse_mul_pre: bool = False,
    fuse_mul_post: bool = False,
    num_warps: int = 4,
    num_stages: int = 2,
    flatten: bool = True,
    autotune: bool = False,
    debug: bool = False,
) -> torch.Tensor:
    """
    X: (M, K) hidden states where M is the num_tokens if `permute_x` is True, otherwise `total_tokens` where `total_tokens = num_tokens * topk`.
    dY: (M, N)
    topk: number of experts to choose per token.
    m_sizes: tokens assigned to each expert which correspond to the size of M in the respective GEMMs in the grouped GEMM.
    gather_indices: (total_tokens,) indices of tokens assigned to each expert.  E.g., slicing gather_indices by cumsum of m_sizes gives the indices of tokens assigned to each expert.
    permute_x: whether X was permuted on load in the forward pass, typically only used for the first grouped GEMM in an MoE MLP to group tokens by expert.
    - for the first grouped GEMM, we permuted on load -> X was [num_tokens, K] and stored y in expert grouped order [num_tokens * topk, K]
    - in the backwards pass, we need to permute on load of X while loading dy in contiguous (expert grouped) order
    - since we are writing out dW, there is no need to permute on store
    permute_y: whether the output was permuted on store in the forward pass, typically only used for the second grouped GEMM in an MoE MLP to restore to the original token order.
    - for the second grouped GEMM, we permuted on store -> y was permuted from expert grouped order to token order while X was loaded in expert grouped order since it was the output of the first grouped GEMM
    - in the backwards pass, we need to permute on load of dy to get from token order to expert grouped order to match the order of X
    - since we are writing out dW, there is no need to permute on store
    use_tma_load_dy: use TMA for loading dy. use_tma_load_dy is incompatible with permute_y.  TODO: add TMA gather / scatter support for Blackwell+ which will enable permute_y and use_tma_load_dy.
    use_tma_load_x: use TMA for loading x. use_tma_load_x is incompatible with permute_x.  TODO: add TMA gather / scatter support for Blackwell+ which will enable permute_x and use_tma_load_x.
    use_tma_store: use TMA for storing dW.  If TMA supported, this should always be enabled as it is faster than global memory store.
    """
    assert not fuse_mul_pre, "fuse_mul_pre not supported"
    assert not fuse_mul_post, "fuse_mul_post not supported"
    NUM_SMS = (
        torch.cuda.get_device_properties("cuda").multi_processor_count
        if not debug
        else 1
    )
    X = X.view(-1, X.shape[-1]).contiguous()
    dY = dY.contiguous()
    m_sizes = m_sizes.contiguous()

    # Preconditions
    assert not (permute_x and permute_y), "Cannot permute both X and Y"
    assert not (permute_y and use_tma_load_dy), "Cannot use both TMA load and permute_y"
    assert not (permute_x and use_tma_load_x), "Cannot use both TMA load and permute_x"

    use_tma = use_tma_load_dy or use_tma_load_x or use_tma_store
    if not supports_tma() and use_tma:
        warnings.warn("TMA not supported, tma_load will be set to False")
        use_tma_load_x = False
        use_tma_load_dy = False
        use_tma_store = False

    if use_tma or autotune:

        def alloc_fn(size: int, alignment: int, stream: int):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

    if permute_x or permute_y:
        assert gather_indices is not None
        assert gather_indices.is_contiguous()
        assert gather_indices.device.type == "cuda"
        assert gather_indices.ndim == 1
        total_tokens = gather_indices.shape[0]
        num_tokens = total_tokens // topk
        if permute_x:
            assert X.shape[0] == num_tokens
        else:
            assert X.shape[0] == total_tokens
    else:
        total_tokens = X.shape[0]
        num_tokens = total_tokens // topk

    num_experts = m_sizes.shape[0]
    # Get dimensions
    _, K = X.shape
    M_grad, N = dY.shape

    assert M_grad == total_tokens, f"dY M ({M_grad}) != total_tokens ({total_tokens})"

    dW = torch.zeros((num_experts, N, K), device=X.device, dtype=X.dtype)

    if not autotune:
        BLOCK_SIZE_M = min(total_tokens, BLOCK_SIZE_M)
        BLOCK_SIZE_N = min(N, BLOCK_SIZE_N)
        BLOCK_SIZE_K = min(K, BLOCK_SIZE_K)

    def grid(META):
        return (NUM_SMS,)

    if debug:
        print(
            f"DEBUG::GROUPED_GEMM_DW_TMA {num_experts=} {N=} {K=} {BLOCK_SIZE_M=} {BLOCK_SIZE_N=} {BLOCK_SIZE_K=} {NUM_SMS=}"
        )

        print(f"DEBUG::GROUPED_GEMM_DW_TMA {m_sizes.tolist()=}")
        print(f"DEBUG::GROUPED_GEMM_DW_TMA {gather_indices.tolist()=}")
        m_start = 0
        for i in range(num_experts):
            expert_token_idx = gather_indices[m_start : m_start + m_sizes[i]]
            t_start = 0
            while t_start < m_sizes[i]:
                token_idx = expert_token_idx[t_start : t_start + BLOCK_SIZE_M]
                if permute_x:
                    token_idx = token_idx // topk
                print(
                    f"DEBUG::GROUPED_GEMM_DW_TMA Token expert {i} indices: {token_idx.tolist()}"
                )
                t_start += BLOCK_SIZE_M

            m_start += m_sizes[i]

    kernel_args = {
        # Inputs
        "x_ptr": X,
        "dY_ptr": dY,
        "m_sizes_ptr": m_sizes,
        "gather_indices_ptr": gather_indices,
        # Output
        "dW_ptr": dW,
        # Problem sizes
        "NUM_TOKENS": num_tokens,
        "TOPK": topk,
        "NUM_EXPERTS": num_experts,
        "N": N,
        "K": K,
        "NUM_SMS": NUM_SMS,
        # Gather / Scatter
        "PERMUTE_X": permute_x,
        "PERMUTE_Y": permute_y,
        # Loop pipelining
        "FLATTEN": flatten,
    }

    if not autotune:
        kernel_args.update(
            {
                "BLOCK_SIZE_M": BLOCK_SIZE_M,
                "BLOCK_SIZE_N": BLOCK_SIZE_N,
                "BLOCK_SIZE_K": BLOCK_SIZE_K,
                "USE_TMA_LOAD_dY": use_tma_load_dy,
                "USE_TMA_LOAD_X": use_tma_load_x,
                "USE_TMA_STORE": use_tma_store,
                "num_warps": num_warps,
                "num_stages": num_stages,
            }
        )

    kernel = _autotuned_grouped_gemm_dW_kernel if autotune else _grouped_gemm_dW_kernel
    compiled_kernel: triton.compiler.CompiledKernel = kernel[grid](**kernel_args)

    if autotune:
        log_kernel_info(compiled_kernel, kernel.best_config)
    else:
        log_kernel_info(compiled_kernel)

    return dW


class GroupedGemm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        X,
        W,
        m_sizes,
        topk,
        gather_indices,
        permute_x,
        permute_y,
        topk_weights,
        fuse_mul_post,
        kernel_config_fwd,
        kernel_config_bwd_dX,
        kernel_config_bwd_dW,
        autotune,
        dX_only,
        dW_only,
    ):
        ctx.topk = topk
        ctx.permute_x = permute_x
        ctx.permute_y = permute_y
        ctx.fuse_mul_post = fuse_mul_post
        ctx.kernel_config_fwd = kernel_config_fwd
        ctx.kernel_config_bwd_dX = kernel_config_bwd_dX
        ctx.kernel_config_bwd_dW = kernel_config_bwd_dW
        ctx.autotune = autotune
        ctx.dX_only = dX_only
        ctx.dW_only = dW_only

        # NOTE: we don't save topk_weights for backward since we do not support training with fused_mul
        ctx.save_for_backward(X, W, m_sizes, gather_indices)

        fwd_config = {}
        if kernel_config_fwd is not None:
            fwd_config["BLOCK_SIZE_M"] = kernel_config_fwd.BLOCK_SIZE_M
            fwd_config["BLOCK_SIZE_N"] = kernel_config_fwd.BLOCK_SIZE_N
            fwd_config["BLOCK_SIZE_K"] = kernel_config_fwd.BLOCK_SIZE_K
            fwd_config["num_warps"] = kernel_config_fwd.num_warps
            fwd_config["num_stages"] = kernel_config_fwd.num_stages
            fwd_config["use_tma_load_x"] = kernel_config_fwd.use_tma_load_x
            fwd_config["use_tma_load_w"] = kernel_config_fwd.use_tma_load_w
            fwd_config["use_tma_store"] = kernel_config_fwd.use_tma_store

        return grouped_gemm_forward(
            X=X,
            W=W,
            topk=topk,
            m_sizes=m_sizes,
            gather_indices=gather_indices,
            topk_weights=topk_weights,
            permute_x=permute_x,
            permute_y=permute_y,
            fuse_mul_post=fuse_mul_post,
            # Autotune -- this will override the manual kernel config if true
            autotune=autotune,
            # Manual kernel config
            **fwd_config,
        )

    @staticmethod
    def backward(ctx, dY):
        X, W, m_sizes, gather_indices = ctx.saved_tensors
        topk = ctx.topk
        permute_x = ctx.permute_x
        permute_y = ctx.permute_y
        fuse_mul_post = ctx.fuse_mul_post
        kernel_config_bwd_dX = ctx.kernel_config_bwd_dX
        kernel_config_bwd_dW = ctx.kernel_config_bwd_dW
        autotune = ctx.autotune
        dX_only = ctx.dX_only
        dW_only = ctx.dW_only

        if not autotune:
            if not dW_only:
                assert kernel_config_bwd_dX is not None, (
                    "kernel_config_bwd_dX must be provided if autotune is False"
                )
            if not dX_only:
                assert kernel_config_bwd_dW is not None, (
                    "kernel_config_bwd_dW must be provided if autotune is False"
                )

        assert not fuse_mul_post, (
            "fused_mul should only be used for inference, not for training"
        )

        if not dX_only:
            bwd_dW_config = {}

            if kernel_config_bwd_dW is not None:
                bwd_dW_config["use_tma_load_dy"] = kernel_config_bwd_dW.use_tma_load_dy
                bwd_dW_config["use_tma_load_x"] = kernel_config_bwd_dW.use_tma_load_x
                bwd_dW_config["use_tma_store"] = kernel_config_bwd_dW.use_tma_store
                bwd_dW_config["BLOCK_SIZE_M"] = kernel_config_bwd_dW.BLOCK_SIZE_M
                bwd_dW_config["BLOCK_SIZE_N"] = kernel_config_bwd_dW.BLOCK_SIZE_N
                bwd_dW_config["BLOCK_SIZE_K"] = kernel_config_bwd_dW.BLOCK_SIZE_K
                bwd_dW_config["num_warps"] = kernel_config_bwd_dW.num_warps
                bwd_dW_config["num_stages"] = kernel_config_bwd_dW.num_stages

            dW = grouped_gemm_dW(
                X=X,
                dY=dY,
                m_sizes=m_sizes,
                gather_indices=gather_indices,
                topk=topk,
                permute_x=permute_x,
                permute_y=permute_y,
                # Autotune -- this will override the manual kernel config if true
                autotune=autotune,
                # Manual kernel config
                **bwd_dW_config,
            )
        else:
            dW = None

        if not dW_only:
            bwd_dX_config = {}
            if kernel_config_bwd_dX is not None:
                bwd_dX_config["use_tma_load_dy"] = kernel_config_bwd_dX.use_tma_load_dy
                bwd_dX_config["use_tma_load_w"] = kernel_config_bwd_dX.use_tma_load_w
                bwd_dX_config["use_tma_store"] = kernel_config_bwd_dX.use_tma_store
                bwd_dX_config["BLOCK_SIZE_M"] = kernel_config_bwd_dX.BLOCK_SIZE_M
                bwd_dX_config["BLOCK_SIZE_N"] = kernel_config_bwd_dX.BLOCK_SIZE_N
                bwd_dX_config["BLOCK_SIZE_K"] = kernel_config_bwd_dX.BLOCK_SIZE_K
                bwd_dX_config["num_warps"] = kernel_config_bwd_dX.num_warps
                bwd_dX_config["num_stages"] = kernel_config_bwd_dX.num_stages

            dX = grouped_gemm_dX(
                dY=dY,
                W=W,
                m_sizes=m_sizes,
                gather_indices=gather_indices,
                topk=topk,
                permute_x=permute_x,
                permute_y=permute_y,
                # Autotune -- this will override the manual kernel config if true
                autotune=autotune,
                # Manual kernel config
                **bwd_dX_config,
            )

            if topk > 1 and permute_x:
                dX = dX.view(X.shape[0], topk, -1).sum(dim=1)
        else:
            dX = None

        return (
            dX,
            dW,
            None,  # m_sizes
            None,  # gather_indices
            None,  # topk
            None,  # permute_x
            None,  # permute_y
            None,  # topk_weights
            None,  # fuse_mul_post
            None,  # kernel_config_fwd
            None,  # kernel_config_bwd_dX
            None,  # kernel_config_bwd_dW
            None,  # autotune
            None,  # dX_only
            None,  # dW_only
        )


def check_valid_config_fwd(
    permute_x,
    permute_y,
    use_tma_load_x,
    use_tma_load_w,
    use_tma_store,
    fuse_mul_post,
    is_first_gemm,
):
    """
    Check if the configuration is valid for the forward pass.
    """
    is_second_gemm = not is_first_gemm

    assert not (permute_x and permute_y), "Cannot permute both X and Y"
    assert not (is_second_gemm and permute_x), (
        "Cannot permute X for the second grouped GEMM"
    )
    assert not (is_first_gemm and permute_y), (
        "Cannot permute Y for the first grouped GEMM"
    )
    assert not (fuse_mul_post and is_first_gemm), (
        "Cannot fuse mul for the first grouped GEMM"
    )
    assert not (use_tma_load_x and permute_x), (
        "Cannot use TMA load and permute X unless on sm100+ (Blackwell+)"
    )
    assert not (use_tma_store and permute_y and is_second_gemm), (
        "Cannot use TMA store and permute Y for the second grouped GEMM unless on sm100+ (Blackwell+)"
    )


def check_valid_config_bwd_dW(
    permute_x,
    permute_y,
    use_tma_load_dY,
    use_tma_load_x,
    use_tma_store,
    fuse_mul_post,
    is_first_gemm,
):
    """
    Check if the configuration is valid for the backward pass of dW.
    """
    is_second_gemm = not is_first_gemm
    if fuse_mul_post:
        assert False, "Cannot fuse_mul is not supported for backward pass"
    if is_second_gemm and permute_y and use_tma_load_dY:
        assert False, "Cannot use TMA load and permute Y for the second grouped GEMM"
    if is_first_gemm and permute_x and use_tma_load_x:
        assert False, "Cannot use TMA load and permute X for the first grouped GEMM"


def check_valid_config_bwd_dX(
    permute_x,
    permute_y,
    use_tma_load_dY,
    use_tma_load_w,
    use_tma_store,
    fuse_mul_post,
    is_first_gemm,
):
    """
    Check if the configuration is valid for the backward pass of dW.
    """
    is_second_gemm = not is_first_gemm
    if fuse_mul_post:
        assert False, "Cannot fuse_mul is not supported for backward pass"
    if is_second_gemm and permute_y and use_tma_load_dY:
        assert False, "Cannot use TMA load and permute Y for the second grouped GEMM"
    if use_tma_store and permute_x and is_first_gemm:
        assert False, "Cannot use TMA store and permute X for the first grouped GEMM"


def grouped_gemm(
    X: torch.Tensor,
    W: torch.Tensor,
    m_sizes: torch.Tensor,
    topk: int,
    gather_indices: torch.Tensor = None,
    permute_x: bool = False,
    permute_y: bool = False,
    topk_weights=None,
    fuse_mul_post=False,
    kernel_config_fwd: KernelConfigForward = None,
    kernel_config_bwd_dX: KernelConfigBackward_dX = None,
    kernel_config_bwd_dW: KernelConfigBackward_dW = None,
    autotune: bool = False,
    is_first_gemm: bool = True,
    # Only for debugging
    dX_only: bool = False,
    dW_only: bool = False,
):
    """
    Grouped GEMM for MoE MLPs.

    The implementation offers a number of fusions specific to MoE:
    - `permute_x`: fuse the permutation of hidden states from token order (original order) to grouped expert order, typically only needed for the first grouped GEMM in an MoE MLP.
        - When `permute_x` is True, `X` is expected to be of shape (num_tokens, K).
        - When `permute_x` is False, `X` is expected to be of shape (total_tokens, K) where `total_tokens = num_tokens * topk` AND already permuted to grouped expert order, i.e., hidden states are sorted such that tokens assigned to each expert are contiguous.
    - `permute_y`: fused the permutation of the output from expert grouped order back to original token order, typically only needed for the second grouped GEMM in an MoE MLP.
    - `fuse_mul`: fuse the multiplication of the routed output with topk_weights, used only when `permute_y` is True. NOTE: this should only be used when using this kernel for inference, not for training.

    X: (M, K) hidden states where M is the num_tokens if `permute_x` is True, otherwise `total_tokens` where `total_tokens = num_tokens * topk`.
    W: (E, N, K) expert weights, where E is number of experts, N in the intermediate (output) dim, and K is the reduction dim
    m_sizes: tokens assigned to each expert which correspond to the size of M in the respective GEMMs in the grouped GEMM.
    gather_indices: (total_tokens,) indices of tokens assigned to each expert.  E.g., slicing gather_indices by cumsum of m_sizes gives the indices of tokens assigned to each expert. Needed when either `permute_x` or `permute_y` is True.
    topk_weights: (total_tokens,) weights to multiply routed output by in expert MLP calculation, used only when `fuse_mul` is True (see note on `fuse_mul`).
    kernel_config_fwd: KernelConfigForward for forward pass.
    kernel_config_bwd_dX: KernelConfigBackward_dX for backward pass of dX.
    kernel_config_bwd_dW: KernelConfigBackward_dW for backward pass of dW.
    autotune: whether to autotune the kernel, if yes, kernel_config_fwd, kernel_config_bwd_dX, and kernel_config_bwd_dW will be ignored.
    is_first_gemm: whether this is the first grouped GEMM in an MoE MLP.  This is needed to check whether kernel configs are valid.  `permute_x` should only be used for first gemm; `permute_y` should only be used for second gemm.
    This will impact whether TMA can be used for loading and storing.

    """
    if not autotune:
        assert kernel_config_fwd is not None, (
            "kernel_config_fwd must be provided if autotune is False"
        )

        check_valid_config_fwd(
            permute_x,
            permute_y,
            use_tma_load_x=kernel_config_fwd.use_tma_load_x,
            use_tma_load_w=kernel_config_fwd.use_tma_load_w,
            use_tma_store=kernel_config_fwd.use_tma_store,
            fuse_mul_post=fuse_mul_post,
            is_first_gemm=is_first_gemm,
        )
        if kernel_config_bwd_dW is not None and not dX_only:
            check_valid_config_bwd_dW(
                permute_x,
                permute_y,
                use_tma_load_dY=kernel_config_bwd_dW.use_tma_load_dy,
                use_tma_load_x=kernel_config_bwd_dW.use_tma_load_x,
                use_tma_store=kernel_config_bwd_dW.use_tma_store,
                fuse_mul_post=fuse_mul_post,
                is_first_gemm=is_first_gemm,
            )
        if kernel_config_bwd_dX is not None and not dW_only:
            check_valid_config_bwd_dX(
                permute_x,
                permute_y,
                use_tma_load_dY=kernel_config_bwd_dX.use_tma_load_dy,
                use_tma_load_w=kernel_config_bwd_dX.use_tma_load_w,
                use_tma_store=kernel_config_bwd_dX.use_tma_store,
                fuse_mul_post=fuse_mul_post,
                is_first_gemm=is_first_gemm,
            )

    if permute_x or permute_y:
        assert gather_indices is not None, (
            "gather_indices is required when either permute_x or permute_y is True"
        )

    if fuse_mul_post:
        assert topk_weights is not None, (
            "topk_weights is required when fuse_mul_post is True"
        )

    X = X.view(-1, X.shape[-1])
    m_sizes = m_sizes.view(-1)
    gather_indices = gather_indices.view(-1)

    return GroupedGemm.apply(
        X,
        W,
        m_sizes,
        topk,
        gather_indices,
        permute_x,
        permute_y,
        topk_weights,
        fuse_mul_post,
        kernel_config_fwd,
        kernel_config_bwd_dX,
        kernel_config_bwd_dW,
        autotune,
        dX_only,
        dW_only,
    )

__name__ = _prev_name
_module_exports = {k: globals()[k] for k in set(globals()) - _module_before}
_register_module(_module_name, _module_exports)
del _module_exports
del _module_before
del _module_name
del _prev_name

# ===== Module: grouped_gemm.reference.moe_ops =====
_prev_name = __name__
_module_name = 'grouped_gemm.reference.moe_ops'
__name__ = _module_name
_ensure_package(_module_name.rpartition('.')[0])
_module_before = set(globals())

# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

import torch
import torch.nn.functional as F


def permute(X: torch.Tensor, gather_indices: torch.Tensor, topk: int):
    """
    Scatters X to a new tensor with shape [total_tokens, hidden_dim] where total_tokens is num_tokens * topk,
    permuting the tokens according to sorted_token_idx.

    Helper for grouped gemm where hidden states need be ordered by expert.
    X: [num_tokens, hidden_dim]
    sorted_token_idx: [num_tokens * topk]
    topk: int

    Returns:
        [total_tokens, hidden_dim]
    """
    assert gather_indices.ndim == 1
    X = X.view(-1, X.shape[-1])
    # Shortcut for topk == 1
    if topk == 1:
        return X[gather_indices]

    return X[gather_indices // topk]


def unpermute(X: torch.Tensor, gather_indices: torch.Tensor):
    X = X.view(-1, X.shape[-1]) if X.ndim > 2 else X
    unpermuted = torch.empty_like(X)
    unpermuted.index_copy_(0, gather_indices, X)
    return unpermuted.view_as(X)


def calculate_topk(
    gating_output: torch.Tensor,
    top_k: int,
    use_sigmoid: bool,
    renormalize: bool,
    pre_act: bool = True,
    post_act: bool = False,
):
    """
    If post_act is True, then activation function is run AFTER topk
    If post_act is False, then activation function is run BEFORE topk

    This is to align with triton_bench implementation (post_act) whereas most models use pre_act (e.g. llama4, deepseek)
    """
    assert pre_act ^ post_act, "only one of pre_act or post_act can be True"

    def _activation(gating_output: torch.Tensor):
        if use_sigmoid:
            scores = torch.sigmoid(gating_output.to(torch.float32)).to(gating_output.dtype)
        else:
            scores = F.softmax(gating_output.to(torch.float32), dim=1).to(gating_output.dtype)

        return scores

    if pre_act:
        scores = _activation(gating_output)
    else:
        scores = gating_output

    topk_weights, topk_ids = torch.topk(scores, k=top_k, dim=1)

    if post_act:
        topk_weights = _activation(topk_weights)

    if renormalize:
        topk_weights /= torch.sum(topk_weights, dim=-1, keepdim=True).to(gating_output.dtype)

    return topk_weights, topk_ids


@torch.no_grad()
def get_routing_indices(selected_experts, num_experts, return_scatter_indices: bool = False):
    """
    Returns:
        token_counts_by_expert: [num_experts]
        gather_indices: [num_tokens]
        scatter_indices [Optional] (torch.Tensor):
            Indices for unpermuting gathered inputs back to token order, shape ``(bs * seqlen * top_k,)``.
    """
    # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
    token_counts_by_expert = torch.histc(
        selected_experts.view(-1),
        bins=num_experts,
        min=0,
        max=num_experts,
    )
    # token_indices_experts_sorted shape (bs*slen*top_k,)
    gather_indices = torch.argsort(selected_experts.view(-1), stable=True)
    if return_scatter_indices:
        scatter_indices = gather_indices.argsort()
        return token_counts_by_expert, gather_indices, scatter_indices
    else:
        return token_counts_by_expert, gather_indices


def torch_grouped_gemm(X, W, m_sizes, transpose=True):
    """
    X: [M, K] if forward, else [M, N]
    W: [E, N, K]
    m_sizes: [E]

    Returns:
        Y: [M, N] if forward, else [M, K]
    """
    X = X.view(-1, X.shape[-1])
    M, K = X.shape

    assert m_sizes.ndim == 1
    E = m_sizes.shape[0]

    assert W.ndim == 3
    assert W.shape[0] == E

    N = W.shape[1]

    result = torch.zeros((M, N), dtype=X.dtype, device=X.device)

    m_start = 0
    for g in range(E):
        m_size = m_sizes[g]
        if m_size > 0:
            m_end = m_start + m_size

            # Extract group input
            # m_size x K
            X_g = X[m_start:m_end]
            # N x K
            W_g = W[g]

            # Y_g = X_g @ W_g.T -> [m_size, N]
            W_g = W_g.T if transpose else W_g
            Y_g = X_g @ W_g

            result[m_start:m_end] = Y_g

            m_start = m_end
    return result

__name__ = _prev_name
_module_exports = {k: globals()[k] for k in set(globals()) - _module_before}
_register_module(_module_name, _module_exports)
del _module_exports
del _module_before
del _module_name
del _prev_name

# ===== Module: grouped_gemm.reference.layers.llama4_moe =====
_prev_name = __name__
_module_name = 'grouped_gemm.reference.layers.llama4_moe'
__name__ = _module_name
_ensure_package(_module_name.rpartition('.')[0])
_module_before = set(globals())

# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from transformers.models.llama4 import Llama4TextConfig
from transformers.models.llama4.modeling_llama4 import Llama4TextMoe

from grouped_gemm.interface import grouped_gemm
from grouped_gemm.kernels.tuning import (
    KernelConfigBackward_dW,
    KernelConfigBackward_dX,
    KernelConfigForward,
)
from grouped_gemm.reference.moe_ops import (
    get_routing_indices,
    permute,
    torch_grouped_gemm,
    unpermute,
)

"""
Reference implementation of Llama4 MoE block using triton grouped gemm.

`Llama4GroupedGemmTextMoe` is the HF `Llama4TextMoe` block implemented with a torch-native grouped gemm.
`Llama4TritonTextMoe` is the HF `Llama4TextMoe` implemented with triton grouped gemm.
"""


@dataclass
class Llama4MoeResult:
    token_counts_by_expert: torch.Tensor
    gather_indices: torch.Tensor
    topk_weights: torch.Tensor
    hidden_states_after_weight_merge: torch.Tensor
    first_gemm: torch.Tensor
    intermediate: torch.Tensor
    second_gemm: torch.Tensor
    hidden_states_unpermute: torch.Tensor
    shared_expert_out: torch.Tensor
    final_out: torch.Tensor
    router_logits: torch.Tensor = None


class Llama4GroupedGemmTextMoe(Llama4TextMoe):
    EXPERT_WEIGHT_NAMES = ["experts.gate_up_proj", "experts.down_proj"]

    def __init__(
        self,
        config: Llama4TextConfig,
        overlap_router_shared=False,
        verbose=False,
        debug=False,
    ):
        super().__init__(config)
        self.overlap_router_shared = overlap_router_shared
        self.verbose = verbose
        self.debug = debug

        # Permute in-place expert weights
        E, K, N = self.num_experts, self.hidden_dim, self.experts.expert_dim
        assert self.experts.gate_up_proj.shape == torch.Size([E, K, 2 * N]), (
            f"{self.experts.gate_up_proj.shape} != {[E, K, 2 * N]}"
        )
        permuted_shape = [E, 2 * N, K]
        permuted_stride = [2 * N * K, K, 1]
        if verbose:
            print(
                f"Changing gate_up_proj from {self.experts.gate_up_proj.size()}:{self.experts.gate_up_proj.stride()} to {permuted_shape}:{permuted_stride}"
            )
        with torch.no_grad():
            self.experts.gate_up_proj.as_strided_(permuted_shape, permuted_stride)

        if verbose:
            print(
                f"{self.experts.gate_up_proj.shape}:{self.experts.gate_up_proj.stride()}"
            )

        assert self.experts.down_proj.shape == torch.Size([E, N, K]), (
            f"{self.experts.down_proj.shape} != {[E, N, K]}"
        )
        permuted_shape = [E, K, N]
        permuted_stride = [K * N, N, 1]
        if verbose:
            print(
                f"Changing down_proj from {self.experts.down_proj.size()}:{self.experts.down_proj.stride()} to {permuted_shape}:{permuted_stride}"
            )

        with torch.no_grad():
            self.experts.down_proj.as_strided_(permuted_shape, permuted_stride)

        if verbose:
            print(f"{self.experts.down_proj.shape}:{self.experts.down_proj.stride()}")

        if overlap_router_shared:
            self.shared_expert_stream = torch.cuda.Stream()
            self.default_event = torch.cuda.Event()
            self.shared_expert_end_event = torch.cuda.Event()

    @torch.no_grad
    def copy_weights(self, other: Llama4TextMoe):
        for name, param_to_copy in other.named_parameters():
            if self.verbose:
                print(f"Copying {name} with shape {param_to_copy.shape}")
            param = self.get_parameter(name)

            if any(n in name for n in self.EXPERT_WEIGHT_NAMES):
                param_to_copy = param_to_copy.permute(0, 2, 1)

            assert param.shape == param_to_copy.shape, (
                f"{param.shape} != {param_to_copy.shape}"
            )
            param.copy_(param_to_copy)

        return self

    def check_weights(self, other: Llama4TextMoe):
        for name, other_param in other.named_parameters():
            if any(n in name for n in self.EXPERT_WEIGHT_NAMES):
                other_param = other_param.permute(0, 2, 1)
            param = self.get_parameter(name)
            assert param.equal(other_param), f"Param {name} not equal!"
            assert param.is_contiguous(), f"{name} not contiguous!"

    def act_and_mul(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 2 * self.experts.expert_dim
        gate_proj = x[..., : self.experts.expert_dim]
        up_proj = x[..., self.experts.expert_dim :]
        return self.experts.act_fn(gate_proj) * up_proj

    def run_router(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # router_logits: (batch * sequence_length, n_experts)
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        router_logits = self.router(hidden_states)
        routing_weights, selected_experts = torch.topk(
            router_logits, self.top_k, dim=-1
        )

        routing_weights = F.sigmoid(routing_weights.float()).to(hidden_states.dtype)

        return router_logits, routing_weights, selected_experts

    def get_token_counts_and_gather_indices(
        self, selected_experts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        token_counts_by_expert, gather_indices = get_routing_indices(
            selected_experts, self.num_experts
        )
        assert not token_counts_by_expert.requires_grad
        assert not gather_indices.requires_grad
        return token_counts_by_expert, gather_indices

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        num_tokens = batch_size * sequence_length
        total_tokens = num_tokens * self.top_k
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.overlap_router_shared:
            # Marker for all prior ops on default stream
            self.default_event.record()

        router_logits, routing_weights, selected_experts = self.run_router(
            hidden_states
        )
        assert routing_weights.shape == (num_tokens, self.top_k), (
            f"{routing_weights.shape} != {(num_tokens, self.top_k)}"
        )

        if self.overlap_router_shared:
            with torch.cuda.stream(self.shared_expert_stream):
                # Ensure prior kernels on default stream complete
                self.default_event.wait()

                shared_expert_out = self.shared_expert(hidden_states)
                # Ensure hidden states remains valid on this stream
                hidden_states.record_stream(self.shared_expert_stream)

                self.shared_expert_end_event.record()

            # Ensure shared expert still valid on default stream
            shared_expert_out.record_stream(torch.cuda.current_stream())
            self.shared_expert_end_event.wait()
        else:
            shared_expert_out = self.shared_expert(hidden_states)

        hidden_states = (
            hidden_states.view(num_tokens, self.top_k, hidden_dim)
            * routing_weights[..., None]
        )

        if self.top_k > 1:
            hidden_states = hidden_states.sum(dim=1)
        hidden_states_after_weight_merge = hidden_states.view(-1, hidden_dim)

        # 1. Compute tokens per expert and indices for gathering tokes from token order to expert order
        # NOTE: these are auxiliary data structs which don't need to be recorded in autograd graph
        token_counts_by_expert, gather_indices = (
            self.get_token_counts_and_gather_indices(selected_experts)
        )

        # 2. Permute tokens from token order to expert order
        hidden_states = permute(
            hidden_states_after_weight_merge, gather_indices, self.top_k
        )
        assert hidden_states.shape == (total_tokens, hidden_dim)

        # Start expert computation
        first_gemm = torch_grouped_gemm(
            X=hidden_states, W=self.experts.gate_up_proj, m_sizes=token_counts_by_expert
        )
        assert first_gemm.shape == (total_tokens, 2 * self.experts.expert_dim)

        intermediate = self.act_and_mul(first_gemm)
        assert intermediate.shape == (total_tokens, self.experts.expert_dim)

        # See comment above
        second_gemm = torch_grouped_gemm(
            X=intermediate, W=self.experts.down_proj, m_sizes=token_counts_by_expert
        )
        assert second_gemm.shape == (total_tokens, hidden_dim)

        # Post-processing
        hidden_states_unpermute = unpermute(second_gemm, gather_indices)
        assert hidden_states_unpermute.shape == (total_tokens, hidden_dim)
        # grouped_gemm_out = hidden_states.view(batch_size, sequence_length, hidden_dim)

        final_out = hidden_states_unpermute + shared_expert_out

        result = (
            Llama4MoeResult(
                token_counts_by_expert=token_counts_by_expert,
                gather_indices=gather_indices,
                topk_weights=routing_weights,
                hidden_states_after_weight_merge=hidden_states_after_weight_merge,
                first_gemm=first_gemm,
                intermediate=intermediate,
                second_gemm=second_gemm,
                hidden_states_unpermute=hidden_states_unpermute,
                shared_expert_out=shared_expert_out,
                final_out=final_out,
                router_logits=router_logits,
            )
            if self.debug
            else (final_out, routing_weights)
        )

        return result


class Llama4TritonTextMoe(Llama4GroupedGemmTextMoe):
    def __init__(
        self,
        config: Llama4TextConfig,
        overlap_router_shared=False,
        permute_x: bool = False,
        permute_y: bool = True,
        autotune: bool = True,
        kernel_config_fwd: KernelConfigForward = None,
        kernel_config_bwd_dW: KernelConfigBackward_dW = None,
        kernel_config_bwd_dX: KernelConfigBackward_dX = None,
        dW_only: bool = False,
        dX_only: bool = False,
        verbose=False,
    ):
        super().__init__(config, overlap_router_shared=overlap_router_shared)
        assert not permute_x, (
            "Llama4 triton grouped gemm does not support permute x due to pre-multiplication of router weights"
        )
        self.permute_x = permute_x
        self.permute_y = permute_y
        self.autotune = autotune
        if not autotune:
            assert (
                kernel_config_fwd is not None
                and kernel_config_bwd_dW is not None
                and kernel_config_bwd_dX is not None
            ), "Kernel configs must be provided if autotune is False"
        self.kernel_config_fwd = kernel_config_fwd
        self.kernel_config_bwd_dW = kernel_config_bwd_dW
        self.kernel_config_bwd_dX = kernel_config_bwd_dX
        self.dW_only = dW_only
        self.dX_only = dX_only

    @torch.no_grad
    def copy_weights(self, other: Llama4TextMoe):
        for name, param_to_copy in other.named_parameters():
            if self.verbose:
                print(f"Copying {name} with shape {param_to_copy.shape}")
            param = self.get_parameter(name)

            if any(n in name for n in self.EXPERT_WEIGHT_NAMES):
                param_to_copy = param_to_copy.permute(0, 2, 1)

            assert param.shape == param_to_copy.shape, (
                f"{param.shape} != {param_to_copy.shape}"
            )
            param.copy_(param_to_copy)

        return self

    def check_weights(self, other: Llama4TextMoe):
        for name, other_param in other.named_parameters():
            if any(n in name for n in self.EXPERT_WEIGHT_NAMES):
                other_param = other_param.permute(0, 2, 1)
            param = self.get_parameter(name)
            assert param.equal(other_param), f"Param {name} not equal!"
            assert param.is_contiguous(), f"{name} not contiguous!"

    def act_and_mul(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 2 * self.experts.expert_dim
        gate_proj = x[..., : self.experts.expert_dim]
        up_proj = x[..., self.experts.expert_dim :]
        return self.experts.act_fn(gate_proj) * up_proj

    def run_router(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # router_logits: (batch * sequence_length, n_experts)
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        router_logits = self.router(hidden_states)
        routing_weights, selected_experts = torch.topk(
            router_logits, self.top_k, dim=-1
        )

        routing_weights = F.sigmoid(routing_weights.float()).to(hidden_states.dtype)

        return router_logits, routing_weights, selected_experts

    def get_token_counts_and_gather_indices(
        self, selected_experts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        token_counts_by_expert, gather_indices = get_routing_indices(
            selected_experts, self.num_experts
        )
        assert not token_counts_by_expert.requires_grad
        assert not gather_indices.requires_grad
        return token_counts_by_expert, gather_indices

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        num_tokens = batch_size * sequence_length
        total_tokens = num_tokens * self.top_k
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.overlap_router_shared:
            # Marker for all prior ops on default stream
            self.default_event.record()

        router_logits, routing_weights, selected_experts = self.run_router(
            hidden_states
        )
        assert routing_weights.shape == (num_tokens, self.top_k), (
            f"{routing_weights.shape} != {(num_tokens, self.top_k)}"
        )

        if self.overlap_router_shared:
            with torch.cuda.stream(self.shared_expert_stream):
                # Ensure prior kernels on default stream complete
                self.default_event.wait()

                shared_expert_out = self.shared_expert(hidden_states)
                # Ensure hidden states remains valid on this stream
                hidden_states.record_stream(self.shared_expert_stream)

                self.shared_expert_end_event.record()

            # Ensure shared expert still valid on default stream
            shared_expert_out.record_stream(torch.cuda.current_stream())
            self.shared_expert_end_event.wait()
        else:
            shared_expert_out = self.shared_expert(hidden_states)

        hidden_states = (
            hidden_states.view(num_tokens, self.top_k, hidden_dim)
            * routing_weights[..., None]
        )

        if self.top_k > 1:
            hidden_states = hidden_states.sum(dim=1)
        hidden_states = hidden_states.view(-1, hidden_dim)

        # 1. Compute tokens per expert and indices for gathering tokes from token order to expert order
        # NOTE: these are auxiliary data structs which don't need to be recorded in autograd graph
        token_counts_by_expert, gather_indices = (
            self.get_token_counts_and_gather_indices(selected_experts)
        )

        # 2. Permute tokens from token order to expert order
        hidden_states = permute(hidden_states, gather_indices, self.top_k)
        assert hidden_states.shape == (total_tokens, hidden_dim)

        # Start expert computation
        hidden_states = grouped_gemm(
            X=hidden_states,
            W=self.experts.gate_up_proj,
            m_sizes=token_counts_by_expert,
            gather_indices=gather_indices,
            topk=self.top_k,
            permute_x=self.permute_x,
            permute_y=False,  # output of first grouped gemm should never be permuted
            autotune=self.autotune,
            kernel_config_fwd=self.kernel_config_fwd,
            kernel_config_bwd_dW=self.kernel_config_bwd_dW,
            kernel_config_bwd_dX=self.kernel_config_bwd_dX,
            is_first_gemm=True,
            dW_only=self.dW_only,
            dX_only=self.dX_only,
        )
        hidden_states = self.act_and_mul(hidden_states)
        hidden_states = grouped_gemm(
            X=hidden_states,
            W=self.experts.down_proj,
            m_sizes=token_counts_by_expert,
            gather_indices=gather_indices,
            topk=self.top_k,
            permute_x=False,
            permute_y=self.permute_y,
            autotune=self.autotune,
            kernel_config_fwd=self.kernel_config_fwd,
            kernel_config_bwd_dW=self.kernel_config_bwd_dW,
            kernel_config_bwd_dX=self.kernel_config_bwd_dX,
            is_first_gemm=False,
            dW_only=self.dW_only,
            dX_only=self.dX_only,
        )

        # Post-processing
        # 1. Unpermute from expert order to token order
        if not self.permute_y:
            hidden_states = unpermute(hidden_states, gather_indices)
        hidden_states += shared_expert_out

        return hidden_states, routing_weights

__name__ = _prev_name
_module_exports = {k: globals()[k] for k in set(globals()) - _module_before}
_register_module(_module_name, _module_exports)
del _module_exports
del _module_before
del _module_name
del _prev_name

# ===== Module: grouped_gemm.reference.layers.qwen3_moe =====
_prev_name = __name__
_module_name = 'grouped_gemm.reference.layers.qwen3_moe'
__name__ = _module_name
_ensure_package(_module_name.rpartition('.')[0])
_module_before = set(globals())

# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    ACT2FN,
    Qwen3MoeSparseMoeBlock,
)

from grouped_gemm.interface import grouped_gemm
from grouped_gemm.kernels.tuning import (
    KernelConfigBackward_dW,
    KernelConfigBackward_dX,
    KernelConfigForward,
)
from grouped_gemm.reference.moe_ops import (
    get_routing_indices,
    permute,
    torch_grouped_gemm,
    unpermute,
)

"""
Reference implementation of HF Qwen3 MoE block using grouped gemm.

The Qwen3MoeGroupedGEMMBlock is a reference torch-native implementation.
Qwen3MoeFusedGroupedGEMMBlock is a version using the triton grouped gemm kernel.

NOTE: This is NOT to be used for production as it contains many extra checks and saves all intermediate results for debugging.
"""


@dataclass
class GroupedGEMMResult:
    token_counts_by_expert: torch.Tensor
    gather_indices: torch.Tensor
    topk_weights: torch.Tensor
    first_gemm: torch.Tensor
    intermediate: torch.Tensor
    second_gemm: torch.Tensor
    hidden_states_unpermute: torch.Tensor
    hidden_states: torch.Tensor  # final output


class Qwen3MoeGroupedGEMMBlock(torch.nn.Module):
    def __init__(
        self,
        config,
        gate: torch.Tensor,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size

        assert gate.shape == (config.num_experts, config.hidden_size)
        assert gate_up_proj.shape == (
            config.num_experts,
            2 * config.moe_intermediate_size,
            config.hidden_size,
        )
        assert down_proj.shape == (
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
        )

        # gating
        self.gate = torch.nn.Parameter(gate)

        # experts
        self.gate_up_proj = torch.nn.Parameter(gate_up_proj, requires_grad=True)
        self.down_proj = torch.nn.Parameter(down_proj, requires_grad=True)
        self.act_fn = ACT2FN[config.hidden_act]

    @staticmethod
    def extract_hf_weights(moe_block: Qwen3MoeSparseMoeBlock):
        config: Qwen3MoeConfig = moe_block.experts[0].config
        num_experts = config.num_experts

        gate = moe_block.gate.weight.data
        gate_proj = torch.stack(
            [moe_block.experts[i].gate_proj.weight.data for i in range(num_experts)],
            dim=0,
        )
        up_proj = torch.stack(
            [moe_block.experts[i].up_proj.weight.data for i in range(num_experts)],
            dim=0,
        )
        down_proj = torch.stack(
            [moe_block.experts[i].down_proj.weight.data for i in range(num_experts)],
            dim=0,
        )
        gate_up_proj = torch.cat([gate_proj, up_proj], dim=1)
        return gate, gate_up_proj, down_proj

    @classmethod
    def from_hf(cls, moe_block: Qwen3MoeSparseMoeBlock):
        config: Qwen3MoeConfig = moe_block.experts[0].config
        gate, gate_up_proj, down_proj = cls.extract_hf_weights(moe_block)
        return cls(config, gate, gate_up_proj, down_proj)

    def check_weights(self, moe_block: Qwen3MoeSparseMoeBlock):
        for i in range(self.num_experts):
            assert self.gate_up_proj[i].equal(
                torch.cat(
                    [
                        moe_block.experts[i].gate_proj.weight.data,
                        moe_block.experts[i].up_proj.weight.data,
                    ],
                    dim=0,
                )
            )
            assert self.down_proj[i].equal(moe_block.experts[i].down_proj.weight.data)

    def act_and_mul(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 2 * self.moe_intermediate_size
        gate_proj = x[..., : self.moe_intermediate_size]
        up_proj = x[..., self.moe_intermediate_size :]
        return self.act_fn(gate_proj) * up_proj

    def run_router(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = torch.nn.functional.linear(hidden_states, self.gate)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        return router_logits, routing_weights, selected_experts

    def get_token_counts_and_gather_indices(
        self, selected_experts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        token_counts_by_expert, gather_indices = get_routing_indices(
            selected_experts, self.num_experts
        )
        assert not token_counts_by_expert.requires_grad
        assert not gather_indices.requires_grad
        return token_counts_by_expert, gather_indices

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        num_tokens = batch_size * sequence_length
        total_tokens = num_tokens * self.top_k

        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits, routing_weights, selected_experts = self.run_router(
            hidden_states
        )

        # 1. Compute tokens per expert and indices for gathering tokes from token order to expert order
        # NOTE: these are auxiliary data structs which don't need to be recorded in autograd graph
        token_counts_by_expert, gather_indices = (
            self.get_token_counts_and_gather_indices(selected_experts)
        )

        # 2. Permute tokens from token order to expert order
        hidden_states = permute(hidden_states, gather_indices, self.top_k)
        assert hidden_states.shape == (total_tokens, hidden_dim)

        # Start expert computation
        first_gemm = torch_grouped_gemm(
            X=hidden_states, W=self.gate_up_proj, m_sizes=token_counts_by_expert
        )
        assert first_gemm.shape == (total_tokens, 2 * self.moe_intermediate_size)
        intermediate = self.act_and_mul(first_gemm)
        assert intermediate.shape == (total_tokens, self.moe_intermediate_size)
        second_gemm = torch_grouped_gemm(
            X=intermediate, W=self.down_proj, m_sizes=token_counts_by_expert
        )
        assert second_gemm.shape == (total_tokens, hidden_dim)

        # Post-processing
        # 1. Unpermute from expert order to token order
        hidden_states_unpermute = unpermute(second_gemm, gather_indices)
        assert hidden_states_unpermute.shape == (total_tokens, hidden_dim)

        # 2. Merge topk weights
        hidden_states = (
            hidden_states_unpermute.view(num_tokens, self.top_k, hidden_dim)
            * routing_weights[..., None]
        )
        hidden_states = hidden_states.sum(dim=1)
        assert hidden_states.shape == (num_tokens, hidden_dim)

        hidden_states = hidden_states.view(batch_size, sequence_length, hidden_dim)
        return GroupedGEMMResult(
            token_counts_by_expert=token_counts_by_expert,
            gather_indices=gather_indices,
            topk_weights=routing_weights,
            first_gemm=first_gemm,
            intermediate=intermediate,
            second_gemm=second_gemm,
            hidden_states_unpermute=hidden_states_unpermute,
            hidden_states=hidden_states,
        ), router_logits


class Qwen3MoeFusedGroupedGEMMBlock(Qwen3MoeGroupedGEMMBlock):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        gate: torch.Tensor,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        permute_x: bool = True,
        permute_y: bool = True,
        autotune: bool = True,
        kernel_config_fwd: KernelConfigForward = None,
        kernel_config_bwd_dW: KernelConfigBackward_dW = None,
        kernel_config_bwd_dX: KernelConfigBackward_dX = None,
        dW_only: bool = False,
        dX_only: bool = False,
    ):
        super().__init__(config, gate, gate_up_proj, down_proj)
        self.permute_x = permute_x
        self.permute_y = permute_y
        self.autotune = autotune
        if not autotune:
            assert (
                kernel_config_fwd is not None
                and kernel_config_bwd_dW is not None
                and kernel_config_bwd_dX is not None
            ), "Kernel configs must be provided if autotune is False"
        self.kernel_config_fwd = kernel_config_fwd
        self.kernel_config_bwd_dW = kernel_config_bwd_dW
        self.kernel_config_bwd_dX = kernel_config_bwd_dX
        self.dW_only = dW_only
        self.dX_only = dX_only

    @classmethod
    def from_hf(
        cls,
        moe_block: Qwen3MoeSparseMoeBlock,
        permute_x: bool = True,
        permute_y: bool = True,
        autotune: bool = True,
        kernel_config_fwd: KernelConfigForward = None,
        kernel_config_bwd_dW: KernelConfigBackward_dW = None,
        kernel_config_bwd_dX: KernelConfigBackward_dX = None,
        dW_only: bool = False,
        dX_only: bool = False,
    ):
        config: Qwen3MoeConfig = moe_block.experts[0].config
        gate, gate_up_proj, down_proj = Qwen3MoeGroupedGEMMBlock.extract_hf_weights(
            moe_block
        )
        return cls(
            config,
            gate,
            gate_up_proj,
            down_proj,
            permute_x=permute_x,
            permute_y=permute_y,
            autotune=autotune,
            kernel_config_fwd=kernel_config_fwd,
            kernel_config_bwd_dW=kernel_config_bwd_dW,
            kernel_config_bwd_dX=kernel_config_bwd_dX,
            dW_only=dW_only,
            dX_only=dX_only,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        num_tokens = batch_size * sequence_length
        total_tokens = num_tokens * self.top_k

        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits, routing_weights, selected_experts = self.run_router(
            hidden_states
        )
        # Pre-processing
        # 1. Compute tokens per expert and indices for gathering tokes from token order to expert order
        # NOTE: these are auxiliary data structs which don't need to be recorded in autograd graph
        token_counts_by_expert, gather_indices = (
            self.get_token_counts_and_gather_indices(selected_experts)
        )

        # 2. permute_x -> permutation will be fused in prologue of first grouped gemm
        if not self.permute_x:
            hidden_states = permute(hidden_states, gather_indices, self.top_k)
        # Start expert computation
        hidden_states = grouped_gemm(
            X=hidden_states,
            W=self.gate_up_proj,
            m_sizes=token_counts_by_expert,
            gather_indices=gather_indices,
            topk=self.top_k,
            permute_x=self.permute_x,
            permute_y=False,  # output of first grouped gemm should never be permuted
            autotune=self.autotune,
            kernel_config_fwd=self.kernel_config_fwd,
            kernel_config_bwd_dW=self.kernel_config_bwd_dW,
            kernel_config_bwd_dX=self.kernel_config_bwd_dX,
            is_first_gemm=True,
            dW_only=self.dW_only,
            dX_only=self.dX_only,
        )
        hidden_states = self.act_and_mul(hidden_states)
        hidden_states = grouped_gemm(
            X=hidden_states,
            W=self.down_proj,
            m_sizes=token_counts_by_expert,
            gather_indices=gather_indices,
            topk=self.top_k,
            permute_x=False,
            permute_y=self.permute_y,
            autotune=self.autotune,
            kernel_config_fwd=self.kernel_config_fwd,
            kernel_config_bwd_dW=self.kernel_config_bwd_dW,
            kernel_config_bwd_dX=self.kernel_config_bwd_dX,
            is_first_gemm=False,
            dW_only=self.dW_only,
            dX_only=self.dX_only,
        )

        # Post-processing
        # 1. Unpermute from expert order to token order
        if not self.permute_y:
            hidden_states = unpermute(hidden_states, gather_indices)

        # 2. Merge topk weights
        hidden_states = (
            hidden_states.view(num_tokens, self.top_k, hidden_dim)
            * routing_weights[..., None]
        )
        hidden_states = hidden_states.sum(dim=1)

        hidden_states = hidden_states.view(batch_size, sequence_length, hidden_dim)
        return hidden_states, router_logits

__name__ = _prev_name
_module_exports = {k: globals()[k] for k in set(globals()) - _module_before}
_register_module(_module_name, _module_exports)
del _module_exports
del _module_before
del _module_name
del _prev_name

# ===== Module: utils =====
_prev_name = __name__
_module_name = 'utils'
__name__ = _module_name
_ensure_package(_module_name.rpartition('.')[0])
_module_before = set(globals())

import argparse
import datetime
import json
import logging
import math
import os
from itertools import product

import pandas as pd
import torch

from grouped_gemm.kernels.tuning import (
    KernelConfigBackward_dW,
    KernelConfigBackward_dX,
    KernelConfigForward,
    KernelResult,
)

SEED = 42


def create_merged_results(
    df: pd.DataFrame, mode: str, seqlen: int, dtype: torch.dtype, autotune: bool
):
    kernel_result_cols = df.columns.to_list()
    test_config_dict = {
        "mode": mode,
        "seqlen": seqlen,
        "dtype": dtype,
        "autotune": autotune,
    }
    test_config_cols = list(test_config_dict.keys())
    for col in test_config_cols:
        df[col] = test_config_dict[col]
    # Reorder columns so that test config cols are first
    df = df[test_config_cols + kernel_result_cols]
    return df


def post_process_results(
    results: list[KernelResult],
    mode: str,
    seqlen: int,
    dtype: torch.dtype,
    autotune: bool,
):
    df = KernelResult.to_dataframe(results, sort_by="speedup")
    df = create_merged_results(df, mode, seqlen, dtype, autotune)
    return df


def save_results(
    df: pd.DataFrame,
    results_dir: str,
    mode: str,
    seqlen: int,
    dtype: torch.dtype,
    autotune: bool,
):
    dt = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = f"{results_dir}/{mode}"
    save_path = f"{save_dir}/{dt}_{seqlen}_{str(dtype).split('.')[-1]}.csv"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"Saving results to {save_path}")
    df.to_csv(save_path, index=False)


def create_kernel_configs(args: argparse.Namespace, permute_x: bool, permute_y: bool):
    block_m_range = power_of_two_range(args.BLOCK_SIZE_M[0], args.BLOCK_SIZE_M[1])
    block_n_range = power_of_two_range(args.BLOCK_SIZE_N[0], args.BLOCK_SIZE_N[1])
    block_k_range = power_of_two_range(args.BLOCK_SIZE_K[0], args.BLOCK_SIZE_K[1])
    num_warps_range = multiples_of_range(args.num_warps[0], args.num_warps[1], step=2)
    num_stages_range = multiples_of_range(
        args.num_stages[0], args.num_stages[1], step=1
    )

    mode = args.mode
    kernel_configs = []
    for (
        block_m,
        block_n,
        block_k,
        num_warps,
        num_stages,
        tma_load_a,
        tma_load_b,
    ) in product(
        block_m_range,
        block_n_range,
        block_k_range,
        num_warps_range,
        num_stages_range,
        [True, False],
        [True, False],
    ):
        if mode == "forward":
            kernel_config = KernelConfigForward(
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_N=block_n,
                BLOCK_SIZE_K=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
                use_tma_load_w=tma_load_a,
                use_tma_load_x=tma_load_b,
                permute_x=permute_x,
                permute_y=permute_y,
            )
        elif mode == "dW":
            kernel_config = KernelConfigBackward_dW(
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_N=block_n,
                BLOCK_SIZE_K=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
                use_tma_load_dy=tma_load_a,
                use_tma_load_x=tma_load_b,
                permute_x=permute_x,
                permute_y=permute_y,
            )
        elif mode == "dX":
            kernel_config = KernelConfigBackward_dX(
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_N=block_n,
                BLOCK_SIZE_K=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
                use_tma_load_dy=tma_load_a,
                use_tma_load_w=tma_load_b,
                permute_x=permute_x,
                permute_y=permute_y,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")
        kernel_configs.append(kernel_config)

    logging.info(f"Pruning {len(kernel_configs)} kernel configs")

    pruned_configs = []
    for config in kernel_configs:
        if mode == "forward":
            if permute_x and config.use_tma_load_x:
                continue
        elif mode == "dW":
            if permute_x and config.use_tma_load_x:
                continue
            if permute_y and config.use_tma_load_dy:
                continue
        elif mode == "dX":
            if permute_y and config.use_tma_load_dy:
                continue
        pruned_configs.append(config)
    logging.info(f"After pruning, {len(pruned_configs)} kernel configs")

    return pruned_configs


def power_of_two_range(start, end):
    start = math.log2(start)
    end = math.log2(end)
    return [2**i for i in range(int(start), int(end) + 1)]


def multiples_of_range(start, end, step=1):
    return list(range(start, end + step, step))


def map_key_to_args(key, mode):
    pass


def save_autotune_results(autotune_cache, mode, ref_time, fused_time, results_dir):
    device_name = torch.cuda.get_device_name().replace(" ", "_")
    dt = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = f"{results_dir}/{mode}/autotune/{dt}/{device_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for key, config in autotune_cache.items():
        key = [
            str(k) if not "torch" in str(k) else str(k.split("torch.")[-1]) for k in key
        ]
        filename = "_".join(key)
        save_path = f"{save_dir}/{filename}.json"
        print(f"Saving autotune results to {save_path}")
        with open(save_path, "w") as f:
            result = {
                **config.all_kwargs(),
                "ref_time": ref_time,
                "fused_time": fused_time,
            }
            json.dump(result, f)


def get_autotuner(mode):
    if mode == "forward":
        from grouped_gemm.kernels.forward import _autotuned_grouped_gemm_forward_kernel

        return _autotuned_grouped_gemm_forward_kernel
    elif mode == "dW":
        from grouped_gemm.kernels.backward import _autotuned_grouped_gemm_dW_kernel

        return _autotuned_grouped_gemm_dW_kernel
    elif mode == "dX":
        from grouped_gemm.kernels.backward import _autotuned_grouped_gemm_dX_kernel

        return _autotuned_grouped_gemm_dX_kernel
    elif mode == "backward":
        from grouped_gemm.kernels.backward import (
            _autotuned_grouped_gemm_dW_kernel,
            _autotuned_grouped_gemm_dX_kernel,
        )

        return _autotuned_grouped_gemm_dW_kernel, _autotuned_grouped_gemm_dX_kernel
    else:
        raise ValueError(f"Invalid mode: {mode}")


def postprocess_autotune_results(autotuner, mode, ref_time, fused_time, results_dir):
    for key, value in autotuner.cache.items():
        print(f"{mode} {key}: {value.all_kwargs()}")
    save_autotune_results(
        autotuner.cache,
        mode=mode,
        ref_time=ref_time,
        fused_time=fused_time,
        results_dir=results_dir,
    )

__name__ = _prev_name
_module_exports = {k: globals()[k] for k in set(globals()) - _module_before}
_register_module(_module_name, _module_exports)
del _module_exports
del _module_before
del _module_name
del _prev_name

# ===== Module: benchmark.benchmark_fused_moe =====
import argparse
import time
from contextlib import nullcontext

import torch
from transformers import AutoConfig
from transformers.models.llama4 import Llama4TextConfig
from transformers.models.llama4.modeling_llama4 import Llama4TextMoe
from transformers.models.qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
from triton.testing import do_bench
from utils import (
    create_kernel_configs,
    get_autotuner,
    post_process_results,
    postprocess_autotune_results,
    save_results,
)

from grouped_gemm.kernels.autotuning import (
    DEFAULT_K_BLOCK_SIZES,
    DEFAULT_M_BLOCK_SIZES,
    DEFAULT_N_BLOCK_SIZES,
    DEFAULT_NUM_STAGES,
    DEFAULT_NUM_WARPS,
)
from grouped_gemm.kernels.tuning import (
    KernelConfigBackward_dW,
    KernelConfigBackward_dX,
    KernelConfigForward,
    KernelResult,
    TritonTuningContext,
)
from grouped_gemm.reference.layers.llama4_moe import Llama4TritonTextMoe
from grouped_gemm.reference.layers.qwen3_moe import Qwen3MoeFusedGroupedGEMMBlock

SEED = 42
LLAMA4_ID = "meta-llama/Llama-4-Scout-17B-16E"
QWEN3_MODEL_ID = "Qwen/Qwen3-30B-A3B"


def run_benchmark_forward(
    ref_model: torch.nn.Module,
    tt_model: torch.nn.Module,
    config: AutoConfig,
    seqlen: int,
    dtype: torch.dtype,
    autotune: bool,
    kernel_config_fwd: KernelConfigForward = None,
    bs: int = 1,
):
    torch.manual_seed(
        SEED
    )  # Should not be needed when running using pytest -- autouse fixture in conftest.py
    device = "cuda"
    hidden_size = config.hidden_size

    X = torch.randn(
        bs, seqlen, hidden_size, dtype=dtype, device=device, requires_grad=True
    )

    # Forward
    bench_forward_ref = lambda: ref_model(X)  # noqa: E731
    bench_forward_fused = lambda: tt_model(X)  # noqa: E731

    ref_forward_time = do_bench(bench_forward_ref)

    if not autotune:
        assert kernel_config_fwd is not None
        tuning_context = TritonTuningContext(kernel_config_fwd)
    else:
        tuning_context = nullcontext()

    with tuning_context:
        fused_forward_time = do_bench(bench_forward_fused)

    if (not autotune) and (not tuning_context.success):
        return 0, 1

    print(
        f"Forward: ref {ref_forward_time:.4f}, fused {fused_forward_time:.4f}, speedup {ref_forward_time / fused_forward_time:.1f}x"
    )
    return ref_forward_time, fused_forward_time


def run_benchmark_backward(
    ref_model: torch.nn.Module,
    tt_model: torch.nn.Module,
    config: AutoConfig,
    seqlen: int,
    dtype: torch.dtype,
    bs=1,
):
    torch.manual_seed(
        SEED
    )  # Should not be needed when running using pytest -- autouse fixture in conftest.py
    device = "cuda"
    hidden_size = config.hidden_size

    X = torch.randn(
        bs, seqlen, hidden_size, dtype=dtype, device=device, requires_grad=True
    )
    X_test = X.detach().clone().requires_grad_(True)

    output, _ = ref_model(X)

    # Prevent autotuning forward pass
    from grouped_gemm.kernels.forward import _autotuned_grouped_gemm_forward_kernel

    _autotuned_grouped_gemm_forward_kernel.configs = (
        _autotuned_grouped_gemm_forward_kernel.configs[:20]
    )
    test_output, _ = tt_model(X_test)

    # Bench
    grad_output = torch.randn_like(output)
    bench_backward_ref = lambda: output.backward(grad_output, retain_graph=True)  # noqa: E731
    bench_backward_fused = lambda: test_output.backward(grad_output, retain_graph=True)  # noqa: E731

    ref_backward_time = do_bench(
        bench_backward_ref, grad_to_none=[X, *ref_model.parameters()]
    )
    fused_backward_time = do_bench(
        bench_backward_fused, grad_to_none=[X_test, *tt_model.parameters()]
    )
    print(
        f"Backward: ref {ref_backward_time:.4f}, fused {fused_backward_time:.4f}, speedup {ref_backward_time / fused_backward_time:.1f}x"
    )
    return ref_backward_time, fused_backward_time


def setup_model(
    config: Qwen3MoeConfig | Llama4TextConfig,
    dtype,
    permute_x,
    permute_y,
    autotune,
    kernel_config_fwd,
    kernel_config_bwd_dW,
    kernel_config_bwd_dX,
    dX_only=False,
    dW_only=False,
    overlap_router_shared=False,
    device="cuda",
):
    if isinstance(config, Qwen3MoeConfig):
        ref_model = Qwen3MoeSparseMoeBlock(config).to(device, dtype)

        # Triton kernel grouped gemm version of MoE Block -- this is what we're testing
        tt_model = Qwen3MoeFusedGroupedGEMMBlock.from_hf(
            ref_model,
            permute_x=permute_x,
            permute_y=permute_y,
            autotune=autotune,
            kernel_config_fwd=kernel_config_fwd,
            kernel_config_bwd_dW=kernel_config_bwd_dW,
            kernel_config_bwd_dX=kernel_config_bwd_dX,
            dX_only=dX_only,
            dW_only=dW_only,
        ).to(device, dtype)

    elif isinstance(config, Llama4TextConfig):
        ref_model = Llama4TextMoe(config).to(device, dtype)
        tt_model = Llama4TritonTextMoe(
            config,
            overlap_router_shared=overlap_router_shared,
            permute_x=permute_x,
            permute_y=permute_y,
            autotune=autotune,
            kernel_config_fwd=kernel_config_fwd,
            kernel_config_bwd_dW=kernel_config_bwd_dW,
            kernel_config_bwd_dX=kernel_config_bwd_dX,
            dX_only=dX_only,
            dW_only=dW_only,
        ).to(device, dtype)

    else:
        raise ValueError(f"Unrecognized config {type(config).__name__}")

    return ref_model, tt_model


def run_benchmark(
    mode: str,
    model_config: Qwen3MoeConfig | Llama4TextConfig,
    seqlen: int,
    dtype: torch.dtype,
    permute_x: bool,
    permute_y: bool,
    autotune: bool,
    kernel_config_fwd: KernelConfigForward = None,
    kernel_config_bwd_dW: KernelConfigBackward_dW = None,
    kernel_config_bwd_dX: KernelConfigBackward_dX = None,
    overlap_router_shared: bool = False,
    results_dir: str = None,
):
    if autotune:
        autotuner = get_autotuner(mode)
    if mode == "dW":
        dW_only = True
    elif mode == "dX":
        dX_only = True
    else:
        dW_only = dX_only = False

    ref_model, tt_model = setup_model(
        model_config,
        dtype=dtype,
        permute_x=permute_x,
        permute_y=permute_y,
        autotune=autotune,
        kernel_config_fwd=kernel_config_fwd,
        kernel_config_bwd_dW=kernel_config_bwd_dW,
        kernel_config_bwd_dX=kernel_config_bwd_dX,
        dX_only=dX_only,
        dW_only=dW_only,
        overlap_router_shared=overlap_router_shared,
    )

    if mode == "forward":
        ref_time, fused_time = run_benchmark_forward(
            ref_model,
            tt_model,
            config=model_config,
            seqlen=seqlen,
            dtype=dtype,
            autotune=autotune,
            kernel_config_fwd=kernel_config_fwd,
        )
    else:
        ref_time, fused_time = run_benchmark_backward(
            ref_model, tt_model, config=model_config, seqlen=seqlen, dtype=dtype
        )

    if autotune:
        if mode == "backward":
            autotuner_dW, autotuner_dX = autotuner
            postprocess_autotune_results(
                autotuner_dW, "dW", ref_time, fused_time, results_dir
            )
            postprocess_autotune_results(
                autotuner_dX, "dX", ref_time, fused_time, results_dir
            )
        else:
            postprocess_autotune_results(
                autotuner, mode, ref_time, fused_time, results_dir
            )

    return ref_time, fused_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="benchmark_results")
    parser.add_argument("--model", type=str, choices=["llama4", "qwen3"], required=True)
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument(
        "--dtype", type=str, choices=["bfloat16", "float16"], default="bfloat16"
    )
    parser.add_argument("--permute_x", action="store_true")
    parser.add_argument("--permute_y", action="store_true")
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--overlap_router_shared", action="store_true")
    parser.add_argument(
        "--BLOCK_SIZE_M",
        nargs=2,
        type=int,
        default=[DEFAULT_M_BLOCK_SIZES[0], DEFAULT_M_BLOCK_SIZES[-1]],
    )
    parser.add_argument(
        "--BLOCK_SIZE_N",
        nargs=2,
        type=int,
        default=[DEFAULT_N_BLOCK_SIZES[0], DEFAULT_N_BLOCK_SIZES[-1]],
    )
    parser.add_argument(
        "--BLOCK_SIZE_K",
        nargs=2,
        type=int,
        default=[DEFAULT_K_BLOCK_SIZES[0], DEFAULT_K_BLOCK_SIZES[-1]],
    )
    parser.add_argument(
        "--num_warps",
        nargs=2,
        type=int,
        default=[DEFAULT_NUM_WARPS[0], DEFAULT_NUM_WARPS[-1]],
    )
    parser.add_argument(
        "--num_stages",
        nargs=2,
        type=int,
        default=[DEFAULT_NUM_STAGES[0], DEFAULT_NUM_STAGES[-1]],
    )
    parser.add_argument(
        "--use_tma_load_w", action="store_true"
    )  # No need to specify, will automatically parametrize these for each kernel config
    parser.add_argument(
        "--use_tma_load_x", action="store_true"
    )  # No need to specify, will automatically parametrize these for each kernel config
    parser.add_argument(
        "--use_tma_load_dy", action="store_true"
    )  # No need to specify, will automatically parametrize these for each kernel config
    parser.add_argument(
        "--mode",
        type=str,
        choices=["forward", "backward", "dW", "dX"],
        default="forward",
    )
    args = parser.parse_args()
    args.dtype = getattr(torch, args.dtype)

    model_id = QWEN3_MODEL_ID if args.model == "qwen3" else LLAMA4_ID
    model_config = AutoConfig.from_pretrained(model_id)
    model_config = model_config.text_config if args.model == "llama4" else model_config

    mode = args.mode

    if args.autotune:
        # logging.basicConfig(level=logging.INFO)
        print(
            f"Benchmarking {model_id} {mode}: seqlen={args.seqlen}, dtype={args.dtype}, permute_x={args.permute_x}, permute_y={args.permute_y}, autotune"
        )
        start_time = time.time()
        ref_time, fused_time = run_benchmark(
            args.mode,
            model_config,
            seqlen=args.seqlen,
            dtype=args.dtype,
            permute_x=args.permute_x,
            permute_y=args.permute_y,
            autotune=args.autotune,
            overlap_router_shared=args.overlap_router_shared,
            results_dir=args.results_dir,
        )
        end_time = time.time()
        print(f"Total time: {end_time - start_time:.4f} seconds")

    # NOTE: better to use autotuner for now, since the MoE block needs 2 different kernel configs for forward (2 grouped gemms, gate_up_proj and down_proj)
    # and the backward pass needs 4 different kernel configs (2 grouped gemms each for dW and dX)
    # The benchmark only supports 1 kernel config at a time so the same config will be used for both grouped gemms, which is suboptimal.
    else:
        assert False, "Use autotune for now"
        kernel_configs = create_kernel_configs(args, args.permute_x, args.permute_y)
        print(f"Running {len(kernel_configs)} kernel configs")
        default_kernel_config_fwd = KernelConfigForward(
            permute_x=args.permute_x, permute_y=args.permute_y
        )
        default_kernel_config_bwd_dW = KernelConfigBackward_dW(
            permute_x=args.permute_x, permute_y=args.permute_y
        )
        default_kernel_config_bwd_dX = KernelConfigBackward_dX(
            permute_x=args.permute_x, permute_y=args.permute_y
        )
        results = []
        for kernel_config in kernel_configs:
            if args.mode == "forward":
                kernel_config_fwd = kernel_config
                kernel_config_bwd_dW = default_kernel_config_bwd_dW
                kernel_config_bwd_dX = default_kernel_config_bwd_dX
            elif args.mode == "dW":
                kernel_config_fwd = default_kernel_config_fwd
                kernel_config_bwd_dW = kernel_config
                kernel_config_bwd_dX = default_kernel_config_bwd_dX
            elif args.mode == "dX":
                kernel_config_fwd = default_kernel_config_fwd
                kernel_config_bwd_dW = default_kernel_config_bwd_dW
                kernel_config_bwd_dX = kernel_config
            else:
                raise ValueError(f"Invalid mode: {args.mode}")
            print(
                f"Benchmarking {model_id} {args.mode} with seqlen={args.seqlen}, dtype={args.dtype}, permute_x={args.permute_x}, permute_y={args.permute_y}, kernel_config_fwd={kernel_config_fwd}, kernel_config_bwd_dW={kernel_config_bwd_dW}, kernel_config_bwd_dX={kernel_config_bwd_dX}"
            )

            ref_time, fused_time = run_benchmark(
                args.mode,
                model_config,
                seqlen=args.seqlen,
                dtype=args.dtype,
                permute_x=kernel_config.permute_x,
                permute_y=kernel_config.permute_y,
                autotune=False,
                kernel_config_fwd=kernel_config_fwd,
                kernel_config_bwd_dW=kernel_config_bwd_dW,
                kernel_config_bwd_dX=kernel_config_bwd_dX,
            )
            results.append(
                KernelResult(
                    torch_time=ref_time,
                    triton_time=fused_time,
                    speedup=ref_time / fused_time,
                    kernel_config=kernel_config,
                )
            )
        df = post_process_results(
            results, args.mode, args.seqlen, args.dtype, args.autotune
        )
        save_results(
            df, args.results_dir, args.mode, args.seqlen, args.dtype, args.autotune
        )
