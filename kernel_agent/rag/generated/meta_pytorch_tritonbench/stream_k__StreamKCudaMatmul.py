# SPDX-License-Identifier: BSD-3-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/meta-pytorch/tritonbench
# Source-Files: tritonbench/operators/gemm/stream_k.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_ct7v_342/tritonbench-main/tritonbench/operators/gemm/stream_k.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

def _streamk_cuda_matmul_impl(a, b):
    assert a.dtype == b.dtype, 'Incompatible dtypes'
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    c = torch.zeros((M, N), device=a.device, dtype=dtype)
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)
    a_desc_sk = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc_sk = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    num_sms = torch.cuda.get_device_properties('cuda').multi_processor_count

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META['BLOCK_M']
        BLOCK_N = META['BLOCK_N']
        num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
        W = num_tiles // num_sms
        R = num_tiles % num_sms
        if W == 0 or R == 0:
            total_ddp_tiles = num_tiles
            streamk_sms = 0
        else:
            total_ddp_tiles = (W - 1) * num_sms
            streamk_sms = num_sms
        return total_ddp_tiles + streamk_sms,
    streamk_cuda_gemm[grid](a_desc, b_desc, a_desc_sk, b_desc_sk, c_desc, M,
        N, K, FP8_OUTPUT=dtype == torch.float8_e4m3fn,
        ENABLE_BUFFER_OPS_ASSUMES=True, NUM_SMS=num_sms)
    return c


@triton.autotune(configs=matmul_get_configs(pre_hook=
    matmul_tma_set_block_size_hook), key=['M', 'N', 'K'])
@triton.jit(launch_metadata=_matmul_launch_metadata)
def streamk_cuda_gemm(a_desc, b_desc, a_desc_sk, b_desc_sk, c_desc, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SK_BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, FP8_OUTPUT: tl.
    constexpr, ENABLE_BUFFER_OPS_ASSUMES: tl.constexpr, NUM_SMS: tl.constexpr):
    if ENABLE_BUFFER_OPS_ASSUMES:
        tl.assume(M >= 0)
        tl.assume(N >= 0)
        tl.assume(K >= 0)
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_tile_m = tl.cdiv(M, BLOCK_M)
    num_tile_n = tl.cdiv(N, BLOCK_N)
    num_tile_in_group = GROUP_M * num_tile_n
    total_tiles = num_tile_m * num_tile_n
    W = total_tiles // NUM_SMS
    R = total_tiles % NUM_SMS
    if W == 0 or R == 0:
        total_ddp_tiles = num_pid
        streamk_sms = 0
    else:
        total_ddp_tiles = num_pid - NUM_SMS
        streamk_sms = NUM_SMS
    if pid < total_ddp_tiles:
        group_id = pid // num_tile_in_group
        first_tile_m = group_id * GROUP_M
        group_size_m = min(num_tile_m - first_tile_m, GROUP_M)
        tile_m = first_tile_m + pid % group_size_m
        tile_n = pid % num_tile_in_group // group_size_m
        offs_am = tile_m * BLOCK_M
        offs_bn = tile_n * BLOCK_N
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        work_units_per_tile = tl.cdiv(K, BLOCK_K)
        for k in tl.range(0, work_units_per_tile, warp_specialize=True):
            offs_k = k * BLOCK_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)
        c = accumulator.to(dtype)
        c_desc.store([offs_am, offs_bn], c)
    else:
        worker_id = pid - total_ddp_tiles
        work_units_per_tile = tl.cdiv(K, SK_BLOCK_K)
        total_work_units = (total_tiles - total_ddp_tiles
            ) * work_units_per_tile
        base = total_work_units // streamk_sms
        rem = total_work_units % streamk_sms
        work = tl.where(worker_id < rem, base + 1, base)
        start = tl.where(worker_id < rem, worker_id * (base + 1), rem * (
            base + 1) + (worker_id - rem) * base)
        end = start + work - 1
        if start >= total_work_units:
            return
        st_tile_streamk = start // work_units_per_tile + total_ddp_tiles
        st_k_streamk = start % work_units_per_tile
        en_tile_streamk = end // work_units_per_tile + total_ddp_tiles
        en_k_streamk = end % work_units_per_tile
        for curr_tile in tl.range(st_tile_streamk, en_tile_streamk + 1,
            flatten=True):
            group_id = curr_tile // num_tile_in_group
            first_tile_m = group_id * GROUP_M
            group_size_m = min(num_tile_m - first_tile_m, GROUP_M)
            tile_m = first_tile_m + curr_tile % group_size_m
            tile_n = curr_tile % num_tile_in_group // group_size_m
            offs_am = tile_m * BLOCK_M
            offs_bn = tile_n * BLOCK_N
            curr_st_k = tl.where(curr_tile == st_tile_streamk, st_k_streamk, 0)
            curr_en_k = tl.where(curr_tile == en_tile_streamk, en_k_streamk,
                work_units_per_tile - 1)
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in tl.range(curr_st_k, curr_en_k + 1, warp_specialize=True):
                offs_k = k * SK_BLOCK_K
                if BLOCK_K == SK_BLOCK_K:
                    a = a_desc.load([offs_am, offs_k])
                    b = b_desc.load([offs_bn, offs_k])
                else:
                    a = a_desc_sk.load([offs_am, offs_k])
                    b = b_desc_sk.load([offs_bn, offs_k])
                accumulator = tl.dot(a, b.T, accumulator)
            c = accumulator.to(dtype)
            if curr_st_k == 0 and curr_en_k == work_units_per_tile - 1:
                c_desc.store([offs_am, offs_bn], c)
            else:
                c_desc.atomic_add([offs_am, offs_bn], c)


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

# Forward method (kernel launch code)
def __StreamKCudaMatmul_forward(ctx, a, b):
    ctx.save_for_backward(a, b)
    return _streamk_cuda_matmul_impl(a, b)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

# Backward method (kernel launch code)
def __StreamKCudaMatmul_backward(ctx, grad_output):
    a, b = ctx.saved_tensors
    grad_a = grad_b = None
    if ctx.needs_input_grad[0]:
        grad_a = _streamk_cuda_matmul_impl(grad_output, b.t().contiguous())
    if ctx.needs_input_grad[1]:
        grad_b = _streamk_cuda_matmul_impl(a.t().contiguous(), grad_output)
    return grad_a, grad_b


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _StreamKCudaMatmul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return _streamk_cuda_matmul_impl(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_a = _streamk_cuda_matmul_impl(grad_output, b.t().contiguous())
        if ctx.needs_input_grad[1]:
            grad_b = _streamk_cuda_matmul_impl(a.t().contiguous(), grad_output)
        return grad_a, grad_b
