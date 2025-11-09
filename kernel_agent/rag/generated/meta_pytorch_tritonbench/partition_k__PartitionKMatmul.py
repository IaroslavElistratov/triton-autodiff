# SPDX-License-Identifier: BSD-3-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/meta-pytorch/tritonbench
# Source-Files: tritonbench/operators/gemm/partition_k.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_ct7v_342/tritonbench-main/tritonbench/operators/gemm/partition_k.py
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
from einops import reduce

@triton.autotune(configs=get_mm_configs(), key=['M', 'N', 'K', 'PK'])
@triton.jit
def _matmul_partition_k(a_ptr, b_ptr, c_buf_ptr, M, N, K, PK, PK_SIZE,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cb_m, stride_cb_n,
    stride_cb_k, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_pk = PK
    num_pid_nk = num_pid_n * num_pid_pk
    num_pid_in_group = GROUP_SIZE_M * num_pid_nk
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_nk = pid % num_pid_in_group // group_size_m
    pid_n = pid_nk // num_pid_pk
    pid_pk = pid_nk % num_pid_pk
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = (pid_pk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) % K
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] *
        stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] *
        stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(PK_SIZE, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += PK_SIZE * stride_ak
        b_ptrs += PK_SIZE * stride_bk
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_ck = pid_pk
    c_buf_ptrs = c_buf_ptr + stride_cb_m * offs_cm[:, None, None
        ] + stride_cb_n * offs_cn[None, :, None] + stride_cb_k * offs_ck[
        None, None, :]
    tl.store(c_buf_ptrs, accumulator[:, :, None])


def _matmul_partition_k_impl(a, b, triton_reduce=False):
    assert a.shape[1] == b.shape[0], 'Incompatible dimensions'
    assert a.is_contiguous(), 'Matrix A must be contiguous'
    assert b.is_contiguous(), 'Matrix B must be contiguous'
    partitionK = 32
    M, K = a.shape
    K, N = b.shape
    partitionK_SIZE = K // partitionK
    c_buf = torch.empty((M, N, partitionK), device=a.device, dtype=torch.
        float32)
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv
        (N, META['BLOCK_SIZE_N']) * partitionK,)
    _matmul_partition_k[grid](a, b, c_buf, M, N, K, partitionK,
        partitionK_SIZE, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c_buf.stride(0), c_buf.stride(1), c_buf.stride(2))
    if triton_reduce:
        BLOCK_M = 32
        BLOCK_N = 32
        grid_reduce = lambda META: (triton.cdiv(M, BLOCK_M) * triton.cdiv(N,
            BLOCK_N),)
        _reduce[grid_reduce](c, c_buf, M, N, c.stride(0), c.stride(1),
            c_buf.stride(0), c_buf.stride(1), c_buf.stride(2), partitionK,
            BLOCK_M, BLOCK_N)
        return c
    else:
        return compiled_reduction(c_buf, a)


@triton.jit
def _reduce(c_ptr, c_buf_ptr, M, N, stride_cm, stride_cn, stride_cb_m,
    stride_cb_n, stride_cb_k, PK: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_m
    pid_n = pid % num_pid_n
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, PK)
    c_buf_ptrs = c_buf_ptr + (offs_m[:, None, None] * stride_cb_m + offs_n[
        None, :, None] * stride_cb_n + offs_k[None, None, :] * stride_cb_k)
    c_buf = tl.load(c_buf_ptrs)
    reduced_k = tl.sum(c_buf, axis=2)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        )
    tl.store(c_ptrs, reduced_k)


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

# Forward method (kernel launch code)
def __PartitionKMatmul_forward(ctx, a, b, triton_reduce=False):
    ctx.save_for_backward(a, b)
    ctx.triton_reduce = triton_reduce
    return _matmul_partition_k_impl(a, b, triton_reduce)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

# Backward method (kernel launch code)
def __PartitionKMatmul_backward(ctx, grad_output):
    a, b = ctx.saved_tensors
    grad_a = grad_b = None
    if ctx.needs_input_grad[0]:
        grad_a = _matmul_partition_k_impl(grad_output, b.t().contiguous(),
            ctx.triton_reduce)
    if ctx.needs_input_grad[1]:
        grad_b = _matmul_partition_k_impl(a.t().contiguous(), grad_output,
            ctx.triton_reduce)
    return grad_a, grad_b, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _PartitionKMatmul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b, triton_reduce=False):
        ctx.save_for_backward(a, b)
        ctx.triton_reduce = triton_reduce
        return _matmul_partition_k_impl(a, b, triton_reduce)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_a = _matmul_partition_k_impl(grad_output, b.t().contiguous
                (), ctx.triton_reduce)
        if ctx.needs_input_grad[1]:
            grad_b = _matmul_partition_k_impl(a.t().contiguous(),
                grad_output, ctx.triton_reduce)
        return grad_a, grad_b, None
