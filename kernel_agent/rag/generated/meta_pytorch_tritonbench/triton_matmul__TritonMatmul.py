# SPDX-License-Identifier: BSD-3-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/meta-pytorch/tritonbench
# Source-Files: tritonbench/operators/gemm/triton_matmul.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_ct7v_342/tritonbench-main/tritonbench/operators/gemm/triton_matmul.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

def _matmul_impl(a, b, activation=''):
    assert a.shape[1] == b.shape[0], 'Incompatible dimensions'
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N,
        META['BLOCK_N']),)
    enable_buffer_ops_assumes = a.stride(0) >= 0 and a.stride(1
        ) >= 0 and b.stride(0) >= 0 and b.stride(1) >= 0 and c.stride(0
        ) >= 0 and c.stride(1) >= 0
    matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.
        stride(0), b.stride(1), c.stride(0), c.stride(1), ACTIVATION=
        activation, ENABLE_BUFFER_OPS_ASSUMES=enable_buffer_ops_assumes)
    return c


@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)


@triton.autotune(configs=tuning_configs, key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak,
    stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
    ACTIVATION: tl.constexpr, ENABLE_BUFFER_OPS_ASSUMES: tl.constexpr):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    if ENABLE_BUFFER_OPS_ASSUMES:
        tl.assume(M >= 0)
        tl.assume(N >= 0)
        tl.assume(K >= 0)
        tl.assume(stride_am >= 0)
        tl.assume(stride_ak >= 0)
        tl.assume(stride_bn >= 0)
        tl.assume(stride_bk >= 0)
        tl.assume(stride_cm >= 0)
        tl.assume(stride_cn >= 0)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] *
        stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] *
        stride_bn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    if ACTIVATION == 'leaky_relu':
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :
        ]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

# Forward method (kernel launch code)
def __TritonMatmul_forward(ctx, a, b, activation=''):
    ctx.save_for_backward(a, b)
    ctx.activation = activation
    return _matmul_impl(a, b, activation)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

# Backward method (kernel launch code)
def __TritonMatmul_backward(ctx, grad_output):
    a, b = ctx.saved_tensors
    grad_a = grad_b = None
    if ctx.needs_input_grad[0]:
        grad_a = _matmul_impl(grad_output, b.t().contiguous(), '')
    if ctx.needs_input_grad[1]:
        grad_b = _matmul_impl(a.t().contiguous(), grad_output, '')
    return grad_a, grad_b, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _TritonMatmul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b, activation=''):
        ctx.save_for_backward(a, b)
        ctx.activation = activation
        return _matmul_impl(a, b, activation)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_a = _matmul_impl(grad_output, b.t().contiguous(), '')
        if ctx.needs_input_grad[1]:
            grad_b = _matmul_impl(a.t().contiguous(), grad_output, '')
        return grad_a, grad_b, None
