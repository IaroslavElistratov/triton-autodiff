# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/OpenMOSS/MOSS
# Source-Files: models/quantization.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_566h6b5n/MOSS-main/models/quantization.py
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
from torch.cuda.amp import custom_bwd
from torch.cuda.amp import custom_fwd

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@autotune(configs=[triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64,
    'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K':
    32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({
    'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,
    'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({
    'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32,
    'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({
    'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,
    'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({
    'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32,
    'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({
    'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32,
    'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({
    'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32,
    'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({
    'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32,
    'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({
    'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64,
    'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({
    'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64,
    'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({
    'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64,
    'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({
    'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128,
    'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4)], key=['M', 'N'],
    nearest_power_of_two=True)
@triton.jit
def matmul_248_kernel(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr, g_ptr, M,
    N, K, bits, maxq, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
    stride_cn, stride_scales, stride_zeros, BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M:
    tl.constexpr):
    """
    Compute the matrix multiplication C = A x B.
    A is of shape (M, K) float16
    B is of shape (K//8, N) int32
    C is of shape (M, N) float16
    scales is of shape (G, N) float16
    zeros is of shape (G, N) float16
    g_ptr is of shape (K) int32
    """
    infearure_per_bits = 32 // bits
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] *
        stride_ak)
    a_mask = offs_am[:, None] < M
    b_ptrs = b_ptr + (offs_k[:, None] // infearure_per_bits * stride_bk + 
        offs_bn[None, :] * stride_bn)
    g_ptrs = g_ptr + offs_k
    scales_ptrs = scales_ptr + offs_bn[None, :]
    zeros_ptrs = zeros_ptr + offs_bn[None, :] // infearure_per_bits
    shifter = offs_k % infearure_per_bits * bits
    zeros_shifter = offs_bn % infearure_per_bits * bits
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_pid_k):
        g_idx = tl.load(g_ptrs)
        scales = tl.load(scales_ptrs + g_idx[:, None] * stride_scales)
        zeros = tl.load(zeros_ptrs + g_idx[:, None] * stride_zeros)
        zeros = zeros >> zeros_shifter[None, :] & maxq
        zeros = zeros + 1
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs)
        b = b >> shifter[:, None] & maxq
        b = (b - zeros) * scales
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K // infearure_per_bits * stride_bk
        g_ptrs += BLOCK_SIZE_K
    c = accumulator.to(tl.float16)
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :
        ]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def matmul248(input, qweight, scales, qzeros, g_idx, bits, maxq):
    output = torch.empty((input.shape[0], qweight.shape[1]), device='cuda',
        dtype=torch.float16)
    grid = lambda META: (triton.cdiv(input.shape[0], META['BLOCK_SIZE_M']) *
        triton.cdiv(qweight.shape[1], META['BLOCK_SIZE_N']),)
    matmul_248_kernel[grid](input, qweight, output, scales, qzeros, g_idx,
        input.shape[0], qweight.shape[1], input.shape[1], bits, maxq, input
        .stride(0), input.stride(1), qweight.stride(0), qweight.stride(1),
        output.stride(0), output.stride(1), scales.stride(0), qzeros.stride(0))
    return output


# Forward method (kernel launch code)
@custom_fwd(cast_inputs=torch.float16)
def _QuantLinearFunction_forward(ctx, input, qweight, scales, qzeros, g_idx,
    bits, maxq):
    output = matmul248(input, qweight, scales, qzeros, g_idx, bits, maxq)
    ctx.save_for_backward(qweight, scales, qzeros, g_idx)
    ctx.bits, ctx.maxq = bits, maxq
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

def transpose_matmul248(input, qweight, scales, qzeros, g_idx, bits, maxq):
    output_dim = qweight.shape[0] * 32 // bits
    output = torch.empty((input.shape[0], output_dim), device='cuda', dtype
        =torch.float16)
    grid = lambda META: (triton.cdiv(input.shape[0], META['BLOCK_SIZE_M']) *
        triton.cdiv(output_dim, META['BLOCK_SIZE_K']),)
    transpose_matmul_248_kernel[grid](input, qweight, output, scales,
        qzeros, g_idx, input.shape[0], qweight.shape[1], output_dim, bits,
        maxq, input.stride(0), input.stride(1), qweight.stride(0), qweight.
        stride(1), output.stride(0), output.stride(1), scales.stride(0),
        qzeros.stride(0))
    return output


# Backward method (kernel launch code)
@custom_bwd
def _QuantLinearFunction_backward(ctx, grad_output):
    qweight, scales, qzeros, g_idx = ctx.saved_tensors
    bits, maxq = ctx.bits, ctx.maxq
    grad_input = None
    if ctx.needs_input_grad[0]:
        grad_input = transpose_matmul248(grad_output, qweight, scales,
            qzeros, g_idx, bits, maxq)
    return grad_input, None, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class QuantLinearFunction(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, qweight, scales, qzeros, g_idx, bits, maxq):
        output = matmul248(input, qweight, scales, qzeros, g_idx, bits, maxq)
        ctx.save_for_backward(qweight, scales, qzeros, g_idx)
        ctx.bits, ctx.maxq = bits, maxq
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        qweight, scales, qzeros, g_idx = ctx.saved_tensors
        bits, maxq = ctx.bits, ctx.maxq
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = transpose_matmul248(grad_output, qweight, scales,
                qzeros, g_idx, bits, maxq)
        return grad_input, None, None, None, None, None, None
