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

def _streamk_amd_matmul_impl(a, b, bias=None):
    M, K = a.shape
    _, N = b.shape
    dtype = a.dtype
    c = torch.zeros((M, N), device=a.device, dtype=dtype)
    NUM_SMS = torch.cuda.get_device_properties('cuda').multi_processor_count
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    if K > (M + N) * 2:
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
    total_tiles = triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)
    total_programs_streamk = min(NUM_SMS, total_tiles)
    total_tiles_streamk = total_tiles % total_programs_streamk
    grids = min(total_programs_streamk, total_tiles)
    """
    print("total_tiles: ", total_tiles)
    print("total_tiles_streamk: ", total_tiles_streamk)
    print("total_programs_streamk: ", total_programs_streamk)
    print("grids: ", grids)
    """
    if bias is not None:
        is_bias_1d = bias.dim() == 1
        if not is_bias_1d:
            bias_stride_m = bias.stride(0)
            bias_stride_n = bias.stride(1)
        elif bias.dim() == 1:
            bias_stride_m = 0
            bias_stride_n = bias.stride(0)
    else:
        is_bias_1d = 0
        bias_stride_m = 0
        bias_stride_n = 0
    NUM_XCDS = 1
    if is_hip_mi300():
        if grids % 38 == 0:
            NUM_XCDS = grids // 38
    enable_buffer_ops_assumes = a.stride(0) >= 0 and a.stride(1
        ) >= 0 and b.stride(0) >= 0 and b.stride(1) >= 0 and c.stride(0
        ) >= 0 and c.stride(1) >= 0
    streamk_amd_gemm[grids,](a, b, c, bias, M, N, K, a.stride(0), a.stride(
        1), b.stride(0), b.stride(1), bias_stride_m, bias_stride_n, c.
        stride(0), c.stride(1), HAS_BIAS=bias is not None, NUM_SMS=grids,
        STREAMK_TILES=total_tiles_streamk, NUM_XCDS=NUM_XCDS,
        ENABLE_BUFFER_OPS_ASSUMES=enable_buffer_ops_assumes)
    return c


@triton.autotune(configs=tuning_configs, key=['M', 'N', 'K'])
@triton.heuristics(values={'EVEN_M': lambda args: args['M'] % args[
    'BLOCK_M'] == 0, 'EVEN_N': lambda args: args['N'] % args['BLOCK_N'] == 
    0, 'EVEN_K': lambda args: args['K'] % args['BLOCK_K'] == 0})
@triton.jit
def streamk_amd_gemm(A, B, C, bias_ptr, M, N, K, stride_am, stride_ak,
    stride_bk, stride_bn, stride_bias_m, stride_bias_n, stride_cm,
    stride_cn, HAS_BIAS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.
    constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, NUM_SMS: tl.
    constexpr, NUM_XCDS: tl.constexpr, STREAMK_TILES: tl.constexpr, EVEN_M:
    tl.constexpr, EVEN_N: tl.constexpr, EVEN_K: tl.constexpr,
    ENABLE_BUFFER_OPS_ASSUMES: tl.constexpr):
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
    if stride_bias_m:
        tl.assume(stride_bias_m >= 0)
    if stride_bias_n:
        tl.assume(stride_bias_n >= 0)
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = pid % NUM_XCDS * (NUM_SMS // NUM_XCDS) + pid // NUM_XCDS
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    iters_per_tile = tl.cdiv(K, BLOCK_K)
    total_tiles = num_pid_m * num_pid_n
    total_full_tiles = total_tiles - STREAMK_TILES
    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    for tile_id in range(pid, total_full_tiles, NUM_SMS):
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + tile_id % num_pid_in_group % group_size_m
        pid_n = tile_id % num_pid_in_group // group_size_m
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        rk = tl.arange(0, BLOCK_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        loop_k = tl.cdiv(K, BLOCK_K)
        if not EVEN_K:
            loop_k -= 1
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
        if HAS_BIAS:
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            bias_ = rn[None, :] * stride_bias_n + rm[:, None] * stride_bias_m
            bias = tl.load(bias_ptr + tl.broadcast_to(bias_, (BLOCK_M,
                BLOCK_N)), mask=mask).to(acc.dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_K * stride_ak
            B_BASE += BLOCK_K * stride_bk
        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_K + tl.arange(0, BLOCK_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            A_BASE = tl.multiple_of(A_BASE, (1, 16))
            B_BASE = tl.multiple_of(B_BASE, (16, 1))
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
            acc += tl.dot(a, b)
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        if HAS_BIAS:
            acc += bias
        c = acc.to(C.type.element_ty)
        tl.store(C_, c, mask=mask)
    tl.assume(pid >= 0)
    total_streamk_iters = STREAMK_TILES * iters_per_tile
    streamk_iters_pcu = total_streamk_iters // NUM_SMS
    streamk_remainder_iters = total_streamk_iters % NUM_SMS
    start_iter = (total_full_tiles * iters_per_tile + pid *
        streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters))
    last_iter = total_full_tiles * iters_per_tile + (pid + 1
        ) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        tile_iter_end = start_iter + (iters_per_tile - remainder)
        tile_id = start_iter // iters_per_tile
        end_iter = tl.minimum(tile_iter_end, last_iter)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + tile_id % num_pid_in_group % group_size_m
        pid_n = tile_id % num_pid_in_group // group_size_m
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) % M
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N) % N
        rk = tl.arange(0, BLOCK_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :
            ] * stride_ak + BLOCK_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :
            ] * stride_bn + BLOCK_K * stride_bk * remainder
        A_BASE = tl.multiple_of(A_BASE, (1, 16))
        B_BASE = tl.multiple_of(B_BASE, (16, 1))
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
        if HAS_BIAS:
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            bias_ = rn[None, :] * stride_bias_n + rm[:, None] * stride_bias_m
            bias = tl.load(bias_ptr + tl.broadcast_to(bias_, (BLOCK_M,
                BLOCK_N)), mask=mask).to(acc.dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                if EVEN_M:
                    a = tl.load(A_BASE)
                else:
                    mask_a = (rm < M)[:, None]
                    a = tl.load(A_BASE, mask=mask_a, other=0.0)
                if EVEN_N:
                    b = tl.load(B_BASE)
                else:
                    mask_b = (rn < N)[None, :]
                    b = tl.load(B_BASE, mask=mask_b, other=0.0)
            else:
                global_k_offset = current_iter % iters_per_tile * BLOCK_K
                k_mask = global_k_offset + rk < K
                if EVEN_M:
                    a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                else:
                    mask_a = (rm < M)[:, None]
                    a = tl.load(A_BASE, mask=k_mask[None, :] & mask_a,
                        other=0.0)
                if EVEN_N:
                    b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
                else:
                    mask_b = (rn < N)[None, :]
                    b = tl.load(B_BASE, mask=k_mask[:, None] & mask_b,
                        other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_K * stride_ak
            B_BASE += BLOCK_K * stride_bk
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        if HAS_BIAS:
            acc += bias
        c = acc.to(C.type.element_ty)
        tl.atomic_add(C_, c, mask=mask, sem='relaxed')
        start_iter = end_iter


def is_hip() ->bool:
    return torch.version.hip is not None


def is_hip_mi300():
    target = triton.runtime.driver.active.get_current_target()
    return is_hip() and target.arch == 'gfx942'


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

# Forward method (kernel launch code)
def __StreamKAmdMatmul_forward(ctx, a, b, bias=None):
    ctx.save_for_backward(a, b, bias)
    return _streamk_amd_matmul_impl(a, b, bias)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

# Backward method (kernel launch code)
def __StreamKAmdMatmul_backward(ctx, grad_output):
    a, b, bias = ctx.saved_tensors
    grad_a = grad_b = grad_bias = None
    if ctx.needs_input_grad[0]:
        grad_a = _streamk_amd_matmul_impl(grad_output, b.t().contiguous())
    if ctx.needs_input_grad[1]:
        grad_b = _streamk_amd_matmul_impl(a.t().contiguous(), grad_output)
    if ctx.needs_input_grad[2] and bias is not None:
        grad_bias = grad_output.sum(dim=0)
        if bias.dim() == 2:
            grad_bias = grad_bias.unsqueeze(0)
    return grad_a, grad_b, grad_bias


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _StreamKAmdMatmul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b, bias=None):
        ctx.save_for_backward(a, b, bias)
        return _streamk_amd_matmul_impl(a, b, bias)

    @staticmethod
    def backward(ctx, grad_output):
        a, b, bias = ctx.saved_tensors
        grad_a = grad_b = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_a = _streamk_amd_matmul_impl(grad_output, b.t().contiguous())
        if ctx.needs_input_grad[1]:
            grad_b = _streamk_amd_matmul_impl(a.t().contiguous(), grad_output)
        if ctx.needs_input_grad[2] and bias is not None:
            grad_bias = grad_output.sum(dim=0)
            if bias.dim() == 2:
                grad_bias = grad_bias.unsqueeze(0)
        return grad_a, grad_b, grad_bias
