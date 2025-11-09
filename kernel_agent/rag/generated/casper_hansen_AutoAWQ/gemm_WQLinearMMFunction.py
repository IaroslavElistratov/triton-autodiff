# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/casper-hansen/AutoAWQ
# Source-Files: awq/modules/linear/gemm.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT__05pewlf/AutoAWQ-main/awq/modules/linear/gemm.py
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
from einops import repeat

@triton.jit
def awq_dequantize_kernel(qweight_ptr, scales_ptr, zeros_ptr, group_size,
    result_ptr, num_cols, num_rows, BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offsets = num_cols * offsets_y[:, None] + offsets_x[None, :]
    masks_y = offsets_y < num_rows
    masks_x = offsets_x < num_cols
    masks = masks_y[:, None] & masks_x[None, :]
    result_offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    result_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8
        )
    result_offsets = 8 * num_cols * result_offsets_y[:, None
        ] + result_offsets_x[None, :]
    result_masks_y = result_offsets_y < num_rows
    result_masks_x = result_offsets_x < num_cols * 8
    result_masks = result_masks_y[:, None] & result_masks_x[None, :]
    iweights = tl.load(qweight_ptr + offsets, masks)
    iweights = tl.interleave(iweights, iweights)
    iweights = tl.interleave(iweights, iweights)
    iweights = tl.interleave(iweights, iweights)
    reverse_awq_order_tensor = ((tl.arange(0, 2) * 4)[None, :] + tl.arange(
        0, 4)[:, None]).reshape(8)
    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_Y * BLOCK_SIZE_X, 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))
    iweights = iweights >> shifts & 15
    zero_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    zero_offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    zero_offsets = num_cols * zero_offsets_y[:, None] + zero_offsets_x[None, :]
    zero_masks_y = zero_offsets_y < num_rows // group_size
    zero_masks_x = zero_offsets_x < num_cols
    zero_masks = zero_masks_y[:, None] & zero_masks_x[None, :]
    zeros = tl.load(zeros_ptr + zero_offsets, zero_masks)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))
    zeros = zeros >> shifts & 15
    scale_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    scale_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)
    scale_offsets = num_cols * 8 * scale_offsets_y[:, None] + scale_offsets_x[
        None, :]
    scale_masks_y = scale_offsets_y < num_rows // group_size
    scale_masks_x = scale_offsets_x < num_cols * 8
    scale_masks = scale_masks_y[:, None] & scale_masks_x[None, :]
    scales = tl.load(scales_ptr + scale_offsets, scale_masks)
    scales = tl.broadcast_to(scales, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))
    iweights = (iweights - zeros) * scales
    iweights = iweights.to(result_ptr.type.element_ty)
    tl.store(result_ptr + result_offsets, iweights, result_masks)


def awq_dequantize_triton(qweight: torch.Tensor, scales: torch.Tensor,
    zeros: torch.Tensor, block_size_x: int=32, block_size_y: int=32
    ) ->torch.Tensor:
    K = qweight.shape[0]
    M = scales.shape[1]
    group_size = qweight.shape[0] // scales.shape[0]
    assert K > 0 and M > 0
    assert scales.shape[0] == K // group_size and scales.shape[1] == M
    assert zeros.shape[0] == K // group_size and zeros.shape[1] == M // 8
    assert group_size <= K
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K
    result = torch.empty(qweight.shape[0], qweight.shape[1] * 8, device=
        qweight.device, dtype=scales.dtype)
    Y = qweight.shape[0]
    X = qweight.shape[1]
    grid = lambda META: (triton.cdiv(X, META['BLOCK_SIZE_X']), triton.cdiv(
        Y, META['BLOCK_SIZE_Y']))
    with get_same_device_cm(qweight):
        awq_dequantize_kernel[grid](qweight, scales, zeros, group_size,
            result, X, Y, BLOCK_SIZE_X=block_size_x, BLOCK_SIZE_Y=block_size_y)
    return result


def get_same_device_cm(t):
    if t.device.type == 'xpu':
        return torch.xpu.device(t.device.index)
    else:
        return torch.cuda.device(t.device.index)


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def awq_gemm_kernel(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, M, N, K,
    group_size, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, SPLIT_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(1)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    accumulator_dtype = c_ptr.type.element_ty
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=
        accumulator_dtype)
    reverse_awq_order_tensor = ((tl.arange(0, 2) * 4)[None, :] + tl.arange(
        0, 4)[:, None]).reshape(8)
    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_K * (BLOCK_SIZE_N //
        8), 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_K, BLOCK_SIZE_N))
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    masks_am = offsets_am < M
    offsets_bn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    masks_bn = offsets_bn < N // 8
    offsets_zn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    masks_zn = offsets_zn < N // 8
    offsets_sn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    masks_sn = offsets_sn < N
    offsets_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offsets_a = K * offsets_am[:, None] + offsets_k[None, :]
    offsets_b = N // 8 * offsets_k[:, None] + offsets_bn[None, :]
    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a)
        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        offsets_szk = (BLOCK_SIZE_K * SPLIT_K * k + pid_z * BLOCK_SIZE_K
            ) // group_size + tl.arange(0, 1)
        offsets_z = N // 8 * offsets_szk[:, None] + offsets_zn[None, :]
        masks_zk = offsets_szk < K // group_size
        masks_z = masks_zk[:, None] & masks_zn[None, :]
        zeros_ptrs = zeros_ptr + offsets_z
        zeros = tl.load(zeros_ptrs, mask=masks_z)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_K, BLOCK_SIZE_N))
        offsets_s = N * offsets_szk[:, None] + offsets_sn[None, :]
        masks_sk = offsets_szk < K // group_size
        masks_s = masks_sk[:, None] & masks_sn[None, :]
        scales_ptrs = scales_ptr + offsets_s
        scales = tl.load(scales_ptrs, mask=masks_s)
        scales = tl.broadcast_to(scales, (BLOCK_SIZE_K, BLOCK_SIZE_N))
        b = b >> shifts & 15
        zeros = zeros >> shifts & 15
        b = (b - zeros) * scales
        b = b.to(c_ptr.type.element_ty)
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)
        offsets_k += BLOCK_SIZE_K * SPLIT_K
        a_ptrs += BLOCK_SIZE_K * SPLIT_K
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * (N // 8)
    c = accumulator.to(c_ptr.type.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)


def awq_gemm_triton(input: torch.Tensor, qweight: torch.Tensor, scales:
    torch.Tensor, qzeros: torch.Tensor, split_k_iters: int, block_size_m:
    int=32, block_size_n: int=32, block_size_k: int=32) ->torch.Tensor:
    M, K = input.shape
    N = qweight.shape[1] * 8
    group_size = qweight.shape[0] // qzeros.shape[0]
    assert N > 0 and K > 0 and M > 0
    assert qweight.shape[0] == K and qweight.shape[1] == N // 8
    assert qzeros.shape[0] == K // group_size and qzeros.shape[1] == N // 8
    assert scales.shape[0] == K // group_size and scales.shape[1] == N
    assert split_k_iters & split_k_iters - 1 == 0 and split_k_iters != 0
    assert split_k_iters <= 32
    assert group_size <= K
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv
        (N, META['BLOCK_SIZE_N']), split_k_iters)
    result = torch.zeros((M, N), dtype=scales.dtype, device=input.device)
    with get_same_device_cm(qweight):
        awq_gemm_kernel[grid](input, qweight, result, qzeros, scales, M, N,
            K, group_size, BLOCK_SIZE_M=block_size_m, BLOCK_SIZE_N=
            block_size_n, BLOCK_SIZE_K=block_size_k, SPLIT_K=split_k_iters)
    return result


def dequantize_gemm(qweight, qzeros, scales, bits, group_size):
    iweight, izeros = unpack_awq(qweight, qzeros, bits)
    iweight, izeros = reverse_awq_order(iweight, izeros, bits)
    iweight = torch.bitwise_and(iweight, 2 ** bits - 1)
    izeros = torch.bitwise_and(izeros, 2 ** bits - 1)
    scales = scales.repeat_interleave(group_size, dim=0)
    izeros = izeros.repeat_interleave(group_size, dim=0)
    iweight = (iweight - izeros) * scales
    return iweight


def reverse_awq_order(iweights: torch.Tensor, izeros: torch.Tensor, bits: int):
    reverse_order_tensor = torch.arange(iweights.shape[-1], dtype=torch.
        int32, device=izeros.device)
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)
    if izeros is not None:
        izeros = izeros[:, reverse_order_tensor]
    iweights = iweights[:, reverse_order_tensor]
    return iweights, izeros


def unpack_awq(qweight: torch.Tensor, qzeros: torch.Tensor, bits: int):
    shifts = torch.arange(0, 32, bits, device=qzeros.device)
    iweights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None,
        None, :]).to(torch.int8)
    iweights = iweights.view(iweights.shape[0], -1)
    if qzeros is not None:
        izeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None,
            None, :]).to(torch.int8)
        izeros = izeros.view(izeros.shape[0], -1)
    else:
        izeros = qzeros
    return iweights, izeros


# Forward method (kernel launch code)
def _WQLinearMMFunction_forward(ctx, x, qweight, qzeros, scales, w_bit=4,
    group_size=128, bias=None, out_features=0):
    ctx.save_for_backward(x, qweight, qzeros, scales, bias)
    ctx.out_features = out_features
    out_shape = x.shape[:-1] + (out_features,)
    x = x.to(torch.float16)
    if x.shape[0] == 0:
        return torch.zeros(out_shape, dtype=x.dtype, device=x.device)
    if awq_ext is not None:
        FP16_MATMUL_HEURISTIC_CONDITION = x.shape[0] * x.shape[1] >= 1024
        if FP16_MATMUL_HEURISTIC_CONDITION:
            out = awq_ext.dequantize_weights_cuda(qweight, scales, qzeros, 
                0, 0, 0, False)
            out = torch.matmul(x, out)
        else:
            out = awq_ext.gemm_forward_cuda(x.reshape(-1, x.shape[-1]),
                qweight, scales, qzeros, 8)
    elif TRITON_AVAILABLE:
        FP16_MATMUL_HEURISTIC_CONDITION = x.shape[0] * x.shape[1] >= 1024
        if FP16_MATMUL_HEURISTIC_CONDITION:
            out = awq_dequantize_triton(qweight, scales, qzeros)
            out = torch.matmul(x, out.to(x.dtype))
        else:
            out = awq_gemm_triton(x.reshape(-1, x.shape[-1]), qweight,
                scales, qzeros, split_k_iters=8)
    else:
        global user_has_been_warned
        if not user_has_been_warned:
            warnings.warn('Using naive (slow) implementation.' + msg)
            user_has_been_warned = True
        out = dequantize_gemm(qweight, qzeros, scales, w_bit, group_size)
        out = torch.matmul(x, out)
    out = out + bias if bias is not None else out
    out = out.reshape(out_shape)
    if len(out.shape) == 2:
        out = out.unsqueeze(0)
    return out


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

# Backward method (kernel launch code)
def _WQLinearMMFunction_backward(ctx, grad_output):
    input, qweight, qzeros, scales, bias = ctx.saved_tensors
    if awq_ext is None and not TRITON_AVAILABLE:
        raise ValueError(
            'either triton or autoawq-kernels is needed to be installed to use `.backward()`. Make sure to install the auto-awq kernels by following the installation guides in https://github.com/casper-hansen/AutoAWQ_kernels'
            )
    if awq_ext is not None:
        weights = awq_ext.dequantize_weights_cuda(qweight, scales, qzeros, 
            1, 0, 0, False).to(grad_output.dtype)
    else:
        weights = awq_dequantize_triton(qweight, scales, qzeros).to(grad_output
            .dtype)
    if ctx.needs_input_grad[0]:
        batch_size = grad_output.shape[0]
        grad_input = grad_output.bmm(weights.transpose(0, 1).unsqueeze(0).
            repeat(batch_size, 1, 1))
    return grad_input, None, None, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class WQLinearMMFunction(Function):

    @staticmethod
    def forward(ctx, x, qweight, qzeros, scales, w_bit=4, group_size=128,
        bias=None, out_features=0):
        ctx.save_for_backward(x, qweight, qzeros, scales, bias)
        ctx.out_features = out_features
        out_shape = x.shape[:-1] + (out_features,)
        x = x.to(torch.float16)
        if x.shape[0] == 0:
            return torch.zeros(out_shape, dtype=x.dtype, device=x.device)
        if awq_ext is not None:
            FP16_MATMUL_HEURISTIC_CONDITION = x.shape[0] * x.shape[1] >= 1024
            if FP16_MATMUL_HEURISTIC_CONDITION:
                out = awq_ext.dequantize_weights_cuda(qweight, scales,
                    qzeros, 0, 0, 0, False)
                out = torch.matmul(x, out)
            else:
                out = awq_ext.gemm_forward_cuda(x.reshape(-1, x.shape[-1]),
                    qweight, scales, qzeros, 8)
        elif TRITON_AVAILABLE:
            FP16_MATMUL_HEURISTIC_CONDITION = x.shape[0] * x.shape[1] >= 1024
            if FP16_MATMUL_HEURISTIC_CONDITION:
                out = awq_dequantize_triton(qweight, scales, qzeros)
                out = torch.matmul(x, out.to(x.dtype))
            else:
                out = awq_gemm_triton(x.reshape(-1, x.shape[-1]), qweight,
                    scales, qzeros, split_k_iters=8)
        else:
            global user_has_been_warned
            if not user_has_been_warned:
                warnings.warn('Using naive (slow) implementation.' + msg)
                user_has_been_warned = True
            out = dequantize_gemm(qweight, qzeros, scales, w_bit, group_size)
            out = torch.matmul(x, out)
        out = out + bias if bias is not None else out
        out = out.reshape(out_shape)
        if len(out.shape) == 2:
            out = out.unsqueeze(0)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, qweight, qzeros, scales, bias = ctx.saved_tensors
        if awq_ext is None and not TRITON_AVAILABLE:
            raise ValueError(
                'either triton or autoawq-kernels is needed to be installed to use `.backward()`. Make sure to install the auto-awq kernels by following the installation guides in https://github.com/casper-hansen/AutoAWQ_kernels'
                )
        if awq_ext is not None:
            weights = awq_ext.dequantize_weights_cuda(qweight, scales,
                qzeros, 1, 0, 0, False).to(grad_output.dtype)
        else:
            weights = awq_dequantize_triton(qweight, scales, qzeros).to(
                grad_output.dtype)
        if ctx.needs_input_grad[0]:
            batch_size = grad_output.shape[0]
            grad_input = grad_output.bmm(weights.transpose(0, 1).unsqueeze(
                0).repeat(batch_size, 1, 1))
        return grad_input, None, None, None, None, None, None, None
