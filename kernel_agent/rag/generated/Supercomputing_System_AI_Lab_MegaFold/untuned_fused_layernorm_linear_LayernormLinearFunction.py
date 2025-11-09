# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/Supercomputing-System-AI-Lab/MegaFold
# Source-Files: megafold/model/FusedLayernormLinear/untuned_fused_layernorm_linear.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_qk7vt93p/MegaFold-main/megafold/model/FusedLayernormLinear/untuned_fused_layernorm_linear.py
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

@triton.jit
def _first_forward_kernel(X_ptr, X_row_stride, Mean_ptr, Mean_row_stride,
    RSTD_ptr, RSTD_row_stride, K, eps, BLOCK_SIZE: tl.constexpr):
    """
    First kernel just calculates the mean and variance along the whole row and store back 
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < K
    X_ptr += row_idx * X_row_stride
    Mean_ptr += row_idx * Mean_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride
    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0)
    mean = tl.sum(X_row, axis=0) / K
    var = tl.sum((X_row - mean) * (X_row - mean), axis=0) / K
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean_ptr, mean)
    tl.store(RSTD_ptr, rstd)


@triton.autotune(configs=get_autotune_forward_config(), key=['M', 'N', 'K'])
@triton.jit
def _second_forward_kernel(a_ptr, b_ptr, linear_bias_ptr, c_ptr, M, N, K,
    WEIGHT, BIAS, Mean_ptr, RSTD_ptr, stride_am, stride_ak, stride_bk,
    stride_bn, stride_cm, stride_cn, Mean_row_stride, RSTD_row_stride,
    has_layernorm_bias: tl.constexpr, has_linear_bias: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K:
    tl.constexpr, GROUP_SIZE_M: tl.constexpr, DTYPE: tl.constexpr):
    """
    Second kernel: calculate matmul. Normalize each input block loaded  
    a: [M, K] 
    b: [K, N]
    c: [M, N]
    linear_bias: [N]
    """
    pid = tl.program_id(axis=0).to(tl.int64)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        )
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        )
    mean = tl.load(Mean_ptr + offs_m * Mean_row_stride, mask=offs_m < M,
        other=0.0)
    rstd = tl.load(RSTD_ptr + offs_m * RSTD_row_stride, mask=offs_m < M,
        other=0.0)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] <
            K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) &
            (offs_n[None, :] < N), other=0.0)
        weight = tl.load(WEIGHT + k * BLOCK_SIZE_K + offs_k, mask=offs_k < 
            K - k * BLOCK_SIZE_K, other=0.0)
        if has_layernorm_bias:
            bias = tl.load(BIAS + k * BLOCK_SIZE_K + offs_k, mask=offs_k < 
                K - k * BLOCK_SIZE_K, other=0.0)
            a = (a - mean[:, None]) * rstd[:, None] * weight[None, :] + bias[
                None, :]
        else:
            a = (a - mean[:, None]) * rstd[:, None] * weight[None, :]
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(DTYPE)
    if has_linear_bias:
        linear_bias = tl.load(linear_bias_ptr + offs_n, mask=offs_n < N,
            other=0.0)
        c = c + linear_bias[None, :]
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def layernorm_linear_forward(a, b, linear_bias, WEIGHT, BIAS,
    has_layernorm_bias=True, has_linear_bias=False, forward_config=
    DEFAULT_CONFIG):
    shape = a.shape
    a = a.view(-1, shape[-1])
    assert a.shape[1] == b.shape[0], 'Incompatible dimensions'
    assert a.is_contiguous(), 'Matrix A must be contiguous'
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    Mean = torch.empty((M,), dtype=a.dtype, device=a.device)
    RSTD = torch.empty((M,), dtype=a.dtype, device=a.device)
    BLOCK_SIZE, num_warps = calculate_settings(K)
    _first_forward_kernel[M,](a, a.stride(0), Mean, Mean.stride(0), RSTD,
        RSTD.stride(0), K, 1e-05, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv
        (N, META['BLOCK_SIZE_N']),)
    _second_forward_kernel[grid](a, b, linear_bias, c, M, N, K, WEIGHT,
        BIAS, Mean, RSTD, a.stride(0), a.stride(1), b.stride(0), b.stride(1
        ), c.stride(0), c.stride(1), Mean.stride(0), RSTD.stride(0),
        has_layernorm_bias=has_layernorm_bias, has_linear_bias=
        has_linear_bias, DTYPE=tl.float16 if a.dtype == torch.float16 else 
        tl.bfloat16 if a.dtype == torch.bfloat16 else tl.float32)
    c = c.view(shape[:-1] + (N,))
    return c, a, b, Mean, RSTD


# Forward method (kernel launch code)
@ensure_contiguous
@torch.amp.custom_fwd(device_type=infer_device(), cast_inputs=torch.bfloat16)
def _LayernormLinearFunction_forward(ctx, X, linear_weight, linear_bias,
    WEIGHT, BIAS, has_layernorm_bias=True, has_linear_bias=False,
    supported_configs=(DEFAULT_CONFIG, DEFAULT_CONFIG)):
    c, X, linear_weight, Mean, RSTD = layernorm_linear_forward(X,
        linear_weight, linear_bias, WEIGHT, BIAS, has_layernorm_bias,
        has_linear_bias, supported_configs[0])
    ctx.save_for_backward(X, linear_weight, Mean, RSTD, WEIGHT, BIAS)
    ctx.backward_config = supported_configs[1]
    ctx.has_layernorm_bias = has_layernorm_bias
    ctx.has_linear_bias = has_linear_bias
    return c


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=get_autotune_backward_config(), key=['M', 'K', 'N'])
@triton.jit
def _first_backward_kernel(dOUT_ptr, B, dX, dB, X, Mean, RSTD, WEIGHT, BIAS,
    dWEIGHT, dBIAS, c1, c2, M, K, N, stride_am, stride_an, stride_bk,
    stride_bn, stride_cm, stride_ck, stride_weight, stride_c1, stride_c2,
    stride_mean, has_layernorm_bias: tl.constexpr, BLOCK_SIZE_M: tl.
    constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr):
    """ 
    Fuse dY = dOUT @ B^T and store temporary wdy into dX
    Fuse in dB = Y^T @ dOUT via atomic adds
    Calculate dWEIGHT, dBIAS, c1, c2
    """
    pid = tl.program_id(axis=0).to(tl.int64)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
    pid_k = pid % num_pid_in_group // group_size_m
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    dOUT_ptrs = dOUT_ptr + (offs_m[:, None] * stride_am + offs_n[None, :] *
        stride_an)
    B_T_ptrs = B + (offs_k[None, :] * stride_bk + offs_n[:, None] * stride_bn)
    x = tl.load(X + (offs_m[:, None] * stride_cm + offs_k[None, :] *
        stride_ck), mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
        other=0.0)
    mean = tl.load(Mean + offs_m * stride_mean, mask=offs_m < M, other=0.0)
    rstd = tl.load(RSTD + offs_m * stride_mean, mask=offs_m < M, other=0.0)
    x_hat = (x - mean[:, None]) * rstd[:, None]
    weight = tl.load(WEIGHT + offs_k, mask=offs_k < K, other=0.0)
    if has_layernorm_bias:
        bias = tl.load(BIAS + offs_k, mask=offs_k < K, other=0.0)
        y_T = tl.trans(x_hat * weight[None, :] + bias[None, :])
    else:
        y_T = tl.trans(x_hat * weight[None, :])
    dy = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        dOUT = tl.load(dOUT_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None,
            :] < N), other=0.0)
        db = tl.dot(y_T, dOUT)
        tl.atomic_add(dB + offs_k[:, None] * N + offs_n[None, :], db.to(dB.
            type.element_ty), mask=(offs_k[:, None] < K) & (offs_n[None, :] <
            N), sem='relaxed')
        b_T = tl.load(B_T_ptrs, mask=(offs_n[:, None] < N) & (offs_k[None,
            :] < K), other=0.0)
        dy = tl.dot(dOUT, b_T, dy)
        dOUT_ptrs += BLOCK_SIZE_N * stride_an
        B_T_ptrs += BLOCK_SIZE_N * stride_bn
        offs_n += BLOCK_SIZE_N
    wdy = dy * weight[None, :]
    c1_vals = tl.sum(x_hat * wdy, axis=1) / K
    c2_vals = tl.sum(wdy, axis=1) / K
    tl.atomic_add(c1 + offs_m * stride_c1, c1_vals.to(c1.type.element_ty),
        mask=offs_m < M, sem='relaxed')
    tl.atomic_add(c2 + offs_m * stride_c2, c2_vals.to(c2.type.element_ty),
        mask=offs_m < M, sem='relaxed')
    dWEIGHT_vals = tl.sum(dy * x_hat, axis=0)
    if has_layernorm_bias:
        dBIAS_vals = tl.sum(dy, axis=0)
    tl.atomic_add(dWEIGHT + offs_k * stride_weight, dWEIGHT_vals.to(dWEIGHT
        .type.element_ty), mask=offs_k < K, sem='relaxed')
    if has_layernorm_bias:
        tl.atomic_add(dBIAS + offs_k * stride_weight, dBIAS_vals.to(dBIAS.
            type.element_ty), mask=offs_k < K, sem='relaxed')
    tl.store(dX + offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck,
        wdy.to(dX.type.element_ty), mask=(offs_m[:, None] < M) & (offs_k[
        None, :] < K))


@triton.jit
def _second_backward_kernel(dX, X, c1, c2, Mean, RSTD, M, K, N, stride_x,
    stride_mean, BLOCK_SIZE: tl.constexpr):
    """ 
    Calculate dX by wdy, x_hat, c1, c2
    """
    row = tl.program_id(0).to(tl.int64)
    dX += row * stride_x
    X += row * stride_x
    c1 += row * stride_mean
    c2 += row * stride_mean
    Mean += row * stride_mean
    RSTD += row * stride_mean
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < K
    x = tl.load(X + col_offsets, mask=mask, other=0.0)
    mean = tl.load(Mean)
    rstd = tl.load(RSTD)
    x_hat = (x - mean) * rstd
    c1_val = tl.load(c1)
    c2_val = tl.load(c2)
    wdy = tl.load(dX + col_offsets, mask=mask, other=0.0)
    dx = (wdy - (c1_val * x_hat + c2_val)) * rstd
    tl.store(dX + col_offsets, dx, mask=mask)


def layernorm_linear_backward(dOUT, X, B, Mean, RSTD, WEIGHT, BIAS,
    has_layernorm_bias=True, has_linear_bias=False, backward_config=
    DEFAULT_CONFIG):
    dOUT_shape = dOUT.shape
    dOUT = dOUT.view(-1, dOUT_shape[-1])
    M, K = X.shape
    N = dOUT.shape[-1]
    dX = torch.empty((M, K), dtype=X.dtype, device=X.device)
    dB = torch.zeros((K, N), dtype=torch.float32, device=X.device)
    dWEIGHT = torch.zeros((K,), dtype=torch.float32, device=WEIGHT.device)
    dBIAS = None
    if has_layernorm_bias:
        dBIAS = torch.zeros((K,), dtype=torch.float32, device=X.device)
    c1 = torch.zeros((M,), dtype=torch.float32, device=X.device)
    c2 = torch.zeros((M,), dtype=torch.float32, device=X.device)
    _first_backward_kernel[lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M'
        ]) * triton.cdiv(K, META['BLOCK_SIZE_K']),)](dOUT, B, dX, dB, X,
        Mean, RSTD, WEIGHT, BIAS, dWEIGHT, dBIAS, c1, c2, M, K, N, dOUT.
        stride(0), dOUT.stride(1), B.stride(0), B.stride(1), dX.stride(0),
        dX.stride(1), WEIGHT.stride(0), c1.stride(0), c2.stride(0), Mean.
        stride(0), has_layernorm_bias=has_layernorm_bias)
    BLOCK_SIZE, num_warps = calculate_settings(K)
    _second_backward_kernel[M,](dX, X, c1, c2, Mean, RSTD, M, K, N, X.
        stride(0), Mean.stride(0), BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    dLinearBias = None
    if has_linear_bias:
        dLinearBias = dOUT.sum(axis=0)
    dX = dX.view(dOUT_shape[:-1] + (K,))
    dB, dWEIGHT = dB.to(X.dtype), dWEIGHT.to(X.dtype)
    if has_layernorm_bias:
        dBIAS = dBIAS.to(X.dtype)
    return dX, dB, dLinearBias, dWEIGHT, dBIAS


# Backward method (kernel launch code)
@ensure_contiguous
@torch.amp.custom_bwd(device_type=infer_device())
def _LayernormLinearFunction_backward(ctx, dOUT):
    X, linear_weight, Mean, RSTD, WEIGHT, BIAS = ctx.saved_tensors
    dX, dB, dLinearBias, dWEIGHT, dBIAS = layernorm_linear_backward(dOUT, X,
        linear_weight, Mean, RSTD, WEIGHT, BIAS, ctx.has_layernorm_bias,
        ctx.has_linear_bias, ctx.backward_config)
    return dX, dB, dLinearBias, dWEIGHT, dBIAS, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LayernormLinearFunction(torch.autograd.Function):

    @staticmethod
    @ensure_contiguous
    @torch.amp.custom_fwd(device_type=infer_device(), cast_inputs=torch.
        bfloat16)
    def forward(ctx, X, linear_weight, linear_bias, WEIGHT, BIAS,
        has_layernorm_bias=True, has_linear_bias=False, supported_configs=(
        DEFAULT_CONFIG, DEFAULT_CONFIG)):
        c, X, linear_weight, Mean, RSTD = layernorm_linear_forward(X,
            linear_weight, linear_bias, WEIGHT, BIAS, has_layernorm_bias,
            has_linear_bias, supported_configs[0])
        ctx.save_for_backward(X, linear_weight, Mean, RSTD, WEIGHT, BIAS)
        ctx.backward_config = supported_configs[1]
        ctx.has_layernorm_bias = has_layernorm_bias
        ctx.has_linear_bias = has_linear_bias
        return c

    @staticmethod
    @ensure_contiguous
    @torch.amp.custom_bwd(device_type=infer_device())
    def backward(ctx, dOUT):
        X, linear_weight, Mean, RSTD, WEIGHT, BIAS = ctx.saved_tensors
        dX, dB, dLinearBias, dWEIGHT, dBIAS = layernorm_linear_backward(dOUT,
            X, linear_weight, Mean, RSTD, WEIGHT, BIAS, ctx.
            has_layernorm_bias, ctx.has_linear_bias, ctx.backward_config)
        return dX, dB, dLinearBias, dWEIGHT, dBIAS, None, None, None
