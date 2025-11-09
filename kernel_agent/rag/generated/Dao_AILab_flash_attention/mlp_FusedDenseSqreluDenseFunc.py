# SPDX-License-Identifier: BSD-3-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/Dao-AILab/flash-attention
# Source-Files: flash_attn/ops/triton/mlp.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_aqv8vcou/flash-attention-main/flash_attn/ops/triton/mlp.py
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

@torch.jit.script
def sqrelu_fwd(x):
    r = F.relu(x)
    return (r * r).to(dtype=x.dtype)


@triton.jit
def gelu(x):
    """Gaussian Error Linear Unit (GELU)"""
    return x * 0.5 * (1.0 + tl.libdevice.erf(x * _sqrt1_2))


@triton.jit
def gelu_approx(x):
    """
    GeLU_ activation - Gaussian error linear unit, with tanh approximation

    .. _GeLU: https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1.0 + tanh(_sqrt2pi * x * (1.0 + 0.044715 * x * x)))


@triton.jit
def relu(x):
    """
    ReLU_ activation function

    .. _ReLU: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
    """
    zero = 0.0
    return tl.where(x >= 0, x, zero.to(x.dtype))


@triton.jit
def squared_relu(x):
    """
    Squared ReLU activation, as proposed in the Primer_ paper.

    .. _Primer: https://arxiv.org/abs/2109.08668
    """
    x_ = relu(x)
    return (x_ * x_).to(x.dtype)


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.autotune(configs=[triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256,
    'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8), triton.Config
    ({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1},
    num_stages=3, num_warps=8), triton.Config({'BLOCK_M': 256, 'BLOCK_N': 
    64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.
    Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1},
    num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 
    128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.
    Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1},
    num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 
    128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.
    Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1},
    num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32,
    'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=5, num_warps=2), triton.Config
    ({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1},
    num_stages=3, num_warps=8), triton.Config({'BLOCK_M': 256, 'BLOCK_N': 
    128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8), triton.
    Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'SPLIT_K': 1},
    num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 
    256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.
    Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1},
    num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 
    64, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.
    Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 1},
    num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 
    32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.
    Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1},
    num_stages=5, num_warps=2)] + get_configs_io_bound(), key=[
    'CACHE_KEY_M', 'CACHE_KEY_N', 'CACHE_KEY_K'], prune_configs_by={
    'early_config_prune': early_config_prune, 'perf_model':
    estimate_matmul_time, 'top_k': 10})
@triton.heuristics({'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] *
    args['SPLIT_K']) == 0})
@triton.jit
def kernel_fwd(C, ACT_INPUT, A, B, bias, M, N, K, CACHE_KEY_M, CACHE_KEY_N,
    CACHE_KEY_K, stride_cm, stride_am, stride_ak, stride_bn, stride_bk,
    BLOCK_M: tl.constexpr, GROUP_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr,
    A_ROWMAJOR: tl.constexpr, B_COLMAJOR: tl.constexpr, BIAS: tl.constexpr,
    SAVE_ACT_INPUT: tl.constexpr, ACTIVATION: tl.constexpr):
    """
    Kernel for computing Out = activation(A x W + C)
    - Input has shape (M, K)
    - Weight has shape (K, N)
    - Bias has shape (N,)
    - Output has shape (M, N)
    - ActInputs (optional) has shape (M, N)
    'ActInputs' optionally saves the A x W + C intermediate for backward computations
    This kernel will consolidate over K
    """
    pid = tl.program_id(axis=0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    if A_ROWMAJOR:
        A = A + (ram[:, None] * stride_am + rk[None, :])
    else:
        A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    if B_COLMAJOR:
        B = B + (rk[:, None] + rbn[None, :] * stride_bn)
    else:
        B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b)
        if A_ROWMAJOR:
            A += BLOCK_K
        else:
            A += BLOCK_K * stride_ak
        if B_COLMAJOR:
            B += BLOCK_K
        else:
            B += BLOCK_K * stride_bk
    if BIAS:
        bias = tl.load(bias + rn, mask=rn < N, other=0.0).to(tl.float32)
        acc += bias[None, :]
    if SAVE_ACT_INPUT:
        act_in_ptrs = ACT_INPUT + ram[:, None] * stride_cm + rbn[None, :]
        tl.store(act_in_ptrs, acc)
    if ACTIVATION == 'gelu':
        acc = gelu(acc)
    elif ACTIVATION == 'gelu_approx':
        acc = gelu_approx(acc)
    elif ACTIVATION == 'squared_relu':
        acc = squared_relu(acc)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + rm[:, None] * stride_cm + rn[None, :]
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C, acc)


def triton_linear_act(x: torch.Tensor, weight: torch.Tensor, bias: Optional
    [torch.Tensor]=None, activation: str='id', save_act_input: bool=False
    ) ->torch.Tensor:
    """
    Compute e = activation(x @ weight.T + bias).
    This wrapper kicks the `kernel_fwd` Triton kernel
    :param x: input tensor
    :param weight: weight matrix
    :param bias: an optional bias tensor
    :param activation: Activation name. Needs to be a Triton kernel.
    :param act_input: an optional tensor to save the activation inputs (for backward)
    :return: result tensor
    """
    assert activation in ['id', 'gelu', 'gelu_approx', 'squared_relu']
    batch_shape, n = x.shape[:-1], x.shape[-1]
    batch_dim = batch_shape.numel()
    x_reshaped = x.reshape(batch_dim, n)
    if x_reshaped.stride(0) > 1 and x_reshaped.stride(1) > 1:
        x_reshaped = x_reshaped.contiguous()
    if weight.stride(0) > 1 and weight.stride(1) > 1:
        weight = weight.contiguous()
    bias = bias.contiguous() if bias is not None else None
    assert x.dtype == weight.dtype, f'Input and weight must have the same dtype, got {x.dtype} and {weight.dtype}'
    if bias is not None:
        assert x.dtype == bias.dtype, f'Input and bias must have the same dtype, got {x.dtype} and {bias.dtype}'
    assert x_reshaped.shape[1] == weight.shape[1
        ], f'Incompatible dimensions: {x_reshaped.shape} - {weight.shape}'
    assert bias is None or bias.shape[0] == weight.shape[0
        ], 'Incompatible dimensions in between weight and bias'
    M, K = x_reshaped.shape
    N, K = weight.shape
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    act_input = torch.empty_like(output) if save_act_input else None
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N,
        META['BLOCK_N']),)
    kernel_fwd[grid](output, act_input, x_reshaped, weight, bias if bias is not
        None else x, M, N, K, M // 32, N // 32, K // 32, stride_cm=output.
        stride(0), stride_am=x_reshaped.stride(0), stride_ak=x_reshaped.
        stride(1), stride_bk=weight.stride(1), stride_bn=weight.stride(0),
        BIAS=bias is not None, SAVE_ACT_INPUT=save_act_input, ACTIVATION=
        activation, A_ROWMAJOR=x_reshaped.stride(1) == 1, B_COLMAJOR=weight
        .stride(1) == 1, GROUP_M=8)
    if not save_act_input:
        return output.reshape(*batch_shape, output.shape[-1])
    else:
        return output.reshape(*batch_shape, output.shape[-1]
            ), act_input.reshape(*batch_shape, act_input.shape[-1])


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

# Forward method (kernel launch code)
@custom_fwd
def _FusedDenseSqreluDenseFunc_forward(ctx, x, weight1, bias1, weight2,
    bias2, checkpoint_lvl=0):
    """checkpoint_lvl:
        0: no recomputation in the bwd
        1: recompute gelu_out in the bwd
        2: recompute act_input and gelu_out in the bwd
        """
    if torch.is_autocast_enabled():
        dtype = torch.get_autocast_gpu_dtype()
        x, weight1, bias1, weight2, bias2 = [a.to(dtype=dtype) for a in [x,
            weight1, bias1, weight2, bias2]]
    is_bf16 = x.dtype == torch.bfloat16
    assert checkpoint_lvl in [0, 1, 2]
    x = x.contiguous()
    weight1 = weight1.contiguous()
    bias1 = bias1.contiguous()
    weight2 = weight2.contiguous()
    bias2 = bias2.contiguous()
    batch_shape, n = x.shape[:-1], x.shape[-1]
    batch_dim = batch_shape.numel()
    if is_bf16:
        act_input = fused_dense_cuda.linear_bias_forward(x.reshape(
            batch_dim, n), weight1, bias1)
        output1 = sqrelu_fwd(act_input)
    else:
        save_act_input = checkpoint_lvl != 2
        result = triton_linear_act(x.reshape(batch_dim, n), weight1, bias1,
            activation='squared_relu', save_act_input=save_act_input)
        if save_act_input:
            output1, act_input = result
        else:
            output1 = result
    output2 = fused_dense_cuda.linear_bias_forward(output1, weight2, bias2)
    ctx.checkpoint_lvl = checkpoint_lvl
    if checkpoint_lvl == 0:
        ctx.save_for_backward(x, weight1, bias1, weight2, act_input, output1)
    elif checkpoint_lvl == 1:
        ctx.save_for_backward(x, weight1, bias1, weight2, act_input)
    elif checkpoint_lvl == 2:
        ctx.save_for_backward(x, weight1, bias1, weight2)
    return output2.reshape(*batch_shape, output2.shape[-1])


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def gelu_approx_grad(x):
    tanh_out = tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    return 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 
        0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)


@triton.jit
def gelu_grad(x):
    cdf = 0.5 * (1.0 + tl.libdevice.erf(x * _sqrt1_2))
    pdf = tl.exp(-0.5 * x * x) * _gaussian_pdf_normalization
    return cdf + x * pdf


@triton.jit
def squared_relu_grad(x):
    return tl.where(x >= 0, 2.0 * x, 0.0)


@triton.autotune(configs=[triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256,
    'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8), triton.Config
    ({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1},
    num_stages=3, num_warps=8), triton.Config({'BLOCK_M': 256, 'BLOCK_N': 
    64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.
    Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1},
    num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 
    128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.
    Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1},
    num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 
    128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.
    Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1},
    num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32,
    'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=5, num_warps=2), triton.Config
    ({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1},
    num_stages=3, num_warps=8), triton.Config({'BLOCK_M': 256, 'BLOCK_N': 
    128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8), triton.
    Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'SPLIT_K': 1},
    num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 
    256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.
    Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1},
    num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 
    64, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.
    Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 1},
    num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 
    32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.
    Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1},
    num_stages=5, num_warps=2)] + get_configs_io_bound(), key=[
    'CACHE_KEY_M', 'CACHE_KEY_N', 'CACHE_KEY_K'], prune_configs_by={
    'early_config_prune': early_config_prune, 'perf_model':
    estimate_matmul_time, 'top_k': 10})
@triton.heuristics({'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] *
    args['SPLIT_K']) == 0})
@triton.jit
def kernel_bwd(C, ACT_INPUT, A, B, M, N, K, CACHE_KEY_M, CACHE_KEY_N,
    CACHE_KEY_K, stride_cm, stride_am, stride_ak, stride_bk, stride_bn,
    BLOCK_M: tl.constexpr, GROUP_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr,
    ACTIVATION: tl.constexpr):
    """
    Kernel for computing Out = activation(A x W + C)
    - Input has shape (M, K)
    - Weight has shape (K, N)
    - Output has shape (M, N)
    - ActInputs (optional) has shape (M, N)
    'ActInputs' optionally saves the A x W + C intermediate for backward computations
    This kernel will consolidate over K
    """
    pid = tl.program_id(axis=0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    if ACTIVATION != 'id':
        act_in_ptrs = ACT_INPUT + ram[:, None] * stride_cm + rbn[None, :]
        act_input = tl.load(act_in_ptrs).to(acc.dtype)
    if ACTIVATION == 'gelu':
        acc *= gelu_grad(act_input)
    elif ACTIVATION == 'gelu_approx':
        acc *= gelu_approx_grad(act_input)
    elif ACTIVATION == 'squared_relu':
        acc *= squared_relu_grad(act_input)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + rm[:, None] * stride_cm + rn[None, :]
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C, acc, mask=mask)


@torch.jit.script
def sqrelu_bwd(g, x):
    return (2.0 * g * F.relu(x)).to(dtype=x.dtype)


def triton_dgrad_act(grad_output: torch.Tensor, weight: torch.Tensor,
    activation: str='id', act_input: Optional[torch.Tensor]=None
    ) ->torch.Tensor:
    """
    Compute e = activation(grad_output @ weight + bias).
    This wrapper kicks the `kernel_fwd` Triton kernel
    :param grad_output: input tensor
    :param weight: weight matrix
    :param activation: Activation name. Needs to be a Triton kernel.
    :param act_input: an optional tensor to save the activation inputs (for backward)
    :return: result tensor
    """
    assert activation in ['id', 'gelu', 'gelu_approx', 'squared_relu']
    batch_shape, n = grad_output.shape[:-1], grad_output.shape[-1]
    batch_dim = batch_shape.numel()
    grad_output_reshaped = grad_output.reshape(batch_dim, n)
    if grad_output_reshaped.stride(0) > 1 and grad_output_reshaped.stride(1
        ) > 1:
        grad_output_reshaped = grad_output_reshaped.contiguous()
    if weight.stride(0) > 1 and weight.stride(1) > 1:
        weight = weight.contiguous()
    assert grad_output.dtype == weight.dtype, f'grad_output and weight must have the same dtype, got {grad_output.dtype} and {weight.dtype}'
    assert grad_output_reshaped.shape[1] == weight.shape[0
        ], f'Incompatible dimensions: {grad_output_reshaped.shape} - {weight.shape}'
    if activation != 'id':
        assert act_input is not None, f'act_input is required for activation {activation}'
    M, K = grad_output_reshaped.shape
    K, N = weight.shape
    grad_input = torch.empty((M, N), device=grad_output.device, dtype=
        grad_output.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N,
        META['BLOCK_N']),)
    kernel_bwd[grid](grad_input, act_input, grad_output_reshaped, weight, M,
        N, K, M // 32, N // 32, K // 32, stride_cm=grad_input.stride(0),
        stride_am=grad_output_reshaped.stride(0), stride_ak=
        grad_output_reshaped.stride(1), stride_bk=weight.stride(0),
        stride_bn=weight.stride(1), ACTIVATION=activation, GROUP_M=8)
    return grad_input.reshape(*batch_shape, grad_input.shape[-1])


# Backward method (kernel launch code)
@custom_bwd
def _FusedDenseSqreluDenseFunc_backward(ctx, grad_output):
    grad_output = grad_output.contiguous()
    checkpoint_lvl = ctx.checkpoint_lvl
    x, weight1, bias1, weight2, *rest = ctx.saved_tensors
    batch_shape, n = x.shape[:-1], x.shape[-1]
    batch_dim = batch_shape.numel()
    is_bf16 = x.dtype == torch.bfloat16
    if checkpoint_lvl == 0:
        act_input, output1 = rest
    elif checkpoint_lvl == 1:
        act_input, = rest
        output1 = sqrelu_fwd(act_input)
    elif checkpoint_lvl == 2:
        if is_bf16:
            act_input = fused_dense_cuda.linear_bias_forward(x.reshape(
                batch_dim, n), weight1, bias1)
            output1 = sqrelu_fwd(act_input)
        else:
            output1, act_input = triton_linear_act(x.reshape(batch_dim, n),
                weight1, bias1, activation='squared_relu', save_act_input=True)
    if is_bf16:
        grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
        grad_weight2, grad_bias2 = fused_dense_cuda.linear_bias_wgrad(output1,
            grad_output)
        grad_output1 = grad_output @ weight2
        grad_act_input = sqrelu_bwd(grad_output1, act_input)
        grad_input, grad_weight1, grad_bias1 = (fused_dense_cuda.
            linear_bias_backward(x.reshape(batch_dim, n), weight1,
            grad_act_input))
    else:
        grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
        grad_weight2, grad_bias2 = fused_dense_cuda.linear_bias_wgrad(output1,
            grad_output)
        grad_act_input = triton_dgrad_act(grad_output, weight2, activation=
            'squared_relu', act_input=act_input)
        grad_input, grad_weight1, grad_bias1 = (fused_dense_cuda.
            linear_bias_backward(x.reshape(batch_dim, n), weight1,
            grad_act_input))
    return grad_input.reshape_as(x
        ), grad_weight1, grad_bias1, grad_weight2, grad_bias2, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class FusedDenseSqreluDenseFunc(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, x, weight1, bias1, weight2, bias2, checkpoint_lvl=0):
        """checkpoint_lvl:
        0: no recomputation in the bwd
        1: recompute gelu_out in the bwd
        2: recompute act_input and gelu_out in the bwd
        """
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
            x, weight1, bias1, weight2, bias2 = [a.to(dtype=dtype) for a in
                [x, weight1, bias1, weight2, bias2]]
        is_bf16 = x.dtype == torch.bfloat16
        assert checkpoint_lvl in [0, 1, 2]
        x = x.contiguous()
        weight1 = weight1.contiguous()
        bias1 = bias1.contiguous()
        weight2 = weight2.contiguous()
        bias2 = bias2.contiguous()
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = batch_shape.numel()
        if is_bf16:
            act_input = fused_dense_cuda.linear_bias_forward(x.reshape(
                batch_dim, n), weight1, bias1)
            output1 = sqrelu_fwd(act_input)
        else:
            save_act_input = checkpoint_lvl != 2
            result = triton_linear_act(x.reshape(batch_dim, n), weight1,
                bias1, activation='squared_relu', save_act_input=save_act_input
                )
            if save_act_input:
                output1, act_input = result
            else:
                output1 = result
        output2 = fused_dense_cuda.linear_bias_forward(output1, weight2, bias2)
        ctx.checkpoint_lvl = checkpoint_lvl
        if checkpoint_lvl == 0:
            ctx.save_for_backward(x, weight1, bias1, weight2, act_input,
                output1)
        elif checkpoint_lvl == 1:
            ctx.save_for_backward(x, weight1, bias1, weight2, act_input)
        elif checkpoint_lvl == 2:
            ctx.save_for_backward(x, weight1, bias1, weight2)
        return output2.reshape(*batch_shape, output2.shape[-1])

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        checkpoint_lvl = ctx.checkpoint_lvl
        x, weight1, bias1, weight2, *rest = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = batch_shape.numel()
        is_bf16 = x.dtype == torch.bfloat16
        if checkpoint_lvl == 0:
            act_input, output1 = rest
        elif checkpoint_lvl == 1:
            act_input, = rest
            output1 = sqrelu_fwd(act_input)
        elif checkpoint_lvl == 2:
            if is_bf16:
                act_input = fused_dense_cuda.linear_bias_forward(x.reshape(
                    batch_dim, n), weight1, bias1)
                output1 = sqrelu_fwd(act_input)
            else:
                output1, act_input = triton_linear_act(x.reshape(batch_dim,
                    n), weight1, bias1, activation='squared_relu',
                    save_act_input=True)
        if is_bf16:
            grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
            grad_weight2, grad_bias2 = fused_dense_cuda.linear_bias_wgrad(
                output1, grad_output)
            grad_output1 = grad_output @ weight2
            grad_act_input = sqrelu_bwd(grad_output1, act_input)
            grad_input, grad_weight1, grad_bias1 = (fused_dense_cuda.
                linear_bias_backward(x.reshape(batch_dim, n), weight1,
                grad_act_input))
        else:
            grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
            grad_weight2, grad_bias2 = fused_dense_cuda.linear_bias_wgrad(
                output1, grad_output)
            grad_act_input = triton_dgrad_act(grad_output, weight2,
                activation='squared_relu', act_input=act_input)
            grad_input, grad_weight1, grad_bias1 = (fused_dense_cuda.
                linear_bias_backward(x.reshape(batch_dim, n), weight1,
                grad_act_input))
        return grad_input.reshape_as(x
            ), grad_weight1, grad_bias1, grad_weight2, grad_bias2, None
