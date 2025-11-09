# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/astramind-ai/BitMat
# Source-Files: bitmat/utils/bitmat.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_1p1aj4ky/BitMat-main/bitmat/utils/bitmat.py
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

@custom_autotune.autotune(configs=[triton.Config({'BLOCK_SIZE_M': 16,
    'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=
    1, num_warps=4), triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32,
    'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=1, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 
    32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({
    'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,
    'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({
    'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,
    'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({
    'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32,
    'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({
    'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32,
    'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4)], key=['M', 'N', 'K'],
    nearest_power_of_two=True, prune_configs_by={'early_config_prune':
    custom_autotune.kernel_config_pruner, 'perf_model': None, 'top_k': None})
@triton.jit
def _ternary_mm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, n_bits, stride_am,
    stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: tl
    .constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, ACTIVATION: tl.constexpr):
    """Kernel for computing the matmul C = A x B.
        A has shape (M, K), int8
        B has shape (K//n_bits, N), int8, packed
        C has shape (M, N),
        """
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
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] *
        stride_ak)
    a_mask = offs_am[:, None] < M
    b_ptrs = b_ptr + (offs_k[:, None] // n_bits * stride_bk + offs_bn[None,
        :] * stride_bn)
    c_dtype = tl.load(c_ptr + stride_cm).dtype
    shifter = (offs_k % n_bits)[:, None] * 2
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_pid_k):
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs)
        b = b >> shifter & 3
        b = tl.where(b == 2, -1, b)
        b = b.to(a.dtype)
        accumulator += tl.dot(a, b, out_dtype=c_dtype)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K // n_bits * stride_bk
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :
        ]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def bitmat_(a, b, int_per_2_bits=4, activation='', out_dtype=torch.float16):
    """
        a: int8 tensor (..., K)
        b: int8 packed tensor (K // int_per_2_bit, N)
        c: float16 tensor (..., N)
        n_bits: int, number of bits that each element in b represents
    """
    assert a.shape[-1] == b.shape[-2
        ] * int_per_2_bits, 'Incompatible dimensions'
    assert a.is_contiguous(), 'A must be contiguous'
    assert b.is_contiguous(), 'B must be contiguous'
    assert int_per_2_bits in [4, 8, 16, 32], 'n_bits must be 4, 8, 16, 32'
    x = a.view(-1, a.shape[-1])
    M, K = x.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=out_dtype).contiguous()
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv
        (N, META['BLOCK_SIZE_N']),)
    _ternary_mm_kernel[grid](x, b, c, M, N, K, int_per_2_bits, x.stride(0),
        x.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        ACTIVATION=activation)
    c = c.view(a.shape[:-1] + (N,))
    return c


def quantize_activations(x: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the activations and returns the scale for each row.
    """
    dtype = x.dtype
    scale = (127 / torch.max(x.abs().max(dim=-1).values, torch.tensor(1e-05))
        ).unsqueeze(-1)
    return torch.clamp((x * scale).round(), -127, 128).to(torch.int8
        ), scale.to(dtype)


def terniarize(weights: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Terniarizes the weights and returns the scale.
    """
    dtype = weights.dtype
    scale = 1 / torch.max(weights.abs().mean(), torch.tensor(1e-05))
    return torch.clamp((weights * scale).round().to(torch.int8), -1, 1
        ), scale.to(dtype)


# Forward method (kernel launch code)
@torch.cuda.amp.custom_fwd
def _BitMat_forward(ctx, W, X, scale_w=None):
    """
        During the forward pass, we ternarize the weights, pack them and then quantize the activations.
        We then perform the bit matrix multiplication and return the scaled results.
        ternarization:
        scale_w = 1 / mean(abs(W))                              | STE
        W = clip(round(W * scale_w), -1, 1)                     | STE
        packing:
        packed_w = 4 int8 -> 1 int8                             | STE
        quantization:
        scale_x = 127 / max(abs(X))                             | STE
        X = clip(round(X * scale_x), -127, 128)                 | STE
        bit matrix multiplication:
        Y = X @ w_packed.t()                                    | dot product
        Y = Y / scale_w / scale_x)                              | STE
        """
    if scale_w is None:
        dtype = W.dtype
        W, scale_w = terniarize(W)
        ctx.save_for_backward(X)
        X, scale_x = quantize_activations(X)
        y = X.to(dtype) @ W.to(dtype).t()
        return y / scale_w / scale_x
    else:
        X, scale_x = quantize_activations(X)
        y = bitmat_(X, W.t().contiguous())
        return y / scale_w / scale_x


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

# Backward method (kernel launch code)
@torch.cuda.amp.custom_bwd
def _BitMat_backward(ctx, grad_output):
    X = ctx.saved_tensors[0]
    grad_W = (grad_output.transpose(1, 2) @ X).mean(dim=0)
    return grad_W, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class BitMat(torch.autograd.Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, W, X, scale_w=None):
        """
        During the forward pass, we ternarize the weights, pack them and then quantize the activations.
        We then perform the bit matrix multiplication and return the scaled results.
        ternarization:
        scale_w = 1 / mean(abs(W))                              | STE
        W = clip(round(W * scale_w), -1, 1)                     | STE
        packing:
        packed_w = 4 int8 -> 1 int8                             | STE
        quantization:
        scale_x = 127 / max(abs(X))                             | STE
        X = clip(round(X * scale_x), -127, 128)                 | STE
        bit matrix multiplication:
        Y = X @ w_packed.t()                                    | dot product
        Y = Y / scale_w / scale_x)                              | STE
        """
        if scale_w is None:
            dtype = W.dtype
            W, scale_w = terniarize(W)
            ctx.save_for_backward(X)
            X, scale_x = quantize_activations(X)
            y = X.to(dtype) @ W.to(dtype).t()
            return y / scale_w / scale_x
        else:
            X, scale_x = quantize_activations(X)
            y = bitmat_(X, W.t().contiguous())
            return y / scale_w / scale_x

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        X = ctx.saved_tensors[0]
        grad_W = (grad_output.transpose(1, 2) @ X).mean(dim=0)
        return grad_W, None, None
