# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/JonasGeiping/linear_cross_entropy_loss
# Source-Files: variants/compile_tests_NV5.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_0e62yj5n/linear_cross_entropy_loss-main/variants/compile_tests_NV5.py
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
from triton import cdiv
from math import exp
from math import log

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=[triton.Config({'V_BLOCK_SIZE': 64, 'N_BLOCK_SIZE':
    64, 'H_BLOCK_SIZE': 64, 'V_TILES': 1}), triton.Config({'V_BLOCK_SIZE': 
    128, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES': 1}), triton.
    Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64,
    'V_TILES': 1}), triton.Config({'V_BLOCK_SIZE': 512, 'N_BLOCK_SIZE': 64,
    'H_BLOCK_SIZE': 64, 'V_TILES': 1}), triton.Config({'V_BLOCK_SIZE': 64,
    'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES': 1}), triton.Config({
    'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 128, 'V_TILES':
    1}), triton.Config({'V_BLOCK_SIZE': 64, 'N_BLOCK_SIZE': 64,
    'H_BLOCK_SIZE': 64, 'V_TILES': 1}, num_warps=4), triton.Config({
    'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES':
    1}, num_warps=4), triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 
    64, 'H_BLOCK_SIZE': 64, 'V_TILES': 1}, num_warps=4), triton.Config({
    'V_BLOCK_SIZE': 512, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES':
    1}, num_warps=4), triton.Config({'V_BLOCK_SIZE': 64, 'N_BLOCK_SIZE': 64,
    'H_BLOCK_SIZE': 64, 'V_TILES': 1}, num_warps=4), triton.Config({
    'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 128, 'V_TILES':
    1}, num_warps=4), triton.Config({'V_BLOCK_SIZE': 64, 'N_BLOCK_SIZE': 64,
    'H_BLOCK_SIZE': 64, 'V_TILES': 1}, num_warps=8), triton.Config({
    'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES':
    1}, num_warps=8), triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 
    64, 'H_BLOCK_SIZE': 64, 'V_TILES': 1}, num_warps=8), triton.Config({
    'V_BLOCK_SIZE': 64, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES': 
    1}, num_warps=8), triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 
    64, 'H_BLOCK_SIZE': 128, 'V_TILES': 1}, num_warps=8), triton.Config({
    'V_BLOCK_SIZE': 512, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES':
    1}, num_warps=8), triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 
    64, 'H_BLOCK_SIZE': 64, 'V_TILES': 1}, num_warps=8, num_stages=3),
    triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE':
    128, 'V_TILES': 1}, num_warps=8, num_stages=3), triton.Config({
    'V_BLOCK_SIZE': 512, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES':
    1}, num_warps=8, num_stages=3), triton.Config({'V_BLOCK_SIZE': 256,
    'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES': 1}, num_warps=8,
    num_stages=4), triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64,
    'H_BLOCK_SIZE': 128, 'V_TILES': 1}, num_warps=8, num_stages=4), triton.
    Config({'V_BLOCK_SIZE': 512, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64,
    'V_TILES': 1}, num_warps=8, num_stages=4), triton.Config({
    'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES':
    1}, num_warps=16), triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 
    64, 'H_BLOCK_SIZE': 128, 'V_TILES': 1}, num_warps=16), triton.Config({
    'V_BLOCK_SIZE': 512, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES':
    1}, num_warps=16)], key=['V', 'N', 'H'])
@triton.jit
def linear_xent_fwd_prep_bwd_kernel_matmul_t(x_ptr, y_ptr, A_t_ptr,
    z_nv_ptr, losses_ptr, lse_ptr, stride_x_N, stride_x_H, stride_A_H,
    stride_A_V, stride_z_N, stride_z_V, stride_lse_N, stride_lse_B,
    stride_loss_Nb, stride_loss_B, idx_N_group, N_group: tl.constexpr, V:
    tl.constexpr, N: tl.constexpr, H: tl.constexpr, V_BLOCK_SIZE: tl.
    constexpr, N_BLOCK_SIZE: tl.constexpr, H_BLOCK_SIZE: tl.constexpr,
    V_TILES: tl.constexpr=1):
    idx_N = tl.program_id(axis=0)
    idx_V_group = tl.program_id(axis=1)
    V_GROUP_SIZE: tl.constexpr = V_TILES * V_BLOCK_SIZE
    x_block_ptr = tl.make_block_ptr(base=x_ptr, shape=(N, H), strides=(
        stride_x_N, stride_x_H), offsets=(idx_N_group * N_group + idx_N *
        N_BLOCK_SIZE, 0), block_shape=(N_BLOCK_SIZE, H_BLOCK_SIZE), order=(
        1, 0))
    A_block_ptr = tl.make_block_ptr(base=A_t_ptr, shape=(H, V), strides=(
        stride_A_H, stride_A_V), offsets=(0, idx_V_group * V_GROUP_SIZE),
        block_shape=(H_BLOCK_SIZE, V_BLOCK_SIZE), order=(1, 0))
    z_block_ptr = tl.make_block_ptr(base=z_nv_ptr, shape=(N_group, V),
        strides=(stride_z_N, stride_z_V), offsets=(idx_N * N_BLOCK_SIZE, 
        idx_V_group * V_GROUP_SIZE), block_shape=(N_BLOCK_SIZE,
        V_BLOCK_SIZE), order=(1, 0))
    lse_row_ptr = tl.make_block_ptr(base=lse_ptr, shape=(N, V // 64),
        strides=(stride_lse_N, stride_lse_B), offsets=(idx_N_group *
        N_group + idx_N * N_BLOCK_SIZE, idx_V_group), block_shape=(
        N_BLOCK_SIZE, 1), order=(1, 0))
    loss_val_ptr = losses_ptr + (idx_N + idx_N_group * N_group // N_BLOCK_SIZE
        ) * stride_loss_Nb + idx_V_group * stride_loss_B
    N_range = idx_N_group * N_group + idx_N * N_BLOCK_SIZE + tl.arange(0,
        N_BLOCK_SIZE)
    V_range = idx_V_group * V_GROUP_SIZE + tl.arange(0, V_BLOCK_SIZE)
    y = tl.load(y_ptr + N_range)
    m = tl.zeros((N_BLOCK_SIZE,), dtype=tl.float32) - float(10000000.0)
    s = tl.zeros((N_BLOCK_SIZE,), dtype=tl.float32)
    loss = 0.0
    for _ in range(V_TILES):
        z_j_to_k = tl.zeros((N_BLOCK_SIZE, V_BLOCK_SIZE), dtype=tl.float32)
        for _ in range(H // H_BLOCK_SIZE):
            x_chunk = tl.load(x_block_ptr)
            A_v = tl.load(A_block_ptr)
            z_j_to_k = tl.dot(x_chunk, A_v, z_j_to_k)
            x_block_ptr = tl.advance(x_block_ptr, [0, H_BLOCK_SIZE])
            A_block_ptr = tl.advance(A_block_ptr, [H_BLOCK_SIZE, 0])
        m_new = tl.maximum(m, tl.max(z_j_to_k, 1))
        s_update = tl.sum(tl.exp(z_j_to_k - m_new[:, None]), axis=1)
        s = s * tl.exp(m - m_new) + s_update
        mask = y[:, None] == V_range[None, :]
        loss -= tl.sum(tl.where(mask, z_j_to_k, float(0.0))) / N
        tl.store(z_block_ptr, z_j_to_k.to(z_nv_ptr.type.element_ty))
        m = m_new
        x_block_ptr = tl.advance(x_block_ptr, [0, -H])
        A_block_ptr = tl.advance(A_block_ptr, [-H, V_BLOCK_SIZE])
        z_block_ptr = tl.advance(z_block_ptr, [0, V_BLOCK_SIZE])
        V_range = V_range + V_BLOCK_SIZE
    lse = m + tl.log(s)
    tl.store(loss_val_ptr, loss)
    tl.store(lse_row_ptr, lse[:, None])


@triton.autotune(configs=[triton.Config({'V_BLOCK_SIZE': 64, 'N_BLOCK_SIZE':
    16}), triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 16}), triton.
    Config({'V_BLOCK_SIZE': 512, 'N_BLOCK_SIZE': 16}), triton.Config({
    'V_BLOCK_SIZE': 1024, 'N_BLOCK_SIZE': 16}), triton.Config({
    'V_BLOCK_SIZE': 64, 'N_BLOCK_SIZE': 64}), triton.Config({'V_BLOCK_SIZE':
    256, 'N_BLOCK_SIZE': 64}), triton.Config({'V_BLOCK_SIZE': 256,
    'N_BLOCK_SIZE': 128}), triton.Config({'V_BLOCK_SIZE': 64,
    'N_BLOCK_SIZE': 16}, num_warps=8), triton.Config({'V_BLOCK_SIZE': 256,
    'N_BLOCK_SIZE': 16}, num_warps=8), triton.Config({'V_BLOCK_SIZE': 512,
    'N_BLOCK_SIZE': 16}, num_warps=8), triton.Config({'V_BLOCK_SIZE': 1024,
    'N_BLOCK_SIZE': 16}, num_warps=8), triton.Config({'V_BLOCK_SIZE': 64,
    'N_BLOCK_SIZE': 64}, num_warps=8), triton.Config({'V_BLOCK_SIZE': 256,
    'N_BLOCK_SIZE': 64}, num_warps=8), triton.Config({'V_BLOCK_SIZE': 256,
    'N_BLOCK_SIZE': 128}, num_warps=8), triton.Config({'V_BLOCK_SIZE': 64,
    'N_BLOCK_SIZE': 16}, num_warps=16), triton.Config({'V_BLOCK_SIZE': 256,
    'N_BLOCK_SIZE': 16}, num_warps=16), triton.Config({'V_BLOCK_SIZE': 512,
    'N_BLOCK_SIZE': 16}, num_warps=16), triton.Config({'V_BLOCK_SIZE': 1024,
    'N_BLOCK_SIZE': 16}, num_warps=16), triton.Config({'V_BLOCK_SIZE': 64,
    'N_BLOCK_SIZE': 64}, num_warps=16), triton.Config({'V_BLOCK_SIZE': 256,
    'N_BLOCK_SIZE': 64}, num_warps=16), triton.Config({'V_BLOCK_SIZE': 256,
    'N_BLOCK_SIZE': 128}, num_warps=16)], key=['V', 'N'])
@triton.jit
def linear_xent_mini_bwd_prologue_kernel(z_nv_ptr, y_ptr, lse_ptr,
    stride_z_N, stride_z_V, idx_N_group, N_group: tl.constexpr, V: tl.
    constexpr, N: tl.constexpr, V_BLOCK_SIZE: tl.constexpr, N_BLOCK_SIZE:
    tl.constexpr):
    idx_N = tl.program_id(axis=0)
    idx_V = tl.program_id(axis=1)
    z_block_ptr = tl.make_block_ptr(base=z_nv_ptr, shape=(N_group, V),
        strides=(stride_z_N, stride_z_V), offsets=(idx_N * N_BLOCK_SIZE, 
        idx_V * V_BLOCK_SIZE), block_shape=(N_BLOCK_SIZE, V_BLOCK_SIZE),
        order=(1, 0))
    N_range = idx_N_group * N_group + idx_N * N_BLOCK_SIZE + tl.arange(0,
        N_BLOCK_SIZE)
    v_range = idx_V * V_BLOCK_SIZE + tl.arange(0, V_BLOCK_SIZE)
    y = tl.load(y_ptr + N_range)
    lse = tl.load(lse_ptr + N_range)
    z_j_to_k = tl.load(z_block_ptr)
    mask = y[:, None] == v_range[None, :]
    softmax_z = (z_j_to_k - lse[:, None]).exp()
    z_grad = (softmax_z - tl.where(mask, 1.0, 0.0)) / N
    tl.store(z_block_ptr, z_grad.to(z_nv_ptr.type.element_ty))


# Forward method (kernel launch code)
def _LinearCrossEntropyLoss_forward(ctx, x, y, At, ignore_index=-100,
    N_chunk_size: int=4096):
    N, H = x.shape
    H_A, V = At.shape
    assert H_A == H
    assert y.shape == (N,)
    if ignore_index >= 0:
        y[y == ignore_index] = -100
    At_grad = torch.zeros_like(At)
    x_grad = torch.zeros_like(x)
    N_group = min(N, N_chunk_size)
    lse_local = -1000000.0 * torch.ones(N, V // 64, dtype=torch.float32,
        device=x.device)
    losses = torch.zeros(N // 64, V // 64, dtype=torch.float32, device=x.device
        )
    z_nv_and_grad = torch.empty((N_group, V), device=x.device, dtype=torch.
        float32)
    fwd_grid = lambda meta: (triton.cdiv(N_group, meta['N_BLOCK_SIZE']),
        triton.cdiv(V, meta['V_BLOCK_SIZE'] * meta['V_TILES']))
    prologue_grid = lambda meta: (triton.cdiv(N_group, meta['N_BLOCK_SIZE']
        ), triton.cdiv(V, meta['V_BLOCK_SIZE']))
    for idx_N_group, x_n_chunk in enumerate(x.split(N_group)):
        linear_xent_fwd_prep_bwd_kernel_matmul_t[fwd_grid](x, y, At,
            z_nv_and_grad, losses, lse_local, x.stride(0), x.stride(1), At.
            stride(0), At.stride(1), z_nv_and_grad.stride(0), z_nv_and_grad
            .stride(1), lse_local.stride(0), lse_local.stride(1), losses.
            stride(0), losses.stride(1), idx_N_group=idx_N_group, N_group=
            N_group, V=V, N=N, H=H)
        lse_global = lse_local.logsumexp(dim=1)
        if x.requires_grad or At.requires_grad:
            linear_xent_mini_bwd_prologue_kernel[prologue_grid](z_nv_and_grad,
                y, lse_global, z_nv_and_grad.stride(0), z_nv_and_grad.
                stride(1), idx_N_group=idx_N_group, N_group=N_group, V=V, N=N)
            z_grad = z_nv_and_grad.to(x.dtype)
        if At.requires_grad:
            torch.addmm(At_grad, x_n_chunk.detach().T, z_grad, out=At_grad)
    ctx.mark_non_differentiable(y)
    ctx.save_for_backward(x_grad, At_grad.to(At.dtype))
    return losses.sum() + lse_global.sum() / N


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

# Backward method (kernel launch code)
def _LinearCrossEntropyLoss_backward(ctx, grad_output):
    x_grad, At_grad = ctx.saved_tensors
    return x_grad * grad_output, None, At_grad * grad_output, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LinearCrossEntropyLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, y, At, ignore_index=-100, N_chunk_size: int=4096):
        N, H = x.shape
        H_A, V = At.shape
        assert H_A == H
        assert y.shape == (N,)
        if ignore_index >= 0:
            y[y == ignore_index] = -100
        At_grad = torch.zeros_like(At)
        x_grad = torch.zeros_like(x)
        N_group = min(N, N_chunk_size)
        lse_local = -1000000.0 * torch.ones(N, V // 64, dtype=torch.float32,
            device=x.device)
        losses = torch.zeros(N // 64, V // 64, dtype=torch.float32, device=
            x.device)
        z_nv_and_grad = torch.empty((N_group, V), device=x.device, dtype=
            torch.float32)
        fwd_grid = lambda meta: (triton.cdiv(N_group, meta['N_BLOCK_SIZE']),
            triton.cdiv(V, meta['V_BLOCK_SIZE'] * meta['V_TILES']))
        prologue_grid = lambda meta: (triton.cdiv(N_group, meta[
            'N_BLOCK_SIZE']), triton.cdiv(V, meta['V_BLOCK_SIZE']))
        for idx_N_group, x_n_chunk in enumerate(x.split(N_group)):
            linear_xent_fwd_prep_bwd_kernel_matmul_t[fwd_grid](x, y, At,
                z_nv_and_grad, losses, lse_local, x.stride(0), x.stride(1),
                At.stride(0), At.stride(1), z_nv_and_grad.stride(0),
                z_nv_and_grad.stride(1), lse_local.stride(0), lse_local.
                stride(1), losses.stride(0), losses.stride(1), idx_N_group=
                idx_N_group, N_group=N_group, V=V, N=N, H=H)
            lse_global = lse_local.logsumexp(dim=1)
            if x.requires_grad or At.requires_grad:
                linear_xent_mini_bwd_prologue_kernel[prologue_grid](
                    z_nv_and_grad, y, lse_global, z_nv_and_grad.stride(0),
                    z_nv_and_grad.stride(1), idx_N_group=idx_N_group,
                    N_group=N_group, V=V, N=N)
                z_grad = z_nv_and_grad.to(x.dtype)
            if At.requires_grad:
                torch.addmm(At_grad, x_n_chunk.detach().T, z_grad, out=At_grad)
        ctx.mark_non_differentiable(y)
        ctx.save_for_backward(x_grad, At_grad.to(At.dtype))
        return losses.sum() + lse_global.sum() / N

    @staticmethod
    def backward(ctx, grad_output):
        x_grad, At_grad = ctx.saved_tensors
        return x_grad * grad_output, None, At_grad * grad_output, None, None
