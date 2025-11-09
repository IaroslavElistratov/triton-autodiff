# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/JonasGeiping/linear_cross_entropy_loss
# Source-Files: variants/double_recomp_thinboy.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_0e62yj5n/linear_cross_entropy_loss-main/variants/double_recomp_thinboy.py
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

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=[triton.Config({'V_BLOCK_SIZE': 256,
    'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64})], key=['V', 'N', 'H'],
    reset_to_zero=['losses_ptr', 'lse_ptr'])
@triton.jit
def linear_xent_fwd_kernel_matmul_t(x_ptr, y_ptr, A_t_ptr, losses_ptr,
    lse_ptr, stride_x_N, stride_x_H, stride_A_H, stride_A_V, V: tl.
    constexpr, N: tl.constexpr, H: tl.constexpr, V_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr, H_BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(axis=0)
    tl.static_print(V_BLOCK_SIZE, N_BLOCK_SIZE, H_BLOCK_SIZE)
    x_block_ptr = tl.make_block_ptr(base=x_ptr, shape=(N, H), strides=(
        stride_x_N, stride_x_H), offsets=(idx * N_BLOCK_SIZE, 0),
        block_shape=(N_BLOCK_SIZE, H_BLOCK_SIZE), order=(1, 0))
    A_block_ptr = tl.make_block_ptr(base=A_t_ptr, shape=(H, V), strides=(
        stride_A_H, stride_A_V), offsets=(0, 0), block_shape=(H_BLOCK_SIZE,
        V_BLOCK_SIZE), order=(1, 0))
    offsets = idx * N_BLOCK_SIZE + tl.arange(0, N_BLOCK_SIZE)
    v_range = tl.arange(0, V_BLOCK_SIZE)
    y = tl.load(y_ptr + offsets)
    m = tl.zeros((N_BLOCK_SIZE,), dtype=tl.float32) - float(1000000.0)
    s = tl.zeros((N_BLOCK_SIZE,), dtype=tl.float32)
    loss = 0.0
    for _ in range(V // V_BLOCK_SIZE):
        z_j_to_k = tl.zeros((N_BLOCK_SIZE, V_BLOCK_SIZE), dtype=tl.float32)
        local_x_block_ptr = x_block_ptr
        for _ in range(H // H_BLOCK_SIZE):
            x_chunk = tl.load(local_x_block_ptr)
            A_v = tl.load(A_block_ptr)
            z_j_to_k = tl.dot(x_chunk, A_v, z_j_to_k)
            local_x_block_ptr = tl.advance(local_x_block_ptr, [0, H_BLOCK_SIZE]
                )
            A_block_ptr = tl.advance(A_block_ptr, [H_BLOCK_SIZE, 0])
        m_new = tl.maximum(m, tl.max(z_j_to_k, 1))
        s_update = tl.sum(tl.exp(z_j_to_k - m_new[:, None]), axis=1)
        s = s * tl.exp(m - m_new) + s_update
        mask = y[:, None] == v_range[None, :]
        loss -= tl.sum(tl.where(mask, z_j_to_k, float(0.0))) / N
        m = m_new
        A_block_ptr = tl.advance(A_block_ptr, [-H_BLOCK_SIZE * (H //
            H_BLOCK_SIZE), V_BLOCK_SIZE])
        v_range = v_range + V_BLOCK_SIZE
    lse = m + tl.log(s)
    loss += tl.sum(lse) / N
    tl.store(losses_ptr + idx, loss)
    tl.store(lse_ptr + offsets, lse)


# Forward method (kernel launch code)
def _LinearCrossEntropyLoss_forward(ctx, x, y, At, ignore_index=-100):
    N, H = x.shape
    H_A, V = At.shape
    assert H_A == H
    assert y.shape == (N,)
    x = x.contiguous()
    y = y.contiguous()
    At = At.contiguous()
    assert V % 256 == 0, f'V is {V}'
    assert N % 64 == 0, f'N is {N}'
    assert H % 64 == 0, f'H is {H}'
    lse_global = torch.zeros(N, dtype=torch.float32, device=x.device)
    losses = torch.zeros(N // 16, dtype=torch.float32, device=x.device)
    grid = lambda meta: (triton.cdiv(N, meta['N_BLOCK_SIZE']),)
    with torch.cuda.device(x.device.index):
        linear_xent_fwd_kernel_matmul_t[grid](x, y, At, losses, lse_global,
            x.stride(0), x.stride(1), At.stride(0), At.stride(1), V=V, N=N, H=H
            )
    print('fwd config:', linear_xent_fwd_kernel_matmul_t.best_config)
    ctx.save_for_backward(x, y, At, lse_global)
    return losses.sum()


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=[triton.Config({'V_BLOCK_SIZE': 16, 'N_BLOCK_SIZE':
    4})], key=['V', 'N', 'H'], reset_to_zero=['x_grad_ptr'])
@triton.jit()
def linear_xent_bwd_kernel_matmul_t_dx(x_ptr, y_ptr, A_t_ptr,
    lse_global_ptr, x_grad_ptr, stride_x_N, stride_x_H, stride_A_H,
    stride_A_V, V: tl.constexpr, N: tl.constexpr, H: tl.constexpr,
    V_BLOCK_SIZE: tl.constexpr=16, N_BLOCK_SIZE: tl.constexpr=1,
    H_BLOCK_SIZE: tl.constexpr=16):
    idx_N = tl.program_id(axis=0)
    tl.static_print(V_BLOCK_SIZE, N_BLOCK_SIZE)
    N_offsets = idx_N * N_BLOCK_SIZE + tl.arange(0, N_BLOCK_SIZE)
    V_offsets = tl.arange(0, V_BLOCK_SIZE)
    y = tl.load(y_ptr + N_offsets)
    lse = tl.load(lse_global_ptr + N_offsets)
    x_grad_slice = tl.zeros((N_BLOCK_SIZE, H), tl.float16)
    for idx_V in range(V // V_BLOCK_SIZE):
        z_j_to_k = tl.zeros((N_BLOCK_SIZE, V_BLOCK_SIZE), dtype=tl.float32)
        x_block_ptr = tl.make_block_ptr(base=x_ptr, shape=(N, H), strides=(
            stride_x_N, stride_x_H), offsets=(idx_N * N_BLOCK_SIZE, 0),
            block_shape=(N_BLOCK_SIZE, H), order=(1, 0))
        A_slice_ptr = tl.make_block_ptr(base=A_t_ptr, shape=(H, V), strides
            =(stride_A_H, stride_A_V), offsets=(0, idx_V * V_BLOCK_SIZE),
            block_shape=(H, V_BLOCK_SIZE), order=(1, 0))
        x_chunk = tl.load(x_block_ptr)
        A_v_full = tl.load(A_slice_ptr)
        z_j_to_k = tl.sum(x_chunk[:, :, None] * A_v_full[None, :, :], axis=1
            ).to(tl.float32)
        mask = (y[:, None] == V_offsets[None, :])[:, :, None]
        softmax_z = (z_j_to_k - lse[:, None]).exp().to(tl.float16)
        temp_xgrad = tl.sum(softmax_z[:, :, None] * A_v_full.trans()[None,
            :, :], axis=1) / N
        temp_xgrad -= tl.sum(tl.where(mask, A_v_full.trans()[None, :, :], 
            0.0), axis=1) / N
        temp_xgrad = temp_xgrad.to(tl.float16)
        x_grad_slice += temp_xgrad
        V_offsets += V_BLOCK_SIZE
    x_grad_block_ptr = tl.make_block_ptr(base=x_grad_ptr, shape=(N, H),
        strides=(stride_x_N, stride_x_H), offsets=(idx_N * N_BLOCK_SIZE, 0),
        block_shape=(N_BLOCK_SIZE, H), order=(1, 0))
    tl.store(x_grad_block_ptr, x_grad_slice)


# Backward method (kernel launch code)
def _LinearCrossEntropyLoss_backward(ctx, grad_output):
    x, y, At, lse_global = ctx.saved_tensors
    N, H = x.shape
    _, V = At.shape
    xgrad = torch.zeros_like(x)
    Atgrad = torch.zeros_like(At)
    with torch.cuda.device(x.device.index):
        grid = lambda meta: (triton.cdiv(N, meta['N_BLOCK_SIZE']),)
        linear_xent_bwd_kernel_matmul_t_dx[grid](x, y, At, lse_global,
            xgrad, x.stride(0), x.stride(1), At.stride(0), At.stride(1), V=
            V, N=N, H=H)
        print('bwd config dx:', linear_xent_bwd_kernel_matmul_t_dx.best_config)
    ctx.mark_non_differentiable(y)
    return xgrad * grad_output, None, Atgrad * grad_output, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LinearCrossEntropyLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, y, At, ignore_index=-100):
        N, H = x.shape
        H_A, V = At.shape
        assert H_A == H
        assert y.shape == (N,)
        x = x.contiguous()
        y = y.contiguous()
        At = At.contiguous()
        assert V % 256 == 0, f'V is {V}'
        assert N % 64 == 0, f'N is {N}'
        assert H % 64 == 0, f'H is {H}'
        lse_global = torch.zeros(N, dtype=torch.float32, device=x.device)
        losses = torch.zeros(N // 16, dtype=torch.float32, device=x.device)
        grid = lambda meta: (triton.cdiv(N, meta['N_BLOCK_SIZE']),)
        with torch.cuda.device(x.device.index):
            linear_xent_fwd_kernel_matmul_t[grid](x, y, At, losses,
                lse_global, x.stride(0), x.stride(1), At.stride(0), At.
                stride(1), V=V, N=N, H=H)
        print('fwd config:', linear_xent_fwd_kernel_matmul_t.best_config)
        ctx.save_for_backward(x, y, At, lse_global)
        return losses.sum()

    @staticmethod
    def backward(ctx, grad_output):
        x, y, At, lse_global = ctx.saved_tensors
        N, H = x.shape
        _, V = At.shape
        xgrad = torch.zeros_like(x)
        Atgrad = torch.zeros_like(At)
        with torch.cuda.device(x.device.index):
            grid = lambda meta: (triton.cdiv(N, meta['N_BLOCK_SIZE']),)
            linear_xent_bwd_kernel_matmul_t_dx[grid](x, y, At, lse_global,
                xgrad, x.stride(0), x.stride(1), At.stride(0), At.stride(1),
                V=V, N=N, H=H)
            print('bwd config dx:', linear_xent_bwd_kernel_matmul_t_dx.
                best_config)
        ctx.mark_non_differentiable(y)
        return xgrad * grad_output, None, Atgrad * grad_output, None
