# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/JonasGeiping/linear_cross_entropy_loss
# Source-Files: variants/chunked_highmem.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_0e62yj5n/linear_cross_entropy_loss-main/variants/chunked_highmem.py
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
from math import log

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=[triton.Config({'V_BLOCK_SIZE': 64, 'N_BLOCK_SIZE':
    16, 'H_BLOCK_SIZE': 64}), triton.Config({'V_BLOCK_SIZE': 64,
    'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64}), triton.Config({'V_BLOCK_SIZE':
    256, 'N_BLOCK_SIZE': 16, 'H_BLOCK_SIZE': 256}), triton.Config({
    'V_BLOCK_SIZE': 512, 'N_BLOCK_SIZE': 16, 'H_BLOCK_SIZE': 512}), triton.
    Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64}),
    triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 256, 'H_BLOCK_SIZE':
    64}), triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 256,
    'H_BLOCK_SIZE': 256}), triton.Config({'V_BLOCK_SIZE': 256,
    'N_BLOCK_SIZE': 16, 'H_BLOCK_SIZE': 16})], key=['V', 'N', 'H'],
    reset_to_zero=['loss_ptr', 'lse_ptr'])
@triton.jit
def linear_xent_fwd_kernel_matmul_t(x_ptr, y_ptr, A_t_ptr, loss_ptr,
    lse_ptr, stride_x_N, stride_x_H, stride_A_H, stride_A_V, V: tl.
    constexpr, N: tl.constexpr, H: tl.constexpr, V_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr, H_BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(axis=0)
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
    tl.atomic_add(loss_ptr, loss)
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
    assert V % 16 == 0, f'V is {V}'
    assert N % 16 == 0, f'N is {N}'
    assert H % 16 == 0, f'H is {H}'
    lse_global = torch.zeros(N, dtype=torch.float32, device=x.device)
    loss = torch.zeros(1, dtype=torch.float32, device=x.device)
    grid = lambda meta: (triton.cdiv(N, meta['N_BLOCK_SIZE']),)
    with torch.cuda.device(x.device.index):
        linear_xent_fwd_kernel_matmul_t[grid](x, y, At, loss, lse_global, x
            .stride(0), x.stride(1), At.stride(0), At.stride(1), V=V, N=N, H=H)
    print('fwd config:', linear_xent_fwd_kernel_matmul_t.best_config)
    ctx.save_for_backward(x, y, At, lse_global)
    return loss


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=[triton.Config({'V_BLOCK_SIZE': 16, 'N_BLOCK_SIZE':
    16, 'H_BLOCK_SIZE': 32}), triton.Config({'V_BLOCK_SIZE': 16,
    'N_BLOCK_SIZE': 16, 'H_BLOCK_SIZE': 16}), triton.Config({'V_BLOCK_SIZE':
    16, 'N_BLOCK_SIZE': 16, 'H_BLOCK_SIZE': 64}), triton.Config({
    'V_BLOCK_SIZE': 32, 'N_BLOCK_SIZE': 32, 'H_BLOCK_SIZE': 32}), triton.
    Config({'V_BLOCK_SIZE': 32, 'N_BLOCK_SIZE': 32, 'H_BLOCK_SIZE': 64}),
    triton.Config({'V_BLOCK_SIZE': 32, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 
    64}), triton.Config({'V_BLOCK_SIZE': 64, 'N_BLOCK_SIZE': 64,
    'H_BLOCK_SIZE': 64}), triton.Config({'V_BLOCK_SIZE': 256,
    'N_BLOCK_SIZE': 256, 'H_BLOCK_SIZE': 16}), triton.Config({
    'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 16, 'H_BLOCK_SIZE': 16}), triton.
    Config({'V_BLOCK_SIZE': 16, 'N_BLOCK_SIZE': 16, 'H_BLOCK_SIZE': 128}),
    triton.Config({'V_BLOCK_SIZE': 16, 'N_BLOCK_SIZE': 16, 'H_BLOCK_SIZE': 
    256}), triton.Config({'V_BLOCK_SIZE': 16, 'N_BLOCK_SIZE': 16,
    'H_BLOCK_SIZE': 512}), triton.Config({'V_BLOCK_SIZE': 64,
    'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 16}), triton.Config({'V_BLOCK_SIZE':
    32, 'N_BLOCK_SIZE': 32, 'H_BLOCK_SIZE': 16})], key=['V', 'N', 'H'],
    reset_to_zero=['A_grad_ptr', 'x_grad_ptr'])
@triton.jit()
def linear_xent_bwd_kernel_matmul_t_epilogue(sz_ptr, x_ptr, y_ptr, A_t_ptr,
    x_grad_ptr, A_grad_ptr, stride_x_N, stride_x_H, stride_A_H, stride_A_V,
    stride_x_grad_N, stride_x_grad_H, stride_A_grad_H, stride_A_grad_V,
    stride_sz_N, stride_sz_V, V: tl.constexpr, N: tl.constexpr, H: tl.
    constexpr, V_BLOCK_SIZE: tl.constexpr=16, N_BLOCK_SIZE: tl.constexpr=16,
    H_BLOCK_SIZE: tl.constexpr=16):
    tl.static_print(V_BLOCK_SIZE, N_BLOCK_SIZE, H_BLOCK_SIZE)
    idx_NV = tl.program_id(axis=1)
    if idx_NV < N // N_BLOCK_SIZE:
        linear_xent_bwd_kernel_matmul_t_epilogue_dx(sz_ptr, y_ptr, A_t_ptr,
            x_grad_ptr, stride_x_N, stride_x_H, stride_A_H, stride_A_V,
            stride_x_grad_N, stride_x_grad_H, stride_A_grad_H,
            stride_A_grad_V, stride_sz_N, stride_sz_V, V, N, H,
            V_BLOCK_SIZE, N_BLOCK_SIZE, H_BLOCK_SIZE)
    else:
        linear_xent_bwd_kernel_matmul_t_epilogue_dA(sz_ptr, x_ptr, y_ptr,
            A_grad_ptr, stride_x_N, stride_x_H, stride_A_H, stride_A_V,
            stride_x_grad_N, stride_x_grad_H, stride_A_grad_H,
            stride_A_grad_V, stride_sz_N, stride_sz_V, V, N, H,
            V_BLOCK_SIZE, N_BLOCK_SIZE, H_BLOCK_SIZE)


@triton.jit()
def linear_xent_bwd_kernel_matmul_t_epilogue_dA(sz_ptr, x_ptr, y_ptr,
    A_grad_ptr, stride_x_N, stride_x_H, stride_A_H, stride_A_V,
    stride_x_grad_N, stride_x_grad_H, stride_A_grad_H, stride_A_grad_V,
    stride_sz_N, stride_sz_V, V: tl.constexpr, N: tl.constexpr, H: tl.
    constexpr, V_BLOCK_SIZE: tl.constexpr=16, N_BLOCK_SIZE: tl.constexpr=16,
    H_BLOCK_SIZE: tl.constexpr=16):
    idx_H = tl.program_id(axis=0)
    idx_V = tl.program_id(axis=1) - N // N_BLOCK_SIZE
    x_block_ptr = tl.make_block_ptr(base=x_ptr, shape=(N, H), strides=(
        stride_x_N, stride_x_H), offsets=(0, idx_H * H_BLOCK_SIZE),
        block_shape=(N_BLOCK_SIZE, H_BLOCK_SIZE), order=(1, 0))
    A_t_grad_block_ptr = tl.make_block_ptr(base=A_grad_ptr, shape=(H, V),
        strides=(stride_A_grad_H, stride_A_grad_V), offsets=(idx_H *
        H_BLOCK_SIZE, idx_V * V_BLOCK_SIZE), block_shape=(H_BLOCK_SIZE,
        V_BLOCK_SIZE), order=(1, 0))
    sz_block_ptr = tl.make_block_ptr(base=sz_ptr, shape=(N, V), strides=(
        stride_sz_N, stride_sz_V), offsets=(0, idx_V * V_BLOCK_SIZE),
        block_shape=(N_BLOCK_SIZE, V_BLOCK_SIZE), order=(1, 0))
    N_offsets = tl.arange(0, N_BLOCK_SIZE)
    V_offsets = idx_V * V_BLOCK_SIZE + tl.arange(0, V_BLOCK_SIZE)
    A_grad_acc = tl.zeros((V_BLOCK_SIZE, H_BLOCK_SIZE), tl.float32)
    for idx_N in range(N // N_BLOCK_SIZE):
        y = tl.load(y_ptr + N_offsets)
        mask = (y[:, None] == V_offsets[None, :])[:, :, None]
        x_chunk = tl.load(x_block_ptr)
        sz = tl.load(sz_block_ptr).trans()
        A_grad_acc = tl.dot(sz, x_chunk, A_grad_acc)
        A_grad_acc -= tl.sum(tl.where(mask, x_chunk[:, None, :], 0.0), axis=0)
        x_block_ptr = tl.advance(x_block_ptr, [N_BLOCK_SIZE, 0])
        sz_block_ptr = tl.advance(sz_block_ptr, [N_BLOCK_SIZE, 0])
        N_offsets += N_BLOCK_SIZE
    tl.store(A_t_grad_block_ptr, A_grad_acc.trans() / N)


@triton.jit()
def linear_xent_bwd_kernel_matmul_t_epilogue_dx(sz_ptr, y_ptr, A_t_ptr,
    x_grad_ptr, stride_x_N, stride_x_H, stride_A_H, stride_A_V,
    stride_x_grad_N, stride_x_grad_H, stride_A_grad_H, stride_A_grad_V,
    stride_sz_N, stride_sz_V, V: tl.constexpr, N: tl.constexpr, H: tl.
    constexpr, V_BLOCK_SIZE: tl.constexpr=16, N_BLOCK_SIZE: tl.constexpr=16,
    H_BLOCK_SIZE: tl.constexpr=16):
    idx_H = tl.program_id(axis=0)
    idx_N = tl.program_id(axis=1)
    x_grad_block_ptr = tl.make_block_ptr(base=x_grad_ptr, shape=(N, H),
        strides=(stride_x_grad_N, stride_x_grad_H), offsets=(idx_N *
        N_BLOCK_SIZE, idx_H * H_BLOCK_SIZE), block_shape=(N_BLOCK_SIZE,
        H_BLOCK_SIZE), order=(1, 0))
    A_t_block_ptr = tl.make_block_ptr(base=A_t_ptr, shape=(H, V), strides=(
        stride_A_H, stride_A_V), offsets=(idx_H * H_BLOCK_SIZE, 0),
        block_shape=(H_BLOCK_SIZE, V_BLOCK_SIZE), order=(1, 0))
    sz_block_ptr = tl.make_block_ptr(base=sz_ptr, shape=(N, V), strides=(
        stride_sz_N, stride_sz_V), offsets=(idx_N * N_BLOCK_SIZE, 0),
        block_shape=(N_BLOCK_SIZE, V_BLOCK_SIZE), order=(1, 0))
    N_offsets = idx_N * N_BLOCK_SIZE + tl.arange(0, N_BLOCK_SIZE)
    V_offsets = tl.arange(0, V_BLOCK_SIZE)
    y = tl.load(y_ptr + N_offsets)
    x_grad_acc = tl.zeros((N_BLOCK_SIZE, H_BLOCK_SIZE), tl.float32)
    for idx_V in range(V // V_BLOCK_SIZE):
        mask = (y[:, None] == V_offsets[None, :])[:, :, None]
        A_v = tl.load(A_t_block_ptr).trans()
        sz = tl.load(sz_block_ptr)
        x_grad_acc = tl.dot(sz, A_v, x_grad_acc)
        x_grad_acc -= tl.sum(tl.where(mask, A_v[None, :, :], 0.0), axis=1)
        A_t_block_ptr = tl.advance(A_t_block_ptr, [0, V_BLOCK_SIZE])
        sz_block_ptr = tl.advance(sz_block_ptr, [0, V_BLOCK_SIZE])
        V_offsets += V_BLOCK_SIZE
    tl.store(x_grad_block_ptr, x_grad_acc / N)


@triton.autotune(configs=[triton.Config({'V_BLOCK_SIZE': 16, 'N_BLOCK_SIZE':
    16, 'H_BLOCK_SIZE': 16}), triton.Config({'V_BLOCK_SIZE': 16,
    'N_BLOCK_SIZE': 16, 'H_BLOCK_SIZE': 16}), triton.Config({'V_BLOCK_SIZE':
    16, 'N_BLOCK_SIZE': 16, 'H_BLOCK_SIZE': 64}), triton.Config({
    'V_BLOCK_SIZE': 32, 'N_BLOCK_SIZE': 32, 'H_BLOCK_SIZE': 32}), triton.
    Config({'V_BLOCK_SIZE': 32, 'N_BLOCK_SIZE': 32, 'H_BLOCK_SIZE': 64}),
    triton.Config({'V_BLOCK_SIZE': 32, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 
    64}), triton.Config({'V_BLOCK_SIZE': 16, 'N_BLOCK_SIZE': 16,
    'H_BLOCK_SIZE': 128}), triton.Config({'V_BLOCK_SIZE': 16,
    'N_BLOCK_SIZE': 16, 'H_BLOCK_SIZE': 256}), triton.Config({
    'V_BLOCK_SIZE': 16, 'N_BLOCK_SIZE': 16, 'H_BLOCK_SIZE': 512})], key=[
    'V', 'N', 'H'], reset_to_zero=['sz_ptr'])
@triton.jit()
def linear_xent_bwd_kernel_matmul_t_prologue(sz_ptr, x_ptr, A_t_ptr,
    lse_global_ptr, stride_x_N, stride_x_H, stride_A_H, stride_A_V,
    stride_sz_N, stride_sz_V, V: tl.constexpr, N: tl.constexpr, H: tl.
    constexpr, V_BLOCK_SIZE: tl.constexpr=16, N_BLOCK_SIZE: tl.constexpr=16,
    H_BLOCK_SIZE: tl.constexpr=16):
    idx_N = tl.program_id(axis=0)
    idx_V = tl.program_id(axis=1)
    tl.static_print(V_BLOCK_SIZE, N_BLOCK_SIZE, H_BLOCK_SIZE)
    offsets = idx_N * N_BLOCK_SIZE + tl.arange(0, N_BLOCK_SIZE)
    lse = tl.load(lse_global_ptr + offsets)
    x_block_ptr = tl.make_block_ptr(base=x_ptr, shape=(N, H), strides=(
        stride_x_N, stride_x_H), offsets=(idx_N * N_BLOCK_SIZE, 0),
        block_shape=(N_BLOCK_SIZE, H_BLOCK_SIZE), order=(1, 0))
    A_block_ptr = tl.make_block_ptr(base=A_t_ptr, shape=(H, V), strides=(
        stride_A_H, stride_A_V), offsets=(0, idx_V * V_BLOCK_SIZE),
        block_shape=(H_BLOCK_SIZE, V_BLOCK_SIZE), order=(1, 0))
    sz_block_ptr = tl.make_block_ptr(base=sz_ptr, shape=(N, V), strides=(
        stride_sz_N, stride_sz_V), offsets=(idx_N * N_BLOCK_SIZE, idx_V *
        V_BLOCK_SIZE), block_shape=(N_BLOCK_SIZE, V_BLOCK_SIZE), order=(1, 0))
    z_j_to_k = tl.zeros((N_BLOCK_SIZE, V_BLOCK_SIZE), dtype=tl.float32)
    for _ in range(H // H_BLOCK_SIZE):
        x_chunk = tl.load(x_block_ptr)
        A_v = tl.load(A_block_ptr)
        z_j_to_k = tl.dot(x_chunk, A_v, z_j_to_k)
        x_block_ptr = tl.advance(x_block_ptr, [0, H_BLOCK_SIZE])
        A_block_ptr = tl.advance(A_block_ptr, [H_BLOCK_SIZE, 0])
    softmax_z = (z_j_to_k - lse[:, None]).exp()
    tl.store(sz_block_ptr, softmax_z.to(tl.float16))


# Backward method (kernel launch code)
def _LinearCrossEntropyLoss_backward(ctx, grad_output):
    x, y, At, lse_global = ctx.saved_tensors
    N, H = x.shape
    _, V = At.shape
    xgrad = torch.zeros_like(x, dtype=torch.float32)
    Atgrad = torch.zeros_like(At, dtype=torch.float32)
    with torch.cuda.device(x.device.index):
        V_tile_size = min(H, V)
        N_tile_size = min(H, N)
        for v_idx in range(V // V_tile_size):
            for n_idx in range(N // N_tile_size):
                x_tile = x[n_idx * N_tile_size:(n_idx + 1) * N_tile_size, :]
                A_v_t = At[:, v_idx * V_tile_size:(v_idx + 1) * V_tile_size]
                lse_tile = lse_global[n_idx * N_tile_size:(n_idx + 1) *
                    N_tile_size]
                sz = torch.empty((N_tile_size, V_tile_size), dtype=torch.
                    float16, device=x.device)
                x_grad_tile = torch.zeros_like(x_tile, dtype=torch.float32)
                At_grad_tile = torch.zeros_like(A_v_t, dtype=torch.float32)
                sz = torch.empty((N, V), dtype=torch.float16, device=x.device)
                grid = lambda meta: (triton.cdiv(N, meta['N_BLOCK_SIZE']),
                    triton.cdiv(V, meta['V_BLOCK_SIZE']))
                linear_xent_bwd_kernel_matmul_t_prologue[grid](sz, x_tile,
                    A_v_t, lse_tile, x_tile.stride(0), x_tile.stride(1),
                    A_v_t.stride(0), A_v_t.stride(1), sz.stride(0), sz.
                    stride(1), V_tile_size, N_tile_size, H)
                grid = lambda meta: (triton.cdiv(H, meta['H_BLOCK_SIZE']), 
                    triton.cdiv(N, meta['N_BLOCK_SIZE']) + triton.cdiv(V,
                    meta['V_BLOCK_SIZE']))
                linear_xent_bwd_kernel_matmul_t_epilogue[grid](sz, x_tile,
                    y, A_v_t, x_grad_tile, At_grad_tile, x_tile.stride(0),
                    x_tile.stride(1), A_v_t.stride(0), A_v_t.stride(1),
                    x_grad_tile.stride(0), x_grad_tile.stride(1),
                    At_grad_tile.stride(0), At_grad_tile.stride(1), sz.
                    stride(0), sz.stride(1), V_tile_size, N_tile_size, H)
                xgrad[n_idx * N_tile_size:(n_idx + 1) * N_tile_size, :
                    ] += x_grad_tile
                Atgrad[:, v_idx * V_tile_size:(v_idx + 1) * V_tile_size
                    ] += At_grad_tile
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
        assert V % 16 == 0, f'V is {V}'
        assert N % 16 == 0, f'N is {N}'
        assert H % 16 == 0, f'H is {H}'
        lse_global = torch.zeros(N, dtype=torch.float32, device=x.device)
        loss = torch.zeros(1, dtype=torch.float32, device=x.device)
        grid = lambda meta: (triton.cdiv(N, meta['N_BLOCK_SIZE']),)
        with torch.cuda.device(x.device.index):
            linear_xent_fwd_kernel_matmul_t[grid](x, y, At, loss,
                lse_global, x.stride(0), x.stride(1), At.stride(0), At.
                stride(1), V=V, N=N, H=H)
        print('fwd config:', linear_xent_fwd_kernel_matmul_t.best_config)
        ctx.save_for_backward(x, y, At, lse_global)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        x, y, At, lse_global = ctx.saved_tensors
        N, H = x.shape
        _, V = At.shape
        xgrad = torch.zeros_like(x, dtype=torch.float32)
        Atgrad = torch.zeros_like(At, dtype=torch.float32)
        with torch.cuda.device(x.device.index):
            V_tile_size = min(H, V)
            N_tile_size = min(H, N)
            for v_idx in range(V // V_tile_size):
                for n_idx in range(N // N_tile_size):
                    x_tile = x[n_idx * N_tile_size:(n_idx + 1) * N_tile_size, :
                        ]
                    A_v_t = At[:, v_idx * V_tile_size:(v_idx + 1) * V_tile_size
                        ]
                    lse_tile = lse_global[n_idx * N_tile_size:(n_idx + 1) *
                        N_tile_size]
                    sz = torch.empty((N_tile_size, V_tile_size), dtype=
                        torch.float16, device=x.device)
                    x_grad_tile = torch.zeros_like(x_tile, dtype=torch.float32)
                    At_grad_tile = torch.zeros_like(A_v_t, dtype=torch.float32)
                    sz = torch.empty((N, V), dtype=torch.float16, device=x.
                        device)
                    grid = lambda meta: (triton.cdiv(N, meta['N_BLOCK_SIZE'
                        ]), triton.cdiv(V, meta['V_BLOCK_SIZE']))
                    linear_xent_bwd_kernel_matmul_t_prologue[grid](sz,
                        x_tile, A_v_t, lse_tile, x_tile.stride(0), x_tile.
                        stride(1), A_v_t.stride(0), A_v_t.stride(1), sz.
                        stride(0), sz.stride(1), V_tile_size, N_tile_size, H)
                    grid = lambda meta: (triton.cdiv(H, meta['H_BLOCK_SIZE'
                        ]), triton.cdiv(N, meta['N_BLOCK_SIZE']) + triton.
                        cdiv(V, meta['V_BLOCK_SIZE']))
                    linear_xent_bwd_kernel_matmul_t_epilogue[grid](sz,
                        x_tile, y, A_v_t, x_grad_tile, At_grad_tile, x_tile
                        .stride(0), x_tile.stride(1), A_v_t.stride(0),
                        A_v_t.stride(1), x_grad_tile.stride(0), x_grad_tile
                        .stride(1), At_grad_tile.stride(0), At_grad_tile.
                        stride(1), sz.stride(0), sz.stride(1), V_tile_size,
                        N_tile_size, H)
                    xgrad[n_idx * N_tile_size:(n_idx + 1) * N_tile_size, :
                        ] += x_grad_tile
                    Atgrad[:, v_idx * V_tile_size:(v_idx + 1) * V_tile_size
                        ] += At_grad_tile
        return xgrad * grad_output, None, Atgrad * grad_output, None
