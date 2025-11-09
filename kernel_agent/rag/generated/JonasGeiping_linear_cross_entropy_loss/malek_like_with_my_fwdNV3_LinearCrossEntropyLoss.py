# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/JonasGeiping/linear_cross_entropy_loss
# Source-Files: variants/malek_like_with_my_fwdNV3.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_0e62yj5n/linear_cross_entropy_loss-main/variants/malek_like_with_my_fwdNV3.py
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
from math import ceil
from math import exp
from math import log

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=[triton.Config({'V_BLOCK_SIZE': 128,
    'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 128, 'GROUP_SIZE': 64}, num_warps=
    4), triton.Config({'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 128,
    'H_BLOCK_SIZE': 128, 'GROUP_SIZE': 64}, num_warps=8), triton.Config({
    'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 256,
    'GROUP_SIZE': 64}, num_warps=4), triton.Config({'V_BLOCK_SIZE': 128,
    'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 256, 'GROUP_SIZE': 64}, num_warps=
    8), triton.Config({'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 256,
    'H_BLOCK_SIZE': 128, 'GROUP_SIZE': 64}, num_warps=4), triton.Config({
    'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 256, 'H_BLOCK_SIZE': 128,
    'GROUP_SIZE': 64}, num_warps=8), triton.Config({'V_BLOCK_SIZE': 128,
    'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 128, 'GROUP_SIZE': 64}, num_warps=
    4, num_stages=3), triton.Config({'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 
    128, 'H_BLOCK_SIZE': 128, 'GROUP_SIZE': 64}, num_warps=8, num_stages=3),
    triton.Config({'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE':
    256, 'GROUP_SIZE': 64}, num_warps=4, num_stages=3), triton.Config({
    'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 256,
    'GROUP_SIZE': 64}, num_warps=8, num_stages=3), triton.Config({
    'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 256, 'H_BLOCK_SIZE': 128,
    'GROUP_SIZE': 64}, num_warps=4, num_stages=3), triton.Config({
    'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 256, 'H_BLOCK_SIZE': 128,
    'GROUP_SIZE': 64}, num_warps=8, num_stages=3), triton.Config({
    'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 128,
    'GROUP_SIZE': 32}, num_warps=4), triton.Config({'V_BLOCK_SIZE': 128,
    'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 128, 'GROUP_SIZE': 32}, num_warps=
    8), triton.Config({'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 128,
    'H_BLOCK_SIZE': 256, 'GROUP_SIZE': 32}, num_warps=4), triton.Config({
    'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 256,
    'GROUP_SIZE': 32}, num_warps=8), triton.Config({'V_BLOCK_SIZE': 128,
    'N_BLOCK_SIZE': 256, 'H_BLOCK_SIZE': 128, 'GROUP_SIZE': 32}, num_warps=
    4), triton.Config({'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 256,
    'H_BLOCK_SIZE': 128, 'GROUP_SIZE': 32}, num_warps=8), triton.Config({
    'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 128,
    'GROUP_SIZE': 32}, num_warps=4, num_stages=3), triton.Config({
    'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 128,
    'GROUP_SIZE': 32}, num_warps=8, num_stages=3), triton.Config({
    'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 256,
    'GROUP_SIZE': 32}, num_warps=4, num_stages=3), triton.Config({
    'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 256,
    'GROUP_SIZE': 32}, num_warps=8, num_stages=3), triton.Config({
    'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 256, 'H_BLOCK_SIZE': 128,
    'GROUP_SIZE': 32}, num_warps=4, num_stages=3), triton.Config({
    'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 256, 'H_BLOCK_SIZE': 128,
    'GROUP_SIZE': 32}, num_warps=8, num_stages=3)], key=['V', 'N', 'H'],
    reset_to_zero=['A_grad_ptr'])
@triton.jit()
def linear_xent_bwd_kernel_matmul_t_epilogue_dA(z_nv_ptr, y_ptr, x_ptr,
    A_grad_ptr, lse_ptr, stride_x_N, stride_x_H, stride_A_H, stride_A_V,
    stride_z_N, stride_z_V, idx_N_group, N_group: tl.constexpr, V: tl.
    constexpr, N: tl.constexpr, H: tl.constexpr, V_BLOCK_SIZE: tl.constexpr
    =16, N_BLOCK_SIZE: tl.constexpr=16, H_BLOCK_SIZE: tl.constexpr=16,
    GROUP_SIZE: tl.constexpr=16):
    idx_V = tl.program_id(axis=0)
    idx_H = tl.program_id(axis=1)
    num_idx_V, num_idx_H = tl.num_programs(0), tl.num_programs(1)
    idx_V, idx_H = tl.swizzle2d(idx_V, idx_H, num_idx_V, num_idx_H, GROUP_SIZE)
    x_block_ptr = tl.make_block_ptr(base=x_ptr, shape=(N, H), strides=(
        stride_x_N, stride_x_H), offsets=(idx_N_group * N_group, idx_H *
        H_BLOCK_SIZE), block_shape=(N_BLOCK_SIZE, H_BLOCK_SIZE), order=(1, 0))
    A_grad_T_block_ptr = tl.make_block_ptr(base=A_grad_ptr, shape=(H, V),
        strides=(stride_A_H, stride_A_V), offsets=(idx_H * H_BLOCK_SIZE, 
        idx_V * V_BLOCK_SIZE), block_shape=(H_BLOCK_SIZE, V_BLOCK_SIZE),
        order=(0, 1))
    z_block_ptr = tl.make_block_ptr(base=z_nv_ptr, shape=(N_group, V),
        strides=(stride_z_N, stride_z_V), offsets=(0, idx_V * V_BLOCK_SIZE),
        block_shape=(N_BLOCK_SIZE, V_BLOCK_SIZE), order=(1, 0))
    N_range = tl.arange(0, N_BLOCK_SIZE)
    V_range = idx_V * V_BLOCK_SIZE + tl.arange(0, V_BLOCK_SIZE)
    A_grad_acc = tl.zeros((H_BLOCK_SIZE, V_BLOCK_SIZE), tl.float32)
    for _ in range(N_group // N_BLOCK_SIZE):
        y = tl.load(y_ptr + idx_N_group * N_group + N_range)
        lse = tl.load(lse_ptr + N_range)
        mask = y[:, None] == V_range[None, :]
        x_chunk = tl.load(x_block_ptr)
        z_j_to_k = tl.load(z_block_ptr)
        softmax_z = (z_j_to_k - lse[:, None]).exp()
        z_grad = (softmax_z - tl.where(mask, 1.0, 0.0)).to(tl.float16)
        A_grad_acc = tl.dot(x_chunk.trans(), z_grad, A_grad_acc)
        x_block_ptr = tl.advance(x_block_ptr, [N_BLOCK_SIZE, 0])
        z_block_ptr = tl.advance(z_block_ptr, [N_BLOCK_SIZE, 0])
        N_range += N_BLOCK_SIZE
    tl.store(A_grad_T_block_ptr, tl.load(A_grad_T_block_ptr) + (A_grad_acc /
        N).to(A_grad_ptr.type.element_ty))


@triton.autotune(configs=[triton.Config({'V_BLOCK_SIZE': 128,
    'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 128, 'GROUP_SIZE': 32}, num_warps=
    8), triton.Config({'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 128,
    'H_BLOCK_SIZE': 128, 'GROUP_SIZE': 32}, num_warps=4, num_stages=4),
    triton.Config({'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE':
    128, 'GROUP_SIZE': 32}, num_warps=8, num_stages=4), triton.Config({
    'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 128,
    'GROUP_SIZE': 32}, num_warps=4, num_stages=3), triton.Config({
    'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 128,
    'GROUP_SIZE': 32}, num_warps=8, num_stages=3), triton.Config({
    'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 128,
    'GROUP_SIZE': 32}, num_warps=8), triton.Config({'V_BLOCK_SIZE': 256,
    'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 128, 'GROUP_SIZE': 32}, num_warps=
    4, num_stages=4), triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 
    128, 'H_BLOCK_SIZE': 128, 'GROUP_SIZE': 32}, num_warps=8, num_stages=4),
    triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE':
    128, 'GROUP_SIZE': 32}, num_warps=4, num_stages=3), triton.Config({
    'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 256,
    'GROUP_SIZE': 32}, num_warps=8, num_stages=3), triton.Config({
    'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 256,
    'GROUP_SIZE': 32}, num_warps=4, num_stages=3), triton.Config({
    'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 256,
    'GROUP_SIZE': 32}, num_warps=8, num_stages=3), triton.Config({
    'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 256,
    'GROUP_SIZE': 32}, num_warps=4, num_stages=3), triton.Config({
    'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 256,
    'GROUP_SIZE': 32}, num_warps=8, num_stages=3)], key=['V', 'N', 'H'],
    reset_to_zero=['x_grad_ptr'])
@triton.jit()
def linear_xent_bwd_kernel_matmul_t_epilogue_dx(z_nv_ptr, y_ptr, A_t_ptr,
    x_grad_ptr, lse_ptr, stride_x_N, stride_x_H, stride_A_H, stride_A_V,
    stride_z_N, stride_z_V, idx_N_group, N_group: tl.constexpr, V: tl.
    constexpr, N: tl.constexpr, H: tl.constexpr, V_BLOCK_SIZE: tl.constexpr
    =16, N_BLOCK_SIZE: tl.constexpr=16, H_BLOCK_SIZE: tl.constexpr=16,
    GROUP_SIZE: tl.constexpr=1):
    idx_N = tl.program_id(axis=0)
    idx_H = tl.program_id(axis=1)
    idx_V = 0
    num_idx_N, num_idx_H = tl.num_programs(0), tl.num_programs(1)
    idx_N, idx_H = tl.swizzle2d(idx_N, idx_H, num_idx_N, num_idx_H, GROUP_SIZE)
    x_grad_block_ptr = tl.make_block_ptr(base=x_grad_ptr, shape=(N, H),
        strides=(stride_x_N, stride_x_H), offsets=(idx_N_group * N_group + 
        idx_N * N_BLOCK_SIZE, idx_H * H_BLOCK_SIZE), block_shape=(
        N_BLOCK_SIZE, H_BLOCK_SIZE), order=(1, 0))
    A_t_block_ptr = tl.make_block_ptr(base=A_t_ptr, shape=(H, V), strides=(
        stride_A_H, stride_A_V), offsets=(idx_H * H_BLOCK_SIZE, 0),
        block_shape=(H_BLOCK_SIZE, V_BLOCK_SIZE), order=(0, 1))
    z_block_ptr = tl.make_block_ptr(base=z_nv_ptr, shape=(N_group, V),
        strides=(stride_z_N, stride_z_V), offsets=(idx_N * N_BLOCK_SIZE, 
        idx_V * V_BLOCK_SIZE), block_shape=(N_BLOCK_SIZE, V_BLOCK_SIZE),
        order=(1, 0))
    N_range = idx_N_group * N_group + idx_N * N_BLOCK_SIZE + tl.arange(0,
        N_BLOCK_SIZE)
    v_range = 0 + tl.arange(0, V_BLOCK_SIZE)
    y = tl.load(y_ptr + N_range)
    lse = tl.load(lse_ptr + idx_N * N_BLOCK_SIZE + tl.arange(0, N_BLOCK_SIZE))
    x_grad_acc = tl.zeros((N_BLOCK_SIZE, H_BLOCK_SIZE), tl.float32)
    for _ in range(V // V_BLOCK_SIZE):
        mask = y[:, None] == v_range[None, :]
        A_v = tl.load(A_t_block_ptr)
        z_j_to_k = tl.load(z_block_ptr)
        softmax_z = (z_j_to_k - lse[:, None]).exp()
        z_grad = (softmax_z - tl.where(mask, 1.0, 0.0)).to(tl.float16)
        x_grad_acc = tl.dot(z_grad, A_v.trans(), x_grad_acc)
        A_t_block_ptr = tl.advance(A_t_block_ptr, [0, V_BLOCK_SIZE])
        z_block_ptr = tl.advance(z_block_ptr, [0, V_BLOCK_SIZE])
        v_range += V_BLOCK_SIZE
    tl.store(x_grad_block_ptr, (x_grad_acc / N).to(x_grad_ptr.type.element_ty))


@triton.autotune(configs=[triton.Config({'V_BLOCK_SIZE': 128,
    'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES': 1}), triton.Config({
    'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES':
    1}), triton.Config({'V_BLOCK_SIZE': 512, 'N_BLOCK_SIZE': 64,
    'H_BLOCK_SIZE': 64, 'V_TILES': 1}), triton.Config({'V_BLOCK_SIZE': 256,
    'N_BLOCK_SIZE': 16, 'H_BLOCK_SIZE': 256, 'V_TILES': 1}), triton.Config(
    {'V_BLOCK_SIZE': 512, 'N_BLOCK_SIZE': 16, 'H_BLOCK_SIZE': 16, 'V_TILES':
    1}), triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64,
    'H_BLOCK_SIZE': 128, 'V_TILES': 1}), triton.Config({'V_BLOCK_SIZE': 128,
    'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES': 1}, num_warps=4),
    triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE':
    64, 'V_TILES': 1}, num_warps=4), triton.Config({'V_BLOCK_SIZE': 512,
    'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES': 1}, num_warps=4),
    triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 16, 'H_BLOCK_SIZE':
    256, 'V_TILES': 1}, num_warps=4), triton.Config({'V_BLOCK_SIZE': 512,
    'N_BLOCK_SIZE': 16, 'H_BLOCK_SIZE': 16, 'V_TILES': 1}, num_warps=4),
    triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE':
    128, 'V_TILES': 1}, num_warps=4), triton.Config({'V_BLOCK_SIZE': 128,
    'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES': 1}, num_warps=8),
    triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE':
    64, 'V_TILES': 1}, num_warps=8), triton.Config({'V_BLOCK_SIZE': 256,
    'N_BLOCK_SIZE': 16, 'H_BLOCK_SIZE': 256, 'V_TILES': 1}, num_warps=8),
    triton.Config({'V_BLOCK_SIZE': 512, 'N_BLOCK_SIZE': 16, 'H_BLOCK_SIZE':
    16, 'V_TILES': 1}, num_warps=8), triton.Config({'V_BLOCK_SIZE': 256,
    'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 128, 'V_TILES': 1}, num_warps=8),
    triton.Config({'V_BLOCK_SIZE': 512, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE':
    64, 'V_TILES': 1}, num_warps=8), triton.Config({'V_BLOCK_SIZE': 512,
    'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES': 1}, num_warps=16),
    triton.Config({'V_BLOCK_SIZE': 512, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE':
    64, 'V_TILES': 1}, num_warps=32), triton.Config({'V_BLOCK_SIZE': 512,
    'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES': 1}, num_warps=8,
    num_stages=4), triton.Config({'V_BLOCK_SIZE': 512, 'N_BLOCK_SIZE': 64,
    'H_BLOCK_SIZE': 64, 'V_TILES': 1}, num_warps=8, num_stages=5), triton.
    Config({'V_BLOCK_SIZE': 512, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64,
    'V_TILES': 1}, num_warps=8, num_stages=6), triton.Config({
    'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES':
    1, 'GROUP_SIZE': 32}, num_warps=4), triton.Config({'V_BLOCK_SIZE': 256,
    'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES': 1, 'GROUP_SIZE': 32},
    num_warps=8), triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64,
    'H_BLOCK_SIZE': 64, 'V_TILES': 1, 'GROUP_SIZE': 32}, num_warps=16),
    triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE':
    64, 'V_TILES': 2, 'GROUP_SIZE': 32}, num_warps=4), triton.Config({
    'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES':
    2, 'GROUP_SIZE': 32}, num_warps=8), triton.Config({'V_BLOCK_SIZE': 256,
    'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES': 2, 'GROUP_SIZE': 32},
    num_warps=16), triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64,
    'H_BLOCK_SIZE': 64, 'V_TILES': 4, 'GROUP_SIZE': 32}, num_warps=4),
    triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE':
    64, 'V_TILES': 4, 'GROUP_SIZE': 32}, num_warps=8), triton.Config({
    'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES':
    4, 'GROUP_SIZE': 32}, num_warps=16), triton.Config({'V_BLOCK_SIZE': 256,
    'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES': 1, 'GROUP_SIZE': 32},
    num_warps=4), triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64,
    'H_BLOCK_SIZE': 64, 'V_TILES': 1, 'GROUP_SIZE': 32}, num_warps=8),
    triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE':
    64, 'V_TILES': 1, 'GROUP_SIZE': 32}, num_warps=16), triton.Config({
    'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES':
    2, 'GROUP_SIZE': 32}, num_warps=4), triton.Config({'V_BLOCK_SIZE': 256,
    'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES': 2, 'GROUP_SIZE': 32},
    num_warps=8), triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64,
    'H_BLOCK_SIZE': 64, 'V_TILES': 2, 'GROUP_SIZE': 32}, num_warps=16),
    triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE':
    64, 'V_TILES': 4, 'GROUP_SIZE': 32}, num_warps=4), triton.Config({
    'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES':
    4, 'GROUP_SIZE': 32}, num_warps=8), triton.Config({'V_BLOCK_SIZE': 256,
    'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES': 4, 'GROUP_SIZE': 32},
    num_warps=16), triton.Config({'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 128,
    'H_BLOCK_SIZE': 128, 'V_TILES': 1, 'GROUP_SIZE': 32}, num_warps=4),
    triton.Config({'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE':
    128, 'V_TILES': 1, 'GROUP_SIZE': 32}, num_warps=8), triton.Config({
    'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 128,
    'V_TILES': 1, 'GROUP_SIZE': 32}, num_warps=16), triton.Config({
    'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 128,
    'V_TILES': 2, 'GROUP_SIZE': 32}, num_warps=4), triton.Config({
    'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 128,
    'V_TILES': 2, 'GROUP_SIZE': 32}, num_warps=8), triton.Config({
    'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 128,
    'V_TILES': 2, 'GROUP_SIZE': 32}, num_warps=16), triton.Config({
    'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 128,
    'V_TILES': 4, 'GROUP_SIZE': 32}, num_warps=4), triton.Config({
    'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 128,
    'V_TILES': 4, 'GROUP_SIZE': 32}, num_warps=8), triton.Config({
    'V_BLOCK_SIZE': 128, 'N_BLOCK_SIZE': 128, 'H_BLOCK_SIZE': 128,
    'V_TILES': 4, 'GROUP_SIZE': 32}, num_warps=16), triton.Config({
    'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES':
    1, 'GROUP_SIZE': 64}, num_warps=4), triton.Config({'V_BLOCK_SIZE': 256,
    'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES': 1, 'GROUP_SIZE': 64},
    num_warps=8), triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64,
    'H_BLOCK_SIZE': 64, 'V_TILES': 1, 'GROUP_SIZE': 64}, num_warps=16),
    triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE':
    64, 'V_TILES': 2, 'GROUP_SIZE': 64}, num_warps=4), triton.Config({
    'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES':
    2, 'GROUP_SIZE': 64}, num_warps=8), triton.Config({'V_BLOCK_SIZE': 256,
    'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES': 2, 'GROUP_SIZE': 64},
    num_warps=16), triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64,
    'H_BLOCK_SIZE': 64, 'V_TILES': 4, 'GROUP_SIZE': 64}, num_warps=4),
    triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE':
    64, 'V_TILES': 4, 'GROUP_SIZE': 64}, num_warps=8), triton.Config({
    'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES':
    4, 'GROUP_SIZE': 64}, num_warps=16)], key=['V', 'N', 'H'],
    reset_to_zero=['losses_ptr', 'lse_ptr', 'z_nv_ptr'])
@triton.jit
def linear_xent_fwd_prep_bwd_kernel_matmul_t(x_ptr, y_ptr, A_t_ptr,
    z_nv_ptr, losses_ptr, lse_ptr, stride_x_N, stride_x_H, stride_A_H,
    stride_A_V, stride_z_N, stride_z_V, stride_lse_N, stride_lse_B,
    stride_loss_Nb, stride_loss_B, idx_N_group, N_group: tl.constexpr, V:
    tl.constexpr, N: tl.constexpr, H: tl.constexpr, V_BLOCK_SIZE: tl.
    constexpr=16, N_BLOCK_SIZE: tl.constexpr=16, H_BLOCK_SIZE: tl.constexpr
    =16, V_TILES: tl.constexpr=1, GROUP_SIZE: tl.constexpr=1):
    idx_N = tl.program_id(axis=0)
    idx_V_group = tl.program_id(axis=1)
    num_idx_N, num_idx_V_group = tl.num_programs(0), tl.num_programs(1)
    idx_N, idx_V_group = tl.swizzle2d(idx_N, idx_V_group, num_idx_N,
        num_idx_V_group, GROUP_SIZE)
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
    lse_row_ptr = tl.make_block_ptr(base=lse_ptr, shape=(N_group, V // 128),
        strides=(stride_lse_N, stride_lse_B), offsets=(idx_N * N_BLOCK_SIZE,
        idx_V_group), block_shape=(N_BLOCK_SIZE, 1), order=(1, 0))
    loss_val_ptr = (losses_ptr + idx_N * stride_loss_Nb + idx_V_group *
        stride_loss_B)
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
    loss = 0.0
    logits = torch.empty((N_group, V), device=x.device, dtype=torch.float32)
    lse_local = -1000000.0 * torch.ones(N_group, V // 128, dtype=torch.
        float32, device=x.device)
    losses = torch.zeros(N_group // 64, V // 128, dtype=torch.float32,
        device=x.device)
    fwd_grid = lambda meta: (triton.cdiv(N_group, meta['N_BLOCK_SIZE']),
        triton.cdiv(V, meta['V_BLOCK_SIZE'] * meta['V_TILES']))
    bwd_grid_dx = lambda meta: (triton.cdiv(N_group, meta['N_BLOCK_SIZE']),
        triton.cdiv(H, meta['H_BLOCK_SIZE']))
    bwd_grid_dA = lambda meta: (triton.cdiv(V, meta['V_BLOCK_SIZE']),
        triton.cdiv(H, meta['H_BLOCK_SIZE']))
    for idx_N_group in range(math.ceil(N / N_group)):
        with torch.cuda.device(x.device.index):
            linear_xent_fwd_prep_bwd_kernel_matmul_t[fwd_grid](x, y, At,
                logits, losses, lse_local, x.stride(0), x.stride(1), At.
                stride(0), At.stride(1), logits.stride(0), logits.stride(1),
                lse_local.stride(0), lse_local.stride(1), losses.stride(0),
                losses.stride(1), idx_N_group=idx_N_group, N_group=N_group,
                V=V, N=N, H=H)
            lse_global = lse_local.logsumexp(dim=1)
            loss += losses.sum() + lse_global.sum() / N
            if x.requires_grad:
                linear_xent_bwd_kernel_matmul_t_epilogue_dx[bwd_grid_dx](logits
                    , y, At, x_grad, lse_global, x_grad.stride(0), x_grad.
                    stride(1), At.stride(0), At.stride(1), logits.stride(0),
                    logits.stride(1), idx_N_group=idx_N_group, N_group=
                    N_group, V=V, N=N, H=H)
            if At.requires_grad:
                linear_xent_bwd_kernel_matmul_t_epilogue_dA[bwd_grid_dA](logits
                    , y, x, At_grad, lse_global, x_grad.stride(0), x_grad.
                    stride(1), At.stride(0), At.stride(1), logits.stride(0),
                    logits.stride(1), idx_N_group=idx_N_group, N_group=
                    N_group, V=V, N=N, H=H)
    print('fwd config:', linear_xent_fwd_prep_bwd_kernel_matmul_t.best_config)
    print('dx config:', linear_xent_bwd_kernel_matmul_t_epilogue_dx.best_config
        )
    print('dA config:', linear_xent_bwd_kernel_matmul_t_epilogue_dA.best_config
        )
    ctx.mark_non_differentiable(y)
    ctx.save_for_backward(x_grad, At_grad.to(At.dtype))
    return loss


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
        loss = 0.0
        logits = torch.empty((N_group, V), device=x.device, dtype=torch.float32
            )
        lse_local = -1000000.0 * torch.ones(N_group, V // 128, dtype=torch.
            float32, device=x.device)
        losses = torch.zeros(N_group // 64, V // 128, dtype=torch.float32,
            device=x.device)
        fwd_grid = lambda meta: (triton.cdiv(N_group, meta['N_BLOCK_SIZE']),
            triton.cdiv(V, meta['V_BLOCK_SIZE'] * meta['V_TILES']))
        bwd_grid_dx = lambda meta: (triton.cdiv(N_group, meta[
            'N_BLOCK_SIZE']), triton.cdiv(H, meta['H_BLOCK_SIZE']))
        bwd_grid_dA = lambda meta: (triton.cdiv(V, meta['V_BLOCK_SIZE']),
            triton.cdiv(H, meta['H_BLOCK_SIZE']))
        for idx_N_group in range(math.ceil(N / N_group)):
            with torch.cuda.device(x.device.index):
                linear_xent_fwd_prep_bwd_kernel_matmul_t[fwd_grid](x, y, At,
                    logits, losses, lse_local, x.stride(0), x.stride(1), At
                    .stride(0), At.stride(1), logits.stride(0), logits.
                    stride(1), lse_local.stride(0), lse_local.stride(1),
                    losses.stride(0), losses.stride(1), idx_N_group=
                    idx_N_group, N_group=N_group, V=V, N=N, H=H)
                lse_global = lse_local.logsumexp(dim=1)
                loss += losses.sum() + lse_global.sum() / N
                if x.requires_grad:
                    linear_xent_bwd_kernel_matmul_t_epilogue_dx[bwd_grid_dx](
                        logits, y, At, x_grad, lse_global, x_grad.stride(0),
                        x_grad.stride(1), At.stride(0), At.stride(1),
                        logits.stride(0), logits.stride(1), idx_N_group=
                        idx_N_group, N_group=N_group, V=V, N=N, H=H)
                if At.requires_grad:
                    linear_xent_bwd_kernel_matmul_t_epilogue_dA[bwd_grid_dA](
                        logits, y, x, At_grad, lse_global, x_grad.stride(0),
                        x_grad.stride(1), At.stride(0), At.stride(1),
                        logits.stride(0), logits.stride(1), idx_N_group=
                        idx_N_group, N_group=N_group, V=V, N=N, H=H)
        print('fwd config:', linear_xent_fwd_prep_bwd_kernel_matmul_t.
            best_config)
        print('dx config:', linear_xent_bwd_kernel_matmul_t_epilogue_dx.
            best_config)
        print('dA config:', linear_xent_bwd_kernel_matmul_t_epilogue_dA.
            best_config)
        ctx.mark_non_differentiable(y)
        ctx.save_for_backward(x_grad, At_grad.to(At.dtype))
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        x_grad, At_grad = ctx.saved_tensors
        return x_grad * grad_output, None, At_grad * grad_output, None, None
