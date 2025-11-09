# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/JonasGeiping/linear_cross_entropy_loss
# Source-Files: variants/final_fusion_parallel_splitN_V_usable.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_0e62yj5n/linear_cross_entropy_loss-main/variants/final_fusion_parallel_splitN_V_usable.py
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

@triton.autotune(configs=bwd_configs, key=['V', 'N', 'H'], prune_configs_by
    ={'early_config_prune': early_config_prune}, warmup=100, rep=500)
@triton.jit()
def linear_xent_bwd_dispatcher(logits_ptr, y_ptr, x_ptr, A_t_ptr, x_grad,
    At_grad, lse_global, stride_x_N, stride_x_H, stride_A_H, stride_A_V,
    stride_z_N, stride_z_V, z_regularization: tl.constexpr,
    fp32_grad_accumulators: tl.constexpr, reduction_ptr, ignore_index: tl.
    constexpr, idx_N_group, N_group, V: tl.constexpr, N: tl.constexpr, H:
    tl.constexpr, V_BLOCK_SIZE: tl.constexpr=128, N_BLOCK_SIZE: tl.
    constexpr=128, H_BLOCK_SIZE: tl.constexpr=128, GROUP_SIZE: tl.constexpr
    =32, SPLIT_N: tl.constexpr=2, SPLIT_V: tl.constexpr=2):
    idx_NV = tl.program_id(axis=0)
    tl.static_print(V_BLOCK_SIZE, N_BLOCK_SIZE, H_BLOCK_SIZE, GROUP_SIZE,
        SPLIT_N, SPLIT_V)
    if idx_NV < N_group // N_BLOCK_SIZE * SPLIT_V:
        linear_xent_bwd_kernel_matmul_t_epilogue_dx(logits_ptr, y_ptr,
            A_t_ptr, x_grad, lse_global, stride_x_N, stride_x_H, stride_A_H,
            stride_A_V, stride_z_N, stride_z_V, z_regularization,
            fp32_grad_accumulators, reduction_ptr, ignore_index,
            idx_N_group, N_group, V, N, H, V_BLOCK_SIZE, N_BLOCK_SIZE,
            H_BLOCK_SIZE, GROUP_SIZE, SPLIT_N, SPLIT_V)
    else:
        linear_xent_bwd_kernel_matmul_t_epilogue_dA(logits_ptr, y_ptr,
            x_ptr, At_grad, lse_global, stride_x_N, stride_x_H, stride_A_H,
            stride_A_V, stride_z_N, stride_z_V, z_regularization,
            fp32_grad_accumulators, reduction_ptr, ignore_index,
            idx_N_group, N_group, V, N, H, V_BLOCK_SIZE, N_BLOCK_SIZE,
            H_BLOCK_SIZE, GROUP_SIZE, SPLIT_N, SPLIT_V)


@triton.jit()
def linear_xent_bwd_kernel_matmul_t_epilogue_dA(z_nv_ptr, y_ptr, x_ptr,
    A_grad_ptr, lse_ptr, stride_x_N, stride_x_H, stride_A_H, stride_A_V,
    stride_z_N, stride_z_V, z_regularization: tl.constexpr,
    fp32_grad_accumulators: tl.constexpr, reduction_ptr, ignore_index: tl.
    constexpr, idx_N_group, N_group: tl.constexpr, V: tl.constexpr, N: tl.
    constexpr, H: tl.constexpr, V_BLOCK_SIZE: tl.constexpr, N_BLOCK_SIZE:
    tl.constexpr, H_BLOCK_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr,
    SPLIT_N: tl.constexpr, SPLIT_V: tl.constexpr):
    idx_V = (tl.program_id(axis=0) - N_group // N_BLOCK_SIZE * SPLIT_V
        ) // SPLIT_N
    idx_H = tl.program_id(axis=1)
    idx_N_tile = (tl.program_id(axis=0) - N_group // N_BLOCK_SIZE * SPLIT_V
        ) % SPLIT_N
    num_idx_V, num_idx_H = tl.num_programs(0
        ) - N_group // N_BLOCK_SIZE * SPLIT_V, tl.num_programs(1)
    idx_V, idx_H = tl.swizzle2d(idx_V, idx_H, num_idx_V // SPLIT_N,
        num_idx_H, GROUP_SIZE)
    N_split_offset = idx_N_tile * tl.cdiv(N_group, SPLIT_N)
    x_block_ptr = tl.make_block_ptr(base=x_ptr, shape=(N, H), strides=(
        stride_x_N, stride_x_H), offsets=(idx_N_group * N_group +
        N_split_offset, idx_H * H_BLOCK_SIZE), block_shape=(N_BLOCK_SIZE,
        H_BLOCK_SIZE), order=(1, 0))
    z_block_ptr = tl.make_block_ptr(base=z_nv_ptr, shape=(N_group, V),
        strides=(stride_z_N, stride_z_V), offsets=(N_split_offset, idx_V *
        V_BLOCK_SIZE), block_shape=(N_BLOCK_SIZE, V_BLOCK_SIZE), order=(1, 0))
    N_range = N_split_offset + tl.arange(0, N_BLOCK_SIZE)
    V_range = idx_V * V_BLOCK_SIZE + tl.arange(0, V_BLOCK_SIZE)
    reduction = tl.load(reduction_ptr)
    acc_dtype = (tl.float32 if fp32_grad_accumulators else A_grad_ptr.type.
        element_ty)
    A_grad_acc = tl.zeros((H_BLOCK_SIZE, V_BLOCK_SIZE), acc_dtype)
    for _ in range(0, tl.cdiv(N_group, N_BLOCK_SIZE * SPLIT_N)):
        y = tl.load(y_ptr + idx_N_group * N_group + N_range,
            eviction_policy='evict_last')
        lse = tl.load(lse_ptr + N_range, eviction_policy='evict_last')
        mask = y[:, None] == V_range[None, :]
        x_chunk = tl.load(x_block_ptr, eviction_policy='evict_first')
        z_j_to_k = tl.load(z_block_ptr, eviction_policy='evict_last')
        softmax_z = (z_j_to_k - lse[:, None]).exp()
        if z_regularization > 0:
            softmax_z += 2.0 * z_regularization * lse[:, None] * softmax_z
        z_grad = softmax_z - tl.where(mask, 1 / reduction, 0)
        valid_z_grad = tl.where((y == ignore_index)[:, None], 0.0, z_grad).to(
            x_ptr.type.element_ty)
        A_grad_acc = tl.dot(x_chunk.trans(), valid_z_grad, A_grad_acc,
            out_dtype=acc_dtype)
        x_block_ptr = tl.advance(x_block_ptr, [N_BLOCK_SIZE, 0])
        z_block_ptr = tl.advance(z_block_ptr, [N_BLOCK_SIZE, 0])
        N_range += N_BLOCK_SIZE
    if SPLIT_N == 1:
        A_grad_T_block_ptr = tl.make_block_ptr(base=A_grad_ptr, shape=(H, V
            ), strides=(stride_A_H, stride_A_V), offsets=(idx_H *
            H_BLOCK_SIZE, idx_V * V_BLOCK_SIZE), block_shape=(H_BLOCK_SIZE,
            V_BLOCK_SIZE), order=(0, 1))
        if idx_N_group > 0:
            tl.store(A_grad_T_block_ptr, tl.load(A_grad_T_block_ptr) +
                A_grad_acc.to(A_grad_ptr.type.element_ty))
        else:
            tl.store(A_grad_T_block_ptr, A_grad_acc.to(A_grad_ptr.type.
                element_ty))
    else:
        row_h = idx_H * H_BLOCK_SIZE + tl.arange(0, H_BLOCK_SIZE)
        row_v = idx_V * V_BLOCK_SIZE + tl.arange(0, V_BLOCK_SIZE)
        A_grad_T_simple_ptr = A_grad_ptr + row_h[:, None] * stride_A_H + row_v[
            None, :] * stride_A_V
        tl.atomic_add(A_grad_T_simple_ptr, A_grad_acc.to(A_grad_ptr.type.
            element_ty))


@triton.jit()
def linear_xent_bwd_kernel_matmul_t_epilogue_dx(z_nv_ptr, y_ptr, A_t_ptr,
    x_grad_ptr, lse_ptr, stride_x_N, stride_x_H, stride_A_H, stride_A_V,
    stride_z_N, stride_z_V, z_regularization: tl.constexpr,
    fp32_grad_accumulators: tl.constexpr, reduction_ptr, ignore_index: tl.
    constexpr, idx_N_group, N_group: tl.constexpr, V: tl.constexpr, N: tl.
    constexpr, H: tl.constexpr, V_BLOCK_SIZE: tl.constexpr, N_BLOCK_SIZE:
    tl.constexpr, H_BLOCK_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr,
    SPLIT_N: tl.constexpr, SPLIT_V: tl.constexpr):
    idx_N = tl.program_id(axis=0) // SPLIT_V
    idx_H = tl.program_id(axis=1)
    idx_V_tile = tl.program_id(axis=0) % SPLIT_V
    num_idx_N, num_idx_H = tl.num_programs(0) - triton.cdiv(V, V_BLOCK_SIZE
        ) * SPLIT_N, tl.num_programs(1)
    idx_N, idx_H = tl.swizzle2d(idx_N, idx_H, num_idx_N // SPLIT_V,
        num_idx_H, GROUP_SIZE)
    V_split_offset = idx_V_tile * tl.cdiv(V, SPLIT_V)
    A_t_block_ptr = tl.make_block_ptr(base=A_t_ptr, shape=(H, V), strides=(
        stride_A_H, stride_A_V), offsets=(idx_H * H_BLOCK_SIZE,
        V_split_offset), block_shape=(H_BLOCK_SIZE, V_BLOCK_SIZE), order=(0, 1)
        )
    z_block_ptr = tl.make_block_ptr(base=z_nv_ptr, shape=(N_group, V),
        strides=(stride_z_N, stride_z_V), offsets=(idx_N * N_BLOCK_SIZE,
        V_split_offset), block_shape=(N_BLOCK_SIZE, V_BLOCK_SIZE), order=(1, 0)
        )
    N_range = idx_N_group * N_group + idx_N * N_BLOCK_SIZE + tl.arange(0,
        N_BLOCK_SIZE)
    V_range = V_split_offset + tl.arange(0, V_BLOCK_SIZE)
    y = tl.load(y_ptr + N_range, eviction_policy='evict_last')
    lse = tl.load(lse_ptr + idx_N * N_BLOCK_SIZE + tl.arange(0,
        N_BLOCK_SIZE), eviction_policy='evict_last')
    reduction = tl.load(reduction_ptr)
    acc_dtype = (tl.float32 if fp32_grad_accumulators else x_grad_ptr.type.
        element_ty)
    x_grad_acc = tl.zeros((N_BLOCK_SIZE, H_BLOCK_SIZE), acc_dtype)
    for _ in range(0, tl.cdiv(V, V_BLOCK_SIZE * SPLIT_V)):
        mask = y[:, None] == V_range[None, :]
        A_v = tl.load(A_t_block_ptr, eviction_policy='evict_first')
        z_j_to_k = tl.load(z_block_ptr, eviction_policy='evict_last')
        softmax_z = (z_j_to_k - lse[:, None]).exp()
        if z_regularization > 0:
            softmax_z += 2.0 * z_regularization * lse[:, None] * softmax_z
        z_grad = softmax_z - tl.where(mask, 1 / reduction, 0.0)
        valid_z_grad = tl.where((y == ignore_index)[:, None], 0.0, z_grad).to(
            A_v.type.element_ty)
        x_grad_acc = tl.dot(valid_z_grad, A_v.trans(), x_grad_acc,
            out_dtype=acc_dtype)
        A_t_block_ptr = tl.advance(A_t_block_ptr, [0, V_BLOCK_SIZE])
        z_block_ptr = tl.advance(z_block_ptr, [0, V_BLOCK_SIZE])
        V_range += V_BLOCK_SIZE
    if SPLIT_V == 1:
        x_grad_block_ptr = tl.make_block_ptr(base=x_grad_ptr, shape=(N, H),
            strides=(stride_x_N, stride_x_H), offsets=(idx_N_group *
            N_group + idx_N * N_BLOCK_SIZE, idx_H * H_BLOCK_SIZE),
            block_shape=(N_BLOCK_SIZE, H_BLOCK_SIZE), order=(1, 0))
        tl.store(x_grad_block_ptr, x_grad_acc.to(x_grad_ptr.type.element_ty))
    else:
        row_n = idx_N_group * N_group + idx_N * N_BLOCK_SIZE + tl.arange(0,
            N_BLOCK_SIZE)
        row_h = idx_H * H_BLOCK_SIZE + tl.arange(0, H_BLOCK_SIZE)
        x_grad_simple_ptr = x_grad_ptr + row_n[:, None] * stride_x_N + row_h[
            None, :] * stride_x_H
        tl.atomic_add(x_grad_simple_ptr, x_grad_acc.to(x_grad_ptr.type.
            element_ty))


@triton.autotune(configs=fwd_configs, key=['V', 'N', 'H'], prune_configs_by
    ={'early_config_prune': early_config_prune}, warmup=100, rep=500)
@triton.jit
def linear_xent_fwd_kernel_matmul_t(x_ptr, y_ptr, A_t_ptr, z_nv_ptr,
    losses_ptr, lse_ptr, stride_x_N, stride_x_H, stride_A_H, stride_A_V,
    stride_z_N, stride_z_V, stride_lse_N, stride_lse_B, stride_loss_Nb,
    stride_loss_B, reduction_ptr, ignore_index: tl.constexpr, idx_N_group,
    N_group: tl.constexpr, V: tl.constexpr, N: tl.constexpr, H: tl.
    constexpr, V_BLOCK_SIZE: tl.constexpr, N_BLOCK_SIZE: tl.constexpr,
    H_BLOCK_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr):
    idx_N = tl.program_id(axis=0)
    idx_V_group = tl.program_id(axis=1)
    num_idx_N, num_idx_V_group = tl.num_programs(0), tl.num_programs(1)
    idx_N, idx_V_group = tl.swizzle2d(idx_N, idx_V_group, num_idx_N,
        num_idx_V_group, GROUP_SIZE)
    tl.static_print(V_BLOCK_SIZE, N_BLOCK_SIZE, H_BLOCK_SIZE, GROUP_SIZE)
    V_GROUP_SIZE: tl.constexpr = V_BLOCK_SIZE
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
    z_j_to_k = tl.zeros((N_BLOCK_SIZE, V_BLOCK_SIZE), dtype=tl.float32)
    for _ in range(H // H_BLOCK_SIZE):
        x_chunk = tl.load(x_block_ptr)
        A_v = tl.load(A_block_ptr)
        z_j_to_k = tl.dot(x_chunk, A_v, z_j_to_k)
        x_block_ptr = tl.advance(x_block_ptr, [0, H_BLOCK_SIZE])
        A_block_ptr = tl.advance(A_block_ptr, [H_BLOCK_SIZE, 0])
    V_range = idx_V_group * V_GROUP_SIZE + tl.arange(0, V_BLOCK_SIZE)
    N_range = idx_N_group * N_group + idx_N * N_BLOCK_SIZE + tl.arange(0,
        N_BLOCK_SIZE)
    y = tl.load(y_ptr + N_range)
    reduction = tl.load(reduction_ptr)
    mask = y[:, None] == tl.where(V_range != ignore_index, V_range, -1)[None, :
        ]
    loss = -tl.sum(tl.where(mask, z_j_to_k, float(0.0))) / reduction
    tl.store(z_block_ptr, (z_j_to_k + tl.log(1 / reduction)).to(z_nv_ptr.
        type.element_ty))
    m = tl.max(z_j_to_k, 1)
    zero_lse_constant: tl.constexpr = tl.log(1 / tl.cdiv(V, V_BLOCK_SIZE))
    lse = tl.where(y != ignore_index, tl.log(tl.sum(tl.exp(z_j_to_k - m[:,
        None]), axis=1)) + m, zero_lse_constant)
    lse_row_ptr = tl.make_block_ptr(base=lse_ptr, shape=(N_group, V // 128),
        strides=(stride_lse_N, stride_lse_B), offsets=(idx_N * N_BLOCK_SIZE,
        idx_V_group), block_shape=(N_BLOCK_SIZE, 1), order=(1, 0))
    loss_val_ptr = (losses_ptr + idx_N * stride_loss_Nb + idx_V_group *
        stride_loss_B)
    tl.store(loss_val_ptr, tl.load(loss_val_ptr) + loss)
    tl.store(lse_row_ptr, lse[:, None])


# Forward method (kernel launch code)
def _LinearCrossEntropyLoss_forward(ctx, x, y, At, ignore_index=-100,
    z_regularization=0.0, N_chunk_size: int=4096):
    with torch.cuda.device(x.device.index):
        N, H = x.shape
        H_A, V = At.shape
        assert H_A == H
        assert y.shape == (N,)
        N_group = min(N, N_chunk_size)
        assert N % 64 == 0
        assert V % 128 == 0
        assert H % 64 == 0
        At_grad = torch.zeros_like(At)
        x_grad = torch.zeros_like(x)
        lse_sum = torch.zeros((1,), dtype=torch.float32, device=x.device)
        lse_local = -1000000.0 * torch.ones(N_group, V // 128, dtype=torch.
            float32, device=x.device)
        losses = torch.zeros(N_group // 64, V // 128, dtype=torch.float32,
            device=x.device)
        logits = torch.empty((N_group, V), device=x.device, dtype=torch.float32
            )
        with torch.inference_mode():
            reduction = (y != ignore_index).sum()
            if reduction == 0:
                ctx.mark_non_differentiable(y)
                ctx.save_for_backward(x_grad, At_grad.to(At.dtype))
                return losses.sum()
            fwd_grid = lambda meta: (triton.cdiv(N_group, meta[
                'N_BLOCK_SIZE']), triton.cdiv(V, meta['V_BLOCK_SIZE']))
            bwd_grid_dx_dA = lambda meta: (triton.cdiv(N_group, meta[
                'N_BLOCK_SIZE']) * meta['SPLIT_V'] + triton.cdiv(V, meta[
                'V_BLOCK_SIZE']) * meta['SPLIT_N'], triton.cdiv(H, meta[
                'H_BLOCK_SIZE']))
            for idx_N_group in range(math.ceil(N / N_group)):
                linear_xent_fwd_kernel_matmul_t[fwd_grid](x, y, At, logits,
                    losses, lse_local, x.stride(0), x.stride(1), At.stride(
                    0), At.stride(1), logits.stride(0), logits.stride(1),
                    lse_local.stride(0), lse_local.stride(1), losses.stride
                    (0), losses.stride(1), reduction, ignore_index=
                    ignore_index, idx_N_group=idx_N_group, N_group=N_group,
                    V=V, N=N, H=H)
                V_BLOCK_SIZE = (linear_xent_fwd_kernel_matmul_t.best_config
                    .kwargs['V_BLOCK_SIZE'])
                buffer_extent = V // V_BLOCK_SIZE
                lse_global = lse_local[:, :buffer_extent].logsumexp(dim=1)
                lse_sum += (lse_global.sum() + z_regularization *
                    lse_global.pow(2).sum()) / reduction
                if x.requires_grad or At.requires_grad:
                    linear_xent_bwd_dispatcher[bwd_grid_dx_dA](logits, y, x,
                        At, x_grad, At_grad, lse_global, x_grad.stride(0),
                        x_grad.stride(1), At.stride(0), At.stride(1),
                        logits.stride(0), logits.stride(1),
                        z_regularization, LinearCrossEntropyLoss.
                        fp32_grad_accumulators, reduction, ignore_index=
                        ignore_index, idx_N_group=idx_N_group, N_group=
                        N_group, V=V, N=N, H=H)
        ctx.mark_non_differentiable(y)
        ctx.save_for_backward(x_grad, At_grad.to(At.dtype))
        return lse_sum + losses.sum()


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

# Backward method (kernel launch code)
@torch.inference_mode()
def _LinearCrossEntropyLoss_backward(ctx, grad_output):
    x_grad, At_grad = ctx.saved_tensors
    return x_grad * grad_output, None, At_grad * grad_output, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LinearCrossEntropyLoss(torch.autograd.Function):
    fp32_grad_accumulators: bool = False

    @staticmethod
    def forward(ctx, x, y, At, ignore_index=-100, z_regularization=0.0,
        N_chunk_size: int=4096):
        with torch.cuda.device(x.device.index):
            N, H = x.shape
            H_A, V = At.shape
            assert H_A == H
            assert y.shape == (N,)
            N_group = min(N, N_chunk_size)
            assert N % 64 == 0
            assert V % 128 == 0
            assert H % 64 == 0
            At_grad = torch.zeros_like(At)
            x_grad = torch.zeros_like(x)
            lse_sum = torch.zeros((1,), dtype=torch.float32, device=x.device)
            lse_local = -1000000.0 * torch.ones(N_group, V // 128, dtype=
                torch.float32, device=x.device)
            losses = torch.zeros(N_group // 64, V // 128, dtype=torch.
                float32, device=x.device)
            logits = torch.empty((N_group, V), device=x.device, dtype=torch
                .float32)
            with torch.inference_mode():
                reduction = (y != ignore_index).sum()
                if reduction == 0:
                    ctx.mark_non_differentiable(y)
                    ctx.save_for_backward(x_grad, At_grad.to(At.dtype))
                    return losses.sum()
                fwd_grid = lambda meta: (triton.cdiv(N_group, meta[
                    'N_BLOCK_SIZE']), triton.cdiv(V, meta['V_BLOCK_SIZE']))
                bwd_grid_dx_dA = lambda meta: (triton.cdiv(N_group, meta[
                    'N_BLOCK_SIZE']) * meta['SPLIT_V'] + triton.cdiv(V,
                    meta['V_BLOCK_SIZE']) * meta['SPLIT_N'], triton.cdiv(H,
                    meta['H_BLOCK_SIZE']))
                for idx_N_group in range(math.ceil(N / N_group)):
                    linear_xent_fwd_kernel_matmul_t[fwd_grid](x, y, At,
                        logits, losses, lse_local, x.stride(0), x.stride(1),
                        At.stride(0), At.stride(1), logits.stride(0),
                        logits.stride(1), lse_local.stride(0), lse_local.
                        stride(1), losses.stride(0), losses.stride(1),
                        reduction, ignore_index=ignore_index, idx_N_group=
                        idx_N_group, N_group=N_group, V=V, N=N, H=H)
                    V_BLOCK_SIZE = (linear_xent_fwd_kernel_matmul_t.
                        best_config.kwargs['V_BLOCK_SIZE'])
                    buffer_extent = V // V_BLOCK_SIZE
                    lse_global = lse_local[:, :buffer_extent].logsumexp(dim=1)
                    lse_sum += (lse_global.sum() + z_regularization *
                        lse_global.pow(2).sum()) / reduction
                    if x.requires_grad or At.requires_grad:
                        linear_xent_bwd_dispatcher[bwd_grid_dx_dA](logits,
                            y, x, At, x_grad, At_grad, lse_global, x_grad.
                            stride(0), x_grad.stride(1), At.stride(0), At.
                            stride(1), logits.stride(0), logits.stride(1),
                            z_regularization, LinearCrossEntropyLoss.
                            fp32_grad_accumulators, reduction, ignore_index
                            =ignore_index, idx_N_group=idx_N_group, N_group
                            =N_group, V=V, N=N, H=H)
            ctx.mark_non_differentiable(y)
            ctx.save_for_backward(x_grad, At_grad.to(At.dtype))
            return lse_sum + losses.sum()

    @staticmethod
    @torch.inference_mode()
    def backward(ctx, grad_output):
        x_grad, At_grad = ctx.saved_tensors
        return (x_grad * grad_output, None, At_grad * grad_output, None,
            None, None)
