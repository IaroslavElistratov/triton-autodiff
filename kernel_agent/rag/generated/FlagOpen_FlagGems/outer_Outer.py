# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/FlagOpen/FlagGems
# Source-Files: src/flag_gems/runtime/backend/_cambricon/fused/outer.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_kcv6dce2/FlagGems-master/src/flag_gems/runtime/backend/_cambricon/fused/outer.py
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
from math import log

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@libentry()
@triton.autotune(configs=[triton.Config({'tile_size': 1024}, num_stages=3,
    num_warps=1), triton.Config({'tile_size': 2048}, num_stages=3,
    num_warps=1), triton.Config({'tile_size': 4096}, num_stages=3,
    num_warps=1), triton.Config({'tile_size': 8192}, num_stages=3,
    num_warps=1), triton.Config({'tile_size': 16384}, num_stages=3,
    num_warps=1), triton.Config({'tile_size': 21760}, num_stages=3,
    num_warps=1), triton.Config({'tile_size': 32768}, num_stages=3,
    num_warps=1)], key=['M', 'N'], prune_configs_by={'early_config_prune':
    early_config_prune})
@triton.jit
def outer_kernel(lhs, rhs, res, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.
    constexpr, NEED_LOOP_N: tl.constexpr):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(axis=0)
    m_tasks_num = tl.cdiv(M, BLOCK_M)
    n_tasks_num = tl.cdiv(N, BLOCK_N)
    total_tasks_num = m_tasks_num * n_tasks_num
    if NEED_LOOP_N:
        for task_id in range(pid, total_tasks_num, num_jobs):
            start_m = task_id // n_tasks_num
            start_n = task_id % n_tasks_num
            offset_m = tl.arange(0, BLOCK_M) + start_m * BLOCK_M
            lhs_val = tl.load(lhs + offset_m, mask=offset_m < M)
            offset_n = tl.arange(0, BLOCK_N) + start_n * BLOCK_N
            rhs_val = tl.load(rhs + offset_n, mask=offset_n < N)
            res_val = lhs_val[:, None] * rhs_val[None, :]
            offset_r = offset_m[:, None] * N + offset_n[None, :]
            tl.store(res + offset_r, res_val, mask=(offset_m[:, None] < M) &
                (offset_n[None, :] < N))
    else:
        offset_n = tl.arange(0, BLOCK_N)
        rhs_val = tl.load(rhs + offset_n)
        for task_id in range(pid, total_tasks_num, num_jobs):
            start_m = task_id // n_tasks_num
            offset_m = tl.arange(0, BLOCK_M) + start_m * BLOCK_M
            lhs_val = tl.load(lhs + offset_m, mask=offset_m < M)
            res_val = lhs_val[:, None] * rhs_val[None, :]
            offset_r = offset_m[:, None] * N + offset_n[None, :]
            tl.store(res + offset_r, res_val, mask=(offset_m[:, None] < M) &
                (offset_n[None, :] < N))


def outer_(lhs, rhs):
    m = lhs.shape[0]
    n = rhs.shape[0]
    res_shape = [m, n]
    res = torch.empty(res_shape, dtype=lhs.dtype, device='mlu')
    grid = lambda META: (min(triton.cdiv(m, META['BLOCK_M']) * triton.cdiv(
        n, META['BLOCK_N']), TOTAL_CORE_NUM),)
    outer_kernel[grid](lhs, rhs, res, m, n)
    return res


# Forward method (kernel launch code)
def _Outer_forward(ctx, inp, weight):
    logger.debug('GEMS_CAMBRICON OUTER')
    assert inp.ndim == 1 and weight.ndim == 1, 'Invalid input'
    out = outer_(inp, weight)
    ctx.save_for_backward(inp, weight)
    return out


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@libentry()
@triton.autotune(configs=runtime.get_tuned_config('mv'), key=['M', 'N'],
    prune_configs_by={'early_config_prune': config_prune})
@triton.heuristics(values={'ONE_TILE_PER_CTA': lambda args: args['M'] <=
    args['BLOCK_M']})
@triton.jit
def mv_kernel(A, B, C, N, M, stride_an, stride_am, stride_bm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, ONE_TILE_PER_CTA: tl.
    constexpr):
    pid = tl.program_id(0)
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    offset_m = tl.arange(0, BLOCK_M)[None, :]
    n_mask = offset_n < N
    A_ptrs = A + offset_n * stride_an + offset_m * stride_am
    B_ptrs = B + offset_m * stride_bm
    if ONE_TILE_PER_CTA:
        a = tl.load(A_ptrs, mask=n_mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptrs).to(tl.float32)
        acc = tl.sum(a * b, axis=1)
        C_ptrs = C + offset_n * stride_cn
        tl.store(C_ptrs, acc[:, None], mask=n_mask)
    else:
        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        for m in range(0, M, BLOCK_M):
            m_mask = m + offset_m < M
            a = tl.load(A_ptrs, mask=n_mask & m_mask, other=0.0).to(tl.float32)
            b = tl.load(B_ptrs, mask=m_mask, other=0.0).to(tl.float32)
            acc += a * b
            A_ptrs += BLOCK_M * stride_am
            B_ptrs += BLOCK_M * stride_bm
        acc = tl.sum(acc, axis=1)
        C_ptrs = C + offset_n * stride_cn
        tl.store(C_ptrs, acc[:, None], mask=n_mask)


def mv(inp, vec):
    logger.debug('GEMS_CAMBRICON MV')
    assert inp.shape[1] == vec.shape[0], 'incompatible dimensions'
    N, M = inp.shape
    out = torch.empty((N,), device=inp.device, dtype=inp.dtype)
    grid = lambda META: (triton.cdiv(N, META['BLOCK_N']),)
    with torch_device_fn.device(inp.device):
        mv_kernel[grid](inp, vec, out, N, M, inp.stride(0), inp.stride(1),
            vec.stride(0), out.stride(0))
    return out


# Backward method (kernel launch code)
def _Outer_backward(ctx, out_grad):
    logger.debug('GEMS_CAMBRICON OUTER VJP')
    assert out_grad.ndim == 2, 'invalide out_grad shape'
    inp, weight = ctx.saved_tensors
    inp_grad = mv(out_grad, weight)
    weight_grad = mv(out_grad.t(), inp)
    return inp_grad, weight_grad


# ============================================================
# autograd.Function Class Definition
# ============================================================

class Outer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, weight):
        logger.debug('GEMS_CAMBRICON OUTER')
        assert inp.ndim == 1 and weight.ndim == 1, 'Invalid input'
        out = outer_(inp, weight)
        ctx.save_for_backward(inp, weight)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug('GEMS_CAMBRICON OUTER VJP')
        assert out_grad.ndim == 2, 'invalide out_grad shape'
        inp, weight = ctx.saved_tensors
        inp_grad = mv(out_grad, weight)
        weight_grad = mv(out_grad.t(), inp)
        return inp_grad, weight_grad
