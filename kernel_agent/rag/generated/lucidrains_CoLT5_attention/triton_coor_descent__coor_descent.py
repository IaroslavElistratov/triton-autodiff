# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/lucidrains/CoLT5-attention
# Source-Files: colt5_attention/triton_coor_descent.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_sb6w1bet/CoLT5-attention-main/colt5_attention/triton_coor_descent.py
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
from math import exp

def calc_num_warps(block_size):
    num_warps = 4
    if block_size >= 2048:
        num_warps = 8
    if block_size >= 4096:
        num_warps = 16
    return num_warps


def exists(val):
    return val is not None


def num_to_groups(num, groups):
    assert 0 < groups <= num
    floor = num // groups
    remainder = num % groups
    out = []
    for ind in range(groups):
        out.append(floor + int(ind < remainder))
    assert sum(out) == num
    return out


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def coor_descent_kernel_forward(a_ptr, b_ptr, input_ptr, mask_ptr, k_ptr,
    a_iter_stride, b_row_stride, b_iter_stride, input_row_stride,
    mask_row_stride, n_iters, current_eps, eps_decay, eps, n_cols,
    BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols
    mask_start_ptr = mask_ptr + row_idx * mask_row_stride
    mask_ptrs = mask_start_ptr + col_offsets
    mask_ints = tl.load(mask_ptrs, mask=col_mask, other=0)
    mask = mask_ints == 1
    a_ptr = a_ptr + row_idx
    a = tl.load(a_ptr)
    b_start_ptr = b_ptr + row_idx * b_row_stride
    b_ptrs = b_start_ptr + col_offsets
    b = tl.load(b_ptrs, mask=col_mask, other=0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    input_ptrs = row_start_ptr + col_offsets
    s = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    k_ptr = k_ptr + row_idx
    k = tl.load(k_ptr)
    logk = tl.log(k)
    for _ in range(n_iters):
        a = (s + b) / current_eps
        a = tl.where(mask, a, -float('inf'))
        a_max = tl.max(a, axis=0)
        a_minus_max = tl.where(mask, a - a_max, -float('inf'))
        exp = tl.exp(a_minus_max)
        sum_exp = tl.sum(exp, axis=0)
        log_sum_exp = tl.log(sum_exp) + a_max
        a = current_eps * (logk - log_sum_exp)
        b = s + a
        b = tl.where(b >= 0.0, -b, 0.0)
        current_eps *= eps_decay
        if current_eps < eps:
            current_eps = eps
    next_a_ptrs = a_ptr + a_iter_stride
    next_b_ptrs = b_ptrs + b_iter_stride
    tl.store(next_a_ptrs, a)
    tl.store(next_b_ptrs, b, mask=col_mask)


def default(val, d):
    return val if exists(val) else d


# Forward method (kernel launch code)
@custom_fwd
def __coor_descent_forward(ctx, x, n_iters, k, eps, eps_init, eps_decay,
    mask, checkpoint_segments):
    assert n_iters > 0
    assert x.is_cuda, 'triton coordinate descent must be on cuda'
    batch, requires_grad, device, dtype = x.shape[0
        ], x.requires_grad, x.device, x.dtype
    if not exists(mask):
        mask = torch.ones_like(x, dtype=torch.bool, device=x.device)
    x, shape = pack_one(x, '* n')
    mask, _ = pack_one(mask, '* n')
    x = x.masked_fill(~mask, -torch.finfo(x.dtype).max)
    mask_ints = mask.int()
    epsilons = []
    eps_init = default(eps_init, eps)
    current_eps = float(max(eps_init, eps))
    n_rows, n_cols = x.shape
    if isinstance(k, (int, float)):
        k = torch.full((n_rows,), k)
    assert k.numel() == n_rows
    k = k.to(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    assert BLOCK_SIZE <= 131072, 'the maximum block size allowed is 131072 for triton cuda kernel - set the `route_block_size` for the CoordinateDescentRouter to be this value or less in order to uniformly route to get around this limitation'
    num_warps = calc_num_warps(BLOCK_SIZE)
    checkpointed_a = torch.empty((checkpoint_segments + 1, n_rows), device=
        device, dtype=dtype)
    checkpointed_b = torch.empty((checkpoint_segments + 1, n_rows, n_cols),
        device=device, dtype=dtype)
    checkpointed_a[0] = torch.zeros_like(k)
    checkpointed_b[0] = -x
    for ind, segment_iters in enumerate(num_to_groups(n_iters,
        checkpoint_segments)):
        is_last = ind == checkpoint_segments - 1
        epsilons.append(current_eps)
        coor_descent_kernel_forward[n_rows,](checkpointed_a[ind],
            checkpointed_b[ind], x, mask_ints, k, checkpointed_a.stride(0),
            n_cols, checkpointed_b.stride(0), x.stride(0), mask_ints.stride
            (0), segment_iters, current_eps, eps_decay, eps, n_cols,
            num_warps=num_warps, BLOCK_SIZE=BLOCK_SIZE)
        current_eps *= eps_decay ** segment_iters
        current_eps = max(current_eps, eps)
    last_a, last_b = map(lambda t: t[-1], (checkpointed_a, checkpointed_b))
    y = torch.exp((last_a[..., None] + last_b + x) / current_eps)
    epsilons.append(current_eps)
    if requires_grad:
        checkpointed_a = checkpointed_a[:-1]
        checkpointed_b = checkpointed_b[:-1]
        ctx.args = n_iters, checkpoint_segments, epsilons, eps_decay, eps
        ctx.save_for_backward(x, y, k, mask, checkpointed_a, checkpointed_b)
    y = unpack_one(y, shape, '* n')
    return y


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def coor_descent_kernel_backward(dk_ptr, input_ptr, a_ptr, b_ptr, mask_ptr,
    ds_ptr, db_ptr, k_ptr, last_da_ptr, input_row_stride, b_row_stride,
    mask_row_stride, ds_row_stride, db_row_stride, n_iters, eps_init,
    eps_decay, eps, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols
    mask_start_ptr = mask_ptr + row_idx * mask_row_stride
    mask_ptrs = mask_start_ptr + col_offsets
    mask_ints = tl.load(mask_ptrs, mask=col_mask, other=0)
    mask = mask_ints == 1
    a_ptr = a_ptr + row_idx
    init_a = tl.load(a_ptr)
    b_start_ptr = b_ptr + row_idx * b_row_stride
    b_ptrs = b_start_ptr + col_offsets
    init_b = tl.load(b_ptrs, mask=mask, other=0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    input_ptrs = row_start_ptr + col_offsets
    s = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    k_ptr = k_ptr + row_idx
    k = tl.load(k_ptr)
    logk = tl.log(k)
    last_da_ptr = last_da_ptr + row_idx
    last_da = tl.load(last_da_ptr)
    ds_row_start_ptr = ds_ptr + row_idx * ds_row_stride
    ds_ptrs = ds_row_start_ptr + col_offsets
    ds = tl.load(ds_ptrs, mask=mask, other=0.0)
    db_row_start_ptr = db_ptr + row_idx * db_row_stride
    db_ptrs = db_row_start_ptr + col_offsets
    db = tl.load(db_ptrs, mask=mask, other=0.0)
    dk_ptr = dk_ptr + row_idx
    dk = tl.load(dk_ptr)
    for ind in range(n_iters):
        a = init_a
        b = init_b
        sa = s * 0
        softmax = s * 0
        current_eps = eps_init / eps_decay
        for _ in range(n_iters - ind):
            current_eps *= eps_decay
            if current_eps < eps:
                current_eps = eps
            sb = (s + b) / current_eps
            sb = tl.where(mask, sb, -float('inf'))
            sb_max = tl.max(sb, axis=0)
            sb_minus_max = tl.where(mask, sb - sb_max, -float('inf'))
            exp = tl.exp(sb_minus_max)
            sum_exp = tl.sum(exp, axis=0)
            softmax = exp / sum_exp
            log_sum_exp = tl.log(sum_exp) + sb_max
            a = current_eps * (logk - log_sum_exp)
            sa = s + a
            b = tl.where(sa > 0.0, -sa, 0.0)
        dsa = db * tl.where(sa > 0, -1.0, 0.0)
        ds += dsa
        da = tl.sum(dsa, axis=0) + last_da
        dk += da * current_eps
        dsb = da * -softmax
        ds += dsb
        db = dsb
        last_da *= 0.0
    tl.store(dk_ptr, dk)
    tl.store(ds_ptrs, ds, mask=col_mask)
    tl.store(db_ptrs, db, mask=col_mask)


# Backward method (kernel launch code)
@custom_bwd
def __coor_descent_backward(ctx, grad_probs):
    assert grad_probs.is_cuda
    batch = grad_probs.shape[0]
    n_iters, checkpoint_segments, epsilons, eps_decay, eps = ctx.args
    x, y, k, mask, checkpointed_a, checkpointed_b = ctx.saved_tensors
    grad_probs, shape = pack_one(grad_probs, '* n')
    if exists(mask):
        grad_probs = grad_probs.masked_fill(~mask, 0.0)
    n_rows, n_cols = grad_probs.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = calc_num_warps(BLOCK_SIZE)
    *epsilons, last_eps = epsilons
    ds = grad_probs * y / last_eps
    db = ds.clone()
    dk = torch.zeros_like(k)
    last_da = ds.sum(dim=-1)
    mask_int = mask.int()
    items = zip(reversed(checkpointed_a.unbind(dim=0)), reversed(
        checkpointed_b.unbind(dim=0)), reversed(num_to_groups(n_iters,
        checkpoint_segments)), reversed(epsilons))
    for ind, (init_a, init_b, segment_iters, eps_init) in enumerate(items):
        is_first = ind == 0
        coor_descent_kernel_backward[n_rows,](dk, x, init_a, init_b,
            mask_int, ds, db, k, last_da if is_first else torch.zeros_like(
            last_da), x.stride(0), init_b.stride(0), mask_int.stride(0), ds
            .stride(0), db.stride(0), segment_iters, eps_init, eps_decay,
            eps, n_cols, num_warps=num_warps, BLOCK_SIZE=BLOCK_SIZE)
    ds += -db
    ds = unpack_one(ds, shape, '* n')
    if not k.requires_grad:
        dk = None
    else:
        dk /= k
    return ds, None, dk, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _coor_descent(autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, x, n_iters, k, eps, eps_init, eps_decay, mask,
        checkpoint_segments):
        assert n_iters > 0
        assert x.is_cuda, 'triton coordinate descent must be on cuda'
        batch, requires_grad, device, dtype = x.shape[0
            ], x.requires_grad, x.device, x.dtype
        if not exists(mask):
            mask = torch.ones_like(x, dtype=torch.bool, device=x.device)
        x, shape = pack_one(x, '* n')
        mask, _ = pack_one(mask, '* n')
        x = x.masked_fill(~mask, -torch.finfo(x.dtype).max)
        mask_ints = mask.int()
        epsilons = []
        eps_init = default(eps_init, eps)
        current_eps = float(max(eps_init, eps))
        n_rows, n_cols = x.shape
        if isinstance(k, (int, float)):
            k = torch.full((n_rows,), k)
        assert k.numel() == n_rows
        k = k.to(x)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        assert BLOCK_SIZE <= 131072, 'the maximum block size allowed is 131072 for triton cuda kernel - set the `route_block_size` for the CoordinateDescentRouter to be this value or less in order to uniformly route to get around this limitation'
        num_warps = calc_num_warps(BLOCK_SIZE)
        checkpointed_a = torch.empty((checkpoint_segments + 1, n_rows),
            device=device, dtype=dtype)
        checkpointed_b = torch.empty((checkpoint_segments + 1, n_rows,
            n_cols), device=device, dtype=dtype)
        checkpointed_a[0] = torch.zeros_like(k)
        checkpointed_b[0] = -x
        for ind, segment_iters in enumerate(num_to_groups(n_iters,
            checkpoint_segments)):
            is_last = ind == checkpoint_segments - 1
            epsilons.append(current_eps)
            coor_descent_kernel_forward[n_rows,](checkpointed_a[ind],
                checkpointed_b[ind], x, mask_ints, k, checkpointed_a.stride
                (0), n_cols, checkpointed_b.stride(0), x.stride(0),
                mask_ints.stride(0), segment_iters, current_eps, eps_decay,
                eps, n_cols, num_warps=num_warps, BLOCK_SIZE=BLOCK_SIZE)
            current_eps *= eps_decay ** segment_iters
            current_eps = max(current_eps, eps)
        last_a, last_b = map(lambda t: t[-1], (checkpointed_a, checkpointed_b))
        y = torch.exp((last_a[..., None] + last_b + x) / current_eps)
        epsilons.append(current_eps)
        if requires_grad:
            checkpointed_a = checkpointed_a[:-1]
            checkpointed_b = checkpointed_b[:-1]
            ctx.args = n_iters, checkpoint_segments, epsilons, eps_decay, eps
            ctx.save_for_backward(x, y, k, mask, checkpointed_a, checkpointed_b
                )
        y = unpack_one(y, shape, '* n')
        return y

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_probs):
        assert grad_probs.is_cuda
        batch = grad_probs.shape[0]
        n_iters, checkpoint_segments, epsilons, eps_decay, eps = ctx.args
        x, y, k, mask, checkpointed_a, checkpointed_b = ctx.saved_tensors
        grad_probs, shape = pack_one(grad_probs, '* n')
        if exists(mask):
            grad_probs = grad_probs.masked_fill(~mask, 0.0)
        n_rows, n_cols = grad_probs.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = calc_num_warps(BLOCK_SIZE)
        *epsilons, last_eps = epsilons
        ds = grad_probs * y / last_eps
        db = ds.clone()
        dk = torch.zeros_like(k)
        last_da = ds.sum(dim=-1)
        mask_int = mask.int()
        items = zip(reversed(checkpointed_a.unbind(dim=0)), reversed(
            checkpointed_b.unbind(dim=0)), reversed(num_to_groups(n_iters,
            checkpoint_segments)), reversed(epsilons))
        for ind, (init_a, init_b, segment_iters, eps_init) in enumerate(items):
            is_first = ind == 0
            coor_descent_kernel_backward[n_rows,](dk, x, init_a, init_b,
                mask_int, ds, db, k, last_da if is_first else torch.
                zeros_like(last_da), x.stride(0), init_b.stride(0),
                mask_int.stride(0), ds.stride(0), db.stride(0),
                segment_iters, eps_init, eps_decay, eps, n_cols, num_warps=
                num_warps, BLOCK_SIZE=BLOCK_SIZE)
        ds += -db
        ds = unpack_one(ds, shape, '* n')
        if not k.requires_grad:
            dk = None
        else:
            dk /= k
        return ds, None, dk, None, None, None, None, None
