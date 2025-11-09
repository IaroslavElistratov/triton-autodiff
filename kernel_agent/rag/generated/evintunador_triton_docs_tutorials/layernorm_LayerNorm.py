# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/evintunador/triton_docs_tutorials
# Source-Files: 08_layernorm/layernorm.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_4ctjy3wy/triton_docs_tutorials-main/08_layernorm/layernorm.py
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
import time

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def _layernorm_forward(x_ptr, y_ptr, w_ptr, b_ptr, mean_ptr, rstd_ptr,
    stride_M, N, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    x_ptr += row * stride_M
    y_ptr += row * stride_M
    sum_accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        x_ptrs = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        sum_accumulator += x_ptrs
    mean = tl.sum(sum_accumulator, axis=0) / N
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        x_ptrs = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        diff = tl.where(cols < N, x_ptrs - mean, 0.0)
        acc += diff * diff
    var = tl.sum(acc, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w_ptrs = tl.load(w_ptr + cols, mask=mask)
        b_ptrs = tl.load(b_ptr + cols, mask=mask)
        x_ptrs = tl.load(x_ptr + cols, mask=mask)
        x_hat = (x_ptrs - mean) * rstd
        y = x_hat * w_ptrs + b_ptrs
        tl.store(y_ptr + cols, y, mask=mask)


# Forward method (kernel launch code)
def _LayerNorm_forward(ctx, x, normalized_shape, weight, bias, eps):
    M, N = x.reshape(-1, x.shape[-1]).shape
    mean = torch.empty((M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    y = torch.empty_like(x)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    _layernorm_forward[M,](x, y, weight, bias, mean, rstd, x.stride(0), N,
        eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    ctx.save_for_backward(x, weight, bias, mean, rstd)
    ctx.BLOCK_SIZE = BLOCK_SIZE
    ctx.num_warps = num_warps
    ctx.eps = eps
    return y


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _layernorm_backward_dLdw_dLdb(dLdw_intermediate_ptr,
    dLdb_intermediate_ptr, dLdw_ptr, dLdb_ptr, GROUP_SIZE, N, BLOCK_SIZE_M:
    tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    PID = tl.program_id(0)
    col_ptrs = PID * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dLdw_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    dLdb_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, GROUP_SIZE, BLOCK_SIZE_M):
        row_ptrs = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (row_ptrs[:, None] < GROUP_SIZE) & (col_ptrs[None, :] < N)
        offsets = row_ptrs[:, None] * N + col_ptrs[None, :]
        dLdw_acc += tl.load(dLdw_intermediate_ptr + offsets, mask=mask,
            other=0.0)
        dLdb_acc += tl.load(dLdb_intermediate_ptr + offsets, mask=mask,
            other=0.0)
    sum_dLdw = tl.sum(dLdw_acc, axis=0)
    sum_dLdb = tl.sum(dLdb_acc, axis=0)
    tl.store(dLdw_ptr + col_ptrs, sum_dLdw, mask=col_ptrs < N)
    tl.store(dLdb_ptr + col_ptrs, sum_dLdb, mask=col_ptrs < N)


@triton.jit
def _layernorm_backward_dLdx(x_ptr, dLdx_ptr, dLdy_ptr, w_ptr,
    dLdw_intermediate_ptr, dLdb_intermediate_ptr, mean_ptr, rstd_ptr,
    locks_ptr, stride, N, GROUP_SIZE: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
    ):
    """
    there's a weird grouping strategy being used here for the _dLdw and _dLdb that has visuals on the website
    https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html#sphx-glr-getting-started-tutorials-05-layer-norm-py
    the idea is that each pid is assigned some subset of rows (which are interleaved rather than next to each other)
    and it's that pid's job to accumulate the gradients over all of the rows it has been assigned
    then once each pid is done, in the next kernel we'll accumulate all of those individiual partial sums
    """
    PID = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    x_ptr += PID * stride
    dLdx_ptr += PID * stride
    dLdy_ptr += PID * stride
    x = tl.load(x_ptr + cols, mask=mask, other=0).to(tl.float32)
    dLdy = tl.load(dLdy_ptr + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
    mean = tl.load(mean_ptr + PID)
    rstd = tl.load(rstd_ptr + PID)
    x_normalized = tl.where(mask, (x - mean) * rstd, 0.0)
    dydx_normed = tl.where(mask, w * dLdy, 0.0)
    c1 = tl.sum(x_normalized * dydx_normed, axis=0) / N
    c2 = tl.sum(dydx_normed, axis=0) / N
    dLdx = (dydx_normed - (x_normalized * c1 + c2)) * rstd
    tl.store(dLdx_ptr + cols, dLdx, mask=mask)
    dLdw_contribution = (dLdy * x_normalized).to(w.dtype)
    dLdb_contribution = dLdy.to(w.dtype)
    """
    Now we'd like to take our single contributions to dLdw and dLdb and somehow aggregate them with
    the portions that all of the other PIDs have calculated.
    The reason this aggregation has to happen is because the input x is of shape (M, N) while
    the weights and biases are of shape (N), meaning they receive gradients from all M rows of x,
    and this PID holds the gradient of one of those rows, but it's not easy to communicate
    that information between PIDs.
    The specific operation to do between all these rows is to sum them up, but we can't just
    naively tl.load(), then add our row, then tl.store() because all of the PIDs would do so
    at slightly different and completely unpredictable times, meaning all the tl.store() calls
    would overwrite each other.
    What we need first a way to ensure that only one PID at a time does the read, flop, and 
    write while all the other PIDs others wait their turn.
    For this we can use what's called a lock, which is a way for us to ensure that only one 
    PID can work on a given part of a tensor in DRAM at a time, AKA "locking" it.

    However, even that's not great because if only one PID can do work at a time and we have a 
    lot of PIDs, then that's a whole lot of time leaving a large majority of the GPU sitting 
    idle while they wait in line. 
    What we need then is a way for GROUPS of PIDs to work sequentially with a lock while each
    group works in parallel to the others. 
    This is why we created dLdw_intermediate and dLdb_intermediate, each of which has shape 
    (GROUP_SIZE, N).
    We're going to assign every PID to a group, and then use our locking mechanism to ensure
    that only (M // GROUP_SIZE) PIDs attempt to wait around for their turn to work on a row
    of dLdw_intermediate and dLdb_intermediate at a time.
    In this way we've now gone from a sequential process with M steps to one with 
    (M // GROUP_SIZE) steps. 
    Then in the next kernel we'll take these (GROUP_SIZE, N) matrices and reduce them further
    down to the desired shape (N) matrices of dLdw and dLdb.

    But how do locks actually work?
    In this case we've got a tensor of shape (2 * GROUP_SIZE) and datatype int32 that's
    initialized to all zeroes. 
    The first GROUP_SIZE entries are for holding an indicator of the state of that lock;
    0 means unlocked and 1 means locked for the row of dLdw_intermediate and dLdb_intermediate 
    that it corresponds to.
    The latter GROUP_SIZE entries are for holding an indicator of whether this lock has
    ever been used before, which is useful because we'll want to run different code if
    this PID happens to be the first one to add its values to dLdw_intermediate and dLdb_intermediate.
    To use the lock, we check if the entry corresponding to the group that our PID is
    in is locked or unlocked:
    - if it's locked, then we wait and check again in a moment until it's unlocked
    - if it's unlocked then we'll lock it, load the current value of our group's row of 
        dLdw_intermediate and dLdb_intermediate, add our dLdw_contribution and dLdb_contribution 
        respectively, write those new values back to DRAM, and finally unlock it
    """
    lock_id = PID % GROUP_SIZE
    locks_ptr += lock_id
    count_ptr = locks_ptr + GROUP_SIZE
    dLdw_intermediate_ptrs = dLdw_intermediate_ptr + lock_id * N + cols
    dLdb_intermediate_ptrs = dLdb_intermediate_ptr + lock_id * N + cols
    while tl.atomic_cas(locks_ptr, 0, 1) == 1:
        pass
    count = tl.load(count_ptr)
    if count == 0:
        tl.atomic_xchg(count_ptr, 1)
    else:
        dLdw_contribution += tl.load(dLdw_intermediate_ptrs, mask=mask)
        dLdb_contribution += tl.load(dLdb_intermediate_ptrs, mask=mask)
    tl.store(dLdw_intermediate_ptrs, dLdw_contribution, mask=mask)
    tl.store(dLdb_intermediate_ptrs, dLdb_contribution, mask=mask)
    tl.atomic_xchg(locks_ptr, 0)


# Backward method (kernel launch code)
def _LayerNorm_backward(ctx, dLdy):
    """
        In the backward pass we receive a Tensor containing the gradient of the loss with respect to the output, and 
        we need to compute the gradient of the loss with respect to the input(s).
        """
    x, w, b, mean, rstd = ctx.saved_tensors
    M, N = x.reshape(-1, x.shape[-1]).shape
    dLdw = torch.empty((N,), dtype=w.dtype, device=w.device)
    dLdb = torch.empty((N,), dtype=w.dtype, device=w.device)
    dLdx = torch.empty_like(dLdy)
    GROUP_SIZE = 64
    if N <= 8192:
        GROUP_SIZE = 96
    if N <= 4096:
        GROUP_SIZE = 128
    if N <= 1024:
        GROUP_SIZE = 256
    dLdw_intermediate = torch.zeros((GROUP_SIZE, N), dtype=x.dtype, device=
        w.device)
    dLdb_intermediate = torch.zeros((GROUP_SIZE, N), dtype=x.dtype, device=
        w.device)
    locks = torch.zeros(2 * GROUP_SIZE, dtype=torch.int32, device=w.device)
    _layernorm_backward_dLdx[M,](x, dLdx, dLdy, w, dLdw_intermediate,
        dLdb_intermediate, mean, rstd, locks, x.stride(0), N, GROUP_SIZE=
        GROUP_SIZE, BLOCK_SIZE_N=ctx.BLOCK_SIZE, num_warps=ctx.num_warps)
    grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
    _layernorm_backward_dLdw_dLdb[grid](dLdw_intermediate,
        dLdb_intermediate, dLdw, dLdb, min(GROUP_SIZE, M), N, BLOCK_SIZE_M=
        32, BLOCK_SIZE_N=128)
    return dLdx, None, dLdw, dLdb, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LayerNorm(torch.autograd.Function):
    """
    We can implement our own custom functions that play nice with PyTorch's autograd graph
    by subclassing torch.autograd.Function and implementing the forward and backward passes
    with static methods forward() and backward(). 
    """

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        M, N = x.reshape(-1, x.shape[-1]).shape
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
        y = torch.empty_like(x)
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError(
                "This layer norm doesn't support feature dim >= 64KB.")
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        _layernorm_forward[M,](x, y, weight, bias, mean, rstd, x.stride(0),
            N, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dLdy):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss with respect to the output, and 
        we need to compute the gradient of the loss with respect to the input(s).
        """
        x, w, b, mean, rstd = ctx.saved_tensors
        M, N = x.reshape(-1, x.shape[-1]).shape
        dLdw = torch.empty((N,), dtype=w.dtype, device=w.device)
        dLdb = torch.empty((N,), dtype=w.dtype, device=w.device)
        dLdx = torch.empty_like(dLdy)
        GROUP_SIZE = 64
        if N <= 8192:
            GROUP_SIZE = 96
        if N <= 4096:
            GROUP_SIZE = 128
        if N <= 1024:
            GROUP_SIZE = 256
        dLdw_intermediate = torch.zeros((GROUP_SIZE, N), dtype=x.dtype,
            device=w.device)
        dLdb_intermediate = torch.zeros((GROUP_SIZE, N), dtype=x.dtype,
            device=w.device)
        locks = torch.zeros(2 * GROUP_SIZE, dtype=torch.int32, device=w.device)
        _layernorm_backward_dLdx[M,](x, dLdx, dLdy, w, dLdw_intermediate,
            dLdb_intermediate, mean, rstd, locks, x.stride(0), N,
            GROUP_SIZE=GROUP_SIZE, BLOCK_SIZE_N=ctx.BLOCK_SIZE, num_warps=
            ctx.num_warps)
        grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
        _layernorm_backward_dLdw_dLdb[grid](dLdw_intermediate,
            dLdb_intermediate, dLdw, dLdb, min(GROUP_SIZE, M), N,
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=128)
        return dLdx, None, dLdw, dLdb, None
