# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/toyaix/TritonLLM
# Source-Files: tritonllm/triton_kernels/topk.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_wam_m4a5/TritonLLM-main/tritonllm/triton_kernels/topk.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def _topk_forward(X, stride_xm, Yv, Yi, stride_ym, USE_PROVIDED_INDX: tl.
    constexpr, Bits, stride_rm: tl.constexpr, stride_rn: tl.constexpr,
    n_rows, n_expts_tot, S, BLOCK_S: tl.constexpr, s_blocks, APPLY_SOFTMAX:
    tl.constexpr, BLOCK_M: tl.constexpr, N_EXPTS_PAD: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    if isinstance(n_rows, tl.tensor) and n_rows.dtype.is_ptr():
        n_rows = tl.load(n_rows)
    if pid < s_blocks:
        tl.store(S + BLOCK_S * pid + tl.arange(0, BLOCK_S), tl.zeros([
            BLOCK_S], tl.int32))
    if pid * BLOCK_M >= n_rows:
        return
    tl.static_assert(BLOCK_N % 32 == 0)
    tl.static_assert(N_EXPTS_PAD % BLOCK_N == 0)
    x_dtype: tl.constexpr = X.dtype.element_ty
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_y_n = tl.arange(0, N_EXPTS_ACT)
    mask_m = offs_m[:, None] < n_rows
    if USE_PROVIDED_INDX:
        Yi_ptrs = Yi + offs_m[:, None] * stride_ym + offs_y_n[None, :]
        y_indices = tl.load(Yi_ptrs, mask=mask_m)
        Xv_ptrs = X + offs_m[:, None] * stride_xm + y_indices
        y_values = tl.load(Xv_ptrs, mask=mask_m)
    else:
        y_values, y_indices = streaming_topk(X, stride_xm, n_expts_tot,
            offs_m, mask_m, N_EXPTS_PAD, N_EXPTS_ACT, BLOCK_N)
    if APPLY_SOFTMAX:
        y_values = tl.softmax(y_values.to(tl.float32), dim=1, keep_dims=True
            ).to(x_dtype)
    Yv_ptrs = Yv + offs_m[:, None] * stride_ym + offs_y_n[None, :]
    tl.store(Yv_ptrs, y_values, mask=mask_m)
    if not USE_PROVIDED_INDX:
        Yi_ptrs = Yi + offs_m[:, None] * stride_ym + offs_y_n[None, :]
        tl.store(Yi_ptrs, y_indices, mask=mask_m)
    y_div = y_indices // 32
    y_rem = y_indices % 32
    loop_iterations = N_EXPTS_PAD // BLOCK_N
    for i in range(loop_iterations):
        offs_r_n = tl.arange(0, BLOCK_N // 32) + i * (BLOCK_N // 32)
        y2 = tl.where(y_div[:, :, None] == offs_r_n[None, None, :], (1 <<
            y_rem)[:, :, None], 0)
        r = tl.reduce_or(y2, axis=1)
        BitsPtrs = Bits + offs_m[:, None] * stride_rm + offs_r_n[None, :
            ] * stride_rn
        tl.store(BitsPtrs, r, mask=mask_m)


@triton.jit
def fpval_to_key(x):
    tm, fm = get_topmask_and_fullmask(x)
    return x ^ tl.where(x & tm != 0, fm, tm)


@triton.jit
def get_topmask_and_fullmask(x):
    tl.static_assert(x.dtype.is_int_unsigned(),
        'floating-point value must be passed as bits')
    tm: tl.constexpr = 1 << -1 + x.dtype.primitive_bitwidth
    fm: tl.constexpr = (1 << x.dtype.primitive_bitwidth) - 1
    tm_arr = tl.full(x.shape, tm, dtype=x.dtype)
    fm_arr = tl.full(x.shape, fm, dtype=x.dtype)
    return tm_arr, fm_arr


@triton.jit
def indx_to_key(indx, N_EXPTS_PAD: tl.constexpr):
    return N_EXPTS_PAD - indx


@triton.jit
def key_to_fpval(x):
    tm, fm = get_topmask_and_fullmask(x)
    return x ^ tl.where(x & tm == 0, fm, tm)


@triton.jit
def key_to_indx(indx, N_EXPTS_PAD: tl.constexpr):
    return N_EXPTS_PAD - indx


@triton.jit
def streaming_topk(X, stride_xm, n_expts_tot, offs_m, mask_m, N_EXPTS_PAD:
    tl.constexpr, N_EXPTS_ACT: tl.constexpr, BLOCK_N: tl.constexpr):
    x_nbits: tl.constexpr = X.dtype.element_ty.primitive_bitwidth
    x_utype: tl.constexpr = tl.dtype(f'uint{x_nbits}')
    if x_nbits < 16:
        y_nbits: tl.constexpr = 32
    else:
        y_nbits: tl.constexpr = x_nbits * 2
    x_ultype: tl.constexpr = tl.dtype(f'uint{y_nbits}')
    x_dtype: tl.constexpr = X.dtype.element_ty
    loop_iterations: tl.constexpr = N_EXPTS_PAD // BLOCK_N - 1
    offs_x_n = loop_iterations * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_x_n[None, :] < n_expts_tot
    X_ptrs = X + offs_m[:, None] * stride_xm + offs_x_n[None, :]
    x = tl.load(X_ptrs, mask=mask_m & mask_n, other=float('-inf'))
    x = fpval_to_key(x.to(x_utype, bitcast=True))
    x = x.to(x_ultype) << 16 | indx_to_key(offs_x_n, N_EXPTS_PAD)[None, :]
    acc = tl.topk(x, N_EXPTS_ACT, dim=1)
    for _i in (tl.static_range if loop_iterations <= 4 else range)(
        loop_iterations):
        acc = tl.bitonic_merge(acc)
        X_ptrs -= BLOCK_N
        offs_x_n -= BLOCK_N
        x = tl.load(X_ptrs, mask=mask_m, other=float('-inf'))
        x = fpval_to_key(x.to(x_utype, bitcast=True))
        x = x.to(x_ultype) << 16 | indx_to_key(offs_x_n, N_EXPTS_PAD)[None, :]
        acc = tl.maximum(acc, tl.topk(x, N_EXPTS_ACT, dim=1))
    acc = acc << y_nbits - 16 | acc >> 16
    acc = tl.sort(acc, dim=1, descending=True)
    y_indices_raw = (acc >> y_nbits - 16).to(tl.uint32)
    y_indices = key_to_indx(y_indices_raw, N_EXPTS_PAD)
    y_values_raw = acc.to(x_utype)
    y_values = key_to_fpval(y_values_raw).to(x_dtype, bitcast=True)
    return y_values, y_indices


def stride(self, i=None):
    return self.storage.data.stride(
        ) if i is None else self.storage.data.stride(i)


def topk_forward(x, k, apply_softmax=True, dim=1, return_bitmatrix=True,
    y_indx=None, n_rows=None):
    if not isinstance(x, Tensor):
        x_shape = [x.shape[0] if n_rows is None else n_rows, x.shape[1]]
        x_shape_max = [x.shape[0], x.shape[1]]
        x = Tensor(x, shape=x_shape, shape_max=x_shape_max)
    cdiv = lambda a, b: (a + b - 1) // b
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_S = 128
    assert len(x.shape) == 2
    assert x.shape_max[-1] < 32768
    assert dim == 1
    assert return_bitmatrix
    n_rows, n_cols = x.shape
    n_rows_max, _ = x.shape_max
    dev = x.device
    y_vals = torch.empty((n_rows_max, k), dtype=x.dtype, device=dev)
    if y_indx is not None:
        use_provided_indx = True
    else:
        y_indx = torch.empty((n_rows_max, k), dtype=torch.int16, device=dev)
        use_provided_indx = False
    n_cols_pad = cdiv(n_cols, BLOCK_N) * BLOCK_N
    n_cols_words = n_cols_pad // 32
    bitmatrix = torch.empty((n_cols_words, cdiv(n_rows_max, 32) * 32),
        dtype=torch.uint32, device=dev)
    bitmatrix = torch.transpose(bitmatrix, 0, 1)[:n_rows_max]
    s_blocks = cdiv(n_cols, BLOCK_S)
    s_cols = s_blocks * BLOCK_S
    scratchpad = torch.empty((s_cols,), dtype=torch.int32, device=dev)
    pids = max(cdiv(n_rows_max, BLOCK_M), s_blocks)
    _topk_forward[pids,](x, x.stride(0), y_vals, y_indx, y_vals.stride(0),
        use_provided_indx, bitmatrix, bitmatrix.stride(0), bitmatrix.stride
        (1), n_rows, n_cols, scratchpad, BLOCK_S, s_blocks, BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N, APPLY_SOFTMAX=apply_softmax, N_EXPTS_PAD=
        n_cols_pad, N_EXPTS_ACT=k)
    bitmatrix_shape = [n_rows, n_cols_words * 32]
    bitmatrix_shape_max = [n_rows_max, None]
    bitmatrix = Bitmatrix(bitmatrix, shape=bitmatrix_shape, shape_max=
        bitmatrix_shape_max, scratchpad=scratchpad)
    return y_vals, y_indx, bitmatrix


# Forward method (kernel launch code)
def _TopK_forward(ctx, x, k, apply_softmax, dim, return_bitmatrix, y_indx,
    n_rows):
    y_vals, y_indx, bitmatrix = topk_forward(x, k, apply_softmax, dim,
        return_bitmatrix, y_indx, n_rows)
    ctx.save_for_backward(x, y_indx)
    ctx.apply_softmax = apply_softmax
    ctx.k = k
    ctx.n_rows = n_rows
    return y_vals, y_indx, bitmatrix


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _topk_backward(Yi, stride_ym, DY, stride_dym, X, stride_xm, DX,
    stride_dxm, n_rows, NRows, n_expts_tot, APPLY_SOFTMAX: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr, N_EXPTS_PAD: tl.constexpr):
    pid_m = tl.program_id(0)
    if NRows is not None:
        n_rows = tl.load(NRows)
    if pid_m >= n_rows:
        return
    Yi += pid_m * stride_ym
    DY += pid_m * stride_dym
    X += pid_m * stride_xm
    DX += pid_m * stride_dxm
    offs_xn = tl.arange(0, N_EXPTS_PAD)
    offs_yn = tl.arange(0, N_EXPTS_ACT)
    mask_xn = offs_xn < n_expts_tot
    y_indx = tl.load(Yi + offs_yn)
    x = tl.load(X + y_indx)
    x = x.to(tl.float32)
    y = tl.softmax(x)
    dy = tl.load(DY + offs_yn)
    dy = dy.to(tl.float32)
    s = tl.sum(y * dy, 0)
    tl.store(DX + offs_xn, 0, mask=mask_xn)
    tl.debug_barrier()
    if APPLY_SOFTMAX:
        dx = y * (dy - s)
    else:
        dx = dy
    tl.store(DX + y_indx, dx)


def topk_backward(x, y_indx, dy_vals, k, n_rows, apply_softmax):
    assert dy_vals.shape[-1] == k
    n_expts_pad = triton.next_power_of_2(x.shape[-1])
    dx = torch.empty_like(x)
    _topk_backward[dy_vals.shape[0],](y_indx, y_indx.stride(0), dy_vals,
        dy_vals.stride(0), x, x.stride(0), dx, dx.stride(0), x.shape[0],
        n_rows, x.shape[-1], APPLY_SOFTMAX=apply_softmax, N_EXPTS_ACT=k,
        N_EXPTS_PAD=n_expts_pad)
    return dx


# Backward method (kernel launch code)
def _TopK_backward(ctx, dy_vals, _0, _1):
    x, y_indx = ctx.saved_tensors
    dx = topk_backward(x, y_indx, dy_vals, ctx.k, ctx.n_rows, ctx.apply_softmax
        )
    return dx, None, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class TopK(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, k, apply_softmax, dim, return_bitmatrix, y_indx, n_rows
        ):
        y_vals, y_indx, bitmatrix = topk_forward(x, k, apply_softmax, dim,
            return_bitmatrix, y_indx, n_rows)
        ctx.save_for_backward(x, y_indx)
        ctx.apply_softmax = apply_softmax
        ctx.k = k
        ctx.n_rows = n_rows
        return y_vals, y_indx, bitmatrix

    @staticmethod
    def backward(ctx, dy_vals, _0, _1):
        x, y_indx = ctx.saved_tensors
        dx = topk_backward(x, y_indx, dy_vals, ctx.k, ctx.n_rows, ctx.
            apply_softmax)
        return dx, None, None, None, None, None, None
