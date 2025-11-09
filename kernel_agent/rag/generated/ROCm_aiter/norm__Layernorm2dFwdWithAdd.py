# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/ROCm/aiter
# Source-Files: aiter/ops/triton/norm.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_ciw0cknm/aiter-main/aiter/ops/triton/norm.py
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
def _fused_add_layernorm_kernel(x_ptr, y_ptr, res_in_ptr, res_out_ptr,
    w_ptr, b_ptr, mean_ptr, rstd_ptr, x_row_stride, y_row_stride, n_rows,
    n_cols, eps, BLOCK_SIZE: tl.constexpr):
    """
    Note: this is Triton jited function and not meant to be called directly. Call layernorm2d_fwd_with_add function
    below

    Performs an addition between two inputs and then applies Layer Normalization over
    the addition result.

    Key parameters:
    - X: The input tensor to be normalized with shape (M, N).
    - Y: The output tensor with the same shape as the input one.
    - Res_in: The tensor to be added to the X tensor with shape (M, N).
    - Res_out: The tensor in which the addition result will be stored with shape (M, N).
    - W: The learnable weights tensor with shape (N, ).
    - B: The learnable bias tensor with shape (N, ).
    """
    row = tl.program_id(0)
    x_ptr_start = x_ptr + row * x_row_stride
    y_ptr_start = y_ptr + row * y_row_stride
    res_in_ptr_start = res_in_ptr + row * x_row_stride
    res_out_ptr_start = res_out_ptr + row * x_row_stride
    loop_num = tl.cdiv(n_cols, BLOCK_SIZE) - 1
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        _x_block = tl.load(x_ptr_start + col_offsets)
        res_in_block = tl.load(res_in_ptr_start + col_offsets)
        _x_block += res_in_block
        tl.store(res_out_ptr_start + col_offsets, _x_block)
        _mean += _x_block.to(tl.float32)
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    _x_block = tl.load(x_ptr_start + col_offsets, mask=col_offsets < n_cols,
        other=0.0)
    res_in_block = tl.load(res_in_ptr_start + col_offsets, mask=col_offsets <
        n_cols, other=0.0)
    _x_block += res_in_block
    tl.store(res_out_ptr_start + col_offsets, _x_block, mask=col_offsets <
        n_cols)
    _mean += _x_block.to(tl.float32)
    mean = tl.sum(_mean, axis=0) / n_cols
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x_block = tl.load(res_out_ptr_start + col_offsets).to(tl.float32)
        x_block = x_block - mean
        _var += x_block * x_block
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_block = tl.load(res_out_ptr_start + col_offsets, mask=col_offsets <
        n_cols, other=0.0).to(tl.float32)
    x_block = tl.where(col_offsets < n_cols, x_block - mean, 0.0)
    _var += x_block * x_block
    var = tl.sum(_var, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        w_block = tl.load(w_ptr + col_offsets)
        b_block = tl.load(b_ptr + col_offsets)
        x_block = tl.load(res_out_ptr_start + col_offsets).to(tl.float32)
        y_block = (x_block - mean) * rstd
        y_block = y_block * w_block + b_block
        tl.store(y_ptr_start + col_offsets, y_block)
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    w_block = tl.load(w_ptr + col_offsets, mask=mask, other=0.0)
    b_block = tl.load(b_ptr + col_offsets, mask=mask, other=0.0)
    x_block = tl.load(res_out_ptr_start + col_offsets, mask=mask, other=0.0
        ).to(tl.float32)
    y_block = (x_block - mean) * rstd
    y_block = y_block * w_block + b_block
    tl.store(y_ptr_start + col_offsets, y_block, mask=mask)


def _layernorm_forward_with_add(y: torch.Tensor, x: torch.Tensor, res_in:
    torch.Tensor, res_out: torch.Tensor, weight: torch.Tensor, bias: torch.
    Tensor, mean: torch.Tensor, rstd: torch.Tensor, epsilon: float):
    M, N = x.shape
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    _fused_add_layernorm_kernel[M,](x, y, res_in, res_out, weight, bias,
        mean, rstd, x.stride(0), y.stride(0), M, N, epsilon, BLOCK_SIZE)
    return


# Forward method (kernel launch code)
def __Layernorm2dFwdWithAdd_forward(ctx, y, x, res_in, res_out, weight,
    bias, eps, is_grad_enabled):
    is_grad = is_grad_enabled and any(tensor.requires_grad for tensor in [x,
        weight, bias])
    M = x.shape[0]
    mean = torch.empty((M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    _layernorm_forward_with_add(y, x, res_in, res_out, weight, bias, mean,
        rstd, eps)
    if is_grad:
        ctx.save_for_backward(res_out, weight, mean, rstd)
    return y


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _layernorm_bwd_dwdb_triton(DW, DB, FINAL_DW, FINAL_DB, M, N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw.to(FINAL_DW.type.element_ty), mask=
        cols < N)
    tl.store(FINAL_DB + cols, sum_db.to(FINAL_DB.type.element_ty), mask=
        cols < N)


@triton.jit
def _layernorm_bwd_dwdb_triton_v2(X, DY, Mean, Rstd, stride, FINAL_DW,
    FINAL_DB, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        means = tl.load(Mean + rows, mask=rows < M, other=0.0).to(tl.float32)
        rstds = tl.load(Rstd + rows, mask=rows < M, other=0.0).to(tl.float32)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * stride + cols[None, :]
        x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
        xhat = (x - means[:, None]) * rstds[:, None]
        dw += dy * xhat
        db += dy
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw.to(FINAL_DW.type.element_ty), mask=
        cols < N)
    tl.store(FINAL_DB + cols, sum_db.to(FINAL_DB.type.element_ty), mask=
        cols < N)


@triton.jit
def _layernorm_bwd_dx_fused_triton(DX, DY, DW, DB, X, W, Mean, Rstd, stride,
    N, NUM_ROWS: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, USE_BLOCKED: tl.
    constexpr, IGNORE_DW_DB: tl.constexpr=False):
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    rows_per_tile = NUM_ROWS // tile_num
    if pid < NUM_ROWS % tile_num:
        rows_per_tile += 1
    if USE_BLOCKED:
        col_offsets = tl.arange(0, BLOCK_SIZE_N)
        num_col_blocks = tl.cdiv(N, BLOCK_SIZE_N) - 1
        row = pid
        for _ in range(0, rows_per_tile):
            mean = tl.load(Mean + row)
            rstd = tl.load(Rstd + row)
            x_row_ptr = X + row * stride
            dy_row_ptr = DY + row * stride
            c1 = 0.0
            c2 = 0.0
            for block_idx in tl.range(0, num_col_blocks):
                cols = block_idx * BLOCK_SIZE_N + col_offsets
                x = tl.load(x_row_ptr + cols).to(tl.float32)
                dy = tl.load(dy_row_ptr + cols).to(tl.float32)
                w = tl.load(W + cols).to(tl.float32)
                xhat = (x - mean) * rstd
                wdy = w * dy
                c1 += tl.sum(xhat * wdy, axis=0)
                c2 += tl.sum(wdy, axis=0)
            cols = num_col_blocks * BLOCK_SIZE_N + col_offsets
            mask = cols < N
            x = tl.load(x_row_ptr + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(dy_row_ptr + cols, mask=mask, other=0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0).to(tl.float32)
            xhat = (x - mean) * rstd
            wdy = w * dy
            wdy = tl.where(mask, wdy, 0)
            c1 += tl.sum(xhat * wdy, axis=0)
            c2 += tl.sum(wdy, axis=0)
            c1 /= N
            c2 /= N
            dx_row_ptr = DX + row * stride
            if not IGNORE_DW_DB:
                dw_row_ptr = DW + pid * N
                db_row_ptr = DB + pid * N
            for block_idx in tl.range(0, num_col_blocks):
                cols = block_idx * BLOCK_SIZE_N + col_offsets
                x = tl.load(x_row_ptr + cols).to(tl.float32)
                dy = tl.load(dy_row_ptr + cols).to(tl.float32)
                w = tl.load(W + cols).to(tl.float32)
                xhat = (x - mean) * rstd
                wdy = w * dy
                dx = (wdy - (xhat * c1 + c2)) * rstd
                tl.store(dx_row_ptr + cols, dx.to(DX.type.element_ty))
                if not IGNORE_DW_DB:
                    partial_dw = dy * xhat
                    dw_ptrs = dw_row_ptr + cols
                    partial_dw += tl.load(dw_ptrs).to(tl.float32)
                    tl.store(dw_ptrs, partial_dw.to(DW.type.element_ty))
                    partial_db = dy
                    db_ptrs = db_row_ptr + cols
                    partial_db += tl.load(db_ptrs).to(tl.float32)
                    tl.store(db_ptrs, partial_db.to(DB.type.element_ty))
            cols = num_col_blocks * BLOCK_SIZE_N + col_offsets
            mask = cols < N
            x = tl.load(x_row_ptr + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(dy_row_ptr + cols, mask=mask, other=0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0).to(tl.float32)
            xhat = (x - mean) * rstd
            wdy = w * dy
            dx = (wdy - (xhat * c1 + c2)) * rstd
            tl.store(dx_row_ptr + cols, dx.to(DX.type.element_ty), mask=mask)
            if not IGNORE_DW_DB:
                partial_dw = dy * xhat
                dw_ptrs = dw_row_ptr + cols
                partial_dw += tl.load(dw_ptrs, mask=mask).to(tl.float32)
                tl.store(dw_ptrs, partial_dw.to(DW.type.element_ty), mask=mask)
                partial_db = dy
                db_ptrs = db_row_ptr + cols
                partial_db += tl.load(db_ptrs, mask=mask).to(tl.float32)
                tl.store(db_ptrs, partial_db.to(DB.type.element_ty), mask=mask)
            row += tile_num
    else:
        cols = tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        row = pid
        if not IGNORE_DW_DB:
            dw_row = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
            db_row = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        for _ in range(0, rows_per_tile):
            x_ptrs = X + row * stride
            dy_ptrs = DY + row * stride
            dx_ptrs = DX + row * stride
            x = tl.load(x_ptrs + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(dy_ptrs + cols, mask=mask, other=0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0).to(tl.float32)
            mean = tl.load(Mean + row)
            rstd = tl.load(Rstd + row)
            xhat = (x - mean) * rstd
            wdy = w * dy
            wdy = tl.where(mask, wdy, 0)
            c1 = tl.sum(xhat * wdy, axis=0) / N
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd
            tl.store(dx_ptrs + cols, dx.to(DX.type.element_ty), mask=mask)
            if not IGNORE_DW_DB:
                dw_row += dy * xhat
                db_row += dy
            row += tile_num
        if not IGNORE_DW_DB:
            tl.store(DW + pid * N + cols, dw_row.to(DW.type.element_ty),
                mask=mask)
            tl.store(DB + pid * N + cols, db_row.to(DB.type.element_ty),
                mask=mask)


def _layernorm_backward(dy: torch.Tensor, dx: torch.Tensor, dw: torch.
    Tensor, db: torch.Tensor, x: torch.Tensor, gamma: torch.Tensor, mu:
    torch.Tensor, rsigma: torch.Tensor):
    M, N = x.shape
    IGNORE_DW_DB_IN_FUSED = M <= 512
    tile_num = max(min(256, M // 4), 1)
    if M <= 512 and M * N < 64 * 1024 * 1024:
        tile_num = M
    elif M >= 8192:
        tile_num = 2048
    max_fused_size = 32768 // x.element_size()
    next_power = triton.next_power_of_2(N)
    BLOCK_SIZE = min(max_fused_size, next_power)
    if tile_num == M:
        if tile_num > 256:
            BLOCK_SIZE = min(BLOCK_SIZE, 2048)
        else:
            BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    USE_BLOCKED = N > BLOCK_SIZE
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    if not IGNORE_DW_DB_IN_FUSED:
        _dw = torch.zeros((tile_num, N), dtype=torch.float32, device=gamma.
            device)
        _db = torch.zeros((tile_num, N), dtype=torch.float32, device=gamma.
            device)
    else:
        _dw = None
        _db = None
    grid_bwd = tile_num,
    _layernorm_bwd_dx_fused_triton[grid_bwd](dx, dy, _dw, _db, x, gamma, mu,
        rsigma, x.stride(0), N, NUM_ROWS=M, BLOCK_SIZE_N=BLOCK_SIZE,
        USE_BLOCKED=USE_BLOCKED, num_warps=num_warps, IGNORE_DW_DB=
        IGNORE_DW_DB_IN_FUSED)
    grid_reduce = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    if not IGNORE_DW_DB_IN_FUSED:
        dwdb_block_n = max(16, N // 256)
        dwdb_block_n = triton.next_power_of_2(dwdb_block_n)
        dwdb_block_m = 64 * 128 // dwdb_block_n
        dwdb_block_m = min(triton.next_power_of_2(tile_num), dwdb_block_m)
        _layernorm_bwd_dwdb_triton[grid_reduce](_dw, _db, dw, db, min(
            tile_num, M), N, BLOCK_SIZE_M=dwdb_block_m, BLOCK_SIZE_N=
            dwdb_block_n)
    else:
        dwdb_block_n = max(16, N // 256)
        dwdb_block_n = triton.next_power_of_2(dwdb_block_n)
        dwdb_block_m = 64 * 128 // dwdb_block_n
        dwdb_block_m = min(triton.next_power_of_2(M), dwdb_block_m)
        _layernorm_bwd_dwdb_triton_v2[grid_reduce](x, dy, mu, rsigma, x.
            stride(0), dw, db, M, N, BLOCK_SIZE_M=dwdb_block_m,
            BLOCK_SIZE_N=dwdb_block_n)
    return dx, dw, db


# Backward method (kernel launch code)
def __Layernorm2dFwdWithAdd_backward(ctx, dy):
    x, w, m, v = ctx.saved_tensors
    N = w.shape[0]
    dw = torch.empty((N,), dtype=w.dtype, device=w.device)
    db = torch.empty((N,), dtype=w.dtype, device=w.device)
    dx = torch.empty_like(dy)
    _layernorm_backward(dy, dx, dw, db, x, w, m, v)
    return None, dx, None, None, dw, db, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _Layernorm2dFwdWithAdd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, y, x, res_in, res_out, weight, bias, eps, is_grad_enabled
        ):
        is_grad = is_grad_enabled and any(tensor.requires_grad for tensor in
            [x, weight, bias])
        M = x.shape[0]
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
        _layernorm_forward_with_add(y, x, res_in, res_out, weight, bias,
            mean, rstd, eps)
        if is_grad:
            ctx.save_for_backward(res_out, weight, mean, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, m, v = ctx.saved_tensors
        N = w.shape[0]
        dw = torch.empty((N,), dtype=w.dtype, device=w.device)
        db = torch.empty((N,), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)
        _layernorm_backward(dy, dx, dw, db, x, w, m, v)
        return None, dx, None, None, dw, db, None, None
