# SPDX-License-Identifier: BSD-2-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/linkedin/Liger-Kernel
# Source-Files: src/liger_kernel/ops/rms_norm.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_d1vk6zhm/Liger-Kernel-main/src/liger_kernel/ops/rms_norm.py
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
def _block_rms_norm_forward_kernel(Y_ptr, Y_row_stride, X_ptr, X_row_stride,
    W_ptr, W_row_stride, RSTD_ptr, RSTD_row_stride, n_rows, n_cols, eps,
    offset, casting_mode: tl.constexpr, BLOCK_SIZE: tl.constexpr, BLOCK_ROW:
    tl.constexpr):
    """
    y_i = (x_i / (RMS)) * (offset + wi), RMS = sqrt(sum(x_i^2) / N)

    Reference:
    1. https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    2. https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/rms_layernorm.py#L22
    3. https://arxiv.org/pdf/1910.07467
    """
    row_idx = tl.program_id(0) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_mask = row_idx < n_rows
    col_mask = col_offsets < n_cols
    X_row = tl.load(X_ptr + row_idx[:, None] * X_row_stride + col_offsets[
        None, :], mask=row_mask[:, None] & col_mask[None, :], other=0)
    X_row_dtype = X_row.dtype
    W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0)
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(tl.float32)
    if casting_mode == _CASTING_MODE_GEMMA:
        W_row = W_row.to(tl.float32)
        X_row = X_row.to(tl.float32)
    if casting_mode == _CASTING_MODE_NONE:
        eps = eps.to(X_row_dtype)
        offset = offset.to(X_row_dtype)
    mean_square = tl.sum(X_row * X_row, axis=1) / n_cols
    rstd = rsqrt(mean_square + eps)
    tl.store(RSTD_ptr + row_idx * RSTD_row_stride, rstd, row_mask)
    X_row = X_row * rstd[:, None]
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(X_row_dtype)
    Y_row = X_row * (offset + W_row)[None, :]
    if casting_mode == _CASTING_MODE_GEMMA:
        Y_row = Y_row.to(X_row_dtype)
    tl.store(Y_ptr + row_idx[:, None] * Y_row_stride + col_offsets[None, :],
        Y_row, mask=row_mask[:, None] & col_mask[None, :])


@triton.jit
def _rms_norm_forward_kernel(Y_ptr, Y_row_stride, X_ptr, X_row_stride,
    W_ptr, W_row_stride, RSTD_ptr, RSTD_row_stride, n_cols, eps, offset,
    casting_mode: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    y_i = (x_i / (RMS)) * (offset + wi), RMS = sqrt(sum(x_i^2) / N)

    Reference:
    1. https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    2. https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/rms_layernorm.py#L22
    3. https://arxiv.org/pdf/1910.07467
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride
    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    X_row_dtype = X_row.dtype
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(tl.float32)
    if casting_mode == _CASTING_MODE_GEMMA:
        W_row = W_row.to(tl.float32)
        X_row = X_row.to(tl.float32)
    if casting_mode == _CASTING_MODE_NONE:
        eps = eps.to(X_row_dtype)
        offset = offset.to(X_row_dtype)
    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)
    tl.store(RSTD_ptr, rstd)
    X_row = X_row * rstd
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(X_row_dtype)
    Y_row = X_row * (offset + W_row)
    if casting_mode == _CASTING_MODE_GEMMA:
        Y_row = Y_row.to(X_row_dtype)
    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)


def rms_norm_forward(X, W, eps, offset, casting_mode, row_mode):
    if not isinstance(casting_mode, int):
        assert casting_mode in _str_to_casting_mode, f'Invalid casting mode: {casting_mode}'
        casting_mode = _str_to_casting_mode[casting_mode]
    else:
        assert casting_mode in _str_to_casting_mode.values(
            ), f'Invalid casting mode: {casting_mode}'
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    rstd_dtype = torch.float32 if casting_mode in (_CASTING_MODE_LLAMA.
        value, _CASTING_MODE_GEMMA.value) else X.dtype
    RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X.device)
    assert X.shape[1] == W.shape[0
        ], 'Incompatible hidden size dimension between tensor1.shape[1] and tensor2.shape[0]'
    kernel_args = {}
    if X.device.type == 'xpu':
        kernel_args['grf_mode'] = 'large'
    if BLOCK_SIZE > 256 or n_rows < 4096 * 8 or row_mode:
        _rms_norm_forward_kernel[n_rows,](Y, Y.stride(0), X, X.stride(0), W,
            W.stride(0), RSTD, RSTD.stride(0), n_cols, eps, offset,
            casting_mode, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, **
            kernel_args)
    else:
        BLOCK_ROW = 16
        kernel_args['BLOCK_ROW'] = BLOCK_ROW
        _block_rms_norm_forward_kernel[triton.cdiv(n_rows, BLOCK_ROW),](Y,
            Y.stride(0), X, X.stride(0), W, W.stride(0), RSTD, RSTD.stride(
            0), n_rows, n_cols, eps, offset, casting_mode, BLOCK_SIZE=
            BLOCK_SIZE, num_warps=num_warps, **kernel_args)
    return Y.view(*shape), X, RSTD, BLOCK_SIZE, num_warps, casting_mode


def calculate_settings(n):
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f'Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}.'
            )
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32 if not is_hip() else 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


def is_hip() ->bool:
    return torch.version.hip is not None


# Forward method (kernel launch code)
@ensure_contiguous
def _LigerRMSNormFunction_forward(ctx, X, W, eps, offset=0.0, casting_mode=
    'llama', in_place=True, row_mode=None):
    """
        X: (B, T, H) or (BxT, H)
        W: (H,)
        """
    Y, X, RSTD, BLOCK_SIZE, num_warps, casting_mode = rms_norm_forward(X, W,
        eps, offset, casting_mode, row_mode)
    ctx.offset = offset
    ctx.casting_mode = casting_mode
    ctx.in_place = in_place
    ctx.row_mode = row_mode
    ctx.BLOCK_SIZE = BLOCK_SIZE
    ctx.num_warps = num_warps
    ctx.save_for_backward(X, W, RSTD)
    return Y


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _block_rms_norm_backward_kernel(dY_ptr, dY_row_stride, dX_ptr,
    dX_row_stride, X_ptr, X_row_stride, X_dtype: tl.constexpr, W_ptr,
    W_row_stride, RSTD_ptr, RSTD_row_stride, dW_ptr, dW_row_stride, n_rows,
    n_cols, offset, rows_per_program: tl.constexpr, casting_mode: tl.
    constexpr, BLOCK_SIZE: tl.constexpr, BLOCK_ROW: tl.constexpr):
    """
    dx = (1 / RMS) * [dy * (w + offset - (1 / N) * (1 / RMS^2) * ((dy * (w + offset)) dot x) * x]. * means element-wise multiplication, whileas dot means dot product
    dw = sum(dy * (x / RMS)). summation over BxT dimension
    """
    pid = tl.program_id(0).cast(tl.int64)
    NUM_SMS = tl.num_programs(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols
    dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0.0)
    W_row = W_row + offset
    for start in range(pid * BLOCK_ROW, n_rows, NUM_SMS * BLOCK_ROW):
        row_idx = start + tl.arange(0, BLOCK_ROW)
        row_mask = row_idx < n_rows
        dY_row = tl.load(dY_ptr + row_idx[:, None] * dY_row_stride +
            col_offsets[None, :], mask=row_mask[:, None] & col_mask[None, :
            ], other=0.0)
        X_row = tl.load(X_ptr + row_idx[:, None] * X_row_stride +
            col_offsets[None, :], mask=row_mask[:, None] & col_mask[None, :
            ], other=0.0)
        rstd_row = tl.load(RSTD_ptr + row_idx * RSTD_row_stride, row_mask)
        X_row = X_row.to(tl.float32)
        if casting_mode == _CASTING_MODE_LLAMA:
            m = (dY_row * W_row[None, :]).to(tl.float32)
        elif casting_mode == _CASTING_MODE_GEMMA:
            dY_row = dY_row.to(tl.float32)
            m = dY_row * W_row[None, :]
        else:
            m = dY_row * W_row[None, :]
        dX_row = rstd_row[:, None] * m
        dX_row += rstd_row[:, None] * (-(1 / n_cols) * (rstd_row * rstd_row *
            tl.sum(m * X_row, axis=1))[:, None] * X_row)
        if casting_mode == _CASTING_MODE_LLAMA:
            dW_row += tl.sum(dY_row * (X_row * rstd_row[:, None]).to(
                X_dtype), 0)
        else:
            dW_row += tl.sum(dY_row * (X_row * rstd_row[:, None]), 0)
        tl.store(dX_ptr + row_idx[:, None] * dX_row_stride + col_offsets[
            None, :], dX_row, mask=row_mask[:, None] & col_mask[None, :])
    tl.store(dW_ptr + pid * dW_row_stride + col_offsets, dW_row, mask=col_mask)


@triton.jit
def _rms_norm_backward_kernel(dY_ptr, dY_row_stride, dX_ptr, dX_row_stride,
    X_ptr, X_row_stride, X_dtype: tl.constexpr, W_ptr, W_row_stride,
    RSTD_ptr, RSTD_row_stride, dW_ptr, dW_row_stride, n_rows, n_cols,
    offset, rows_per_program: tl.constexpr, casting_mode: tl.constexpr,
    BLOCK_SIZE: tl.constexpr):
    """
    dx = (1 / RMS) * [dy * (w + offset - (1 / N) * (1 / RMS^2) * ((dy * (w + offset)) dot x) * x]. * means element-wise multiplication, whileas dot means dot product
    dw = sum(dy * (x / RMS)). summation over BxT dimension
    """
    row_block_id = tl.program_id(0).to(tl.int64)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    dY_ptr += row_start * dY_row_stride
    dX_ptr += row_start * dX_row_stride
    X_ptr += row_start * X_row_stride
    RSTD_ptr += row_start
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    W_row = W_row + offset
    for _ in range(row_start, row_end):
        dY_row = tl.load(dY_ptr + col_offsets, mask=mask, other=0.0)
        X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0)
        rstd_row = tl.load(RSTD_ptr)
        X_row = X_row.to(tl.float32)
        if casting_mode == _CASTING_MODE_LLAMA:
            m = (dY_row * W_row).to(tl.float32)
        elif casting_mode == _CASTING_MODE_GEMMA:
            dY_row = dY_row.to(tl.float32)
            m = dY_row * W_row
        else:
            m = dY_row * W_row
        dX_row = rstd_row * m
        dX_row += rstd_row * (-(1 / n_cols) * rstd_row * rstd_row * tl.sum(
            m * X_row, axis=0) * X_row)
        if casting_mode == _CASTING_MODE_LLAMA:
            dW_row += dY_row * (X_row * rstd_row).to(X_dtype)
        else:
            dW_row += dY_row * (X_row * rstd_row)
        tl.store(dX_ptr + col_offsets, dX_row.to(X_dtype), mask=mask)
        dY_ptr += dY_row_stride
        dX_ptr += dX_row_stride
        X_ptr += X_row_stride
        RSTD_ptr += RSTD_row_stride
    tl.store(dW_ptr + row_block_id * dW_row_stride + col_offsets, dW_row,
        mask=mask)


def rms_norm_backward(dY, X, W, RSTD, offset, casting_mode, BLOCK_SIZE,
    num_warps, in_place, row_mode):
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape
    sm_count = 1
    if X.device.type == 'cuda':
        sm_count = torch.cuda.get_device_properties(X.device
            ).multi_processor_count
    elif X.device.type == 'xpu':
        sm_count = torch.xpu.get_device_properties(X.device).gpu_eu_count
    _dW = torch.empty((sm_count, n_cols), dtype=torch.float32, device=W.device)
    if n_cols > BLOCK_SIZE:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    rows_per_program = math.ceil(n_rows / sm_count)
    grid = sm_count,
    if in_place is True:
        dX = dY
    else:
        dX = torch.zeros_like(dY)
    kernel_args = {}
    if X.device.type == 'xpu':
        kernel_args['grf_mode'] = 'large'
    if BLOCK_SIZE > 256 or n_rows < 4096 * 8 or row_mode:
        _rms_norm_backward_kernel[grid](dY, dY.stride(0), dX, dX.stride(0),
            X, X.stride(0), torch_to_triton_dtype[X.dtype], W, W.stride(0),
            RSTD, RSTD.stride(0), _dW, _dW.stride(0), n_rows, n_cols,
            offset, rows_per_program, casting_mode, BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps, **kernel_args)
    else:
        BLOCK_ROW = 16
        kernel_args['BLOCK_ROW'] = BLOCK_ROW
        _block_rms_norm_backward_kernel[grid](dY, dY.stride(0), dX, dX.
            stride(0), X, X.stride(0), torch_to_triton_dtype[X.dtype], W, W
            .stride(0), RSTD, RSTD.stride(0), _dW, _dW.stride(0), n_rows,
            n_cols, offset, rows_per_program, casting_mode, BLOCK_SIZE=
            BLOCK_SIZE, num_warps=num_warps, **kernel_args)
    dX = dX.view(*shape)
    dW = _dW.sum(dim=0).to(W.dtype)
    return dX, dW


# Backward method (kernel launch code)
@ensure_contiguous
def _LigerRMSNormFunction_backward(ctx, dY):
    """
        Y: (B, T, H) or (BxT, H)
        """
    X, W, RSTD = ctx.saved_tensors
    dX, dW = rms_norm_backward(dY, X, W, RSTD, ctx.offset, ctx.casting_mode,
        ctx.BLOCK_SIZE, ctx.num_warps, ctx.in_place, ctx.row_mode)
    return dX, dW, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LigerRMSNormFunction(torch.autograd.Function):
    """
    Performs RMSNorm (Root Mean Square Normalization), which normalizes the input tensor `X` using the
    weight tensor `W`, with an optional offset and casting mode.

    Some models use an 'offset' to shift the weight tensor `W` by a constant value. For example, Gemma
    uses an offset of 1.0, so the computation becomes `(X / RMS(X)) * (W + 1.0)` instead of the usual
    `(X / RMS(X)) * W`. You can pass the offset value as an argument to the forward function.

    In addition, different models cast their inputs at different places during RMSNorm computation. For
    example, Gemma casts everything to fp32 nefore starting the computation, while Llama casts only the
    inverse RMS to fp32. You can specify the casting mode using the `casting_mode` argument. We currently
    support the following casting modes (they match HuggingFace Transformers' implementations):
    - 'llama': matches the Llama implementation, where only the inverse RMS is computed on fp32.
    - 'gemma': matches the Gemma implementation, where everything is cast to fp32, then computed, then cast back to the original dtype.
    - 'none': no casting is done. The computation is done in the original dtype. This saves memory and is slightly faster, but has more error w.r.t. the original implementation.

    `in_place` option means whether to in_place modify dY to store dX. This is default to `True` to save memory. However, under certain cases, it can produce incorrect inputs.
        For example, gemma2 uses two rmsnorm sequentially with residual in between. The resesidual part needs dY so it cannot be modified in-place.
        Therefore, for the patching of RMSNorm in gemma2, we set `in_place` to `False`
    """

    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, eps, offset=0.0, casting_mode='llama', in_place=
        True, row_mode=None):
        """
        X: (B, T, H) or (BxT, H)
        W: (H,)
        """
        Y, X, RSTD, BLOCK_SIZE, num_warps, casting_mode = rms_norm_forward(X,
            W, eps, offset, casting_mode, row_mode)
        ctx.offset = offset
        ctx.casting_mode = casting_mode
        ctx.in_place = in_place
        ctx.row_mode = row_mode
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.save_for_backward(X, W, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        """
        Y: (B, T, H) or (BxT, H)
        """
        X, W, RSTD = ctx.saved_tensors
        dX, dW = rms_norm_backward(dY, X, W, RSTD, ctx.offset, ctx.
            casting_mode, ctx.BLOCK_SIZE, ctx.num_warps, ctx.in_place, ctx.
            row_mode)
        return dX, dW, None, None, None, None, None
