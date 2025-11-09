# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/stanford-futuredata/stk
# Source-Files: stk/backend/sputnik.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_nby9i_3k/stk-main/stk/backend/sputnik.py
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

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=[triton.Config({'BLOCK_M': TritonConfig.BLOCK_M,
    'BLOCK_N': TritonConfig.BLOCK_N, 'BLOCK_K': TritonConfig.BLOCK_K,
    'BLOCK_SIZE': TritonConfig.BLOCK_SIZE}, num_stages=TritonConfig.
    NUM_STAGES, num_warps=TritonConfig.NUM_WARPS)], key=['M', 'N', 'K'])
@triton.jit
def _dds_kernel(A, B, C, M, N, K, stride_am, stride_ak, stride_bk,
    stride_bn, stride_cm, stride_cn, row_indices, column_indices, offsets,
    block_offsets_t, trans_A: tl.constexpr, trans_B: tl.constexpr, BLOCK_M:
    tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_SIZE:
    tl.constexpr, GROUP_M: tl.constexpr, ACC_TYPE: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_pid_m = tl.num_programs(0)
    num_pid_n = tl.num_programs(1)
    pid_n, pid_m = tl.swizzle2d(pid_n, pid_m, num_pid_n, num_pid_m, GROUP_M)
    start_inx = tl.load(offsets + pid_n)
    end_inx = tl.load(offsets + pid_n + 1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rak = tl.arange(0, BLOCK_K)
    A += rm[:, None] * stride_am + rak[None, :] * stride_ak
    rn = tl.arange(0, BLOCK_N)
    rbk = tl.arange(0, BLOCK_K)
    B += rbk[:, None] * stride_bk + rn[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    nsub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_K)
    BLOCK_ELEMENTS = BLOCK_SIZE * BLOCK_SIZE
    ak_sub_incr = BLOCK_K * stride_ak
    ak_block_incr = BLOCK_SIZE * stride_ak
    bk_sub_incr = BLOCK_K * stride_bk
    for k in range(nsub_blocks * (end_inx - start_inx)):
        sub_block_inx = k % nsub_blocks
        block_inx = k // nsub_blocks
        if trans_B:
            ptr_B = B + (start_inx + block_inx
                ) * BLOCK_ELEMENTS + sub_block_inx * bk_sub_incr
        else:
            ptr_B = B + tl.load(block_offsets_t + start_inx + block_inx
                ) * BLOCK_ELEMENTS + sub_block_inx * bk_sub_incr
        ptr_A = A + tl.load(column_indices + start_inx + block_inx
            ) * ak_block_incr + sub_block_inx * ak_sub_incr
        a = tl.load(ptr_A)
        b = tl.load(ptr_B)
        acc += tl.dot(a, b)
    acc = acc.to(C.dtype.element_ty)
    cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (cm[:, None] * stride_cm + cn[None, :] * stride_cn)
    tl.store(C, acc, mask=True)


def _standardize_shape(x, transpose):
    if transpose:
        return torch.Size((x[1], x[0]))
    return x


def _validate_matmul_dims(M: int, K: int, N: int):
    error_string = (
        'incompatible dimensions: tensor has dim with length: {}, which must be divisible by {}'
        )
    assert M % TritonConfig.BLOCK_M == 0, error_string.format(M,
        TritonConfig.BLOCK_M)
    assert K % TritonConfig.BLOCK_K == 0, error_string.format(K,
        TritonConfig.BLOCK_K)
    assert N % TritonConfig.BLOCK_N == 0, error_string.format(N,
        TritonConfig.BLOCK_N)


def dds(lhs, shape, data, offsets, row_indices, column_indices, offsets_t,
    column_indices_t, block_offsets_t, transpose_b, out):
    device = lhs.device
    trans_B = transpose_b
    trans_A = False
    if lhs.stride(0) > 1 and lhs.stride(1) > 1:
        trans_A = True
    assert lhs.shape[1] == shape[0], 'incompatible dimensions'
    M, K = lhs.shape
    _, N = shape
    _validate_matmul_dims(M, K, N)
    ACC_TYPE = tl.float32 if lhs.dtype in [torch.float16, torch.bfloat16,
        torch.float32] else tl.int32
    stride_am, stride_ak = lhs.stride(0), lhs.stride(1)
    stride_bk, stride_bn = data.stride(1), data.stride(2)
    b_column_indices = column_indices_t
    b_offsets = offsets_t
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N,
        META['BLOCK_N']))
    if trans_A:
        stride_am, stride_ak = lhs.stride(1), lhs.stride(0)
    if trans_B:
        stride_bk, stride_bn = data.stride(2), data.stride(1)
        b_column_indices, b_offsets = column_indices, offsets
    _dds_kernel[grid](lhs, data, out, M, N, K, stride_am, stride_ak,
        stride_bk, stride_bn, out.stride(0), out.stride(1), row_indices,
        b_column_indices, b_offsets, block_offsets_t, trans_A, trans_B,
        GROUP_M=128, ACC_TYPE=ACC_TYPE)


# Forward method (kernel launch code)
@custom_fwd
def _DDS_forward(ctx, lhs, shape, data, offsets, row_indices,
    column_indices, offsets_t, column_indices_t, block_offsets_t, transpose_b):
    ctx.save_for_backward(lhs, data, offsets, row_indices, column_indices,
        offsets_t, column_indices_t, block_offsets_t)
    ctx.shape = _standardize_shape(shape, transpose_b)
    ctx.transpose_b = transpose_b
    out = torch.empty((lhs.size()[0], shape[1]), dtype=lhs.dtype, device=
        lhs.device)
    backend.dds(lhs, shape, data, offsets, row_indices, column_indices,
        offsets_t, column_indices_t, block_offsets_t, transpose_b, out)
    return out


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

def _call_helper(op, out, a, b, trans_a, trans_b):
    args = _wrap(_transpose_helper(a, trans_a)) + _wrap(_transpose_helper(b,
        trans_b))
    if isinstance(out, tuple):
        args = args + out
    return op(*args)


def _is_transposed(x):
    return not x.is_contiguous() and x.stride()[0] == 1 and x.stride()[1
        ] == x.size()[0]


def _lhs_gradient(op, lhs, rhs, dy, trans_lhs, trans_rhs):
    lhs, rhs, dy = _preprocess_inputs(lhs, rhs, dy)
    a, b = (rhs, dy) if trans_lhs else (dy, rhs)
    trans_a = trans_lhs and trans_rhs
    trans_b = trans_lhs or not trans_rhs
    out = _call_helper(op, lhs, a, b, trans_a, trans_b)
    return _postprocess_outputs(lhs, trans_lhs, out)


def _postprocess_outputs(x, transpose, grad):
    if isinstance(x, torch.Tensor) and transpose:
        return grad.t()
    return grad


def _preprocess_inputs(lhs, rhs, dy):
    if isinstance(lhs, torch.Tensor) and _is_transposed(lhs):
        lhs = lhs.t()
    if isinstance(rhs, torch.Tensor) and _is_transposed(rhs):
        rhs = rhs.t()
    if isinstance(dy, torch.Tensor) and not dy.is_contiguous(
        ) and not _is_transposed(dy):
        dy = dy.contiguous()
    if isinstance(dy, tuple) and not dy[1].is_contiguous():
        dy = (dy[0], dy[1].contiguous()) + dy[2:]
    return lhs, rhs, dy


def _rhs_gradient(op, lhs, rhs, dy, trans_lhs, trans_rhs):
    lhs, rhs, dy = _preprocess_inputs(lhs, rhs, dy)
    a, b = (dy, lhs) if trans_rhs else (lhs, dy)
    trans_a = not trans_lhs or trans_rhs
    trans_b = trans_lhs and trans_rhs
    out = _call_helper(op, rhs, a, b, trans_a, trans_b)
    return _postprocess_outputs(rhs, trans_rhs, out)


def _sparse_transpose(x):
    return (torch.Size((x[0][1], x[0][0])),) + x[1:]


def _transpose_helper(x, transpose):
    if isinstance(x, torch.Tensor):
        return x.t() if transpose else x
    if transpose:
        x = _sparse_transpose(x)
    return x + (transpose,)


def _wrap(x):
    if isinstance(x, torch.Tensor):
        return x,
    return x


# Backward method (kernel launch code)
@custom_bwd
def _DDS_backward(ctx, dy):
    saved_tensors = ctx.saved_tensors
    lhs = saved_tensors[0]
    rhs = (ctx.shape,) + saved_tensors[1:]
    trans_a = _is_transposed(lhs)
    trans_b = ctx.transpose_b
    dlhs = None
    if ctx.needs_input_grad[0]:
        op = dsd if trans_a else dds
        dlhs = _lhs_gradient(op, lhs, rhs, dy, trans_a, trans_b)
    ddata = None
    if ctx.needs_input_grad[2]:
        ddata = _rhs_gradient(sdd, lhs, rhs, dy, trans_a, trans_b)
    return dlhs, None, ddata, None, None, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class DDS(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, lhs, shape, data, offsets, row_indices, column_indices,
        offsets_t, column_indices_t, block_offsets_t, transpose_b):
        ctx.save_for_backward(lhs, data, offsets, row_indices,
            column_indices, offsets_t, column_indices_t, block_offsets_t)
        ctx.shape = _standardize_shape(shape, transpose_b)
        ctx.transpose_b = transpose_b
        out = torch.empty((lhs.size()[0], shape[1]), dtype=lhs.dtype,
            device=lhs.device)
        backend.dds(lhs, shape, data, offsets, row_indices, column_indices,
            offsets_t, column_indices_t, block_offsets_t, transpose_b, out)
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, dy):
        saved_tensors = ctx.saved_tensors
        lhs = saved_tensors[0]
        rhs = (ctx.shape,) + saved_tensors[1:]
        trans_a = _is_transposed(lhs)
        trans_b = ctx.transpose_b
        dlhs = None
        if ctx.needs_input_grad[0]:
            op = dsd if trans_a else dds
            dlhs = _lhs_gradient(op, lhs, rhs, dy, trans_a, trans_b)
        ddata = None
        if ctx.needs_input_grad[2]:
            ddata = _rhs_gradient(sdd, lhs, rhs, dy, trans_a, trans_b)
        return dlhs, None, ddata, None, None, None, None, None, None, None
