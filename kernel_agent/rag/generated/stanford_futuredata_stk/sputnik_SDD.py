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
def _sdd_kernel(A, B, C, M, N, K, stride_am, stride_ak, stride_bk,
    stride_bn, stride_cm, stride_cn, row_indices, column_indices, BLOCK_M:
    tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_SIZE:
    tl.constexpr, GROUP_M: tl.constexpr, ACC_TYPE: tl.constexpr):
    pid = tl.program_id(0)
    pid_m = tl.load(row_indices + pid)
    pid_n = tl.load(column_indices + pid)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A)
        b = tl.load(B)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    BLOCK_ELEMENTS = BLOCK_SIZE * BLOCK_SIZE
    cm = tl.arange(0, BLOCK_M)
    cn = tl.arange(0, BLOCK_N)
    C = C + pid * BLOCK_ELEMENTS + (cm[:, None] * stride_cm + cn[None, :] *
        stride_cn)
    tl.store(C, acc, mask=True)


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


def sdd(lhs, rhs, shape, out, offsets, row_indices, column_indices):
    device = out.device
    trans_A = False
    trans_B = False
    if lhs.stride(0) > 1 and lhs.stride(1) > 1:
        trans_A = True
    if rhs.stride(0) > 1 and rhs.stride(1) > 1:
        trans_B = True
    assert lhs.shape[1] == rhs.shape[0], 'incompatible dimensions'
    M, K = lhs.shape
    _, N = rhs.shape
    _validate_matmul_dims(M, K, N)
    ACC_TYPE = tl.float32 if out.dtype in [torch.float16, torch.bfloat16,
        torch.float32] else tl.int32
    nnz_blocks = len(row_indices)
    grid = lambda META: (nnz_blocks,)
    stride_am, stride_ak = lhs.stride(0), lhs.stride(1)
    stride_bk, stride_bn = rhs.stride(0), rhs.stride(1)
    if trans_A:
        stride_am, stride_ak = lhs.stride(1), lhs.stride(0)
    if trans_B:
        stride_bk, stride_bn = rhs.stride(1), rhs.stride(0)
    _sdd_kernel[grid](lhs, rhs, out, M, N, K, stride_am, stride_ak,
        stride_bk, stride_bn, out.stride(1), out.stride(2), row_indices,
        column_indices, GROUP_M=128, ACC_TYPE=ACC_TYPE)


# Forward method (kernel launch code)
@custom_fwd
def _SDD_forward(ctx, lhs, rhs, shape, data, offsets, row_indices,
    column_indices, offsets_t, column_indices_t, block_offsets_t):
    ctx.save_for_backward(lhs, rhs, offsets, row_indices, column_indices,
        offsets_t, column_indices_t, block_offsets_t)
    ctx.shape = shape
    out = torch.empty(data.shape, dtype=lhs.dtype, device=lhs.device)
    backend.sdd(lhs, rhs, shape, out, offsets, row_indices, column_indices)
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
def _SDD_backward(ctx, dy):
    saved_tensors = ctx.saved_tensors
    lhs, rhs = saved_tensors[:2]
    dy = (ctx.shape, dy) + saved_tensors[2:]
    trans_a = _is_transposed(lhs)
    trans_b = _is_transposed(rhs)
    dlhs = None
    if ctx.needs_input_grad[0]:
        op = dds if trans_a else dsd
        dlhs = _lhs_gradient(op, lhs, rhs, dy, trans_a, trans_b)
    drhs = None
    if ctx.needs_input_grad[1]:
        op = dsd if trans_b else dds
        drhs = _rhs_gradient(op, lhs, rhs, dy, trans_a, trans_b)
    return dlhs, drhs, None, None, None, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class SDD(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, lhs, rhs, shape, data, offsets, row_indices,
        column_indices, offsets_t, column_indices_t, block_offsets_t):
        ctx.save_for_backward(lhs, rhs, offsets, row_indices,
            column_indices, offsets_t, column_indices_t, block_offsets_t)
        ctx.shape = shape
        out = torch.empty(data.shape, dtype=lhs.dtype, device=lhs.device)
        backend.sdd(lhs, rhs, shape, out, offsets, row_indices, column_indices)
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, dy):
        saved_tensors = ctx.saved_tensors
        lhs, rhs = saved_tensors[:2]
        dy = (ctx.shape, dy) + saved_tensors[2:]
        trans_a = _is_transposed(lhs)
        trans_b = _is_transposed(rhs)
        dlhs = None
        if ctx.needs_input_grad[0]:
            op = dds if trans_a else dsd
            dlhs = _lhs_gradient(op, lhs, rhs, dy, trans_a, trans_b)
        drhs = None
        if ctx.needs_input_grad[1]:
            op = dsd if trans_b else dds
            drhs = _rhs_gradient(op, lhs, rhs, dy, trans_a, trans_b)
        return dlhs, drhs, None, None, None, None, None, None, None, None
