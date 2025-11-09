# SPDX-License-Identifier: BSD-3-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/meta-pytorch/tritonbench
# Source-Files: tritonbench/operators/gemm/kernels/matmul.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_ct7v_342/tritonbench-main/tritonbench/operators/gemm/kernels/matmul.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

@autotune(configs=tuning_configs, key=['M', 'N', 'K'], prune_configs_by=
    prune_configs_by)
@heuristics({'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args[
    'SPLIT_K']) == 0})
@jit
def _splitk_kernel(A, B, C, M, N, K, stride_am, stride_ak, stride_bk,
    stride_bn, stride_cm, stride_cn, acc_dtype: tl.constexpr,
    input_precision: tl.constexpr, fp8_fast_accum: tl.constexpr, BLOCK_M:
    tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl
    .constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr, AB_DTYPE: tl.
    constexpr, ENABLE_BUFFER_OPS_ASSUMES: tl.constexpr):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    if ENABLE_BUFFER_OPS_ASSUMES:
        tl.assume(M >= 0)
        tl.assume(N >= 0)
        tl.assume(K >= 0)
        tl.assume(stride_am >= 0)
        tl.assume(stride_ak >= 0)
        tl.assume(stride_bn >= 0)
        tl.assume(stride_bk >= 0)
        tl.assume(stride_cm >= 0)
        tl.assume(stride_cn >= 0)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if AB_DTYPE is not None:
            a = a.to(AB_DTYPE)
            b = b.to(AB_DTYPE)
        if fp8_fast_accum:
            acc = tl.dot(a, b, acc, out_dtype=acc_dtype, input_precision=
                input_precision)
        else:
            acc += tl.dot(a, b, out_dtype=acc_dtype, input_precision=
                input_precision)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask, sem='relaxed')


def get_higher_dtype(a, b):
    a = upcast_if_fp8(a)
    b = upcast_if_fp8(b)
    if a is b:
        return a
    assert a in _ordered_datatypes
    assert b in _ordered_datatypes
    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


def upcast_if_fp8(a):
    if 'fp8' in str(a):
        return torch.float16
    return a


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

# Forward method (kernel launch code)
def __matmul_forward(ctx, a, b, acc_dtype=None, input_precision=None,
    fp8_fast_accum=True, output_dtype=None):
    ctx.save_for_backward(a, b)
    ctx.acc_dtype = acc_dtype
    ctx.input_precision = input_precision
    ctx.fp8_fast_accum = fp8_fast_accum
    ctx.output_dtype = output_dtype
    return _matmul._call(a, b, acc_dtype=acc_dtype, input_precision=
        input_precision, fp8_fast_accum=fp8_fast_accum, output_dtype=
        output_dtype)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

# Backward method (kernel launch code)
def __matmul_backward(ctx, grad_output):
    a, b = ctx.saved_tensors
    grad_a = grad_b = None
    if ctx.needs_input_grad[0]:
        grad_a = _matmul._call(grad_output, b.t(), acc_dtype=ctx.acc_dtype,
            input_precision=ctx.input_precision, fp8_fast_accum=ctx.
            fp8_fast_accum, output_dtype=None)
    if ctx.needs_input_grad[1]:
        grad_b = _matmul._call(a.t(), grad_output, acc_dtype=ctx.acc_dtype,
            input_precision=ctx.input_precision, fp8_fast_accum=ctx.
            fp8_fast_accum, output_dtype=None)
    return grad_a, grad_b, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _matmul(torch.autograd.Function):
    kernel = _splitk_kernel
    _locks = {}

    @staticmethod
    def _call(a, b, acc_dtype, input_precision, fp8_fast_accum, output_dtype):
        device = a.device
        if a.stride(0) > 1 and a.stride(1) > 1:
            a = a.contiguous()
        if b.stride(0) > 1 and b.stride(1) > 1:
            b = b.contiguous()
        assert a.shape[1] == b.shape[0], 'incompatible dimensions'
        M, K = a.shape
        _, N = b.shape
        ab_dtype = get_higher_dtype(a.dtype, b.dtype)
        if output_dtype is None:
            output_dtype = ab_dtype
        c = torch.empty((M, N), device=device, dtype=output_dtype)
        supported_acc_dtypes = {torch.float16: (torch.float32, torch.
            float16), torch.bfloat16: (torch.float32, torch.bfloat16),
            torch.float32: (torch.float32,), torch.int8: (torch.int32,)}
        if acc_dtype is None:
            acc_dtype = supported_acc_dtypes[ab_dtype][0]
        else:
            assert isinstance(acc_dtype, torch.dtype
                ), 'acc_dtype must be a torch.dtype'
            assert acc_dtype in supported_acc_dtypes[a.dtype
                ], 'acc_dtype not compatible with the type of a'
            assert acc_dtype in supported_acc_dtypes[b.dtype
                ], 'acc_dtype not compatible with the type of b'

        def to_tl_type(ty):
            return getattr(tl, str(ty).split('.')[-1])
        acc_dtype = to_tl_type(acc_dtype)
        ab_dtype = to_tl_type(ab_dtype)
        output_dtype = to_tl_type(output_dtype)
        if a.dtype in [tl.float8e4nv, tl.float8e5] and b.dtype in [tl.
            float8e4nv, tl.float8e5]:
            ab_dtype = None
        grid = lambda META: (cdiv(M, META['BLOCK_M']) * cdiv(N, META[
            'BLOCK_N']), META['SPLIT_K'])
        enable_buffer_ops_assumes = a.stride(0) >= 0 and a.stride(1
            ) >= 0 and b.stride(0) >= 0 and b.stride(1) >= 0 and c.stride(0
            ) >= 0 and c.stride(1) >= 0
        _splitk_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.
            stride(0), b.stride(1), c.stride(0), c.stride(1), acc_dtype=
            acc_dtype, input_precision=input_precision, fp8_fast_accum=
            fp8_fast_accum, AB_DTYPE=ab_dtype, ENABLE_BUFFER_OPS_ASSUMES=
            enable_buffer_ops_assumes)
        return c

    @staticmethod
    def forward(ctx, a, b, acc_dtype=None, input_precision=None,
        fp8_fast_accum=True, output_dtype=None):
        ctx.save_for_backward(a, b)
        ctx.acc_dtype = acc_dtype
        ctx.input_precision = input_precision
        ctx.fp8_fast_accum = fp8_fast_accum
        ctx.output_dtype = output_dtype
        return _matmul._call(a, b, acc_dtype=acc_dtype, input_precision=
            input_precision, fp8_fast_accum=fp8_fast_accum, output_dtype=
            output_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_a = _matmul._call(grad_output, b.t(), acc_dtype=ctx.
                acc_dtype, input_precision=ctx.input_precision,
                fp8_fast_accum=ctx.fp8_fast_accum, output_dtype=None)
        if ctx.needs_input_grad[1]:
            grad_b = _matmul._call(a.t(), grad_output, acc_dtype=ctx.
                acc_dtype, input_precision=ctx.input_precision,
                fp8_fast_accum=ctx.fp8_fast_accum, output_dtype=None)
        return grad_a, grad_b, None, None, None, None
