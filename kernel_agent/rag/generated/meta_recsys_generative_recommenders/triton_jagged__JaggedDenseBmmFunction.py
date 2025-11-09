# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/meta-recsys/generative-recommenders
# Source-Files: generative_recommenders/ops/triton/triton_jagged.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rzwbk0zd/generative-recommenders-main/generative_recommenders/ops/triton/triton_jagged.py
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
from einops import reduce

def autotune_max_seq_len(runtime_max_seq_len: int) ->int:
    global USE_RUNTIME_MAX_SEQ_LEN
    if USE_RUNTIME_MAX_SEQ_LEN:
        return prev_power_of_2(runtime_max_seq_len)
    else:
        if STATIC_MAX_SEQ_LENS == []:
            return 1
        for max_len in STATIC_MAX_SEQ_LENS:
            if max_len >= runtime_max_seq_len:
                return max_len
        return STATIC_MAX_SEQ_LENS[-1]


@torch.fx.wrap
def prev_power_of_2(x: int) ->int:
    if torch.compiler.is_compiling():
        x_tensor = torch.scalar_tensor(x, dtype=torch.int64)
        x_tensor_orig = x_tensor.clone()
        out = triton.next_power_of_2(x_tensor)
        return int(torch.where(torch.lt(x_tensor_orig, out), out // 2, out)
            .item())
    else:
        out = triton.next_power_of_2(x)
        return out // 2 if out > x else out


@triton_autotune(configs=_get_bmm_configs(), key=['AUTOTUNE_MAX_SEQ_LEN',
    'N', 'K'])
@triton.jit
def jagged_dense_bmm_broadcast_add_kernel(seq_offsets, Jagged, Dense, Bias,
    Out, AUTOTUNE_MAX_SEQ_LEN, N, K, stride_jm, stride_db, stride_dk,
    stride_dn, stride_bias_b, stride_om, HAS_BIAS: tl.constexpr, ALLOW_TF32:
    tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl
    .constexpr, ELEMENTWISE: tl.constexpr):
    """
    Computing bmm Out = Jagged x Dense + Bias
    M is the jagged dimension
    Jagged has shape (sum_B(M_i), K), Dense has shape (B, K, N), Bias has shape (B, N), and Out has shape (sum_B(M_i), N)
    """
    off_n = tl.program_id(0)
    off_m = tl.program_id(1).to(tl.int64)
    off_b = tl.program_id(2)
    seq_start = tl.load(seq_offsets + off_b).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start
    start_m = off_m * BLOCK_M
    start_n = off_n * BLOCK_N
    if start_m >= seq_len:
        return
    Jagged += (seq_start + start_m) * stride_jm
    Dense += off_b.to(tl.int64) * stride_db
    Out += seq_start * stride_om
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    jg_ptrs = Jagged + offs_m[:, None] * stride_jm + offs_k[None, :]
    dn_ptrs = Dense + offs_k[:, None] * stride_dk + offs_n[None, :] * stride_dn
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        jg = tl.load(jg_ptrs, mask=(offs_m[:, None] < seq_len - start_m) &
            ((k + offs_k)[None, :] < K), other=0.0)
        dn = tl.load(dn_ptrs, mask=(k + offs_k)[:, None] < K and offs_n[
            None, :] < N, other=0.0)
        accumulator += tl.dot(jg, dn, allow_tf32=ALLOW_TF32)
        jg_ptrs += BLOCK_K
        dn_ptrs += BLOCK_K * stride_dk
    if HAS_BIAS:
        if ELEMENTWISE:
            Bias += (seq_start + start_m) * stride_bias_b
            bias_ptrs = Bias + offs_m[:, None] * stride_bias_b + offs_n[None, :
                ]
            bias = tl.load(bias_ptrs, mask=(offs_m[:, None] < seq_len -
                start_m) & (offs_n[None, :] < N), other=0.0)
            accumulator += bias.to(tl.float32)
        else:
            bias_ptrs = Bias + off_b.to(tl.int64) * stride_bias_b + offs_n
            bias = tl.load(bias_ptrs, mask=offs_n < N)
            accumulator += bias[None, :].to(tl.float32)
    out = accumulator.to(Out.dtype.element_ty)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    Out += start_m * stride_om
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < seq_len - start_m) & (
        offs_n[None, :] < N))


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

def switch_to_contiguous_if_needed(x: torch.Tensor) ->torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10 ** 9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


# Forward method (kernel launch code)
def __JaggedDenseBmmFunction_forward(ctx, max_seq_len: int, seq_offsets:
    torch.Tensor, jagged: torch.Tensor, dense: torch.Tensor):
    jagged = switch_to_contiguous_if_needed(jagged)
    L, D = jagged.shape
    B, _, K = dense.shape
    bmm_out = torch.empty((L, K), dtype=jagged.dtype, device=jagged.device)
    grid = lambda meta: (triton.cdiv(K, meta['BLOCK_N']), triton.cdiv(
        max_seq_len, meta['BLOCK_M']), B)
    jagged_dense_bmm_broadcast_add_kernel[grid](seq_offsets=seq_offsets,
        Jagged=jagged, Dense=dense, Bias=0, Out=bmm_out,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len), N=K, K=D,
        stride_jm=jagged.stride(0), stride_db=dense.stride(0), stride_dk=
        dense.stride(1), stride_dn=dense.stride(2), stride_bias_b=0,
        stride_om=bmm_out.stride(0), HAS_BIAS=False, ALLOW_TF32=torch.
        backends.cuda.matmul.allow_tf32, ELEMENTWISE=False)
    ctx.save_for_backward(seq_offsets, jagged, dense)
    ctx.B = B
    ctx.max_seq_len = max_seq_len
    ctx.K = K
    ctx.D = D
    return bmm_out


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton_autotune(configs=_get_bmm_reduce_sum_configs(), key=['M', 'N',
    'AUTOTUNE_MAX_SEQ_LEN'])
@triton.jit
def _jagged_jagged_bmm_reduce_sum(seq_offsets, JaggedA, JaggedB, Out,
    ReduceOut, M, N, AUTOTUNE_MAX_SEQ_LEN, stride_ak, stride_bk, stride_ob,
    stride_om, stride_on, stride_orb, stride_orn, REDUCE_JAGGEDB: tl.
    constexpr, ALLOW_TF32: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl
    .constexpr, BLOCK_K: tl.constexpr):
    """
    Computing bmm Out = Jagged x Jagged
    K is the jagged dimension
    JaggedA has shape (sum_B(K_i), M), JaggedB has shape (sum_B(K_i), N), and Out has shape (B, M, N)
    """
    off_m = tl.program_id(0).to(tl.int64)
    off_n = tl.program_id(1)
    off_b = tl.program_id(2)
    seq_start = tl.load(seq_offsets + off_b).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start
    start_m = off_m * BLOCK_M
    start_n = off_n * BLOCK_N
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    Out += off_b.to(tl.int64) * stride_ob
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    Out += start_m * stride_om
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    if REDUCE_JAGGEDB:
        out_reduce_ptrs = ReduceOut + off_b.to(tl.int64
            ) * stride_orb + offs_n * stride_orn
        acc_reduce = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if seq_len == 0:
        out = accumulator.to(Out.dtype.element_ty)
        tl.store(out_ptrs, out, mask=(offs_m[:, None] < M - start_m) & (
            offs_n[None, :] < N))
        if REDUCE_JAGGEDB:
            if off_m == 0:
                tl.store(out_reduce_ptrs, acc_reduce.to(ReduceOut.dtype.
                    element_ty), mask=offs_n < N)
        return
    JaggedA += seq_start * stride_ak
    JaggedB += seq_start * stride_bk
    offs_k = tl.arange(0, BLOCK_K)
    jg_a_ptrs = JaggedA + offs_k[None, :] * stride_ak + (start_m + offs_m)[
        :, None]
    jg_b_ptrs = JaggedB + offs_k[:, None] * stride_bk + offs_n[None, :]
    for k in range(0, seq_len, BLOCK_K):
        jg_a = tl.load(jg_a_ptrs, mask=(offs_m[:, None] < M - start_m) & ((
            k + offs_k)[None, :] < seq_len), other=0.0)
        jg_b = tl.load(jg_b_ptrs, mask=offs_n[None, :] < N and (k + offs_k)
            [:, None] < seq_len, other=0.0)
        accumulator += tl.dot(jg_a, jg_b, allow_tf32=ALLOW_TF32)
        if REDUCE_JAGGEDB:
            if off_m == 0:
                acc_reduce += tl.sum(jg_b.to(tl.float32), axis=0)
        jg_a_ptrs += BLOCK_K * stride_ak
        jg_b_ptrs += BLOCK_K * stride_bk
    out = accumulator.to(Out.dtype.element_ty)
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < M - start_m) & (offs_n[
        None, :] < N))
    if REDUCE_JAGGEDB:
        if off_m == 0:
            tl.store(out_reduce_ptrs, acc_reduce.to(ReduceOut.dtype.
                element_ty), mask=offs_n < N)


# Backward method (kernel launch code)
def __JaggedDenseBmmFunction_backward(ctx, d_bmm_out: torch.Tensor) ->Tuple[
    None, None, torch.Tensor, torch.Tensor]:
    seq_offsets, jagged, dense = ctx.saved_tensors
    d_jagged = torch.empty_like(jagged)
    d_dense = torch.empty_like(dense)
    grid = lambda meta: (triton.cdiv(ctx.D, meta['BLOCK_N']), triton.cdiv(
        ctx.max_seq_len, meta['BLOCK_M']), ctx.B)
    jagged_dense_bmm_broadcast_add_kernel[grid](seq_offsets=seq_offsets,
        Jagged=d_bmm_out, Dense=dense, Bias=None, Out=d_jagged,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(ctx.max_seq_len), N=ctx.D,
        K=ctx.K, stride_jm=d_bmm_out.stride(0), stride_db=dense.stride(0),
        stride_dk=dense.stride(2), stride_dn=dense.stride(1), stride_bias_b
        =0, stride_om=d_jagged.stride(0), HAS_BIAS=False, ALLOW_TF32=torch.
        backends.cuda.matmul.allow_tf32, ELEMENTWISE=False)
    grid = lambda meta: (triton.cdiv(ctx.D, meta['BLOCK_M']), triton.cdiv(
        ctx.K, meta['BLOCK_N']), ctx.B)
    _jagged_jagged_bmm_reduce_sum[grid](seq_offsets=seq_offsets, JaggedA=
        jagged, JaggedB=d_bmm_out, Out=d_dense, ReduceOut=None, M=ctx.D, N=
        ctx.K, AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(ctx.max_seq_len),
        stride_ak=jagged.stride(0), stride_bk=d_bmm_out.stride(0),
        stride_ob=d_dense.stride(0), stride_om=d_dense.stride(1), stride_on
        =d_dense.stride(2), stride_orb=0, stride_orn=0, REDUCE_JAGGEDB=
        False, ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32)
    return None, None, d_jagged, d_dense


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _JaggedDenseBmmFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, max_seq_len: int, seq_offsets: torch.Tensor, jagged:
        torch.Tensor, dense: torch.Tensor):
        jagged = switch_to_contiguous_if_needed(jagged)
        L, D = jagged.shape
        B, _, K = dense.shape
        bmm_out = torch.empty((L, K), dtype=jagged.dtype, device=jagged.device)
        grid = lambda meta: (triton.cdiv(K, meta['BLOCK_N']), triton.cdiv(
            max_seq_len, meta['BLOCK_M']), B)
        jagged_dense_bmm_broadcast_add_kernel[grid](seq_offsets=seq_offsets,
            Jagged=jagged, Dense=dense, Bias=0, Out=bmm_out,
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len), N=K, K=
            D, stride_jm=jagged.stride(0), stride_db=dense.stride(0),
            stride_dk=dense.stride(1), stride_dn=dense.stride(2),
            stride_bias_b=0, stride_om=bmm_out.stride(0), HAS_BIAS=False,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32, ELEMENTWISE=False
            )
        ctx.save_for_backward(seq_offsets, jagged, dense)
        ctx.B = B
        ctx.max_seq_len = max_seq_len
        ctx.K = K
        ctx.D = D
        return bmm_out

    @staticmethod
    def backward(ctx, d_bmm_out: torch.Tensor) ->Tuple[None, None, torch.
        Tensor, torch.Tensor]:
        seq_offsets, jagged, dense = ctx.saved_tensors
        d_jagged = torch.empty_like(jagged)
        d_dense = torch.empty_like(dense)
        grid = lambda meta: (triton.cdiv(ctx.D, meta['BLOCK_N']), triton.
            cdiv(ctx.max_seq_len, meta['BLOCK_M']), ctx.B)
        jagged_dense_bmm_broadcast_add_kernel[grid](seq_offsets=seq_offsets,
            Jagged=d_bmm_out, Dense=dense, Bias=None, Out=d_jagged,
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(ctx.max_seq_len), N=
            ctx.D, K=ctx.K, stride_jm=d_bmm_out.stride(0), stride_db=dense.
            stride(0), stride_dk=dense.stride(2), stride_dn=dense.stride(1),
            stride_bias_b=0, stride_om=d_jagged.stride(0), HAS_BIAS=False,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32, ELEMENTWISE=False
            )
        grid = lambda meta: (triton.cdiv(ctx.D, meta['BLOCK_M']), triton.
            cdiv(ctx.K, meta['BLOCK_N']), ctx.B)
        _jagged_jagged_bmm_reduce_sum[grid](seq_offsets=seq_offsets,
            JaggedA=jagged, JaggedB=d_bmm_out, Out=d_dense, ReduceOut=None,
            M=ctx.D, N=ctx.K, AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(ctx
            .max_seq_len), stride_ak=jagged.stride(0), stride_bk=d_bmm_out.
            stride(0), stride_ob=d_dense.stride(0), stride_om=d_dense.
            stride(1), stride_on=d_dense.stride(2), stride_orb=0,
            stride_orn=0, REDUCE_JAGGEDB=False, ALLOW_TF32=torch.backends.
            cuda.matmul.allow_tf32)
        return None, None, d_jagged, d_dense
