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

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton_autotune(configs=_get_jagged_dense_broadcast_add_configs(), key=[
    'AUTOTUNE_MAX_SEQ_LEN'])
@triton.jit
def jagged_dense_broadcast_add_kernel(seq_offsets, Jagged, Dense, Out,
    AUTOTUNE_MAX_SEQ_LEN, D, stride_jn, stride_db, stride_on, BLOCK_N: tl.
    constexpr, BLOCK_D: tl.constexpr):
    """
    Computing Out = Jagged + Dense
    JaggedA has shape (sum_B(N_i), D), Dense has shape (B, D), and Out has shape (sum_B(N_i), D)
    """
    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    seq_start = tl.load(seq_offsets + off_b)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start
    start_n = off_n * BLOCK_N
    if start_n >= seq_len:
        return
    Jagged += seq_start * stride_jn
    Dense += off_b * stride_db
    Out += seq_start * stride_on
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    jagged_ptrs = Jagged + offs_n[:, None] * stride_jn + offs_d[None, :]
    dense_ptrs = Dense + offs_d
    out_ptrs = Out + offs_n[:, None] * stride_jn + offs_d[None, :]
    for d in range(0, D, BLOCK_D):
        jg = tl.load(jagged_ptrs, mask=offs_n[:, None] < seq_len and (d +
            offs_d)[None, :] < D)
        dn = tl.load(dense_ptrs, mask=d + offs_d < D)
        out = jg + dn[None, :]
        tl.store(out_ptrs, out, mask=offs_n[:, None] < seq_len and (d +
            offs_d)[None, :] < D)
        dense_ptrs += BLOCK_D
        jagged_ptrs += BLOCK_D
        out_ptrs += BLOCK_D


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


def switch_to_contiguous_if_needed(x: torch.Tensor) ->torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10 ** 9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


# Forward method (kernel launch code)
def __JaggedDenseBroadcastAddFunction_forward(ctx, max_seq_len: int,
    seq_offsets: torch.Tensor, jagged: torch.Tensor, dense: torch.Tensor):
    jagged = switch_to_contiguous_if_needed(jagged)
    dense = switch_to_contiguous_if_needed(dense)
    L, D = jagged.shape
    B, _ = dense.shape
    out = torch.empty_like(jagged)
    grid = lambda meta: (B, triton.cdiv(max_seq_len, meta['BLOCK_N']))
    BLOCK_D = triton.next_power_of_2(D) if D < 64 else 64
    jagged_dense_broadcast_add_kernel[grid](seq_offsets=seq_offsets, Jagged
        =jagged, Dense=dense, Out=out, AUTOTUNE_MAX_SEQ_LEN=
        autotune_max_seq_len(max_seq_len), D=D, stride_jn=jagged.stride(0),
        stride_db=dense.stride(0), stride_on=out.stride(0), BLOCK_D=BLOCK_D)
    ctx.save_for_backward(seq_offsets)
    ctx.max_seq_len = max_seq_len
    ctx.B = B
    ctx.D = D
    return out


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def jagged_reduce_sum(seq_offsets, Jagged, Out, D, stride_jn, stride_ob,
    BLOCK_D: tl.constexpr):
    """
    Computing Out = Jagged + Dense
    JaggedA has shape (sum_B(N_i), D), Dense has shape (B, D), and Out has shape (sum_B(N_i), D)
    """
    off_b = tl.program_id(0)
    off_d = tl.program_id(1) * BLOCK_D
    seq_start = tl.load(seq_offsets + off_b)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start
    Jagged += seq_start * stride_jn
    Out += off_b * stride_ob
    offs_d = off_d + tl.arange(0, BLOCK_D)
    jagged_ptrs = Jagged + offs_d
    out_ptrs = Out + offs_d
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for _ in range(0, seq_len):
        jg = tl.load(jagged_ptrs, mask=offs_d < D)
        accumulator += jg
        jagged_ptrs += stride_jn
    out = accumulator.to(Out.dtype.element_ty)
    tl.store(out_ptrs, out, mask=offs_d < D)


# Backward method (kernel launch code)
def __JaggedDenseBroadcastAddFunction_backward(ctx, d_out: torch.Tensor
    ) ->Tuple[None, None, torch.Tensor, torch.Tensor]:
    seq_offsets = ctx.saved_tensors[0]
    d_dense = torch.empty((ctx.B, ctx.D), device=d_out.device, dtype=d_out.
        dtype)
    BLOCK_D = triton.next_power_of_2(ctx.D) if ctx.D < 64 else 64
    jagged_reduce_sum[ctx.B, triton.cdiv(ctx.D, BLOCK_D)](seq_offsets=
        seq_offsets, Jagged=d_out, Out=d_dense, D=ctx.D, stride_jn=d_out.
        stride(0), stride_ob=d_dense.stride(0), BLOCK_D=BLOCK_D)
    return None, None, d_out, d_dense


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _JaggedDenseBroadcastAddFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, max_seq_len: int, seq_offsets: torch.Tensor, jagged:
        torch.Tensor, dense: torch.Tensor):
        jagged = switch_to_contiguous_if_needed(jagged)
        dense = switch_to_contiguous_if_needed(dense)
        L, D = jagged.shape
        B, _ = dense.shape
        out = torch.empty_like(jagged)
        grid = lambda meta: (B, triton.cdiv(max_seq_len, meta['BLOCK_N']))
        BLOCK_D = triton.next_power_of_2(D) if D < 64 else 64
        jagged_dense_broadcast_add_kernel[grid](seq_offsets=seq_offsets,
            Jagged=jagged, Dense=dense, Out=out, AUTOTUNE_MAX_SEQ_LEN=
            autotune_max_seq_len(max_seq_len), D=D, stride_jn=jagged.stride
            (0), stride_db=dense.stride(0), stride_on=out.stride(0),
            BLOCK_D=BLOCK_D)
        ctx.save_for_backward(seq_offsets)
        ctx.max_seq_len = max_seq_len
        ctx.B = B
        ctx.D = D
        return out

    @staticmethod
    def backward(ctx, d_out: torch.Tensor) ->Tuple[None, None, torch.Tensor,
        torch.Tensor]:
        seq_offsets = ctx.saved_tensors[0]
        d_dense = torch.empty((ctx.B, ctx.D), device=d_out.device, dtype=
            d_out.dtype)
        BLOCK_D = triton.next_power_of_2(ctx.D) if ctx.D < 64 else 64
        jagged_reduce_sum[ctx.B, triton.cdiv(ctx.D, BLOCK_D)](seq_offsets=
            seq_offsets, Jagged=d_out, Out=d_dense, D=ctx.D, stride_jn=
            d_out.stride(0), stride_ob=d_dense.stride(0), BLOCK_D=BLOCK_D)
        return None, None, d_out, d_dense
