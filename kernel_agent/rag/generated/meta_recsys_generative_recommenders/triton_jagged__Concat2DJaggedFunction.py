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

def is_sm100() ->bool:
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major == 10 and props.minor == 0


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def concat_2D_jagged(OffsetsA, ValuesA, OffsetsB, ValuesB, DenseSize, Out,
    D, stride_ad, stride_bd, stride_dense_batch, stride_od, IS_DENSE_A: tl.
    constexpr, IS_DENSE_B: tl.constexpr, BLOCK_D: tl.constexpr, IS_REPLACE:
    tl.constexpr):
    concat_2D_jagged_w_prefix(OffsetsA, ValuesA, OffsetsB, ValuesB,
        DenseSize, Out, D, stride_ad, stride_bd, stride_dense_batch,
        stride_od, 0, IS_DENSE_A, IS_DENSE_B, BLOCK_D, IS_REPLACE)


@triton.jit
def concat_2D_jagged_jagged_w_prefix(OffsetsA, ValuesA, OffsetsB, ValuesB,
    Out, D, stride_ad, stride_bd, stride_od, n_prefix_from_B, BLOCK_D: tl.
    constexpr):
    concat_2D_jagged_w_prefix(OffsetsA, ValuesA, OffsetsB, ValuesB, 0, Out,
        D, stride_ad, stride_bd, 0, stride_od, n_prefix_from_B, IS_DENSE_A=
        False, IS_DENSE_B=False, BLOCK_D=BLOCK_D, IS_REPLACE=False)


@triton_autotune(configs=_get_split_concat_2d_jagged_multirow_configs(),
    key=['BLOCK_D'])
@triton.jit
def concat_2D_jagged_jagged_w_prefix_multirow(OffsetsA, ValuesA, OffsetsB,
    ValuesB, Out, D, stride_ad, stride_bd, stride_od, n_prefix_from_B,
    BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr):
    concat_2D_jagged_w_prefix_multirow(OffsetsA, ValuesA, OffsetsB, ValuesB,
        0, Out, D, stride_ad, stride_bd, 0, stride_od, n_prefix_from_B,
        IS_DENSE_A=False, IS_DENSE_B=False, BLOCK_D=BLOCK_D, BLOCK_N=
        BLOCK_N, IS_REPLACE=False)


@triton_autotune(configs=_get_split_concat_2d_jagged_multirow_configs(),
    key=['BLOCK_D'])
@triton.jit
def concat_2D_jagged_multirow(OffsetsA, ValuesA, OffsetsB, ValuesB,
    DenseSize, Out, D, stride_ad, stride_bd, stride_dense_batch, stride_od,
    IS_DENSE_A: tl.constexpr, IS_DENSE_B: tl.constexpr, BLOCK_D: tl.
    constexpr, BLOCK_N: tl.constexpr, IS_REPLACE: tl.constexpr):
    concat_2D_jagged_w_prefix_multirow(OffsetsA, ValuesA, OffsetsB, ValuesB,
        DenseSize, Out, D, stride_ad, stride_bd, stride_dense_batch,
        stride_od, 0, IS_DENSE_A, IS_DENSE_B, BLOCK_D, BLOCK_N, IS_REPLACE)


@triton.jit
def concat_2D_jagged_w_prefix(OffsetsA, ValuesA, OffsetsB, ValuesB,
    DenseSize, Out, D, stride_ad, stride_bd, stride_dense_batch, stride_od,
    n_prefix_from_B, IS_DENSE_A: tl.constexpr, IS_DENSE_B: tl.constexpr,
    BLOCK_D: tl.constexpr, IS_REPLACE: tl.constexpr):
    off_z = tl.program_id(1)
    off_n = tl.program_id(0)
    if IS_DENSE_A:
        seq_start_a = off_z * DenseSize
        seq_len_a = DenseSize
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_len_b = seq_end_b - seq_start_b
    elif IS_DENSE_B:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
        seq_start_b = off_z * DenseSize
        seq_len_b = DenseSize
    else:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_len_b = seq_end_b - seq_start_b
    if IS_REPLACE:
        seq_len = seq_len_a
    else:
        seq_len = seq_len_a + seq_len_b
    if off_n >= seq_len:
        return
    offs_d = tl.arange(0, BLOCK_D)
    if IS_REPLACE:
        out_seq_start = seq_start_a + off_n
        out_seq_b_start = seq_len_a - seq_len_b
    else:
        out_seq_start = seq_start_a + seq_start_b + off_n
        out_seq_b_start = seq_len_a + n_prefix_from_B
    out_ptrs = Out + out_seq_start.to(tl.int64) * stride_od + offs_d
    if off_n < out_seq_b_start and off_n >= n_prefix_from_B:
        off_a = off_n - n_prefix_from_B
        if IS_DENSE_A:
            in_ptrs = ValuesA + off_a.to(tl.int64) * stride_ad + off_z.to(tl
                .int64) * stride_dense_batch + offs_d
        else:
            in_ptrs = ValuesA + (off_a + seq_start_a).to(tl.int64
                ) * stride_ad + offs_d
    else:
        off_b = off_n - out_seq_b_start + n_prefix_from_B
        if off_n < n_prefix_from_B:
            off_b += out_seq_b_start - n_prefix_from_B
        if IS_DENSE_B:
            in_ptrs = ValuesB + off_b.to(tl.int64) * stride_bd + off_z.to(tl
                .int64) * stride_dense_batch + offs_d
        else:
            in_ptrs = ValuesB + (off_b + seq_start_b).to(tl.int64
                ) * stride_bd + offs_d
    v = tl.load(in_ptrs, mask=offs_d < D)
    tl.store(out_ptrs, v, mask=offs_d < D)


@triton.jit
def concat_2D_jagged_w_prefix_multirow(OffsetsA, ValuesA, OffsetsB, ValuesB,
    DenseSize, Out, D, stride_ad, stride_bd, stride_dense_batch, stride_od,
    n_prefix_from_B, IS_DENSE_A: tl.constexpr, IS_DENSE_B: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr, IS_REPLACE: tl.constexpr):
    off_z = tl.program_id(1)
    off_block_n = tl.program_id(0)
    if IS_DENSE_A:
        seq_start_a = off_z * DenseSize
        seq_len_a = DenseSize
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_len_b = seq_end_b - seq_start_b
    elif IS_DENSE_B:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
        seq_start_b = off_z * DenseSize
        seq_len_b = DenseSize
    else:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_len_b = seq_end_b - seq_start_b
    if IS_REPLACE:
        seq_len = seq_len_a
        out_seq_start = seq_start_a
        out_seq_b_start = seq_len_a - seq_len_b
    else:
        seq_len = seq_len_a + seq_len_b
        out_seq_start = seq_start_a + seq_start_b
        out_seq_b_start = seq_len_a + n_prefix_from_B
    start_n = off_block_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    if start_n >= seq_len:
        return
    valid_mask = offs_n < seq_len
    out_ptrs = Out + (out_seq_start + offs_n[:, None]).to(tl.int64
        ) * stride_od + offs_d[None, :]
    to_a_mask = (offs_n < out_seq_b_start) & (offs_n >= n_prefix_from_B
        ) & valid_mask
    to_b_mask = ~to_a_mask & valid_mask
    off_a = offs_n - n_prefix_from_B
    if IS_DENSE_A:
        in_a_ptrs = ValuesA + off_a[:, None].to(tl.int64
            ) * stride_ad + off_z.to(tl.int64) * stride_dense_batch + offs_d[
            None, :]
    else:
        in_a_ptrs = ValuesA + (off_a[:, None] + seq_start_a).to(tl.int64
            ) * stride_ad + offs_d[None, :]
    v_a = tl.load(in_a_ptrs, mask=to_a_mask[:, None] & (offs_d[None, :] < D
        ), other=0.0)
    tl.store(out_ptrs, v_a, mask=to_a_mask[:, None] & (offs_d[None, :] < D))
    prefix_mask = offs_n < n_prefix_from_B
    off_b = tl.where(prefix_mask, offs_n, offs_n - out_seq_b_start +
        n_prefix_from_B)
    if IS_DENSE_B:
        in_b_ptrs = ValuesB + off_b[:, None].to(tl.int64
            ) * stride_bd + off_z.to(tl.int64) * stride_dense_batch + offs_d[
            None, :]
    else:
        in_b_ptrs = ValuesB + (off_b[:, None] + seq_start_b).to(tl.int64
            ) * stride_bd + offs_d[None, :]
    v_b = tl.load(in_b_ptrs, mask=to_b_mask[:, None] & (offs_d[None, :] < D
        ), other=0.0)
    tl.store(out_ptrs, v_b, mask=to_b_mask[:, None] & (offs_d[None, :] < D))


def switch_to_contiguous_if_needed(x: torch.Tensor) ->torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10 ** 9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


def _triton_concat_2D_jagged_internal(values_a: torch.Tensor, values_b:
    torch.Tensor, values_out: torch.Tensor, max_seq_len: int, B: int,
    offsets_a: Optional[torch.Tensor], offsets_b: Optional[torch.Tensor], D:
    int, dense_size: int, stride_dense_batch: int, n_prefix: int,
    is_dense_a: bool, is_dense_b: bool, is_replace: bool, BLOCK_D: int) ->None:
    if n_prefix != 0:
        if is_sm100():

            def grid(meta):
                return triton.cdiv(max_seq_len, meta['BLOCK_N']), B
            concat_2D_jagged_jagged_w_prefix_multirow[grid](OffsetsA=
                offsets_a, ValuesA=values_a, OffsetsB=offsets_b, ValuesB=
                values_b, Out=values_out, D=D, stride_ad=values_a.stride(-2
                ), stride_bd=values_b.stride(-2), stride_od=values_out.
                stride(0), n_prefix_from_B=n_prefix, BLOCK_D=BLOCK_D)
        else:
            concat_2D_jagged_jagged_w_prefix[max_seq_len, B](OffsetsA=
                offsets_a, ValuesA=values_a, OffsetsB=offsets_b, ValuesB=
                values_b, Out=values_out, D=D, stride_ad=values_a.stride(-2
                ), stride_bd=values_b.stride(-2), stride_od=values_out.
                stride(0), n_prefix_from_B=n_prefix, BLOCK_D=BLOCK_D)
    elif is_sm100():

        def grid(meta):
            return triton.cdiv(max_seq_len, meta['BLOCK_N']), B
        concat_2D_jagged_multirow[grid](OffsetsA=offsets_a, ValuesA=
            values_a, OffsetsB=offsets_b, ValuesB=values_b, DenseSize=
            dense_size, Out=values_out, D=D, stride_ad=values_a.stride(-2),
            stride_bd=values_b.stride(-2), stride_dense_batch=
            stride_dense_batch, stride_od=values_out.stride(0), IS_DENSE_A=
            is_dense_a, IS_DENSE_B=is_dense_b, BLOCK_D=BLOCK_D, IS_REPLACE=
            is_replace)
    else:
        concat_2D_jagged[max_seq_len, B](OffsetsA=offsets_a, ValuesA=
            values_a, OffsetsB=offsets_b, ValuesB=values_b, DenseSize=
            dense_size, Out=values_out, D=D, stride_ad=values_a.stride(-2),
            stride_bd=values_b.stride(-2), stride_dense_batch=
            stride_dense_batch, stride_od=values_out.stride(0), IS_DENSE_A=
            is_dense_a, IS_DENSE_B=is_dense_b, BLOCK_D=BLOCK_D, IS_REPLACE=
            is_replace)


# Forward method (kernel launch code)
def __Concat2DJaggedFunction_forward(ctx, max_seq_len: int, values_a: torch
    .Tensor, values_b: torch.Tensor, offsets_a: Optional[torch.Tensor]=None,
    offsets_b: Optional[torch.Tensor]=None, is_replace: bool=False,
    n_prefix_from_right: int=0):
    values_a = switch_to_contiguous_if_needed(values_a)
    values_b = switch_to_contiguous_if_needed(values_b)
    is_dense_a = offsets_a is None
    is_dense_b = offsets_b is None
    dense_size: int = 0
    if is_dense_a:
        assert offsets_b is not None
        B, dense_size, D = values_a.shape
        seq_len_a = dense_size * B
        seq_len_b, _ = values_b.shape
        device = values_b.device
        dtype = values_b.dtype
        stride_dense_batch = values_a.stride(0)
    elif is_dense_b:
        assert offsets_a is not None
        B, dense_size, D = values_b.shape
        seq_len_a, _ = values_a.shape
        seq_len_b = dense_size * B
        device = values_a.device
        dtype = values_a.dtype
        stride_dense_batch = values_b.stride(0)
    else:
        assert offsets_a is not None and offsets_b is not None
        B = offsets_a.shape[0] - 1
        seq_len_a, D = values_a.shape
        seq_len_b, _ = values_b.shape
        device = values_a.device
        dtype = values_a.dtype
        stride_dense_batch = 0
    BLOCK_D = triton.next_power_of_2(D)
    if is_replace:
        values_out = torch.empty_like(values_a)
    else:
        values_out = torch.empty((seq_len_a + seq_len_b, D), device=device,
            dtype=dtype)
    _triton_concat_2D_jagged_internal(values_a=values_a, values_b=values_b,
        values_out=values_out, max_seq_len=max_seq_len, B=B, offsets_a=
        offsets_a, offsets_b=offsets_b, D=D, dense_size=dense_size,
        stride_dense_batch=stride_dense_batch, n_prefix=n_prefix_from_right,
        is_dense_a=is_dense_a, is_dense_b=is_dense_b, is_replace=is_replace,
        BLOCK_D=BLOCK_D)
    ctx.save_for_backward(offsets_a, offsets_b)
    ctx.max_seq_len = max_seq_len
    ctx.seq_len_a = seq_len_a
    ctx.seq_len_b = seq_len_b
    ctx.is_dense_a = is_dense_a
    ctx.is_dense_b = is_dense_b
    ctx.dense_size = dense_size
    ctx.is_replace = is_replace
    ctx.n_prefix_from_right = n_prefix_from_right
    return values_out


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def split_2D_jagged(JaggedIn, DenseSize, OffsetsA, OffsetsB, OutA, OutB, D,
    stride_id, stride_ad, stride_bd, IS_DENSE_A: tl.constexpr, IS_DENSE_B:
    tl.constexpr, BLOCK_D: tl.constexpr, IS_REPLACE: tl.constexpr):
    split_2D_jagged_w_prefix(JaggedIn, DenseSize, OffsetsA, OffsetsB, OutA,
        OutB, D, stride_id, stride_ad, stride_bd, 0, IS_DENSE_A, IS_DENSE_B,
        BLOCK_D, IS_REPLACE)


@triton.jit
def split_2D_jagged_jagged_w_prefix(JaggedIn, OffsetsA, OffsetsB, OutA,
    OutB, D, stride_id, stride_ad, stride_bd, n_prefix_to_B, BLOCK_D: tl.
    constexpr):
    split_2D_jagged_w_prefix(JaggedIn, 0, OffsetsA, OffsetsB, OutA, OutB, D,
        stride_id, stride_ad, stride_bd, n_prefix_to_B, IS_DENSE_A=False,
        IS_DENSE_B=False, BLOCK_D=BLOCK_D, IS_REPLACE=False)


@triton_autotune(configs=_get_split_concat_2d_jagged_multirow_configs(),
    key=['BLOCK_D'])
@triton.jit
def split_2D_jagged_jagged_w_prefix_multirow(JaggedIn, OffsetsA, OffsetsB,
    OutA, OutB, D, stride_id, stride_ad, stride_bd, n_prefix_to_B, BLOCK_D:
    tl.constexpr, BLOCK_N: tl.constexpr):
    split_2D_jagged_w_prefix_multirow(JaggedIn, 0, OffsetsA, OffsetsB, OutA,
        OutB, D, stride_id, stride_ad, stride_bd, n_prefix_to_B, IS_DENSE_A
        =False, IS_DENSE_B=False, BLOCK_D=BLOCK_D, BLOCK_N=BLOCK_N,
        IS_REPLACE=False)


@triton_autotune(configs=_get_split_concat_2d_jagged_multirow_configs(),
    key=['BLOCK_D'])
@triton.jit
def split_2D_jagged_multirow(JaggedIn, DenseSize, OffsetsA, OffsetsB, OutA,
    OutB, D, stride_id, stride_ad, stride_bd, IS_DENSE_A: tl.constexpr,
    IS_DENSE_B: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_REPLACE: tl.constexpr):
    split_2D_jagged_w_prefix_multirow(JaggedIn, DenseSize, OffsetsA,
        OffsetsB, OutA, OutB, D, stride_id, stride_ad, stride_bd, 0,
        IS_DENSE_A, IS_DENSE_B, BLOCK_D, BLOCK_N, IS_REPLACE)


@triton.jit
def split_2D_jagged_w_prefix(JaggedIn, DenseSize, OffsetsA, OffsetsB, OutA,
    OutB, D, stride_id, stride_ad, stride_bd, n_prefix_to_B, IS_DENSE_A: tl
    .constexpr, IS_DENSE_B: tl.constexpr, BLOCK_D: tl.constexpr, IS_REPLACE:
    tl.constexpr):
    off_z = tl.program_id(1)
    off_n = tl.program_id(0)
    if IS_DENSE_A:
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_start_a = off_z * DenseSize
        seq_len_a = DenseSize
        seq_len_b = seq_end_b - seq_start_b
    elif IS_DENSE_B:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
        seq_start_b = off_z * DenseSize
        seq_len_b = DenseSize
    else:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_len_b = seq_end_b - seq_start_b
    if IS_REPLACE:
        seq_len = seq_len_a
    else:
        seq_len = seq_len_a + seq_len_b
    if off_n >= seq_len:
        return
    if IS_REPLACE:
        seq_start = seq_start_a
        out_seq_b_start = seq_len_a - seq_len_b
    else:
        seq_start = seq_start_a + seq_start_b
        out_seq_b_start = seq_len_a + n_prefix_to_B
    offs_d = tl.arange(0, BLOCK_D)
    in_ptrs = JaggedIn + (seq_start + off_n).to(tl.int64) * stride_id + offs_d
    if off_n < out_seq_b_start and off_n >= n_prefix_to_B:
        off_a = off_n - n_prefix_to_B
        out_ptrs = OutA + (off_a + seq_start_a).to(tl.int64
            ) * stride_ad + offs_d
    else:
        off_b = off_n - out_seq_b_start + n_prefix_to_B
        if off_n < n_prefix_to_B:
            off_b += out_seq_b_start - n_prefix_to_B
        out_ptrs = OutB + (off_b + seq_start_b).to(tl.int64
            ) * stride_bd + offs_d
    v = tl.load(in_ptrs, mask=offs_d < D)
    tl.store(out_ptrs, v, mask=offs_d < D)


@triton.jit
def split_2D_jagged_w_prefix_multirow(JaggedIn, DenseSize, OffsetsA,
    OffsetsB, OutA, OutB, D, stride_id, stride_ad, stride_bd, n_prefix_to_B,
    IS_DENSE_A: tl.constexpr, IS_DENSE_B: tl.constexpr, BLOCK_D: tl.
    constexpr, BLOCK_N: tl.constexpr, IS_REPLACE: tl.constexpr):
    off_z = tl.program_id(1)
    off_block_n = tl.program_id(0)
    if IS_DENSE_A:
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_start_a = off_z * DenseSize
        seq_len_a = DenseSize
        seq_len_b = seq_end_b - seq_start_b
    elif IS_DENSE_B:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
        seq_start_b = off_z * DenseSize
        seq_len_b = DenseSize
    else:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_len_b = seq_end_b - seq_start_b
    if IS_REPLACE:
        seq_len = seq_len_a
    else:
        seq_len = seq_len_a + seq_len_b
    if IS_REPLACE:
        seq_start = seq_start_a
        out_seq_b_start = seq_len_a - seq_len_b
    else:
        seq_start = seq_start_a + seq_start_b
        out_seq_b_start = seq_len_a + n_prefix_to_B
    start_n = off_block_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    if start_n >= seq_len:
        return
    valid_mask = offs_n < seq_len
    in_ptrs = JaggedIn + (seq_start + offs_n[:, None]).to(tl.int64
        ) * stride_id + offs_d[None, :]
    v = tl.load(in_ptrs, mask=valid_mask[:, None] & (offs_d[None, :] < D),
        other=0.0)
    to_a_mask = (offs_n < out_seq_b_start) & (offs_n >= n_prefix_to_B
        ) & valid_mask
    to_b_mask = ~to_a_mask & valid_mask
    off_a = offs_n - n_prefix_to_B
    out_a_ptrs = OutA + (off_a[:, None] + seq_start_a).to(tl.int64
        ) * stride_ad + offs_d[None, :]
    tl.store(out_a_ptrs, v, mask=to_a_mask[:, None] & (offs_d[None, :] < D))
    prefix_mask = offs_n < n_prefix_to_B
    off_b = tl.where(prefix_mask, offs_n, offs_n - out_seq_b_start +
        n_prefix_to_B)
    out_b_ptrs = OutB + (off_b[:, None] + seq_start_b).to(tl.int64
        ) * stride_bd + offs_d[None, :]
    tl.store(out_b_ptrs, v, mask=to_b_mask[:, None] & (offs_d[None, :] < D))


def _triton_split_2D_jagged_internal(jagged_in: torch.Tensor, max_seq_len:
    int, B: int, offsets_a: Optional[torch.Tensor], offsets_b: Optional[
    torch.Tensor], out_a: torch.Tensor, out_b: torch.Tensor, D: int,
    dense_size: int, n_prefix: int, is_dense_a: bool, is_dense_b: bool,
    is_replace: bool, BLOCK_D: int) ->None:
    if n_prefix != 0:
        if is_sm100():

            def grid(meta):
                return triton.cdiv(max_seq_len, meta['BLOCK_N']), B
            split_2D_jagged_jagged_w_prefix_multirow[grid](JaggedIn=
                jagged_in, OffsetsA=offsets_a, OffsetsB=offsets_b, OutA=
                out_a, OutB=out_b, D=D, stride_id=jagged_in.stride(0),
                stride_ad=out_a.stride(0), stride_bd=out_b.stride(0),
                n_prefix_to_B=n_prefix, BLOCK_D=BLOCK_D)
        else:
            split_2D_jagged_jagged_w_prefix[max_seq_len, B](JaggedIn=
                jagged_in, OffsetsA=offsets_a, OffsetsB=offsets_b, OutA=
                out_a, OutB=out_b, D=D, stride_id=jagged_in.stride(0),
                stride_ad=out_a.stride(0), stride_bd=out_b.stride(0),
                n_prefix_to_B=n_prefix, BLOCK_D=BLOCK_D)
    elif is_sm100():

        def grid(meta):
            return triton.cdiv(max_seq_len, meta['BLOCK_N']), B
        split_2D_jagged_multirow[grid](JaggedIn=jagged_in, DenseSize=
            dense_size, OffsetsA=offsets_a, OffsetsB=offsets_b, OutA=out_a,
            OutB=out_b, D=D, stride_id=jagged_in.stride(0), stride_ad=out_a
            .stride(0), stride_bd=out_b.stride(0), IS_DENSE_A=is_dense_a,
            IS_DENSE_B=is_dense_b, BLOCK_D=BLOCK_D, IS_REPLACE=is_replace)
    else:
        split_2D_jagged[max_seq_len, B](JaggedIn=jagged_in, DenseSize=
            dense_size, OffsetsA=offsets_a, OffsetsB=offsets_b, OutA=out_a,
            OutB=out_b, D=D, stride_id=jagged_in.stride(0), stride_ad=out_a
            .stride(0), stride_bd=out_b.stride(0), IS_DENSE_A=is_dense_a,
            IS_DENSE_B=is_dense_b, BLOCK_D=BLOCK_D, IS_REPLACE=is_replace)


# Backward method (kernel launch code)
def __Concat2DJaggedFunction_backward(ctx, d_out: torch.Tensor) ->Tuple[
    None, torch.Tensor, torch.Tensor, None, None, None, None]:
    offsets_a, offsets_b = ctx.saved_tensors
    is_dense_a, is_dense_b, is_replace = (ctx.is_dense_a, ctx.is_dense_b,
        ctx.is_replace)
    dense_size = ctx.dense_size
    if is_dense_a:
        B = offsets_b.shape[0] - 1
    else:
        B = offsets_a.shape[0] - 1
    _, D = d_out.shape
    BLOCK_D = triton.next_power_of_2(D)
    values_a = torch.zeros((ctx.seq_len_a, D), device=d_out.device, dtype=
        d_out.dtype)
    values_b = torch.empty((ctx.seq_len_b, D), device=d_out.device, dtype=
        d_out.dtype)
    _triton_split_2D_jagged_internal(jagged_in=d_out, max_seq_len=ctx.
        max_seq_len, B=B, offsets_a=offsets_a, offsets_b=offsets_b, out_a=
        values_a, out_b=values_b, D=D, dense_size=dense_size, n_prefix=ctx.
        n_prefix_from_right, is_dense_a=is_dense_a, is_dense_b=is_dense_b,
        is_replace=is_replace, BLOCK_D=BLOCK_D)
    if is_dense_a:
        values_a = values_a.reshape((B, dense_size, D))
    elif is_dense_b:
        values_b = values_b.reshape((B, dense_size, D))
    return None, values_a, values_b, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _Concat2DJaggedFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, max_seq_len: int, values_a: torch.Tensor, values_b:
        torch.Tensor, offsets_a: Optional[torch.Tensor]=None, offsets_b:
        Optional[torch.Tensor]=None, is_replace: bool=False,
        n_prefix_from_right: int=0):
        values_a = switch_to_contiguous_if_needed(values_a)
        values_b = switch_to_contiguous_if_needed(values_b)
        is_dense_a = offsets_a is None
        is_dense_b = offsets_b is None
        dense_size: int = 0
        if is_dense_a:
            assert offsets_b is not None
            B, dense_size, D = values_a.shape
            seq_len_a = dense_size * B
            seq_len_b, _ = values_b.shape
            device = values_b.device
            dtype = values_b.dtype
            stride_dense_batch = values_a.stride(0)
        elif is_dense_b:
            assert offsets_a is not None
            B, dense_size, D = values_b.shape
            seq_len_a, _ = values_a.shape
            seq_len_b = dense_size * B
            device = values_a.device
            dtype = values_a.dtype
            stride_dense_batch = values_b.stride(0)
        else:
            assert offsets_a is not None and offsets_b is not None
            B = offsets_a.shape[0] - 1
            seq_len_a, D = values_a.shape
            seq_len_b, _ = values_b.shape
            device = values_a.device
            dtype = values_a.dtype
            stride_dense_batch = 0
        BLOCK_D = triton.next_power_of_2(D)
        if is_replace:
            values_out = torch.empty_like(values_a)
        else:
            values_out = torch.empty((seq_len_a + seq_len_b, D), device=
                device, dtype=dtype)
        _triton_concat_2D_jagged_internal(values_a=values_a, values_b=
            values_b, values_out=values_out, max_seq_len=max_seq_len, B=B,
            offsets_a=offsets_a, offsets_b=offsets_b, D=D, dense_size=
            dense_size, stride_dense_batch=stride_dense_batch, n_prefix=
            n_prefix_from_right, is_dense_a=is_dense_a, is_dense_b=
            is_dense_b, is_replace=is_replace, BLOCK_D=BLOCK_D)
        ctx.save_for_backward(offsets_a, offsets_b)
        ctx.max_seq_len = max_seq_len
        ctx.seq_len_a = seq_len_a
        ctx.seq_len_b = seq_len_b
        ctx.is_dense_a = is_dense_a
        ctx.is_dense_b = is_dense_b
        ctx.dense_size = dense_size
        ctx.is_replace = is_replace
        ctx.n_prefix_from_right = n_prefix_from_right
        return values_out

    @staticmethod
    def backward(ctx, d_out: torch.Tensor) ->Tuple[None, torch.Tensor,
        torch.Tensor, None, None, None, None]:
        offsets_a, offsets_b = ctx.saved_tensors
        is_dense_a, is_dense_b, is_replace = (ctx.is_dense_a, ctx.
            is_dense_b, ctx.is_replace)
        dense_size = ctx.dense_size
        if is_dense_a:
            B = offsets_b.shape[0] - 1
        else:
            B = offsets_a.shape[0] - 1
        _, D = d_out.shape
        BLOCK_D = triton.next_power_of_2(D)
        values_a = torch.zeros((ctx.seq_len_a, D), device=d_out.device,
            dtype=d_out.dtype)
        values_b = torch.empty((ctx.seq_len_b, D), device=d_out.device,
            dtype=d_out.dtype)
        _triton_split_2D_jagged_internal(jagged_in=d_out, max_seq_len=ctx.
            max_seq_len, B=B, offsets_a=offsets_a, offsets_b=offsets_b,
            out_a=values_a, out_b=values_b, D=D, dense_size=dense_size,
            n_prefix=ctx.n_prefix_from_right, is_dense_a=is_dense_a,
            is_dense_b=is_dense_b, is_replace=is_replace, BLOCK_D=BLOCK_D)
        if is_dense_a:
            values_a = values_a.reshape((B, dense_size, D))
        elif is_dense_b:
            values_b = values_b.reshape((B, dense_size, D))
        return None, values_a, values_b, None, None, None, None
