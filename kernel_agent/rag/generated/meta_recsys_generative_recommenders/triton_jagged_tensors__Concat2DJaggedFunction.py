# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/meta-recsys/generative-recommenders
# Source-Files: generative_recommenders/ops/triton/triton_jagged_tensors.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rzwbk0zd/generative-recommenders-main/generative_recommenders/ops/triton/triton_jagged_tensors.py
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
def _concat_2D_jagged(ValuesA, ValuesB, OffsetsA, OffsetsB, MaxLenA,
    MaxLenB, Out, D, stride_ad, stride_bd, stride_od, n_prefix_from_B,
    IS_DENSE_A: tl.constexpr, IS_DENSE_B: tl.constexpr, BLOCK_D: tl.constexpr):
    off_z = tl.program_id(1)
    off_n = tl.program_id(0)
    if IS_DENSE_A:
        seq_start_a = off_z * MaxLenA
        seq_len_a = MaxLenA
    else:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
    if IS_DENSE_B:
        seq_start_b = off_z * MaxLenB
        seq_len_b = MaxLenB
    else:
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_len_b = seq_end_b - seq_start_b
    seq_len = seq_len_a + seq_len_b
    if off_n >= seq_len:
        return
    offs_d = tl.arange(0, BLOCK_D)
    out_seq_start = seq_start_a + seq_start_b + off_n
    out_ptrs = Out + out_seq_start.to(tl.int64) * stride_od + offs_d
    if off_n < n_prefix_from_B:
        in_ptrs = ValuesB + (off_n + seq_start_b).to(tl.int64
            ) * stride_bd + offs_d
    elif off_n < seq_len_a + n_prefix_from_B:
        in_ptrs = ValuesA + (off_n - n_prefix_from_B + seq_start_a).to(tl.int64
            ) * stride_ad + offs_d
    else:
        in_ptrs = ValuesB + (off_n - seq_len_a + seq_start_b).to(tl.int64
            ) * stride_bd + offs_d
    v = tl.load(in_ptrs, mask=offs_d < D)
    tl.store(out_ptrs, v, mask=offs_d < D)


@triton.jit
def _concat_2D_jagged_multirow(ValuesA, ValuesB, OffsetsA, OffsetsB,
    MaxLenA, MaxLenB, Out, D, stride_ad, stride_bd, stride_od,
    n_prefix_from_B, IS_DENSE_A: tl.constexpr, IS_DENSE_B: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr):
    off_z = tl.program_id(1)
    block_n = tl.program_id(0)
    if IS_DENSE_A:
        seq_start_a = off_z * MaxLenA
        seq_len_a = MaxLenA
    else:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
    if IS_DENSE_B:
        seq_start_b = off_z * MaxLenB
        seq_len_b = MaxLenB
    else:
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_len_b = seq_end_b - seq_start_b
    seq_len = seq_len_a + seq_len_b
    start_n = block_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    valid_mask = offs_n < seq_len
    out_seq_start = seq_start_a + seq_start_b + offs_n
    out_ptrs = Out + out_seq_start[:, None].to(tl.int64) * stride_od + offs_d[
        None, :]
    from_prefix_b_mask = (offs_n < n_prefix_from_B) & valid_mask
    from_a_mask = (offs_n >= n_prefix_from_B) & (offs_n < seq_len_a +
        n_prefix_from_B) & valid_mask
    from_suffix_b_mask = (offs_n >= seq_len_a + n_prefix_from_B) & valid_mask
    in_b1_ptrs = ValuesB + (offs_n[:, None] + seq_start_b).to(tl.int64
        ) * stride_bd + offs_d[None, :]
    v_b1 = tl.load(in_b1_ptrs, mask=from_prefix_b_mask[:, None] & (offs_d[
        None, :] < D), other=0.0)
    tl.store(out_ptrs, v_b1, mask=from_prefix_b_mask[:, None] & (offs_d[
        None, :] < D))
    off_a = offs_n - n_prefix_from_B
    in_a_ptrs = ValuesA + (off_a[:, None] + seq_start_a).to(tl.int64
        ) * stride_ad + offs_d[None, :]
    v_a = tl.load(in_a_ptrs, mask=from_a_mask[:, None] & (offs_d[None, :] <
        D), other=0.0)
    tl.store(out_ptrs, v_a, mask=from_a_mask[:, None] & (offs_d[None, :] < D))
    off_b = offs_n - seq_len_a
    in_b2_ptrs = ValuesB + (off_b[:, None] + seq_start_b).to(tl.int64
        ) * stride_bd + offs_d[None, :]
    v_b2 = tl.load(in_b2_ptrs, mask=from_suffix_b_mask[:, None] & (offs_d[
        None, :] < D), other=0.0)
    tl.store(out_ptrs, v_b2, mask=from_suffix_b_mask[:, None] & (offs_d[
        None, :] < D))


@triton_autotune(configs=_get_concat_split_2d_jagged_multirow_configs(),
    key=['BLOCK_D'])
@triton.jit
def concat_2D_jagged_multirow(ValuesA, ValuesB, OffsetsA, OffsetsB, MaxLenA,
    MaxLenB, Out, D, stride_ad, stride_bd, stride_od, n_prefix_from_B,
    IS_DENSE_A: tl.constexpr, IS_DENSE_B: tl.constexpr, BLOCK_D: tl.
    constexpr, BLOCK_N: tl.constexpr):
    _concat_2D_jagged_multirow(ValuesA, ValuesB, OffsetsA, OffsetsB,
        MaxLenA, MaxLenB, Out, D, stride_ad, stride_bd, stride_od,
        n_prefix_from_B, IS_DENSE_A, IS_DENSE_B, BLOCK_D, BLOCK_N)


def switch_to_contiguous_if_needed(x: torch.Tensor) ->torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10 ** 9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


def _triton_concat_2D_jagged_internal(values_a: torch.Tensor, values_b:
    torch.Tensor, values_out: torch.Tensor, max_seq_len: int, B: int,
    offsets_a: Optional[torch.Tensor], offsets_b: Optional[torch.Tensor],
    max_len_a: Optional[int], max_len_b: Optional[int], D: int,
    n_prefix_from_B: int, is_dense_a: bool, is_dense_b: bool, BLOCK_D: int
    ) ->None:
    if is_sm100():

        def grid(meta):
            return triton.cdiv(max_seq_len, meta['BLOCK_N']), B
        concat_2D_jagged_multirow[grid](ValuesA=values_a, ValuesB=values_b,
            OffsetsA=offsets_a, OffsetsB=offsets_b, MaxLenA=max_len_a,
            MaxLenB=max_len_b, Out=values_out, D=D, stride_ad=values_a.
            stride(-2), stride_bd=values_b.stride(-2), stride_od=values_out
            .stride(-2), n_prefix_from_B=n_prefix_from_B, IS_DENSE_A=
            is_dense_a, IS_DENSE_B=is_dense_b, BLOCK_D=BLOCK_D)
    else:
        _concat_2D_jagged[max_seq_len, B](ValuesA=values_a, ValuesB=
            values_b, OffsetsA=offsets_a, OffsetsB=offsets_b, MaxLenA=
            max_len_a, MaxLenB=max_len_b, Out=values_out, D=D, stride_ad=
            values_a.stride(-2), stride_bd=values_b.stride(-2), stride_od=
            values_out.stride(-2), n_prefix_from_B=n_prefix_from_B,
            IS_DENSE_A=is_dense_a, IS_DENSE_B=is_dense_b, BLOCK_D=BLOCK_D)


def is_sm100() ->bool:
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major == 10 and props.minor == 0


# Forward method (kernel launch code)
def __Concat2DJaggedFunction_forward(ctx, max_seq_len: int, values_a: torch
    .Tensor, values_b: torch.Tensor, max_len_a: Optional[int], max_len_b:
    Optional[int], offsets_a: Optional[torch.Tensor], offsets_b: Optional[
    torch.Tensor], n_prefix_from_B: int):
    values_a = switch_to_contiguous_if_needed(values_a)
    values_b = switch_to_contiguous_if_needed(values_b)
    is_dense_a = offsets_a is None
    is_dense_b = offsets_b is None
    total_len_a, D = values_a.shape
    total_len_b, _ = values_b.shape
    if is_dense_a:
        assert max_len_a is not None
        B = total_len_a // max_len_a
    else:
        assert offsets_a is not None
        B = offsets_a.shape[0] - 1
    if is_dense_b:
        assert max_len_b is not None
        B = total_len_b // max_len_b
    else:
        assert offsets_b is not None
        B = offsets_b.shape[0] - 1
    total_seq_len = total_len_a + total_len_b
    BLOCK_D = triton.next_power_of_2(D)
    values_out = torch.empty((total_seq_len, D), device=values_a.device,
        dtype=values_a.dtype)
    _triton_concat_2D_jagged_internal(values_a=values_a, values_b=values_b,
        values_out=values_out, max_seq_len=max_seq_len, B=B, offsets_a=
        offsets_a, offsets_b=offsets_b, max_len_a=max_len_a, max_len_b=
        max_len_b, D=D, n_prefix_from_B=n_prefix_from_B, is_dense_a=
        is_dense_a, is_dense_b=is_dense_b, BLOCK_D=BLOCK_D)
    ctx.save_for_backward(offsets_a, offsets_b)
    ctx.max_seq_len = max_seq_len
    ctx.total_len_a = total_len_a
    ctx.total_len_b = total_len_b
    ctx.is_dense_a = is_dense_a
    ctx.is_dense_b = is_dense_b
    ctx.max_len_a = max_len_a
    ctx.max_len_b = max_len_b
    ctx.B = B
    ctx.n_prefix_from_B = n_prefix_from_B
    return values_out


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _split_2D_jagged(JaggedIn, OffsetsA, OffsetsB, MaxLenA, MaxLenB, OutA,
    OutB, D, stride_id, stride_ad, stride_bd, n_prefix_to_B, IS_DENSE_A: tl
    .constexpr, IS_DENSE_B: tl.constexpr, BLOCK_D: tl.constexpr):
    off_z = tl.program_id(1)
    off_n = tl.program_id(0)
    if IS_DENSE_A:
        seq_start_a = off_z * MaxLenA
        seq_len_a = MaxLenA
    else:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
    if IS_DENSE_B:
        seq_start_b = off_z * MaxLenB
        seq_len_b = MaxLenB
    else:
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_len_b = seq_end_b - seq_start_b
    seq_len = seq_len_a + seq_len_b
    if off_n >= seq_len:
        return
    seq_start = seq_start_a + seq_start_b
    offs_d = tl.arange(0, BLOCK_D)
    in_ptrs = JaggedIn + (seq_start + off_n).to(tl.int64) * stride_id + offs_d
    if off_n < n_prefix_to_B:
        out_ptrs = OutB + (off_n + seq_start_b).to(tl.int64
            ) * stride_bd + offs_d
    elif off_n < seq_len_a + n_prefix_to_B:
        out_ptrs = OutA + (off_n - n_prefix_to_B + seq_start_a).to(tl.int64
            ) * stride_ad + offs_d
    else:
        out_ptrs = OutB + (off_n - seq_len_a + seq_start_b).to(tl.int64
            ) * stride_bd + offs_d
    v = tl.load(in_ptrs, mask=offs_d < D)
    tl.store(out_ptrs, v, mask=offs_d < D)


# Backward method (kernel launch code)
def __Concat2DJaggedFunction_backward(ctx, d_out: torch.Tensor) ->Tuple[
    None, torch.Tensor, torch.Tensor, None, None, None, None, None]:
    offsets_a, offsets_b = ctx.saved_tensors
    _, D = d_out.shape
    BLOCK_D = triton.next_power_of_2(D)
    d_values_a = torch.zeros((ctx.total_len_a, D), device=d_out.device,
        dtype=d_out.dtype)
    d_values_b = torch.empty((ctx.total_len_b, D), device=d_out.device,
        dtype=d_out.dtype)
    _split_2D_jagged[ctx.max_seq_len, ctx.B](JaggedIn=d_out, OffsetsA=
        offsets_a, OffsetsB=offsets_b, MaxLenA=ctx.max_len_a, MaxLenB=ctx.
        max_len_b, OutA=d_values_a, OutB=d_values_b, D=D, stride_id=d_out.
        stride(-2), stride_ad=d_values_a.stride(-2), stride_bd=d_values_b.
        stride(-2), n_prefix_to_B=ctx.n_prefix_from_B, BLOCK_D=BLOCK_D,
        IS_DENSE_A=ctx.is_dense_a, IS_DENSE_B=ctx.is_dense_b)
    return None, d_values_a, d_values_b, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _Concat2DJaggedFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, max_seq_len: int, values_a: torch.Tensor, values_b:
        torch.Tensor, max_len_a: Optional[int], max_len_b: Optional[int],
        offsets_a: Optional[torch.Tensor], offsets_b: Optional[torch.Tensor
        ], n_prefix_from_B: int):
        values_a = switch_to_contiguous_if_needed(values_a)
        values_b = switch_to_contiguous_if_needed(values_b)
        is_dense_a = offsets_a is None
        is_dense_b = offsets_b is None
        total_len_a, D = values_a.shape
        total_len_b, _ = values_b.shape
        if is_dense_a:
            assert max_len_a is not None
            B = total_len_a // max_len_a
        else:
            assert offsets_a is not None
            B = offsets_a.shape[0] - 1
        if is_dense_b:
            assert max_len_b is not None
            B = total_len_b // max_len_b
        else:
            assert offsets_b is not None
            B = offsets_b.shape[0] - 1
        total_seq_len = total_len_a + total_len_b
        BLOCK_D = triton.next_power_of_2(D)
        values_out = torch.empty((total_seq_len, D), device=values_a.device,
            dtype=values_a.dtype)
        _triton_concat_2D_jagged_internal(values_a=values_a, values_b=
            values_b, values_out=values_out, max_seq_len=max_seq_len, B=B,
            offsets_a=offsets_a, offsets_b=offsets_b, max_len_a=max_len_a,
            max_len_b=max_len_b, D=D, n_prefix_from_B=n_prefix_from_B,
            is_dense_a=is_dense_a, is_dense_b=is_dense_b, BLOCK_D=BLOCK_D)
        ctx.save_for_backward(offsets_a, offsets_b)
        ctx.max_seq_len = max_seq_len
        ctx.total_len_a = total_len_a
        ctx.total_len_b = total_len_b
        ctx.is_dense_a = is_dense_a
        ctx.is_dense_b = is_dense_b
        ctx.max_len_a = max_len_a
        ctx.max_len_b = max_len_b
        ctx.B = B
        ctx.n_prefix_from_B = n_prefix_from_B
        return values_out

    @staticmethod
    def backward(ctx, d_out: torch.Tensor) ->Tuple[None, torch.Tensor,
        torch.Tensor, None, None, None, None, None]:
        offsets_a, offsets_b = ctx.saved_tensors
        _, D = d_out.shape
        BLOCK_D = triton.next_power_of_2(D)
        d_values_a = torch.zeros((ctx.total_len_a, D), device=d_out.device,
            dtype=d_out.dtype)
        d_values_b = torch.empty((ctx.total_len_b, D), device=d_out.device,
            dtype=d_out.dtype)
        _split_2D_jagged[ctx.max_seq_len, ctx.B](JaggedIn=d_out, OffsetsA=
            offsets_a, OffsetsB=offsets_b, MaxLenA=ctx.max_len_a, MaxLenB=
            ctx.max_len_b, OutA=d_values_a, OutB=d_values_b, D=D, stride_id
            =d_out.stride(-2), stride_ad=d_values_a.stride(-2), stride_bd=
            d_values_b.stride(-2), n_prefix_to_B=ctx.n_prefix_from_B,
            BLOCK_D=BLOCK_D, IS_DENSE_A=ctx.is_dense_a, IS_DENSE_B=ctx.
            is_dense_b)
        return None, d_values_a, d_values_b, None, None, None, None, None
