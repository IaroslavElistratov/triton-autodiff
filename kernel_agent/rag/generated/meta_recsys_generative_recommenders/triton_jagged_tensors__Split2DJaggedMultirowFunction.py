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
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

# Common helper imports
from triton import cdiv

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def _split_2D_jagged_multirow(JaggedIn, OffsetsA, OffsetsB, MaxLenA,
    MaxLenB, OutA, OutB, D, stride_id, stride_ad, stride_bd, n_prefix_to_B,
    IS_DENSE_A: tl.constexpr, IS_DENSE_B: tl.constexpr, BLOCK_D: tl.
    constexpr, BLOCK_N: tl.constexpr):
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
    seq_start = seq_start_a + seq_start_b
    start_n = block_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    valid_mask = offs_n < seq_len
    in_ptrs = JaggedIn + (seq_start + offs_n[:, None]).to(tl.int64
        ) * stride_id + offs_d[None, :]
    v = tl.load(in_ptrs, mask=valid_mask[:, None] & (offs_d[None, :] < D),
        other=0.0)
    to_prefix_b_mask = (offs_n < n_prefix_to_B) & valid_mask
    to_a_mask = (offs_n >= n_prefix_to_B) & (offs_n < seq_len_a + n_prefix_to_B
        ) & valid_mask
    to_suffix_b_mask = (offs_n >= seq_len_a + n_prefix_to_B) & valid_mask
    out_b1_ptrs = OutB + (offs_n[:, None] + seq_start_b).to(tl.int64
        ) * stride_bd + offs_d[None, :]
    tl.store(out_b1_ptrs, v, mask=to_prefix_b_mask[:, None] & (offs_d[None,
        :] < D))
    off_a = offs_n - n_prefix_to_B
    out_a_ptrs = OutA + (off_a[:, None] + seq_start_a).to(tl.int64
        ) * stride_ad + offs_d[None, :]
    tl.store(out_a_ptrs, v, mask=to_a_mask[:, None] & (offs_d[None, :] < D))
    off_b = offs_n - seq_len_a
    out_b2_ptrs = OutB + (off_b[:, None] + seq_start_b).to(tl.int64
        ) * stride_bd + offs_d[None, :]
    tl.store(out_b2_ptrs, v, mask=to_suffix_b_mask[:, None] & (offs_d[None,
        :] < D))


@triton_autotune(configs=_get_concat_split_2d_jagged_multirow_configs(),
    key=['BLOCK_D'])
@triton.jit
def split_2D_jagged_multirow(JaggedIn, OffsetsA, OffsetsB, MaxLenA, MaxLenB,
    OutA, OutB, D, stride_id, stride_ad, stride_bd, n_prefix_to_B,
    IS_DENSE_A: tl.constexpr, IS_DENSE_B: tl.constexpr, BLOCK_D: tl.
    constexpr, BLOCK_N: tl.constexpr):
    _split_2D_jagged_multirow(JaggedIn, OffsetsA, OffsetsB, MaxLenA,
        MaxLenB, OutA, OutB, D, stride_id, stride_ad, stride_bd,
        n_prefix_to_B, IS_DENSE_A, IS_DENSE_B, BLOCK_D, BLOCK_N)


def switch_to_contiguous_if_needed(x: torch.Tensor) ->torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10 ** 9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


# Forward method (kernel launch code)
def __Split2DJaggedMultirowFunction_forward(ctx, max_seq_len: int, values:
    torch.Tensor, total_len_left: Optional[int], total_len_right: Optional[
    int], max_len_left: Optional[int], max_len_right: Optional[int],
    offsets_left: Optional[torch.Tensor], offsets_right: Optional[torch.Tensor]
    ) ->Tuple[torch.Tensor, torch.Tensor]:
    values = switch_to_contiguous_if_needed(values)
    is_dense_left: bool = offsets_left is None
    is_dense_right: bool = offsets_right is None
    total_seq_len, D = values.shape
    if is_dense_left:
        assert is_dense_right is False
        assert offsets_right is not None
        assert max_len_left is not None
        B = offsets_right.shape[0] - 1
        total_len_a = max_len_left * B
        total_len_b = total_seq_len - total_len_a
    elif is_dense_right:
        assert is_dense_left is False
        assert offsets_left is not None
        assert max_len_right is not None
        B = offsets_left.shape[0] - 1
        total_len_b = max_len_right * B
        total_len_a = total_seq_len - total_len_b
    else:
        assert offsets_left is not None and offsets_right is not None
        B = offsets_left.shape[0] - 1
        if total_len_left is not None and total_len_right is not None:
            assert total_len_left + total_len_right == total_seq_len
            total_len_a = total_len_left
            total_len_b = total_len_right
        else:
            total_len_a = int(offsets_left[-1].item())
            total_len_b = values.size(0) - total_len_a
    BLOCK_D = triton.next_power_of_2(D)
    values_a = torch.empty((total_len_a, D), device=values.device, dtype=
        values.dtype)
    values_b = torch.empty((total_len_b, D), device=values.device, dtype=
        values.dtype)

    def grid(meta):
        return triton.cdiv(max_seq_len, meta['BLOCK_N']), B
    split_2D_jagged_multirow[grid](JaggedIn=values, OffsetsA=offsets_left,
        OffsetsB=offsets_right, MaxLenA=max_len_left, MaxLenB=max_len_right,
        OutA=values_a, OutB=values_b, D=D, stride_id=values.stride(-2),
        stride_ad=values_a.stride(-2), stride_bd=values_b.stride(-2),
        n_prefix_to_B=0, IS_DENSE_A=is_dense_left, IS_DENSE_B=
        is_dense_right, BLOCK_D=BLOCK_D)
    ctx.save_for_backward(offsets_left, offsets_right)
    ctx.max_seq_len = max_seq_len
    ctx.total_seq_len = total_seq_len
    ctx.max_len_left = max_len_left
    ctx.max_len_right = max_len_right
    ctx.is_dense_left = is_dense_left
    ctx.is_dense_right = is_dense_right
    ctx.B = B
    ctx.D = D
    return values_a, values_b


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

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


# Backward method (kernel launch code)
def __Split2DJaggedMultirowFunction_backward(ctx, *d_values) ->Tuple[None,
    torch.Tensor, None, None, None, None, None, None]:
    offsets_left, offsets_right = ctx.saved_tensors
    d_values_a, d_values_b = d_values
    BLOCK_D = triton.next_power_of_2(ctx.D)
    d_jagged_in = torch.empty((ctx.total_seq_len, ctx.D), device=d_values_a
        .device, dtype=d_values_a.dtype)

    def grid(meta):
        return triton.cdiv(ctx.max_seq_len, meta['BLOCK_N']), ctx.B
    concat_2D_jagged_multirow[grid](ValuesA=d_values_a, ValuesB=d_values_b,
        OffsetsA=offsets_left, OffsetsB=offsets_right, MaxLenA=ctx.
        max_len_left, MaxLenB=ctx.max_len_right, Out=d_jagged_in, D=ctx.D,
        stride_ad=d_values_a.stride(-2), stride_bd=d_values_b.stride(-2),
        stride_od=d_jagged_in.stride(-2), n_prefix_from_B=0, IS_DENSE_A=ctx
        .is_dense_left, IS_DENSE_B=ctx.is_dense_right, BLOCK_D=BLOCK_D)
    return None, d_jagged_in, None, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _Split2DJaggedMultirowFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, max_seq_len: int, values: torch.Tensor, total_len_left:
        Optional[int], total_len_right: Optional[int], max_len_left:
        Optional[int], max_len_right: Optional[int], offsets_left: Optional
        [torch.Tensor], offsets_right: Optional[torch.Tensor]) ->Tuple[
        torch.Tensor, torch.Tensor]:
        values = switch_to_contiguous_if_needed(values)
        is_dense_left: bool = offsets_left is None
        is_dense_right: bool = offsets_right is None
        total_seq_len, D = values.shape
        if is_dense_left:
            assert is_dense_right is False
            assert offsets_right is not None
            assert max_len_left is not None
            B = offsets_right.shape[0] - 1
            total_len_a = max_len_left * B
            total_len_b = total_seq_len - total_len_a
        elif is_dense_right:
            assert is_dense_left is False
            assert offsets_left is not None
            assert max_len_right is not None
            B = offsets_left.shape[0] - 1
            total_len_b = max_len_right * B
            total_len_a = total_seq_len - total_len_b
        else:
            assert offsets_left is not None and offsets_right is not None
            B = offsets_left.shape[0] - 1
            if total_len_left is not None and total_len_right is not None:
                assert total_len_left + total_len_right == total_seq_len
                total_len_a = total_len_left
                total_len_b = total_len_right
            else:
                total_len_a = int(offsets_left[-1].item())
                total_len_b = values.size(0) - total_len_a
        BLOCK_D = triton.next_power_of_2(D)
        values_a = torch.empty((total_len_a, D), device=values.device,
            dtype=values.dtype)
        values_b = torch.empty((total_len_b, D), device=values.device,
            dtype=values.dtype)

        def grid(meta):
            return triton.cdiv(max_seq_len, meta['BLOCK_N']), B
        split_2D_jagged_multirow[grid](JaggedIn=values, OffsetsA=
            offsets_left, OffsetsB=offsets_right, MaxLenA=max_len_left,
            MaxLenB=max_len_right, OutA=values_a, OutB=values_b, D=D,
            stride_id=values.stride(-2), stride_ad=values_a.stride(-2),
            stride_bd=values_b.stride(-2), n_prefix_to_B=0, IS_DENSE_A=
            is_dense_left, IS_DENSE_B=is_dense_right, BLOCK_D=BLOCK_D)
        ctx.save_for_backward(offsets_left, offsets_right)
        ctx.max_seq_len = max_seq_len
        ctx.total_seq_len = total_seq_len
        ctx.max_len_left = max_len_left
        ctx.max_len_right = max_len_right
        ctx.is_dense_left = is_dense_left
        ctx.is_dense_right = is_dense_right
        ctx.B = B
        ctx.D = D
        return values_a, values_b

    @staticmethod
    def backward(ctx, *d_values) ->Tuple[None, torch.Tensor, None, None,
        None, None, None, None]:
        offsets_left, offsets_right = ctx.saved_tensors
        d_values_a, d_values_b = d_values
        BLOCK_D = triton.next_power_of_2(ctx.D)
        d_jagged_in = torch.empty((ctx.total_seq_len, ctx.D), device=
            d_values_a.device, dtype=d_values_a.dtype)

        def grid(meta):
            return triton.cdiv(ctx.max_seq_len, meta['BLOCK_N']), ctx.B
        concat_2D_jagged_multirow[grid](ValuesA=d_values_a, ValuesB=
            d_values_b, OffsetsA=offsets_left, OffsetsB=offsets_right,
            MaxLenA=ctx.max_len_left, MaxLenB=ctx.max_len_right, Out=
            d_jagged_in, D=ctx.D, stride_ad=d_values_a.stride(-2),
            stride_bd=d_values_b.stride(-2), stride_od=d_jagged_in.stride(-
            2), n_prefix_from_B=0, IS_DENSE_A=ctx.is_dense_left, IS_DENSE_B
            =ctx.is_dense_right, BLOCK_D=BLOCK_D)
        return None, d_jagged_in, None, None, None, None, None, None
