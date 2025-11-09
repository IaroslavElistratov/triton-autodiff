# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/meta-recsys/generative-recommenders
# Source-Files: generative_recommenders/ops/triton/triton_position.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rzwbk0zd/generative-recommenders-main/generative_recommenders/ops/triton/triton_position.py
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
import time

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


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton_autotune(configs=_autotune_configs(), key=['AUTOTUNE_MAX_SEQ_LEN'])
@triton.jit
def _add_timestamp_position_embeddings_kernel(SeqEmb, Offsets, Lengths,
    PosEmb, TsEmb, Out, TS, PosInds, TsInds, NumTargets,
    AUTOTUNE_MAX_SEQ_LEN, D, num_time_buckets, time_bucket_increments,
    time_bucket_scale, time_delta, max_contextual_seq_len, max_pos_ind,
    stride_sn, stride_pn, stride_tn, stride_on, TRAINING: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr, INTERLEAVE_TARGETS: tl.constexpr,
    TIME_BUCKET_FN: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr
    ):
    """
    SeqEmb has shape (sum_B(N_i), D),
    PosEmb has shape (N_p, D),
    TsEmb has shape (N_t, D),
    Out has shape (sum_B(N_i), D)
    """
    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    seq_start = tl.load(Offsets + off_b)
    seq_end = tl.load(Offsets + off_b + 1)
    seq_len = seq_end - seq_start
    start_n = off_n * BLOCK_N
    if start_n >= seq_len:
        return
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    seq_emb_offsets = offs_n[:, None] * stride_sn + offs_d[None, :]
    SeqEmb += seq_start.to(tl.int64) * stride_sn
    mask_n = offs_n < seq_len
    seq_len = tl.load(Lengths + off_b)
    if HAS_MULTIPLE_TARGETS:
        num_targets = tl.load(NumTargets + off_b)
        if INTERLEAVE_TARGETS:
            high_ind = seq_len - num_targets * 2
        else:
            high_ind = seq_len - num_targets
    else:
        high_ind = seq_len
    pos_inds = tl.where(offs_n < high_ind, offs_n, high_ind)
    pos_inds = high_ind - pos_inds + max_contextual_seq_len
    pos_inds = tl.where(pos_inds < max_pos_ind - 1, pos_inds, max_pos_ind - 1)
    pos_inds = tl.where(offs_n < max_contextual_seq_len, offs_n, pos_inds)
    if TRAINING:
        tl.store(PosInds + seq_start + offs_n, pos_inds, mask=mask_n)
    pos_emb_offsets = pos_inds[:, None] * stride_pn + offs_d[None, :]
    ts = tl.load(TS + seq_start + offs_n, mask=mask_n)
    query_time = tl.load(TS + seq_end - 1)
    ts = query_time - ts + time_delta
    ts = tl.where(ts > 1e-06, ts, 1e-06) / time_bucket_increments
    if TIME_BUCKET_FN == 'log':
        ts = tl.log(ts)
    else:
        ts = tl.sqrt(ts)
    ts = ts * time_bucket_scale
    ts = ts.to(tl.int32)
    ts = tl.where(ts > 0, ts, 0)
    ts = tl.where(ts < num_time_buckets, ts, num_time_buckets)
    if TRAINING:
        tl.store(TsInds + seq_start + offs_n, ts, mask=mask_n)
    ts_emb_offsets = ts[:, None] * stride_tn + offs_d[None, :]
    Out += seq_start.to(tl.int64) * stride_on
    out_offsets = Out + offs_n[:, None] * stride_on + offs_d[None, :]
    for _d in range(0, D, BLOCK_D):
        mask = offs_n[:, None] < seq_len and offs_d[None, :] < D
        seq_emb = tl.load(SeqEmb + seq_emb_offsets, mask=mask)
        pos_emb = tl.load(PosEmb + pos_emb_offsets, mask=mask)
        ts_emb = tl.load(TsEmb + ts_emb_offsets, mask=mask)
        tl.store(out_offsets, seq_emb + (pos_emb + ts_emb).to(seq_emb.dtype
            ), mask=mask)
        seq_emb_offsets += BLOCK_D
        pos_emb_offsets += BLOCK_D
        ts_emb_offsets += BLOCK_D
        out_offsets += BLOCK_D
        offs_d += BLOCK_D


def switch_to_contiguous_if_needed(x: torch.Tensor) ->torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10 ** 9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


# Forward method (kernel launch code)
def __AddTimestampPositionEmbeddingsFunction_forward(ctx, seq_embeddings:
    torch.Tensor, seq_offsets: torch.Tensor, pos_embeddings: torch.Tensor,
    ts_embeddings: torch.Tensor, timestamps: torch.Tensor, max_seq_len: int,
    max_contextual_seq_len: int, seq_lengths: torch.Tensor, num_targets:
    Optional[torch.Tensor], interleave_targets: bool, time_bucket_fn: str):
    seq_embeddings = switch_to_contiguous_if_needed(seq_embeddings)
    pos_embeddings = switch_to_contiguous_if_needed(pos_embeddings)
    ts_embeddings = switch_to_contiguous_if_needed(ts_embeddings)
    max_pos_ind = pos_embeddings.shape[0]
    B = seq_lengths.shape[0]
    N, D = seq_embeddings.shape
    assert len(pos_embeddings.shape) == 2
    assert len(ts_embeddings.shape) == 2
    assert pos_embeddings.shape[1
        ] == D, 'shape[1] of pos_embeddings much match seq_embeddings'
    assert ts_embeddings.shape[1
        ] == D, 'shape[1] of ts_embeddings much match seq_embeddings'
    out = torch.empty_like(seq_embeddings)
    timestamps = switch_to_contiguous_if_needed(timestamps)
    ts_inds = torch.empty_like(seq_embeddings[:, 0], dtype=torch.int32)
    pos_inds = torch.empty_like(seq_embeddings[:, 0], dtype=torch.int32)
    ts_emb_size = ts_embeddings.shape[0]
    grid = lambda meta: (B, triton.cdiv(max_seq_len, meta['BLOCK_N']))
    BLOCK_D = triton.next_power_of_2(D) if D < 64 else 64
    _add_timestamp_position_embeddings_kernel[grid](SeqEmb=seq_embeddings,
        Offsets=seq_offsets, Lengths=seq_lengths, PosEmb=pos_embeddings,
        TsEmb=ts_embeddings, Out=out, TS=timestamps, PosInds=pos_inds,
        TsInds=ts_inds, NumTargets=num_targets, AUTOTUNE_MAX_SEQ_LEN=
        autotune_max_seq_len(max_seq_len), D=D, num_time_buckets=
        ts_emb_size - 1, time_bucket_increments=60.0, time_bucket_scale=1.0,
        time_delta=0, max_contextual_seq_len=max_contextual_seq_len,
        max_pos_ind=max_pos_ind, stride_sn=seq_embeddings.stride(0),
        stride_pn=pos_embeddings.stride(0), stride_tn=ts_embeddings.stride(
        0), stride_on=out.stride(0), TRAINING=True, HAS_MULTIPLE_TARGETS=
        num_targets is not None, INTERLEAVE_TARGETS=interleave_targets,
        TIME_BUCKET_FN=time_bucket_fn, BLOCK_D=BLOCK_D)
    try:
        values = torch.arange(0, N, dtype=torch.int32, device=timestamps.device
            )
        sorted_ts_key_inds, sorted_ts_value_inds = (torch.ops.hammer.
            sort_kv_pairs(ts_inds, values))
        sorted_pos_key_inds, sorted_pos_value_inds = (torch.ops.hammer.
            sort_kv_pairs(pos_inds, values))
    except Exception:
        sorted_ts_key_inds, sorted_ts_value_inds = torch.sort(ts_inds)
        sorted_pos_key_inds, sorted_pos_value_inds = torch.sort(pos_inds)
    ctx.save_for_backward(sorted_pos_key_inds, sorted_pos_value_inds,
        sorted_ts_key_inds, sorted_ts_value_inds)
    ctx.B = B
    ctx.D = D
    ctx.max_seq_len = max_seq_len
    ctx.pos_emb_size = pos_embeddings.shape[0]
    ctx.ts_emb_size = ts_emb_size
    ctx.pos_dtype = pos_embeddings.dtype
    ctx.ts_dtype = ts_embeddings.dtype
    return out


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton_autotune(configs=_add_embeddings_bwd_configs(), key=[
    'AUTOTUNE_MAX_SEQ_LEN', 'AUTOTUNE_B', 'D'])
@triton.jit
def _add_embeddings_bwd_kernel(In, KeyInds, ValueInds, Out,
    AUTOTUNE_MAX_SEQ_LEN, AUTOTUNE_B, D, jagged_size, stride_in, stride_on,
    BLOCK_D: tl.constexpr, BLOCK: tl.constexpr):
    off_block = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    key_ind = -1
    key_ind = key_ind.to(KeyInds.dtype.element_ty)
    accumulator = tl.zeros((BLOCK_D,), dtype=In.dtype.element_ty)
    for off_i in range(0, BLOCK):
        off = off_block * BLOCK + off_i
        if off < jagged_size:
            value_ind = tl.load(ValueInds + off)
            in_offset = In + value_ind.to(tl.int64) * stride_in
            jagged_in = tl.load(in_offset + offs_d, mask=mask_d)
            key_ind_new = tl.load(KeyInds + off)
            if key_ind == key_ind_new:
                accumulator += jagged_in
            else:
                if key_ind >= 0:
                    out_offset = Out + key_ind.to(tl.int64) * stride_on
                    tl.atomic_add(out_offset + offs_d, accumulator.to(Out.
                        dtype.element_ty), mask=mask_d, sem='relaxed')
                key_ind = key_ind_new
                accumulator = jagged_in
    if key_ind >= 0:
        out_offset = Out + key_ind.to(tl.int64) * stride_on
        tl.atomic_add(out_offset + offs_d, accumulator.to(Out.dtype.
            element_ty), mask=mask_d, sem='relaxed')


# Backward method (kernel launch code)
def __AddTimestampPositionEmbeddingsFunction_backward(ctx, d_out: torch.Tensor
    ) ->Tuple[torch.Tensor, None, torch.Tensor, torch.Tensor, None, None,
    None, None, None, None, None]:
    (sorted_pos_key_inds, sorted_pos_value_inds, sorted_ts_key_inds,
        sorted_ts_value_inds) = ctx.saved_tensors
    d_pos_embeddings = torch.empty((ctx.pos_emb_size, ctx.D), device=d_out.
        device, dtype=torch.float32)
    d_ts_embeddings = torch.empty((ctx.ts_emb_size, ctx.D), device=d_out.
        device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(d_out.shape[0], meta['BLOCK']),)
    AUTOTUNE_B = prev_power_of_2(ctx.B)
    _add_embeddings_bwd_kernel[grid](In=d_out, KeyInds=sorted_pos_key_inds,
        ValueInds=sorted_pos_value_inds, Out=d_pos_embeddings,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(ctx.max_seq_len),
        AUTOTUNE_B=AUTOTUNE_B, D=ctx.D, jagged_size=d_out.shape[0],
        stride_in=d_out.stride(0), stride_on=d_pos_embeddings.stride(0),
        BLOCK_D=triton.next_power_of_2(ctx.D))
    _add_embeddings_bwd_kernel[grid](In=d_out, KeyInds=sorted_ts_key_inds,
        ValueInds=sorted_ts_value_inds, Out=d_ts_embeddings,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(ctx.max_seq_len),
        AUTOTUNE_B=AUTOTUNE_B, D=ctx.D, jagged_size=d_out.shape[0],
        stride_in=d_out.stride(0), stride_on=d_ts_embeddings.stride(0),
        BLOCK_D=triton.next_power_of_2(ctx.D))
    return d_out, None, d_pos_embeddings.to(ctx.pos_dtype), d_ts_embeddings.to(
        ctx.ts_dtype), None, None, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _AddTimestampPositionEmbeddingsFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, seq_embeddings: torch.Tensor, seq_offsets: torch.
        Tensor, pos_embeddings: torch.Tensor, ts_embeddings: torch.Tensor,
        timestamps: torch.Tensor, max_seq_len: int, max_contextual_seq_len:
        int, seq_lengths: torch.Tensor, num_targets: Optional[torch.Tensor],
        interleave_targets: bool, time_bucket_fn: str):
        seq_embeddings = switch_to_contiguous_if_needed(seq_embeddings)
        pos_embeddings = switch_to_contiguous_if_needed(pos_embeddings)
        ts_embeddings = switch_to_contiguous_if_needed(ts_embeddings)
        max_pos_ind = pos_embeddings.shape[0]
        B = seq_lengths.shape[0]
        N, D = seq_embeddings.shape
        assert len(pos_embeddings.shape) == 2
        assert len(ts_embeddings.shape) == 2
        assert pos_embeddings.shape[1
            ] == D, 'shape[1] of pos_embeddings much match seq_embeddings'
        assert ts_embeddings.shape[1
            ] == D, 'shape[1] of ts_embeddings much match seq_embeddings'
        out = torch.empty_like(seq_embeddings)
        timestamps = switch_to_contiguous_if_needed(timestamps)
        ts_inds = torch.empty_like(seq_embeddings[:, 0], dtype=torch.int32)
        pos_inds = torch.empty_like(seq_embeddings[:, 0], dtype=torch.int32)
        ts_emb_size = ts_embeddings.shape[0]
        grid = lambda meta: (B, triton.cdiv(max_seq_len, meta['BLOCK_N']))
        BLOCK_D = triton.next_power_of_2(D) if D < 64 else 64
        _add_timestamp_position_embeddings_kernel[grid](SeqEmb=
            seq_embeddings, Offsets=seq_offsets, Lengths=seq_lengths,
            PosEmb=pos_embeddings, TsEmb=ts_embeddings, Out=out, TS=
            timestamps, PosInds=pos_inds, TsInds=ts_inds, NumTargets=
            num_targets, AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(
            max_seq_len), D=D, num_time_buckets=ts_emb_size - 1,
            time_bucket_increments=60.0, time_bucket_scale=1.0, time_delta=
            0, max_contextual_seq_len=max_contextual_seq_len, max_pos_ind=
            max_pos_ind, stride_sn=seq_embeddings.stride(0), stride_pn=
            pos_embeddings.stride(0), stride_tn=ts_embeddings.stride(0),
            stride_on=out.stride(0), TRAINING=True, HAS_MULTIPLE_TARGETS=
            num_targets is not None, INTERLEAVE_TARGETS=interleave_targets,
            TIME_BUCKET_FN=time_bucket_fn, BLOCK_D=BLOCK_D)
        try:
            values = torch.arange(0, N, dtype=torch.int32, device=
                timestamps.device)
            sorted_ts_key_inds, sorted_ts_value_inds = (torch.ops.hammer.
                sort_kv_pairs(ts_inds, values))
            sorted_pos_key_inds, sorted_pos_value_inds = (torch.ops.hammer.
                sort_kv_pairs(pos_inds, values))
        except Exception:
            sorted_ts_key_inds, sorted_ts_value_inds = torch.sort(ts_inds)
            sorted_pos_key_inds, sorted_pos_value_inds = torch.sort(pos_inds)
        ctx.save_for_backward(sorted_pos_key_inds, sorted_pos_value_inds,
            sorted_ts_key_inds, sorted_ts_value_inds)
        ctx.B = B
        ctx.D = D
        ctx.max_seq_len = max_seq_len
        ctx.pos_emb_size = pos_embeddings.shape[0]
        ctx.ts_emb_size = ts_emb_size
        ctx.pos_dtype = pos_embeddings.dtype
        ctx.ts_dtype = ts_embeddings.dtype
        return out

    @staticmethod
    def backward(ctx, d_out: torch.Tensor) ->Tuple[torch.Tensor, None,
        torch.Tensor, torch.Tensor, None, None, None, None, None, None, None]:
        (sorted_pos_key_inds, sorted_pos_value_inds, sorted_ts_key_inds,
            sorted_ts_value_inds) = ctx.saved_tensors
        d_pos_embeddings = torch.empty((ctx.pos_emb_size, ctx.D), device=
            d_out.device, dtype=torch.float32)
        d_ts_embeddings = torch.empty((ctx.ts_emb_size, ctx.D), device=
            d_out.device, dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(d_out.shape[0], meta['BLOCK']),)
        AUTOTUNE_B = prev_power_of_2(ctx.B)
        _add_embeddings_bwd_kernel[grid](In=d_out, KeyInds=
            sorted_pos_key_inds, ValueInds=sorted_pos_value_inds, Out=
            d_pos_embeddings, AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(ctx
            .max_seq_len), AUTOTUNE_B=AUTOTUNE_B, D=ctx.D, jagged_size=
            d_out.shape[0], stride_in=d_out.stride(0), stride_on=
            d_pos_embeddings.stride(0), BLOCK_D=triton.next_power_of_2(ctx.D))
        _add_embeddings_bwd_kernel[grid](In=d_out, KeyInds=
            sorted_ts_key_inds, ValueInds=sorted_ts_value_inds, Out=
            d_ts_embeddings, AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(ctx.
            max_seq_len), AUTOTUNE_B=AUTOTUNE_B, D=ctx.D, jagged_size=d_out
            .shape[0], stride_in=d_out.stride(0), stride_on=d_ts_embeddings
            .stride(0), BLOCK_D=triton.next_power_of_2(ctx.D))
        return d_out, None, d_pos_embeddings.to(ctx.pos_dtype
            ), d_ts_embeddings.to(ctx.ts_dtype
            ), None, None, None, None, None, None, None
