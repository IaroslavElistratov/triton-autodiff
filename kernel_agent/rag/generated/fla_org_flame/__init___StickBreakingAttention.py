# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/fla-org/flame
# Source-Files: custom_models/sba/stickbreaking_attention/sb_attn/__init__.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_n4iy29j4/flame-main/custom_models/sba/stickbreaking_attention/sb_attn/__init__.py
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
from math import log

@triton.jit
def compute_block(q, k, qk_scale, neg_log_acc, M_blk_idxs, N_blk_idxs, cm,
    on_band: tl.constexpr, ALLOW_TF32: tl.constexpr, backward: tl.constexpr,
    attend_current: tl.constexpr=False, use_cumsum: tl.constexpr=False,
    is_compiling: tl.constexpr=False):
    qk = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * qk_scale
    log_om_beta = -softplus(qk, is_compiling=is_compiling)
    if on_band:
        if attend_current:
            block_mask = M_blk_idxs[:, None] >= N_blk_idxs[None, :]
        else:
            block_mask = M_blk_idxs[:, None] > N_blk_idxs[None, :]
        log_om_beta = tl.where(block_mask, log_om_beta, 0.0)
        if backward:
            neg_log_acc -= tl.sum(log_om_beta, axis=1)
        log_p = qk + neg_log_acc[:, None]
        if use_cumsum:
            log_p += tl.cumsum(log_om_beta.to(q.dtype), axis=1, reverse=True)
        else:
            log_p = tl.dot(log_om_beta.to(q.dtype), cm, acc=log_p,
                allow_tf32=ALLOW_TF32)
        p = tl.math.exp2(log_p)
        p = tl.where(block_mask, p, 0.0)
    else:
        if backward:
            neg_log_acc -= tl.sum(log_om_beta, axis=1)
        log_p = qk + neg_log_acc[:, None]
        if use_cumsum:
            log_p += tl.cumsum(log_om_beta.to(q.dtype), axis=1, reverse=True)
        else:
            log_p = tl.dot(log_om_beta.to(q.dtype), cm, acc=log_p,
                allow_tf32=ALLOW_TF32)
        p = tl.math.exp2(log_p)
    if not backward:
        neg_log_acc += tl.sum(log_om_beta, axis=1)
    return p, log_om_beta, neg_log_acc


@triton.jit
def load_kv(K_blk_ptrs, V_blk_ptrs, N_mask, NO_N_MASK, D_mask, NO_D_MASK:
    tl.constexpr):
    if NO_D_MASK:
        if NO_N_MASK:
            k = tl.load(K_blk_ptrs)
            v = tl.load(V_blk_ptrs)
        else:
            k = tl.load(K_blk_ptrs, mask=N_mask[:, None])
            v = tl.load(V_blk_ptrs, mask=N_mask[:, None])
    else:
        mask = N_mask[:, None] & D_mask[None, :]
        k = tl.load(K_blk_ptrs, mask=mask)
        v = tl.load(V_blk_ptrs, mask=mask)
    return k, v


@triton.jit
def softplus(x, is_compiling: tl.constexpr=False):
    if is_compiling:
        tl.static_print('Using triton softplus.')
        out = tl.where(x < 15.0, tl.math.log2(1 + tl.math.exp2(x)), x)
        return out
    else:
        out = tl.inline_asm_elementwise(asm=asm_str, constraints=
            constraints_str, pack=NUM_REG, args=[x], dtype=tl.float32,
            is_pure=True)
        return out


def _dispatch(func: Callable, compileable_fn: Callable, *args, **kwargs):
    if torch.compiler.is_compiling():
        output = compileable_fn(*args, **kwargs)
    else:
        output = func(*args, **kwargs)
    return output


def custom_op(name: str=None, mutates_args: (str | Iterable[str])=None,
    device_types: (str | Sequence[str] | None)=None, schema: (str | None)=None
    ) ->Callable:
    compileable_name = f'{PACKAGE_NAME}::{name}'

    def _inner(func: Callable):
        compileable_func = torch.library.custom_op(compileable_name, func,
            mutates_args=mutates_args, device_types=device_types, schema=schema
            )

        def _run(*args, **kwargs):
            return _dispatch(func, compileable_func, *args, **kwargs)
        return _run
    return _inner


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=get_configs(), key=['token_size', 'head_size'])
@triton.jit
def _forward(Q_ptr, stride_qb, stride_qh, stride_qm: tl.constexpr,
    stride_qd: tl.constexpr, K_ptr, stride_kb, stride_kh, stride_kn: tl.
    constexpr, stride_kd: tl.constexpr, V_ptr, stride_vb, stride_vh,
    stride_vn: tl.constexpr, stride_vd: tl.constexpr, O_ptr, stride_ob,
    stride_oh, stride_om: tl.constexpr, stride_od: tl.constexpr, R_ptr,
    stride_rb, stride_rh, stride_rm: tl.constexpr, A_ptr, stride_ab,
    stride_ah, stride_am: tl.constexpr, W_ptr, stride_wb, stride_wh,
    stride_wm, stride_wn, logit_scale: tl.constexpr, attend_current: tl.
    constexpr, batch_size, token_size, head_size: tl.constexpr, num_heads:
    tl.constexpr, BLOCK_D: tl.constexpr, NO_D_MASK: tl.constexpr, NO_M_MASK:
    tl.constexpr, NO_N_MASK: tl.constexpr, ALLOW_TF32: tl.constexpr,
    inv_log2: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    no_grad: tl.constexpr=False, acc_dtype: tl.constexpr=tl.float32,
    return_attention: tl.constexpr=False, is_compiling: tl.constexpr=False):
    tl.static_assert(BLOCK_M % BLOCK_N == 0)
    batch_id = tl.program_id(0)
    head_pid = tl.program_id(1)
    prog_id = tl.program_id(2)
    tl.num_programs(2)
    seq_length = token_size
    qk_scale = inv_log2 * logit_scale
    M_range = tl.arange(0, BLOCK_M)
    N_range = tl.arange(0, BLOCK_N)
    D_range = tl.arange(0, BLOCK_D)
    D_mask = D_range < head_size
    cm = tl.where(N_range[:, None] >= N_range[None, :], 1.0, 0.0).to(Q_ptr.
        type.element_ty)
    head_id = head_pid
    seq_prog_id = prog_id
    Q_head_seq_ptr = Q_ptr + stride_qb * batch_id + stride_qh * head_id
    K_head_seq_ptr = K_ptr + stride_kb * batch_id + stride_kh * head_id
    V_head_seq_ptr = V_ptr + stride_vb * batch_id + stride_vh * head_id
    O_head_seq_ptr = O_ptr + stride_ob * batch_id + stride_oh * head_id
    R_head_seq_ptr = R_ptr + stride_rb * batch_id + stride_rh * head_id
    A_head_seq_ptr = A_ptr + stride_ab * batch_id + stride_ah * head_id
    W_head_seq_ptr = W_ptr + stride_wb * batch_id + stride_wh * head_id
    _forward_one_row(seq_prog_id, seq_length, qk_scale, M_range, N_range,
        D_range, D_mask, cm, Q_head_seq_ptr, stride_qm, stride_qd,
        K_head_seq_ptr, stride_kn, stride_kd, V_head_seq_ptr, stride_vn,
        stride_vd, O_head_seq_ptr, stride_om, stride_od, R_head_seq_ptr,
        stride_rm, A_head_seq_ptr, stride_am, W_head_seq_ptr, stride_wm,
        stride_wn, BLOCK_D, NO_D_MASK, NO_M_MASK, NO_N_MASK, ALLOW_TF32,
        BLOCK_M, BLOCK_N, no_grad, acc_dtype, return_attention,
        attend_current=attend_current, is_compiling=is_compiling)


@triton.jit
def _forward_one_row(seq_block_id, seq_length, qk_scale, M_range, N_range,
    D_range, D_mask, cm, Q_head_seq_ptr, stride_qm, stride_qd: tl.constexpr,
    K_head_seq_ptr, stride_kn, stride_kd: tl.constexpr, V_head_seq_ptr,
    stride_vn, stride_vd: tl.constexpr, O_head_seq_ptr, stride_om,
    stride_od: tl.constexpr, R_head_seq_ptr, stride_rm, A_head_seq_ptr,
    stride_am, W_head_seq_ptr, stride_wm, stride_wn, BLOCK_D: tl.constexpr,
    NO_D_MASK: tl.constexpr, NO_M_MASK: tl.constexpr, NO_N_MASK: tl.
    constexpr, ALLOW_TF32: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl
    .constexpr, no_grad: tl.constexpr=False, acc_dtype: tl.constexpr=tl.
    float32, return_attention: tl.constexpr=False, is_compiling: tl.
    constexpr=False, use_cumsum: tl.constexpr=False, attend_current: tl.
    constexpr=False):
    block_start_offset = BLOCK_M * seq_block_id
    M_blk_idxs = block_start_offset + M_range
    M_mask = M_blk_idxs < seq_length
    NO_M_MASK = block_start_offset + BLOCK_M - 1 < seq_length
    N_blk_idxs_start = block_start_offset + BLOCK_M
    N_blk_idxs = N_blk_idxs_start + N_range
    Q_blk_ptrs = Q_head_seq_ptr + (stride_qm * M_blk_idxs[:, None] + 
        stride_qd * D_range[None, :])
    K_blk_ptrs = K_head_seq_ptr + (stride_kn * N_blk_idxs[:, None] + 
        stride_kd * D_range[None, :])
    V_blk_ptrs = V_head_seq_ptr + (stride_vn * N_blk_idxs[:, None] + 
        stride_vd * D_range[None, :])
    O_blk_ptrs = O_head_seq_ptr + (stride_om * M_blk_idxs[:, None] + 
        stride_od * D_range[None, :])
    R_blk_ptrs = R_head_seq_ptr + stride_rm * M_blk_idxs
    A_blk_ptrs = A_head_seq_ptr + stride_am * M_blk_idxs
    if NO_D_MASK:
        if NO_M_MASK:
            q = tl.load(Q_blk_ptrs)
        else:
            q = tl.load(Q_blk_ptrs, mask=M_mask[:, None], other=0.0)
    else:
        q = tl.load(Q_blk_ptrs, mask=M_mask[:, None] & D_mask[None, :],
            other=0.0)
    iters = N_blk_idxs_start // BLOCK_N
    neg_log_acc = tl.zeros([BLOCK_M], dtype=acc_dtype)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=acc_dtype)
    for i in range(iters):
        N_blk_idxs -= BLOCK_N
        N_blk_idxs_start -= BLOCK_N
        K_blk_ptrs -= BLOCK_N * stride_kn
        V_blk_ptrs -= BLOCK_N * stride_vn
        N_mask = N_blk_idxs < seq_length
        k, v = load_kv(K_blk_ptrs, V_blk_ptrs, N_mask=N_mask, NO_N_MASK=
            N_blk_idxs_start + BLOCK_N - 1 < seq_length, D_mask=D_mask,
            NO_D_MASK=NO_D_MASK)
        on_band = i < BLOCK_M // BLOCK_N
        p, _, neg_log_acc = compute_block(q, k, qk_scale, neg_log_acc,
            M_blk_idxs, N_blk_idxs, cm, on_band, ALLOW_TF32, attend_current
            =attend_current, backward=False, is_compiling=is_compiling,
            use_cumsum=use_cumsum)
        acc = tl.dot(p.to(v.dtype), v, acc, allow_tf32=ALLOW_TF32)
        if return_attention:
            tl.store(W_head_seq_ptr + stride_wm * M_blk_idxs[:, None] + 
                stride_wn * N_blk_idxs[None, :], p, mask=(M_blk_idxs <
                seq_length)[:, None] & (N_blk_idxs < seq_length)[None, :])
    if NO_M_MASK:
        tl.store(R_blk_ptrs, tl.math.exp2(neg_log_acc))
        tl.store(A_blk_ptrs, neg_log_acc.to(A_head_seq_ptr.type.element_ty))
    else:
        tl.store(R_blk_ptrs, tl.math.exp2(neg_log_acc), mask=M_mask)
        tl.store(A_blk_ptrs, neg_log_acc.to(A_head_seq_ptr.type.element_ty),
            mask=M_mask)
    if NO_D_MASK:
        tl.store(O_blk_ptrs, acc.to(O_head_seq_ptr.type.element_ty), mask=
            M_mask[:, None])
    else:
        tl.store(O_blk_ptrs, acc.to(O_head_seq_ptr.type.element_ty), mask=
            M_mask[:, None] & D_mask[None, :])


@custom_op('attn_fwd', mutates_args={'o', 'rem', 'neg_log_acc', 'W'})
def _compileable_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    logit_scale: float, no_grad: bool, return_attention: bool, BLOCK_M: int,
    BLOCK_N: int, batch_size: int, num_heads: int, token_size: int,
    dim_size: int, o: torch.Tensor, rem: torch.Tensor, neg_log_acc: torch.
    Tensor, W: torch.Tensor, attend_current: bool) ->None:
    num_folded_heads = num_heads
    num_seq_blocks = triton.cdiv(token_size, BLOCK_M)
    BLOCK_D = triton.next_power_of_2(dim_size)
    grid = batch_size, num_folded_heads, num_seq_blocks
    _forward[grid](q, q.stride(0), q.stride(1), q.stride(2), q.stride(3), k,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3), v, v.stride(0),
        v.stride(1), v.stride(2), v.stride(3), o, o.stride(0), o.stride(1),
        o.stride(2), o.stride(3), rem, rem.stride(0), rem.stride(1), rem.
        stride(2), neg_log_acc, neg_log_acc.stride(0), neg_log_acc.stride(1
        ), neg_log_acc.stride(2), W, W.stride(0), W.stride(1), W.stride(2),
        W.stride(3), logit_scale=logit_scale, batch_size=batch_size,
        token_size=token_size, head_size=dim_size, num_heads=num_heads,
        no_grad=no_grad, attend_current=attend_current, BLOCK_D=BLOCK_D,
        NO_D_MASK=BLOCK_D == dim_size, NO_M_MASK=token_size % BLOCK_M == 0,
        NO_N_MASK=token_size % BLOCK_N == 0, BLOCK_M=BLOCK_M, BLOCK_N=
        BLOCK_N, ALLOW_TF32=ALLOW_TF32, inv_log2=inv_log2, return_attention
        =return_attention, acc_dtype=tl.float32, is_compiling=False)


def _fwd(q, k, v, logit_scale, attend_current=False, no_grad=False,
    return_attention=False, BLOCK_M: int=64, BLOCK_N: int=32):
    batch_size, num_heads, token_size, dim_size = q.size()
    o = torch.empty_like(q)
    rem = torch.zeros_like(q[:, :, :, 0], device=q.device)
    neg_log_acc = torch.zeros_like(rem, device=q.device, dtype=torch.float32)
    if return_attention:
        W = torch.full((batch_size, num_heads, token_size, token_size), 0.0,
            dtype=torch.float32, device=q.device)
    else:
        W = torch.empty((1, 1, 1, 1), device=q.device)
    _compileable_fwd(q, k, v, logit_scale, no_grad, return_attention,
        BLOCK_M, BLOCK_N, batch_size, num_heads, token_size, dim_size, o,
        rem, neg_log_acc, W, attend_current=attend_current)
    if return_attention:
        return o, rem, neg_log_acc, W
    else:
        return o, rem, neg_log_acc


# Forward method (kernel launch code)
def _StickBreakingAttention_forward(ctx, q: torch.Tensor, k: torch.Tensor,
    v: torch.Tensor, inv_temp: float, attend_current: bool=False):
    no_grad = not ctx.needs_input_grad[0]
    logit_scale = inv_temp
    BLOCK_M = FWD_BLOCK_M
    BLOCK_N = FWD_BLOCK_N
    o, rem, neg_log_acc = _fwd(q, k, v, logit_scale=inv_temp, no_grad=
        no_grad, return_attention=False, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        attend_current=attend_current)
    ctx.save_for_backward(q, k, v, neg_log_acc)
    ctx.logit_scale = logit_scale
    ctx.attend_current = attend_current
    return o, rem


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=get_configs(), key=['token_size', 'head_size'])
@triton.jit()
def _backward(DO_ptr, stride_dob, stride_doh, stride_dom: tl.constexpr,
    stride_dod: tl.constexpr, DR_ptr, stride_drb, stride_drh, stride_drm:
    tl.constexpr, A_ptr, stride_ab, stride_ah, stride_am: tl.constexpr,
    Q_ptr, stride_qb, stride_qh, stride_qm: tl.constexpr, stride_qd: tl.
    constexpr, K_ptr, stride_kb, stride_kh, stride_kn: tl.constexpr,
    stride_kd: tl.constexpr, V_ptr, stride_vb, stride_vh, stride_vn: tl.
    constexpr, stride_vd: tl.constexpr, DQ_ptr, stride_dqb, stride_dqh,
    stride_dqm: tl.constexpr, stride_dqd: tl.constexpr, DK_ptr, stride_dkb,
    stride_dkh, stride_dkn: tl.constexpr, stride_dkd: tl.constexpr, DV_ptr,
    stride_dvb, stride_dvh, stride_dvn: tl.constexpr, stride_dvd: tl.
    constexpr, KV_Lock_ptr, KV_Count_ptr, stride_kvb: tl.constexpr,
    stride_kvl: tl.constexpr, logit_scale, batch_size, token_size,
    head_size: tl.constexpr, num_heads: tl.constexpr, BLOCK_D: tl.constexpr,
    NO_D_MASK: tl.constexpr, NO_M_MASK: tl.constexpr, NO_N_MASK: tl.
    constexpr, ALLOW_TF32: tl.constexpr, inv_log2: tl.constexpr, BLOCK_M:
    tl.constexpr, BLOCK_N: tl.constexpr, acc_dtype: tl.constexpr=tl.float32,
    is_compiling: tl.constexpr=False, attend_current: tl.constexpr=False):
    tl.static_assert(BLOCK_M % BLOCK_N == 0)
    batch_id = tl.program_id(0)
    head_pid = tl.program_id(1)
    prog_id = tl.program_id(2)
    qk_scale = inv_log2 * logit_scale
    M_range = tl.arange(0, BLOCK_M)
    N_range = tl.arange(0, BLOCK_N)
    D_range = tl.arange(0, BLOCK_D)
    D_mask = D_range < head_size
    cm = tl.where(N_range[:, None] >= N_range[None, :], 1.0, 0.0).to(Q_ptr.
        type.element_ty)
    head_id = head_pid
    seq_prog_id = prog_id
    seq_length = token_size
    DO_head_seq_ptr = DO_ptr + stride_dob * batch_id + stride_doh * head_id
    DR_head_seq_ptr = DR_ptr + stride_drb * batch_id + stride_drh * head_id
    A_head_seq_ptr = A_ptr + stride_ab * batch_id + stride_ah * head_id
    Q_head_seq_ptr = Q_ptr + stride_qb * batch_id + stride_qh * head_id
    K_head_seq_ptr = K_ptr + stride_kb * batch_id + stride_kh * head_id
    V_head_seq_ptr = V_ptr + stride_vb * batch_id + stride_vh * head_id
    DQ_head_seq_ptr = DQ_ptr + stride_dqb * batch_id + stride_dqh * head_id
    DK_head_seq_ptr = DK_ptr + stride_dkb * batch_id + stride_dkh * head_id
    DV_head_seq_ptr = DV_ptr + stride_dvb * batch_id + stride_dvh * head_id
    KV_Lock_head_seq_ptr = (KV_Lock_ptr + stride_kvb * batch_id + 
        stride_kvl * head_id)
    KV_Count_head_seq_ptr = (KV_Count_ptr + stride_kvb * batch_id + 
        stride_kvl * head_id)
    _backward_one_row(seq_prog_id, seq_length, qk_scale, M_range, N_range,
        D_range, D_mask, cm, DO_head_seq_ptr, stride_dom, stride_dod,
        DR_head_seq_ptr, stride_drm, A_head_seq_ptr, stride_am,
        Q_head_seq_ptr, stride_qm, stride_qd, K_head_seq_ptr, stride_kn,
        stride_kd, V_head_seq_ptr, stride_vn, stride_vd, DQ_head_seq_ptr,
        stride_dqm, stride_dqd, DK_head_seq_ptr, stride_dkn, stride_dkd,
        DV_head_seq_ptr, stride_dvn, stride_dvd, KV_Lock_head_seq_ptr,
        KV_Count_head_seq_ptr, logit_scale, BLOCK_D, NO_D_MASK, NO_M_MASK,
        ALLOW_TF32, BLOCK_M, BLOCK_N, acc_dtype, is_compiling=is_compiling,
        attend_current=attend_current)


@triton.jit
def _backward_one_row(seq_prog_id, seq_length, qk_scale, M_range, N_range,
    D_range, D_mask, cm, DO_head_seq_ptr, stride_dom, stride_dod: tl.
    constexpr, DR_head_seq_ptr, stride_drm, A_head_seq_ptr, stride_am: tl.
    constexpr, Q_head_seq_ptr, stride_qm, stride_qd: tl.constexpr,
    K_head_seq_ptr, stride_kn, stride_kd: tl.constexpr, V_head_seq_ptr,
    stride_vn, stride_vd: tl.constexpr, DQ_head_seq_ptr, stride_dqm,
    stride_dqd: tl.constexpr, DK_head_seq_ptr, stride_dkn, stride_dkd: tl.
    constexpr, DV_head_seq_ptr, stride_dvn, stride_dvd: tl.constexpr,
    KV_Lock_ptr, KV_Count_ptr, logit_scale, BLOCK_D: tl.constexpr,
    NO_D_MASK: tl.constexpr, NO_M_MASK: tl.constexpr, ALLOW_TF32: tl.
    constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, acc_dtype: tl.
    constexpr=tl.float32, is_compiling: tl.constexpr=False, attend_current:
    tl.constexpr=False):
    block_start_offset = BLOCK_M * seq_prog_id
    M_blk_idxs = block_start_offset + M_range
    M_mask = M_blk_idxs < seq_length
    NO_M_MASK = block_start_offset + BLOCK_M - 1 < seq_length
    N_blk_idxs_start = 0
    N_blk_idxs = N_blk_idxs_start + N_range
    DO_blk_ptrs = DO_head_seq_ptr + (stride_dom * M_blk_idxs[:, None] + 
        stride_dod * D_range[None, :])
    K_blk_ptrs = K_head_seq_ptr + (stride_kn * N_blk_idxs[:, None] + 
        stride_kd * D_range[None, :])
    Q_blk_ptrs = Q_head_seq_ptr + (stride_qm * M_blk_idxs[:, None] + 
        stride_qd * D_range[None, :])
    V_blk_ptrs = V_head_seq_ptr + (stride_vn * N_blk_idxs[:, None] + 
        stride_vd * D_range[None, :])
    A_blk_ptrs = A_head_seq_ptr + stride_am * M_blk_idxs
    DQ_blk_ptrs = DQ_head_seq_ptr + (stride_dqm * M_blk_idxs[:, None] + 
        stride_dqd * D_range[None, :])
    DK_blk_ptrs = DK_head_seq_ptr + (stride_dkn * N_blk_idxs[:, None] + 
        stride_dkd * D_range[None, :])
    DV_blk_ptrs = DV_head_seq_ptr + (stride_dvn * N_blk_idxs[:, None] + 
        stride_dvd * D_range[None, :])
    DR_blk_ptrs = DR_head_seq_ptr + stride_drm * M_blk_idxs
    if NO_D_MASK:
        if NO_M_MASK:
            q = tl.load(Q_blk_ptrs)
            do = tl.load(DO_blk_ptrs)
            dr = tl.load(DR_blk_ptrs)
            neg_log_acc = tl.load(A_blk_ptrs, mask=M_mask)
        else:
            q = tl.load(Q_blk_ptrs, mask=M_mask[:, None])
            do = tl.load(DO_blk_ptrs, mask=M_mask[:, None])
            dr = tl.load(DR_blk_ptrs, mask=M_mask)
            neg_log_acc = tl.load(A_blk_ptrs, mask=M_mask)
    else:
        MD_mask = M_mask[:, None] & D_mask[None, :]
        q = tl.load(Q_blk_ptrs, mask=MD_mask)
        do = tl.load(DO_blk_ptrs, mask=MD_mask)
        dr = tl.load(DR_blk_ptrs, mask=M_mask)
        neg_log_acc = tl.load(A_blk_ptrs, mask=M_mask)
    neg_log_acc = neg_log_acc.to(dtype=acc_dtype)
    grad_prev_acc = tl.zeros((BLOCK_M,), dtype=acc_dtype)
    dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=acc_dtype)
    fwd_cm = tl.trans(cm)
    iters = (block_start_offset + BLOCK_M) // BLOCK_N
    for i in range(iters):
        on_band = iters - i - 1 < BLOCK_M // BLOCK_N
        N_mask = N_blk_idxs < seq_length
        NO_N_MASK = N_blk_idxs_start + BLOCK_N - 1 < seq_length
        k, v = load_kv(K_blk_ptrs, V_blk_ptrs, N_mask=N_mask, NO_N_MASK=
            N_blk_idxs_start + BLOCK_N - 1 < seq_length, D_mask=D_mask,
            NO_D_MASK=NO_D_MASK)
        p, log_om_beta, neg_log_acc = compute_block(q, k, qk_scale,
            neg_log_acc, M_blk_idxs, N_blk_idxs, cm, on_band, ALLOW_TF32,
            attend_current=attend_current, backward=True, is_compiling=
            is_compiling)
        if not NO_M_MASK:
            neg_log_acc = tl.where(M_mask, neg_log_acc, 0.0)
        att_dA = p * (tl.dot(do, tl.trans(v), allow_tf32=ALLOW_TF32) - dr[:,
            None])
        cumul_att_dA = tl.dot(att_dA.to(cm.dtype), fwd_cm, allow_tf32=
            ALLOW_TF32) + grad_prev_acc[:, None]
        grad_prev_acc += tl.sum(att_dA, axis=1)
        beta = 1 - tl.exp2(log_om_beta)
        dqk = att_dA - beta * cumul_att_dA
        dq = tl.dot(dqk.to(k.dtype), k, acc=dq, allow_tf32=ALLOW_TF32)
        block_dk = tl.dot(tl.trans(dqk).to(q.dtype), q, allow_tf32=ALLOW_TF32
            ) * logit_scale
        block_dv = tl.dot(tl.trans(p), do.to(p.dtype), allow_tf32=ALLOW_TF32)
        locked_add(KV_Lock_ptr + i, KV_Count_ptr + i, DK_blk_ptrs, block_dk,
            DV_blk_ptrs, block_dv, N_mask, NO_N_MASK, D_mask, NO_D_MASK)
        N_blk_idxs += BLOCK_N
        N_blk_idxs_start += BLOCK_N
        K_blk_ptrs += BLOCK_N * stride_kn
        V_blk_ptrs += BLOCK_N * stride_vn
        DK_blk_ptrs += BLOCK_N * stride_dkn
        DV_blk_ptrs += BLOCK_N * stride_dvn
    dq = (logit_scale * dq).to(DQ_head_seq_ptr.type.element_ty)
    if NO_D_MASK:
        tl.store(DQ_blk_ptrs, dq, mask=M_mask[:, None])
    else:
        tl.store(DQ_blk_ptrs, dq, mask=M_mask[:, None] & D_mask[None, :])


@triton.jit
def locked_add(Lock_ptr, Count_ptr, A_ptrs, a, B_ptrs, b, N_mask, NO_N_MASK,
    D_mask, NO_D_MASK: tl.constexpr):
    while tl.atomic_cas(Lock_ptr, 0, 1) == 1:
        pass
    count = tl.load(Count_ptr, eviction_policy='evict_last')
    if NO_D_MASK:
        if NO_N_MASK:
            if count == 0:
                tl.store(Count_ptr, True, eviction_policy='evict_last')
            else:
                a += tl.load(A_ptrs, eviction_policy='evict_last')
                b += tl.load(B_ptrs, eviction_policy='evict_last')
            tl.store(A_ptrs, a, eviction_policy='evict_last')
            tl.store(B_ptrs, b, eviction_policy='evict_last')
        else:
            if count == 0:
                tl.store(Count_ptr, True, eviction_policy='evict_last')
            else:
                a += tl.load(A_ptrs, mask=N_mask[:, None], eviction_policy=
                    'evict_last')
                b += tl.load(B_ptrs, mask=N_mask[:, None], eviction_policy=
                    'evict_last')
            tl.store(A_ptrs, a, mask=N_mask[:, None], eviction_policy=
                'evict_last')
            tl.store(B_ptrs, b, mask=N_mask[:, None], eviction_policy=
                'evict_last')
    else:
        mask = N_mask[:, None] & D_mask[None, :]
        if count == 0:
            tl.store(Count_ptr, True, eviction_policy='evict_last')
        else:
            a += tl.load(A_ptrs, mask=mask, eviction_policy='evict_last')
            b += tl.load(B_ptrs, mask=mask, eviction_policy='evict_last')
        tl.store(A_ptrs, a, mask=mask, eviction_policy='evict_last')
        tl.store(B_ptrs, b, mask=mask, eviction_policy='evict_last')
    tl.atomic_xchg(Lock_ptr, 0)


def _bwd(do, dr, q, k, v, neg_log_acc, logit_scale, attend_current=False,
    BLOCK_M=64, BLOCK_N=32):
    batch_size, num_heads, token_size, dim_size = q.size()
    M_count = triton.cdiv(token_size, BLOCK_M)
    N_count = triton.cdiv(token_size, BLOCK_N)
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    M_count = triton.cdiv(token_size, BLOCK_M)
    N_count = M_count * (BLOCK_M // BLOCK_N)
    dkdv_lock = torch.zeros((batch_size, num_heads, N_count), dtype=torch.
        int32, device=q.device)
    dkdv_count = torch.zeros((batch_size, num_heads, N_count), dtype=torch.
        bool, device=q.device)
    _compileable_backward(do, dr, q, k, v, neg_log_acc, logit_scale,
        attend_current, BLOCK_M, BLOCK_N, batch_size, num_heads, token_size,
        dim_size, M_count, N_count, dq, dk, dv, dkdv_lock, dkdv_count)
    return dq, dk, dv


@custom_op('attn_bwd', mutates_args={'dq', 'dk', 'dv', 'dkdv_lock',
    'dkdv_count'})
def _compileable_backward(do: torch.Tensor, dr: torch.Tensor, q: torch.
    Tensor, k: torch.Tensor, v: torch.Tensor, neg_log_acc: torch.Tensor,
    logit_scale: float, attend_current: bool, BLOCK_M: int, BLOCK_N: int,
    batch_size: int, num_heads: int, token_size: int, dim_size: int,
    M_count: int, N_count: int, dq: torch.Tensor, dk: torch.Tensor, dv:
    torch.Tensor, dkdv_lock: torch.Tensor, dkdv_count: torch.Tensor) ->None:
    BLOCK_D = triton.next_power_of_2(dim_size)
    _backward[batch_size, num_heads, M_count](do, do.stride(0), do.stride(1
        ), do.stride(2), do.stride(3), dr, dr.stride(0), dr.stride(1), dr.
        stride(2), neg_log_acc, neg_log_acc.stride(0), neg_log_acc.stride(1
        ), neg_log_acc.stride(2), q, q.stride(0), q.stride(1), q.stride(2),
        q.stride(3), k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v, v.stride(0), v.stride(1), v.stride(2), v.stride(3), dq, dq.
        stride(0), dq.stride(1), dq.stride(2), dq.stride(3), dk, dk.stride(
        0), dk.stride(1), dk.stride(2), dk.stride(3), dv, dv.stride(0), dv.
        stride(1), dv.stride(2), dv.stride(3), dkdv_lock, dkdv_count, 
        num_heads * N_count, N_count, logit_scale=logit_scale,
        attend_current=attend_current, batch_size=batch_size, token_size=
        token_size, head_size=dim_size, num_heads=num_heads, BLOCK_M=
        BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, NO_D_MASK=BLOCK_D ==
        dim_size, NO_M_MASK=token_size % BLOCK_M == 0, NO_N_MASK=token_size %
        BLOCK_N == 0, ALLOW_TF32=ALLOW_TF32, inv_log2=inv_log2, acc_dtype=
        tl.float32, is_compiling=False)


# Backward method (kernel launch code)
def _StickBreakingAttention_backward(ctx, do: torch.Tensor, drem: torch.Tensor
    ):
    logit_scale = ctx.logit_scale
    attend_current = ctx.attend_current
    q, k, v, neg_log_acc = ctx.saved_tensors
    BLOCK_M = BWD_BLOCK_M
    BLOCK_N = BWD_BLOCK_N
    dq, dk, dv = _bwd(do, drem, q, k, v, neg_log_acc, logit_scale,
        attend_current=attend_current, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return dq, dk, dv, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class StickBreakingAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        inv_temp: float, attend_current: bool=False):
        no_grad = not ctx.needs_input_grad[0]
        logit_scale = inv_temp
        BLOCK_M = FWD_BLOCK_M
        BLOCK_N = FWD_BLOCK_N
        o, rem, neg_log_acc = _fwd(q, k, v, logit_scale=inv_temp, no_grad=
            no_grad, return_attention=False, BLOCK_M=BLOCK_M, BLOCK_N=
            BLOCK_N, attend_current=attend_current)
        ctx.save_for_backward(q, k, v, neg_log_acc)
        ctx.logit_scale = logit_scale
        ctx.attend_current = attend_current
        return o, rem

    @staticmethod
    def backward(ctx, do: torch.Tensor, drem: torch.Tensor):
        logit_scale = ctx.logit_scale
        attend_current = ctx.attend_current
        q, k, v, neg_log_acc = ctx.saved_tensors
        BLOCK_M = BWD_BLOCK_M
        BLOCK_N = BWD_BLOCK_N
        dq, dk, dv = _bwd(do, drem, q, k, v, neg_log_acc, logit_scale,
            attend_current=attend_current, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
        return dq, dk, dv, None, None
