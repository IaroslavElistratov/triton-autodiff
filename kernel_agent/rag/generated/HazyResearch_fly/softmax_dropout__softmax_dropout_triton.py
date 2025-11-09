# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/HazyResearch/fly
# Source-Files: src/ops/triton/softmax_dropout.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_lyot2jni/fly-master/src/ops/triton/softmax_dropout.py
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
from einops import repeat

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8), triton.Config({}, num_warps=16), triton.Config({},
    num_warps=32)], key=['K'])
@triton.heuristics({'DEPTH': lambda nargs: get_depth(nargs['K'])})
@triton.heuristics({'IS_FP16': lambda nargs: nargs['Y'].dtype == torch.float16}
    )
@triton.jit
def _softmax(Y, X, M, stride_ym, stride_yn, stride_xm, stride_xn, stride_m,
    K, LOG: tl.constexpr, MASK_TYPE: tl.constexpr, CAUSAL: tl.constexpr,
    DEPTH: tl.constexpr, IS_FP16: tl.constexpr):
    """
    Fused softmax kernel over a 3d tensor.
    The softmax is applied over the last dimension, meaning that this is equivalent to torch.softmax(tensor, dim=-1)

    Note, if the last dimension is large, say 128K elements, the kernel compile time can shot up to many minutes when
    the kernel is run for the first time.
    """
    m = tl.program_id(0)
    n = tl.program_id(1)
    k = tl.arange(0, DEPTH)
    x_ptrs = X + m * stride_xm + n * stride_xn + k
    io_mask = k < K
    if CAUSAL:
        io_mask = io_mask & (k <= n)
    x = tl.load(x_ptrs, mask=io_mask, other=float('-inf'))
    if CAUSAL:
        off = float('-inf')
        off = off.to(x.dtype)
        x = tl.where(k > n, off, x)
    if MASK_TYPE is not None:
        if MASK_TYPE == 'qk':
            mask_ptrs = M + n * stride_m + k
        elif MASK_TYPE == 'bk':
            mask_ptrs = M + m * stride_m + k
        add_mask = tl.load(mask_ptrs, io_mask, other=float('-inf'))
        x += add_mask
    z = x - tl.max(x, axis=0)
    if IS_FP16:
        z = z.to(tl.float32)
    num = tl.exp(z)
    denom = tl.sum(num, axis=0)
    if LOG:
        y = z - tl.log(denom)
    else:
        y = num / denom
    y_ptrs = Y + m * stride_ym + n * stride_yn + k
    tl.store(y_ptrs, y, mask=k < K)


# Forward method (kernel launch code)
@custom_fwd(cast_inputs=torch.float16 if _triton_softmax_fp16_enabled else None
    )
def __softmax_dropout_triton_forward(ctx, x, p, mask, causal, mask_type):
    """
        Fused softmax implementation, using the Triton programming model.
        This only supports a reduction over the last dimension for now
        Argument:
            x: (bs, nheads, q_seqlen, k_seqlen)
            mask: (bs, 1, 1, k_seqlen)
        """
    assert x.ndim == 4
    x_ = x.unsqueeze(0) if x.ndim == 2 else x
    x_ = x_.flatten(0, -3)
    if not x_.is_contiguous():
        x_ = x_.contiguous()
    y = torch.empty_like(x_)
    assert y.stride(2) == 1 and x_.stride(2
        ) == 1, f'{x.shape} - {x_.shape} - {x_.stride()}'
    grid_2d = x_.shape[0], x_.shape[1]
    if mask is None:
        mask = x_
        mask_type = None
    else:
        assert mask.dtype == x.dtype, 'An additive mask is requested'
        if mask_type == 'bk':
            mask = repeat(mask, 'b 1 1 s -> b h 1 s', h=x_.shape[0] // mask
                .shape[0])
        mask = mask.flatten(0, -2).contiguous()
    _softmax[grid_2d](y, x_, mask, y.stride(0), y.stride(1), x_.stride(0),
        x_.stride(1), mask.stride(0), x_.shape[2], LOG=False, MASK_TYPE=
        mask_type, CAUSAL=causal)
    dropout_results, dropout_mask = torch._fused_dropout(y, p=1.0 - p)
    ctx.save_for_backward(y, dropout_mask)
    ctx.dropout_prob = p
    ctx.causal = causal
    ctx.mask_type = mask_type
    return dropout_results.reshape_as(x)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8), triton.Config({}, num_warps=16), triton.Config({},
    num_warps=32)], key=['K'])
@triton.heuristics({'DEPTH': lambda nargs: get_depth(nargs['K'])})
@triton.heuristics({'IS_FP16': lambda nargs: nargs['GradIn'].dtype == torch
    .float16})
@triton.jit
def _softmax_backward(GradIn, GradOut, Out, stride_bm, stride_bn, stride_gm,
    stride_gn, stride_om, stride_on, K, LOG: tl.constexpr, CAUSAL: tl.
    constexpr, DEPTH: tl.constexpr, IS_FP16: tl.constexpr):
    """
    Compute the softmax gradients.
    ..Note: Not autotuning for now because this would lead to broken accumulated gradients
    """
    m = tl.program_id(0)
    n = tl.program_id(1)
    k = tl.arange(0, DEPTH)
    grad_out_ptrs = GradOut + m * stride_gm + n * stride_gn + k
    out_ptrs = Out + m * stride_om + n * stride_on + k
    io_mask = k < K
    if CAUSAL:
        io_mask = io_mask & (k <= n)
    g = tl.load(grad_out_ptrs, mask=io_mask, other=float(0))
    o = tl.load(out_ptrs, mask=io_mask, other=float(0))
    if CAUSAL:
        zero = float(0)
        zero = zero.to(g.dtype)
        g = tl.where(k > n, zero, g)
        o = tl.where(k > n, zero, o)
    if LOG:
        s = tl.sum(g, 0)
        if IS_FP16:
            o = o.to(tl.float32)
        grad_in = g - tl.exp(o) * s
    else:
        s = tl.sum(g * o, 0)
        grad_in = o * (g - s)
    grad_in_ptrs = GradIn + m * stride_bm + n * stride_bn + k
    tl.store(grad_in_ptrs, grad_in, mask=k < K)


# Backward method (kernel launch code)
@custom_bwd
def __softmax_dropout_triton_backward(ctx, grad_out):
    y, dropout_mask = ctx.saved_tensors
    grad_out_ = grad_out.unsqueeze(0) if grad_out.ndim == 2 else grad_out
    grad_out_ = grad_out_.flatten(0, -3)
    grid_2d = grad_out_.shape[0], grad_out_.shape[1]
    grad_out_, y = map(lambda x: x.contiguous(), [grad_out_, y])
    if (FAST_MHA_AVAILABLE and grad_out.dtype == torch.float16 and not ctx.
        causal and ctx.mask_type == 'bk'):
        grad_in = additive_mask_softmax_dropout_backward(True, 1, grad_out_,
            y, dropout_mask, ctx.dropout_prob)
    else:
        dropout_grads = torch._masked_scale(grad_out_, dropout_mask, 1.0 /
            (1.0 - ctx.dropout_prob))
        grad_in = torch.empty_like(y)
        _softmax_backward[grid_2d](grad_in, dropout_grads, y, grad_in.
            stride(0), grad_in.stride(1), grad_out_.stride(0), grad_out_.
            stride(1), y.stride(0), y.stride(1), y.shape[2], LOG=False,
            CAUSAL=ctx.causal)
    return grad_in.reshape_as(grad_out), None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _softmax_dropout_triton(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16 if _triton_softmax_fp16_enabled else
        None)
    def forward(ctx, x, p, mask, causal, mask_type):
        """
        Fused softmax implementation, using the Triton programming model.
        This only supports a reduction over the last dimension for now
        Argument:
            x: (bs, nheads, q_seqlen, k_seqlen)
            mask: (bs, 1, 1, k_seqlen)
        """
        assert x.ndim == 4
        x_ = x.unsqueeze(0) if x.ndim == 2 else x
        x_ = x_.flatten(0, -3)
        if not x_.is_contiguous():
            x_ = x_.contiguous()
        y = torch.empty_like(x_)
        assert y.stride(2) == 1 and x_.stride(2
            ) == 1, f'{x.shape} - {x_.shape} - {x_.stride()}'
        grid_2d = x_.shape[0], x_.shape[1]
        if mask is None:
            mask = x_
            mask_type = None
        else:
            assert mask.dtype == x.dtype, 'An additive mask is requested'
            if mask_type == 'bk':
                mask = repeat(mask, 'b 1 1 s -> b h 1 s', h=x_.shape[0] //
                    mask.shape[0])
            mask = mask.flatten(0, -2).contiguous()
        _softmax[grid_2d](y, x_, mask, y.stride(0), y.stride(1), x_.stride(
            0), x_.stride(1), mask.stride(0), x_.shape[2], LOG=False,
            MASK_TYPE=mask_type, CAUSAL=causal)
        dropout_results, dropout_mask = torch._fused_dropout(y, p=1.0 - p)
        ctx.save_for_backward(y, dropout_mask)
        ctx.dropout_prob = p
        ctx.causal = causal
        ctx.mask_type = mask_type
        return dropout_results.reshape_as(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        y, dropout_mask = ctx.saved_tensors
        grad_out_ = grad_out.unsqueeze(0) if grad_out.ndim == 2 else grad_out
        grad_out_ = grad_out_.flatten(0, -3)
        grid_2d = grad_out_.shape[0], grad_out_.shape[1]
        grad_out_, y = map(lambda x: x.contiguous(), [grad_out_, y])
        if (FAST_MHA_AVAILABLE and grad_out.dtype == torch.float16 and not
            ctx.causal and ctx.mask_type == 'bk'):
            grad_in = additive_mask_softmax_dropout_backward(True, 1,
                grad_out_, y, dropout_mask, ctx.dropout_prob)
        else:
            dropout_grads = torch._masked_scale(grad_out_, dropout_mask, 
                1.0 / (1.0 - ctx.dropout_prob))
            grad_in = torch.empty_like(y)
            _softmax_backward[grid_2d](grad_in, dropout_grads, y, grad_in.
                stride(0), grad_in.stride(1), grad_out_.stride(0),
                grad_out_.stride(1), y.stride(0), y.stride(1), y.shape[2],
                LOG=False, CAUSAL=ctx.causal)
        return grad_in.reshape_as(grad_out), None, None, None, None
