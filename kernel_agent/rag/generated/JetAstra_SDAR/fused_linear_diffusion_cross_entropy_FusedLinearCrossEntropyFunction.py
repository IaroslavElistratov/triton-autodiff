# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/JetAstra/SDAR
# Source-Files: training/model/SDAR-4B-Chat/fused_linear_diffusion_cross_entropy.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_yh1v2u1p/SDAR-main/training/model/SDAR-4B-Chat/fused_linear_diffusion_cross_entropy.py
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

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def cross_entropy_kernel(logits, lse, target, p_mask, loss, total,
    ignore_index, label_smoothing: tl.constexpr, logit_scale: tl.constexpr,
    reduction: tl.constexpr, V: tl.constexpr, BV: tl.constexpr):
    """
    This kernel computes both cross entropy loss and the gradient of the input.
    We only consider hard label + mean reduction for now.
    Please refer to https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for the math.

    Args:
        logits:
            Pointer to logits tensor.
        lse:
            Pointer to logsumexp tensor.
        target: Pointer to target tensor.
        loss:
            Pointer to tensor to store the loss.
        V (int):
            The number of columns in the input tensor.
        total (int):
            The number of non-ignored classes.
        ignore_index (int):
            The index to ignore in the target.
        label_smoothing (float):
            The amount of smoothing when computing the loss, where 0.0 means no smoothing.
        reduction (str):
            The string for the reduction to apply
        BV (int):
            The block size for vocab.
    """
    i_n = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)
    b_y = tl.load(target + i_n)
    b_p_mask = tl.load(p_mask + i_n)
    logits += i_n * V
    if b_y == ignore_index:
        for i in range(0, V, BV):
            o_v = i + tl.arange(0, BV)
            tl.store(logits + o_v, 0.0, mask=o_v < V)
        return
    b_l = tl.load(logits + b_y) * logit_scale
    b_lse = tl.load(lse + i_n)
    b_loss = (b_lse - b_l) / b_p_mask
    b_z = 0.0
    eps = label_smoothing / V
    tl.debug_barrier()
    for iv in range(0, NV):
        o_v = iv * BV + tl.arange(0, BV)
        b_logits = tl.load(logits + o_v, mask=o_v < V, other=float('-inf')
            ) * logit_scale
        if label_smoothing > 0:
            b_z += tl.sum(tl.where(o_v < V, -eps * b_logits, 0.0))
        b_p = (tl.exp(b_logits - b_lse) - eps) * logit_scale
        b_p /= b_p_mask
        if reduction == 'mean':
            b_p = b_p / total
        tl.store(logits + o_v, b_p, mask=o_v < V)
        tl.debug_barrier()
    if label_smoothing > 0:
        b_loss = b_loss * (1 - label_smoothing) + (b_z + label_smoothing *
            b_lse)
    b_l = tl.load(logits + b_y)
    if reduction == 'mean':
        b_loss = b_loss / total
        b_l += (label_smoothing - 1) / b_p_mask / total * logit_scale
    else:
        b_l += (label_smoothing - 1) / b_p_mask * logit_scale
    tl.store(loss + i_n, b_loss)
    tl.store(logits + b_y, b_l)


@triton.heuristics({'HAS_SCALE': lambda args: args['scale'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for
    num_warps in [1, 2, 4, 8, 16, 32]], key=['D'])
@triton.jit
def logsumexp_fwd_kernel(x, z, scale, D: tl.constexpr, B: tl.constexpr,
    HAS_SCALE: tl.constexpr):
    i_n, i_d = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    o_d = i_d * B + tl.arange(0, B)
    m_d = o_d < D
    b_x = tl.load(x + i_n * D + o_d, mask=m_d, other=-float('inf'))
    if HAS_SCALE:
        b_x = b_x * scale
    b_m = tl.max(b_x, 0)
    b_z = tl.log(tl.sum(tl.exp(b_x - b_m), 0)) + b_m
    tl.store(z + i_n * tl.cdiv(D, B) + i_d, b_z)


def fused_linear_cross_entropy_forward(x: torch.Tensor, target: torch.
    LongTensor, weight: torch.Tensor, bias: torch.Tensor=None, p_mask:
    torch.Tensor=None, ignore_index: int=-100, label_smoothing: float=0.0,
    logit_scale: float=1.0, num_chunks: int=8, reduction: str='mean'):
    device = x.device
    N, H, V = *x.shape, weight.shape[0]
    BV = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    NC = min(num_chunks, triton.cdiv(V, H))
    C = triton.next_power_of_2(triton.cdiv(N, NC))
    NC = triton.cdiv(N, C)
    dx = torch.zeros_like(x, device=device)
    dw = torch.zeros_like(weight, device=device, dtype=torch.float
        ) if weight is not None else None
    db = torch.zeros_like(bias, device=device, dtype=torch.float
        ) if bias is not None else None
    loss = torch.zeros(N, device=device, dtype=torch.float)
    total = target.ne(ignore_index).sum().item()
    for ic in range(NC):
        start, end = ic * C, min((ic + 1) * C, N)
        c_x = x[start:end]
        c_logits = F.linear(c_x, weight, bias)
        c_target = target[start:end]
        c_p_mask = p_mask[start:end]
        c_lse = logsumexp_fwd(c_logits, scale=logit_scale, dtype=torch.float)
        c_loss = loss[start:end]
        cross_entropy_kernel[c_logits.shape[0],](logits=c_logits, lse=c_lse,
            target=c_target, p_mask=c_p_mask, loss=c_loss, total=total,
            ignore_index=ignore_index, label_smoothing=label_smoothing,
            logit_scale=logit_scale, reduction=reduction, V=V, BV=BV,
            num_warps=32)
        dx[start:end] = torch.mm(c_logits, weight)
        if weight is not None:
            dw += c_logits.t() @ c_x
        if bias is not None:
            torch.add(input=db, other=c_logits.sum(0), out=db)
    loss = loss.sum()
    if dw is not None:
        dw = dw.to(weight)
    if db is not None:
        db = db.to(bias)
    return loss, dx, dw, db


def logsumexp_fwd(x, scale: Optional[float]=None, dtype: Optional[torch.
    dtype]=None):
    """
    Compute the logsumexp of the input tensor over the last dimension.

    Args:
        x (Tensor):
            The input tensor of any shape.
        scale (Optional[float]):
            The scale applied to the input tensor. Default: `None`.
        dtype (Optional[torch.dtype]):
            The data type of the output tensor. Default: `None`.
    Returns:
        Tensor: The logsumexp of the input tensor.
    """
    shape = x.shape
    x = x.view(-1, shape[-1])
    N, D = x.shape
    B = min(triton.next_power_of_2(D), 64 * 1024)
    ND = triton.cdiv(D, B)
    z = x.new_empty(N, ND, dtype=torch.float)
    logsumexp_fwd_kernel[N, ND](x=x, z=z, scale=scale, D=D, B=B)
    z = z.logsumexp(-1).view(*shape[:-1])
    if dtype is not None and dtype != torch.float:
        z = z.to(dtype)
    return z


# Forward method (kernel launch code)
def _FusedLinearCrossEntropyFunction_forward(ctx, x: torch.Tensor, target:
    torch.LongTensor, weight: torch.Tensor, bias: torch.Tensor=None, p_mask:
    torch.Tensor=None, ignore_index: int=-100, label_smoothing: float=0.0,
    logit_scale: float=1.0, num_chunks: int=8, reduction: str='mean'):
    """
        Fusing the last linear layer with cross-entropy loss
            Reference: https://github.com/mgmalek/efficient_cross_entropy

        Handle the forward and backward pass of the final linear layer via cross-entropy loss by avoiding
        the materialization of the large logits tensor. Since Cross Entropy Loss is the last layer, we can
        compute the gradient at the forward pass. By doing so, we don't have to store the x and target
        for the backward pass.

        x (torch.Tensor): [batch_size * seq_len, hidden_size]
        target (torch.LongTensor): [batch_size * seq_len]
            where each value is in [0, vocab_size).
        weight (torch.Tensor): [vocab_size, hidden_size]
            where `vocab_size` is the number of classes.
        bias (Optional[torch.Tensor]): [vocab_size]
            where `vocab_size` is the number of classes.
        p_mask(torch.Tensor): [batch_size * seq_len]
            Its shape should be same as target. 
        ignore_index:
            the index to ignore in the target.
        label_smoothing:
            the amount of smoothing when computing the loss, where 0.0 means no smoothing.
        logit_scale: float = 1.0,
            A scaling factor applied to the logits. Default: 1.0
        num_chunks: int
            The number of chunks to split the input tensor into for processing.
            This can help optimize memory usage and computation speed.
            Default: 8
        reduction:
            Specifies the reduction to apply to the output: 'mean' | 'sum'.
            'mean': the weighted mean of the output is taken,
            'sum': the output will be summed.
            Default: 'mean'.
        """
    loss, dx, dw, db = fused_linear_cross_entropy_forward(x, target, weight,
        bias, p_mask, ignore_index, label_smoothing, logit_scale,
        num_chunks, reduction)
    ctx.save_for_backward(dx.detach(), dw.detach() if weight is not None else
        None, db.detach() if bias is not None else None)
    return loss


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def elementwise_mul_kernel(x, g, N: tl.constexpr, B: tl.constexpr):
    """
    This function multiplies each element of the tensor pointed by x with the value pointed by g.
    The multiplication is performed in-place on the tensor pointed by x.

    Parameters:
    x:
        Pointer to the input tensor.
    g:
        Pointer to the gradient output value.
    N (int):
        The number of columns in the input tensor.
    B (int):
        The block size for Triton operations.
    """
    i_x = tl.program_id(0).to(tl.int64)
    o_x = i_x * B + tl.arange(0, B)
    b_g = tl.load(g)
    b_x = tl.load(x + o_x, mask=o_x < N)
    tl.store(x + o_x, b_x * b_g, mask=o_x < N)


def fused_linear_cross_entropy_backward(do: torch.Tensor, dx: torch.Tensor,
    dw: torch.Tensor, db: torch.Tensor):
    if torch.ne(do, torch.tensor(1.0, device=do.device)):
        N, H = dx.shape
        B = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))
        elementwise_mul_kernel[triton.cdiv(N * H, B),](x=dx, g=do, N=N * H,
            B=B, num_warps=32)
        if dw is not None:
            V, H = dw.shape
            elementwise_mul_kernel[triton.cdiv(V * H, B),](x=dw, g=do, N=V *
                H, B=B, num_warps=32)
        if db is not None:
            V = db.shape[0]
            elementwise_mul_kernel[triton.cdiv(V, B),](x=db, g=do, N=V, B=B,
                num_warps=32)
    return dx, dw, db


# Backward method (kernel launch code)
def _FusedLinearCrossEntropyFunction_backward(ctx, do):
    dx, dw, db = ctx.saved_tensors
    dx, dw, db = fused_linear_cross_entropy_backward(do, dx, dw, db)
    return dx, None, dw, db, None, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class FusedLinearCrossEntropyFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, target: torch.LongTensor, weight:
        torch.Tensor, bias: torch.Tensor=None, p_mask: torch.Tensor=None,
        ignore_index: int=-100, label_smoothing: float=0.0, logit_scale:
        float=1.0, num_chunks: int=8, reduction: str='mean'):
        """
        Fusing the last linear layer with cross-entropy loss
            Reference: https://github.com/mgmalek/efficient_cross_entropy

        Handle the forward and backward pass of the final linear layer via cross-entropy loss by avoiding
        the materialization of the large logits tensor. Since Cross Entropy Loss is the last layer, we can
        compute the gradient at the forward pass. By doing so, we don't have to store the x and target
        for the backward pass.

        x (torch.Tensor): [batch_size * seq_len, hidden_size]
        target (torch.LongTensor): [batch_size * seq_len]
            where each value is in [0, vocab_size).
        weight (torch.Tensor): [vocab_size, hidden_size]
            where `vocab_size` is the number of classes.
        bias (Optional[torch.Tensor]): [vocab_size]
            where `vocab_size` is the number of classes.
        p_mask(torch.Tensor): [batch_size * seq_len]
            Its shape should be same as target. 
        ignore_index:
            the index to ignore in the target.
        label_smoothing:
            the amount of smoothing when computing the loss, where 0.0 means no smoothing.
        logit_scale: float = 1.0,
            A scaling factor applied to the logits. Default: 1.0
        num_chunks: int
            The number of chunks to split the input tensor into for processing.
            This can help optimize memory usage and computation speed.
            Default: 8
        reduction:
            Specifies the reduction to apply to the output: 'mean' | 'sum'.
            'mean': the weighted mean of the output is taken,
            'sum': the output will be summed.
            Default: 'mean'.
        """
        loss, dx, dw, db = fused_linear_cross_entropy_forward(x, target,
            weight, bias, p_mask, ignore_index, label_smoothing,
            logit_scale, num_chunks, reduction)
        ctx.save_for_backward(dx.detach(), dw.detach() if weight is not
            None else None, db.detach() if bias is not None else None)
        return loss

    @staticmethod
    def backward(ctx, do):
        dx, dw, db = ctx.saved_tensors
        dx, dw, db = fused_linear_cross_entropy_backward(do, dx, dw, db)
        return dx, None, dw, db, None, None, None, None, None, None
