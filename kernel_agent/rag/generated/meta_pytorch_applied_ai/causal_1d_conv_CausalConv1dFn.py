# SPDX-License-Identifier: BSD-3-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/meta-pytorch/applied-ai
# Source-Files: kernels/triton/inference/mamba/causal_1d_conv/causal_1d_conv/causal_1d_conv.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_6p51e771/applied-ai-main/kernels/triton/inference/mamba/causal_1d_conv/causal_1d_conv/causal_1d_conv.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=[triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256},
    num_stages=3, num_warps=8), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 
    128}, num_stages=3, num_warps=8), triton.Config({'BLOCK_M': 128,
    'BLOCK_N': 64}, num_stages=3, num_warps=8), triton.Config({'BLOCK_M': 
    128, 'BLOCK_N': 32}, num_stages=3, num_warps=8), triton.Config({
    'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=8), triton.
    Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=3, num_warps=8
    ), triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256}, num_stages=3,
    num_warps=8), triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_stages
    =3, num_warps=8), triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64},
    num_stages=3, num_warps=8), triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32
    }, num_stages=3, num_warps=8)], key=['seqlen', 'dim', 'batch'])
@triton.jit()
def _causal_conv1d_fwd_kernel(x_ptr, w_ptr, bias_ptr, initial_states_ptr,
    o_ptr, batch, dim, seqlen, stride_x_seq, stride_x_dim, stride_x_token,
    stride_weight_dim, stride_weight_width, stride_istate_seq,
    stride_istate_dim, stride_istate_token, stride_o_seq, stride_o_dim,
    stride_o_token, HAS_BIAS: tl.constexpr, KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr, HAS_INITIAL_STATES: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    indices_0 = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_seqs = indices_0 // seqlen
    idx_tokens = indices_0 % seqlen
    x_base = x_ptr + (idx_seqs * stride_x_seq)[:, None]
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    w_base = w_ptr + idx_feats * stride_weight_dim
    load_init_state = False
    if HAS_INITIAL_STATES:
        load_init_state = tl.min(idx_tokens) < KERNEL_WIDTH - 1
        initial_states_base = initial_states_ptr + (idx_seqs *
            stride_istate_seq)[:, None] + (idx_feats * stride_istate_dim)[
            None, :]
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)[None, :]
        acc = tl.broadcast_to(acc, (BLOCK_M, BLOCK_N))
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    PADDING_W = KERNEL_WIDTH - 1
    for j in range(KERNEL_WIDTH):
        idx_x_w = j - PADDING_W + idx_tokens
        x_ptrs = x_base + ((idx_x_w * stride_x_token)[:, None] + (idx_feats *
            stride_x_dim)[None, :])
        mask_x = (idx_seqs < batch)[:, None] & (idx_x_w >= 0)[:, None] & (
            idx_x_w < seqlen)[:, None] & (idx_feats < dim)[None, :]
        if HAS_INITIAL_STATES:
            if load_init_state:
                initial_states_ptrs = initial_states_base + ((idx_x_w +
                    KERNEL_WIDTH - 1) * stride_istate_token)[:, None]
                mask_w = (idx_seqs < batch)[:, None] & (idx_x_w < 0)[:, None
                    ] & (idx_feats < dim)[None, :]
                initial_states = tl.load(initial_states_ptrs, mask_w, 0.0)
            else:
                initial_states = tl.zeros((BLOCK_M, BLOCK_N), dtype=x_ptr.
                    dtype.element_ty)
            matrix_x = tl.load(x_ptrs, mask=mask_x, other=initial_states)
        else:
            matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w_ptrs = w_base[None, :] + j * stride_weight_width
        mask_w = (idx_feats < dim)[None, :]
        matrix_w = tl.load(w_ptrs, mask_w, other=0.0)
        acc += matrix_x * matrix_w
    if SILU_ACTIVATION:
        acc = acc / (1 + tl.exp(-acc))
    mask = (idx_seqs < batch)[:, None] & (idx_tokens < seqlen)[:, None] & (
        idx_feats < dim)[None, :]
    o_ptrs = o_ptr + (idx_seqs * stride_o_seq)[:, None] + (idx_tokens *
        stride_o_token)[:, None] + (idx_feats * stride_o_dim)[None, :]
    tl.store(o_ptrs, acc, mask=mask)


def causal_conv1d_fwd(x: torch.Tensor, weight: torch.Tensor, bias: Optional
    [torch.Tensor]=None, seq_idx: Optional[torch.Tensor]=None,
    initial_states: Optional[torch.Tensor]=None, return_final_states:
    Optional[torch.Tensor]=False, final_states_out: Optional[torch.Tensor]=
    None, activation: Optional[Literal['silu', 'swish']]=None):
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    assert (dim, width) == weight.shape
    assert x.stride(2) == 1 or x.stride(1) == 1
    assert weight.stride(1) == 1
    is_channel_last = (x.stride(1) == 1) & (x.stride(2) > 1)
    stride_w_dim = weight.stride(0)
    stride_w_width = weight.stride(1)
    weight = weight.transpose(0, 1).contiguous()
    stride_w_dim = weight.stride(1)
    stride_w_width = weight.stride(0)
    assert return_final_states is False
    stride_istate_seq = 0
    stride_istate_dim = 0
    stride_istate_token = 0
    if initial_states is not None:
        assert (batch, dim, width - 1) == initial_states.shape
        stride_istate_seq = initial_states.stride(0)
        stride_istate_dim = initial_states.stride(1)
        stride_istate_token = initial_states.stride(2)
        assert stride_istate_dim == 1
    out = torch.empty_like(x)
    if not is_channel_last:
        assert 0, 'Need to run in channel-last layout'
    else:

        def grid(META):
            return triton.cdiv(batch * seqlen, META['BLOCK_M']), triton.cdiv(
                dim, META['BLOCK_N'])
        with torch.cuda.device(x.device.index):
            _causal_conv1d_fwd_kernel[grid](x, weight, bias, initial_states,
                out, batch, dim, seqlen, x.stride(0), x.stride(1), x.stride
                (2), stride_w_dim, stride_w_width, stride_istate_seq,
                stride_istate_dim, stride_istate_token, out.stride(0), out.
                stride(1), out.stride(2), HAS_BIAS=bias is not None,
                KERNEL_WIDTH=width, SILU_ACTIVATION=activation in ['silu',
                'swish'], HAS_INITIAL_STATES=initial_states is not None)
    return out


# Forward method (kernel launch code)
def _CausalConv1dFn_forward(ctx, x, weight, bias=None, seq_idx=None,
    initial_states=None, return_final_states: bool=False, final_states_out=
    None, activation: Optional[Literal['silu', 'swish']]=None):
    if x.stride(2) != 1 and x.stride(1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None
    if seq_idx is not None:
        assert initial_states is None, 'initial_states must be None if seq_idx is not None'
        assert not return_final_states, "If seq_idx is not None, we don't return final_states_out"
    seq_idx = seq_idx.contiguous() if seq_idx is not None else None
    if initial_states is not None and (initial_states.stride(2) != 1 and 
        initial_states.stride(1) != 1):
        initial_states = initial_states.contiguous()
    if return_final_states:
        assert x.stride(1
            ) == 1, 'Only channel-last layout support returning final_states_out'
        if final_states_out is not None:
            assert final_states_out.stride(2) == 1 or final_states_out.stride(1
                ) == 1
        else:
            batch, dim, seqlen = x.shape
            width = weight.shape[1]
            final_states_out = torch.empty(batch, width - 1, dim, device=x.
                device, dtype=x.dtype).transpose(1, 2)
    else:
        final_states_out = None
    ctx.activation = activation
    out = causal_conv1d_fwd(x, weight, bias=bias, seq_idx=seq_idx,
        initial_states=initial_states, return_final_states=
        return_final_states, final_states_out=final_states_out, activation=
        ctx.activation)
    ctx.save_for_backward(x, weight, bias, seq_idx, initial_states)
    ctx.return_final_states = return_final_states
    ctx.return_dinitial_states = (initial_states is not None and
        initial_states.requires_grad)
    return out if not return_final_states else (out, final_states_out)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

# Backward method (kernel launch code)
def _CausalConv1dFn_backward(ctx, dout, *args):
    """dout = dL/dy
        RETURN: dL/dx, dL/dweight, dL/dbias, ...
        GIVEN THAT: def forward(ctx, x, weight, bias=None...)
        """
    x, weight, bias, seq_idx, initial_states = ctx.saved_tensors
    dfinal_states = args[0] if ctx.return_final_states else None
    if dout.stride(2) != 1 and dout.stride(1) != 1:
        dout = dout.contiguous()
    dx, dweight, dbias, dinitial_states = causal_conv1d_bwd(x, weight, bias,
        dout, seq_idx, initial_states, dfinal_states, None, ctx.
        return_dinitial_states, ctx.activation)
    return (dx, dweight, dbias if bias is not None else None, None, 
        dinitial_states if initial_states is not None else None, None, None,
        None)


# ============================================================
# autograd.Function Class Definition
# ============================================================

class CausalConv1dFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias=None, seq_idx=None, initial_states=
        None, return_final_states: bool=False, final_states_out=None,
        activation: Optional[Literal['silu', 'swish']]=None):
        if x.stride(2) != 1 and x.stride(1) != 1:
            x = x.contiguous()
        bias = bias.contiguous() if bias is not None else None
        if seq_idx is not None:
            assert initial_states is None, 'initial_states must be None if seq_idx is not None'
            assert not return_final_states, "If seq_idx is not None, we don't return final_states_out"
        seq_idx = seq_idx.contiguous() if seq_idx is not None else None
        if initial_states is not None and (initial_states.stride(2) != 1 and
            initial_states.stride(1) != 1):
            initial_states = initial_states.contiguous()
        if return_final_states:
            assert x.stride(1
                ) == 1, 'Only channel-last layout support returning final_states_out'
            if final_states_out is not None:
                assert final_states_out.stride(2
                    ) == 1 or final_states_out.stride(1) == 1
            else:
                batch, dim, seqlen = x.shape
                width = weight.shape[1]
                final_states_out = torch.empty(batch, width - 1, dim,
                    device=x.device, dtype=x.dtype).transpose(1, 2)
        else:
            final_states_out = None
        ctx.activation = activation
        out = causal_conv1d_fwd(x, weight, bias=bias, seq_idx=seq_idx,
            initial_states=initial_states, return_final_states=
            return_final_states, final_states_out=final_states_out,
            activation=ctx.activation)
        ctx.save_for_backward(x, weight, bias, seq_idx, initial_states)
        ctx.return_final_states = return_final_states
        ctx.return_dinitial_states = (initial_states is not None and
            initial_states.requires_grad)
        return out if not return_final_states else (out, final_states_out)

    @staticmethod
    def backward(ctx, dout, *args):
        """dout = dL/dy
        RETURN: dL/dx, dL/dweight, dL/dbias, ...
        GIVEN THAT: def forward(ctx, x, weight, bias=None...)
        """
        x, weight, bias, seq_idx, initial_states = ctx.saved_tensors
        dfinal_states = args[0] if ctx.return_final_states else None
        if dout.stride(2) != 1 and dout.stride(1) != 1:
            dout = dout.contiguous()
        dx, dweight, dbias, dinitial_states = causal_conv1d_bwd(x, weight,
            bias, dout, seq_idx, initial_states, dfinal_states, None, ctx.
            return_dinitial_states, ctx.activation)
        return (dx, dweight, dbias if bias is not None else None, None, 
            dinitial_states if initial_states is not None else None, None,
            None, None)
