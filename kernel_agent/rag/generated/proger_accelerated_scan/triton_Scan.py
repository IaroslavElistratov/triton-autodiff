# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/proger/accelerated-scan
# Source-Files: accelerated_scan/triton.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_u0r250fl/accelerated-scan-main/accelerated_scan/triton.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

@triton.jit
def pack64(a, b):
    tl.static_assert(a.dtype == tl.float32)
    tl.static_assert(b.dtype == tl.float32)
    a = a.to(dtype=tl.uint32, bitcast=True).to(tl.uint64)
    a = a << 32
    b = b.to(dtype=tl.uint32, bitcast=True).to(tl.uint64)
    return a | b


@triton.jit
def unpack64(merged):
    tl.static_assert(merged.dtype == tl.uint64)
    b = (merged & 4294967295).to(tl.uint32).to(tl.float32, bitcast=True)
    a = (merged >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    return a, b


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def forward_scan(gates, tokens, outputs, SEQUENCE_LENGTH: tl.constexpr):
    sequence_id = tl.num_programs(axis=1) * tl.program_id(axis=0
        ) + tl.program_id(axis=1)
    strides = tl.arange(0, SEQUENCE_LENGTH) + sequence_id * SEQUENCE_LENGTH
    tokens_ = tl.load(tokens + strides)
    gates_ = tl.load(gates + strides)
    tuples = pack64(tokens_, gates_)
    output_tuples_ = tl.associative_scan(tuples, axis=0, combine_fn=
        first_order_op)
    output_tokens_, output_gates_ = unpack64(output_tuples_)
    tl.store(outputs + strides, output_tokens_)


# Forward method (kernel launch code)
def _Scan_forward(ctx, gates, tokens):
    B, C, T = gates.shape
    assert tokens.shape == (B, C, T)
    assert gates.is_contiguous()
    assert tokens.is_contiguous()
    states = torch.zeros_like(tokens)
    forward_scan[B, C](gates, tokens, states, SEQUENCE_LENGTH=T,
        enable_fp_fusion=False)
    ctx.save_for_backward(states, gates)
    return states


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def backward_scan(gates, tokens, outputs, SEQUENCE_LENGTH: tl.constexpr):
    sequence_id = tl.num_programs(axis=1) * tl.program_id(axis=0
        ) + tl.program_id(axis=1)
    forward_strides = tl.arange(0, SEQUENCE_LENGTH
        ) + sequence_id * SEQUENCE_LENGTH
    reverse_strides = tl.num_programs(axis=0) * tl.num_programs(axis=1
        ) * SEQUENCE_LENGTH - 1 - forward_strides
    tokens_ = tl.load(tokens + reverse_strides)
    gates_ = tl.load(gates + reverse_strides)
    tuples = pack64(tokens_, gates_)
    output_tuples_ = tl.associative_scan(tuples, axis=0, combine_fn=
        first_order_op)
    output_tokens_, output_gates_ = unpack64(output_tuples_)
    tl.store(outputs + reverse_strides, output_tokens_)


# Backward method (kernel launch code)
def _Scan_backward(ctx, grad_output):
    states, gates = ctx.saved_tensors
    B, C, T = gates.shape
    grad_output = grad_output.contiguous()
    assert states.is_contiguous()
    assert gates.is_contiguous()
    d_states = torch.empty_like(states)
    padded_shifted_gates = torch.cat([gates, torch.ones_like(gates[:, :, :1
        ])], dim=-1)[:, :, 1:].contiguous()
    backward_scan[B, C](padded_shifted_gates, grad_output, d_states,
        SEQUENCE_LENGTH=T, enable_fp_fusion=False)
    padded_outputs = torch.cat([torch.zeros_like(states[:, :, :1]), states],
        dim=-1)[:, :, :-1]
    d_gates = padded_outputs * d_states
    d_tokens = d_states
    return d_gates, d_tokens


# ============================================================
# autograd.Function Class Definition
# ============================================================

class Scan(torch.autograd.Function):

    @staticmethod
    def forward(ctx, gates, tokens):
        B, C, T = gates.shape
        assert tokens.shape == (B, C, T)
        assert gates.is_contiguous()
        assert tokens.is_contiguous()
        states = torch.zeros_like(tokens)
        forward_scan[B, C](gates, tokens, states, SEQUENCE_LENGTH=T,
            enable_fp_fusion=False)
        ctx.save_for_backward(states, gates)
        return states

    @staticmethod
    def backward(ctx, grad_output):
        states, gates = ctx.saved_tensors
        B, C, T = gates.shape
        grad_output = grad_output.contiguous()
        assert states.is_contiguous()
        assert gates.is_contiguous()
        d_states = torch.empty_like(states)
        padded_shifted_gates = torch.cat([gates, torch.ones_like(gates[:, :,
            :1])], dim=-1)[:, :, 1:].contiguous()
        backward_scan[B, C](padded_shifted_gates, grad_output, d_states,
            SEQUENCE_LENGTH=T, enable_fp_fusion=False)
        padded_outputs = torch.cat([torch.zeros_like(states[:, :, :1]),
            states], dim=-1)[:, :, :-1]
        d_gates = padded_outputs * d_states
        d_tokens = d_states
        return d_gates, d_tokens
