# %%
# Copied from official triton tutorial (before blackwell support):
# https://github.com/triton-lang/triton/blob/105cb56487cd8a433b8fbfe9cc63c1f1c04a4b2a/python/tutorials/06-fused-attention.py

# CHANGE LOG
#   - non causal only (so no need for the 2nd call to _attn_fwd_inner)
#   - made kernels args that are used as loop bounds -- tl.constexpr
#     - start_m
#     - almost all args to _attn_fwd

# %%
import pytest
import torch

import triton
import triton.language as tl

from triton.backends.autodiff import autodiff



DEVICE = torch.device("cuda:0")


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64]\
    for s in ([1] if is_hip() else [3, 4, 7])\
    for w in [4, 8]\
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr):
    # range of values handled by this stage
    # causal = False
    lo, hi = 0, N_CTX

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        # todo-now: the only place (in this kenrel) where acc is used (besides the "+=" across itters) is here
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    # todo-now: this loop has 3 iter-arg accumulators not just o ne
    return acc, l_i, m_i



@autodiff(idxs_buffers=(4, 5),
          axis_names=[
        # not specifying -- "B", "NUM_HEADS" bc these are paralized over
                ["BLOCK_M", "HEAD_DIM"],  # Q
                ["HEAD_DIM", "BLOCK_N"],  # K
                ["BLOCK_N", "HEAD_DIM"],  # V
                ["BLOCK_M"],              # M
                ["BLOCK_M", "HEAD_DIM"],  # Out
          ]

    )
# todo: rm do_not_specialize
@triton.jit(do_not_specialize=["stride_qz", "stride_qh", "stride_qm", "stride_qk",  "stride_kn", "stride_kk",  "stride_vk", "stride_vn",  "stride_om", "stride_on", "Z", "H"]) # , "N_CTX"
def _attn_fwd(Q, K, V, sm_scale: tl.constexpr, M, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kn, stride_kk,  #
              stride_vk, stride_vn,  #
              stride_om, stride_on,  #
              Z, H, N_CTX: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):


    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0) # walks along the sequence dimension in blocks of BLOCK_M rows of Q/K/V/Output.
    off_z = tl.program_id(1) # batch
    off_h = tl.program_id(2) # head
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)

    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                    start_m, qk_scale,  #
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                    4 - STAGE, offs_m, offs_n, N_CTX
                                    )

    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]

    # question-now: correct change?
    m_ptrs = M + off_z*off_h * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


BLOCK_M = 16
BLOCK_N = 16

def stub(q, k, v, causal, sm_scale):
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    o = torch.empty_like(q)
    stage = 3 if causal else 1
    print("stage: ", stage)

    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0], q.shape[1])
    print("grid: ", grid)

    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    _attn_fwd[grid](
        q, k, v, sm_scale, M, o,  #
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        # k.stride(0), k.stride(1),
        k.stride(2), k.stride(3),  #
        # v.stride(0), v.stride(1),
        v.stride(2), v.stride(3),  #
        # o.stride(0), o.stride(1),
        o.stride(2), o.stride(3),  #

        q.shape[0],  # Z
        q.shape[1],  # H
        N_CTX=q.shape[2],  #
        HEAD_DIM=HEAD_DIM_K,  #
        STAGE=stage,  #
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return o


# NOTE:
#   this test was part of the official tutorial so I included it here as well;
#   my actual forward / backward test is in run.py
@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM", [(1, 2, 128, 64)])
@pytest.mark.parametrize("causal", [True])
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype=torch.float16):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    dout = torch.randn_like(q)

    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)

    # todo: commented this out bc I'm using smaller shapes now but the bwd
    #   has this assert (requires a larger shape)
    # PRE_BLOCK = 128
    # assert N_CTX % PRE_BLOCK == 0

    tri_out = stub(q, k, v, causal, sm_scale).half()
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)

    # ref_out.backward(dout)
    # tri_out.backward(dout)

    # rtol = 0.0
    # Relative tolerance workaround for known hardware limitation of MI200 GPU.
    # For details see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    # if torch.version.hip is not None and triton.runtime.driver.active.get_current_target().arch == "gfx90a":
    #     rtol = 1e-2
    # assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=rtol)
    # assert torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=rtol)
    # assert torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=rtol)

    return tri_out, ref_out

# todo: works but unrolls for loop too many -- so try smaller shapes
# test_op(256, 64, 128, 64, causal=True)

test_op(256, 64, 32, 16, causal=False)
