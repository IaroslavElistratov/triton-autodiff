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


configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64]\
    for s in ([3, 4, 7])\
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
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i



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
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
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

    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))




@autodiff(_attn_fwd, idxs_buffers=(4, 5))
def stub(q, k, v, causal=False, sm_scale=0.5, BLOCK_M=16, BLOCK_N=16):
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    o = torch.empty_like(q)
    stage = 3 if causal else 1
    print("stage: ", stage)

    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
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


def torch_fn(q, k, v, causal=False, sm_scale=0.5):
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale

    if causal:
        M = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device=DEVICE))
        p[:, :, M == 0] = float("-inf")

    # p = torch.softmax(p.float(), dim=-1).half()

    p = p.float()
    p = p - p.max(axis=3, keepdim=True)[0]
    p = torch.exp(p)
    p = (p / p.sum(dim=3, keepdim=True))
    p = p.half()

    ref_out = torch.matmul(p, v)
    return ref_out





SWEEP = [
    {"B": 256, "NUM_HEADS": 64, "SEQ_LEN": N, "HEAD_DIM": 16, "causal": False, "sm_scale": 0.5}
    for N in (16, 128, 256, 512, 1024, 2048, 4096)
]

# todo-now:
# becuase my naive autodiff unrolls, the first dim should be extremely small
# in fact it should be one because otherwise confuses the LLM

def make_args(dims, device="cuda", dtype=torch.float16):
    B, NUM_HEADS, SEQ_LEN, HEAD_DIM = dims["B"], dims["NUM_HEADS"], dims["SEQ_LEN"], dims["HEAD_DIM"]
    q = torch.empty((B, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    k = torch.empty((B, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    v = torch.empty((B, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    # todo: add support for python types args in gradcheck
    # return (q, k, v, dims["causal"], dims["sm_scale"]), {}
    return (q, k, v), {}

# optional
def flops(dims, mode):
    B, H, N, D = dims["Z"], dims["H"], dims["N_CTX"], dims["HEAD_DIM"]
    total = 2.0 * (2.0 * B * H * N * N * D)  # forward+backward baseline
    if dims.get("causal", False):
        total *= 0.5
    return total



def setup():
    # q = torch.empty((B, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    # k = torch.empty((B, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    # v = torch.empty((B, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    (q, k, v), _ = make_args(SWEEP[0])
    stub(q, k, v)

