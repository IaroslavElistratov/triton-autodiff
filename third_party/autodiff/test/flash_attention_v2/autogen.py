import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd(Q, K, V, M, Out, stride_qz, stride_qh, stride_qm, stride_qk, stride_kn, stride_kk, stride_vk, stride_vn, stride_om, stride_on, Z, H, grad_Q, grad_K, grad_V, grad_M, grad_Out):
    fwd_off_hz = tl.program_id(axis=1)
    fwd_off_z = fwd_off_hz // H  # assumes non-negative
    fwd_qvk_offset = tl.cast(fwd_off_z, tl.int64)
    fwd_qvk_offset_1 = tl.cast(stride_qz, tl.int64)
    fwd_qvk_offset_2 = fwd_qvk_offset * fwd_qvk_offset_1
    fwd_off_h = fwd_off_hz % H  # assumes non-negative
    fwd_qvk_offset_3 = tl.cast(fwd_off_h, tl.int64)
    fwd_qvk_offset_4 = tl.cast(stride_qh, tl.int64)
    fwd_qvk_offset_5 = fwd_qvk_offset_3 * fwd_qvk_offset_4
    fwd_qvk_offset_6 = fwd_qvk_offset_2 + fwd_qvk_offset_5
    fwd_O_block_ptr = Out + fwd_qvk_offset_6
    fwd_O_block_ptr_1 = fwd_O_block_ptr + tl.zeros((16, 16), dtype=tl.int64)
    fwd_start_m = tl.program_id(axis=0)
    fwd_unnamed = 16
    fwd_Q_block_ptr = fwd_start_m * fwd_unnamed
    fwd_Q_block_ptr_1 = tl.cast(fwd_Q_block_ptr, tl.int64)
    fwd_offs_m = tl.arange(0, 16)
    fwd_q_1 = tl.cast(fwd_offs_m, tl.int64)
    fwd_q_2 = fwd_Q_block_ptr_1 + fwd_q_1
    fwd_q_3 = tl.expand_dims(fwd_q_2, axis=1)
    fwd_O_block_ptr_2 = tl.cast(stride_om, tl.int64)
    fwd_O_block_ptr_4 = fwd_q_3 * fwd_O_block_ptr_2
    fwd_q_4 = tl.expand_dims(fwd_q_1, axis=0)
    fwd_O_block_ptr_6 = tl.cast(stride_on, tl.int64)
    fwd_O_block_ptr_8 = fwd_q_4 * fwd_O_block_ptr_6
    fwd_O_block_ptr_10 = fwd_O_block_ptr_4 + fwd_O_block_ptr_8
    fwd_O_block_ptr_11 = fwd_O_block_ptr_1 + fwd_O_block_ptr_10
    fwd_Q_block_ptr_2 = Q + fwd_qvk_offset_6
    fwd_q_5 = fwd_Q_block_ptr_2 + tl.zeros((16, 16), dtype=tl.int64)
    fwd_Q_block_ptr_3 = tl.cast(stride_qm, tl.int64)
    fwd_q_7 = fwd_q_3 * fwd_Q_block_ptr_3
    fwd_Q_block_ptr_4 = tl.cast(stride_qk, tl.int64)
    fwd_q_10 = fwd_q_4 * fwd_Q_block_ptr_4
    fwd_q_12 = fwd_q_7 + fwd_q_10
    fwd_q_13 = fwd_q_5 + fwd_q_12
    fwd_q_14 = tl.load(fwd_q_13)
    fwd_K_block_ptr = K + fwd_qvk_offset_6
    fwd_k = fwd_K_block_ptr + tl.zeros((16, 16), dtype=tl.int64)
    fwd_k_1 = tl.expand_dims(fwd_q_1, axis=1)
    fwd_K_block_ptr_1 = tl.cast(stride_kk, tl.int64)
    fwd_k_3 = fwd_k_1 * fwd_K_block_ptr_1
    fwd_K_block_ptr_2 = tl.cast(stride_kn, tl.int64)
    fwd_k_6 = fwd_q_4 * fwd_K_block_ptr_2
    fwd_k_8 = fwd_k_3 + fwd_k_6
    fwd_k_9 = fwd_k + fwd_k_8
    fwd_k_10 = tl.load(fwd_k_9)
    fwd_unnamed_1 = tl.full((16, 16), 0.0, dtype=tl.float32)
    fwd_qk = tl.dot(fwd_q_14, fwd_k_10)
    fwd_unnamed_2 = tl.full((16, 16), 0.7213475108146667, dtype=tl.float32)
    fwd_qk_1 = fwd_qk * fwd_unnamed_2
    fwd__elementwise_max = tl.max(fwd_qk, axis=1)
    fwd_unnamed_3 = tl.full((16,), 0.7213475108146667, dtype=tl.float32)
    fwd_m_ij = fwd__elementwise_max * fwd_unnamed_3
    fwd_unnamed_4 = tl.full((16,), float('-inf'), dtype=tl.float32)
    fwd_m_ij_1 = tl.maximum(fwd_m_ij, fwd_unnamed_4)
    fwd_qk_2 = tl.expand_dims(fwd_m_ij_1, axis=1)
    fwd_qk_4 = fwd_qk_1 - fwd_qk_2
    fwd_p = tl.exp2(fwd_qk_4)
    fwd_p_1 = tl.cast(fwd_p, tl.float16)
    fwd_V_block_ptr = V + fwd_qvk_offset_6
    fwd_v = fwd_V_block_ptr + tl.zeros((16, 16), dtype=tl.int64)
    fwd_V_block_ptr_1 = tl.cast(stride_vk, tl.int64)
    fwd_v_2 = fwd_k_1 * fwd_V_block_ptr_1
    fwd_V_block_ptr_2 = tl.cast(stride_vn, tl.int64)
    fwd_v_5 = fwd_q_4 * fwd_V_block_ptr_2
    fwd_v_7 = fwd_v_2 + fwd_v_5
    fwd_v_8 = fwd_v + fwd_v_7
    fwd_v_9 = tl.load(fwd_v_8)
    fwd_alpha = fwd_unnamed_4 - fwd_m_ij_1
    fwd_alpha_1 = tl.exp2(fwd_alpha)
    fwd_acc = tl.expand_dims(fwd_alpha_1, axis=1)
    fwd_unnamed_5 = tl.full((16, 1), 0.0, dtype=tl.float32)
    fwd_acc_1 = fwd_acc * fwd_unnamed_5
    fwd_acc_3 = tl.dot(fwd_p_1, fwd_v_9) + fwd_acc_1
    fwd__sum_combine = tl.sum(fwd_p, axis=1)
    fwd_l_i = fwd_alpha_1 + fwd__sum_combine
    fwd_acc_4 = tl.expand_dims(fwd_l_i, axis=1)
    fwd_acc_6 = fwd_acc_3 / fwd_acc_4
    fwd_element_ty = tl.cast(fwd_acc_6, tl.float16)
    tl.store(fwd_O_block_ptr_11, fwd_element_ty)
    fwd_m_ptrs = fwd_off_hz * fwd_unnamed
    fwd_m_ptrs_1 = M + tl.cast(fwd_m_ptrs, tl.int64)
    fwd_m_ptrs_2 = fwd_m_ptrs_1 + tl.zeros((16,), dtype=tl.int64)
    fwd_offs_m_2 = fwd_Q_block_ptr + fwd_offs_m
    fwd_m_ptrs_3 = fwd_m_ptrs_2 + tl.cast(fwd_offs_m_2, tl.int64)
    fwd_m_i = tl.log2(fwd_l_i)
    fwd_m_i_1 = fwd_m_ij_1 + fwd_m_i
    tl.store(fwd_m_ptrs_3, fwd_m_i_1)

    # ~~~~~~~~~~ grad branch for {grad_Q,grad_K} ~~~~~~~~~~
    fwd_m_ptrs_4 = grad_M + tl.cast(fwd_m_ptrs, tl.int64)
    fwd_m_ptrs_5 = fwd_m_ptrs_4 + tl.zeros((16,), dtype=tl.int64)
    fwd_m_ptrs_6 = fwd_m_ptrs_5 + tl.cast(fwd_offs_m_2, tl.int64)
    bwd_m_i = tl.load(fwd_m_ptrs_6)

    # ~~~~~~~~~~ grad branch for {grad_Q,grad_K,grad_V} ~~~~~~~~~~
    fwd_O_block_ptr_12 = grad_Out + fwd_qvk_offset_6
    fwd_O_block_ptr_13 = fwd_O_block_ptr_12 + tl.zeros((16, 16), dtype=tl.int64)
    fwd_O_block_ptr_14 = fwd_O_block_ptr_13 + fwd_O_block_ptr_10
    bwd_O_block_ptr = tl.load(fwd_O_block_ptr_14)

    # local grads for fwd_element_ty
    bwd_element_ty = tl.cast(bwd_O_block_ptr, tl.float32)

    # local grads for fwd_acc_6
    bwd_acc_3 = (1.0 / fwd_acc_4) * bwd_element_ty

    # local grads for fwd_acc_3
    bwd_acc_4 = tl.cast(bwd_acc_3, tl.float16)

    # ~~~~~~~~~~ grad branch for {grad_Q,grad_K} ~~~~~~~~~~
    bwd_acc_8 = tl.dot(bwd_acc_4, tl.trans(fwd_v_9))

    # local grads for fwd_p_1
    bwd_p = tl.cast(bwd_acc_8, tl.float32)

    # ~~~~~~~~~~ grad branch for grad_V ~~~~~~~~~~

    # local grads for fwd_acc_3
    bwd_acc_12 = tl.dot(tl.trans(fwd_p_1), bwd_acc_4)

    # ~~~~~~~~~~ grad branch for {grad_Q,grad_K} ~~~~~~~~~~

    # local grads for fwd_acc_1
    bwd_acc_15 = tl.expand_dims(tl.sum(bwd_acc_3, axis=1), axis=1)
    bwd_acc_16 = fwd_acc * bwd_acc_15
    bwd_acc_17 = fwd_unnamed_5 * bwd_acc_15

    # local grads for fwd_acc_6
    bwd_acc_24 = (-1.0 * (fwd_acc_3 / (fwd_acc_4 * fwd_acc_4))) * bwd_element_ty

    # local grads for fwd_acc_4
    bwd_acc_27 = tl.expand_dims(tl.sum(bwd_acc_24, axis=1), axis=1)
    bwd_acc_28 = tl.reshape(bwd_acc_27, (16,))

    # local grads for fwd_m_i
    bwd_acc_29 = bwd_acc_28 + ((1.0 / (fwd_l_i * 0.6931470036506653)) * bwd_m_i)

    # local grads for fwd_acc
    bwd_acc_30 = bwd_acc_29 + tl.reshape(bwd_acc_17, (16,))

    # local grads for fwd_alpha_1
    bwd_alpha_3 = (0.6931470036506653 * fwd_alpha_1) * bwd_acc_30

    # local grads for fwd_alpha
    bwd_m_i_8 = bwd_m_i + (bwd_alpha_3 * -1.0)

    # local grads for fwd__sum_combine
    bwd_p_1 = bwd_p + tl.expand_dims(bwd_acc_29, axis=1)

    # local grads for fwd_p
    bwd_p_5 = (0.6931470036506653 * fwd_p) * bwd_p_1

    # local grads for fwd_qk_4
    bwd_qk_2 = bwd_p_5 * -1.0

    # local grads for fwd_qk_2
    bwd_qk_5 = tl.expand_dims(tl.sum(bwd_qk_2, axis=1), axis=1)
    bwd_m_i_9 = bwd_m_i_8 + tl.reshape(bwd_qk_5, (16,))

    # local grads for fwd_m_ij_1
    bwd_m_ij_3 = 1.0
    bwd_m_ij_5 = 0.0
    bwd_m_ij_9 = tl.where((fwd_m_ij >= fwd_unnamed_4), bwd_m_ij_3, bwd_m_ij_5) * bwd_m_i_9

    # local grads for fwd_m_ij
    bwd_m_ij_10 = fwd__elementwise_max * bwd_m_ij_9
    bwd_m_ij_11 = fwd_unnamed_3 * bwd_m_ij_9

    # local grads for fwd__elementwise_max
    bwd__elementwise_max_10 = tl.where((fwd_qk == tl.expand_dims(fwd__elementwise_max, axis=1)), 1.0, 0.0) * tl.expand_dims(bwd_m_ij_11, axis=1)

    # local grads for fwd_m_ij_1
    bwd_alpha_7 = bwd_alpha_3 + (tl.where((fwd_unnamed_4 > fwd_m_ij), bwd_m_ij_3, bwd_m_ij_5) * bwd_m_i_9)

    # local grads for fwd_qk_1
    bwd_qk_7 = fwd_qk * bwd_p_5
    bwd_qk_8 = fwd_unnamed_2 * bwd_p_5

    # local grads for fwd__elementwise_max
    bwd_qk_9 = bwd_qk_8 + bwd__elementwise_max_10

    # local grads for fwd_qk
    bwd_qk_10 = tl.cast(bwd_qk_9, tl.float16)

    # ~~~~~~~~~~ grad branch for grad_Q ~~~~~~~~~~
    bwd_qk_14 = tl.dot(bwd_qk_10, tl.trans(fwd_k_10))

    # ~~~~~~~~~~ grad branch for grad_K ~~~~~~~~~~
    bwd_qk_18 = tl.dot(tl.trans(fwd_q_14), bwd_qk_10)

    # ~~~~~~~~~~ grad branch for grad_Q ~~~~~~~~~~
    fwd_Q_block_ptr_5 = grad_Q + fwd_qvk_offset_6
    fwd_q_15 = fwd_Q_block_ptr_5 + tl.zeros((16, 16), dtype=tl.int64)
    fwd_q_16 = fwd_q_15 + fwd_q_12

    # local grads for fwd_q_14
    bwd_Q_block_ptr_1 = tl.atomic_add(fwd_q_16, tl.cast(bwd_qk_14, tl.float16))

    # ~~~~~~~~~~ grad branch for grad_K ~~~~~~~~~~
    fwd_K_block_ptr_3 = grad_K + fwd_qvk_offset_6
    fwd_k_11 = fwd_K_block_ptr_3 + tl.zeros((16, 16), dtype=tl.int64)
    fwd_k_12 = fwd_k_11 + fwd_k_8

    # local grads for fwd_k_10
    bwd_k = tl.atomic_add(fwd_k_12, tl.cast(bwd_qk_18, tl.float16))

    # ~~~~~~~~~~ grad branch for grad_V ~~~~~~~~~~
    fwd_V_block_ptr_3 = grad_V + fwd_qvk_offset_6
    fwd_v_10 = fwd_V_block_ptr_3 + tl.zeros((16, 16), dtype=tl.int64)
    fwd_v_11 = fwd_v_10 + fwd_v_7

    # local grads for fwd_v_9
    bwd_v = tl.atomic_add(fwd_v_11, tl.cast(bwd_acc_12, tl.float16))



# commenting manually
# N_CTX = 128
# N_CTX = (stride_qh // stride_qm)
# fwd_m_ptrs = (fwd_off_hz * N_CTX)


BLOCK_M = 16
BLOCK_N = 16




def stub(q, k, v, causal, sm_scale, upstream):
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

    grad_q = torch.zeros_like(q)
    grad_k = torch.zeros_like(k)
    grad_v = torch.zeros_like(v)
    grad_o = upstream


    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    grad_m = torch.zeros_like(M)
    _attn_fwd[grid](
        q, k, v,
        M, o,  #
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        k.stride(2), k.stride(3),  #
        v.stride(2), v.stride(3),  #
        o.stride(2), o.stride(3),  #

        q.shape[0],  # Z
        q.shape[1],  # H
        # N_CTX=q.shape[2],  #
        # HEAD_DIM=HEAD_DIM_K,  #
        # STAGE=stage,  #
        # BLOCK_M=BLOCK_M,
        # BLOCK_N=BLOCK_N,

        grad_q,  # arg17
        grad_k, # arg18
        grad_v, # arg19
        # question-now: not sure: m at this postion?
        grad_m,  # arg20
        grad_o, # arg21
    )

    return grad_q, grad_k, grad_v


#   my actual forward / backward test is in run.py
# Z = 1
# H = 2
# N_CTX = 16
# HEAD_DIM = 64

# answer-now:MUST be same as fwd shapes! bc the kernel is speciazed
Z = 256
H = 64
N_CTX = 16
HEAD_DIM = 16

causal=False
dtype=torch.float16
DEVICE = torch.device("cuda:0")

torch.manual_seed(20)
q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
sm_scale = 0.5
upstream = torch.randn_like(q)

torch_q = q.clone().detach().requires_grad_(True)
torch_k = k.clone().detach().requires_grad_(True)
torch_v = v.clone().detach().requires_grad_(True)
torch_upstream = upstream.clone().detach()


# reference implementation
def torch_fn(q, k, v):
    M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    return ref_out



grad_q, grad_k, grad_v = stub(q, k, v, causal, sm_scale, upstream)

print("grad_q", grad_q[0, 0, :5, :5])
print("grad_k", grad_k[0, 0, :5, :5])
print("grad_v", grad_v[0, 0, :5, :5])


_torch_output = torch_fn(torch_q, torch_k, torch_v)
_torch_output.backward(torch_upstream)

print("torch_q.grad", torch_q.grad[0, 0, :5, :5])
print("torch_k.grad", torch_k.grad[0, 0, :5, :5])
print("torch_v.grad", torch_v.grad[0, 0, :5, :5])



if torch.allclose(grad_q, torch_q.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(grad_k, torch_k.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(grad_v, torch_v.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")