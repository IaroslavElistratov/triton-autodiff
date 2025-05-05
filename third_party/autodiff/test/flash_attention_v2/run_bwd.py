import numpy as np
import torch

torch.manual_seed(20)
DEVICE = torch.device("cuda:0")

import triton
from triton.compiler import compile
from triton.backends.compiler import GPUTarget


# Create a GPU target
target = GPUTarget("cuda", arch=89, warp_size=32)

# Compile the IR
bwd_kernel = compile("out.ttir", target=target)




def test_op(upstream, q, k, v, causal, sm_scale):

    ######## COPY FROM FWD ########

    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    o = torch.empty_like(q)
    stage = 3 if causal else 1
    print("stage: ", stage)
    extra_kern_args = {}


    grid = (1, q.shape[0] * q.shape[1], 1)

    BLOCK_M = 16
    BLOCK_N = 16

    # assert q.shape[2] / BLOCK_M == 1
    assert q.shape[2] == BLOCK_M
    # comment: this tensor is float32 -- while others are float16
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    ################################



    grad_q = torch.zeros_like(q)
    grad_k = torch.zeros_like(k)
    grad_v = torch.zeros_like(v)
    grad_m = torch.zeros_like(M)
    # note:
    grad_o = upstream

    # enqueue kernel
    compiled_kernel = bwd_kernel[grid](  #

        # fwd values:
        q,      # FLOAT PTR!
        k,      # FLOAT PTR!
        v,      # FLOAT PTR!
        # sm_scale,     # constexpr
        M,      # FLOAT PTR!
        o,      # FLOAT PTR!
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),

        # k.stride(0), k.stride(1), # not used in the fwd kernel
        k.stride(2), k.stride(3),

        # v.stride(0), v.stride(1), # not used in the fwd kernel
        v.stride(2), v.stride(3),

        # o.stride(0), o.stride(1), # not used in the fwd kernel
        o.stride(2), o.stride(3),

        q.shape[0], q.shape[1], q.shape[2],

        # HEAD_DIM=HEAD_DIM_K,  # constexpr
        # STAGE=stage,          # constexpr
        # BLOCK_M=BLOCK_M,      # constexpr
        # BLOCK_N=BLOCK_N       # constexpr

        # grads:
        grad_q,
        grad_k,
        grad_v,
        grad_m,
        grad_o,

    )
    return compiled_kernel, o, grad_q, grad_k, grad_v, grad_m




Z=1
H=2
N_CTX=16
HEAD_DIM=16
dtype=torch.float16

causal=False
sm_scale=0.5

q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)

# note:
# dout = torch.randn_like(q)
dout = torch.ones_like(q)






# my autodif

# NOTE: this works but unrolls for 128 times -- so try smaller shapes
# tri_out, ref_out = test_op(1, 1, 128, 64, True)

compiled_kernel, out, grad_q, grad_k, grad_v, grad_m = test_op(dout, q, k, v, causal, sm_scale)
print("out", out[0, 0, :4, :4])
print("grad_q", grad_q[0, 0, :4, :4])
print("grad_k", grad_k[0, 0, :4, :4])
print("grad_v", grad_v[0, 0, :4, :4])



# reference implementation

def torch_fn(q, k, v, causal, sm_scale):
    M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")

    # ORIGINAL:
    p = torch.softmax(p.float(), dim=-1).half()

    #  todo-now: using softmax wt min-max trick to correspond to the inp.ttir
    # # MY:
    # p = p.float()
    # exp = torch.exp(p)
    # p = (exp / exp.sum(dim=-1, keepdim=True)).half()

    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    return ref_out

q.requires_grad = True
k.requires_grad = True
v.requires_grad = True

ref_out = torch_fn(q, k, v, causal, sm_scale)
ref_out.backward(dout)

print("ref_out", ref_out[0, 0, :4, :4])
print("ref_grad_q", q.grad[0, 0, :4, :4])
print("ref_grad_k", k.grad[0, 0, :4, :4])
print("ref_grad_v", v.grad[0, 0, :4, :4])

# ref_dv, v.grad = v.grad.clone(), None
# ref_dk, k.grad = k.grad.clone(), None
# ref_dq, q.grad = q.grad.clone(), None


# # triton implementation
# tri_out = attention(q, k, v, causal, sm_scale).half()

# assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
# assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=0.0)
# assert torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=0.0)
# assert torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=0.0)
# return tri_out, ref_out


if torch.allclose(out, ref_out, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(grad_q, q.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(grad_k, k.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(grad_v, v.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")




"""


M = 1151
N = 8192
dtype = torch.float16
device = DEVICE

# create data
x_shape = (M, N)
w_shape = (x_shape[-1], )
weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
dy = .1 * torch.randn_like(x)
# x.requires_grad_(True)
# quantiles = [0.5, 0.2, 0.8]


# # Run our implementation
with torch.no_grad():
    compiled_kernel, grad_x_arg, grad_weight, grad_bias, grad_mean, grad_rstd = bwd(x, weight, bias)
    print("grad_x_arg: ", grad_x_arg)
    print("grad_weight: ", grad_weight)
    print("grad_bias: ", grad_bias)

print("\n" * 3)
# # Forward pass with PyTorch's LayerNorm
x.requires_grad = True
weight.requires_grad = True
bias.requires_grad = True
torch_ln = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps=1e-5)
torch_ln.backward(torch.ones_like(torch_ln))
print("x.grad: ", x.grad)
print("weight.grad: ", weight.grad)
print("bias.grad: ", bias.grad)
"""