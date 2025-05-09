import numpy as np
import torch
import triton

# comment: at import this also runs asserts comparing fwd out with torch's
from utils import kernel, stub


torch.manual_seed(20)
DEVICE = torch.device("cuda:0")


# comment: M tensor is float32 -- while others are float16

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

grid = (1, q.shape[0] * q.shape[1], 1)



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



#### test forward ####

output_torch = torch_fn(q, k, v, causal, sm_scale)
output_triton = stub(q, k, v, causal, sm_scale)
max_difference = torch.max(torch.abs(output_torch - output_triton))

# print("output_torch:", output_torch[:3, :3])
# print("output_triton:", output_triton[:3, :3])
# print(f'The maximum difference between torch and triton is '
#       f'{max_difference}')

# assert max_difference == 0.0
if torch.allclose(output_torch, output_triton, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")



#### test backward ####

upstream = torch.randn_like(q)
q.requires_grad = True
k.requires_grad = True
v.requires_grad = True

from triton.backends.api import autodiff


# todo: need to support passing no grad args (in this case "causal, sm_scale") -- and my torch.Function needs to know to not output derivatives wrt them
# my_out = my_op(q, k, v, causal, sm_scale)
from functools import partial
# here binding trailing arguments (not begging arguments), bc kernel happen to have these at the end
def right_partial(func, *args):
    return lambda *fargs: func(*fargs, *args)
stub = right_partial(stub, causal, sm_scale))

# now this uses the stub which only needs grad args (non grad args have been bound)
my_op, bwd_kernel = autodiff(kernel, stub, idx_upstream=5)

# todo-now: to do the warmup you need to somehow get all the args -- this requires replicating logic of the enitre stub
# todo: rm warmup
bwd_kernel[grid](a, b, torch.ones_like(a))
my_out = my_op(q, k, v, causal, sm_scale)
my_out.backward(upstream)
print("grad a[:3, :3]: ", a.grad[:3, :3])
print("grad b[:3, :3]: ", b.grad[:3, :3])
print()


# # NOTE: this works but unrolls for 128 times -- so try smaller shapes
# # tri_out, ref_out = test_op(1, 1, 128, 64, True)

# out, grad_q, grad_k, grad_v, grad_m = test_op(dout, q, k, v, causal, sm_scale)
# print("out", out[0, 0, :4, :4])
# print("grad_q", grad_q[0, 0, :4, :4])
# print("grad_k", grad_k[0, 0, :4, :4])
# print("grad_v", grad_v[0, 0, :4, :4])




# q.requires_grad = True
# k.requires_grad = True
# v.requires_grad = True

# ref_out = torch_fn(q, k, v, causal, sm_scale)
# ref_out.backward(dout)

# print("ref_out", ref_out[0, 0, :4, :4])
# print("ref_grad_q", q.grad[0, 0, :4, :4])
# print("ref_grad_k", k.grad[0, 0, :4, :4])
# print("ref_grad_v", v.grad[0, 0, :4, :4])

# # ref_dv, v.grad = v.grad.clone(), None
# # ref_dk, k.grad = k.grad.clone(), None
# # ref_dq, q.grad = q.grad.clone(), None


# # # triton implementation
# # tri_out = attention(q, k, v, causal, sm_scale).half()

# # assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
# # assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=0.0)
# # assert torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=0.0)
# # assert torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=0.0)
# # return tri_out, ref_out


# if torch.allclose(out, ref_out, atol=1e-2, rtol=0):
#     print("✅ Triton and Torch match")
# else:
#     print("❌ Triton and Torch differ")

# if torch.allclose(grad_q, q.grad, atol=1e-2, rtol=0):
#     print("✅ Triton and Torch match")
# else:
#     print("❌ Triton and Torch differ")

# if torch.allclose(grad_k, k.grad, atol=1e-2, rtol=0):
#     print("✅ Triton and Torch match")
# else:
#     print("❌ Triton and Torch differ")

# if torch.allclose(grad_v, v.grad, atol=1e-2, rtol=0):
#     print("✅ Triton and Torch match")
# else:
#     print("❌ Triton and Torch differ")




# """


# M = 1151
# N = 8192
# dtype = torch.float16
# device = DEVICE

# # create data
# x_shape = (M, N)
# w_shape = (x_shape[-1], )
# weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
# bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
# x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
# dy = .1 * torch.randn_like(x)
# # x.requires_grad_(True)
# # quantiles = [0.5, 0.2, 0.8]


# # # Run our implementation
# with torch.no_grad():
#     compiled_kernel, grad_x_arg, grad_weight, grad_bias, grad_mean, grad_rstd = bwd(x, weight, bias)
#     print("grad_x_arg: ", grad_x_arg)
#     print("grad_weight: ", grad_weight)
#     print("grad_bias: ", grad_bias)

# print("\n" * 3)
# # # Forward pass with PyTorch's LayerNorm
# x.requires_grad = True
# weight.requires_grad = True
# bias.requires_grad = True
# torch_ln = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps=1e-5)
# torch_ln.backward(torch.ones_like(torch_ln))
# print("x.grad: ", x.grad)
# print("weight.grad: ", weight.grad)
# print("bias.grad: ", bias.grad)
# """