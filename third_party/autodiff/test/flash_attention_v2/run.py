
import numpy as np
import torch
import triton
# torch.set_printoptions(sci_mode=False, linewidth=1000)

# comment: at import this also runs asserts comparing fwd out with torch's
from utils import kernel, stub, BLOCK_M


torch.manual_seed(20)
DEVICE = torch.device("cuda:0")


# comment: M tensor is float32 -- while others are float16

B=1
NUM_HEADS=1
SEQ_LEN=32
HEAD_DIM=16
dtype=torch.float16

causal=False
sm_scale=0.5

q = torch.empty((B, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
k = torch.empty((B, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
v = torch.empty((B, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)

grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)



# reference implementation

def torch_fn(q, k, v, causal, sm_scale):
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



#### test forward ####
print("\n" * 4, "forward:")
output_torch = torch_fn(q, k, v, causal, sm_scale)
output_triton = stub(kernel, q, k, v, causal, sm_scale)
max_difference = torch.max(torch.abs(output_torch - output_triton))

print("output_torch:", output_torch[0, 0, :4, :4])
print("output_triton:", output_triton[0, 0, :4, :4])
print(f'The maximum difference between torch and triton is '
      f'{max_difference}')

# assert max_difference == 0.0
if torch.allclose(output_torch, output_triton, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")



#### test backward ####
print("\n" * 4, "backward warmup:")

upstream = torch.randn_like(q)
q.requires_grad = True
k.requires_grad = True
v.requires_grad = True

from triton.backends.autodiff import autodiff


# todo: need to support passing no grad args (in this case "causal, sm_scale") -- and my torch.Function needs to know to not output derivatives wrt them
# my_out = my_op(q, k, v, causal, sm_scale)
from functools import partial
# here binding trailing arguments (not arguments at the beginning), bc kernel happen to have these at the end
def right_partial(func, *args):
    return lambda *fargs: func(*fargs, *args)
stub = right_partial(stub, causal, sm_scale)

# now this uses the stub which only needs grad args (non grad args have been bound)
my_op, bwd_kernel = autodiff(kernel, stub, grid, idx_upstream=5, non_stub_args_idxs=[3])

# todo: rm warmup
stub(bwd_kernel, q, k, v)

print("\n" * 4, "backward:")

my_out = my_op(q, k, v)
my_out.backward(upstream)
print("grad q[0, 0, :4, :4]: ", q.grad[0, 0, :4, :4])
print("grad k[0, 0, :4, :4]: ", k.grad[0, 0, :4, :4])
print("grad v[0, 0, :4, :4]: ", v.grad[0, 0, :4, :4])
print()




# # NOTE: this works but unrolls for loop too many times -- so try smaller shapes
# # tri_out, ref_out = test_op(1, 1, 128, 64, True)

# out, grad_q, grad_k, grad_v, grad_m = test_op(dout, q, k, v, causal, sm_scale)
# print("out", out[0, 0, :4, :4])
# print("grad_q", grad_q[0, 0, :4, :4])
# print("grad_k", grad_k[0, 0, :4, :4])
# print("grad_v", grad_v[0, 0, :4, :4])


torch_q = q.clone().detach().requires_grad_(True)
torch_k = k.clone().detach().requires_grad_(True)
torch_v = v.clone().detach().requires_grad_(True)

torch_out = torch_fn(torch_q, torch_k, torch_v, causal, sm_scale)
torch_out.backward(upstream)

# print("torch_out", torch_out[0, 0, :4, :4])
print("torch_grad_q", torch_q.grad[0, 0, :4, :4])
print("torch_grad_k", torch_k.grad[0, 0, :4, :4])
print("torch_grad_v", torch_v.grad[0, 0, :4, :4])

# torch_dv, v.grad = v.grad.clone(), None
# torch_dk, k.grad = k.grad.clone(), None
# torch_dq, q.grad = q.grad.clone(), None


# # triton implementation
# tri_out = attention(q, k, v, causal, sm_scale).half()

# assert torch.allclose(torch_out, tri_out, atol=1e-2, rtol=0)
# assert torch.allclose(torch_dv, tri_dv, atol=1e-2, rtol=0.0)
# assert torch.allclose(torch_dk, tri_dk, atol=1e-2, rtol=0.0)
# assert torch.allclose(torch_dq, tri_dq, atol=1e-2, rtol=0.0)
# return tri_out, torch_out


rtol = 0.1
if torch.allclose(my_out, torch_out, atol=1e-2, rtol=rtol):
    print("✅ [out] Triton and Torch match")
else:
    print("❌ out Triton and Torch differ")

if torch.allclose(q.grad, torch_q.grad, atol=1e-2, rtol=rtol):
    print("✅ [q grad] Triton and Torch match")
else:
    print("❌ [q grad] Triton and Torch differ")

if torch.allclose(k.grad, torch_k.grad, atol=1e-2, rtol=rtol):
    print("✅ [k grad] Triton and Torch match")
else:
    print("❌ [k grad] Triton and Torch differ")

if torch.allclose(v.grad, torch_v.grad, atol=1e-2, rtol=rtol):
    print("✅ [v grad] Triton and Torch match")
else:
    print("❌ [v grad] Triton and Torch differ")

