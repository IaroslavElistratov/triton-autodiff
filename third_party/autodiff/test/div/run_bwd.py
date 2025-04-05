import numpy as np
import torch
DEVICE = torch.device("cuda:0")

import triton
from triton.compiler import compile
from triton.backends.compiler import GPUTarget


target = GPUTarget("cuda", arch=89, warp_size=32)
bwd_kernel = compile("out.ttir", target=target)

def bwd(a, b, upstream, BLOCK_SIZE=4):
    grid = (1, 1, 1)
    out = torch.empty_like(a)
    grad_a = torch.zeros_like(a)
    grad_b = torch.zeros_like(b)
    out_grad = upstream
    _compiled_kernel = bwd_kernel[grid](a, b, out, grad_a, grad_b, out_grad)
    return _compiled_kernel, grad_a, grad_b

np_a = np.array([0.3990, 0.5167, 0.0249, 0.9401])
np_b = np.array([0.9722, 0.7910, 0.4690, 0.3300])

a = torch.from_numpy(np_a).to(dtype=torch.float32, device='cuda:0')
b = torch.from_numpy(np_b).to(dtype=torch.float32, device='cuda:0')

upstream = torch.ones_like(a)
_compiled_kernel, grad_a, grad_b = bwd(a, b, upstream)
print("grad a: ", grad_a)
print("grad b: ", grad_b)
print()

# compare with pytorch

torch_a = torch.from_numpy(np_a).to(device='cuda:0')
torch_a.requires_grad = True
torch_b = torch.from_numpy(np_b).to(device='cuda:0')
torch_b.requires_grad = True

torch_output = torch_a / torch_b
torch_output.backward(torch.ones_like(torch_output))

print("torch grad a: ", torch_a.grad)
print("torch grad b: ", torch_b.grad)
print()


if torch.allclose(grad_a, torch_a.grad.to(dtype=torch.float32), atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(grad_b, torch_b.grad.to(dtype=torch.float32), atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
