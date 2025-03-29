import numpy as np
import torch
DEVICE = torch.device("cuda:0")

import triton
from triton.compiler import compile
from triton.backends.compiler import GPUTarget


target = GPUTarget("cuda", arch=89, warp_size=32)
bwd_kernel = compile("out.ttir", target=target)

def bwd(a, b, c, upstream):
    grid = (1, 1, 1)
    return bwd_kernel[grid](a, b, c, upstream)

np_a = np.array([ 0.7578,  2.1274, -0.5905,  0.8350])
np_b = np.array([ 0.6962,  0.5211,  1.3674, -0.7933])
np_c = np.array([ 0.9104,  1.3097, -0.1426, -1.3289])


a = torch.from_numpy(np_a).to(dtype=torch.float32, device='cuda:0')
b = torch.from_numpy(np_b).to(dtype=torch.float32, device='cuda:0')
c = torch.from_numpy(np_c).to(dtype=torch.float32, device='cuda:0')


upstream = torch.ones_like(a)
_compiled_kernel = bwd(a, b, c, upstream)
print("grad a: ", a)
print("grad b: ", b)
print("grad c: ", c)
print()

# compare with pytorch

torch_a = torch.from_numpy(np_a).to(device='cuda:0')
torch_a.requires_grad = True
torch_b = torch.from_numpy(np_b).to(device='cuda:0')
torch_b.requires_grad = True
torch_c = torch.from_numpy(np_c).to(device='cuda:0')
torch_c.requires_grad = True

def torch_fn(a, b, c):
    x = a + 0.5
    y = x * b
    z = y / c
    return z

torch_output = torch_fn(torch_a, torch_b, torch_c)
torch_output.backward(torch.ones_like(torch_output))
print("torch grad a: ", torch_a.grad)
print("torch grad b: ", torch_b.grad)
print("torch grad c: ", torch_c.grad)
print()


if torch.allclose(a, torch_a.grad.to(dtype=torch.float32), atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(b, torch_b.grad.to(dtype=torch.float32), atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(c, torch_c.grad.to(dtype=torch.float32), atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")