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
    out = torch.empty_like(a)
    grad_a = torch.zeros_like(a)
    grad_b = torch.zeros_like(b)
    grad_c = torch.zeros_like(c)
    grad_out = upstream
    _compiled_kernel = bwd_kernel[grid](a, b, c, out, grad_a, grad_b, grad_c, grad_out)
    return _compiled_kernel, grad_a, grad_b, grad_c

np_a = np.array([ 0.7578,  2.1274, -0.5905,  0.8350])
np_b = np.array([ 0.6962,  0.5211,  1.3674, -0.7933])
np_c = np.array([ 0.9104,  1.3097, -0.1426, -1.3289])


a = torch.from_numpy(np_a).to(dtype=torch.float32, device='cuda:0')
b = torch.from_numpy(np_b).to(dtype=torch.float32, device='cuda:0')
c = torch.from_numpy(np_c).to(dtype=torch.float32, device='cuda:0')


upstream = torch.ones_like(a)
_compiled_kernel, grad_a, grad_b, grad_c = bwd(a, b, c, upstream)
print("grad a: ", grad_a)
print("grad b: ", grad_b)
print("grad c: ", grad_c)
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
    e = torch.cos(z)
    q = torch.sin(e)
    u = torch.sqrt(q)
    t = torch.log(u)
    l = torch.exp(t)
    return l

torch_output = torch_fn(torch_a, torch_b, torch_c)
torch_output.backward(torch.ones_like(torch_output))
print("torch grad a: ", torch_a.grad)
print("torch grad b: ", torch_b.grad)
print("torch grad c: ", torch_c.grad)
print()


if torch.allclose(grad_a, torch_a.grad.to(dtype=torch.float32), atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(grad_b, torch_b.grad.to(dtype=torch.float32), atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(grad_c, torch_c.grad.to(dtype=torch.float32), atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")