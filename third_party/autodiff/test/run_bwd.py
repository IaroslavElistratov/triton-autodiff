import numpy as np
import torch
DEVICE = torch.device("cuda:0")

import triton
from triton.compiler import compile
from triton.backends.compiler import GPUTarget


# Create a GPU target
target = GPUTarget("cuda", arch=89, warp_size=32)

# Compile the IR
bwd_kernel = compile("third_party/autodiff/test/out.ttir", target=target)

# The IRSource class handles parsing the IR file and setting up the compilation pipeline
# The rest of the compilation process (from IR to PTX to cubin) remains the same as the normal workflow
def bwd(a, b, upstream, BLOCK_SIZE=4):
    # We need to preallocate the output.
    # a_grad = torch.empty_like(upstream, device=upstream.device)
    # assert upstream.device == DEVICE and a_grad.device == DEVICE
    grid = (triton.cdiv(upstream.numel(), BLOCK_SIZE), 1, 1)
    # The kernel expects 3 arguments: two float pointers and an integer
    # comment: note this is entirely new approach: pass a, b (forward values), and upstream grad at the same time -- without duplciaitng numebr of forward args and passding forwaard args and buffers for bwd at the same time
    _compiled_kernel = bwd_kernel[grid](a, b, upstream) # BLOCK_SIZE
    # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    return _compiled_kernel

np_a = np.array([0.3990, 0.5167, 0.0249, 0.9401])
np_b = np.array([0.9722, 0.7910, 0.4690, 0.3300])

a = torch.from_numpy(np_a).to(device='cuda:0')
b = torch.from_numpy(np_b).to(device='cuda:0')

upstream = torch.ones(4, device=DEVICE)
# comment: grads have been written inplace of the original values
_compiled_kernel = bwd(a, b, upstream)
print("grad a: ", a)
print("grad b: ", b)
print()

# compare with pytorch

torch_a = torch.from_numpy(np_a).to(device='cuda:0')
torch_a.requires_grad = True
torch_b = torch.from_numpy(np_b).to(device='cuda:0')
torch_b.requires_grad = True

torch_output = (torch_a + 0.5) * torch_b
torch_output.backward(torch.ones_like(torch_output))

print("torch grad a: ", torch_a.grad)
print("torch grad b: ", torch_b.grad)
