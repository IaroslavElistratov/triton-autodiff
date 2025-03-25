import numpy as np

import torch
DEVICE = torch.device("cuda:0")

import triton
from triton.compiler import compile
from triton.backends.compiler import GPUTarget

# ir_code = """
# 
# """

# IRSource (created inside triton.compile) uses extension of the file to figure
# out to call "ir.parse_mlir_module(self.path, context)""

# with open("third_party/autodiff/test/kernel.ttir", "w") as f:
#     f.write(ir_code)

# Create a GPU target
target = GPUTarget("cuda", arch=89, warp_size=32)

# Compile the IR
add_bwd_kernel = compile("./out.ttir", target=target)

# The IRSource class handles parsing the IR file and setting up the compilation pipeline
# The rest of the compilation process (from IR to PTX to cubin) remains the same as the normal workflow
def add_bwd(a, upstream, BLOCK_SIZE=4):
    assert upstream.device == DEVICE and a.device == DEVICE
    grid = (1, 1, 1)
    return add_bwd_kernel[grid](a, upstream)

np_a = np.array([0.3990, 0.5167, 0.0249, 0.9401])

a = torch.from_numpy(np_a).to(dtype=torch.float32, device='cuda:0')
upstream = torch.ones(4, device=DEVICE)
_compiled_kernel = add_bwd(a, upstream)

# comment: grads have been written inplace of the original values
print("grad a: ", a)
print()

# compare with pytorch

torch_a = torch.from_numpy(np_a).to(device='cuda:0')
torch_a.requires_grad = True
torch_output = torch_a + 42.0
torch_output.backward(torch.ones_like(torch_output))

print("torch grad a: ", torch_a.grad)
print()


if torch.allclose(a, torch_a.grad.to(dtype=torch.float32), atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
