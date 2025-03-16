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
add_bwd_kernel = compile("third_party/autodiff/test/out.ttir", target=target)

# The IRSource class handles parsing the IR file and setting up the compilation pipeline
# The rest of the compilation process (from IR to PTX to cubin) remains the same as the normal workflow
def add_bwd(upstream, BLOCK_SIZE=4):
    # We need to preallocate the output.
    a_grad = torch.empty_like(upstream, device=upstream.device)
    assert upstream.device == DEVICE and a_grad.device == DEVICE
    grid = (triton.cdiv(upstream.numel(), BLOCK_SIZE), 1, 1)
    # The kernel expects 3 arguments: two float pointers and an integer
    _compiled_kernel = add_bwd_kernel[grid](a_grad, upstream) # BLOCK_SIZE
    # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    return a_grad, _compiled_kernel
  
upstream = torch.ones(4, device=DEVICE)
a_grad, _compiled_kernel = add_bwd(upstream)
print(a_grad)