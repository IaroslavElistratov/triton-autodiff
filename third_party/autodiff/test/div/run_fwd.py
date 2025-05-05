import os
os.environ['TRITON_ALWAYS_COMPILE']='1'


import torch

import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")


@triton.jit
def kernel(
        a_ptr,
        b_ptr,
        output_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
    offsets = tl.arange(0, BLOCK_SIZE)

    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)

    z = a / b

    tl.store(output_ptr + offsets, z)

def stub(a, b):
    output = torch.empty_like(a)
    grid = lambda meta: (triton.cdiv(output.numel(), meta['BLOCK_SIZE']), )
    _compiled_kernel = kernel[grid](a, b, output, BLOCK_SIZE=4)
    return output, _compiled_kernel

torch.manual_seed(0)
size = 4
a = torch.rand(size, device=DEVICE)
b = torch.rand(size, device=DEVICE)
# print("a: ", a)
# print("b: ", b)
# print()

output_torch = a / b
output_triton, _compiled_kernel = stub(a, b)
max_difference = torch.max(torch.abs(output_torch - output_triton))
assert max_difference == 0.0

# print(output_torch)
# print(output_triton)
# print(f'The maximum difference between torch and triton is '
#       f'{max_difference}')

with open("inp.ttir", "w") as f:
  f.write(_compiled_kernel.asm['ttir'])
