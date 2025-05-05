import os
os.environ['TRITON_ALWAYS_COMPILE']='1'


import torch

import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")


@triton.jit
def kernel(a_ptr,
               b_ptr,
               output_1_ptr,
               output_2_ptr,
               ):
    offsets = tl.arange(0, 4)

    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)

    _1 = a * b

    # branch 1
    _2 = _1 + 2
    tl.store(output_1_ptr + offsets, _2) # _3

    # branch 2
    _4 = _1 - 4
    # _5 = _4 / 2.0
    tl.store(output_2_ptr + offsets, _4) # _6

def stub(a, b):
    output_1 = torch.empty_like(a)
    output_2 = torch.empty_like(a)
    grid = (1, 1, 1)
    _compiled_kernel = kernel[grid](a, b, output_1, output_2)
    return _compiled_kernel, output_1, output_2

torch.manual_seed(0)

# output_torch = a / b
size = 4
a = torch.rand(size, device=DEVICE)
b = torch.rand(size, device=DEVICE)
_compiled_kernel, output_1, output_2 = stub(a, b)
# max_difference = torch.max(torch.abs(output_torch - output_triton))
# assert max_difference == 0.0

# print(output_torch)
# print(output_triton)
# print(f'The maximum difference between torch and triton is '
#       f'{max_difference}')

with open("inp.ttir", "w") as f:
    f.write(_compiled_kernel.asm['ttir'])
