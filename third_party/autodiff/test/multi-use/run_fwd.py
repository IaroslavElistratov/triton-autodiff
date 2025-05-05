import os
os.environ['TRITON_ALWAYS_COMPILE']='1'


import torch

import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")


# @triton.jit
# def kernel(a_ptr,
#             output_ptr,
#             BLOCK_SIZE: tl.constexpr,
#           ):
#     offsets = tl.arange(0, BLOCK_SIZE)

#     a = tl.load(a_ptr + offsets)

#     x = a + 0.5
#     y = a * x
#     z = a * y

#     tl.store(output_ptr + offsets, z)


@triton.jit
def kernel(a_ptr,
            output_ptr,
            BLOCK_SIZE: tl.constexpr,
          ):
    offsets = tl.arange(0, BLOCK_SIZE)

    a = tl.load(a_ptr + offsets)

    x = a + 0.5
    y = a * x

    tl.store(output_ptr + offsets, y)

def stub(a):
    output = torch.empty_like(a)
    assert a.device == DEVICE and output.device == DEVICE
    _compiled_kernel = kernel[(1, 1, 1)](a, output, BLOCK_SIZE=4)
    return output, _compiled_kernel

torch.manual_seed(0)
size = 4
a = torch.rand(size, device=DEVICE)
# print("a: ", a)

output_torch = (a + 0.5) * a
output_triton, _compiled_kernel = stub(a)
max_difference = torch.max(torch.abs(output_torch - output_triton))
assert max_difference == 0.0

# print(output_torch)
# print(output_triton)
# print(f'The maximum difference between torch and triton is '
#       f'{max_difference}')


with open("inp.ttir", "w") as f:
    f.write(_compiled_kernel.asm['ttir'])
