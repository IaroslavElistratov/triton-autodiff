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
        c_ptr,
        output_ptr,
    ):
    offsets = tl.arange(0, 4)

    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)
    c = tl.load(c_ptr + offsets)

    x = a + 0.5
    y = x * b
    z = y / c
    e = tl.cos(z)
    q = tl.sin(e)
    u = tl.sqrt(q)
    t = tl.log(u)
    l = tl.exp(t)

    # todo-high: decomposes into multiple ops
    # o = tl.sigmoid(l)
    # o = tl.softmax(l)

    tl.store(output_ptr + offsets, l)

def stub(a, b, c):
    output = torch.empty_like(a)
    # grid = lambda meta: (triton.cdiv(output.numel(), meta['BLOCK_SIZE']), )
    grid = (1, 1, 1)
    _compiled_kernel = kernel[grid](a, b, c, output)
    return output, _compiled_kernel

torch.manual_seed(0)
size = 4
a = torch.rand(size, device=DEVICE)
b = torch.rand(size, device=DEVICE)
c = torch.rand(size, device=DEVICE)
# print("a: ", a)
# print("b: ", b)
# print("c: ", c)
# print()

def torch_fn(a, b, c):
    x = a + 0.5
    y = x * b
    z = y / c
    e = torch.cos(z)
    q = torch.sin(e)
    u = torch.sqrt(q)
    t = torch.log(u)
    l = torch.exp(t)
    # o = tl.sigmoid(l)
    return l


output_torch = torch_fn(a, b, c)
output_triton, _compiled_kernel = stub(a, b, c)
max_difference = torch.max(torch.abs(output_torch - output_triton))
assert max_difference == 0.0

# print(output_torch)
# print(output_triton)
# print(f'The maximum difference between torch and triton is '
#       f'{max_difference}')

with open("inp.ttir", "w") as f:
    f.write(_compiled_kernel.asm['ttir'])
