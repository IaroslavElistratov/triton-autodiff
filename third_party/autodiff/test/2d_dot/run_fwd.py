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
    ):
    offsets = tl.arange(0, 16)

    offsets_2d = (16 * offsets[:, None]) + offsets[None, :]

    a = tl.load(a_ptr + offsets_2d) # , mask=offsets<16
    b = tl.load(b_ptr + offsets_2d) # , mask=offsets<16

    l = tl.dot(a, b)
    tl.store(output_ptr + offsets_2d, l) # , mask=offsets<16

def stub(a, b):
    output = torch.empty(a.shape[0], b.shape[1]).to(dtype=torch.float32, device='cuda:0')
    # grid = lambda meta: (triton.cdiv(output.numel(), meta['BLOCK_SIZE']), )
    grid = (1, 1, 1)
    _compiled_kernel = kernel[grid](a, b, output) # BLOCK_SIZE=4
    return output, _compiled_kernel

torch.manual_seed(0)
# fits in a single block -- less complex kernel (and thus the IR) bc not computing idx using block_size and pid in this case;
# Also, bc it's exactly the size of the block -- no need to add masks when loading -- further simplifies loading
size = 16
a = torch.rand((size, size), device=DEVICE).to(dtype=torch.float32, device='cuda:0')
b = torch.rand((size, size), device=DEVICE).to(dtype=torch.float32, device='cuda:0')

def torch_fn(a, b):
    return torch.matmul(a, b)


output_torch = torch_fn(a, b)
output_triton, _compiled_kernel = stub(a, b)
# print("output_torch:", output_torch[:3, :3])
# print("output_triton:", output_triton[:3, :3])
max_difference = torch.max(torch.abs(output_torch - output_triton))
# print(f'The maximum difference between torch and triton is '
#       f'{max_difference}')
assert max_difference == 0.0

with open("inp.ttir", "w") as f:
    f.write(_compiled_kernel.asm['ttir'])
