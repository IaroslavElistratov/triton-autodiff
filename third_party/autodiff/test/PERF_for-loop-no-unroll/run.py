import os
os.environ['TRITON_ALWAYS_COMPILE']='1'

import torch
import triton
import triton.language as tl

from triton.backends.autodiff import autodiff

torch.manual_seed(0)
DEVICE = torch.device("cuda:0")


#################
#    Step 1     #
#################

# @triton.jit
# def kernel_v1(
#       a_ptr,
#       b_ptr,
#       output_ptr,
#       BLOCK_SIZE: tl.constexpr,
#     ):
#     offsets = tl.arange(0, BLOCK_SIZE) # (4, )

#     accum = tl.load(output_ptr + offsets) # (4, )

#     # num_iters -- num elements (8), elements-per iter (4)
#     for i in range(2):
#       a = tl.load(a_ptr + offsets) # (4, )
#       b = tl.load(b_ptr + offsets) # (4, )
#       out = a * b

#       # NOTE: make this "offsets" be the iter arg
#       tl.store(output_ptr + offsets, out)
#       # shift offsets by 4
#       offsets += 4


# COMPILED_KERNEL = None

# def stub(a, b):
#     output = torch.zeros_like(a)
#     global COMPILED_KERNEL
#     COMPILED_KERNEL = kernel_v1[(1, )](a, b, output, BLOCK_SIZE=4)
#     return output


# def torch_fn(torch_a, torch_b):
#     return torch_a * torch_b

# with open("inp.ttir", "w") as f:
#     f.write(COMPILED_KERNEL.asm['ttir'])



#################
#    Step 2     #
#################
# comment:
#   Now there's a dependency between iterations -- they write to the same accum
#   That's close to what flash-attention kernel does

@autodiff(
    # pattern="elementwise-like",
    idxs_buffers=(2, )
)
@triton.jit
def kernel_v2(
      a_ptr,
      b_ptr,
      output_ptr,
      BLOCK_SIZE: tl.constexpr,
    ):

    offsets = tl.arange(0, BLOCK_SIZE) # (4, )

    accum = tl.load(output_ptr + offsets) # (4, )

    for i in range(2):
      a = tl.load(a_ptr + offsets) # (4, )
      b = tl.load(b_ptr + offsets) # (4, )

      out = a * b

      # accum into the same buffer
      accum += out

      offsets += 4

    # create new offsets overwise previous was overwritten inside the loop
    offsets = tl.arange(0, BLOCK_SIZE) # (4, )
    tl.store(output_ptr + offsets, accum)

def stub(a, b):
    output = torch.zeros((4, ), device=a.device)
    kernel_v2[(1, )](a, b, output, BLOCK_SIZE=4)
    return output

def torch_fn(torch_a, torch_b):
    output = torch.zeros((4, ), device=torch_a.device)
    output += torch_a[:4] * torch_b[:4]
    output += torch_a[4:] * torch_b[4:]
    return output


#################
#    common     #
#################

size = 8
a = torch.randn(size, device=DEVICE)
b = torch.randn(size, device=DEVICE)

output_torch = torch_fn(a, b)
output_triton = stub(a, b)

print("output_torch:", output_torch)
print("output_triton:", output_triton)


if torch.allclose(output_torch, output_triton, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

#### test backward ####

upstream = torch.randn_like(output_triton)
a.requires_grad = True
b.requires_grad = True
torch_a = a.clone().detach().requires_grad_(True)
torch_b = b.clone().detach().requires_grad_(True)

my_out = stub(a, b)
my_out.backward(upstream)

# compare with pytorch

torch_output = torch_fn(torch_a, torch_b)
torch_output.backward(upstream)

if torch.allclose(a.grad, torch_a.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(b.grad, torch_b.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

print("grad a: ", a.grad)
print("grad b: ", b.grad)
print()

print("torch grad a: ", torch_a.grad)
print("torch grad b: ", torch_b.grad)
print()