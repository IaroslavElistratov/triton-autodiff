import os
os.environ['TRITON_ALWAYS_COMPILE']='1'

import torch
import triton
import triton.language as tl

from triton.backends.autodiff import autodiff

torch.manual_seed(0)
DEVICE = torch.device("cuda:0")


#################
#    Type 1     #
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



# #################
# #    Type 2     #
# #################
# # comment:
# #   Now there's a dependency between iterations -- they write to the same accum
# #   That's close to what flash-attention kernel does

# @autodiff(
#     # pattern="elementwise-like",
#     idxs_buffers=(2, )
# )
# @triton.jit
# def kernel_v2(
#       a_ptr,
#       b_ptr,
#       output_ptr,
#       BLOCK_SIZE: tl.constexpr,
#     ):

#     offsets = tl.arange(0, BLOCK_SIZE) # (4, )

#     accum = tl.load(output_ptr + offsets) # (4, )

#     for i in range(2):
#       a = tl.load(a_ptr + offsets) # (4, )
#       b = tl.load(b_ptr + offsets) # (4, )

#       out = a * b

#       # accum into the same buffer
#       accum += out

#       offsets += 4

#     # create new offsets overwise previous was overwritten inside the loop
#     offsets = tl.arange(0, BLOCK_SIZE) # (4, )
#     tl.store(output_ptr + offsets, accum)

# def stub(a, b):
#     output = torch.zeros((4, ), device=a.device)
#     kernel_v2[(1, )](a, b, output, BLOCK_SIZE=4)
#     return output

# def torch_fn(torch_a, torch_b):
#     output = torch.zeros((4, ), device=torch_a.device)
#     output += torch_a[:4] * torch_b[:4]
#     output += torch_a[4:] * torch_b[4:]
#     return output


#################
#    Type 4     #
#################
# comment:
#   upstream needs to change for each bwd iteration (not just upstream wrt each yield == upstream wrt final for-op outputs)
#   seems this also requires to reconstruct the fwd buffer at each iteration -- applying the inverse (not derivative) of the accumulation function
#   e.g. when accum is "accum = accum * curr" -- grad wrt prev iteration's accum requires knowing the result of the current fwd iteration (curr)


@autodiff(
    # pattern="elementwise-like",
    idxs_buffers=(2, )
)
@triton.jit
def kernel_v4(
      a_ptr,
      b_ptr,
      output_ptr,
      BLOCK_SIZE: tl.constexpr,
    ):

    offsets = tl.arange(0, BLOCK_SIZE) # (4, )

    accum = tl.load(output_ptr + offsets) # (4, )

    # answer-now: the loop varaible is not used
    for i in range(2):
      a = tl.load(a_ptr + offsets) # (4, )
      b = tl.load(b_ptr + offsets) # (4, )

      out = a * b

      # accum into the same buffer
      accum *= out

      offsets += 4

    # create new offsets overwise previous was overwritten inside the loop
    offsets = tl.arange(0, BLOCK_SIZE) # (4, )
    tl.store(output_ptr + offsets, accum)

def stub(a, b, buff):
    kernel_v4[(1, )](a, b, buff, BLOCK_SIZE=4)
    return buff

def torch_fn(torch_a, torch_b, torch_buff):
    # inplace mutation -- autograd err
    # torch_buff *= torch_a[:4] * torch_b[:4]
    # torch_buff *= torch_a[4:] * torch_b[4:]
    torch_buff = torch_buff * (torch_a[:4] * torch_b[:4])
    torch_buff = torch_buff * (torch_a[4:] * torch_b[4:])
    return torch_buff



#################
#    common     #
#################

# todo-now:
# current autodiff logic still feeds the same upstream gradient
# (the one for the final loop result) to every iteration, so the factors
# that should appear in front of early elements are missing. That is why
# the gradients coming out of my pass are different from torch's

size = 8
a = torch.randn(size, device=DEVICE, requires_grad=True)
b = torch.randn(size, device=DEVICE, requires_grad=True)
# obviously can't start with zeros when accum is "*=", bc all local grads will be zeros;
# to keep the numerics the same, passing similarly initialized buff to both my and torch_fn
# todo-now:
# buff = torch.randn((4, ), device=DEVICE)
buff = torch.ones((4, ), device=DEVICE)

upstream = torch.randn_like(buff)

torch_a = a.clone().detach().requires_grad_(True)
torch_b = b.clone().detach().requires_grad_(True)
torch_buff = buff.clone().detach() # .requires_grad_(True)
torch_upstream = upstream.clone().detach()

output_triton = stub(a, b, buff)
output_torch = torch_fn(torch_a, torch_b, torch_buff)

print("output_triton:", output_triton)
print("output_torch:", output_torch)

if torch.allclose(output_triton, output_torch, atol=1e-2, rtol=0):
    print("✅ [output] Triton and Torch match")
else:
    print("❌ [output] Triton and Torch differ")

#### test backward ####

output_triton.backward(upstream)
output_torch.backward(torch_upstream)

if torch.allclose(a.grad, torch_a.grad, atol=1e-2, rtol=0):
    print("✅ [a's grad] Triton and Torch match")
else:
    print("❌ [a's grad] Triton and Torch differ")

if torch.allclose(b.grad, torch_b.grad, atol=1e-2, rtol=0):
    print("✅ [b's grad] Triton and Torch match")
else:
    print("❌ [b's grad] Triton and Torch differ")

# if torch.allclose(buff.grad, torch_buff.grad, atol=1e-2, rtol=0):
#     print("✅ Triton and Torch match")
# else:
#     print("❌ Triton and Torch differ")

print("grad a: ", a.grad)
print("grad b: ", b.grad)
# print("grad buff: ", buff.grad)
print()

print("torch grad a: ", torch_a.grad)
print("torch grad b: ", torch_b.grad)
# print("torch grad buff: ", torch_buff.grad)
# print()