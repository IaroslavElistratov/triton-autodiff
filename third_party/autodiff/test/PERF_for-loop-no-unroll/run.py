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
    # pattern="elementwise-like", # todo: expose the pattern
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



size = 8
a = torch.randn(size, device=DEVICE, requires_grad=True)
b = torch.randn(size, device=DEVICE, requires_grad=True)
# obviously can't start with zeros when accum is "*=", bc all local grads will be zeros;
# to keep the numerics the same, passing similarly initialized buff to both my and torch_fn
# todo-now:
# buff = torch.randn((4, ), device=DEVICE)
buff = torch.ones((4, ), device=DEVICE)



# #################
# #    Type 5     #
# #################

# # comment:
# # requires changing the tile/stream structure,
# # generating 2 for-loops.

# @autodiff(
#     # pattern="affine-like",
#     idxs_buffers=(2, ),
#     axis_names=[["m","k"],["k","n"],["m", "n"]]
# )
# @triton.jit
# def kernel(
#         # Pointers to matrices
#         a_ptr, b_ptr, c_ptr,
#         # Matrix dimensions
#         M,
#         N,
#         K,
#         # Strides
#         stride_am,
#         stride_ak,
#         stride_bk,
#         stride_bn,
#         stride_cm,
#         stride_cn,
#         # Meta-parameters
#         BLOCK_SIZE_M: tl.constexpr,
#         BLOCK_SIZE_N: tl.constexpr,
#         BLOCK_SIZE_K: tl.constexpr,
# ):
#     """Kernel for computing the matmul C = A x B.
#     A has shape (M, K), B has shape (K, N) and C has shape (M, N)
#     """
#     pid = tl.program_id(axis=0)
#     grid_n = tl.cdiv(N, BLOCK_SIZE_N)
#     pid_m = pid // grid_n
#     pid_n = pid % grid_n

#     # ----------------------------------------------------------
#     # Create pointers for the first blocks of A and B.
#     # We will advance this pointer as we move in the K direction
#     # and accumulate
#     offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
#     offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
#     offs_k = tl.arange(0, BLOCK_SIZE_K)
#     a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
#     b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

#     # -----------------------------------------------------------
#     accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
#     # for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
#     # using static bounds for simplicity
#     for k in range(0, 2):
#         # Load the next block of A and B, generate a mask by checking the K dimension.
#         # If it is out of bounds, set it to 0.
#         a = tl.load(a_ptrs)
#         b = tl.load(b_ptrs)
#         # We accumulate along the K dimension.
#         accumulator = tl.dot(a, b, accumulator)
#         # Advance the ptrs to the next K block.
#         a_ptrs += BLOCK_SIZE_K * stride_ak
#         b_ptrs += BLOCK_SIZE_K * stride_bk

#     c = accumulator.to(tl.float16)

#     # -----------------------------------------------------------
#     # Write back the block of the output matrix C with masks.
#     offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
#     tl.store(c_ptrs, c)

# def stub(
#         a,
#         b,
#         BLOCK_SIZE_M=16,
#         BLOCK_SIZE_N=16,
#         BLOCK_SIZE_K=16
#     ):

#     # Check constraints.
#     assert a.shape[1] == b.shape[0], "Incompatible dimensions"
#     assert a.is_contiguous(), "Matrix A must be contiguous"
#     M, K = a.shape
#     K, N = b.shape
#     # Allocates output.
#     c = torch.empty((M, N), device=a.device, dtype=torch.float16)
#     # 1D launch kernel where each block gets its own program.
#     # todo: passing grid with meta args isn't supported yet
#     grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), 1, 1)
#     print("grid: ", grid)
#     kernel[grid](
#         a, b, c,
#         M, N, K,
#         a.stride(0), a.stride(1),
#         b.stride(0), b.stride(1),
#         c.stride(0), c.stride(1),

#         BLOCK_SIZE_M,
#         BLOCK_SIZE_N,
#         BLOCK_SIZE_K,
#     )
#     return c


# def torch_fn(a, b):
#     return torch.matmul(a, b)



# size = (32, 32)
# a = torch.randn(size, device=DEVICE, requires_grad=True)
# b = torch.randn(size, device=DEVICE, requires_grad=True)

# upstream = torch.randn_like(a)

# output_triton = stub(a, b)
# output_torch = torch_fn(torch_a, torch_b)

#################
#    common     #
#################


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