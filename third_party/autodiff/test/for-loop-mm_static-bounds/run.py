import os
os.environ['TRITON_ALWAYS_COMPILE']='1'

import torch
DEVICE = torch.device("cuda:0")

import triton
import triton.language as tl



# - removed
#   - masks
#   - % M, % N
#   - hardcoded block sizes instead of providing them as meta params

@triton.jit
def kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,

        # answer-now:
        # made them all constexpr, bc had a problem were some the args were hardcoded in their kernel -- and I didn't know which one, so I didn't know which args to pass to the kernel (if pass wrong shape, CUDA memory alignment error)
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,

        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,

        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    # answer-now: for now using static bounds for simplicity
    for k in range(0, 2):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c)

def stub(
        kernel,
        a,
        b,
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=16,
        BLOCK_SIZE_K=16
    ):

    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), 1, 1)
    print("grid: ", grid)
    kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),

        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    return c

torch.manual_seed(0)
a = torch.randn((32, 32), device=DEVICE, dtype=torch.float16)
b = torch.randn((32, 32), device=DEVICE, dtype=torch.float16)

def torch_fn(a, b):
    return torch.matmul(a, b)

triton_output = stub(kernel, a, b)
torch_output = torch_fn(a, b)
# print(f"triton_output_with_fp16_inputs={triton_output[:4, :4]}")
# print(f"torch_output_with_fp16_inputs={torch_output[:4, :4]}")

if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")


#### test backward ####

from triton.backends.api import autodiff

# todo: passing grid with meta args isn't supported yet
my_op, bwd_kernel = autodiff(kernel, stub, grid=(4, 1, 1), idx_upstream=2)

upstream = torch.randn_like(torch_output)
a.requires_grad = True
b.requires_grad = True

stub(bwd_kernel, a, b)
my_out = my_op(a, b)
my_out.backward(upstream)
print("grad a: ", a.grad)
print("grad b: ", b.grad)
print()

# compare with pytorch

torch_a = a.clone().detach().requires_grad_(True)
torch_b = b.clone().detach().requires_grad_(True)

torch_output = torch_fn(torch_a, torch_b)
torch_output.backward(upstream)
print("torch grad a: ", torch_a.grad)
print("torch grad b: ", torch_b.grad)
print()

print("abs diff a.grad", torch.abs(a.grad - torch_a.grad).mean())
print("abs diff b.grad", torch.abs(b.grad - torch_b.grad).mean())
print()


if torch.allclose(a.grad, torch_a.grad, atol=1e-2, rtol=0.001):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")


if torch.allclose(b.grad, torch_b.grad, atol=1e-2, rtol=0.001):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")