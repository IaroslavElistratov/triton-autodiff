import os
os.environ['TRITON_ALWAYS_COMPILE']='1'

import torch
import triton
import triton.language as tl

from triton.backends.autodiff import autodiff


torch.manual_seed(0)
DEVICE = torch.device("cuda:0")

# Copied from official triton tutorial: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#sphx-glr-getting-started-tutorials-03-matrix-multiplication-py
# removed
#   - masks
#   - % M, % N

@triton.jit
def kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M,
        N,
        K,
        # Strides
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        # using static bounds for simplicity
        NUM_ITERS: tl.constexpr,
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
    for k in range(0, NUM_ITERS):
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


@autodiff(kernel=kernel, idxs_buffers=2)
def stub(
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
    # todo: passing grid with meta args isn't supported yet
    num_iters = triton.cdiv(K, BLOCK_SIZE_K)
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), 1, 1)
    # print("grid: ", grid)
    kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        num_iters,
    )
    return c


def torch_fn(a, b):
    return torch.matmul(a, b)


SWEEP = [
    # {"M": size, "N": size, "K": 16}
    # for size in (256, 512, 1024, 2048)

    # {"M": 256, "N": 256, "K": 16},
    # {"M": 256, "N": 256, "K": 32},

    # {"M": 2**i, "N": 2**i, "K": 2**i}
    # for i in range(8, 13)

    {"M": 256, "N": 256, "K": 16},
]
SWEEP += [{"M": 2**i, "N": 2**i, "K": 2**i} for i in range(8, 13)]


def make_args(dims, device="cuda", dtype=torch.float16):
    M, N, K = dims["M"], dims["N"], dims["K"]
    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((K, N), device=device, dtype=dtype)
    return (a, b), {}

def setup():
  (a, b), _ = make_args(SWEEP[0])
  c = stub(a, b)


# optional, for benchmarking
def flops(dims, mode):
    M, N, K = dims["M"], dims["N"], dims["K"]
    return 2.0 * M * N * K