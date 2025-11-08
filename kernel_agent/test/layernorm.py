import torch

import triton
import triton.language as tl

from kernel_agent.autodiff import autodiff

# torch.manual_seed(20)
# # torch.set_printoptions(sci_mode=False, linewidth=1000)
# DEVICE = torch.device("cuda:0")


# NOTE: copied from official triton tutorial -- https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html#sphx-glr-getting-started-tutorials-05-layer-norm-py




@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)


# todo-now: add decorator to specify that some outputs don't require grads
@autodiff(idxs_buffers=(0, 2, 3))
def forward_layernorm(x, normalized_shape, weight, bias, eps):
    # allocate output
    y = torch.empty_like(x)
    # reshape input data into 2D tensor
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    # enqueue kernel
    _layer_norm_fwd_fused[(M, )](  #
        x_arg, y, weight, bias, mean, rstd,  #
        x_arg.stride(0), N, eps,  #
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
    return y, mean, rstd, num_warps, BLOCK_SIZE


SWEEP = [
    {"M": 4096, "N": 512 * i}
    for i in range(2, 32)  # N from 1024 to 15872
]


def make_args(dims, device="cuda", dtype=torch.float32):
    # layernorm operates on shape (M, N) where:
    # - M is typically batch_size * seq_len (rows to normalize)
    # - N is feature_dim (normalized dimension)
    M, N = dims["M"], dims["N"]
    x = -2.3 + 0.5 * torch.randn((M, N), device=device, dtype=dtype)
    weight = torch.randn((N,), device=device, dtype=dtype)
    bias = torch.randn((N,), device=device, dtype=dtype)
    normalized_shape = (N,)
    eps = 1e-5
    # forward_layernorm signature includes normalized_shape metadata and explicit eps
    return (x, normalized_shape, weight, bias, eps), {}

def flops(dims, mode):
    """optional, for benchmarking"""
    M, N = dims["M"], dims["N"]
    # Forward pass:
    # - mean computation: M*N additions + M divisions
    # - variance computation: M*N subtractions + M*N multiplications + M*N additions + M divisions
    # - normalization: M*N subtractions + M*N multiplications
    # - affine transform: M*N multiplications + M*N additions
    # Approximation: ~6*M*N ops for forward
    # Backward adds similar cost, so total ~12*M*N
    return 12.0 * M * N


def setup():
    (x, normalized_shape, weight, bias, eps), _ = make_args(SWEEP[0])
    forward_layernorm(x, normalized_shape, weight, bias, eps)
