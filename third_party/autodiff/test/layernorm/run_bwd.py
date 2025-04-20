import numpy as np
import torch
DEVICE = torch.device("cuda:0")

import triton
from triton.compiler import compile
from triton.backends.compiler import GPUTarget


# Create a GPU target
target = GPUTarget("cuda", arch=89, warp_size=32)

# Compile the IR
bwd_kernel = compile("out.ttir", target=target)



def bwd(x, weight, bias, eps=1e-5):

    ######## COPY FROM FWD ########
    # allocate output
    y = torch.empty_like(x)
    # reshape input data into 2D tensor
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    # answer-now: some input types to the fn are float16 and some are float32
    mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    # num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    ################################

    # todo-now: which of the inputs actually needs to be considered upstream in this case
    #   ==> The y

    grad_x_arg = torch.zeros_like(x_arg)
    # note: ones!
    grad_y = torch.ones_like(y)
    grad_weight = torch.zeros_like(weight)
    grad_bias = torch.zeros_like(bias)
    grad_mean = torch.zeros_like(mean)
    grad_rstd = torch.zeros_like(rstd)

    # enqueue kernel
    compiled_kernel = bwd_kernel[(M, 1, 1)](  #
        # fwd values:
        x_arg, y, weight, bias, mean, rstd,
        # grads:
        grad_x_arg, grad_y, grad_weight, grad_bias, grad_mean, grad_rstd,

    )
    return compiled_kernel, grad_x_arg, grad_weight, grad_bias, grad_mean, grad_rstd


M = 1151
N = 8192
dtype = torch.float16
device = DEVICE

# create data
x_shape = (M, N)
w_shape = (x_shape[-1], )
weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
dy = .1 * torch.randn_like(x)
# x.requires_grad_(True)
# quantiles = [0.5, 0.2, 0.8]


# # Run our implementation
with torch.no_grad():
    compiled_kernel, grad_x_arg, grad_weight, grad_bias, grad_mean, grad_rstd = bwd(x, weight, bias)
    print("grad_x_arg: ", grad_x_arg)
    print("grad_weight: ", grad_weight)
    print("grad_bias: ", grad_bias)
# print("Small test gradients:")
# print("dx shape:", dx_small.shape)
# print("dweight shape:", dweight_small.shape)
# print("dbias shape:", dbias_small.shape)

# # Compare with PyTorch's implementation
# x_torch = x_small.clone().detach().requires_grad_(True)
# weight_torch = weight_small.clone().detach().requires_grad_(True)
# bias_torch = bias_small.clone().detach().requires_grad_(True)

print("\n" * 3)
# # Forward pass with PyTorch's LayerNorm
x.requires_grad = True
weight.requires_grad = True
bias.requires_grad = True
torch_ln = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps=1e-5)
torch_ln.backward(torch.ones_like(torch_ln))
print("x.grad: ", x.grad)
print("weight.grad: ", weight.grad)
print("bias.grad: ", bias.grad)
# torch_ln.weight.data = weight_torch
# torch_ln.bias.data = bias_torch
# y_torch = torch_ln(x_torch)
# y_torch.backward(dy_small)

# print("\nPyTorch reference gradients:")
# print("dx norm difference:", torch.norm(dx_small - x_torch.grad).item())
# print("dweight norm difference:", torch.norm(dweight_small - weight_torch.grad).item())
# print("dbias norm difference:", torch.norm(dbias_small - bias_torch.grad).item())

