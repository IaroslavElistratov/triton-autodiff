# SPDX-License-Identifier: BSD-2-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/linkedin/Liger-Kernel
# Source-Files: src/liger_kernel/ops/swiglu.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)


import torch
import triton
import triton.language as tl


from kernel_agent.autodiff import autodiff


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######

def is_hip() -> bool:
    return torch.version.hip is not None


def calculate_settings(n):
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32 if not is_hip() else 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


######


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _swiglu_forward_kernel(a_ptr, b_ptr, c_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    c_row = silu(a_row).cast(b_row.dtype) * b_row
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


@autodiff(inputs_require_grad=(0, 1))
def swiglu_forward(a, b):
    ori_shape = a.shape

    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_forward_kernel[(n_rows,)](
        a,
        b,
        c,
        c.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a, b, c.view(*ori_shape)

SWEEP = [
    {"B": 4, "SEQ_LEN": 1024, "INTERMEDIATE_SIZE": 11008, "dtype": torch.bfloat16, "required": True},
    {"B": 4, "SEQ_LEN": 2048, "INTERMEDIATE_SIZE": 11008, "dtype": torch.bfloat16, "required": True},
    {"B": 4, "SEQ_LEN": 4096, "INTERMEDIATE_SIZE": 11008, "dtype": torch.bfloat16, "required": False},
    {"B": 4, "SEQ_LEN": 8192, "INTERMEDIATE_SIZE": 11008, "dtype": torch.bfloat16, "required": False},
]


def make_args(dims, device=DEVICE, dtype=None):
    dims = dict(dims)
    batch = dims["B"]
    seq_len = dims["SEQ_LEN"]
    intermediate = dims["INTERMEDIATE_SIZE"]
    tensor_dtype = dims.get("dtype", dtype or torch.bfloat16)
    a = torch.randn((batch, seq_len, intermediate), dtype=tensor_dtype, device=device)
    b = torch.randn_like(a)
    return (a, b), {}

# temporarily disable no bwd-file flop reference to validate it
# todo: replace with actual flop count for bwd
# def flops(dims, mode):
#     if mode == "bwd":
#         # backward recomputes sigmoid/SiLU and emits two grads (~12 scalar ops per element)
#         elems = dims["B"] * dims["SEQ_LEN"] * dims["INTERMEDIATE_SIZE"]
#         return float(elems) * 12.0
#     raise ValueError(f"flops only implemented for backward mode; got {mode}")


def setup():
    (a, b), _ = make_args(SWEEP[0])
    swiglu_forward(a, b)


if __name__ == "__main__":
    # catch errors quickly
    setup()
