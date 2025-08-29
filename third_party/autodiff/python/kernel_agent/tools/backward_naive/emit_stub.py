# stub_codegen_minimal.py
import os
os.environ['TRITON_ALWAYS_COMPILE'] = '1'

import torch
import triton
import triton.language as tl
from textwrap import dedent


# ---------------------------
# 1) Example Triton kernels
# ---------------------------

@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    tl.store(out_ptr + offs, a + b)


@triton.jit
def add_kernel_bwd(a_ptr, b_ptr, out_ptr,
                   grad_a_ptr, grad_b_ptr, grad_out_ptr,
                   BLOCK_SIZE: tl.constexpr):
    # d(a+b)/da = 1, d(a+b)/db = 1  -> grad_a += upstream, grad_b += upstream
    offs = tl.arange(0, BLOCK_SIZE)
    upstream = tl.load(grad_out_ptr + offs)
    tl.atomic_add(grad_a_ptr + offs, upstream)
    tl.atomic_add(grad_b_ptr + offs, upstream)


# -------------------------------------------
# 2) Tiny stub codegen that emits callsites
# -------------------------------------------

def build_stubs(fwd_kernel, bwd_kernel, inputs, need_grads, out_name='out', block_size_expr=None):
    """
    Emits two functions with hard-coded callsites:
      stub_fwd(a, b, ...)
      stub_bwd(a, b, upstream, ...)
    """
    if block_size_expr is None:
        block_size_expr = f"{inputs[0]}.numel()"

    inp_list = ", ".join(inputs)
    grad_alloc = "\n        ".join([f"grad_{n} = torch.zeros_like({n})" for n in need_grads])
    grad_tuple = ", ".join([f"grad_{n}" for n in need_grads])

    src = dedent(f"""
    def stub_fwd({inp_list}):
        {out_name} = torch.empty_like({inputs[0]})
        grid = (1,)
        {fwd_kernel.__name__}[grid]({", ".join(inputs + [out_name])}, BLOCK_SIZE={block_size_expr})
        return {out_name}

    def stub_bwd({inp_list}, upstream):
        {out_name} = torch.empty_like({inputs[0]})
        grid = (1,)
        {fwd_kernel.__name__}[grid]({", ".join(inputs + [out_name])}, BLOCK_SIZE={block_size_expr})
        {grad_alloc}
        grad_out = upstream.clone()
        {bwd_kernel.__name__}[grid]({", ".join(inputs + [out_name] + [f"grad_{n}" for n in need_grads] + ["grad_out"])}, BLOCK_SIZE={block_size_expr})
        return ({grad_tuple},)
    """).strip()

    print(src)
    ns = dict(torch=torch, triton=triton, tl=tl,
              **{fwd_kernel.__name__: fwd_kernel, bwd_kernel.__name__: bwd_kernel})
    exec(src, ns, ns)
    return ns['stub_fwd'], ns['stub_bwd'], src


# ---------------------------
# 3) Demo / quick test
# ---------------------------

if __name__ == "__main__":
    DEVICE = torch.device("cuda:0")  # requires CUDA
    torch.manual_seed(0)

    # Build hard-coded callsite stubs
    stub_fwd, stub_bwd, emitted_src = build_stubs(
        add_kernel, add_kernel_bwd,
        inputs=["a", "b"],
        need_grads=["a", "b"],
    )

    # Data
    a = torch.randn(8, device=DEVICE, dtype=torch.float32)
    b = torch.randn(8, device=DEVICE, dtype=torch.float32)

    # Forward check
    out_ref = a + b
    out_triton = stub_fwd(a, b)
    print("fwd match:", torch.allclose(out_ref, out_triton, atol=1e-6))

    # Backward check
    upstream = torch.randn_like(out_ref)
    grad_a, grad_b = stub_bwd(a, b, upstream)

    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    (a_ref + b_ref).backward(upstream)

    print("bwd match a:", torch.allclose(grad_a, a_ref.grad, atol=1e-6))
    print("bwd match b:", torch.allclose(grad_b, b_ref.grad, atol=1e-6))
