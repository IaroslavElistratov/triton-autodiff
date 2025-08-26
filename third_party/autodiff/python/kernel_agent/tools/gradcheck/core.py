import torch
from typing import Callable, Sequence, Tuple, Union, Optional, Any, Dict

Tensor = torch.Tensor
Tensors = Tuple[Tensor, ...]
MaybeTensors = Union[Tensor, Sequence[Tensor]]
OutputSel = Union[str, int, Sequence[int], Callable[[MaybeTensors], MaybeTensors]]

def _as_tuple(x: MaybeTensors) -> Tensors:
    return (x,) if isinstance(x, torch.Tensor) else tuple(x)

def _select_outputs(y: MaybeTensors, sel: OutputSel) -> Tensors:
    yt = _as_tuple(y)
    if sel == "auto":
        # default: trust the op's return (e.g., user stub returning true outputs)
        return yt
    if isinstance(sel, int):
        return (yt[sel],)
    if callable(sel):
        return _as_tuple(sel(y))
    # assume sequence of indices
    return tuple(yt[i] for i in sel)

def check_op_backward_parity(
    ref_fwd: Callable[..., MaybeTensors],
    my_op: Callable[..., MaybeTensors],
    inputs: Sequence[Tensor],
    *,
    outputs: OutputSel = "auto",          # how to pick my_op's real outputs
    upstream: Optional[Sequence[Tensor]] = None,
    compare_dtype: torch.dtype = torch.float32,
    atol: float = 2e-2,
    rtol: float = 1e-2,
    seed: int = 0,
    only_floating_inputs: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    """Parity of grads: Torch(ref_fwd) vs my fused op (my_op). No finite differences."""
    # Clone into two graphs
    ref_ins = [t.detach().clone() for t in inputs]
    op_ins  = [t.detach().clone() for t in inputs]

    # Enable grad where useful
    for i in range(len(inputs)):
        need = inputs[i].requires_grad or (inputs[i].is_floating_point() if only_floating_inputs else True)
        ref_ins[i].requires_grad_(need)
        op_ins[i].requires_grad_(need)

    # Forward: reference
    y_ref = _as_tuple(ref_fwd(*ref_ins))
    if not y_ref:
        raise ValueError("ref_fwd returned no tensors")

    # Upstream
    if upstream is None:
        torch.manual_seed(seed)
        ups = tuple(torch.randn_like(y) for y in y_ref)
    else:
        ups = _as_tuple(upstream)
        if len(ups) != len(y_ref):
            raise ValueError("len(upstream) must match number of ref outputs")

    # Forward: my op, then select true outputs if it returns extras
    y_my_all = my_op(*op_ins)
    y_my = _select_outputs(y_my_all, outputs)
    if len(y_my) != len(y_ref):
        raise ValueError("Selected my_op outputs must match ref_fwd outputs in count")

    # Backward passes
    grads_ref = torch.autograd.grad(y_ref, tuple(ref_ins), grad_outputs=ups, allow_unused=True)
    ups_my = tuple(u.to(y.dtype) for u, y in zip(ups, y_my))
    grads_my  = torch.autograd.grad(y_my,  tuple(op_ins),  grad_outputs=ups_my, allow_unused=True)

    # Compare per input
    per_input = []
    ok = True
    for i, (g_r, g_m, x) in enumerate(zip(grads_ref, grads_my, inputs)):
        # Skip non-floating or no-grad inputs
        if not (x.is_floating_point() and (ref_ins[i].requires_grad or op_ins[i].requires_grad)):
            per_input.append(dict(index=i, compared=False, reason="non_floating_or_nograd"))
            continue
        if g_r is None and g_m is None:
            per_input.append(dict(index=i, compared=False, reason="both_none"))
            continue
        if (g_r is None) ^ (g_m is None):
            ok = False
            per_input.append(dict(index=i, compared=True, equal=False, reason="one_none"))
            continue
        a = g_r.to(compare_dtype)
        b = g_m.to(compare_dtype)
        equal = torch.allclose(a, b, atol=atol, rtol=rtol)
        ok &= bool(equal)
        diff = (a - b).abs()
        per_input.append(dict(
            index=i, compared=True, equal=bool(equal),
            max_abs=float(diff.max().item()),
            max_rel=float((diff / torch.maximum(a.abs(), b.abs()).clamp_min(1.)).max().item())
        ))

    return ok, dict(ok=ok, per_input=per_input, atol=atol, rtol=rtol, compare_dtype=str(compare_dtype))

"""
def torch_ref(a, b):
    return (a.float() @ b.float()).to(a.dtype)    # fp32 accum → fp16 store  :contentReference[oaicite:2]{index=2}

# If my autograd op returns all tensor args, pick only the real output:
ok, info = check_op_backward_parity(
    ref_fwd=torch_ref,
    my_op=my_op,                  # my torch.autograd.Function-wrapped op
    inputs=(a, b),
    outputs="auto",               # or outputs=[-1] / a lambda to select real outputs  :contentReference[oaicite:3]{index=3}
    atol=2e-2, rtol=1e-2,
)
print(ok); print(info)
"""