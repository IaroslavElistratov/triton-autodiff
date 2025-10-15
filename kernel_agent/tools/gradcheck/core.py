import torch
from typing import Callable, Sequence, Tuple, Union, Optional, Any, Dict

Tensor = torch.Tensor
Tensors = Tuple[Tensor, ...]
MaybeTensors = Union[Tensor, Sequence[Tensor]]
# How to select the *true* outputs from my_op if it returns auxiliaries
OutputSel = Union[str, int, Sequence[int], Callable[[MaybeTensors], MaybeTensors]]

def _as_tuple(x: MaybeTensors) -> Tensors:
    """Normalize a single Tensor or a sequence of Tensors into a tuple."""
    return (x,) if isinstance(x, torch.Tensor) else tuple(x)

def _select_outputs(y: MaybeTensors, sel: OutputSel) -> Tensors:
    """
    Pick which tensors from my_op(...) to treat as *actual* outputs.
    Many autograd.Function wrappers return extra tensors; this isolates the ones
    that participate in the reference comparison.
    """
    yt = _as_tuple(y)
    if sel == "auto":
        # Trust the op's return as-is (common case: op returns only its true outputs).
        return yt
    if isinstance(sel, int):
        # Single index -> single-output tuple
        return (yt[sel],)
    if callable(sel):
        # Custom selector callable for complex returns
        return _as_tuple(sel(y))
    # Otherwise assume a sequence of indices
    return tuple(yt[i] for i in sel)

def check_op_backward_parity(
    ref_fwd: Callable[..., MaybeTensors],
    my_op:   Callable[..., MaybeTensors],
    inputs:  Sequence[Tensor],
    *,
    outputs: OutputSel = "auto",          # how to pick my_op's real outputs
    upstream: Optional[Sequence[Tensor]] = None,
    compare_dtype: torch.dtype = torch.float32,  # compare grads in this dtype (fp32 recommended)
    atol: float = 2e-2,                   # absolute tolerance for grad comparison
    rtol: float = 1e-2,                   # relative tolerance for grad comparison
    seed: int = 0,                        # RNG seed for synthetic upstream if not provided
    only_floating_inputs: bool = True,    # compute grads only for float inputs by default
) -> Tuple[bool, Dict[str, Any]]:
    """
    Parity test for backward:
      - Build two graphs: (A) Torch reference fwd, (B) fused op my_op.
      - Run the *same* upstream gradients through both.
      - Compare input gradients per-argument.

    Notes:
      - This is an *analytic vs analytic* comparison (no finite differences).
      - Typical usage: ref_fwd is a composition of Torch ops; my_op is fused
        autograd.Function that computes the same outputs.
      - If my_op.forward returns auxiliary tensors (e.g., returns all args),
        select the true outputs via `outputs`.
      - Compare in fp32 to absorb fp16/bf16 store quantization when kernels
        accumulate in fp32 but store lower precision.
    """
    if not inputs:
        raise ValueError("inputs must be a non-empty sequence of tensors")

    # Clone inputs into two *disjoint* graphs so autograd trees do not interfere.
    ref_ins = [t.detach().clone() for t in inputs]
    op_ins  = [t.detach().clone() for t in inputs]

    # Enable grad only where useful to reduce work and avoid None mismatches.
    # If only_floating_inputs=True, require grad for floating tensors; otherwise
    # honor the original requires_grad flags (and allow non-float if requested).
    for i in range(len(inputs)):
        need = inputs[i].requires_grad or (inputs[i].is_floating_point() if only_floating_inputs else True)
        ref_ins[i].requires_grad_(need)
        op_ins[i].requires_grad_(need)

    # Forward pass: reference implementation defines the *shape/arity* of outputs.
    y_ref = _as_tuple(ref_fwd(*ref_ins))
    if not y_ref:
        raise ValueError("ref_fwd returned no tensors")

    # Upstream selection:
    # If not provided, synthesize random upstreams matching each ref output.
    # Using the same upstreams for both graphs ensures apples-to-apples VJP.
    if upstream is None:
        torch.manual_seed(seed)
        ups = tuple(torch.randn_like(y) for y in y_ref)
    else:
        ups = _as_tuple(upstream)
        if len(ups) != len(y_ref):
            raise ValueError("len(upstream) must match number of ref outputs")

    # Forward pass: fused op. Then select only the true outputs if extras exist.
    y_my_all = my_op(*op_ins)
    y_my = _select_outputs(y_my_all, outputs)
    if len(y_my) != len(y_ref):
        raise ValueError("Selected my_op outputs must match ref_fwd outputs in count")

    # Backward passes:
    # Compute analytic input-grads for both graphs under identical upstreams.
    # allow_unused=True so inputs that do not influence outputs can yield None.
    grads_ref = torch.autograd.grad(y_ref, tuple(ref_ins), grad_outputs=ups, allow_unused=True)

    # Match upstream dtype to each my_op output to avoid implicit casts inside autograd.
    ups_my = tuple(u.to(y.dtype) for u, y in zip(ups, y_my))
    grads_my  = torch.autograd.grad(y_my,  tuple(op_ins),  grad_outputs=ups_my, allow_unused=True)

    # Compare per input:
    # - Skip non-floating or no-grad inputs (they either cannot have grads or we disabled them).
    # - Handle None vs None (both unused) and None vs Tensor (mismatch).
    # - Compare in `compare_dtype` with given tolerances, and collect max abs/rel errors.
    per_input = []
    ok = True
    for i, (g_r, g_m, x) in enumerate(zip(grads_ref, grads_my, inputs)):
        # Skip non-floating or inputs with requires_grad=False in both graphs.
        if not (x.is_floating_point() and (ref_ins[i].requires_grad or op_ins[i].requires_grad)):
            per_input.append(dict(index=i, compared=False, reason="non_floating_or_nograd"))
            continue

        # Both grads are structurally absent -> parity holds for "unused" input.
        if g_r is None and g_m is None:
            per_input.append(dict(index=i, compared=False, reason="both_none"))
            continue

        # One path produced a grad while the other did not -> definite mismatch.
        if (g_r is None) ^ (g_m is None):
            ok = False
            per_input.append(dict(index=i, compared=True, equal=False, reason="one_none"))
            continue

        # Cast to a common dtype for robust numeric comparison.
        a = g_r.to(compare_dtype)
        b = g_m.to(compare_dtype)
        equal = torch.allclose(a, b, atol=atol, rtol=rtol)
        ok &= bool(equal)

        # Report worst absolute and relative errors to guide tolerance tuning.
        diff = (a - b).abs()
        max_abs = float(diff.max().item())
        denom = torch.maximum(a.abs(), b.abs()).clamp_min(1.0)  # avoid div-by-zero in relative error
        max_rel = float((diff / denom).max().item())

        per_input.append(dict(
            index=i, compared=True, equal=bool(equal),
            max_abs=max_abs, max_rel=max_rel
        ))

    # Summary suitable for logging or test assertions.
    return ok, dict(ok=ok, per_input=per_input, atol=atol, rtol=rtol, compare_dtype=str(compare_dtype))


def check_op_backward_parity_sweep(
    ref_fwd: Callable[..., MaybeTensors],
    my_op:   Callable[..., MaybeTensors],
    *,
    sidecar: Dict[str, Any],
    outputs: OutputSel = "auto",
    upstream: Optional[Sequence[Tensor]] = None,
    compare_dtype: torch.dtype = torch.float32,
    atol: float = 2e-2,
    rtol: float = 1e-2,
    seed: int = 0,
    only_floating_inputs: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    """Run backward parity once per dims in SWEEP and aggregate results."""

    make_args = sidecar["make_args"]
    sweep = sidecar["SWEEP"]

    grad_pass: list[Dict[str, Any]] = []
    grad_fail: list[Dict[str, Any]] = []
    ok_all = True
    total = 0
    passed = 0
    for i, dims in enumerate(list(sweep or [])):
        try:
            args, _ = make_args(dims)
            ok_i, stats_i = check_op_backward_parity(
                ref_fwd=ref_fwd,
                my_op=my_op,
                inputs=args,
                outputs=outputs,
                upstream=upstream,
                compare_dtype=compare_dtype,
                atol=atol,
                rtol=rtol,
                seed=seed + i,
                only_floating_inputs=only_floating_inputs,
            )
        except Exception as e:
            ok_i, stats_i = False, {"error": f"{type(e).__name__}: {e}"}
        try:
            d = dict(dims)
        except Exception:
            d = {"dims": str(dims)}
        total += 1
        if ok_i:
            grad_pass.append(d)
            passed += 1
        else:
            # Include per_input details for failed cases so LLM can see which gradient is wrong
            if "per_input" in stats_i:
                d["per_input"] = stats_i["per_input"]
            # Also include exception info if this shape failed with an error
            if "error" in stats_i:
                d["error"] = stats_i["error"]
            grad_fail.append(d)
        ok_all = ok_all and bool(ok_i)

    # Per-input error details (max_abs, max_rel) are now included in grad_fail entries
    # to help LLM identify which specific gradient is wrong and by how much

    summary: Dict[str, Any] = {
        "ok": bool(ok_all),
        "num_total": total,
        "num_passed": passed,
        "num_failed": (total - passed),
        "grad_pass": grad_pass,
        "grad_fail": grad_fail,
    }

    # Add formatted text to make errors prominent while preserving shape association.
    # Each error stays tied to its shape because different shapes trigger different errors:
    # - Small shapes might hit NameError (structural bug)
    # - Large shapes might hit OOM (memory allocation bug)
    # This tells LLM whether errors are universal or shape-specific.
    summary["summary_text"] = _format_summary_for_llm(summary)

    return ok_all, summary


def _format_summary_for_llm(summary: Dict[str, Any]) -> str:
    """Format gradcheck summary to make errors and numerical failures prominent."""
    lines = []
    lines.append(f"ok: {summary['ok']}")
    lines.append(f"passed: {summary['num_passed']}/{summary['num_total']}")

    grad_fail = summary.get('grad_fail', [])
    if grad_fail:
        lines.append("")
        lines.append("Failures:")
        for entry in grad_fail:
            # Extract shape dimensions (exclude 'error' and 'per_input' keys)
            shape_items = [(k, v) for k, v in entry.items() if k not in ('error', 'per_input')]
            shape_str = ", ".join(f"{k}={v}" for k, v in shape_items)
            lines.append(f"  - Shape: {shape_str}")

            # Show error prominently if present (structural errors like NameError, OOM)
            if 'error' in entry:
                lines.append(f"    ERROR: {entry['error']}")

            # Show per-input numerical details if present (which gradients are wrong and by how much)
            if 'per_input' in entry:
                lines.append("    per_input:")
                for inp in entry['per_input']:
                    idx = inp.get('index', '?')
                    if not inp.get('compared', True):
                        reason = inp.get('reason', 'not_compared')
                        lines.append(f"      [{idx}] {reason}")
                    elif inp.get('equal', True):
                        max_abs = inp.get('max_abs', 0)
                        max_rel = inp.get('max_rel', 0)
                        lines.append(f"      [{idx}] grad OK: max_abs={max_abs:.4g}, max_rel={max_rel:.4g}")
                    else:
                        max_abs = inp.get('max_abs', 0)
                        max_rel = inp.get('max_rel', 0)
                        lines.append(f"      [{idx}] grad WRONG: max_abs={max_abs:.4g}, max_rel={max_rel:.4g}")

            lines.append("")  # blank line between failures

    return "\n".join(lines)


# def torch_ref(a, b):
#     # Typical matmul semantics at fp16: compute in fp32, cast to fp16 for output
#     return (a.float() @ b.float()).to(a.dtype)    # fp32 accum -> fp16 store
#
# # If autograd op returns *all* tensor args, select only the real outputs:
# ok, info = check_op_backward_parity(
#     ref_fwd=torch_ref,
#     my_op=my_op,                  # torch.autograd.Function-wrapped fused op
#     inputs=(a, b),
#     outputs="auto",               # or e.g. outputs=[-1] / a lambda to pick true outputs
#     atol=2e-2, rtol=1e-2,
# )
# print(ok); print(info)
