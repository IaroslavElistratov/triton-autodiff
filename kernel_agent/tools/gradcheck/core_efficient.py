"""
Efficient gradcheck implementation using PyTorch's built-in torch.autograd.gradcheck.

This uses the Triton forward kernel for numerical gradients (via finite differences)
and compares against analytical gradients from the Triton backward kernel.
Avoids O(n²) memory issues by never calling the naive torch reference implementation.
"""

import torch
from typing import Callable, Sequence, Tuple, Union, Optional, Any, Dict

Tensor = torch.Tensor
Tensors = Tuple[Tensor, ...]
MaybeTensors = Union[Tensor, Sequence[Tensor]]
OutputSel = Union[str, int, Sequence[int], Callable[[MaybeTensors], MaybeTensors]]


def check_op_backward_numerical(
    my_op: Callable[..., MaybeTensors],
    inputs: Sequence[Tensor],
    *,
    outputs: OutputSel = "auto",
    upstream: Optional[Sequence[Tensor]] = None,
    compare_dtype: torch.dtype = torch.float32,
    atol: float = 0.0001,  # Float32 finite difference tolerance
    rtol: float = 0.01,
    seed: int = 0,
    only_floating_inputs: bool = True,
    eps: float = 0.005,
    fast_mode: bool = True,
    nondet_tol: float = 0.0,
    numerical_method: str = "central",
) -> Tuple[bool, Dict[str, Any]]:
    """
    Test backward correctness using PyTorch's gradcheck with numerical gradients.

    This uses PyTorch's torch.autograd.gradcheck which:
      - Computes numerical gradients via finite differences using my_op forward
      - Computes analytical gradients via my_op backward
      - Compares them with specified tolerances

    Key advantage: Uses ONLY the efficient Triton kernel (my_op), never calls
    naive torch_fn, so avoids O(n²) memory issues.

    Args:
        my_op: The efficient Triton kernel wrapped in autograd.Function
        inputs: Input tensors (must have requires_grad=True for grad computation)
        outputs: Output selector (currently ignored, kept for API compatibility)
        upstream: Upstream gradients (currently ignored, gradcheck generates internally)
        compare_dtype: dtype for comparison (currently ignored, gradcheck uses input dtype)
        atol: Absolute tolerance (default 0.0001 for float32 central differences)
        rtol: Relative tolerance (default 0.01 = 1% relative error tolerance)
        seed: RNG seed (currently ignored by gradcheck)
        only_floating_inputs: Only compute grads for float tensors
        eps: Finite difference epsilon (default 0.005, optimal for float32)
        fast_mode: Use fast mode (random projections instead of full Jacobian)
        nondet_tol: Tolerance for non-deterministic operations
        numerical_method: Finite difference method ("central" or "forward")
                         NOTE: Ignored - PyTorch's gradcheck always uses central differences.
                         Kept for API compatibility with worker.py.

    Returns:
        (ok, stats) where:
            ok: True if gradients match within tolerance
            stats: Dict with detailed results
    """
    if not inputs:
        raise ValueError("inputs must be a non-empty sequence of tensors")

    # ========== CRITICAL: dtype handling for Triton kernels with gradcheck ==========
    #
    # PyTorch's torch.autograd.gradcheck normally expects float64 (double precision):
    # - Gradcheck computes numerical gradients via finite differences: (f(x+eps) - f(x-eps))/(2*eps)
    # - This subtraction amplifies floating-point errors, requiring high precision
    # - PyTorch docs state: "The default values are designed for input of double precision.
    #   This check will likely fail if input is of less precision, e.g., FloatTensor."
    # - If inputs are not float64, gradcheck issues warning and may fail with tight tolerances
    #
    # BUT: Triton kernels do NOT support float64:
    # - tl.dot (matrix multiplication) only supports float32/float16/bfloat16
    # - Most Triton ops are designed for GPU performance at lower precision
    # - Attempting to run Triton kernels with float64 inputs will raise runtime errors
    #
    # Solution: Keep original dtypes (float32/float16) with appropriately loose tolerances
    # Float32 central difference settings:
    # - eps=0.005: Optimal step size (h ≈ ε_mach^(1/3) ≈ 0.005 for float32)
    # - atol=0.0001: Absolute tolerance above error floor (~0.00001 to 0.00002)
    # - rtol=0.01: 1% relative tolerance for gradient comparison
    # - Use fast_mode=True to use random projections (v^T·J·u) which is more tolerant
    #   of precision issues than full Jacobian computation
    #
    # Float32 finite difference analysis:
    # - Machine epsilon: ~0.000000119
    # - Optimal central diff step: h* ≈ 0.0049
    # - Best-case relative error floor: ~0.00002
    # - atol=0.0001 provides margin above error floor while catching bugs
    # - rtol=0.01 balances strictness with float32 accumulation noise
    #
    # This approach is validated by:
    # - PyTorch forums discuss "Why does gradcheck fail for floats?" - common workaround
    # - Fast mode reduces sensitivity to individual element errors
    # ================================================================================
    inputs_list = list(inputs)
    for i, inp in enumerate(inputs_list):
        if only_floating_inputs and not inp.is_floating_point():
            continue

        # Keep original dtype - do NOT convert to float64
        # Triton kernels will fail if given float64 inputs
        inputs_list[i] = inp.detach().clone().requires_grad_(True)

    try:
        # Call PyTorch's gradcheck directly
        # This computes numerical gradients using my_op(inputs + eps) - my_op(inputs - eps)
        # and compares against analytical gradients from my_op.backward()
        ok = torch.autograd.gradcheck(
            my_op,
            tuple(inputs_list),
            eps=eps,
            atol=atol,
            rtol=rtol,
            raise_exception=False,  # Return False instead of raising
            fast_mode=fast_mode,
            nondet_tol=nondet_tol,
        )

        stats = {
            "ok": bool(ok),
            "method": "torch.autograd.gradcheck",
            "eps": eps,
            "atol": atol,
            "rtol": rtol,
            "fast_mode": fast_mode,
            "message": "Gradients match" if ok else "Gradient mismatch detected",
        }

        return bool(ok), stats

    except Exception as e:
        # Catch any errors from gradcheck (OOM, runtime errors, etc.)
        return False, {
            "ok": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "method": "torch.autograd.gradcheck",
            "eps": eps,
            "atol": atol,
            "rtol": rtol,
        }


def check_op_backward_numerical_sweep(
    my_op: Callable[..., MaybeTensors],
    *,
    sidecar: Dict[str, Any],
    outputs: OutputSel = "auto",
    upstream: Optional[Sequence[Tensor]] = None,
    compare_dtype: torch.dtype = torch.float32,
    atol: float = 0.0001,  # Float32 finite difference tolerance
    rtol: float = 0.01,
    seed: int = 0,
    only_floating_inputs: bool = True,
    eps: float = 0.005,
    fast_mode: bool = True,
    nondet_tol: float = 0.0,
    numerical_method: str = "central",
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run numerical gradient check once per dims in SWEEP and aggregate results.

    Uses PyTorch's torch.autograd.gradcheck for each test case, which:
      - Uses the efficient Triton forward kernel for numerical gradients
      - Avoids calling naive torch_fn reference implementation
      - Prevents O(n²) memory issues on large sequences

    Args:
        my_op: The efficient Triton kernel wrapped in autograd.Function
        sidecar: Dict containing 'make_args' and 'SWEEP'
        outputs: Output selector (kept for API compatibility)
        upstream: Upstream gradients (kept for API compatibility)
        compare_dtype: Comparison dtype (kept for API compatibility)
        atol: Absolute tolerance (default 0.0001 for float32 central differences)
        rtol: Relative tolerance (default 0.01 = 1% relative error tolerance)
        seed: RNG seed
        only_floating_inputs: Only compute grads for float tensors
        eps: Finite difference epsilon (default 0.005, optimal for float32)
        fast_mode: Use fast mode (random projections)
        nondet_tol: Tolerance for non-deterministic ops
        numerical_method: Finite difference method ("central" or "forward")
                         NOTE: Ignored - PyTorch's gradcheck always uses central differences.
                         Kept for API compatibility with worker.py.

    Returns:
        (ok_all, summary) where:
            ok_all: True if all test cases pass
            summary: Dict with aggregated results
    """
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
            ok_i, stats_i = check_op_backward_numerical(
                my_op=my_op,
                inputs=args,
                outputs=outputs,
                upstream=upstream,
                compare_dtype=compare_dtype,
                atol=atol,
                rtol=rtol,
                seed=seed + i,
                only_floating_inputs=only_floating_inputs,
                eps=eps,
                fast_mode=fast_mode,
                nondet_tol=nondet_tol,
                numerical_method=numerical_method,
            )
        except Exception as e:
            # Catch exceptions during test setup or execution
            ok_i = False
            stats_i = {"error": f"{type(e).__name__}: {str(e)}"}

        # Record the test shape
        try:
            d = dict(dims)
        except Exception:
            d = {"dims": str(dims)}

        total += 1
        if ok_i:
            grad_pass.append(d)
            passed += 1
        else:
            # Include error details for failed cases
            if "error" in stats_i:
                d["error"] = stats_i["error"]
            if "message" in stats_i:
                d["message"] = stats_i["message"]
            grad_fail.append(d)

        ok_all = ok_all and bool(ok_i)

    summary: Dict[str, Any] = {
        "ok": bool(ok_all),
        "num_total": total,
        "num_passed": passed,
        "num_failed": (total - passed),
        "grad_pass": grad_pass,
        "grad_fail": grad_fail,
        "method": "torch.autograd.gradcheck",
        "eps": eps,
        "atol": atol,
        "rtol": rtol,
        "fast_mode": fast_mode,
    }

    # Format summary for readability
    summary["summary_text"] = _format_summary_for_llm(summary)

    return ok_all, summary


def _format_summary_for_llm(summary: Dict[str, Any]) -> str:
    """Format gradcheck summary for readability."""
    lines = []
    lines.append(f"ok: {summary['ok']}")
    lines.append(f"method: {summary.get('method', 'unknown')}")
    lines.append(f"passed: {summary['num_passed']}/{summary['num_total']}")

    if summary.get('eps'):
        lines.append(f"eps: {summary['eps']}")
    if summary.get('fast_mode') is not None:
        lines.append(f"fast_mode: {summary['fast_mode']}")

    grad_fail = summary.get('grad_fail', [])
    if grad_fail:
        lines.append("")
        lines.append("Failures:")
        for entry in grad_fail:
            # Extract shape dimensions (exclude 'error' and 'message' keys)
            shape_items = [(k, v) for k, v in entry.items() if k not in ('error', 'message')]
            shape_str = ", ".join(f"{k}={v}" for k, v in shape_items)
            lines.append(f"  - Shape: {shape_str}")

            # Show error if present
            if 'error' in entry:
                lines.append(f"    ERROR: {entry['error']}")

            # Show message if present
            if 'message' in entry:
                lines.append(f"    {entry['message']}")

            lines.append("")  # blank line between failures

    return "\n".join(lines)