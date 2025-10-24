"""
Efficient gradcheck implementation using PyTorch's built-in torch.autograd.gradcheck.

Only one verifier: torch.autograd.gradcheck is the check. Everything else improves test quality and debuginfo.

Problem: in f32, if |g| < g* = atol/rtol, the relative term is inert. Gradcheck then tests
almost-absolute error and can mislead (passes on skeleton bugs when grads ~ 0).

Solution: do a cheap analytic scale check before gradcheck; run gradcheck once.
- Diagnostic: _grad_magnitude_diagnostics (analytic-only, no FD)
- Validation: torch.autograd.gradcheck (THE central-difference check)
"""

import warnings

import torch
from torch.autograd.gradcheck import GradcheckError
from typing import Callable, Sequence, Tuple, Union, Optional, Any, Dict

Tensor = torch.Tensor
Tensors = Tuple[Tensor, ...]
MaybeTensors = Union[Tensor, Sequence[Tensor]]
OutputSel = Union[str, int, Sequence[int], Callable[[MaybeTensors], MaybeTensors]]


def _safe_quantiles_1d(g: torch.Tensor, qs=(0.5, 0.95), max_samples=200_000):
    """
    Purpose: compute median (p50) and tail (p95) safely on huge/NaN-contaminated tensors.

    Median (p50) unlike mean robust to outliers, reflects TYPICAL gradient magnitude.

    The need this wrapper around torch.quantile() -- torch.quantile() directly is BRITTLE:
    1. Crashes on large tensors: "tensor too large" error on >200k elements
    2. NaN contamination: any NaN → all quantiles become NaN
    3. No subsample safeguard: can't handle huge gradient vectors

    This wrapper makes torch.quantile() safe. We still use PyTorch's quantile
    algorithm, just with preprocessing (filter NaN, subsample if huge, use CPU).

    Returns: ([p50, p95, ...], max_from_full_tensor)
    """
    g = g.reshape(-1)

    # Step 1: Filter non-finite values BEFORE computing quantiles
    # Problem: torch.quantile([1, 2, nan, 4]) → [nan, nan] (NaN poisons all results)
    # Solution: filter to finite subset first
    finite_mask = torch.isfinite(g)
    if not finite_mask.any():
        return [float('nan') for _ in qs], float('nan')

    g_fin = g[finite_mask]

    # Step 2: Compute max from FULL finite tensor BEFORE subsampling
    # Why: max must be from full tensor, not subsample (subsample might miss true max)
    # Bug in old code: max was computed from subsample, losing accuracy
    gmax = float(g_fin.max().item())

    # Step 3: Subsample if tensor too large (>200k elements)
    # Problem: torch.quantile(huge_tensor) → RuntimeError: "tensor too large"
    # Solution: randomly sample 200k elements for quantile (median estimate still accurate)
    # Use randperm for sampling without replacement (better than randint with replacement)
    if g_fin.numel() > max_samples:
        idx = torch.randperm(g_fin.numel(), device=g_fin.device)[:max_samples]
        g_fin = g_fin.index_select(0, idx)

    # Step 4: Call torch.quantile on CPU (more numerically stable)
    # We're not replacing torch.quantile's algorithm, just making it safe to call
    g_cpu = g_fin.float().cpu()
    q = torch.quantile(g_cpu, torch.tensor(qs))
    return q.tolist(), gmax


def _grad_magnitude_diagnostics(
    my_op: Callable,
    inputs: Sequence[Tensor],
    *,
    atol: float,
    rtol: float,
    margin: float = 2.0,
    frac_threshold: float = 0.5,
    sample_show: int = 10
) -> Dict[str, Any]:
    """
    Purpose: analytic-only precheck. Computes grads once, reports fraction |g| < k·g* and median.
    If most grads sit below g* = atol/rtol, chosen atol/rtol won't give a meaningful test.

    No finite differences.
    _grad_magnitude_diagnostics does one forward+backward. It only flags when most
    grads are below g*, so you know gradcheck will be in the absolute regime. Earlier code
    mixed diagnostics with FD (removed).

    Cost: 1 forward + 1 backward pass, 0 finite-difference passes.
    Guards quality of the central-difference check.
    """
    if rtol <= 0:
        warnings.warn("rtol <= 0 → absolute-only regime; tiny grads cannot be judged reliably.")
        return {}

    # SEMANTIC CHANGE 1: Preserve ALL inputs (not just float tensors)
    # Old code dropped non-float args (int/bool tensors), causing op to fail
    # Fix: Only set requires_grad on float tensors, pass others as-is
    args = [
        x.detach().clone().requires_grad_(True)
        if (isinstance(x, torch.Tensor) and x.is_floating_point())
        else x
        for x in inputs
    ]

    # Run forward + backward on simple sum loss
    out = my_op(*args)
    loss = (
        out.sum()
        if not isinstance(out, (list, tuple))
        else sum(t.sum() for t in out if isinstance(t, torch.Tensor))
    )
    loss.backward()

    # Collect gradient magnitudes
    mags = []
    for x in args:
        if (isinstance(x, torch.Tensor) and x.is_floating_point()
            and x.grad is not None and x.grad.numel() > 0):
            mags.append(x.grad.detach().abs().reshape(-1))

    if not mags:
        return {}

    g = torch.cat(mags)

    # SEMANTIC CHANGE 3: Explicit finiteness filtering BEFORE stats
    # Old code: called torch.quantile(g, 0.5) directly on raw gradients
    # Problem: Any NaN → all quantiles become NaN (contamination)
    # Fix: Filter to finite subset, early-out if all NaN
    finite = torch.isfinite(g)
    n_all = g.numel()
    n_fin = int(finite.sum().item())

    # SEMANTIC CHANGE 5: Early-out on all-NaN (prevent running gradcheck on garbage)
    # Old code returned stats with NaN values
    # Fix: Return immediately with non_finite_grads flag, caller early-outs
    if n_fin == 0:
        print(f"\nERROR: All {n_all} analytical gradients are non-finite (NaN/Inf).")
        print("Your backward kernel has a catastrophic bug. Fix this before gradcheck can run.")
        return {
            "ok": False,
            "non_finite_grads": True,
            "num_total": n_all,
            "num_finite": 0
        }

    # Use only finite gradients for stats (prevents NaN contamination)
    g_fin = g[finite]

    # Dead-zone analysis: g* = atol/rtol is the crossover point
    # When |grad| < g*, relative tolerance is inactive (absolute tolerance dominates)
    # This makes gradcheck very loose: |num - ana| < atol regardless of magnitude
    g_star = atol / max(rtol, 1e-20)
    frac_dead = (g_fin < margin * g_star).float().mean().item()

    # Compute median (p50) and tail (p95) to understand gradient scale distribution
    # median (unlike mean) robust typical scale
    (p50, p95), gmax = _safe_quantiles_1d(g_fin, qs=(0.5, 0.95))

    # SEMANTIC CHANGE 4: Show gradient samples for ALL shapes (not just failures)
    # Always print for visibility (helps spot patterns across shapes)
    k = min(sample_show, g_fin.numel())
    if k > 0:
        # Use randperm for sampling without replacement (better than randint)
        idx = torch.randperm(g_fin.numel(), device=g_fin.device)[:k]
        sample = g_fin.index_select(0, idx).tolist()
        formatted_sample = [f'{v:.3g}' for v in sample]
        print(f"\n[GRAD STATS] g*={g_star:.3g}  dead_frac={100*frac_dead:.1f}%  p50={p50:.3g}  p95={p95:.3g}  max={gmax:.3g}")
        print(f"[GRAD STATS] sample({k}): {formatted_sample}")

    # Warn if some (but not all) gradients are non-finite
    if n_fin < n_all:
        frac_nonfinite = (n_all - n_fin) / n_all
        print(f"\nWARNING: {n_all - n_fin}/{n_all} ({100*frac_nonfinite:.1f}%) gradients are non-finite (NaN/Inf).")
        print("Your backward kernel has numerical stability issues. Gradcheck results will be unreliable.")

    # Determine if warning should be issued (return as flag)
    warning_issued = frac_dead >= frac_threshold

    # Warn if most gradients fall in dead zone
    if warning_issued:
        rec_target = 5.0 * g_star
        scale = (rec_target / max(p50, 1e-12)) if p50 > 0 else float("inf")
        print(f"\nWARNING: Your gradients are too small/broken for gradcheck to meaningfully validate correctness.")
        print(f"Details: {100*frac_dead:.1f}% of grads < {margin:.1f}·g* (g* = atol/rtol = {g_star:.3g})")
        print(f"Stats: p50={p50:.3g}, p95={p95:.3g}, max={gmax:.3g}")
        print(f"Tolerances: atol={atol}, rtol={rtol}")
        print(f"Suggested fix: Rescale inputs/loss by ~{scale:.2g}x to get median |grad| ≥ {rec_target:.3g}")

    return {
        "g_star": g_star,
        "frac_in_dead_zone": frac_dead,
        "grad_p50": p50,
        "grad_p95": p95,
        "grad_max": gmax,
        "num_total": n_all,
        "num_finite": n_fin,
        "warning_issued": warning_issued,
    }


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

    # DIAGNOSTIC PHASE: Analytic-only gradient magnitude check
    # Cost: 1 forward + 1 backward pass, 0 finite-difference passes
    # Old code: _warn_if_unreliable_gradient_zone computed FD on first 3 elements (6 fwd passes)
    # New code: _grad_magnitude_diagnostics is purely analytic (0 FD passes)
    # This catches NaN/dead-zone issues before expensive gradcheck (~200 fwd passes)
    dead_zone_stats = _grad_magnitude_diagnostics(my_op, inputs_list, atol=atol, rtol=rtol)

    # SEMANTIC CHANGE 5: Early-out if all gradients are non-finite
    # Old code: continued to gradcheck with NaN grads, got mysterious failures
    # New code: abort immediately with clear error message
    # Prevents wasting ~200 forward passes on garbage gradients
    if dead_zone_stats.get("non_finite_grads", False):
        return False, {
            "ok": False,
            "error": "All analytical gradients are non-finite (NaN/Inf). Cannot run gradcheck.",
            "method": "torch.autograd.gradcheck",
            "dead_zone_stats": dead_zone_stats,
        }

    # Make eps magnitude-aware to handle different input scales
    # eps_eff = eps * max(1.0, max_abs_x) scales step size with input magnitude
    max_abs_x = max(
        (inp.abs().max().item() for inp in inputs_list if inp.is_floating_point()),
        default=1.0
    )
    eps_eff = eps * max(1.0, max_abs_x)

    try:
        # VALIDATION PHASE: torch.autograd.gradcheck - THE ONLY FINITE DIFFERENCE CHECK
        # Cost: ~200 forward passes (100 random directions × 2 for central differences)
        # This is the single source of truth for gradient correctness
        #
        # In the revised setup:
        # - Diagnostic phase (above): 1 backward, 0 forward (analytic-only)
        # - Validation phase (here): ~200 forward (central differences via gradcheck)
        # - Debug phase (future): _numeric_probe_slice runs only on fail/warn
        #
        # Old setup was running 3 separate FD checks (wasteful!):
        # 1. Diagnostic FD on first 3 elements (6 fwd passes) - NOW REMOVED
        # 2. This gradcheck (~200 fwd passes) - KEPT (only mandatory FD)
        # 3. Debug FD in failure path - moved to optional probe
        torch.autograd.gradcheck(
            my_op,
            tuple(inputs_list),
            eps=eps_eff,
            atol=atol,
            rtol=rtol,
            raise_exception=True,  # Raise to get detailed error with input_idx
            fast_mode=fast_mode,
            nondet_tol=nondet_tol,
        )

        # Success case: all gradients match
        stats = {
            "ok": True,
            "method": "torch.autograd.gradcheck",
            "eps": eps,
            "eps_eff": eps_eff,
            "atol": atol,
            "rtol": rtol,
            "fast_mode": fast_mode,
            "message": "Gradients match",
        }
        # Include dead-zone diagnostics if available
        if dead_zone_stats:
            stats["dead_zone_stats"] = dead_zone_stats
        return True, stats

    except GradcheckError as e:
        # Gradcheck failed - include the raw error message which contains all relevant details
        #
        # IMPORTANT: PyTorch's gradcheck with raise_exception=True provides detailed errors:
        # - Identifies which input failed: "Jacobian mismatch for output i with respect to input j"
        # - Fast mode: runs scalar check first, reports max elementwise difference
        # - On failure: reruns in slow mode, reconstructs full Jacobians and compares elementwise
        # - Slow mode message explicitly states it's recomputing after fast mode failure
        #
        # NO EXTRA POST-PROCESSING NEEDED - don't parse with regex, message already includes input index
        error_msg = str(e)

        stats = {
            "ok": False,
            "method": "torch.autograd.gradcheck",
            "eps": eps,
            "eps_eff": eps_eff,
            "atol": atol,
            "rtol": rtol,
            "fast_mode": fast_mode,
            "error": error_msg,
            "message": "Gradient mismatch (see error for details)",
        }
        # Include dead-zone diagnostics even on failure - helps diagnose if failure
        # is due to wrong tolerances (dead-zone) vs actual gradient bug
        if dead_zone_stats:
            stats["dead_zone_stats"] = dead_zone_stats
        return False, stats

    except Exception as e:
        # Catch other errors: ValueError (invalid inputs), MemoryError (OOM),
        # TypeError, or exceptions from the operation itself
        # Note: eps_eff is always defined by the time we reach this handler
        # (it's computed before the try block)
        return False, {
            "ok": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "method": "torch.autograd.gradcheck",
            "eps": eps,
            "eps_eff": eps_eff,  # Always exists - no NameError possible
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
    dead_zone_stats = None  # Collect from first successful test

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

        # Collect dead-zone stats from first test (success or failure) for diagnostics
        # We compute these before gradcheck runs, so they're available even on failure
        if dead_zone_stats is None and "dead_zone_stats" in stats_i:
            dead_zone_stats = stats_i["dead_zone_stats"]

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

    # Include dead-zone diagnostics if available (from first successful test)
    if dead_zone_stats:
        summary["dead_zone_stats"] = dead_zone_stats

    # Format summary for readability
    summary["summary_text"] = _format_summary_for_llm(summary)

    return ok_all, summary


def _format_summary_for_llm(summary: Dict[str, Any]) -> str:
    """Format gradcheck summary to make errors and numerical failures prominent."""
    lines = []
    lines.append(f"ok: {summary['ok']}")
    lines.append(f"passed: {summary['num_passed']}/{summary['num_total']}")

    # Include dead-zone diagnostics if present
    if "dead_zone_stats" in summary:
        dz = summary["dead_zone_stats"]
        lines.append("")
        lines.append("Gradient magnitude diagnostics:")
        lines.append(f"  dead-zone threshold (g* = atol/rtol): {dz.get('g_star', 'N/A'):.4g}")
        lines.append(f"  fraction in dead-zone: {dz.get('frac_in_dead_zone', 0)*100:.1f}%")
        lines.append(f"  gradient magnitudes: p50={dz.get('grad_p50', 0):.4g}, p95={dz.get('grad_p95', 0):.4g}, max={dz.get('grad_max', 0):.4g}")
        if dz.get('warning_issued'):
            lines.append("  ⚠ WARNING: Most gradients in dead-zone (absolute tolerance dominates)")

    grad_fail = summary.get('grad_fail', [])
    if grad_fail:
        lines.append("")
        lines.append("Failures:")
        for entry in grad_fail:
            # Extract shape dimensions (exclude 'error' and 'message' keys)
            shape_items = [(k, v) for k, v in entry.items() if k not in ('error', 'message')]
            shape_str = ", ".join(f"{k}={v}" for k, v in shape_items)
            lines.append(f"  - Shape: {shape_str}")

            # Show error message with full PyTorch gradcheck details
            # The error includes which input failed, numerical/analytical values, etc.
            if 'error' in entry:
                lines.append(f"    ERROR: {entry['error']}")

            lines.append("")  # blank line between failures

    return "\n".join(lines)