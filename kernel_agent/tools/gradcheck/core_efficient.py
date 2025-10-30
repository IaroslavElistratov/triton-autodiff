"""
Reference-based gradcheck implementation using PyTorch autograd.

Instead of finite differences, this uses a PyTorch reference implementation
to validate Triton backward kernels via autograd gradients.
"""

import torch
from typing import Callable, Sequence, Tuple, Union, Optional, Any, Dict, List
import traceback

# Import telemetry counter to detect PyTorch/stub bypass
from ...utils import _triton_launch_counter


# Validation contract: “true” means gradients match the PyTorch reference.
# Any other failure that prevents producing comparable tensors (runtime error, telemetry check, compile breakage)
# raises immediately instead of pretending the sweep succeeded.

class ValidationRuntimeError(RuntimeError):
    """Raised when runtime failures prevent gradcheck from completing.

    Subclass RuntimeError (instead of a generic Exception) to make the intent explicit: this
    represents execution-time faults (CUDA asserts, telemetry checks, compile breakage) rather
    than “gradients mismatched”. The worker wrapper catches it and surfaces the payload to
    run_with_fix so structural issues are routed as hard failures instead of being mistaken for
    parity mismatches.
    """

    def __init__(self,
                 message: str,
                 traceback_text: Optional[str] = None,
                 summary: Optional[str] = None) -> None:
        super().__init__(message)
        self.traceback_text = traceback_text
        self.summary = summary

Tensor = torch.Tensor
Tensors = Tuple[Tensor, ...]


def check_forward_outputs_match(
    triton_op: Callable,
    pytorch_ref: Callable,
    test_inputs: Sequence[Tensor],
    test_kwargs: Dict[str, Any],
    *,
    # tolerances copied form triton tutorials
    atol: float = 1e-2,
    rtol: float = 0.0,
    verbose: bool = True
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate that PyTorch reference forward outputs match Triton forward outputs.
    Used in Phase 1 to validate the PyTorch reference implementation.

    Args:
        triton_op: Triton operation (forward only)
        pytorch_ref: PyTorch reference implementation
        test_inputs: Input tensors to test
        atol: Absolute tolerance for output comparison
        rtol: Relative tolerance
        verbose: Print detailed information

    Returns:
        (passed, stats) where passed is True if outputs match
    """
    stats = {}

    try:
        # Clone inputs for both paths
        triton_inputs = [x.detach().clone() for x in test_inputs]
        ref_inputs = [x.detach().clone() for x in test_inputs]

        # Forward pass - Triton
        triton_out = triton_op(*triton_inputs, **test_kwargs)
        if not isinstance(triton_out, (list, tuple)):
            triton_out = (triton_out,)

        # Forward pass - PyTorch reference
        ref_out = pytorch_ref(*ref_inputs, **test_kwargs)
        if not isinstance(ref_out, (list, tuple)):
            ref_out = (ref_out,)

        # Check output arity (must have same number of outputs)
        if len(triton_out) != len(ref_out):
            msg = f"Output count mismatch: Triton returned {len(triton_out)} outputs, reference returned {len(ref_out)}"
            raise ValidationRuntimeError(
                msg,
                summary=f"Forward reference validation failed: {msg}",
            )

        # Check: forward outputs must match
        all_match = True
        forward_mismatches = []

        for i, (t_out, r_out) in enumerate(zip(triton_out, ref_out)):
            if isinstance(t_out, torch.Tensor) and isinstance(r_out, torch.Tensor):
                try:
                    torch.testing.assert_close(t_out, r_out, atol=atol, rtol=rtol)
                    # Also compute actual max difference for transparency
                    max_diff = (t_out - r_out).abs().max().item()
                    if verbose:
                        print(f"[Reference Check] Output {i} matches ✓ (max_diff={max_diff:.6f}, atol={atol})")
                except AssertionError as e:
                    all_match = False
                    forward_mismatches.append(f"Output {i}: {e}")
                    if verbose:
                        print(f"[Reference Check] Output {i} mismatch: {e}")

        stats["forward_match"] = all_match
        stats["forward_mismatches"] = forward_mismatches

        return all_match, stats

    except torch.cuda.OutOfMemoryError:
        # NOTE: Naive reference impl can OOM, it's ok later we gonna omit errs on shapes marked as "required" in the user land.
        #
        # Let OOM propagate to sweep level for proper optional shape handling
        # check_op_backward_numerical_sweep catches this and decides whether to skip optional shapes or fail required ones.
        raise
    except Exception as e:
        # Non-OOM errors: raise as failure
        full_tb = traceback.format_exc()
        # Include exception type in error message (e.g., "CompilationError: <message>")
        # Triton errors show code location but str(e) may not include WHAT error occurred,
        # so prepend type (CompilationError, NameError, TypeError) giving LLM context.
        # Example: "CompilationError: Both operands must be same dtype" vs just "at 51:14: dv += tl.dot(...) ^"
        error_msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        if verbose:
            print(f"[Reference Check] Exception during forward validation: {error_msg}")
            print(full_tb)
        raise ValidationRuntimeError(
            error_msg,
            traceback_text=full_tb,
            summary=f"Forward reference failure: {error_msg}",
        ) from e


def check_op_backward_with_reference(
    triton_op: Callable,
    pytorch_ref: Callable,
    test_inputs: Sequence[Tensor],
    test_kwargs: Dict[str, Any],
    *,
    atol: float = 1e-2,
    rtol: float = 0.0,
    verbose: bool = True
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate Triton backward gradients against PyTorch reference gradients.

    Args:
        triton_op: Triton operation with forward and backward
        pytorch_ref: PyTorch reference implementation (pytorch_reference_impl)
        test_inputs: Input tensors to test
        atol: Absolute tolerance for gradient comparison
        rtol: Relative tolerance (usually 0 for reference-based check)
        verbose: Print detailed information

    Returns:
        (passed, stats) where passed is True if gradients match
    """
    stats = {}

    try:
        # Clone inputs for both paths
        triton_inputs = [x.detach().clone().requires_grad_(True) if x.is_floating_point() else x
                        for x in test_inputs]
        ref_inputs = [x.detach().clone().requires_grad_(True) if x.is_floating_point() else x
                     for x in test_inputs]

        # Guard against PyTorch/stub bypass: Ensure forward AND backward call Triton kernels
        # LLM might write stubs using pure PyTorch (no Triton kernel calls),
        # which would pass gradcheck but defeats the purpose.
        # Solution: count Triton kernel launches and fail if zero.

        # Install telemetry (idempotent - safe to call multiple times)
        _triton_launch_counter.install()
        # Forward pass - Triton (with telemetry)
        _triton_launch_counter.reset()

        triton_out = triton_op(*triton_inputs, **test_kwargs)
        if not isinstance(triton_out, (list, tuple)):
            triton_out = (triton_out,)

        # Verify that forward launched at least one Triton kernel
        # Catch here because validation err, not structual err
        try:
            _triton_launch_counter.assert_launched(context="forward")
        except RuntimeError as e:
            msg = str(e)
            raise ValidationRuntimeError(
                msg,
                summary=f"Triton forward launch telemetry failure: {msg}",
            ) from e

        # Forward pass - PyTorch reference
        ref_out = pytorch_ref(*ref_inputs, **test_kwargs)
        if not isinstance(ref_out, (list, tuple)):
            ref_out = (ref_out,)

        if len(triton_out) != len(ref_out):
            msg = f"Output count mismatch: Triton returned {len(triton_out)} outputs, reference returned {len(ref_out)}"
            raise ValidationRuntimeError(
                msg,
                summary=f"Backward validation failed: {msg}",
            )

        # First check: forward outputs must match. A mismatch means the kernel executed but produced
        # different values, so we capture it in stats and let the parity path handle the fix.
        forward_match = True
        forward_mismatches = []
        for i, (t_out, r_out) in enumerate(zip(triton_out, ref_out)):
            if isinstance(t_out, torch.Tensor) and isinstance(r_out, torch.Tensor):
                # comment:
                # numeric mismatch errs are not raised (so child doesn't fails and orchestrator.run_with_fix doesn't trigger),
                # instead the main orchestrator loop checks flag parity_ok (here named forward_match/backward_match) and if not, advances to the next iter
                #
                # If the kernel ran but produced a different output (which doesn't match with reference) we
                # treat it as a numerical failure so the orchestrator can surface it via the parity branch
                # instead of raising that err (which would trigger orchestrator's run_with_fix to re-prompt)
                #
                # Numerical mismatches fall through so the orchestrator can handle them via the parity branch.
                try:
                    torch.testing.assert_close(t_out, r_out, atol=atol, rtol=rtol)
                except AssertionError as e:
                    if verbose:
                        print(f"[Reference Check] Forward output {i} mismatch: {e}")
                    forward_mismatches.append(f"Output {i}: {e}")
                    forward_match = False
                    break

        if not forward_match:
            stats["forward_match"] = False
            stats["forward_mismatches"] = forward_mismatches
            stats["error"] = "Forward outputs don't match between Triton and PyTorch reference"
            return False, stats

        # reset telemetry
        _triton_launch_counter.reset()

        # Backward pass - Triton
        triton_loss = sum(o.sum() for o in triton_out if isinstance(o, torch.Tensor))
        triton_loss.backward()

        # Verify that backward launched at least one Triton kernel
        # Catch here because validation err, not structual err
        try:
            _triton_launch_counter.assert_launched(context="backward")
        except RuntimeError as e:
            msg = str(e)
            raise ValidationRuntimeError(
                msg,
                summary=f"Triton backward launch telemetry failure: {msg}",
            ) from e

        # Backward pass - PyTorch reference
        ref_loss = sum(o.sum() for o in ref_out if isinstance(o, torch.Tensor))
        ref_loss.backward()

        # Compare gradients
        backward_match = True
        backward_mismatches = []

        for i, (t_inp, r_inp) in enumerate(zip(triton_inputs, ref_inputs)):
            # Skip non-tensor or non-floating inputs
            if not (isinstance(t_inp, torch.Tensor) and t_inp.is_floating_point()):
                continue

            # Check gradient presence parity (both None or both not None)
            t_has_grad = hasattr(t_inp, 'grad') and t_inp.grad is not None
            r_has_grad = hasattr(r_inp, 'grad') and r_inp.grad is not None

            if t_has_grad != r_has_grad:
                backward_match = False
                backward_mismatches.append(f"Input {i}: gradient presence mismatch (Triton grad={'present' if t_has_grad else 'None'}, reference grad={'present' if r_has_grad else 'None'})")
                if verbose:
                    print(f"[Reference Check] Input {i} gradient presence mismatch: Triton={'present' if t_has_grad else 'None'}, reference={'present' if r_has_grad else 'None'}")
                continue

            # Both are None - skip comparison
            if not t_has_grad:
                continue

            # Both have gradients - compare them
            try:
                torch.testing.assert_close(t_inp.grad, r_inp.grad, atol=atol, rtol=rtol)
                if verbose:
                    print(f"[Reference Check] Input {i} gradients match ✓")
            # comment:
            # Numerical mismatches fall through so the orchestrator can handle them via the parity branch
            except AssertionError as e:
                backward_match = False
                backward_mismatches.append(f"Input {i}: {e}")
                if verbose:
                    print(f"[Reference Check] Input {i} gradient mismatch: {e}")
            finally:
                if verbose:
                    # DEBUG: Print first 20 elements for manual inspection (not shown to LLM)
                    triton_flat = t_inp.grad.flatten()[:20].cpu().tolist()
                    ref_flat = r_inp.grad.flatten()[:20].cpu().tolist()
                    print(f"  [DEBUG] Ground truth (ref) first 20: {ref_flat}")
                    print(f"  [DEBUG] Model (triton) first 20:    {triton_flat}")

        stats["gradient_match"] = backward_match
        stats["backward_mismatches"] = backward_mismatches

        return backward_match, stats

    except torch.cuda.OutOfMemoryError:
        # Let OOM propagate to sweep level for proper optional shape handling
        # Sweep level will catch this and either fail (required) or skip (optional)
        raise
    except Exception as e:
        full_tb = traceback.format_exc()
        # Include exception type in error message (e.g., "CompilationError: <message>")
        # Example: "CompilationError: Both operands must be same dtype" vs just "at 51:14: dv += tl.dot(...) ^"
        error_msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        if verbose:
            print(f"[Reference Check] Exception during validation: {error_msg}")
            print(full_tb)
        raise ValidationRuntimeError(
            error_msg,
            traceback_text=full_tb,
            summary=f"Backward validation failure: {error_msg}",
        ) from e


def check_op_backward_numerical_sweep(
    my_op: Callable,
    sidecar: Dict[str, Any],
    outputs: Any = "auto",
    *,
    atol: float = 1e-2,
    rtol: float = 0.0,
    eps: float = 0.005,
    verbose: bool = True,
    forward_only: bool = False
) -> Tuple[bool, Dict[str, Any]]:
    """
    Main entry point for gradcheck using reference-based validation.

    Validates Triton kernels against PyTorch reference implementation via autograd.

    Args:
        my_op: Triton operation to test
        sidecar: Dict containing test configuration (SWEEP, make_args, pytorch_reference_impl)
        outputs: Which outputs to check (unused, kept for compatibility)
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison
        eps: Step size (unused, kept for compatibility)
        verbose: Print detailed information
        forward_only: If True, only validate forward outputs (Phase 1)

    Returns:
        (all_passed, stats) where all_passed is True if all shapes pass
    """
    # Get PyTorch reference implementation
    pytorch_ref = sidecar.get("pytorch_reference_impl")
    if pytorch_ref is None:
        raise ValueError("pytorch_reference_impl not found in sidecar. "
                        "PyTorch reference must be provided for reference-based gradcheck.")

    # Get test configuration
    sweep = sidecar.get("SWEEP", [])
    make_args = sidecar.get("make_args")

    if not sweep:
        raise ValueError("SWEEP not found or empty in sidecar")
    if not make_args:
        raise ValueError("make_args not found in sidecar")

    stats = {
        "method": "reference",
        "validation_type": "forward_only" if forward_only else "full_gradcheck",
        "num_shapes": len(sweep),
        "shapes_passed": 0,
        "shapes_failed": 0,
        "shapes_oom": 0,
        "shape_details": []
    }

    all_passed = True

    # FEATURE: Required Flag for Selective Shape Testing
    # Allows users to mark shapes as required=True (must pass) or required=False (OOM OK)
    # RATIONALE: Naive PyTorch references materialize full tensors (e.g., N×N attention matrix)
    # which OOM on large shapes. Optional shapes allow testing on smaller shapes for correctness
    # while skipping memory-intensive large shapes that would fail the naive reference.
    for i, shape_dict in enumerate(sweep):
        # Extract 'required' flag (default True - strict by default for safety)
        is_required = shape_dict.get("required", True)
        shape_params = {k: v for k, v in shape_dict.items() if k != "required"}

        if verbose:
            validation_type = "forward outputs" if forward_only else "gradients"
            req_tag = "[REQUIRED]" if is_required else "[OPTIONAL]"
            print(f"\n[Reference Check] {req_tag} Testing shape {i+1}/{len(sweep)} ({validation_type}): {shape_params}")

        # Generate test inputs
        args, kwargs = make_args(shape_dict)

        # Wrap in try-except for OOM handling (BOTH phases - see comment below)
        try:
            # Choose validation function based on phase
            if forward_only:
                # Phase 1: Only validate forward outputs
                shape_passed, shape_stats = check_forward_outputs_match(
                    triton_op=my_op,
                    pytorch_ref=pytorch_ref,
                    test_inputs=args,
                    test_kwargs=kwargs,
                    atol=atol,
                    rtol=rtol,
                    verbose=verbose
                )
            else:
                # Phase 2: Full gradient validation (OOM handling applies - see below)
                shape_passed, shape_stats = check_op_backward_with_reference(
                    triton_op=my_op,
                    pytorch_ref=pytorch_ref,
                    test_inputs=args,
                    test_kwargs=kwargs,
                    atol=atol,
                    rtol=rtol,
                    verbose=verbose
                )

            # Validation completed (no OOM)
            shape_info = {
                "shape": shape_params,
                "required": is_required,
                "passed": shape_passed,
                "stats": shape_stats
            }
            stats["shape_details"].append(shape_info)

            if shape_passed:
                stats["shapes_passed"] += 1
                if verbose:
                    print(f"[Reference Check] Shape {i+1} PASSED ✓")
            else:
                stats["shapes_failed"] += 1
                all_passed = False
                if verbose:
                    print(f"[Reference Check] Shape {i+1} FAILED ✗")
                    if "error" in shape_stats:
                        print(f"  Error: {shape_stats['error']}")

        except torch.cuda.OutOfMemoryError as e:
            # FEATURE: OOM Handling for Naive References
            # APPLIES TO BOTH PHASES: If naive reference OOMs in forward pass, it cannot
            # provide reference gradients in backward pass either (must run forward first).
            # CORNER CASE: Large shapes with naive implementations (e.g., materializing full
            # N×N attention matrix) will OOM consistently in both Phase 1 and Phase 2.
            # SOLUTION: Track OOM and continue testing other shapes. At the end, if no required
            # shapes passed, we'll determine if it's user's fault (no required) or LLM's fault (all failed/OOMed).
            stats["shapes_oom"] += 1
            stats["shape_details"].append({
                "shape": shape_params,
                "required": is_required,
                "passed": None,
                "oom": True
            })
            phase_name = "forward" if forward_only else "backward"
            req_tag = "[REQUIRED]" if is_required else "[OPTIONAL]"
            if verbose:
                print(f"[Reference Check] {req_tag} Shape {i+1} OOMed in {phase_name}")
            # Continue to next shape (don't return immediately - test all shapes)
            continue

        # we’re not swallowing ValidationRuntimeError. We catch the exception purely to enrich
        # it with shape context before re-raising. Without that wrapper we’d only get
        # “ValidationRuntimeError: cuda illegal access,” which is hard to debug when there
        # are multiple shapes in the sweep. The block builds a message like: ValidationRuntimeError(...)  [shape: B=1, NUM_HEADS=8, ...]
        # and then rethrows, preserving the original traceback (and summary if we had one). That
        # augmented exception bubbles up to _gradcheck_child, which in turn hands it to run_with_fix,
        # so the orchestrator still treats it as a hard runtime failure
        except ValidationRuntimeError as err:
            shape_str = ", ".join(f"{k}={v}" for k, v in shape_params.items())
            summary = err.summary or f"Shape {shape_str}: {err}"
            raise ValidationRuntimeError(
                f"{err} [shape: {shape_str}]",
                getattr(err, "traceback_text", None),
                summary,
            ) from err

        # Force any sticky CUDA errors to surface NOW before next iteration
        # Explicit synchronize after each shape test to catch errors before next make_args().
        # - Prevents misleading tracebacks: Error surfaces HERE (after kernel execution),
        #   NOT at torch.empty() in next iteration's make_args() (wrong location)
        # todo:
        # - BUT traceback NOT precise: Points to synchronize() call, not the actual failing kernel line
        #     (stack unwound by the time async error surfaces)
        # Example: Illegal memory access in _attn_bwd kernel at iteration N:
        #   Without this sync: Error appears at "torch.empty()" in iteration N+1 make_args()
        #   With this sync: Error caught here after shape N test, before shape N+1 begins
        try:
            torch.cuda.synchronize()
        except Exception as e:
            sync_msg = f"{type(e).__name__}: {e}"
            if verbose:
                print(f"[Reference Check] CUDA sync error after shape {i+1}: {sync_msg}")
            raise ValidationRuntimeError(
                f"CUDA sync error after shape {i+1}: {sync_msg}",
                traceback.format_exc(),
                f"CUDA sync error after shape {i+1}: {sync_msg}",
            ) from e

    # FEATURE: Vacuous Truth Prevention
    # LOGIC:
    # - If user didn't specify ANY required shapes (num_required == 0) → User's fault
    # - If user specified required shapes (num_required > 0) but all failed/OOMed → LLM's fault
    if stats["shapes_passed"] == 0:
        num_required = sum(1 for s in sweep if s.get("required", True))

        if num_required == 0:
            # User's fault: No required shapes in SWEEP
            # All shapes are optional → vacuous truth (nothing was REQUIRED to pass)
            phase_name = "Phase 1" if forward_only else "Phase 2"
            error_msg = (
                f"No shapes validated in {phase_name}!\n"
                f"  Total shapes: {len(sweep)} (all optional)\n"
                f"  OOMed: {stats['shapes_oom']}\n"
                f"  Failed: {stats['shapes_failed']}\n"
                f"\n"
                f"SWEEP must contain at least ONE required shape.\n"
                f"\n"
                f"Solutions:\n"
                f"  1. Mark at least one small shape as required=True in SWEEP\n"
                f"  2. Add smaller shapes to SWEEP (e.g., SEQ=128, required=True)\n"
            )
            return False, {
                "ok": False,
                "error": error_msg,
                "summary_text": error_msg,
                "results": stats["shape_details"]
            }

        # else: num_required > 0, but shapes_passed == 0
        # → Required shapes exist but all failed validation or OOMed
        # → LLM's fault (implementation needs fixing)
        all_passed = False  # Ensure we signal failure to orchestrator for LLM retry
        # → Fall through to normal return path for LLM retry with error details

    # Format summary for LLM
    # I only surface the condensed summary_text to the orchestrator here, so the detailed per-shape
    # tracebacks stored in stats["shape_details"] never reach the prompt the model sees.
    stats["summary_text"] = _format_summary_for_llm(stats, forward_only)

    # Print minimal summary
    if verbose:
        phase = "Phase 1 (forward)" if forward_only else "Phase 2 (gradients)"
        status = "✓ PASSED" if all_passed else "✗ FAILED"
        print(f"\n[Reference Check - {phase}] {status}: {stats['shapes_passed']}/{stats['num_shapes']} shapes validated")

    return all_passed, stats


# todo: simplify
def _format_summary_for_llm(stats: Dict[str, Any], forward_only: bool) -> str:
    """
    Format validation results into a clean summary for LLM prompts.
    Focus on failures and actionable errors - omit passed shapes.

    Error taxonomy:
    - forward_mismatches / backward_mismatches: numerical correctness failures
    - OOM entries are handled separately at sweep level

    Tracebacks NOT included in stats (printed directly in verbose mode for debugging).
    This function extracts ERROR MESSAGES ONLY (not tracebacks) from stats.
    Orchestrator calls this function to get summary_text, which is what LLM sees.
    Only uncaught exceptions (that bubble up to orchestrator) show full tracebacks.
    """
    lines = []

    # Status line
    phase = "Forward" if forward_only else "Gradient"
    status = "PASSED" if stats["shapes_passed"] > 0 and stats["shapes_failed"] == 0 else "FAILED"
    lines.append(f"{phase} Validation: {status} ({stats['shapes_passed']}/{stats['num_shapes']} shapes passed)")

    # Success case: early return with clean message
    if stats["shapes_failed"] == 0 and stats["shapes_passed"] > 0:
        if stats["shapes_oom"] > 0:
            lines.append(f"OOM: {stats['shapes_oom']} shapes skipped (naive reference)")
        return "\n".join(lines)

    # Failed shapes - only show these (LLM needs to fix them)
    failed_shapes = [s for s in stats["shape_details"] if s.get("passed") is False]
    if failed_shapes:
        lines.append("\nFailed shapes:")
        for i, shape_info in enumerate(failed_shapes, 1):
            shape_params = shape_info["shape"]
            required = "[REQUIRED]" if shape_info.get("required", True) else "[OPTIONAL]"
            shape_str = ", ".join(f"{k}={v}" for k, v in shape_params.items())
            lines.append(f"\n{required} Shape: {shape_str}")

            # Extract error details from shape stats
            shape_stats = shape_info.get("stats", {})

            # At this point only numerical mismatches remain; all runtime failures raise and are handled via run_with_fix.
            # note there can only be numeric mismatch errs and no structural errs
            # all errs besides numeric mismatches -- should be raised so that
            # child catches them and returns child_run_ok=False to run_with_fix
            # causes to re-prompt llm
            if forward_only:
                numerical_failures = shape_stats.get("forward_mismatches", [])
                for err in numerical_failures[:3]:
                    lines.append(f"  {err.split(':')[0]}: mismatch")
            else:
                numerical_failures = shape_stats.get("backward_mismatches", [])
                for err in numerical_failures[:3]:
                    lines.append(f"  {err.split(':')[0]}: mismatch")

    # OOM shapes - distinguish between required (problem) and optional (expected)
    oom_shapes = [s for s in stats["shape_details"] if s.get("oom")]
    if oom_shapes:
        required_oom = [s for s in oom_shapes if s.get("required", True)]
        optional_oom = [s for s in oom_shapes if not s.get("required", True)]

        if required_oom:
            # Required shapes OOMed - this is a problem LLM needs to fix
            lines.append(f"\nRequired shapes OOMed ({len(required_oom)} shapes):")
            for shape_info in required_oom:
                shape_str = ", ".join(f"{k}={v}" for k, v in shape_info["shape"].items())
                lines.append(f"  [REQUIRED] {shape_str}: OOM")

        if optional_oom:
            lines.append(f"\nOptional shapes OOMed: {len(optional_oom)} (expected for naive reference)")

    return "\n".join(lines)
