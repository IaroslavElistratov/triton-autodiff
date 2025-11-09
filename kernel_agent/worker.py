import os
import multiprocessing as mp
import queue
import traceback

import sys

from .generate_initial import wrap_with_kwargs


# High-level workflow:
#
# 0. Orchestrator writes raised.py (forward copy + editable backward scaffold) before
#    any worker starts. Rollback snapshots that file between iterations.
#
# 1. Each worker child calls compile_kernel(fwd_fp=..., generated_fp=raised.py).
#    compile_kernel imports the forward module to build the "sidecar" namespace
#    (make_args, SWEEP, etc.) we need for gradcheck/bench; it no longer patches
#    the stub at runtime.
#
# 2. Import raised.py directly, fetch DifferentiableStub and the orchestrator-supplied stub name,
#    and create a wrapper that mirrors the stub signature but forwards into DifferentiableStub.apply.
#    The original stub inside raised.py untouched; only the worker holds the wrapper.
#
# 3. Gradcheck/benchmark call that wrapper; it binds kwargs/defaults, hands the positional tuple to
#    DifferentiableStub.apply, and therefore executes whatever the LLM last wrote to raised.py without
#    altering the stub definition itself.


# Dedicated timeouts for long-running child tasks (env-overridable)
GRADCHECK_TIMEOUT_S = float(os.environ.get("TB_GRADCHECK_TIMEOUT_S", "180"))
BENCH_TIMEOUT_S     = float(os.environ.get("TB_BENCH_TIMEOUT_S", "180"))



def _load_generated_op(generated_fp: str):
    import runpy

    # Execute generated/raised.py to inspect its namespace (forward copy, backward stub, DifferentiableStub).
    ns = runpy.run_path(generated_fp)

    # DifferentiableStub is the autograd bridge the generated file exports. Without it the backward cannot run.
    override_cls = ns.get("DifferentiableStub")
    if override_cls is None:
        raise RuntimeError("DifferentiableStub missing from generated backward file")

    target_stub_name = os.environ.get("KERNEL_AGENT_STUB_NAME")
    if not target_stub_name:
        raise RuntimeError("KERNEL_AGENT_STUB_NAME env var missing; orchestrator must pass stub name")

    candidate = ns.get(target_stub_name)
    if not callable(candidate):
        raise RuntimeError(
            f"Autodiff stub '{target_stub_name}' not found or not callable in {generated_fp}"
        )

    # Rely solely on this orchestrator-provided name so the worker ignores the forward module's stub and always executes raised.py edits.
    stub_name = target_stub_name
    stub_fn = candidate

    # Wrap the generated stub so kwargs/defaults still work, but bodies go through DifferentiableStub.apply().
    # Autograd Function.apply only accepts positional args, so we preserve the original signature
    # (including keyword-only parameters and defaults) via inspect.signature.bind(). This happens in
    # the worker only; the original forward module isn’t modified.
    #
    # Important: do NOT overwrite the stub living in raised.py. This helper creates a thin wrapper
    # that mirrors the stub's signature and forwards into DifferentiableStub.apply. The gradcheck/bench children
    # call that wrapper so kwargs/defaults bind correctly before apply() (which only accepts positional args).
    # Because the wrapper sits purely in the worker, the actual stub definition inside raised.py remains untouched.
    #
    # IOW: only borrow the stub’s signature (and call it to bind kwargs/defaults), then hand the positional tuple
    # off to DifferentiableStub.apply; the stub inside raised.py stays untouched
    op = wrap_with_kwargs(stub_fn, override_cls)
    return stub_name, op, ns


# these aren't called in the main loop anymore, because each (run_gradcheck_child, run_bench_child) already create an agent
# so from inside run_gradcheck_child and run_bench_child can just directly call compile_kernel (will run in the main process as which e.g. run_gradcheck_child spawned)
# these are only called once outside the main loop


def _compile_child(fwd_fp: str, generated_fp: str, q):
    """
    Child process worker for the compile probe.

    Rationale:
      - Run compile+preflight in an isolated CUDA context so OOB or device faults
        cannot poison the parent's context. Reconstruct everything in the child
        to avoid passing non-picklable callables/tensors across processes.
      - Do NOT return live tensors; only a small status/message via a Queue.
    """
    try:
        # Make launch sync so device-side assert triggers here, not later.
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
        # Mark worker type for any child-side gating; also guard recursion elsewhere.
        os.environ["KERNEL_AGENT_WORKER"] = "compile"
        # Lazy import to avoid circular imports at module load time
        from .utils import compile_kernel
        # Import torch lazily inside the child process only
        import torch as _t
        # If CUDA is unavailable, treat as a hard error and exit child.
        if not _t.cuda.is_available():
            # Signal fatal probe failure to parent and return cleanly; parent raises.
            q.put({"etype": "RuntimeError", "emsg": "no_cuda"})
            q.close(); q.join_thread()
            os._exit(1)

        # Reconstruct everything fresh in the child so any CUDA faults stay isolated.
        # FLOW: compile_kernel executes fwd_fp (user's forward file), not the generated backward path:
        # - runpy.run_path(fwd_fp) creates fresh kernel objects, runs @autodiff (tags stub)
        # - setup() runs once, compiling the Triton kernel and populating make_args / SWEEP
        # - Runs setup() to surface runtime faults; nothing is returned
        compile_kernel(fwd_fp, generated_fp)

        # Force async CUDA errors to surface NOW at compile_kernel, not later at unrelated code.
        # Prevents kernel bugs from surfacing at torch.empty() with misleading tracebacks.
        # See _gradcheck_child:207-212 for detailed explanation with real example.
        _t.cuda.synchronize()
        q.close(); q.join_thread()
        os._exit(0)
    except Exception as e:
        # Surface compile errors to LLM with full traceback. Without traceback, LLM sees generic "KeyError"
        # but can't tell what key or where, leading to blind guessing instead of targeted fixes.
        # Examples caught here:
        #   - SyntaxError: invalid Python syntax in generated backward (LLM mangled indentation/syntax)
        #   - NameError: name 'triton' not defined (missing import in generated backward)
        #   - AttributeError: 'NoneType' has no attribute 'shape' (LLM used wrong variable)
        #   - KeyError: 'backward_stub' (RAG kernel uses wrong stub name, expected specific name)
        #   - TypeError: kernel() missing required argument (signature mismatch between stub and kernel)
        import traceback
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        q.put({"etype": type(e).__name__, "emsg": str(e), "traceback": tb})
        q.close(); q.join_thread()
        # avoid torch.cuda.synchronize() and any cudaDeviceReset() here, bc
        # exiting the child process is sufficient to release GPU resources
        # and isolate device faults; additional sync/reset is unnecessary and brittle
        # across CUDA backends/builds. Process exit will destroy the child's context.
        # Hard-exit to guarantee isolation; parent inspects exit code and status
        os._exit(1)

# def run_compile_child(fwd_fp: str, generated_fp: str):
#     """
#     Spawn a short-lived child that runs compile_kernel(...) to validate the forward module in isolation.
#     Returns True on success; otherwise raises with the child’s structured error payload.
#     """
#     # Lazy import to avoid circular imports and pick up runtime-configured timeout
#     from .utils import CODE_EXEC_TIMEOUT_S
#     ctx = mp.get_context("spawn")  # never 'fork' with CUDA (unsafe with GPU)
#     q = ctx.Queue(1)
#     # Use non-daemon child so resources flush cleanly; explicitly join below.
#     p = ctx.Process(target=_compile_child, args=(fwd_fp, generated_fp, q), daemon=False)
#     p.start()
#     try:
#         payload = q.get(timeout=CODE_EXEC_TIMEOUT_S)
#     except (queue.Empty, EOFError):
#         # print(f"[probe] queue wait failed: {type(e).__name__}: {e}; alive={p.is_alive()}, exitcode={p.exitcode}")
#         # Child wedged or died before posting a status – kill and escalate
#         p.terminate(); p.join(1.0)
#         if p.is_alive():
#             getattr(p, "kill", p.terminate)()
#             p.join()
#         raise TimeoutError("compile probe hung or child died without reporting")
#     finally:
#         q.close()
#     p.join()
#     # Use child exit code to decide success; queue carries payload only.
#     if p.exitcode != 0:
#         # Attach structured payload from child so orchestrator can extract full traceback.
#         # payload = {"etype": "KeyError", "emsg": "'backward_stub'", "traceback": "...full frames..."}
#         # Orchestrator's run_with_fix will extract this via hasattr(err, 'worker_payload') and
#         # filter the traceback before showing to LLM (removing middleware frames, keeping user code).
#         err = RuntimeError(f"compile probe failed: {payload.get('etype', 'Unknown')}: {payload.get('emsg', str(payload))}")
#         err.worker_payload = payload  # Attach payload dict as attribute
#         raise err
#     return payload



# -------------------- Child runners for full isolation -------------------- #



def _gradcheck_child(fwd_fp: str, generated_fp: str, q):
    # Mark worker so compile_kernel can run preflight safely here only.
    os.environ["KERNEL_AGENT_WORKER"] = "gradcheck"
    # Make launch sync so device-side assert triggers here, not later.
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    status_ok = False
    try:
        # Late import inside the child to avoid importing CUDA stacks in parent.
        from .utils import compile_kernel
        import torch as _t

        # Validates Triton backward gradients against the PyTorch reference instead of finite differences
        from .tools.gradcheck.core_efficient import (
            check_op_backward_reference_sweep as gradcheck_fn,
        )

        # CRITICAL: compile_kernel runs FIRST in this child process to materialize the
        # forward namespace (make_args, SWEEP, etc.). Later steps load DifferentiableStub from raised.py
        # and call the wrapper produced by _load_generated_op so tests route through DifferentiableStub.apply().
        ns = compile_kernel(fwd_fp, generated_fp)

        # Force async CUDA errors to surface NOW at compile_kernel, not later at unrelated code.
        # Real bug example (iterations 2-7): stride error in kernel → surfaced at torch.empty()
        # Without sync: LLM sees "Error at torch.empty()" (misleading - input generation not broken)
        # With sync: LLM sees "Error after compile_kernel()" (correct - points to actual kernel bug)
        # Cost: Nearly free with CUDA_LAUNCH_BLOCKING=1 already enabled
        _t.cuda.synchronize()

        sidecar = dict(ns)

        _, op, raised_ns = _load_generated_op(generated_fp)
        pytorch_ref = raised_ns.get("pytorch_reference_impl")
        if pytorch_ref:
            sidecar["pytorch_reference_impl"] = pytorch_ref
            if os.environ.get("KERNEL_AGENT_VERBOSE"):
                print(f"[gradcheck] Loaded pytorch_reference_impl from {generated_fp}")

        # Reference mode tolerances (compare against PyTorch autograd gradients)
        atol, rtol = 1e-2, 0.0

        # Check if we're in Phase 1 (forward validation only)
        forward_only = os.environ.get("GRADCHECK_FORWARD_ONLY", "0") == "1"

        ok, stats = gradcheck_fn(
            my_op=op,
            sidecar=sidecar,
            atol=atol,
            rtol=rtol,
            forward_only=forward_only,
        )
        # Temporarily disable final unguarded synchronize here.
        # (Per-shape sync inside core_efficient.py already surfaced and recorded sticky CUDA faults;
        # a final global sync would simply re-raise the same accelerator error and replace the detailed
        # per-shape stats with a generic synchronize traceback.)
        # _t.cuda.synchronize()
        q.put((bool(ok), stats))
        status_ok = True
    # comment:
    # all exceptions (including the structural runtime errors raised by gradcheck helpers)
    except BaseException as e:
        # Catch exceptions during gradcheck and send traceback to parent for LLM.
        # Flow: child puts traceback in queue + exits with code 1 → parent raises RuntimeError
        # with worker_payload → orchestrator's run_with_fix catches it → filters traceback → shows to LLM.
        # So exceptions ARE shown to LLM (not hidden by exit code).
        #
        # Examples caught here (STRUCTURAL errors - exceptions thrown):
        #   - KeyError: 'backward_stub' (stub name mismatch - RAG kernel uses different name)
        #   - TypeError: backward_stub() takes 2 args but 5 given (stub signature doesn't match forward inputs)
        #   - RuntimeError: shape mismatch in backward (stub returns wrong number/shape of gradients)
        #   - AttributeError: accessing undefined variable in backward kernel
        #   - IndexError: out of bounds access in backward kernel logic
        #
        # These structural errors show FULL FILTERED TRACEBACKS to LLM (via orchestrator).
        # This is different from validation errors (dtype mismatches, numerical failures) which are
        # caught inside check functions and only show ERROR MESSAGES (not tracebacks) via summary_text.
        #
        # Note: Numerical failures (gradients numerically wrong by max_abs=0.05) are NOT exceptions.
        # Those complete successfully (no exception) and return (False, stats) above (exit code 0).
        # Orchestrator receives stats dict and shows it to LLM separately (not as filtered traceback).
        import traceback
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        payload = {"etype": type(e).__name__, "emsg": str(e), "traceback": tb}
        q.put(payload)
    finally:
        try:
            q.close(); q.join_thread()
        except Exception:
            pass
        # Exit 0 on success (payload posted), 1 on error; parent gates on exit code
        os._exit(0 if status_ok else 1)


def run_gradcheck_child(fwd_fp: str, generated_fp: str):
    ctx = mp.get_context("spawn")
    q = ctx.Queue(1)
    p = ctx.Process(target=_gradcheck_child, args=(fwd_fp, generated_fp, q), daemon=False)
    p.start()
    try:
        # Queue returns only the payload (parity_ok, stats)
        payload = q.get(timeout=GRADCHECK_TIMEOUT_S)
    except (queue.Empty, EOFError):
        try: p.terminate()
        except Exception: pass
        try:
            p.join(1.0)
            getattr(p, "kill", p.terminate)()
            p.join()
        except Exception:
            pass
        raise TimeoutError("gradcheck worker hung")
    finally:
        try: q.close()
        except Exception: pass
    p.join()
    # Use child exit code to decide success; payload contains (parity_ok, stats).
    # Partial parity is a successful run (exit 0). Parent can inspect stats and run rollback/LLM fix.
    if p.exitcode != 0:
        # Child encountered exception (exit code 1) - raise to trigger orchestrator's run_with_fix.
        # Attach structured payload from child so orchestrator can extract full traceback.
        # payload = {"etype": "TypeError", "emsg": "takes 2 args but 5 given", "traceback": "..."}
        # Flow: raise here → run_with_fix catches → filters traceback → shows to LLM.
        # This handles EXCEPTIONS during gradcheck (stub signature issues, runtime errors).
        # Numerical failures (wrong gradients) exit 0 and return (False, stats) below - not exceptions.
        err = RuntimeError(f"gradcheck failed: {payload.get('etype', 'Unknown')}: {payload.get('emsg', str(payload))}")
        err.worker_payload = payload  # Attach payload dict as attribute
        raise err
    return payload  # (ok: bool, stats: dict)


def _bench_child(fwd_fp: str, generated_fp: str, q):
    os.environ["KERNEL_AGENT_WORKER"] = "bench"
    # Make launch sync so device-side assert triggers here, not later.
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    status_ok = False
    try:
        from .utils import compile_kernel
        # autodiff API is imported on demand inside utils.compile_kernel
        from .tools.benchmark import bench_op
        import torch as _t

        # CRITICAL: compile_kernel runs FIRST in this child process to materialize the
        # forward namespace. Later steps replace op with the wrapper from _load_generated_op so execution routes
        # through DifferentiableStub.apply without touching the stub definition in raised.py.
        ns = compile_kernel(fwd_fp, generated_fp)

        # Force async CUDA errors to surface NOW at compile_kernel, not later at unrelated code.
        # Real bug example (iterations 2-7): stride error in kernel → surfaced at torch.empty()
        # Without sync: LLM sees "Error at torch.empty()" (misleading - input generation not broken)
        # With sync: LLM sees "Error after compile_kernel()" (correct - points to actual kernel bug)
        # Cost: Nearly free with CUDA_LAUNCH_BLOCKING=1 already enabled
        _t.cuda.synchronize()

        sidecar = dict(ns)

        # Load the generated stub wrapper so benchmarking exercises the latest backward implementation
        _, op, _ = _load_generated_op(generated_fp)

        cand = bench_op(op, sidecar, mode="bwd")
        # Force async CUDA errors to surface - if this raises, outer except will handle it
        _t.cuda.synchronize()
        # # Note: GPU cleanup could be done here, but outer except/finally handles it
        # # Try to clean up GPU state before exiting
        # _t.cuda.empty_cache()
        q.put(cand)
        status_ok = True
    except BaseException as e:
        # Surface benchmark execution errors to LLM with full traceback.
        # Examples caught here (rare - bench only runs after parity passes):
        #   - RuntimeError: CUDA OOM during benchmarking (kernel uses too much memory under load)
        #   - RuntimeError: CUDA illegal memory access (kernel has bounds bugs exposed under benchmark stress)
        #   - RuntimeError: device-side assert triggered (kernel correctness bug that didn't show in gradcheck)
        import traceback
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        q.put({"etype": type(e).__name__, "emsg": str(e), "traceback": tb})
    finally:
        try:
            q.close(); q.join_thread()
        except Exception:
            pass
        # Exit 0 on success (payload posted), 1 on error; parent gates on exit code
        os._exit(0 if status_ok else 1)


def run_bench_child(fwd_fp: str, generated_fp: str):
    ctx = mp.get_context("spawn")
    q = ctx.Queue(1)
    p = ctx.Process(target=_bench_child, args=(fwd_fp, generated_fp, q), daemon=False)
    p.start()
    try:
        # Queue returns only the payload (benchmark summary)
        payload = q.get(timeout=BENCH_TIMEOUT_S)
    except (queue.Empty, EOFError):
        try: p.terminate()
        except Exception: pass
        try:
            p.join(1.0)
            getattr(p, "kill", p.terminate)()
            p.join()
        except Exception:
            pass
        raise TimeoutError("bench worker hung")
    finally:
        try: q.close()
        except Exception: pass
    p.join()
    # Use child exit code to decide success; queue payload contains the benchmark summary.
    if p.exitcode != 0:
        # Attach structured payload from child so orchestrator can extract full traceback.
        # payload = {"etype": "RuntimeError", "emsg": "CUDA OOM", "traceback": "..."}
        # Bench errors are rare (only runs after parity passes) and usually catastrophic
        # (OOM, illegal memory access, device asserts). Traceback helps diagnose if fixable.
        err = RuntimeError(f"bench failed: {payload.get('etype', 'Unknown')}: {payload.get('emsg', str(payload))}")
        err.worker_payload = payload  # Attach payload dict as attribute
        raise err
    return payload  # cand dict
