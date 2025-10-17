import os
import multiprocessing as mp
import queue
import traceback

import sys

# ============================================================================
# KERNEL AGENT WORKER EXECUTION FLOW (RAG MODE)
# ============================================================================
#
# High-level workflow:
#
# 0. Orchestrator retrieves backward kernel from RAG index → writes to raised.py (backward only)
#
# 1. Orchestrator spawns child process → calls compile_kernel(fwd_fp=attention.py, overwrite_fp=raised.py)
#    - compile_kernel EXECUTES attention.py (forward file), NOT raised.py
#    - overwrite_fp tells hook: "use this backward file, skip MLIR passes"
#
# 2. Inside compile_kernel:
#    - runpy.run_path(attention.py) creates fresh kernel objects, runs @autodiff decorator
#    - @autodiff creates proxies (_fwd_stub_proxy, _bwd_stub_proxy) that late-bind to stubs
#    - Proxies needed because decorator runs at import time (backward doesn't exist yet)
#    - setup() calls stub → triggers kernel compilation → hook fires
#
# 3. Hook (in api/new.py) executes during kernel compilation:
#    - Prepends forward kernel+stub source to raised.py (now has both forward and backward)
#    - Loads both stubs: runpy.run_path(raised.py) → creates separate namespace
#    - Sets attributes: kernel._generated_fwd_stub = fwd_stub_from_raised
#                       kernel._generated_bwd_stub = bwd_stub_from_raised
#
# 4. Proxy mechanism enables runtime binding:
#    - First call: proxy uses original stub from attention.py (triggers compilation)
#    - After hook: proxy uses stub from raised.py (getattr(kernel, "_generated_fwd_stub"))
#    - This stub calls kernels from raised.py namespace (different JITFunction objects)
#    - LLM edits to raised.py take effect immediately via proxy redirection
#
# 5. Gradcheck/benchmark execute in SAME child process:
#    - They call op(q, k, v) many times
#    - Each call → proxy → uses stub from raised.py (set by hook in step 3)
#    - Result: Gradcheck ALWAYS uses stubs/kernels from raised.py, not from user file
#
# 6. Across iterations:
#    - Each iteration spawns fresh child → compile_kernel runs AGAIN
#    - Hook fires AGAIN → reloads edited raised.py → sets attributes in new child
#    - Gradcheck uses newly loaded stubs (LLM edits picked up)
#
# Key insight: Process-local attributes are fine because compile_kernel recreates them
# in every child before gradcheck runs. No state needs to persist across processes.
#
# ============================================================================

# Dedicated timeouts for long-running child tasks (env-overridable)
GRADCHECK_TIMEOUT_S = float(os.environ.get("TB_GRADCHECK_TIMEOUT_S", "180"))
BENCH_TIMEOUT_S     = float(os.environ.get("TB_BENCH_TIMEOUT_S", "180"))



# these aren't called in the main loop anymore, because each (run_gradcheck_child, run_bench_child) already create an agent
# so from inside run_gradcheck_child and run_bench_child can just directly call compile_kernel (will run in the main process as which e.g. run_gradcheck_child spawned)
# these are only called once outside the main loop


def _compile_child(fwd_fp: str, overwrite_fp: str | None, q):
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
        # Mark this process as the probe child so compile_kernel won't spawn again.
        os.environ["KERNEL_AGENT_PROBE_CHILD"] = "1"

        # If CUDA is unavailable, treat as a hard error and exit child.
        if not _t.cuda.is_available():
            # Signal fatal probe failure to parent and return cleanly; parent raises.
            q.put({"etype": "RuntimeError", "emsg": "no_cuda"})
            q.close(); q.join_thread()
            os._exit(1)
        # Reconstruct everything fresh in the child; this triggers the same
        # compile path and the in-process backward preflight inside compile_kernel.
        # Capture bwd_fp to return to the parent (ns is not pickleable; return only bwd_fp).
        #
        # FLOW: compile_kernel executes fwd_fp (user's forward file), NOT overwrite_fp:
        # - runpy.run_path(fwd_fp) creates fresh kernel objects, runs @autodiff
        # - setup() triggers compilation → hook fires → prepends forward to raised.py
        # - Hook loads stubs from raised.py → sets kernel._generated_{fwd,bwd}_stub
        # - Result: All subsequent calls use stubs from raised.py via proxy mechanism
        _op, bwd_fp, _ns = compile_kernel(fwd_fp, overwrite_fp=overwrite_fp)
        # Ensure any pending device work (e.g., preflight backward) is observed before exit,
        # so device-side asserts surface in this child, not later in the parent.
        # todo: but that seems to be device-wide
        _t.cuda.synchronize()
        q.put(bwd_fp)
        q.close(); q.join_thread()
        os._exit(0)
    except Exception as e:
        # Surface compile errors to LLM with full traceback. Without traceback, LLM sees generic "KeyError"
        # but can't tell what key or where, leading to blind guessing instead of targeted fixes.
        # Examples caught here:
        #   - SyntaxError: invalid Python syntax in generated backward (LLM mangled indentation/syntax)
        #   - NameError: name 'triton' not defined (missing import in generated backward)
        #   - AttributeError: 'NoneType' has no attribute 'shape' (LLM used wrong variable)
        #   - KeyError: 'backward_stub' (RAG kernel uses wrong stub name, compiler expects specific name)
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

def run_compile_child(fwd_fp: str, overwrite_fp: str | None = None):
    """
    Spawn a short-lived child that runs compile_kernel(...) as a probe.

    Why here and why child:
      - compile_kernel includes a minimal backward preflight; running it in a child
        contains OOB/device faults so the main agent loop can continue and let the LLM fix.
      - Only the backward file path (string) is returned to avoid pickling large objects.
    """
    # Lazy import to avoid circular imports and pick up runtime-configured timeout
    from .utils import CODE_EXEC_TIMEOUT_S
    ctx = mp.get_context("spawn")  # never 'fork' with CUDA (unsafe with GPU)
    q = ctx.Queue(1)
    # Use non-daemon child so resources flush cleanly; explicitly join below.
    p = ctx.Process(target=_compile_child, args=(fwd_fp, overwrite_fp, q), daemon=False)
    p.start()
    try:
        payload = q.get(timeout=CODE_EXEC_TIMEOUT_S)
    except (queue.Empty, EOFError):
        # print(f"[probe] queue wait failed: {type(e).__name__}: {e}; alive={p.is_alive()}, exitcode={p.exitcode}")
        # Child wedged or died before posting a status – kill and escalate
        p.terminate(); p.join(1.0)
        if p.is_alive():
            getattr(p, "kill", p.terminate)()
            p.join()
        raise TimeoutError("compile probe hung or child died without reporting")
    finally:
        q.close()
    p.join()
    # Use child exit code to decide success; queue carries payload only.
    if p.exitcode != 0:
        # Attach structured payload from child so orchestrator can extract full traceback.
        # payload = {"etype": "KeyError", "emsg": "'backward_stub'", "traceback": "...full frames..."}
        # Orchestrator's run_with_fix will extract this via hasattr(err, 'worker_payload') and
        # filter the traceback before showing to LLM (removing middleware frames, keeping user code).
        err = RuntimeError(f"compile probe failed: {payload.get('etype', 'Unknown')}: {payload.get('emsg', str(payload))}")
        err.worker_payload = payload  # Attach payload dict as attribute
        raise err
    # payload is the bwd_fp (string)
    return payload



# -------------------- Child runners for full isolation -------------------- #



def _gradcheck_child(fwd_fp: str, overwrite_fp: str | None, q):
    # Mark worker so compile_kernel can run preflight safely here only.
    os.environ["KERNEL_AGENT_WORKER"] = "gradcheck"
    # Prevent compile_kernel from spawning another probe child recursively.
    os.environ["KERNEL_AGENT_PROBE_CHILD"] = "1"
    # Make launch sync so device-side assert triggers here, not later.
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    status_ok = False
    try:
        # Late import inside the child to avoid importing CUDA stacks in parent.
        from .utils import compile_kernel
        import torch as _t

        # Import gradcheck function - uses PyTorch's torch.autograd.gradcheck
        # Computes numerical gradients via finite differences: (f(x+eps) - f(x-eps))/(2*eps)
        # Compares against analytical gradients from backward kernel
        # Uses ONLY efficient Triton kernel (no naive torch_fn) → O(n) memory
        from .tools.gradcheck.core_efficient import check_op_backward_numerical_sweep as gradcheck_fn

        # CRITICAL: compile_kernel runs FIRST in this child process:
        # - Executes fwd_fp (attention.py) → hook fires → loads stubs from raised.py
        # - Sets kernel._generated_{fwd,bwd}_stub in THIS process
        # - Returns op that uses proxies pointing to these attributes
        op, _, ns = compile_kernel(fwd_fp, overwrite_fp=overwrite_fp)

        # Phase-0 throttling: optionally limit SWEEP to the first shape via env.
        # None -- a sentinel meaning "all shapes"
        limit = os.environ.get("KERNEL_AGENT_SWEEP_LIMIT")
        sidecar = dict(ns)
        if limit is not None and isinstance(sidecar.get("SWEEP"), (list, tuple)):
            limit = int(limit)
            sidecar["SWEEP"] = list(sidecar["SWEEP"])[:limit]

        # Gradcheck calls op many times, each call:
        # - op(q, k, v) → proxy → getattr(kernel, "_generated_fwd_stub")
        # - Uses stub from raised.py (loaded by hook above)
        # - Stub calls kernels from raised.py namespace (different JITFunction objects)
        # Result: Gradcheck ALWAYS uses stubs/kernels from raised.py, NOT user file

        # Call gradcheck function with numerical gradients via PyTorch's gradcheck
        ok, stats = gradcheck_fn(
            my_op=op, sidecar=sidecar, outputs="auto",
            atol=0.07, rtol=0.02,
            eps=float(os.environ.get("GRADCHECK_EPS", "1e-4")),
            numerical_method=os.environ.get("GRADCHECK_NUMERICAL_METHOD", "central")
        )
        try:
            _t.cuda.synchronize()
        except Exception:
            pass
        q.put((bool(ok), stats))
        status_ok = True
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
        # Note: Numerical failures (gradients numerically wrong by max_abs=0.05) are NOT exceptions.
        # Those complete successfully (no exception) and return (False, stats) above (exit code 0).
        # Orchestrator receives stats dict and shows it to LLM separately (not as filtered traceback).
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


def run_gradcheck_child(fwd_fp: str, overwrite_fp: str | None):
    ctx = mp.get_context("spawn")
    q = ctx.Queue(1)
    p = ctx.Process(target=_gradcheck_child, args=(fwd_fp, overwrite_fp, q), daemon=False)
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


def _bench_child(fwd_fp: str, overwrite_fp: str | None, q):
    os.environ["KERNEL_AGENT_WORKER"] = "bench"
    # Prevent compile_kernel from spawning another probe child recursively.
    os.environ["KERNEL_AGENT_PROBE_CHILD"] = "1"
    # Make launch sync so device-side assert triggers here, not later.
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    status_ok = False
    try:
        from .utils import compile_kernel
        # autodiff API is imported on demand inside utils.compile_kernel
        from .tools.benchmark import bench_op
        import torch as _t

        # CRITICAL: compile_kernel runs FIRST in this child process:
        # - Executes fwd_fp (attention.py) → hook fires → loads stubs from raised.py
        # - Sets kernel._generated_{fwd,bwd}_stub in THIS process
        # - Returns op that uses proxies pointing to these attributes
        # Result: Benchmark ALWAYS uses stubs/kernels from raised.py, NOT user file
        op, _, ns = compile_kernel(fwd_fp, overwrite_fp=overwrite_fp)

        # Phase-0 throttling: optionally limit SWEEP to the first shape via env.
        limit = os.environ.get("KERNEL_AGENT_SWEEP_LIMIT")
        sidecar = dict(ns)
        if limit is not None and isinstance(sidecar.get("SWEEP"), (list, tuple)):
            limit = int(limit)
            sidecar["SWEEP"] = list(sidecar["SWEEP"])[:limit]
        cand = bench_op(op, sidecar, mode="bwd")
        try:
            _t.cuda.synchronize()
        except Exception:
            pass
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


def run_bench_child(fwd_fp: str, overwrite_fp: str | None):
    ctx = mp.get_context("spawn")
    q = ctx.Queue(1)
    p = ctx.Process(target=_bench_child, args=(fwd_fp, overwrite_fp, q), daemon=False)
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


