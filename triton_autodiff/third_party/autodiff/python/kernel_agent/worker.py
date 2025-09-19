import os
import multiprocessing as mp
import queue
import traceback
 
import sys

# todo-high: ugly, restructure project folders to solve that

# Dedicated timeouts for long-running child tasks (env-overridable)
def _ensure_triton_autodiff_api_alias() -> None:
    """Ensure sys.modules has 'triton_autodiff_api' registered.
    Tries normal import; on failure, imports api.py by path which registers the alias.
    """
    if "triton_autodiff_api" in sys.modules:
        return
    # Prefer loading the package __init__ directly under the canonical alias,
    # regardless of whether the dotted import is available, to guarantee aliasing.
    import importlib.util
    here = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(here, "../../../.."))
    api_dir = os.path.join(repo_root, "third_party", "autodiff", "python", "api")
    api_init = os.path.join(api_dir, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        "triton_autodiff_api", api_init, submodule_search_locations=[api_dir]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["triton_autodiff_api"] = mod
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)

# Dedicated timeouts for long-running child tasks (env-overridable)
GRADCHECK_TIMEOUT_S = float(os.environ.get("TB_GRADCHECK_TIMEOUT_S", "180"))
BENCH_TIMEOUT_S     = float(os.environ.get("TB_BENCH_TIMEOUT_S", "180"))


# comment:
# these aren't called anymore, because each (run_gradcheck_child, run_bench_child) already create an agent
# so from inside run_gradcheck_child and run_bench_child can just directly call compile_kernel (will run in the main process as which e.g. run_gradcheck_child spawned)


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
        # Ensure triton_autodiff_api alias is registered for _autodiff_api consumers
        _ensure_triton_autodiff_api_alias()
        # Import torch lazily inside the child process only
        import torch as _t
        # Mark this process as the probe child so compile_kernel won't spawn again.
        os.environ["KERNEL_AGENT_PROBE_CHILD"] = "1"

        # If CUDA is unavailable, treat as a hard error and exit child.
        if not _t.cuda.is_available():
            # Signal fatal probe failure to parent and return cleanly; parent raises.
            q.put(("err", "no_cuda"))
            q.close(); q.join_thread()
            os._exit(1)
        # Reconstruct everything fresh in the child; this triggers the same
        # compile path and the in-process backward preflight inside compile_kernel.
        # Capture bwd_fp to return to the parent (ns is not pickleable; return only bwd_fp).
        _op, _bwd_fp, _ns = compile_kernel(fwd_fp, overwrite_fp=overwrite_fp)
        # Ensure any pending device work (e.g., preflight backward) is observed before exit,
        # so device-side asserts surface in this child, not later in the parent.
        _t.cuda.synchronize()
        q.put(("ok", _bwd_fp))
        q.close(); q.join_thread()
        os._exit(0)
    except Exception as e:
        # First report error to parent and flush queue so parent never hangs.
        # Deliberately omit traceback from payload to keep LLM-facing messages concise
        # and avoid leaking file paths. Full tracebacks are still available in logs.
        try:
            q.put(("err", {
                "etype": type(e).__name__,
                "emsg": str(e),
            }))
        except Exception:
            pass
        q.close(); q.join_thread()
        # Intentionally avoid torch.cuda.synchronize() and any cudaDeviceReset() here.
        # Rationale: exiting the child process is sufficient to release GPU resources
        # and isolate device faults; additional sync/reset is unnecessary and brittle
        # across CUDA backends/builds. Process exit will destroy the child's context.
        # Hard-exit to guarantee isolation; parent inspects exit code and status.
        # Reason for not throwing exception instead of os.exit is it seems the child process
        # shouldn’t rely on raising exceptions across process boundaries?
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
        status, payload = q.get(timeout=CODE_EXEC_TIMEOUT_S)
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
    if status != "ok":
        raise RuntimeError(f"compile probe failed: {payload}")
    if p.exitcode != 0:
        raise RuntimeError(f"compile probe child exited with code {p.exitcode} (status={status})")
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
    try:
        # Late import inside the child to avoid importing CUDA stacks in parent.
        from .utils import compile_kernel
        _ensure_triton_autodiff_api_alias()
        from .tools.gradcheck.core import check_op_backward_parity_sweep
        import torch as _t

        op, _, ns = compile_kernel(fwd_fp, overwrite_fp=overwrite_fp)
        ok, stats = check_op_backward_parity_sweep(
            ref_fwd=ns["torch_fn"], my_op=op, sidecar=ns, outputs="auto",
            # tests/mamtul: backward casts to fp16 before dot and accumulates/atomics in fp16, while Torch grads accumulate in fp32;
            # later proper fix: keep accumulators fp32 and cast only at tl.atomic_add
            atol=0.07, rtol=0.02
        )
        try:
            _t.cuda.synchronize()
        except Exception:
            pass
        q.put(("ok", (bool(ok), stats)))
    except BaseException as e:
        q.put(("err", {"etype": type(e).__name__, "emsg": str(e)}))
    finally:
        try:
            q.close(); q.join_thread()
        except Exception:
            pass
        os._exit(0 if 'ok' in locals() and ok else 1)


def run_gradcheck_child(fwd_fp: str, overwrite_fp: str | None):
    ctx = mp.get_context("spawn")
    q = ctx.Queue(1)
    p = ctx.Process(target=_gradcheck_child, args=(fwd_fp, overwrite_fp, q), daemon=False)
    p.start()
    try:
        status, payload = q.get(timeout=GRADCHECK_TIMEOUT_S)
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
    if status != "ok" or p.exitcode != 0:
        raise RuntimeError(f"gradcheck failed: {payload}")
    return payload  # (ok: bool, stats: dict)


def _bench_child(fwd_fp: str, overwrite_fp: str | None, q):
    os.environ["KERNEL_AGENT_WORKER"] = "bench"
    # Prevent compile_kernel from spawning another probe child recursively.
    os.environ["KERNEL_AGENT_PROBE_CHILD"] = "1"
    # Make launch sync so device-side assert triggers here, not later.
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    try:
        from .utils import compile_kernel
        _ensure_triton_autodiff_api_alias()
        from .tools.benchmark import bench_op, reduce_bench
        import torch as _t

        op, _, ns = compile_kernel(fwd_fp, overwrite_fp=overwrite_fp)
        recs = bench_op(op, ns, mode="bwd")
        cand = reduce_bench(recs)
        try:
            _t.cuda.synchronize()
        except Exception:
            pass
        q.put(("ok", cand))
    except BaseException as e:
        # Keep payload minimal: no traceback for model consumption.
        q.put(("err", {
            "etype": type(e).__name__,
            "emsg": str(e),
        }))
    finally:
        try:
            q.close(); q.join_thread()
        except Exception:
            pass
        os._exit(0 if 'cand' in locals() else 1)


def run_bench_child(fwd_fp: str, overwrite_fp: str | None):
    ctx = mp.get_context("spawn")
    q = ctx.Queue(1)
    p = ctx.Process(target=_bench_child, args=(fwd_fp, overwrite_fp, q), daemon=False)
    p.start()
    try:
        status, payload = q.get(timeout=BENCH_TIMEOUT_S)
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
    if status != "ok" or p.exitcode != 0:
        raise RuntimeError(f"bench failed: {payload}")
    return payload  # cand dict


