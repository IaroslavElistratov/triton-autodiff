import os
import queue
import traceback
import threading
from typing import Any
from triton.runtime.jit import JITFunction

# def _norm_bench(x) -> dict[str, float]:
#     """Normalize benchmark() result to {'throughput': float} if possible."""
#     if isinstance(x, (int, float)):
#         return {"throughput": float(x)}
#     if isinstance(x, dict):
#         out = {}
#         for k, v in x.items():
#             if isinstance(v, (int, float)):
#                 out[k.lower()] = float(v)
#         return out
#     # last resort: loose parse from string
#     try:
#         return {"throughput": float(str(x).strip().split()[0])}
#     except Exception:
#         return {}

# def _better(new: dict[str, float], best: dict[str, float] | None, min_rel: float) -> bool:
#     """Strict improvement gate (relative throughput)."""
#     if not new: return False
#     if best is None or "throughput" not in best: return "throughput" in new
#     if "throughput" not in new: return False
#     base = best["throughput"]
#     return new["throughput"] >= base * (1.0 + min_rel)

# def _summ_bench(m: dict[str, float] | None) -> str:
#     if not m: return "no bench yet"
#     return ", ".join(f"{k}={v:.4g}" for k, v in m.items() if isinstance(v, (int, float)))




class CompileError(Exception):
    def __init__(self, info: dict):
        self.info = info
        super().__init__(f"compile_error: {info.get('error_type')}: {info.get('error_message')}")


def compile_kernel(file_path: str, overwrite_fp: str | None = None):
    """
    Execute user's forward module, run its setup(), and return:
        (op_fn, backward_file_pointer, module_namespace)

    - file_path: path to the original forward user file (e.g., matmul.py)
    - overwrite_fp: path to an existing _raised.py with edited bwd kernel+stub.
      If provided, do not re-run the MLIR pass; reuse that file instead.
    """

    def exec_module(src: str) -> dict[str, Any]:
        import types, sys  # local to avoid polluting module scope
        module_name = "__kernel_agent_user__"
        # previsoly, the hook crashed due to fwd_stub.__module__ being None because user code was executed in a plain dict,
        # changed utils.exec_module to create a real module (types.ModuleType("__kernel_agent_user__")), insert it into
        # sys.modules, and execute code in that module’s dict. This guarantees functions (including the stub) have a valid __module__
        mod = types.ModuleType(module_name)
        mod.__file__ = file_path
        sys.modules[module_name] = mod
        code = compile(src, file_path, "exec")
        # execute user code in a real module namespace (gives functions a stable __module__)
        run_with_timeout(lambda: exec(code, mod.__dict__, mod.__dict__), CODE_EXEC_TIMEOUT_S)
        return mod.__dict__

    with open(file_path, "r", encoding="utf-8") as f:
        src = f.read()

    try:
        # sometimes even this step failed, e.g. when model messed up
        # indentation of python kernel function declaration
        ns = exec_module(src)
    except BaseException as e:
        # surface structured error upwards; let the orchestrator decide how to recover
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        msg = str(e)
        err = {
            "phase": "exec_module",
            "error_type": type(e).__name__,
            "error_message": msg,
            "fwd_file": file_path,
            "traceback": tb,
            "context_snippet": _read_snippet(file_path, 200),
        }
        raise CompileError(err) from e


    # accept a Triton JITFunction decorated with autodiff-wrapped helper class
    from ._autodiff_api import StubOverrideDCK as StubDCK

    has_dck = any(isinstance(v, type) and issubclass(v, Stub) for v in ns.values())
    has_kernel = any(isinstance(v, JITFunction) for v in ns.values())
    if not (has_dck or has_kernel):
        # these errors for invalid *fwd* kernel -- these aren't designed for llm, but for a human
        # so not wrapping into CompileError
        raise RuntimeError(
            "Expected your code to define (1) top-level kernel decorated with @triton.jit and (2) a stub function decorated with @autodiff, e.g.:\n"
            "@triton.jit\n"
            "def my_kernel(...): ...\n\n"
            "@autodiff(kernel=my_kernel, ...)\n"
            "@my_stub(...): ...\n"
        )

    setup_fn = ns.get("setup")
    if not callable(setup_fn):
        raise RuntimeError("Expected a top-level setup() that runs the stub once. The stub must call the kernel.")


    bwd_fp = None
    def _exec_setup():
        nonlocal bwd_fp
        if overwrite_fp:
            from ._autodiff_api import autodiff_overwrite_fp
            # this adds the "overwrite_fp" argument to my autograd function
            # so that the hook knows to use the backward from "overwrite_fp",
            # and not the backward created by my mlir pass
            with autodiff_overwrite_fp(overwrite_fp):
                # execute the function body in the same namespace so it can populate
                # names like `compiled_kernel` directly into `ns`
                exec(setup_fn.__code__, ns, ns)
                # not used, keeping for clarity
                bwd_fp = overwrite_fp
        else:
            from ._autodiff_api import record_autodiff_artifacts, get_last_bwd_fp
            with record_autodiff_artifacts():
                exec(setup_fn.__code__, ns, ns)
                # record path to the last generated/selected backward before context resets
                bwd_fp = get_last_bwd_fp()

    # comment: this triggers my callback
    try:
        run_with_timeout(_exec_setup, CODE_EXEC_TIMEOUT_S)
    except BaseException as e:
        # surface structured error upwards; let the orchestrator decide how to recover
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        msg = str(e)
        err = {
            "phase": "compile_triton_kernel",
            "error_type": type(e).__name__,
            "error_message": msg,
            "fwd_file": file_path,
            "traceback": tb,
            "context_snippet": _read_snippet(file_path, 200),
        }
        raise CompileError(err) from e
    op = ns.get("stub")
    if not callable(op):
        raise RuntimeError("Expected a top-level stub(...) to call the kernel.")


    # todo: cleanup the code below
    # Preemptively run backward to be certain that both fwd and bwd well formed;
    # run one small forward + backward to surface syntax/import/JIT issues early.
    # Previously, errors would only show up during the first actual backward (e.g., gradcheck),
    # which could terminate the process late (since nothing will catch an exeption at that time).
    # This preflight keeps the same semantics but fails here where the errors are being caught (as part
    # of compile_kernel and not later, when StubOverrideDCK.backward will be called unguarded e.g.
    # during gradcheck, where a raised exception will crash the program)
    sweep = ns.get("SWEEP")
    make_args = ns.get("make_args")
    if isinstance(sweep, (list, tuple)) and sweep and callable(make_args):
        def _preflight_backward():
            import torch
            dims = sweep[0]
            # Obtain exemplar inputs using the user-provided helper
            args, kwargs = make_args(dims)
            kwargs = dict(kwargs or {})

            # Enable grads for floating-point tensors only (matches typical autograd expectations)
            pos_args = list(args or [])
            for i, v in enumerate(pos_args):
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    pos_args[i] = v.detach().requires_grad_(True)
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    kwargs[k] = v.detach().requires_grad_(True)

            # Forward once via the fused op (stub)
            y = op(*pos_args, **kwargs)
            ys = y if isinstance(y, (tuple, list)) else (y,)
            ys = tuple(t for t in ys if isinstance(t, torch.Tensor))
            if not ys:
                return  # nothing to differentiate

            # Reduce outputs to a scalar to avoid constructing explicit upstreams.
            # This mirrors gradcheck behavior without needing grad_outputs.
            scalar = None
            for t in ys:
                scalar = (t.sum() if scalar is None else scalar + t.sum())

            # Collect grad-requiring inputs (positional + keyword)
            grad_ins = []
            for v in pos_args:
                if isinstance(v, torch.Tensor) and v.requires_grad:
                    grad_ins.append(v)
            for v in kwargs.values():
                if isinstance(v, torch.Tensor) and v.requires_grad:
                    grad_ins.append(v)
            if grad_ins and scalar is not None:
                torch.autograd.grad(scalar, tuple(grad_ins), allow_unused=True)

        try:
            # Use the same timeout guard as other compile-time steps
            run_with_timeout(_preflight_backward, CODE_EXEC_TIMEOUT_S)
        except BaseException as e:
            # Surface early as a structured CompileError so orchestrator can recover
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            msg = str(e)
            err = {
                "phase": "jit_backward",
                "error_type": type(e).__name__,
                "error_message": msg,
                "fwd_file": file_path,
                "bwd_file": bwd_fp,
                "traceback": tb,
                "context_snippet": _read_snippet(bwd_fp, 200) if isinstance(bwd_fp, str) else "",
            }
            raise CompileError(err) from e

    return op, bwd_fp, ns




# Tunable limits (seconds) — configurable via env
CODE_EXEC_TIMEOUT_S = float(os.environ.get("TB_CODE_TIMEOUT_S", "15"))

def run_with_timeout(fn, timeout_s: float):
    """
    Run `fn()` in a background thread and wait up to `timeout_s` seconds.
    Returns fn()'s value or re-raises its exception.
    Raises TimeoutError if it doesn't finish in time.
    """
    q: "queue.Queue[tuple[bool, object]]" = queue.Queue(maxsize=1)

    def _target():
        try:
            q.put((True, fn()))
        except BaseException as e:
            q.put((False, e))

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        # We can't kill the thread, but we can stop blocking the tool.
        raise TimeoutError(f"Execution timed out after {timeout_s:.1f}s")

    ok, payload = q.get_nowait()
    if ok:
        return payload
    raise payload  # type: ignore[misc]






# todo-high: use slicing, don't feed entire file
def _read_snippet(path: str, max_lines: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return "".join(lines[:max_lines])
    except Exception as e:
        return f"(snippet unavailable: {e})"


# def try_make_slice_payload(text: str) -> Optional[str]:
#     """
#     Optional fast path for chunked reads of the generated TTIR file.

#     Input (text): JSON string possibly containing:
#       {"slice": {"digest": "<digest10|full>", "offset": int, "limit": int}}

#     On success: returns a JSON payload string with fields
#       {"digest", "path", "offset", "limit", "data"}
#     If the request is not a slice request, returns None.
#     Raises ValueError on malformed slice requests.
#     """
#     try:
#         obj = json.loads(text)
#     except Exception:
#         return None

#     if not isinstance(obj, dict):
#         return None

#     # Support both legacy {"slice": {...}} and direct parameter objects used by the
#     # triton_backward.slice function-call interface.
#     if "slice" in obj:
#         s = obj["slice"] or {}
#     else:
#         # When called via function interface the JSON itself IS the slice payload.
#         s = obj

#     digest = str(s.get("digest", "")).strip()
#     if not digest:
#         raise ValueError("slice.digest is required")
#     digest10 = digest[:10]
#     offset = int(s.get("offset", 0))
#     limit = int(s.get("limit", 64 * 1024))
#     if offset < 0 or limit <= 0:
#         raise ValueError("slice.offset must be >= 0 and slice.limit must be > 0")

#     out_path = f"generated/{digest10}/out.ttir"
#     with open(out_path, "rb") as fh:
#         fh.seek(offset)
#         chunk = fh.read(limit)

#     payload = json.dumps({
#         "digest": digest10,
#         "path": out_path,
#         "offset": offset,
#         "limit": limit,
#         "data": chunk.decode("utf-8", "replace"),
#     })
#     return payload

# if fmt == "json" or (fmt == "raw" and len(backward_code.encode("utf-8")) > MAX_RAW_BYTES):
# fwd_ttir = compiled_kernel.asm["ttir"]
# digest10 = hashlib.sha256(fwd_ttir.encode()).hexdigest()[:10]
# path = f"generated/{digest10}/out.ttir"
# payload_dict: dict[str, Any] = {"digest": digest10, "path": path}
# if fmt == "raw":
#     payload_dict["note"] = (
#         f"raw TTIR exceeded {MAX_RAW_BYTES} bytes; returning json pointer instead. "
#         "Use triton_backward.slice to read windows."
#     )
# payload = json.dumps(payload_dict)
