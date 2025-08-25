import os
import queue
import threading
from typing import Any, Callable
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



def compile_kernel(file_path: str, return_ns:bool = False) -> Callable[..., Any]:

    # no need for extract_request -- instead make input file to be a python not json

    def load_function_from_code(code: str) -> dict[str, Any]:
        """
        Execute the provided Python `code` and extract top-level Triton JITFunction.
        * executes code in an isolated namespace,
        * enumerates ALL JITFunction instances,
        """

        local_ns: dict[str, Any] = {}
        try:
            # execute user-provided code in an isolated namespace
            run_with_timeout(lambda: exec(code, local_ns), CODE_EXEC_TIMEOUT_S)
        except Exception as e:
            raise RuntimeError(
                "Failed to execute `code`. Ensure it is valid Python and defines a Triton kernel decorated with @triton.jit.\n"
                "Tip: Import triton and triton.language as tl, and bind the kernel to a top-level name.\n"
                f"Exec error: {e}"
            ) from e

        kernels: dict[str, Any] = {name: value for name, value in local_ns.items() if isinstance(value, JITFunction)}

        if not kernels:
            raise RuntimeError(
                "No Triton JITFunction found.\n"
                "Expected your `code` to define top-level function decorated with @triton.jit, e.g.:\n"
                "@triton.jit\n"
                "def my_kernel(...): ...\n"
            )

        return local_ns

    with open(file_path, "r", encoding="utf-8") as f:
        src = f.read()

    # todo: use "mod = importlib.import_module(file_path)" instead of the below?
    code = compile(src, file_path, "exec")
    ns = load_function_from_code(code)

    setup_fn = ns.get("setup", None)
    try:
        run_with_timeout(lambda: exec(setup_fn.__code__, ns, ns), CODE_EXEC_TIMEOUT_S)
    except Exception as e:
        raise RuntimeError(
            "Failed to execute `setup`. Ensure it creates CUDA tensors and launches the kernel once.\n"
            f"Setup error: {e}"
        ) from e

    # if fn_name not in ns or not callable(ns[fn_name]):
    #     raise RuntimeError(f"File {file_path} must define a callable `{fn_name}` function")

    # todo-now: use decorator to intercept CompiledKernel
    # todo: programatically pick top level kernel (to know which JitFunction to attatch the callback to). For now assume single JITFunction
    if return_ns:
        return ns["stub"], ns
    return ns["stub"]




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






# todo-now: use slicing, don't feed entire file
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
