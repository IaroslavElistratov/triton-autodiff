from __future__ import annotations

from typing import Tuple, Any

import json
import queue
import threading


def extract_request(text: str) -> Tuple[str, str]:
    """
    Validate the JSON payload for the Triton backward tool.

    Required:
      - code:  str (Python module defining exactly one @triton.jit kernel)
      - setup: str (Python snippet that prepares inputs AND runs the kernel once)

    Returns (code, setup).
    """
    try:
        obj = json.loads(text)
    except Exception as e:
        raise ValueError(f"Invalid JSON: {e}")

    if not isinstance(obj, dict):
        raise ValueError("Invalid request: top-level JSON must be an object.")

    REQUIRED = {"code", "setup"}
    ALLOWED = REQUIRED

    missing = sorted(REQUIRED - obj.keys())
    extra = sorted(set(obj.keys()) - ALLOWED)
    if missing or extra:
        problems: list[str] = []
        if missing:
            problems.append(f"missing={missing}")
        if extra:
            problems.append(f"unexpected={extra}")
        raise ValueError(
            "Invalid request. " + "; ".join(problems) + ". Allowed keys: " + ", ".join(sorted(ALLOWED))
        )

    if "code" not in obj or not isinstance(obj["code"], str) or not obj["code"].strip():
        raise ValueError("Field `code` must be a non-empty string.")
    if "setup" not in obj or not isinstance(obj["setup"], str) or not obj["setup"].strip():
        raise ValueError("Field `setup` must be a non-empty string.")

    code = obj["code"]
    setup = obj["setup"]
    return code, setup


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


def pick_compiled_kernel(ns: dict[str, Any]) -> Any:
    """
    Locate a compiled Triton kernel object in the given namespace.

    Preference order:
      1) `_compiled_kernel` if present and valid
      2) Otherwise, search for exactly one compiled kernel among values and
         shallow elements of tuples/lists.

    Raises RuntimeError with an actionable message if none or multiple are found.
    """
    def _is_compiled_kernel(x: Any) -> bool:
        return hasattr(x, "asm") and isinstance(getattr(x, "asm"), dict) and "ttir" in x.asm

    # 1) prefer explicit `_compiled_kernel`
    if "_compiled_kernel" in ns and _is_compiled_kernel(ns["_compiled_kernel"]):
        return ns["_compiled_kernel"]

    # 2) Otherwise, search namespace values (direct and shallow tuples/lists)
    found: list[Any] = []
    for v in ns.values():
        if _is_compiled_kernel(v):
            found.append(v)
        elif isinstance(v, (tuple, list)):
            for e in v:
                if _is_compiled_kernel(e):
                    found.append(e)

    if len(found) == 1:
        return found[0]
    if len(found) == 0:
        raise RuntimeError(
            "Setup did not yield a compiled kernel. Fix: Modify your stub to return the compiled kernel object and assign it to `_compiled_kernel`, "
            "e.g. `_compiled_kernel = my_kernel[grid](...)` or `_, _compiled_kernel = stub(...)`."
        )
    raise RuntimeError(
        "Multiple compiled kernels found. Assign the one you want to `_compiled_kernel` to disambiguate."
    )

