from __future__ import annotations

from typing import Optional, Tuple, List

import json
import queue
import threading


def extract_request(text: str) -> Tuple[str, List[str], Optional[str]]:

    """
    Validate the JSON payload for the Triton backward tool.

    Required fields:
      - code:        str (non-empty)
      - setup:       str (non-empty)
      - warmup_call: str (non-empty)

    Optional fields: none (kernel selection is automatic; exactly one @triton.jit kernel must be present).

    Returns (code, [warmup_call], setup).
    Raises ValueError with actionable messages on failure.
    """
    REQUIRED = {"code", "setup", "warmup_call"}
    ALLOWED = REQUIRED

    try:
        obj = json.loads(text)
    except Exception as e:
        raise ValueError(f"Invalid JSON: {e}")

    if not isinstance(obj, dict):
        raise ValueError(
            "Invalid request: top-level JSON must be an object with fields: code, setup, warmup_call, (optional) kernel_name."
        )

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

    def _need_nonempty_str(k: str) -> str:
        v = obj.get(k)
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"Field `{k}` must be a non-empty string.")
        return v

    code = _need_nonempty_str("code")
    setup = _need_nonempty_str("setup")
    warmup_call = _need_nonempty_str("warmup_call")

    return code, [warmup_call], setup


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


