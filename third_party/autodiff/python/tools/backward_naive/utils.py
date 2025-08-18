from __future__ import annotations

from typing import Tuple, Any, Optional

import json
import re
import queue
import threading


def _unfence_json(text: str) -> Optional[str]:
    m = re.search(r"```json\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else None


def _from_fenced_sections(text: str) -> Optional[dict]:
    def grab(tag: str) -> Optional[str]:
        m = re.search(rf"```{tag}\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else None
    code, setup = grab("code"), grab("setup")
    if code and setup:
        obj: dict[str, Any] = {"code": code, "setup": setup}
        return obj
    return None


def extract_request(text: str) -> Tuple[str, str, str]:
    """
    Returns: (code, setup, fmt) where fmt ∈ {"raw", "json"}.
    Accepts:
      - JSON with fields: code, setup, optional format ("raw"|"json") or json (bool)
      - Fenced ```json ...```
      - Fenced blocks: ```code```, ```setup```
    """
    obj: Any = None
    # 1) plain JSON
    try:
        obj = json.loads(text)
    except Exception:
        pass
    # 2) fenced JSON
    if obj is None:
        fenced = _unfence_json(text)
        if fenced:
            try:
                obj = json.loads(fenced)
            except Exception:
                pass
    # 3) fenced sections
    if obj is None:
        obj = _from_fenced_sections(text)

    if not isinstance(obj, dict):
        raise ValueError(
            "Invalid request. Provide JSON ({code, setup, [format|json]}) "
            "or fenced ```code``` and ```setup``` blocks."
        )

    # Required
    code = obj.get("code", "")
    setup = obj.get("setup", "")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Field `code` must be a non-empty string.")
    if not isinstance(setup, str) or not setup.strip():
        raise ValueError("Field `setup` must be a non-empty string.")

    # setup must perform any necessary compilation/calls

    # Output format
    fmt = obj.get("format", None)
    if fmt is None:
        # Default to JSON; honor legacy json: true/false knob if present
        fmt = "json" if bool(obj.get("json", True)) else "raw"
    if fmt not in ("raw", "json"):
        raise ValueError("`format` must be 'raw' or 'json'.")

    return code, setup, fmt


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

    # 1) prefer explicit `COMPILED_KERNEL` / `_compiled_kernel`
    if "COMPILED_KERNEL" in ns and _is_compiled_kernel(ns["COMPILED_KERNEL"]):
        return ns["COMPILED_KERNEL"]
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
            "Setup did not yield a compiled kernel. Fix: Capture the LAUNCH object from a Triton call and assign it to COMPILED_KERNEL (or _compiled_kernel),\n"
            "e.g. `COMPILED_KERNEL = my_kernel[(1,1,1)](a,b,c,o)` or `_, COMPILED_KERNEL = stub(a,b,c)`."
        )
    raise RuntimeError(
        "Multiple compiled kernels found. Assign the one you want to `_compiled_kernel` to disambiguate."
    )


def try_make_slice_payload(text: str) -> Optional[str]:
    """
    Optional fast path for chunked reads of the generated TTIR file.

    Input (text): JSON string possibly containing:
      {"slice": {"digest": "<digest10|full>", "offset": int, "limit": int}}

    On success: returns a JSON payload string with fields
      {"digest", "path", "offset", "limit", "data"}
    If the request is not a slice request, returns None.
    Raises ValueError on malformed slice requests.
    """
    try:
        obj = json.loads(text)
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None

    # Support both legacy {"slice": {...}} and direct parameter objects used by the
    # triton_backward.slice function-call interface.
    if "slice" in obj:
        s = obj["slice"] or {}
    else:
        # When called via function interface the JSON itself IS the slice payload.
        s = obj

    digest = str(s.get("digest", "")).strip()
    if not digest:
        raise ValueError("slice.digest is required")
    digest10 = digest[:10]
    offset = int(s.get("offset", 0))
    limit = int(s.get("limit", 64 * 1024))
    if offset < 0 or limit <= 0:
        raise ValueError("slice.offset must be >= 0 and slice.limit must be > 0")

    out_path = f"generated/{digest10}/out.ttir"
    with open(out_path, "rb") as fh:
        fh.seek(offset)
        chunk = fh.read(limit)

    payload = json.dumps({
        "digest": digest10,
        "path": out_path,
        "offset": offset,
        "limit": limit,
        "data": chunk.decode("utf-8", "replace"),
    })
    return payload

