import os
import queue
import traceback
import threading
import multiprocessing as mp

from typing import Any

import torch
from triton.runtime.jit import JITFunction


# todo:
# Normalize repetitive wrappers from child/parent to keep the message concise.
# Root cause of repetition: the child stringifies its exception and sends it via
# Queue; the parent wraps that into a RuntimeError; then compile_kernel wraps again
# into CompileError, and log sites often prepend f"{type(e).__name__}: {e}".
# Each layer adds its own "...Error: ..." prefix, producing repeated headers.
# Solution: prefer a structured payload (etype/emsg) and collapse to one line.
# Examples:
#   "CompileError: compile_error: TypeError: ..." -> type="TypeError", msg="..."
#   "TypeError: ..." -> type="TypeError", msg="..."


# todo: remove, not needed anymore. Can just raise e
class CompileError(Exception):
    def __init__(self, info: dict):
        self.info = info
        super().__init__(f"compile_error: {info.get('error_type')}: {info.get('error_message')}")


#  my whole thing with catating compileERROR is that it was there are orther ers taht compile_Kernel raises ad these were meant for the user. so maybe i should create some UserError class? and basically in my run_with_fix check for User error in the expct block so if dedect i cought user er i'l re-raise it
class UserError(Exception):
    pass


def filter_traceback_for_llm(tb_string: str) -> str:
    """
    Filter traceback to show only relevant frames for LLM error fixing.

    Motivation:
    1. Full tracebacks contain 10+ frames of middleware (triton runtime, autodiff backend,
       torch internals) that LLM has no control over - this is noise that distracts from the
       actual problem in user code.
    2. But we can't just filter to user-controlled files, because the LAST frame (where the
       actual error occurs) might be in internal code (e.g., generated code calls triton.language.func()
       which throws internally). If we filter that out, LLM won't see the actual error location.
    3. Solution: Keep LAST frame always (actual error location) + keep frames from user-controlled
       paths (generated/, test/, tools/) + skip middleware frames.
    4. This gives LLM: error location (last frame) + how user code led to it (user frames) without
       20 lines of triton/torch/autodiff internals.

    Examples of what gets filtered:
    - KEPT: generated/abc123/raised.py (LLM-generated backward code - can modify)
    - KEPT: kernel_agent/test/matmul.py (user forward code - shows call context)
    - KEPT: kernel_agent/tools/gradcheck/core.py (user tools - shows what triggered error)
    - FILTERED: kernel_agent/worker.py (infrastructure - LLM can't modify)
    - FILTERED: kernel_agent/utils.py (infrastructure - just plumbing)
    - FILTERED: triton/runtime/jit.py (framework - LLM can't modify)
    - FILTERED: triton/backends/autodiff/hooks.py (framework internals)
    - KEPT: Last frame even if in filtered path (shows actual error location)

    Why string-based filtering instead of traceback object reconstruction (like TensorFlow/Django):
    1. Workers run in separate processes (multiprocessing) - traceback objects cannot be
       pickled/serialized across process boundaries. Workers must convert to strings via
       format_exception() before sending through the queue.
    2. Orchestrator receives strings, not live traceback objects - no access to tb_frame, tb_next, etc.
    3. Don't need to re-raise exceptions with modified tracebacks - just formatting error messages
       for LLM consumption, so string output is the end goal anyway.
    4. String parsing is simpler and sufficient for this use case given the architecture constraints.
    """
    lines = tb_string.split('\n')

    # User-controlled paths that LLM can modify
    # Note: 'test/' and 'tools/' match via substring check (f'/{prefix}' in path),
    # so kernel_agent/test/*.py and kernel_agent/tools/*.py are still matched,
    # but kernel_agent/utils.py, worker.py, orchestrator.py are excluded (infrastructure)
    USER_CONTROLLED_PREFIXES = (
        'generated/',
        # 'test/',
        # 'tools/',
    )

    # Parse traceback into frames
    frames = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith('File "'):
            # Extract file path from: File "path/to/file.py", line 123, in function
            frame_lines = [line]
            i += 1
            # Collect code context lines (typically 1-2 lines after File line)
            # Context lines are indented (start with spaces), stop at non-indented lines (error messages)
            while i < len(lines) and not lines[i].strip().startswith('File "'):
                # Stop at non-indented, non-empty lines (like "NameError: ..." at the end)
                # Check BEFORE appending to avoid including error message in last frame
                if lines[i].strip() and not lines[i].startswith('  '):
                    break
                frame_lines.append(lines[i])
                i += 1

            # Check if this frame is from user-controlled path
            file_path = line.split('"')[1] if '"' in line else ""
            is_user_controlled = any(file_path.startswith(prefix) or f'/{prefix}' in file_path
                                     for prefix in USER_CONTROLLED_PREFIXES)

            frames.append({
                'lines': frame_lines,
                'is_user_controlled': is_user_controlled,
                'file_path': file_path
            })
        else:
            i += 1

    if not frames:
        # No frames found, return original
        return tb_string

    # Always keep last frame (actual error location)
    # Keep all user-controlled frames
    # Skip middleware frames (triton, torch, autodiff internals)
    kept_frames = []
    skipped_count = 0

    for idx, frame in enumerate(frames):
        is_last = (idx == len(frames) - 1)
        if is_last or frame['is_user_controlled']:
            # Insert skipped marker if we skipped frames just before this one
            if skipped_count > 0:
                kept_frames.append(f'  [...skipped {skipped_count} internal frames...]')
                skipped_count = 0
            kept_frames.extend(frame['lines'])
        else:
            skipped_count += 1

    # Reconstruct traceback
    result = '\n'.join(kept_frames)
    return result


def compile_kernel(file_path: str, overwrite_fp: str | None = None):
    """
    Execute user's forward module, run its setup(), and return:
        (op_fn, backward_file_pointer, module_namespace)

    - file_path: path to the original forward user file (e.g., matmul.py)
    - overwrite_fp: path to an existing _raised.py with edited bwd kernel+stub.
      If provided, do not re-run the MLIR pass; reuse that file instead.
    """

    # probe compile in a spawned child before using a newly edited backward in‑process.
    # Needed because if LLM edits introduce OOB or device faults, running compile+preflight
    # in a child contains the failure to that child’s CUDA context. The parent remains
    # healthy and can immediately ask the LLM to fix instead of crashing the whole loop.
    # The probe ignores return values and only reports status; the parent then rebuilds
    # the op in‑process as usual on success.
    #
    # i put the probe call inside the compile_op, because i already define _create_op_with_fix
    # which catches the errs from _create_op and shows them to llm, and to avoid duplicating all the logic;
    # That keeps logic in one place, _create_op_with_fix surface both compile and probe failures
    # to the LLM, and avoids duplicating paths

    # In parent (orchestrator.py) run a probe compile in a child before executing user code here.
    # Avoid recursion by checking a child flag (because run_compile_child calls compile_kernel)

    # If i launch a child once and it OOB's and messes the context, then (e.g. on the next iteration of the orchestrator loop)
    # I'll launch another child and that child will have a fresh working context
    if not os.environ.get("KERNEL_AGENT_PROBE_CHILD"):
        try:
            # Import here to avoid circular import when worker imports utils.
            from .worker import run_compile_child
            run_compile_child(file_path, overwrite_fp=overwrite_fp)
        except Exception as e:
            # surface as CompileError so the orchestrator can prompt the LLM to fix
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            msg = str(e)
            err = {
                "phase": "compile_probe",
                "error_type": type(e).__name__,
                "error_message": msg,
                # "fwd_file": file_path,
                "traceback": tb,
                "context_snippet": _read_snippet(file_path, 200),
            }
            raise CompileError(err) from e


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
            # "fwd_file": file_path,
            "traceback": tb,
            "context_snippet": _read_snippet(file_path, 200),
        }
        raise CompileError(err) from e

    # Validate presence of a Triton kernel and an @autodiff-decorated stub.
    # Don't check for a StubDCK subclass here, bc the decorator returns a callable function,
    # and the internal autograd class is not exposed as a public subclass in the user module.
    # Instead i made the @autograd decorator tag the stub (__is_autodiff_stub__), so check for that
    has_kernel = any(isinstance(v, JITFunction) for v in ns.values())
    has_stub = any(
        callable(v) and bool(getattr(v, "__is_autodiff_stub__", False))
        for v in ns.values()
    )
    if not (has_kernel and has_stub):
        # these errors for invalid *fwd* kernel -- these aren't designed for llm, but for a human
        # so not wrapping into CompileError
        raise UserError(
            "Expected your code to define (1) top-level kernel decorated with @triton.jit and (2) a stub function decorated with @autodiff, e.g.:\n"
            "@triton.jit\n"
            "def my_kernel(...): ...\n\n"
            "@autodiff(kernel=my_kernel, ...)\n"
            "def stub(...): ...\n"
        )

    setup_fn = ns.get("setup")
    if not callable(setup_fn):
        raise UserError("Expected a top-level setup() that runs the stub once. The stub must call the kernel.")

    if not callable(ns.get("make_args")) or not isinstance(ns.get("SWEEP"), (list, tuple)):
        raise UserError("User kernel must define make_args and SWEEP")

    bwd_fp = None
    def _exec_setup():
        nonlocal bwd_fp
        if overwrite_fp:
            # Use the canonical backend re-export to control overwrite path
            from triton.backends.autodiff import autodiff_overwrite_fp
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
            # Record/collect the generated backward path via backend re-export
            from triton.backends.autodiff import record_autodiff_artifacts, get_last_bwd_fp
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
            # "fwd_file": file_path,
            "traceback": tb,
            "context_snippet": _read_snippet(file_path, 200),
        }
        raise CompileError(err) from e
    op = ns.get("stub")
    if not callable(op):
        raise UserError("Expected a top-level stub(...) to call the kernel.")

    # don't run pre-flight bwd here anymore, instead just directly run gradcheck / bench
    # in the child -- if it oob's then no problem. Preemptively running backward was needed here when
    # i tried to do a canary compile_kernel in the child to try to guard oob's before the main
    # process runs but now since gracheck and bench both run in child -- isn't needed anymore.

    return op, bwd_fp, ns



# Tunable limits (seconds) — configurable via env
CODE_EXEC_TIMEOUT_S = float(os.environ.get("TB_CODE_TIMEOUT_S", "90"))

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



def _read_snippet(path: str, max_lines: int | None) -> str:
    """Read file snippet, optionally limiting lines.

    If max_lines is None, reads entire file (no limit).
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        if max_lines is None:
            return "".join(lines)
        return "".join(lines[:max_lines])
    except Exception as e:
        return f"(snippet unavailable: {e})"


FN_NAMES_TO_STRIP = {"torch_fn", "make_args", "setup", "flops"}


# todo: instead of removing unwanted code, maybe change instead
# to select only the desired code (kernel, stub) -- seems cleaner

def redact_torch_fn(path: str, max_lines: int | None = None) -> str:
    """Redact prompt-only helpers from the forward source for LLM.

    Removes these top-level items (module scope only):
      - def torch_fn(...): (with decorators)
      - def make_args(...):, def setup(...):, def flops(...):
      - Any assignment to SWEEP (Assign, AnnAssign, AugAssign)

    Runtime source remains untouched (this function only returns text). If AST
    parsing fails, the original text is returned. When max_lines is provided,
    the result is cropped to the first max_lines lines.
    """
    import ast
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            src = fh.read()
    except Exception:
        return ""

    # Collect spans (1-based inclusive line ranges) to delete
    spans: list[tuple[int, int]] = []

    # Parse AST; if syntax is invalid, keep the text unchanged
    try:
        tree = ast.parse(src)
    except SyntaxError:
        tree = None

    if tree is not None:
        for node in getattr(tree, "body", []):
            # Strip selected top-level functions (with decorators)
            if isinstance(node, ast.FunctionDef) and getattr(node, "name", "") in FN_NAMES_TO_STRIP:
                decos = getattr(node, "decorator_list", []) or []
                start_line = min([getattr(d, "lineno", node.lineno) for d in decos] + [node.lineno])
                end_line = getattr(node, "end_lineno", None) or node.lineno
                spans.append((int(start_line), int(end_line)))
                continue

            # Strip top-level SWEEP assignments (Assign / AnnAssign / AugAssign)
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                # Normalize targets to a list of top-level names
                names: list[str] = []
                if isinstance(node, ast.Assign):
                    for tgt in getattr(node, "targets", []) or []:
                        if isinstance(tgt, ast.Name):
                            names.append(getattr(tgt, "id", ""))
                else:
                    tgt = getattr(node, "target", None)
                    if isinstance(tgt, ast.Name):
                        names.append(getattr(tgt, "id", ""))
                if "SWEEP" in names:
                    start_line = getattr(node, "lineno", None)
                    end_line = getattr(node, "end_lineno", None) or start_line
                    if start_line is not None:
                        spans.append((int(start_line), int(end_line)))

        if spans:
            # Delete from bottom to top to keep line indices stable
            lines = src.splitlines(True)
            for start_line, end_line in sorted(spans, key=lambda t: t[0], reverse=True):
                start_line = max(1, int(start_line))
                end_line = max(start_line, int(end_line))
                del lines[start_line - 1:end_line]
            src = "".join(lines)

    if isinstance(max_lines, int) and max_lines > 0:
        # When cropping is requested, first reduce to the first `max_lines` lines
        # but do not alter internal formatting beyond that.
        src = "".join(src.splitlines(True)[:max_lines])

    # Normalize trailing newlines at EOF:
    # - Remove only newline characters to avoid extra blank lines
    # - If non-empty, one final newline
    # This keeps internal spacing intact while removing only the suffix clutter.
    src = src.rstrip("\n\r")
    if src:
        src = src + "\n"
    return src


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



def save_file_bytes(path: str) -> tuple[bool, bytes]:
    """Capture existence and raw bytes of a file for potential rollback.

    Returns (existed_before, bytes). On read errors returns empty bytes.
    """
    existed = os.path.isfile(path)
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except Exception:
        data = b""
    return existed, data


def restore_file_bytes(path: str, existed_before: bool, prev_bytes: bytes) -> Exception | None:
    """Best-effort restore of a file's previous bytes after a failed write.

    - If the file existed before, rewrites previous bytes.
    - If it did not exist before, removes any newly created file.
    Returns an Exception on restore failure, else None.
    """
    try:
        if existed_before:
            with open(path, "wb") as fh:
                fh.write(prev_bytes)
        else:
            if os.path.exists(path):
                os.remove(path)
        return None
    except Exception as e:
        return e

def _env_truthy(name: str, default: str = "0") -> bool:
    """Parse boolean-like env flags from environment."""
    val = os.environ.get(name, default)
    return str(val).lower() not in ("0", "", "false", "no", "off")
