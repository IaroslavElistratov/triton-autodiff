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


def filter_traceback_for_llm(tb_string: str, bwd_fp: str) -> str:
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
       paths (generated backward file being edited) + skip middleware frames.
    4. This gives LLM: error location (last frame) + how user code led to it (user frames) without
       20 lines of triton/torch/autodiff internals.

    Args:
        tb_string: Formatted traceback string from format_exception()
        bwd_fp: Path to generated backward file (e.g., "generated/abc123/raised.py")
                Used for exact path matching to identify user-controlled frames.

                FIX: Old prefix-based matching ('generated/' in path) was too aggressive -
                it would hide generated frames in the skip marker when they should be shown to LLM.
                This caused misleading tracebacks like:
                    [...skipped 2 internal frames...]  ← Generated kernel frame hidden here!
                    File "test/attention.py", line 235, in make_args
                        q = torch.empty(...)  ← CUDA error surfaces here (misleading!)

                New approach: Exact path matching ensures generated code frames (WHERE error actually
                occurred) are ALWAYS shown to LLM.

    Examples of what gets filtered:
    - KEPT: generated/abc123/raised.py (matches bwd_fp - LLM is editing this)
    - KEPT: Last frame (shows actual error location)
    - FILTERED: kernel_agent/worker.py (infrastructure - LLM can't modify)
    - FILTERED: kernel_agent/utils.py (infrastructure - just plumbing)
    - FILTERED: triton/runtime/jit.py (framework - LLM can't modify)
    - FILTERED: triton/backends/autodiff/hooks.py (framework internals)

    # todo: include this?
    - [?] KEPT: kernel_agent/tools/gradcheck/core.py (user tools - shows what triggered error)

    Why string-based filtering instead of traceback object reconstruction (like TensorFlow/Django):
    1. Workers run in separate processes (multiprocessing) - traceback objects cannot be
       pickled/serialized across process boundaries. Workers must convert to strings via
       format_exception() before sending through the queue.
    2. Orchestrator receives strings, not live traceback objects - no access to tb_frame, tb_next, etc.
    3. Don't need to re-raise exceptions with modified tracebacks - just formatting error messages
       for LLM consumption, so string output is the end goal anyway.
    4. String parsing is simpler and sufficient for this use case given the architecture constraints.
    """
    import os
    lines = tb_string.split('\n')

    # Normalize bwd_fp for exact path matching
    bwd_fp_normalized = os.path.normpath(bwd_fp)

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

            # Check if this frame is from the generated backward file being edited
            # FIX: Exact path matching prevents overly aggressive filtering that was hiding
            # generated kernel frames in skip markers, causing LLM to see misleading error locations
            file_path = line.split('"')[1] if '"' in line else ""
            is_user_controlled = False

            if bwd_fp_normalized:
                # Check if frame is from the generated file we're editing (exact path match)
                frame_path_normalized = os.path.normpath(file_path)
                # Match if paths are equal or if absolute frame path ends with relative bwd_fp
                # Use os.sep to ensure we match at path boundaries (not substrings)
                is_user_controlled = (
                    frame_path_normalized == bwd_fp_normalized or
                    frame_path_normalized.endswith(os.sep + bwd_fp_normalized) or
                    frame_path_normalized.endswith('/' + bwd_fp_normalized)  # Handle both separators
                )

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


class _TritonLaunchCounter:
    """
    Triton kernel launch counter.

    Guards against PyTorch/stub bypass by counting actual Triton kernel launches.
    Reset before backward pass and check after to ensure backward launches Triton kernels.

    Thread safety note:
    - No lock on count increment: We only check (count == 0), so lost increments don't
      create false negatives. If any launch happened, count > 0 regardless of races.
    - Lock only in install() for idempotent monkey-patching.
    """
    def __init__(self):
        self.count = 0
        self._wrapped = False
        self._orig_run = None
        self._lock = threading.Lock()

    def install(self):
        """
        Monkey-patch JITFunction.run to count launches.
        Idempotent - safe to call multiple times (lock prevents double-patching).

        Fails hard if Triton API changed (no silent degradation).
        """
        if self._wrapped:
            return

        with self._lock:
            if self._wrapped:  # double-check after lock
                return

            from triton.runtime.jit import JITFunction
            orig = JITFunction.run

            # Fail hard if already wrapped by another instance (shouldn't happen with singleton)
            if getattr(orig, '__telemetry_wrapped__', False):
                raise RuntimeError(
                    "Telemetry already installed by another instance. "
                    "Multiple counter instances violate singleton pattern."
                )

            self._orig_run = orig
            counter = self  # closure capture

            def _run_with_count(self, *args, **kwargs):
                # No lock needed: we only check count==0, lost increments don't matter
                counter.count += 1
                return counter._orig_run(self, *args, **kwargs)

            # Mark as wrapped to prevent double-patching
            _run_with_count.__telemetry_wrapped__ = True
            JITFunction.run = _run_with_count
            self._wrapped = True

    def reset(self):
        """Reset counter to zero. No lock needed (single write is atomic)."""
        self.count = 0

    def assert_launched(self, context="operation"):
        """
        Raise RuntimeError if no Triton kernel launches detected.

        Args:
            context: Description of the operation being checked (e.g., "backward")

        Raises:
            RuntimeError: If telemetry not installed or no launches detected
        """
        # Fail hard if install() was never called or failed
        if not self._wrapped:
            raise RuntimeError(
                "Triton launch telemetry not installed; cannot verify kernel launches. "
                "This indicates a setup issue."
            )

        # No lock needed: single read, and we only care about 0 vs non-zero
        if self.count == 0:
            raise RuntimeError(
                f"No Triton kernel launches detected during {context}. "
                "PyTorch/stub bypass detected - backward must call a @triton.jit kernel."
            )


# Module-level singleton
_triton_launch_counter = _TritonLaunchCounter()


def compile_kernel(file_path: str, generated_fp: str | None = None):
    """
    - Executes user’s forward module to get a fresh namespace for make_args, SWEEP, setup, etc.
    - Runs setup() once to force Triton to compile the forward kernel (and surface syntax/runtime issues early).
    - Validates that the forward file actually defines a @triton.jit kernel plus an autodiff-tagged stub, and raises UserError if not.
    - If user fwd kernel call blows up, the failure stays in that short-lived process because we invoke this fn from probe child.
    """

    # Spawn a short-lived child before touching the freshly edited backward. The child runs compile_kernel to
    # materialize the forward namespace (make_args, SWEEP, setup) and to surface syntax/device faults inside
    # its own CUDA context; the parent stays clean. On success nothing is returned—only a "passed" marker
    # so the parent can rebuild in-process. A KERNEL_AGENT_PROBE_CHILD flag prevents recursion because
    # run_compile_child ultimately calls back into compile_kernel.

    if not os.environ.get("KERNEL_AGENT_PROBE_CHILD"):
        try:
            # Import here to avoid circular import when worker imports utils.
            from .worker import run_compile_child
            run_compile_child(file_path, generated_fp=generated_fp)
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
        # Run user code inside a real module so functions get a stable __module__.
        # (Historically a plain dict caused __module__ = None and downstream crashes.)
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
        # this can fail, e.g. wrong indentation of python kernel function declaration
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

    # Validate presence of a Triton kernel and an @autodiff-decorated stub
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
            "@autodiff(...)\n"
            "def stub(...): ...\n"
        )

    setup_fn = ns.get("setup")
    if not callable(setup_fn):
        raise UserError("Expected a top-level setup() that runs the stub once. The stub must call the kernel.")

    if not callable(ns.get("make_args")) or not isinstance(ns.get("SWEEP"), (list, tuple)):
        raise UserError("User kernel must define make_args and SWEEP")

    def _exec_setup():
        exec(setup_fn.__code__, ns, ns)

    # NOTE: execute user stub and kernels
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
    if not callable(ns.get("stub")):
        raise UserError("Expected a top-level stub(...) to call the kernel.")

    return ns



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
        if len(lines) > max_lines:
            snippet = "".join(lines[:max_lines])
            snippet += f"\n\n# ... [TRUNCATED: {len(lines) - max_lines} lines omitted] ...\n"
            return snippet
        return "".join(lines[:max_lines])
    except Exception as e:
        return f"(snippet unavailable: {e})"


def strip_backward_section(content: str) -> str:
    """Remove backward section from file content (for Phase 1 filtering).

    During Phase 1 (PyTorch reference generation), the backward section contains
    only a stub skeleton with "raise NotImplementedError", which should be hidden
    from the LLM to avoid confusion.

    File structure: [forward kernel] [forward stub] [backward kernel] [backward stub] [autograd.Function] <EOF>
    Once the backward marker is found, everything from that point to EOF is removed.
    """
    # Look for the standard backward section marker
    marker = "# Backward kernel and stub"
    idx = content.find(marker)
    if idx != -1:
        return content[:idx]
    return content


FN_NAMES_TO_STRIP = {"torch_fn", "make_args", "setup", "flops", "pytorch_reference_impl"}


# todo: instead of removing unwanted code, maybe change instead
# to select only the desired code (kernel, stub) -- seems cleaner

def redact_torch_fn(path: str, max_lines: int | None = None) -> str:
    """Redact prompt-only helpers from the forward source for LLM.

    Removes these top-level items (module scope only):
      - def torch_fn(...): (with decorators)
      - def pytorch_reference_impl(...): (validation reference - prevents LLM from emitting PyTorch code)
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
        lines = src.splitlines(True)
        if len(lines) > max_lines:
            src = "".join(lines[:max_lines])
            src += f"\n\n# ... [TRUNCATED: {len(lines) - max_lines} lines omitted] ...\n"
        else:
            src = "".join(lines[:max_lines])

    # Normalize trailing newlines at EOF:
    # - Remove only newline characters to avoid extra blank lines
    # - If non-empty, one final newline
    # This keeps internal spacing intact while removing only the suffix clutter.
    src = src.rstrip("\n\r")
    if src:
        src = src + "\n"
    return src



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
