# utils.py
# Naming helpers for the raiser: derive clean stems from MLIR NameLocs.
#
# Design:
# - Do NOT uniquify here. Codegen stamps meaningful NameLocs (assignment/op-centric
#   stems). We sanitize into Python-safe identifiers, but final uniqueness/reuse
#   is deferred to the raiser where liveness/context is available. This keeps a
#   single authority for emitted Python identifiers and avoids unstable suffixes
#   when TTIR transforms/cloning occur.

from __future__ import annotations
from typing import Dict, Optional, Set
import keyword
import re

from triton._C.libtriton import ir as mlir


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _sanitize_identifier(name: str) -> str:
    """Turn an arbitrary source/loc name into a safe Python identifier."""
    name = name.strip()
    # Common MLIR-ish or Triton-ish artifacts to drop
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    # Avoid leading digits
    if not name or name[0].isdigit():
        name = f"v_{name}"
    # Avoid keywords / builtins
    if keyword.iskeyword(name):
        name = f"{name}_"
    # Final guard
    if not _IDENT_RE.match(name):
        name = "v"
    return name


def _normalize_stem(name: Optional[str]) -> str:
    """Pass-through stem normalization for NameLoc-derived names.

    Minimal by design: we do not strip counters or rewrite legacy patterns here.
    The raiser has full liveness/context to decide uniqueness or reuse safely.
    """
    if not name:
        return "v"
    return str(name)


def build_value_name_hints(module: mlir.module) -> Dict[int, str]:
    """
    Walk the module and derive a stable, readable STEM for every Value that has
    an associated (best) NameLoc, using ir.value_best_name(v).

    Returns: Dict[value_id -> sanitized_stem]
    - Not guaranteed unique. The raiser is the single authority for final
      uniqueness/reuse (it has liveness and grouping info).
    """
    vid2name: Dict[int, str] = {}
    # Prefer semantic/attr-based stems when available (set earlier in the flow),
    # but avoid importing helpers that may not exist here. The raiser will still
    # enforce uniqueness/reuse.
    composer = None

    def on_op(op: mlir.operation):
        # Results
        for i in range(op.get_num_results()):
            v = op.get_result(i)
            pref = None
            # No external composer available here; rely on MLIR best name or
            # any stems injected upstream (e.g., via attributes).
            pref = None
            if isinstance(pref, str) and pref:
                stem = pref
            else:
                s = mlir.value_best_name(v)   # -> py.str or None (C++ binding)
                stem = _normalize_stem(str(s) if s is not None else None)
            nm = _sanitize_identifier(stem)
            vid2name[int(v.id())] = nm

    # Also name function arguments if present
    def on_func(func: mlir.function):
        for i in range(func.get_num_args()):
            v = func.args(i)
            s = mlir.value_best_name(v)
            stem = _normalize_stem(str(s) if s is not None else f"arg{i}")
            nm = _sanitize_identifier(stem)
            vid2name[int(v.id())] = nm

    module.walk(on_op)
    # Best-effort to find the entry and bind args (harmless if none)
    try:
        fname = module.get_entry_func_name()
        if fname:
            on_func(module.get_function(fname))
    except Exception:
        pass

    return vid2name
