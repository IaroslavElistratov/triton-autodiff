# utils.py
# Naming helpers: use C++ binding ir.value_best_name(v) and ensure uniqueness.

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


class _NamePool:
    def __init__(self) -> None:
        self.used: Set[str] = set()

    def claim(self, base: Optional[str]) -> str:
        if not base:
            base = "v"
        base = _sanitize_identifier(base)
        if base not in self.used:
            self.used.add(base)
            return base
        i = 1
        while True:
            cand = f"{base}_{i}"
            if cand not in self.used:
                self.used.add(cand)
                return cand
            i += 1


def build_value_name_hints(module: mlir.module) -> Dict[int, str]:
    """
    Walk the module and derive a stable, readable name for every Value
    that has an associated (best) NameLoc, using ir.value_best_name(v).
    Returns a mapping: value_id(int) -> unique variable name (str).
    """
    pool = _NamePool()
    vid2name: Dict[int, str] = {}

    def on_op(op: mlir.operation):
        # Results
        for i in range(op.get_num_results()):
            v = op.get_result(i)
            s = mlir.value_best_name(v)   # -> py.str or None (C++ binding)
            nm = pool.claim(str(s) if s is not None else None)
            vid2name[int(v.id())] = nm

    # Also name function arguments if present
    def on_func(func: mlir.function):
        for i in range(func.get_num_args()):
            v = func.args(i)
            s = mlir.value_best_name(v)
            nm = pool.claim(str(s) if s is not None else f"arg{i}")
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
