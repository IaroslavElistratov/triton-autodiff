# utils.py -- helpers for TTIR -> Triton-language raising
from __future__ import annotations
from typing import Dict, Optional
import keyword
import re

from triton._C.libtriton import ir as mlir

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def _sanitize_identifier(name: Optional[str]) -> str:
    """Turn an arbitrary source/loc name into a safe Python identifier."""
    if not name:
        return ""
    name = name.strip()
    # Replace non-identifier chars with '_'
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    # Avoid leading digits
    if not name or name[0].isdigit():
        name = f"v_{name}"
    # Avoid keywords
    if keyword.iskeyword(name):
        name = f"{name}_"
    # Final guard
    if not _IDENT_RE.match(name):
        name = ""
    return name

class ValueNamer:
    """Binds MLIR Value ids to stable, human-friendly names.

    Prefers the outermost NameLoc (via ir.value_best_name), sanitizes, and
    uniquifies. Falls back to v1, v2, ... when no name is available.
    """
    def __init__(self):
        self.vid2name: Dict[int, str] = {}
        self.used: set[str] = set()
        self._counter: int = 0

    def _uniq(self, base: Optional[str]) -> str:
        s = _sanitize_identifier(base) if base else ""
        if not s:
            # fallback monotonic v{n}
            self._counter += 1
            s = f"v{self._counter}"
        name = s
        i = 0
        while name in self.used:
            i += 1
            name = f"{s}_{i}"
        self.used.add(name)
        return name

    @staticmethod
    def _vid(v: mlir.value) -> int:
        return int(v.id())

    def bind(self, v: mlir.value) -> str:
        vid = self._vid(v)
        if vid in self.vid2name:
            return self.vid2name[vid]
        src = mlir.value_best_name(v)
        src = str(src) if src is not None else None
        nm = self._uniq(src)
        self.vid2name[vid] = nm
        return nm

    def name(self, v: mlir.value) -> str:
        vid = self._vid(v)
        return self.vid2name.get(vid) or self.bind(v)

    # Build upfront for determinism
    def prebind_module(self, mod: mlir.module) -> None:
        # Function args first
        entry = mod.get_entry_func_name() or ""
        if entry and mod.has_function(entry):
            fn = mod.get_function(entry)
            for i in range(fn.get_num_args()):
                self.bind(fn.args(i))
        # Then all op results in walk order
        def on_op(op):
            for i in range(op.get_num_results()):
                self.bind(op.get_result(i))
        mod.walk(on_op)

def build_value_name_map(mod: mlir.module) -> Dict[int, str]:
    namer = ValueNamer()
    namer.prebind_module(mod)
    return dict(namer.vid2name)
