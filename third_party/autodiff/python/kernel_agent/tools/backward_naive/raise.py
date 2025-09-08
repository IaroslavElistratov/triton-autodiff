# raise.py
# Minimal, robust TTIR -> Triton-language raiser using the new operation bindings.
# Uses: operation.mnemonic, operation.str_nodebug(), operation.get_attr_text(),
#       operation.get_int_attr(), operation.get_i64_array_attr(),
#       operation.get_reduce_combiner() to avoid brittle regex fallbacks.

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import re
import struct
import math
import sys

import triton
import triton.language as tl
from triton._C.libtriton import ir as mlir

from utils import build_value_name_hints


# ----------------------------- Options ---------------------------------------

@dataclass
class RaiserOptions:
    # If True, print arith with symbols (a+b) instead of tl.add(a,b)
    infix_arith: bool = True
    # Emit headers grouping gradients by their source forward op
    emit_grad_groups: bool = True


# ----------------------------- Type helpers ----------------------------------

def _dtype_expr_from_type_string(t: str) -> Optional[str]:
    if "bf16" in t: return "tl.bfloat16"
    if "f16"  in t: return "tl.float16"
    if "f32"  in t: return "tl.float32"
    if "f64"  in t: return "tl.float64"
    if "i1"   in t: return "tl.int1"
    m = re.search(r"(?:^|[x<,])i(8|16|32|64)(?:[>x,]|$)", t)
    if m:
        return {"8":"tl.int8","16":"tl.int16","32":"tl.int32","64":"tl.int64"}[m.group(1)]
    return None

def _shape_from_tensor_type_string(t: str) -> Optional[List[int]]:
    # "tensor<128x64xf16>" -> [128, 64]
    m = re.search(r"tensor<([^>]+)>", t)
    if not m:
        return None
    parts = m.group(1).split("x")
    try:
        return [int(d) for d in parts[:-1]]
    except ValueError:
        return None  # dynamic dims -> give up


# Helper: format a Python tuple literal for a static block shape, e.g. [16, 16] -> "(16, 16)"
def _fmt_shape(shp): return "(" + ", ".join(str(d) for d in shp) + ("," if len(shp) == 1 else "") + ")"
# Helper: best-effort pointer type detection from MLIR type text
# We must not broadcast pointers (doing so loses pointer-ness and breaks tl.load/tl.store)
def _is_ptr_type(ty) -> bool: return "ptr<" in str(ty)


def _typed_zero(dst_ty: mlir.type) -> str:
    """Emit a neutral literal or zeros tensor matching the type."""
    t = str(dst_ty)
    shp = _shape_from_tensor_type_string(t)
    dty = _dtype_expr_from_type_string(t) or "None"
    if shp:
        tup = "(" + ", ".join(str(d) for d in shp) + ("," if len(shp) == 1 else "") + ")"
        return f"tl.zeros({tup}, dtype={dty})"
    if dty.startswith("tl.float") or dty == "tl.bfloat16":
        return "0.0"
    if dty == "tl.int1":
        return "False"
    return "0"


# --- small kwarg builder (avoids None/empty values) ---
def _kw_items(*pairs):
    """Return ['k=v', ...] only for non-empty values. Values should be preformatted strings."""
    return [f"{k}={v}" for k, v in pairs if v not in (None, "", [])]


# ----------------------------- Attr helpers ----------------------------------

def _text_attr(op, name: str) -> Optional[str]:
    """Return textual attr value with quotes stripped, or None."""
    try:
        txt = op.get_attr_text(name)
        if txt is None:
            return None
        s = str(txt)
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            return s[1:-1]
        return s
    except Exception:
        return None


class Attr:
    @staticmethod
    def int_attr(op, name, default=None):
        try:
            v = op.get_int_attr(name)
            if v is not None:
                return int(v)
        except Exception:
            pass
        m = re.search(rf"\b{name}\s*=\s*(-?\d+)\b", op.str_nodebug())
        return int(m.group(1)) if m else default

    @staticmethod
    def bool_attr(op, name, default=False):
        # Prefer textual attr; falls back to generic print
        s = _text_attr(op, name)
        if s is not None:
            s = s.strip().lower()
            if s in ("true", "false"):
                return s == "true"
        m = re.search(rf"\b{name}\s*=\s*(true|false)\b", op.str_nodebug())
        return {"true": True, "false": False}.get(m.group(1), default) if m else default

    @staticmethod
    def list_int_attr(op, name):
        try:
            arr = op.get_i64_array_attr(name)
            if arr is not None:
                return [int(x) for x in arr]
        except Exception:
            pass
        m = re.search(rf"\b{name}\s*=\s*\[([0-9,\s-]+)\]", op.str_nodebug())
        return [int(x) for x in m.group(1).replace(" ", "").split(",")] if m else None

    # Common shorthands
    axis        = staticmethod(lambda op, default=0: Attr.int_attr(op, "axis", default))
    order       = staticmethod(lambda op: Attr.list_int_attr(op, "order"))
    boundary_ck = staticmethod(lambda op: Attr.list_int_attr(op, "boundary_check"))
    start_end   = staticmethod(lambda op: (
        (Attr.int_attr(op, "start"), Attr.int_attr(op, "end"))
        if (Attr.int_attr(op, "start") is not None and Attr.int_attr(op, "end") is not None)
        else (tuple(map(int, re.search(r"\bstart\s*=\s*(-?\d+).*?end\s*=\s*(-?\d+)", op.str_nodebug()).groups()))
              if re.search(r"\bstart\s*=\s*(-?\d+).*?end\s*=\s*(-?\d+)", op.str_nodebug()) else None)
    ))

    @staticmethod
    def cache_modifier(op):
        s = (_text_attr(op, "cache_modifier") or op.str_nodebug()).lower()
        if ".ca" in s or " cache_modifier = ca" in s or "cache_modifier=ca" in s: return ".ca"
        if ".cg" in s or " cache_modifier = cg" in s or "cache_modifier=cg" in s: return ".cg"
        if ".cs" in s or " cache_modifier = cs" in s or "cache_modifier=cs" in s: return ".cs"
        if ".wb" in s or " cache_modifier = wb" in s or "cache_modifier=wb" in s: return ".wb"
        if ".wt" in s or " cache_modifier = wt" in s or "cache_modifier=wt" in s: return ".wt"
        if ".cv" in s or " cache_modifier = cv" in s or "cache_modifier=cv" in s: return ".cv"
        return ""

    @staticmethod
    def eviction_policy(op):
        s = (_text_attr(op, "eviction_policy") or op.str_nodebug()).lower()
        for k in ("evict_last", "evict_first"):
            if k in s: return k
        return ""

    @staticmethod
    def padding_option(op):
        s = (_text_attr(op, "padding_option") or op.str_nodebug()).lower()
        if "nan" in s:  return "nan"
        if "zero" in s: return "zero"
        return ""

    # @staticmethod
    # def reduce_kinds(op, module):
    #     # Prefer the direct helper.
    #     try:
    #         comb = op.get_reduce_combiner()
    #     except Exception:
    #         comb = None
    #     if comb:
    #         return [str(comb)] * max(1, op.get_num_results())

    #     # Fallback: walk the reduce region to infer the combiner.
    #     reg = op.get_region(0); target_rid = reg.id()
    #     val_owner = {}
    #     reduce_ret = None

    #     def visit(inner):
    #         b = inner.get_block(); r = b.get_parent() if b is not None else None
    #         if b is None:
    #             return
    #         while r:
    #             if r.id() == target_rid:
    #                 for i in range(inner.get_num_results()):
    #                     v = inner.get_result(i)
    #                     val_owner[int(v.id())] = inner.get_name()
    #                 if inner.get_name().endswith("reduce.return"):
    #                     nonlocal reduce_ret; reduce_ret = inner
    #                 break
    #             r = r.get_parent_region()
    #     module.walk(visit)

    #     kinds = []
    #     if reduce_ret:
    #         MAP = {
    #             "arith.addi":"sum","arith.addf":"sum",
    #             "arith.maxsi":"max","arith.maxui":"max","arith.maximumf":"max","arith.maxnumf":"max",
    #             "arith.minsi":"min","arith.minui":"min","arith.minimumf":"min","arith.minnumf":"min",
    #             "arith.andi":"and","arith.ori":"or","arith.xori":"xor",
    #         }
    #         for i in range(reduce_ret.get_num_operands()):
    #             vid = int(reduce_ret.get_operand(i).id())
    #             kinds.append(MAP.get(val_owner.get(vid, ""), "custom"))
    #     return kinds


# ----------------------------- Raiser ----------------------------------------

class Raiser:
    def __init__(self, module: mlir.module, func_name: Optional[str], *, opts: Optional[RaiserOptions] = None):
        self.m = module
        self.func_name = func_name or self.m.get_entry_func_name() or "raised_kernel"
        self.opts = opts or RaiserOptions()
        self.lines: List[str] = []
        self._n = 0
        self._hints: Dict[int, str] = build_value_name_hints(self.m)
        self.env: Dict[int, str] = {}
        self.registry = self._build_registry()
        # Alias map for elided value-only ops: alias name -> surviving name
        # Why: After we drop explicit value broadcasts/splats, some forward temps
        # (e.g., fwd_qk_3) become pure aliases and are not emitted at all. Downstream
        # code and headers could still refer to these names. We record the mapping
        # when we elide the alias so later we can resolve labels (and optionally
        # normalize bwd_* names) to the surviving producer (e.g., fwd_qk).
        self._alias_of: Dict[str, str] = {}
        # Track current gradient group label to reduce noisy headers
        self._last_grad_of: Optional[str] = None
        # Track finer-grained per-handler label to show local groups
        self._last_local_grad_of: Optional[str] = None
        # Python argument names by kernel-arg index for grouping via raise.gradIdx
        # Inline note: previously headers came from raise.gradOf text and could show
        # stride_* due to provenance landing on stride args. We now prefer
        # raise.gradIdx -> python arg name, then raise.gradOf, then legacy raise.gradOf.
        self._arg_names: List[str] = []
        # Map from a stable forward-op tag id (raise.gradOfTag) -> chosen Python name.
        # High-level: tags identify cloned forward ops; I resolve tags to the
        # actual minted Python names for those forward values so local headers can
        # show readable identifiers without re-deriving names here.
        self._fwd_tag_to_py: Dict[int, str] = self._build_tag_to_py_map()

    # ---- small utils
    def _fresh(self, base="v") -> str:
        self._n += 1
        return f"{base}{self._n}"

    @staticmethod
    def _vid(v: mlir.value) -> int:
        return int(v.id())

    def _bind(self, v: mlir.value, name: Optional[str] = None) -> str:
        vid = self._vid(v)
        if vid in self.env:
            return self.env[vid]
        if name is None:
            name = self._hints.get(vid, self._fresh("v"))
        # Normalize names that reference elided forward aliases:
        # If a bwd name mirrors a fwd alias we removed (e.g., bwd_acc_2 and we elided fwd_acc_2 → fwd_acc),
        # adopt the surviving base (bwd_acc) when it doesn't collide.
        name = self._canonicalize_name(name)
        self.env[vid] = name
        return name

    def _canonicalize_name(self, name: str) -> str:
        # Name canonicalization for bwd_*:
        # If a backward temp mirrors a forward alias we've removed (e.g., bwd_acc_2
        # while fwd_acc_2 was elided to fwd_acc), adopt the surviving base (bwd_acc)
        # when it doesn't collide. This keeps bwd_* names aligned with visible fwd_*.
        if name.startswith("bwd_"):
            tail = name[4:]
            key = f"fwd_{tail}"
            mapped = self._alias_of.get(key)
            if isinstance(mapped, str):
                # mapped may be either fwd_* or a bare base; derive bwd_* accordingly
                base = mapped[4:] if mapped.startswith("fwd_") else mapped
                candidate = f"bwd_{base}"
                # Avoid collisions: only adopt if not already used
                if candidate not in self.env.values():
                    return candidate
        return name

    def _get(self, v: mlir.value, hint: str = "v") -> str:
        vid = self._vid(v)
        if vid not in self.env:
            self._bind(v, self._hints.get(vid, self._fresh(hint)))
        return self.env[vid]

    # ---- emission helpers
    def _arith(self, a: str, b: str, sym: str, fn: str) -> str:
        return f"{a} {sym} {b}" if self.opts.infix_arith else f"tl.{fn}({a}, {b})"

    def _cast(self, x: str, dst_ty: mlir.type) -> str:
        dty = _dtype_expr_from_type_string(str(dst_ty)) or "None"
        return f"tl.cast({x}, {dty})"

    def _bitcast(self, x: str, dst_ty: mlir.type) -> str:
        dty = _dtype_expr_from_type_string(str(dst_ty)) or "None"
        return f"tl.bitcast({x}, {dty})"

    def _attr_text(self, op, name: str) -> Optional[str]:
        # Prefer typed string attribute when available to avoid parsing quotes
        s = op.get_str_attr(name)
        if s is not None:
            return str(s)

        # # Fallback: textual attr representation
        # try:
        #     v = op.get_attr_text(name)
        #     if v is None:
        #         return None
        #     s = str(v).strip()
        #     if len(s) >= 2 and s[0] == s[-1] and s[0] in ('\"', "'"):
        #         s = s[1:-1]
        #     return s or None
        # except Exception:
        #     return None

    def _int_attr(self, op, name: str) -> Optional[int]:
        # Prefer typed integer attribute
        v = op.get_int_attr(name)
        if v is not None:
            return int(v)
        return None

    def _group_label(self, op) -> Optional[str]:
        # Grouping strategy with union support:
        # 1) If raise.gradIdxs (array) is present:
        #    - singleton -> "<arg_name>"
        #    - multi     -> "shared:{name1,name2}"
        try:
            arr = op.get_i64_array_attr("raise.gradIdxs")
        except Exception:
            arr = None
        if arr:
            uniq = sorted(set(int(x) for x in arr if isinstance(x, int)))
            if len(uniq) == 1:
                i = uniq[0]
                return self._arg_names[i] if 0 <= i < len(self._arg_names) else f"arg{i}"
            names = [self._arg_names[i] if 0 <= i < len(self._arg_names) else f"arg{i}" for i in uniq]
            return "shared:{" + ",".join(names) + "}"
        # 2) Else use canonical index (raise.gradIdx)
        idx = self._int_attr(op, "raise.gradIdx")
        if idx is not None and 0 <= idx < len(self._arg_names):
            return self._arg_names[idx]
        # 3) No explicit human label fallback (legacy path removed)
        return None

    def _maybe_emit_grad_header(self, op) -> None:
        if not self.opts.emit_grad_groups:
            return
        # Prefer kernel arg mapping; fallback removed (only idx-based grouping)
        src = self._group_label(op)
        if not src:
            return
        if src != self._last_grad_of:
            self._last_grad_of = src
            # Visual spacer between groups
            if self.lines and not self.lines[-1].strip() == "":
                self.lines.append("")
            if src.startswith("shared:{"):
                inner = src[len("shared:"):]  # keep the {...}
                self.lines.append(f"    # ~~~~~~~~~~ grad branch for {inner} ~~~~~~~~~~")
            else:
                self.lines.append(f"    # ~~~~~~~~~~ grad branch for {src} ~~~~~~~~~~")

    def _maybe_emit_local_gradof(self, op) -> None:
        if not self.opts.emit_grad_groups:
            return
        # Emit finer-grained header based on raise.gradOfTag (stable id)
        py_lbl = self._resolve_local_label(op, None)
        if not py_lbl:
            return
        if py_lbl != self._last_local_grad_of:
            self._last_local_grad_of = py_lbl
            if self.lines and not self.lines[-1].strip() == "":
                self.lines.append("")
            self.lines.append(f"    # local grads for {py_lbl}")

    def _build_tag_to_py_map(self) -> Dict[int, str]:
        tag2name: Dict[int, str] = {}
        ops: List[mlir.operation] = []
        self.m.walk(lambda o: ops.append(o))
        for o in ops:
            tag = self._int_attr(o, "raise.gradOfTag")
            if tag is None:
                continue
            # Only consider cloned forward ops
            if not Attr.bool_attr(o, "isCloned", False):
                continue
            # Deterministic selection: prefer first "fwd_" result name, else first available
            first_fwd = None
            first_any = None
            for i in range(o.get_num_results()):
                nm = self._hints.get(self._vid(o.get_result(i)))
                if isinstance(nm, str):
                    if first_any is None:
                        first_any = nm
                    if first_fwd is None and nm.startswith("fwd_"):
                        first_fwd = nm
            chosen = first_fwd if first_fwd is not None else first_any
            if chosen is None:
                continue
            tag2name[int(tag)] = chosen
        return tag2name

    def _resolve_local_label(self, op, fallback: str) -> str:
        tag = self._int_attr(op, "raise.gradOfTag")
        if tag is None:
            return fallback
        name = self._fwd_tag_to_py.get(int(tag), fallback)
        if not isinstance(name, str):
            return fallback
        # Follow alias chain to a surviving identifier if this tag pointed to an elided alias
        seen = set()
        while name in self._alias_of and name not in seen:
            seen.add(name)
            name = self._alias_of[name]
        return name

    # ---- registry
    def _build_registry(self) -> Dict[str, Callable[[mlir.operation], Optional[str]]]:
        R: Dict[str, Callable[[mlir.operation], Optional[str]]] = {}

        # --- arith.constant
        def emit_constant(op: mlir.operation) -> str:
            ty = op.get_result(0).get_type() if op.get_num_results() else None
            ty_str = str(ty) if ty else ""
            shp = _shape_from_tensor_type_string(ty_str) or []
            dty = _dtype_expr_from_type_string(ty_str) or "tl.float32"

            v = None
            try:
                v = op.get_splat_value("value")
            except Exception:
                v = None

            if v is None:
                return _typed_zero(ty) if ty else "0"

            def _lit(x):
                if isinstance(x, float):
                    if math.isnan(x):
                        return "float('nan')"
                    if math.isinf(x):
                        return "float('inf')" if x > 0 else "float('-inf')"
                    return repr(x)
                if isinstance(x, bool):
                    return "True" if x else "False"
                return str(int(x))

            return (f"tl.full({tuple(shp)}, {_lit(v)}, dtype={dty})" if shp else _lit(v))
        R["arith.constant"] = emit_constant

        # --- binary arithmetic
        def bin2(op, sym, fn):
            return self._arith(self._get(op.get_operand(0)),
                               self._get(op.get_operand(1)), sym, fn)
        for k, sym, fn in (
            ("arith.addf", "+", "add"), ("arith.addi", "+", "add"),
            ("arith.subf", "-", "sub"), ("arith.subi", "-", "sub"),
            ("arith.mulf", "*", "mul"), ("arith.muli", "*", "mul"),
            ("arith.divf", "/", "fdiv"),
        ):
            R[k] = (lambda op, s=sym, f=fn: bin2(op, s, f))

        # --- integer division & remainder
        # MLIR divsi/remsi are trunc‑toward‑zero; Python `//`/`%` are floor‑based for negatives
        R["arith.divsi"] = lambda op: f"{self._get(op.get_operand(0))} // {self._get(op.get_operand(1))}  # assumes non-negative"
        R["arith.divui"] = lambda op: self._arith(self._get(op.get_operand(0)), self._get(op.get_operand(1)), "//", "floordiv")
        R["arith.remsi"] = lambda op: f"{self._get(op.get_operand(0))} % {self._get(op.get_operand(1))}  # assumes non-negative"
        R["arith.remui"] = R["arith.remsi"]

        # --- casts
        R["arith.extf"]    = lambda op: self._cast(self._get(op.get_operand(0)), op.get_result(0).get_type())
        R["arith.truncf"]  = lambda op: self._cast(self._get(op.get_operand(0)), op.get_result(0).get_type())
        R["arith.fptosi"]  = lambda op: self._cast(self._get(op.get_operand(0)), op.get_result(0).get_type())
        R["arith.fptoui"]  = lambda op: self._cast(self._get(op.get_operand(0)), op.get_result(0).get_type())
        R["arith.sitofp"]  = lambda op: self._cast(self._get(op.get_operand(0)), op.get_result(0).get_type())
        R["arith.uitofp"]  = lambda op: self._cast(self._get(op.get_operand(0)), op.get_result(0).get_type())
        R["arith.bitcast"] = lambda op: self._bitcast(self._get(op.get_operand(0)), op.get_result(0).get_type())
        # integer width casts
        for _k in ("arith.extsi", "arith.extui", "arith.trunci"):
            R[_k] = (lambda op: self._cast(self._get(op.get_operand(0)), op.get_result(0).get_type()))

        # --- select (ternary)
        R["arith.select"] = lambda op: f"tl.where({self._get(op.get_operand(0))}, {self._get(op.get_operand(1))}, {self._get(op.get_operand(2))})"

        # --- compares with robust predicate extraction
        # Decode cmp predicates robustly to avoid silently generating wrong masks.
        # 1) Prefer the symbolic 'predicate' attribute (e.g., #arith.cmpipred<slt>),
        #    returning the token inside <>.
        # 2) If the attribute is numeric-coded (e.g., "2 : i64") or otherwise non-symbolic,
        #    ignore it and parse the textual op ("arith.cmpi slt, %a, %b : ...").
        # 3) As a last resort for integer cmps, map numeric codes to tokens (2 -> slt, ...).
        # Never default to '=='. The previous behavior mis-parsed '<' as '=='
        # and produced always-false masks like 'offsets == N', which zeroed grads.
        def _extract_cmp_pred(op: mlir.operation) -> Optional[str]:
            # 1) Prefer explicit 'predicate' attribute
            attr_txt = None
            try:
                attr_txt = op.get_attr_text("predicate")
                if attr_txt:
                    s = str(attr_txt).strip().lower()
                    # Handles enum prints like "#arith.cmpipred<slt>"
                    m = re.search(r"<\s*([a-z]+)\s*>", s)
                    if m:
                        return m.group(1)
                    # If it's already a bare predicate token like "slt"
                    if re.fullmatch(r"[a-z]+", s):
                        return s
                    # If numeric-coded like "2 : i64" or "2", don't trust it; fall back to textual op parse
            except Exception:
                attr_txt = None
            # 2) Fallback: parse generic print, e.g.: "arith.cmpi slt, %a, %b : ..."
            txt = op.str_nodebug().lower()
            m = re.search(r"cmp[fi]\s+([a-z]+)\s*,", txt)
            if m:
                return m.group(1)
            # 3) Last resort: decode numeric-coded predicate for integer cmps
            try:
                s = str(attr_txt).strip().lower() if attr_txt else ""
                mnum = re.fullmatch(r"\s*(-?\d+)\s*(?::\s*i\d+)?\s*", s)
                if mnum:
                    code = int(mnum.group(1))
                    name = op.mnemonic if hasattr(op, "mnemonic") else op.get_name()
                    if name.endswith("arith.cmpi") or name.endswith("cmpi") or "arith.cmpi" in name:
                        int_map = {
                            0: "eq", 1: "ne", 2: "slt", 3: "sle",
                            4: "sgt", 5: "sge", 6: "ult", 7: "ule",
                            8: "ugt", 9: "uge",
                        }
                        return int_map.get(code)
            except Exception:
                pass
            return None

        def _emit_cmp_with_pred(op: mlir.operation) -> str:
            a = self._get(op.get_operand(0))
            b = self._get(op.get_operand(1))
            pred = _extract_cmp_pred(op)
            table = {
                "eq":"==","oeq":"==","ueq":"==",
                "ne":"!=","one":"!=","une":"!=",
                "slt":"<","ult":"<","olt":"<",
                "sle":"<=","ule":"<=","ole":"<=",
                "sgt":">","ugt":">","ogt":">",
                "sge":">=","uge":">=","oge":">=",
            }
            # Fail loudly on unknown preds to avoid silently emitting incorrect equality.
            if pred not in table:
                raise RuntimeError(f"Unsupported/unknown cmp predicate: {pred} in {op.str_nodebug()}")
            return f"{a} {table[pred]} {b}"

        R["arith.cmpi"] = _emit_cmp_with_pred
        R["arith.cmpf"] = _emit_cmp_with_pred

        # --- math unary
        R["math.cos"]   = lambda op: f"tl.cos({self._get(op.get_operand(0))})"
        R["math.sin"]   = lambda op: f"tl.sin({self._get(op.get_operand(0))})"
        R["math.exp"]   = lambda op: f"tl.exp({self._get(op.get_operand(0))})"
        R["math.exp2"]  = lambda op: f"tl.exp2({self._get(op.get_operand(0))})"
        R["math.log"]   = lambda op: f"tl.log({self._get(op.get_operand(0))})"
        R["math.log2"]  = lambda op: f"tl.log2({self._get(op.get_operand(0))})"
        R["math.sqrt"]  = lambda op: f"tl.sqrt({self._get(op.get_operand(0))})"
        R["math.rsqrt"] = lambda op: f"tl.rsqrt({self._get(op.get_operand(0))})"
        R["math.absf"]  = lambda op: f"tl.abs({self._get(op.get_operand(0))})"
        # elementwise min/max with *num semantics* (from Answer 2)
        R["arith.maxnumf"] = lambda op: f"tl.maximum({self._get(op.get_operand(0))}, {self._get(op.get_operand(1))})"
        R["arith.minnumf"] = lambda op: f"tl.minimum({self._get(op.get_operand(0))}, {self._get(op.get_operand(1))})"

        # --- Triton builtins (axis attr)
        R["tt.get_program_id"]   = lambda op: f"tl.program_id(axis={max(0, min(2, Attr.axis(op, 0)))})"
        R["tt.get_num_programs"] = lambda op: f"tl.num_programs(axis={max(0, min(2, Attr.axis(op, 0)))})"

        # --- memory: load/store with cache/evict/padding/boundary_check
        # Keep masked-load semantics safe by injecting a typed zero 'other='
        # when a mask is provided without an explicit 'other'. This prevents
        # garbage reads under false masks and mirrors TTIR semantics.
        def emit_load(op: mlir.operation) -> str:
            ptr  = self._get(op.get_operand(0))
            mask = self._get(op.get_operand(1)) if op.get_num_operands() >= 2 else None
            if op.get_num_operands() >= 3:
                other = self._get(op.get_operand(2))
            else:
                # masked load with no explicit 'other' => inject typed zero (TTIR semantics)
                other = _typed_zero(op.get_result(0).get_type()) if mask is not None else None

            # optional attrs
            bc  = Attr.boundary_ck(op)
            pad = Attr.padding_option(op)
            cm  = Attr.cache_modifier(op)
            ev  = Attr.eviction_policy(op)

            kws = _kw_items(("mask", mask), ("other", other))
            if bc:  kws.append(f"boundary_check={tuple(bc)}")
            if pad: kws.append(f"padding_option='{pad}'")
            if cm:  kws.append(f"cache_modifier='{cm}'")
            if ev:  kws.append(f"eviction_policy='{ev}'")
            return f"tl.load({ptr}{', ' if kws else ''}{', '.join(kws)})"
        R["tt.load"] = emit_load

        # tt.store: shape-match value to pointer; keep pointer untouched
        # Why: Broadcasting/reshaping the pointer erases pointer-ness and breaks tl.store.
        def emit_store(op):
            ptr = self._get(op.get_operand(0))
            val = self._get(op.get_operand(1))

            ptr_sh = _shape_from_tensor_type_string(str(op.get_operand(0).get_type())) or []
            val_sh = _shape_from_tensor_type_string(str(op.get_operand(1).get_type())) or []

            # Make value conform to pointer shape (keep pointer untouched).
            if ptr_sh and val_sh and ptr_sh != val_sh:
                if math.prod(ptr_sh) == math.prod(val_sh):
                    val = f"tl.reshape({val}, {_fmt_shape(ptr_sh)})"
                elif len(val_sh) <= len(ptr_sh) and all(
                    (b == 1 or a == b) for a, b in zip(ptr_sh, val_sh + [1]*(len(ptr_sh)-len(val_sh)))
                ):
                    val = f"tl.broadcast_to({val}, {_fmt_shape(ptr_sh)})"

            args = [ptr, val]
            if op.get_num_operands() >= 3:
                args.append(f"mask={self._get(op.get_operand(2))}")
            cm = Attr.cache_modifier(op); ev = Attr.eviction_policy(op)
            if cm: args.append(f"cache_modifier='{cm}'")
            if ev: args.append(f"eviction_policy='{ev}'")
            return f"tl.store({', '.join(args)})"
        R["tt.store"] = emit_store


        # --- atomic read-modify-write (merge of A1 + A3)
        def _enum_from_attrs(op, names: List[str], fallback: Optional[str]) -> Optional[str]:
            for nm in names:
                s = _text_attr(op, nm)
                if s:
                    # e.g., "acq_rel" or "gpu"
                    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s)
                    if toks:
                        return toks[-1].lower()
            return fallback

        # Always forward the TTIR mask to atomics. Previously, some raised variants
        # emitted mask=None even when TTIR had a mask, which broke partial tiles/multi-program
        # launches and led to missing gradient updates. Also cast the value to the
        # pointee type when known to match Triton expectations.
        def emit_atomic_rmw(op: mlir.operation) -> str:
            # op/sem/scope decoding
            opc = _enum_from_attrs(op, ["op", "operation", "atomic_op"], None)
            sem = _enum_from_attrs(op, ["sem", "semantics", "memory_semantics", "ordering"], "relaxed")
            scope = _enum_from_attrs(op, ["scope", "mem_scope", "memory_scope"], "gpu")
            if opc is None:
                txt = op.str_nodebug()
                m = re.search(r"atomic_rmw\s+([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)", txt)
                opc, sem, scope = (m.group(1), m.group(2), m.group(3)) if m else ("add","relaxed","gpu")

            MAP = {
                "fadd":"atomic_add", "add":"atomic_add",
                "fmax":"atomic_max", "max":"atomic_max",
                "fmin":"atomic_min", "min":"atomic_min",
                "umin":"atomic_min", "umax":"atomic_max",
                "and":"atomic_and", "or":"atomic_or", "xor":"atomic_xor",
                "xchg":"atomic_xchg", "exchange":"atomic_xchg",
            }
            fn = MAP.get(opc, "atomic_add")

            ptr = self._get(op.get_operand(0))
            val = self._get(op.get_operand(1))
            # cast to pointee type when known
            try:
                pty = _dtype_expr_from_type_string(str(op.get_operand(0).get_type()))
                if pty:
                    val = f"tl.cast({val}, {pty})"
            except Exception:
                pass

            mask_kw = "None"
            if op.get_num_operands() >= 3:
                mask_kw = self._get(op.get_operand(2))

            return f"tl.{fn}({ptr}, {val}, mask={mask_kw}, sem='{sem}', scope='{scope}')"
        R["tt.atomic_rmw"] = emit_atomic_rmw

        # --- simple shape ops
        def emit_reshape(op: mlir.operation) -> str:
            x = self._get(op.get_operand(0))
            shp = _shape_from_tensor_type_string(str(op.get_result(0).get_type()))
            if shp:
                tup = "(" + ", ".join(str(d) for d in shp) + ("," if len(shp) == 1 else "") + ")"
                return f"tl.reshape({x}, {tup})"
            return f"tl.reshape({x}, None)  # TODO: dynamic shape"
        R["tt.reshape"] = emit_reshape

        R["tt.expand_dims"] = lambda op: f"tl.expand_dims({self._get(op.get_operand(0))}, axis={Attr.axis(op, 0)})"

        # Rationale: tl.broadcast returns a PAIR (lhs, rhs) and can accidentally feed a Python tuple
        # into pointer math/memory ops, leading to tuple_type errors. tl.broadcast_to returns a single
        # tensor of the target shape.
        # Also: never broadcast pointers here, because broadcasting
        # a pointer strips pointer-ness (becomes a block of values), which breaks tl.load/tl.store.
        # Instead, widen pointers only via broadcasting OFFSETS inside tt.addptr.
        # Why: Broadcasting a pointer strips pointer-ness; use tl.broadcast_to for values.
        def emit_broadcast(op):
            x = self._get(op.get_operand(0))
            # Keep pointers scalar; for values, rely on Triton's implicit broadcasting
            if "ptr<" in str(op.get_operand(0).get_type()):
                return x
            return x
        R["tt.broadcast"] = emit_broadcast


        # --- tt.addptr: keep base scalar; rely on implicit broadcasting of offsets (cast to int64)
        # Why: Widen pointers via offsets; Triton will broadcast values implicitly as needed.
        def emit_addptr(op):
            base = self._get(op.get_operand(0))  # scalar ptr
            offs = []
            for i in range(1, op.get_num_operands()):
                o = f"tl.cast({self._get(op.get_operand(i))}, tl.int64)"
                offs.append(o)
            return base if not offs else f"{base} + {' + '.join(offs)}"
        R["tt.addptr"] = emit_addptr


        # tt.splat: values return as-is (implicit broadcast). For pointers, keep pointer grid via zeros.
        def emit_splat(op):
            x = self._get(op.get_operand(0))
            shp = _shape_from_tensor_type_string(str(op.get_result(0).get_type())) or []
            if not shp:
                return x
            if _is_ptr_type(op.get_operand(0).get_type()) or _is_ptr_type(op.get_result(0).get_type()):
                return f"{x} + tl.zeros({_fmt_shape(shp)}, dtype=tl.int64)"
            # Value splat: avoid explicit broadcast; rely on Triton's implicit broadcasting.
            # Rationale: explicit tl.broadcast_to(...) is redundant noise for values.
            # Safety: stores still reshape/broadcast the VALUE (never the pointer) to pointer shape,
            # so removing value-side materialization does not change semantics.
            return x
        R["tt.splat"] = emit_splat

        # --- make_range: prefer explicit start/end
        def emit_make_range(op: mlir.operation) -> str:
            se = Attr.start_end(op)
            if se:
                return f"tl.arange({se[0]}, {se[1]})"
            shp = _shape_from_tensor_type_string(str(op.get_result(0).get_type())) or []
            n = shp[0] if len(shp) >= 1 else 0
            return f"tl.arange(0, {n})" + ("  # TODO: dynamic shape" if n == 0 else "")
        R["tt.make_range"] = emit_make_range

        # --- linear algebra
        def emit_dot(op: mlir.operation) -> str:
            a = self._get(op.get_operand(0))
            b = self._get(op.get_operand(1))
            if op.get_num_operands() >= 3:
                c = self._get(op.get_operand(2))
                return f"tl.dot({a}, {b}) + {c}"
            return f"tl.dot({a}, {b})"
        R["tt.dot"] = emit_dot

        # --- trans / permute
        def emit_trans(op):
            x = self._get(op.get_operand(0))

            # Result rank (best-effort from the result type)
            out_ty = op.get_result(0).get_type() if op.get_num_results() else None
            shp = _shape_from_tensor_type_string(str(out_ty)) or []
            rank = len(shp) if shp else None

            # Preferred path: explicit 'order' attribute (list of ints)
            ord = Attr.order(op)  # uses operation.get_i64_array_attr("order")
            if ord is None:
                # No 'order' in TTIR. If this is a 2-D transpose, assume the canonical swap.
                # Otherwise, we can't safely guess; keep the value and leave a TODO.
                return f"tl.trans({x})" if rank == 2 else f"{x}  # TODO: missing 'order' attr"

            # Identity permutation -> no-op
            if rank is not None and ord == list(range(rank)):
                return x

            # 2-D swap -> emit the idiomatic tl.trans
            if (rank == 2 and ord in ([1, 0], (1, 0))) or (ord == [1, 0] and (rank is None or rank == 2)):
                return f"tl.trans({x})"

            # Fallback if Triton doesn't expose tl.permute in your build
            return f"{x}  # TODO: unsupported permute {tuple(ord)}"
        R["tt.trans"] = emit_trans

        # --- reductions
        def emit_reduce(op: mlir.operation) -> str:
            axis = Attr.axis(op, 0)
            try:
                kind = op.get_reduce_combiner()
            except Exception:
                kind = None
            # if not kind:
            #     kinds = Attr.reduce_kinds(op, self.m)
            #     kind = kinds[0] if kinds else "sum"
            x = self._get(op.get_operand(0))
            return {
                "sum": f"tl.sum({x}, axis={axis})",
                "max": f"tl.max({x}, axis={axis})",
                "min": f"tl.min({x}, axis={axis})",
                "and": f"tl.and_reduce({x}, axis={axis})",
                "or":  f"tl.or_reduce({x}, axis={axis})",
                "xor": f"tl.xor_reduce({x}, axis={axis})",
            }.get(str(kind), f"{x}  # TODO: custom reduce")
        R["tt.reduce"] = emit_reduce
        R["tt.reduce.return"] = lambda op: None

        # --- return
        def emit_return(op: mlir.operation) -> Optional[str]:
            return
            # if op.get_num_operands() == 0:
            #     return None
            # vals = ", ".join(self._get(op.get_operand(i)) for i in range(op.get_num_operands()))
            # return f"return {vals}"
        R["tt.return"] = emit_return

        return R

    # ---- emit a single op
    def _emit_op(self, op: mlir.operation):
        name = op.mnemonic if hasattr(op, "mnemonic") else op.get_name()
        if name in ("module", "builtin.module", "tt.func", "func.func"):
            return
        if name.startswith(("scf.", "cf.")):
            self.lines.append(f"    # TODO: raise structured control-flow: {name}")
            return

        # Emit gradient grouping header if available (debounced on change)
        self._maybe_emit_grad_header(op)
        # Emit local per-handler header for non-cloned ops carrying a gradOfTag
        has_tag = (self._int_attr(op, "raise.gradOfTag") is not None)
        is_cloned = Attr.bool_attr(op, "isCloned", False)
        if has_tag and not is_cloned:
            self._maybe_emit_local_gradof(op)

        # Pre-bind results with friendly names (or minted as fallback)
        res_vars = [self._bind(op.get_result(i)) for i in range(op.get_num_results())]
        emit = self.registry.get(name)
        if emit is None:
            # Include attribute names to ease future handler additions
            try:
                attrs = list(op.get_attr_names())
                self.lines.append(f"    # TODO: raise {name} (attrs: {attrs})")
            except Exception:
                self.lines.append(f"    # TODO: raise {name}")
            return

        rhs = emit(op)
        if rhs is None:
            return
        # Elide alias-only assignments produced by value broadcast/splat removal
        # Example: "v2 = v1" where v2 came from tt.broadcast/tt.splat after elision
        if op.get_num_results() == 1 and name in ("tt.broadcast", "tt.splat"):
            rvid2 = int(op.get_result(0).id())
            if re.fullmatch(r"[A-Za-z_]\w*", rhs or ""):
                # Alias-elision: This op is a value-only alias (broadcast/splat) we decided
                # not to materialize. Bind its SSA result directly to the surviving rhs name,
                # and record alias->survivor for later label/name normalization.
                # Example: fwd_qk_3 (alias) → fwd_qk (survivor). We'll later print
                #   "# local grads for fwd_qk" instead of a dangling fwd_qk_3.
                existing = self.env.get(rvid2)
                if isinstance(existing, str) and existing != rhs:
                    self._alias_of[existing] = rhs
                self.env[rvid2] = rhs
                return
        if res_vars:
            lhs = ", ".join(res_vars) if len(res_vars) > 1 else res_vars[0]
            self.lines.append(f"    {lhs} = {rhs}")
        else:
            self.lines.append(f"    {rhs}")

    # ---- kernel top level
    def raise_kernel(self) -> str:
        self.lines.append("import triton")
        self.lines.append("import triton.language as tl")
        self.lines.append("")
        self.lines.append("# Legend:")
        self.lines.append("#    local grads for <y>                         (fine-grained: backward ops emitted when differentiating a single forward value y)")
        self.lines.append("#    ~~~~~~~~~~ grad branch for <X> ~~~~~~~~~~   (coarse: ops contributing to grad of kernel input X)") # groups of fine-grained nodes computing
        self.lines.append("")
        func = self.m.get_function(self.func_name) if self.m.has_function(self.func_name) else None
        arg_names: List[str] = []
        if func:
            for i in range(func.get_num_args()):
                v = func.args(i)
                nm = self._hints.get(self._vid(v), f"arg{i}")
                arg_names.append(self._bind(v, nm))
        # Save for grouping labels by kernel-arg index
        self._arg_names = list(arg_names)

        self.lines.append("@triton.jit")
        self.lines.append(f"def backward_{self.func_name}({', '.join(arg_names)}):")

        # Walk & emit only operations belonging to the kernel entry region
        ops: List[mlir.operation] = []
        self.m.walk(lambda o: ops.append(o))
        target_region_id = None
        if func:
            try:
                target_region_id = func.get_region(0).id()
            except Exception:
                target_region_id = None
        body_started = False
        for op in ops:
            if target_region_id is not None:
                try:
                    blk = op.get_block()
                    parent_region = blk.get_parent()
                    if parent_region.id() != target_region_id:
                        continue
                except Exception:
                    pass
            oname = op.mnemonic if hasattr(op, "mnemonic") else op.get_name()
            if not body_started and any(oname.startswith(p) for p in ("arith.","math.","tt.","scf.","cf.","triton.")):
                body_started = True
            if oname.startswith(("arith.","math.","tt.","scf.","cf.","triton.")):
                self._emit_op(op)
        if not body_started:
            self.lines.append("    pass")
        return "\n".join(self.lines)




# # in _build_registry()
# # Add a tiny helper `self._emit_region(...)` that walks the region’s ops and, when it sees `scf.yield`, assigns `iter_out[i] = <yield_i>`.

# def emit_scf_for(op):
#     lb  = self._get(op.get_operand(0)); ub = self._get(op.get_operand(1)); st = self._get(op.get_operand(2))
#     # Bind iter_args -> Python vars
#     iter_in  = [self._get(op.get_operand(i)) for i in range(3, op.get_num_operands())]
#     iter_out = [self._bind(op.get_result(i)) for i in range(op.get_num_results())]
#     # Hoist initial values
#     for dst, src in zip(iter_out, iter_in): self.lines.append(f"    {dst} = {src}")
#     self.lines.append(f"    for {self._fresh('k')} in range({lb}, {ub}, {st}):")
#     # Emit region body; capture last scf.yield operands into iter_out
#     reg = op.get_region(0)
#     self._emit_region(reg, indent=8, yield_targets=iter_out)
#     return None
# R["scf.for"] = emit_scf_for




# ----------------------------- Convenience API -------------------------------

def raise_from_module(module: mlir.module, func_name: Optional[str] = None, *, options: Optional[RaiserOptions] = None) -> str:
    return Raiser(module, func_name, opts=options).raise_kernel()

def raise_from_file(ttir_path: str, *, func_name: Optional[str] = None, options: Optional[RaiserOptions] = None) -> str:
    ctx = mlir.context()
    mlir.load_dialects(ctx)
    mod = mlir.parse_mlir_module(ttir_path, ctx)
    try:
        return Raiser(mod, func_name, opts=options).raise_kernel()
    finally:
        del mod


# ----------------------------- Demo / CLI ------------------------------------

if __name__ == "__main__":

    # print(raise_from_file(
    #     "/root/triton-autodiff/generated/1b763f2bf2/out.ttir",
    #     # "/root/triton-autodiff/third_party/autodiff/python/kernel_agent/tools/backward_naive/test/N_CTX/minimal.ttir",
    #     # "/root/triton-autodiff/third_party/autodiff/python/kernel_agent/tools/backward_naive/test/pointerness/minimal.ttir",
    #     # "/root/triton-autodiff/generated/repro/out.ttir",
    #     # "/root/triton-autodiff/third_party/autodiff/test/add-mul-div/generated/f7fc96a11c/out.ttir",
    #     options=RaiserOptions(infix_arith=True)
    # ))

    if len(sys.argv) != 2:
        print("Usage: python raise.py <path/to/file.ttir>")
        raise SystemExit(1)
    ttir_path = sys.argv[1]
    print(raise_from_file(ttir_path, options=RaiserOptions(infix_arith=True)))
