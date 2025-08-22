# raise.py
# Minimal, robust TTIR -> Triton-language raiser.
# - No use of op.str_nodebug() (that exists on OpState, not operation).
# - Names for SSA values come from ir.value_best_name(v) via utils.build_value_name_hints.
#
# Note: Attribute access for integers/enums is limited in the current pybind
# (operation exposes get_name/operands/results/regions but not generic int attrs).
# Where an integer/enum attr would be required (e.g., cmp predicate, axis, orders),
# we fall back to sensible defaults and emit a clear TODO comment instead of crashing.

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import re

import triton
import triton.language as tl
from triton._C.libtriton import ir as mlir

from utils import build_value_name_hints


# ----------------------------- Options ---------------------------------------

@dataclass
class RaiserOptions:
    # If True, print arith with symbols (a+b) instead of tl.add(a,b)
    infix_arith: bool = True


# ----------------------------- Type helpers ----------------------------------

def _dtype_expr_from_type_string(t: str) -> Optional[str]:
    # quick-and-robust from MLIR type string
    if "bf16" in t: return "tl.bfloat16"
    if "f16"  in t: return "tl.float16"
    if "f32"  in t: return "tl.float32"
    if "f64"  in t: return "tl.float64"
    if "i1"   in t: return "tl.int1"
    m = None
    # i8/i16/i32/i64 anywhere typical
    import re
    m = re.search(r"(?:^|[x<,])i(8|16|32|64)(?:[>x,]|$)", t)
    if m:
        return {"8":"tl.int8","16":"tl.int16","32":"tl.int32","64":"tl.int64"}[m.group(1)]
    return None


def _shape_from_tensor_type_string(t: str) -> Optional[List[int]]:
    # "tensor<128x64xf16>" -> [128,64]
    import re
    m = re.search(r"tensor<([^>]+)>", t)
    if not m:
        return None
    parts = m.group(1).split("x")
    try:
        return [int(d) for d in parts[:-1]]
    except ValueError:
        return None  # dynamic dims -> give up


def _typed_zero(dst_ty: mlir.type) -> str:
    """Emit a neutral literal or zeros tensor matching the type."""
    t = str(dst_ty)
    shp = _shape_from_tensor_type_string(t)
    dty = _dtype_expr_from_type_string(t) or "None"
    if shp:
        # (dim,) tuple rendering
        tup = "(" + ", ".join(str(d) for d in shp) + ("," if len(shp) == 1 else "") + ")"
        return f"tl.zeros({tup}, dtype={dty})"
    # scalar
    if dty.startswith("tl.float") or dty == "tl.bfloat16":
        return "0.0"
    if dty == "tl.int1":
        return "False"
    return "0"


# ----------------------------- Attr helpers ------------------------------------

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
        try:
            b = op.get_bool_attr(name)
            if b is not None:
                return bool(b)
        except Exception:
            pass
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

    # Load/store enums → Triton strings
    @staticmethod
    def cache_modifier(op):
        # accepts textual forms seen in TTIR; normalize to tl.load/store spellings
        txt = None
        try:
            txt = op.get_attr_text("cache_modifier")
        except Exception:
            pass
        s = (txt or op.str_nodebug()).lower()
        if ".ca" in s or " cache_modifier = ca" in s or "cache_modifier=ca" in s: return ".ca"
        if ".cg" in s or " cache_modifier = cg" in s or "cache_modifier=cg" in s: return ".cg"
        if ".cs" in s or " cache_modifier = cs" in s or "cache_modifier=cs" in s: return ".cs"
        if ".wb" in s or " cache_modifier = wb" in s or "cache_modifier=wb" in s: return ".wb"
        if ".wt" in s or " cache_modifier = wt" in s or "cache_modifier=wt" in s: return ".wt"
        if ".cv" in s or " cache_modifier = cv" in s or "cache_modifier=cv" in s: return ".cv"
        return ""

    @staticmethod
    def eviction_policy(op):
        s = None
        try:
            s = op.get_attr_text("eviction_policy")
        except Exception:
            pass
        s = (s or op.str_nodebug()).lower()
        for k in ("evict_last", "evict_first"):
            if k in s: return k
        return ""

    @staticmethod
    def padding_option(op):
        txt = None
        try:
            txt = op.get_attr_text("padding_option")
        except Exception:
            pass
        s = (txt or op.str_nodebug()).lower()
        if "nan" in s:  return "nan"
        if "zero" in s: return "zero"
        return ""

    # Detect per-result reduce combiners by inspecting the region
    @staticmethod
    def reduce_kinds(op, module):
        # Prefer the C++ helper if available
        try:
            comb = op.get_reduce_combiner()
        except Exception:
            comb = None
        if comb:
            return [str(comb)] * max(1, op.get_num_results())

        reg = op.get_region(0); target_rid = reg.id()
        val_owner = {}
        reduce_ret = None

        def visit(inner):
            # keep only ops in this region
            b = inner.get_block(); r = b.get_parent() if b is not None else None
            if b is None:
                return
            while r:
                if r.id() == target_rid:
                    # record defs
                    for i in range(inner.get_num_results()):
                        v = inner.get_result(i)
                        val_owner[int(v.id())] = inner.get_name()
                    # find tt.reduce.return
                    if inner.get_name().endswith("reduce.return"):
                        nonlocal reduce_ret; reduce_ret = inner
                    break
                r = r.get_parent_region()
        module.walk(visit)

        kinds = []
        if reduce_ret:
            MAP = {
                "arith.addi":"sum","arith.addf":"sum",
                "arith.maxsi":"max","arith.maxui":"max","arith.maximumf":"max","arith.maxnumf":"max",
                "arith.minsi":"min","arith.minui":"min","arith.minimumf":"min","arith.minnumf":"min",
                "arith.andi":"and","arith.ori":"or","arith.xori":"xor",
            }
            for i in range(reduce_ret.get_num_operands()):
                vid = int(reduce_ret.get_operand(i).id())
                kinds.append(MAP.get(val_owner.get(vid, ""), "custom"))
        return kinds

# ----------------------------- Raiser ----------------------------------------

class Raiser:
    def __init__(self, module: mlir.module, func_name: Optional[str], *, opts: Optional[RaiserOptions] = None):
        self.m = module
        self.func_name = func_name or self.m.get_entry_func_name() or "raised_kernel"
        self.opts = opts or RaiserOptions()
        self.lines: List[str] = []
        self._n = 0
        # Names for values (results + args), derived from value_best_name
        self._hints: Dict[int, str] = build_value_name_hints(self.m)
        # Final env: value-id -> bound variable
        self.env: Dict[int, str] = {}
        self.registry = self._build_registry()

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
            # prefer hint; if absent, mint
            name = self._hints.get(vid, self._fresh("v"))
        self.env[vid] = name
        return name

    def _get(self, v: mlir.value, hint: str = "v") -> str:
        vid = self._vid(v)
        if vid not in self.env:
            self._bind(v, self._hints.get(vid, self._fresh(hint)))
        return self.env[vid]

    # ---- emission helpers
    def _arith(self, a: str, b: str, sym: str, fn: str) -> str:
        return f"({a} {sym} {b})" if self.opts.infix_arith else f"tl.{fn}({a}, {b})"

    def _cast(self, x: str, dst_ty: mlir.type) -> str:
        dty = _dtype_expr_from_type_string(str(dst_ty)) or "None"
        return f"tl.cast({x}, {dty})"

    def _bitcast(self, x: str, dst_ty: mlir.type) -> str:
        dty = _dtype_expr_from_type_string(str(dst_ty)) or "None"
        return f"tl.bitcast({x}, {dty})"

    # ---- registry
    def _build_registry(self) -> Dict[str, Callable[[mlir.operation], Optional[str]]]:
        R: Dict[str, Callable[[mlir.operation], Optional[str]]] = {}

        # --- arith constant (no text parse; typed zero if value not available)
        def emit_constant(op: mlir.operation) -> str:
            ty = op.get_result(0).get_type() if op.get_num_results() else None
            return _typed_zero(ty) if ty is not None else "0"
        R["arith.constant"] = emit_constant

        # --- binary arithmetic
        def bin2(op, sym, fn):
            return self._arith(self._get(op.get_operand(0)),
                               self._get(op.get_operand(1)), sym, fn)
        for k, sym, fn in (
            ("arith.addf", "+", "add"), ("arith.addi", "+", "add"),
            ("arith.subf", "-", "sub"), ("arith.subi", "-", "sub"),
            ("arith.mulf", "*", "mul"), ("arith.muli", "*", "mul"),
            ("arith.divf", "/", "fdiv"),  # integer div variants omitted for now
        ):
            R[k] = (lambda op, s=sym, f=fn: bin2(op, s, f))

        # --- casts
        R["arith.extf"]   = lambda op: self._cast(self._get(op.get_operand(0)), op.get_result(0).get_type())
        R["arith.truncf"] = lambda op: self._cast(self._get(op.get_operand(0)), op.get_result(0).get_type())
        R["arith.fptosi"] = lambda op: self._cast(self._get(op.get_operand(0)), op.get_result(0).get_type())
        R["arith.fptoui"] = lambda op: self._cast(self._get(op.get_operand(0)), op.get_result(0).get_type())
        R["arith.sitofp"] = lambda op: self._cast(self._get(op.get_operand(0)), op.get_result(0).get_type())
        R["arith.uitofp"] = lambda op: self._cast(self._get(op.get_operand(0)), op.get_result(0).get_type())
        R["arith.bitcast"] = lambda op: self._bitcast(self._get(op.get_operand(0)), op.get_result(0).get_type())

        # --- select (ternary)
        R["arith.select"] = lambda op: f"tl.where({self._get(op.get_operand(0))}, {self._get(op.get_operand(1))}, {self._get(op.get_operand(2))})"

        # --- compares with predicate decoding via op.get_attr_text("predicate")
        def _emit_cmp_with_pred(op: mlir.operation, is_int: bool) -> str:
            a = self._get(op.get_operand(0)); b = self._get(op.get_operand(1))
            pred_txt = None
            try:
                pred_txt = op.get_attr_text("predicate")
            except Exception:
                pred_txt = None
            s = (pred_txt or op.str_nodebug()).lower()
            table = {
                "eq":"==","oeq":"==","ueq":"==",
                "ne":"!=","one":"!=","une":"!=",
                "slt":"<","ult":"<","olt":"<",
                "sle":"<=","ule":"<=","ole":"<=",
                "sgt":">","ugt":">","ogt":">",
                "sge":">=","uge":">=","oge":">=",
            }
            # pick the first key that appears in the predicate text
            order = ("oeq","ueq","one","une","olt","ole","ogt","oge",
                     "eq","ne","slt","sle","sgt","sge","ult","ule","ugt","uge")
            key = next((k for k in order if k in s), None)
            op_sym = table.get(key, "==")
            return f"({a} {op_sym} {b})"
        R["arith.cmpf"] = lambda op: _emit_cmp_with_pred(op, is_int=False)
        R["arith.cmpi"] = lambda op: _emit_cmp_with_pred(op, is_int=True)

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

        # --- Triton builtins (axis not readable -> default axis=0, keep TODO)
        R["tt.get_program_id"]   = lambda op: "tl.program_id(axis=0)  # TODO: axis"
        R["tt.get_num_programs"] = lambda op: "tl.num_programs(axis=0)  # TODO: axis"

        # --- memory
        def emit_load(op: mlir.operation) -> str:
            # tt.load(ptr [, mask [, other]])
            argc = op.get_num_operands()
            ptr = self._get(op.get_operand(0)) if argc >= 1 else "ptr"
            args = [ptr]
            if argc >= 2:
                args.append(f"mask={self._get(op.get_operand(1))}")
            if argc >= 3:
                args.append(f"other={self._get(op.get_operand(2))}")
            return f"tl.load({', '.join(args)})"
        R["tt.load"] = emit_load

        def emit_store(op: mlir.operation) -> Optional[str]:
            # tt.store(ptr, value [, mask])
            argc = op.get_num_operands()
            if argc < 2:
                return "pass  # malformed store"
            ptr = self._get(op.get_operand(0))
            val = self._get(op.get_operand(1))
            if argc >= 3:
                m = self._get(op.get_operand(2))
                return f"tl.store({ptr}, {val}, mask={m})"
            return f"tl.store({ptr}, {val})"
        R["tt.store"] = emit_store

        # --- simple shape ops (derive shape from result type string)
        def emit_reshape(op: mlir.operation) -> str:
            x = self._get(op.get_operand(0))
            shp = _shape_from_tensor_type_string(str(op.get_result(0).get_type()))
            if shp:
                tup = "(" + ", ".join(str(d) for d in shp) + ("," if len(shp) == 1 else "") + ")"
                return f"tl.reshape({x}, {tup})"
            return f"tl.reshape({x}, None)  # TODO: dynamic shape"
        R["tt.reshape"] = emit_reshape

        def emit_expand_dims(op: mlir.operation) -> str:
            # axis attr not readable -> default axis=0
            x = self._get(op.get_operand(0))
            return f"tl.expand_dims({x}, axis=0)  # TODO: axis"
        R["tt.expand_dims"] = emit_expand_dims

        def emit_broadcast(op: mlir.operation) -> str:
            x = self._get(op.get_operand(0))
            shp = _shape_from_tensor_type_string(str(op.get_result(0).get_type()))
            if shp:
                tup = "(" + ", ".join(str(d) for d in shp) + ("," if len(shp) == 1 else "") + ")"
                return f"tl.broadcast({x}, {tup})"
            return f"tl.broadcast({x}, None)  # TODO: dynamic shape"
        R["tt.broadcast"] = emit_broadcast

        # You can keep appending handlers here…

        # --- integer division & remainder (signed/unsigned)
        R["arith.divsi"] = lambda op: self._arith(
            self._get(op.get_operand(0)), self._get(op.get_operand(1)), "//", "floordiv"
        )
        R["arith.divui"] = lambda op: self._arith(
            self._get(op.get_operand(0)), self._get(op.get_operand(1)), "//", "floordiv"
        )
        R["arith.remsi"] = lambda op: self._arith(
            self._get(op.get_operand(0)), self._get(op.get_operand(1)), "%", "mod"
        )
        R["arith.remui"] = R["arith.remsi"]

        # --- sign/zero extend and truncate (use tl.cast to the result type)
        for _k in ("arith.extsi", "arith.extui", "arith.trunci"):
            R[_k] = (lambda op: self._cast(self._get(op.get_operand(0)),
                                           op.get_result(0).get_type()))

        # --- elementwise min/max with "num" semantics
        R["arith.maxnumf"] = lambda op: (
            f"tl.maximum({self._get(op.get_operand(0))}, {self._get(op.get_operand(1))})"
        )
        R["arith.minnumf"] = lambda op: (
            f"tl.minimum({self._get(op.get_operand(0))}, {self._get(op.get_operand(1))})"
        )

        # --- common math (already present for some; harmless to overwrite)
        R["math.exp2"] = lambda op: f"tl.exp2({self._get(op.get_operand(0))})"
        R["math.log2"] = lambda op: f"tl.log2({self._get(op.get_operand(0))})"

        # --- pointer arith / address calc
        def emit_addptr(op: mlir.operation) -> str:
            # tt.addptr(ptr, off[, off2, ...]) -> ptr + off (+ off2 ...)
            terms = [self._get(op.get_operand(i)) for i in range(op.get_num_operands())]
            base, offs = terms[0], terms[1:]
            return f"({base} " + " + ".join([""] + offs) + ")"
        R["tt.addptr"] = emit_addptr

        # --- splat: scalar -> block tensor
        def emit_splat(op: mlir.operation) -> str:
            x = self._get(op.get_operand(0))
            shp = _shape_from_tensor_type_string(str(op.get_result(0).get_type()))
            if shp:
                tup = "(" + ", ".join(str(d) for d in shp) + ("," if len(shp) == 1 else "") + ")"
                return f"tl.broadcast({x}, {tup})"
            return f"tl.broadcast({x}, None)  # TODO: dynamic shape"
        R["tt.splat"] = emit_splat

        # --- make_range: 0..N-1
        def emit_make_range(op: mlir.operation) -> str:
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
                return f"(tl.dot({a}, {b}) + {c})"
            return f"tl.dot({a}, {b})"
        R["tt.dot"] = emit_dot

        # --- trans / permute using order attr when available
        def emit_trans(op: mlir.operation) -> str:
            x = self._get(op.get_operand(0))
            ord = Attr.order(op)
            if ord is not None:
                return f"tl.permute({x}, tuple({ord}))"
            return f"tl.trans({x})  # TODO: order"
        R["tt.trans"] = emit_trans

        # --- reductions (best-effort; can’t inspect combiner/axis robustly here)
        def emit_reduce(op: mlir.operation) -> str:
            axis = Attr.axis(op, 0)
            kinds = Attr.reduce_kinds(op, self.m)
            xs = [self._get(op.get_operand(i)) for i in range(op.get_num_operands())]
            x = xs[0] if len(xs) == 1 else f"({', '.join(xs)})"
            outs: List[str] = []
            for k in kinds:
                if k == "sum":
                    outs.append(f"tl.sum({x}, axis={axis})")
                elif k == "max":
                    outs.append(f"tl.max({x}, axis={axis})")
                elif k == "min":
                    outs.append(f"tl.min({x}, axis={axis})")
                elif k in {"and", "or", "xor"}:
                    outs.append(f"tl.{k}_reduce({x}, axis={axis})")
                else:
                    outs.append(f"{x}  # TODO: custom reduce")
            return (", ".join(outs)) if len(outs) > 1 else outs[0]
        R["tt.reduce"] = emit_reduce
        R["tt.reduce.return"] = lambda op: None

        # --- override: program id / num programs (axis)
        R["tt.get_program_id"]   = lambda op: f"tl.program_id(axis={max(0, min(2, Attr.axis(op, 0)))})"
        R["tt.get_num_programs"] = lambda op: f"tl.num_programs(axis={max(0, min(2, Attr.axis(op, 0)))})"

        # --- override: expand_dims(axis)
        R["tt.expand_dims"] = lambda op: f"tl.expand_dims({self._get(op.get_operand(0))}, axis={Attr.axis(op, 0)})"

        # --- override: load/store with cache/evict/padding/boundary_check
        def emit_load(op: mlir.operation) -> str:
            ptr = self._get(op.get_operand(0))
            args: List[str] = [f"{ptr}"]
            if op.get_num_operands() >= 2:
                args.append(f"mask={self._get(op.get_operand(1))}")
            if op.get_num_operands() >= 3:
                args.append(f"other={self._get(op.get_operand(2))}")
            bc = Attr.boundary_ck(op)
            pad = Attr.padding_option(op)
            cm = Attr.cache_modifier(op)
            ev = Attr.eviction_policy(op)
            if bc:
                args.append(f"boundary_check={tuple(bc)}")
            if pad:
                args.append(f"padding_option='{pad}'")
            if cm:
                args.append(f"cache_modifier='{cm}'")
            if ev:
                args.append(f"eviction_policy='{ev}'")
            return f"tl.load({', '.join(args)})"
        R["tt.load"] = emit_load

        def emit_store(op: mlir.operation) -> str:
            ptr = self._get(op.get_operand(0))
            val = self._get(op.get_operand(1))
            args: List[str] = [f"{ptr}", f"{val}"]
            if op.get_num_operands() >= 3:
                args.append(f"mask={self._get(op.get_operand(2))}")
            cm = Attr.cache_modifier(op)
            ev = Attr.eviction_policy(op)
            if cm:
                args.append(f"cache_modifier='{cm}'")
            if ev:
                args.append(f"eviction_policy='{ev}'")
            return f"tl.store({', '.join(args)})"
        R["tt.store"] = emit_store

        # --- override: make_range(start, end) with fallback to shape-based
        def emit_make_range2(op: mlir.operation) -> str:
            se = Attr.start_end(op)
            if se:
                return f"tl.arange({se[0]}, {se[1]})"
            shp = _shape_from_tensor_type_string(str(op.get_result(0).get_type())) or []
            n = shp[0] if len(shp) >= 1 else 0
            return f"tl.arange(0, {n})" + ("  # TODO: dynamic shape" if n == 0 else "")
        R["tt.make_range"] = emit_make_range2

        return R

    # ---- emit a single op
    def _emit_op(self, op: mlir.operation):
        # Prefer mnemonic if available (new binding), fallback to get_name()
        name = getattr(op, "mnemonic", op.get_name())
        if name in ("module", "builtin.module", "tt.func", "func.func"):
            return
        if name.startswith(("scf.", "cf.")):
            self.lines.append(f"    # TODO: raise structured control-flow: {name}")
            return

        # Pre-bind results with friendly names (or minted as fallback)
        res_vars = [self._bind(op.get_result(i)) for i in range(op.get_num_results())]
        emit = self.registry.get(name)
        if emit is None:
            self.lines.append(f"    # TODO: raise {name}")
            return

        rhs = emit(op)
        if rhs is None:
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
        # Bind entry args if present (names were already hinted in utils)
        func = self.m.get_function(self.func_name) if self.m.has_function(self.func_name) else None
        arg_names: List[str] = []
        if func:
            for i in range(func.get_num_args()):
                v = func.args(i)
                nm = self._hints.get(self._vid(v), f"arg{i}")
                arg_names.append(self._bind(v, nm))

        self.lines.append("@triton.jit")
        self.lines.append(f"def {self.func_name}({', '.join(arg_names)}):")
        self.lines.append("    # Raised from TTIR (best-effort).")

        # Walk & emit
        ops: List[mlir.operation] = []
        self.m.walk(lambda o: ops.append(o))
        body_started = False
        for op in ops:
            oname = getattr(op, "mnemonic", op.get_name())
            if not body_started and any(oname.startswith(p) for p in ("arith.","math.","tt.","scf.","cf.","triton.")):
                body_started = True
            if oname.startswith(("arith.","math.","tt.","scf.","cf.","triton.")):
                self._emit_op(op)
        if not body_started:
            self.lines.append("    pass")
        return "\n".join(self.lines)


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
    # Adjust the path if you want to test on a different TTIR file.
    print(raise_from_file(
        "/root/triton-autodiff/third_party/autodiff/test/flash_attention_v2/generated/annotated/out.ttir",
        options=RaiserOptions(infix_arith=True)
    ))
