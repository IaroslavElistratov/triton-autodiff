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

        # --- compares (predicate not accessible -> placeholder to avoid crash)
        def _cmp_placeholder(op: mlir.operation) -> str:
            a = self._get(op.get_operand(0)); b = self._get(op.get_operand(1))
            # Emit equality as a benign default; keep a TODO to invite filling in later.
            return f"({a} == {b})  # TODO: refine predicate"
        R["arith.cmpi"] = _cmp_placeholder
        R["arith.cmpf"] = _cmp_placeholder

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

        R["tt.trans"] = lambda op: f"tl.trans({self._get(op.get_operand(0))})"

        # --- reductions (best-effort; can’t inspect combiner/axis robustly here)
        def emit_reduce(op: mlir.operation) -> str:
            x = self._get(op.get_operand(0))
            return f"tl.sum({x}, axis=0)  # TODO: combiner/axis"
        R["tt.reduce"] = emit_reduce
        R["tt.reduce.return"] = lambda op: None

        return R

    # ---- emit a single op
    def _emit_op(self, op: mlir.operation):
        name = op.get_name()
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
            if not body_started and any(op.get_name().startswith(p) for p in ("arith.","math.","tt.","scf.","cf.","triton.")):
                body_started = True
            if op.get_name().startswith(("arith.","math.","tt.","scf.","cf.","triton.")):
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
