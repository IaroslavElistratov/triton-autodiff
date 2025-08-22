# raise.py -- End-to-end TTIR -> Triton-language raiser using MLIR bindings.
# Uses value names derived from MLIR locations via ir.value_best_name (no regex).

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import triton
import triton.language as tl
from triton._C.libtriton import ir as mlir

from utils import ValueNamer, build_value_name_map

@dataclass
class RaiserOptions:
    # If True, print a+b instead of tl.add(a,b)
    infix_arith: bool = True

class Raiser:
    def __init__(self, module: mlir.module, func_name: Optional[str], *, opts: Optional[RaiserOptions] = None):
        self.m = module
        self.func_name = func_name or self.m.get_entry_func_name() or "raised_kernel"
        self.opts = opts or RaiserOptions()
        # Names: prebind from NameLoc to ensure deterministic, meaningful ids
        self.namer = ValueNamer()
        self.namer.prebind_module(self.m)
        self.lines: List[str] = []
        self.registry = self._build_registry()

    # --------------- Utils ---------------
    def _name(self, v: mlir.value) -> str:
        return self.namer.name(v)

    def _arith(self, a: str, b: str, sym: str, fn: str) -> str:
        return f"({a} {sym} {b})" if self.opts.infix_arith else f"tl.{fn}({a}, {b})"

    def _dtype_expr_from_type_string(self, t: str) -> Optional[str]:
        # Minimal map for common types
        if "bf16" in t: return "tl.bfloat16"
        if "f16"  in t: return "tl.float16"
        if "f32"  in t: return "tl.float32"
        if "f64"  in t: return "tl.float64"
        if "i1"   in t: return "tl.int1"
        if "i8"   in t: return "tl.int8"
        if "i16"  in t: return "tl.int16"
        if "i32"  in t: return "tl.int32"
        if "i64"  in t: return "tl.int64"
        return None

    # --------------- Registry ---------------
    def _build_registry(self) -> Dict[str, Callable[[mlir.operation], str]]:
        R: Dict[str, Callable[[mlir.operation], str]] = {}

        # Binary arith
        def bin2(op, sym, fn):
            return self._arith(self._name(op.get_operand(0)), self._name(op.get_operand(1)), sym, fn)
        for k, sym, fn in (
            ("arith.addf", "+", "add"), ("arith.addi", "+", "add"),
            ("arith.subf", "-", "sub"), ("arith.subi", "-", "sub"),
            ("arith.mulf", "*", "mul"), ("arith.muli", "*", "mul"),
        ):
            R[k] = (lambda op, s=sym, f=fn: bin2(op, s, f))

        # Div/mod (float div is special in Triton)
        R["arith.divf"]  = lambda op: f"tl.fdiv({self._name(op.get_operand(0))}, {self._name(op.get_operand(1))}, False)"
        R["arith.divsi"] = lambda op: f"tl.floordiv({self._name(op.get_operand(0))}, {self._name(op.get_operand(1))})"
        R["arith.divui"] = R["arith.divsi"]
        R["arith.remf"]  = lambda op: f"tl.mod({self._name(op.get_operand(0))}, {self._name(op.get_operand(1))})"
        R["arith.remsi"] = R["arith.remf"]; R["arith.remui"] = R["arith.remf"]

        # Bitwise
        R["arith.andi"] = lambda op: self._arith(self._name(op.get_operand(0)), self._name(op.get_operand(1)), "&", "and_")
        R["arith.ori"]  = lambda op: self._arith(self._name(op.get_operand(0)), self._name(op.get_operand(1)), "|", "or_")
        R["arith.xori"] = lambda op: self._arith(self._name(op.get_operand(0)), self._name(op.get_operand(1)), "^", "xor_")

        # Casts (best-effort)
        R["arith.truncf"] = lambda op: f"tl.cast({self._name(op.get_operand(0))}, {self._dtype_expr_from_type_string(str(op.get_result(0).get_type())) or 'None'})"
        R["arith.extf"]   = R["arith.truncf"]
        R["arith.sitofp"] = R["arith.truncf"]
        R["arith.uitofp"] = R["arith.truncf"]
        R["arith.fptosi"] = R["arith.truncf"]
        R["arith.fptoui"] = R["arith.truncf"]
        R["arith.bitcast"]= lambda op: f"tl.bitcast({self._name(op.get_operand(0))}, {self._dtype_expr_from_type_string(str(op.get_result(0).get_type())) or 'None'})"
        R["tt.bitcast"]   = R["arith.bitcast"]

        # Compare & select (predicate best-effort)
        R["arith.select"] = lambda op: f"tl.where({self._name(op.get_operand(0))}, {self._name(op.get_operand(1))}, {self._name(op.get_operand(2))})"
        R["arith.cmpf"]   = lambda op: f"/*cmpf*/ tl.equal({self._name(op.get_operand(0))}, {self._name(op.get_operand(1))})"
        R["arith.cmpi"]   = lambda op: f"/*cmpi*/ tl.equal({self._name(op.get_operand(0))}, {self._name(op.get_operand(1))})"

        # Elementary math
        for m in ("floor","ceil","exp","exp2","cos","sin","log","log2","erf","sqrt","rsqrt","abs"):
            R[f"math.{m}"] = (lambda op, mm=m: f"tl.{mm}({self._name(op.get_operand(0))})")

        # Programming model (axis defaults to 0; op int attrs aren't exposed here)
        R["tt.get_program_id"]   = lambda op: f"tl.program_id(0)"
        R["tt.get_num_programs"] = lambda op: f"tl.num_programs(0)"

        # Memory
        def emit_load(op: mlir.operation) -> str:
            n = op.get_num_operands()
            if n == 1:
                return f"tl.load({self._name(op.get_operand(0))})"
            if n == 3:
                return f"tl.load({self._name(op.get_operand(0))}, mask={self._name(op.get_operand(1))}, other={self._name(op.get_operand(2))})"
            args = ", ".join(self._name(op.get_operand(i)) for i in range(n))
            return f"tl.load({args})"
        R["tt.load"] = emit_load

        def emit_store(op: mlir.operation) -> str:
            n = op.get_num_operands()
            if n == 2:
                return f"tl.store({self._name(op.get_operand(0))}, {self._name(op.get_operand(1))})"
            if n == 3:
                return f"tl.store({self._name(op.get_operand(0))}, {self._name(op.get_operand(1))}, mask={self._name(op.get_operand(2))})"
            args = ", ".join(self._name(op.get_operand(i)) for i in range(n))
            return f"tl.store({args})"
        R["tt.store"] = emit_store

        # Pointers / shape
        R["tt.addptr"]      = lambda op: f"({self._name(op.get_operand(0))} + {self._name(op.get_operand(1))})"
        R["tt.make_range"]  = lambda op: f"tl.arange(0, 0)  # TODO: fill start,end if available"
        R["tt.expand_dims"] = lambda op: f"tl.expand_dims({self._name(op.get_operand(0))}, 0)  # axis=0 (unknown)"
        R["tt.reshape"]     = lambda op: f"tl.reshape({self._name(op.get_operand(0))}, /*shape*/None, can_reorder=False)"
        R["tt.trans"]       = lambda op: f"/* tl.permute(x, order) */ {self._name(op.get_operand(0))}"

        # Constants: emit a typed zero as placeholder (attribute getter not exposed on Operation)
        R["arith.constant"] = lambda op: "0"

        return R

    # --------------- Emit one op ---------------
    def _emit_op(self, op: mlir.operation):
        name = op.get_name()
        # skip module/func wrappers
        if name in ("module", "builtin.module", "tt.func", "func.func"):
            return
        emit = self.registry.get(name)
        # Result LHS (use prebound names)
        lhs_vars = [self._name(op.get_result(i)) for i in range(op.get_num_results())]
        if emit:
            rhs = emit(op)
            if lhs_vars:
                lhs = ", ".join(lhs_vars) if len(lhs_vars) > 1 else lhs_vars[0]
                self.lines.append(f"    {lhs} = {rhs}")
            else:
                self.lines.append(f"    {rhs}")
        else:
            # Unknown op: keep as comment
            self.lines.append(f"    # TODO raise {name}")

    # --------------- Kernel top-level ---------------
    def raise_kernel(self) -> str:
        self.lines.append("import triton")
        self.lines.append("import triton.language as tl")
        self.lines.append("")

        # Entry function + args
        func = self.m.get_function(self.func_name) if self.m.has_function(self.func_name) else None
        arg_names: List[str] = []
        if func:
            for i in range(func.get_num_args()):
                v = func.args(i)
                arg_names.append(self._name(v))

        self.lines.append("@triton.jit")
        self.lines.append(f"def {self.func_name}({', '.join(arg_names)}):")
        self.lines.append("    # Raised from TTIR (best-effort).")

        # Walk and emit
        ops: List[mlir.operation] = []
        self.m.walk(lambda op_ptr: ops.append(op_ptr))
        for op in ops:
            if any(op.get_name().startswith(p) for p in ("arith.", "math.", "tt.")):
                self._emit_op(op)

        if self.lines[-1].endswith("):"):  # empty body guard
            self.lines.append("    pass")
        return "\n".join(self.lines)

# --------------- Convenience API ---------------
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

if __name__ == "__main__":
    # Example usage
    print(raise_from_file("/root/triton-autodiff/third_party/autodiff/test/flash_attention_v2/generated/annotated/inp.ttir", options=RaiserOptions(infix_arith=True)))
