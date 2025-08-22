# cell 1 — minimal registry-driven raiser (toy, no Triton deps)
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

@dataclass
class Op:
    name: str
    results: List[int]           # SSA ids of results
    operands: List[int]          # SSA ids of operands
    attrs: Dict = field(default_factory=dict)

class ToyRaiser:
    def __init__(self):
        self.env: Dict[int, str] = {}        # value-id -> python var name
        self.lines: List[str] = []           # emitted tl.* lines
        self._n = 0
        self.registry: Dict[str, Callable[[Op], str]] = self._build_registry()

    # ---- SSA helpers
    def _fresh(self, base="v"):
        self._n += 1
        return f"{base}{self._n}"

    def bind_arg(self, vid: int, name: str):
        self.env[vid] = name

    def _bind_results(self, op: Op) -> List[str]:
        names = []
        for vid in op.results:
            # i think don't need this because graph is SSA
            # name = self.env.get(vid) or self._fresh("v")
            name = self._fresh("v")
            self.env[vid] = name
            names.append(name)
        return names

    def _get(self, vid: int, hint="v"):
        if vid not in self.env:
            self.env[vid] = self._fresh(hint)
        return self.env[vid]

    # ---- registry (op -> emitter that returns RHS string)
    def _build_registry(self):
        R: Dict[str, Callable[[Op], str]] = {}
        R["arith.addi"] = lambda op: f"({self._get(op.operands[0])} + {self._get(op.operands[1])})"
        R["arith.mulf"] = lambda op: f"({self._get(op.operands[0])} * {self._get(op.operands[1])})"
        R["arith.constant"] = lambda op: str(op.attrs.get("value", 0))
        R["arith.cmpi"] = lambda op: f"tl.{op.attrs.get('pred','equal')}({self._get(op.operands[0])}, {self._get(op.operands[1])})"
        R["arith.select"] = lambda op: f"tl.where({self._get(op.operands[0])}, {self._get(op.operands[1])}, {self._get(op.operands[2])})"
        R["tt.get_program_id"] = lambda op: f"tl.program_id({op.attrs.get('axis', 0)})"
        R["tt.load"] = lambda op: f"tl.load({self._get(op.operands[0])})"
        R["tt.store"] = lambda op: f"tl.store({self._get(op.operands[0])}, {self._get(op.operands[1])})"
        return R

    # ---- main ----
    def emit_op(self, op: Op):
        lhs_names = self._bind_results(op)
        rhs = self.registry.get(op.name, lambda o: f"# TODO: {o.name}") (op)
        if lhs_names:
            lhs = ", ".join(lhs_names)
            self.lines.append(f"{lhs} = {rhs}")
        else:
            self.lines.append(rhs)

    def raise_ops(self, ops: List[Op]):
        for op in ops:
            self.emit_op(op)
        return self.lines

# sanity check: show available handlers
ToyRaiser().registry.keys()




### cell 2 — feed a tiny TTIR‑like graph; print **env** (intermediate) and **emitted** lines

This shows SSA→names binding and one‑pass emission.


# cell 2 — build a tiny TTIR-like graph and raise it
r = ToyRaiser()

# pretend these are SSA ids coming from TTIR; bind kernel args
r.bind_arg(0, "ptr")     # pointer arg
r.bind_arg(1, "x")       # some tensor
r.bind_arg(2, "y")       # another tensor

ops = [
    Op("arith.constant", [10], [], {"value": 42}),                 # %10 = constant 42
    Op("tt.get_program_id", [11], [], {"axis": 0}),                # %11 = program_id(0)
    Op("arith.addi", [12], [10, 11]),                              # %12 = addi %10, %11
    Op("tt.load", [13], [0]),                                      # %13 = load %ptr
    Op("arith.cmpi", [14], [12, 10], {"pred": "equal"}),           # %14 = cmpi eq, %12, %10
    Op("arith.select", [15], [14, 1, 2]),                          # %15 = select %14, %x, %y
    Op("arith.mulf", [16], [13, 15]),                              # %16 = mulf %13, %15
    Op("tt.store", [], [0, 16]),                                   # store %ptr, %16
]

emitted = r.raise_ops(ops)

print("— env (SSA id -> name) —")
print({k: r.env[k] for k in sorted(r.env)})

print("\n— emitted lines —")
print("\n".join(emitted))


