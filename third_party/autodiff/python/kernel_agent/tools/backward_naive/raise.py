# raise.py
# Minimal, robust TTIR -> Triton-language raiser using the new operation bindings.
# Uses: operation.mnemonic, operation.str_nodebug(), operation.get_attr_text(),
#       operation.get_int_attr(), operation.get_i64_array_attr(),
#       operation.get_reduce_combiner() to avoid brittle regex fallbacks.

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import re
import math
import sys
from triton._C.libtriton import ir as mlir

from utils import build_value_name_hints
from naming import NameAllocator, NameStyle

# Dialects we emit and common opcode sets
DIALECTS = ("arith.", "math.", "tt.", "scf.", "cf.", "triton.")
CAST_OPS = (
    "arith.extf", "arith.truncf", "arith.fptosi", "arith.fptoui",
    "arith.sitofp", "arith.uitofp", "arith.extsi", "arith.extui", "arith.trunci",
)
CMP_SYMS = {
    "eq":"==","oeq":"==","ueq":"==",
    "ne":"!=","one":"!=","une":"!=",
    "slt":"<","ult":"<","olt":"<",
    "sle":"<=","ule":"<=","ole":"<=",
    "sgt":">","ugt":">","ogt":">",
    "sge":">=","uge":">=","oge":">=",
}

# ----------------------------- Options ---------------------------------------

@dataclass(frozen=True)
class RaiserOptions:
    # If True, print arith with symbols (a+b) instead of tl.add(a,b)
    infix_arith: bool = True
    # Emit headers grouping gradients by their source forward op
    emit_grad_groups: bool = True
    # Inline single-use pure values within the same fine-grained grad tag
    collapse_single_use: bool = True
    # Maximum line length budget for inlined RHS strings
    max_line: int = 110
    # Naming policy: "unique" (default) or "reuse" (reuse base name after last use)
    name_style: str = "unique"


# ----------------------------- Local inliner ----------------------------------

class GradLocalInliner:
    """Tag-local single-use value inliner.

    Responsibility: inline only pure, single-use producers whose fine-grained
    provenance tag (raise.gradOfTag) matches the current consumer's tag.
    Never inline loads/stores/dots/atomics/addptr/expand_dims or pointer
    results. Optionally wrap inlined RHS with parentheses to preserve
    precedence; avoid redundant parens for simple call/name/literal forms.

    The inliner is decoupled from the Raiser; it receives small callbacks for
    emission and env/lines access needed for removing already-emitted producer
    assignments when they later get inlined.
    """

    def __init__(
        self,
        *,
        registry: Dict[str, Callable[[mlir.operation], Optional[str]]],
        emit_rhs_cb: Callable[[mlir.operation, Callable[[mlir.value], str]], Optional[str]],
        op_name_fn: Callable[[mlir.operation], str],
        read_tag_fn: Callable[[mlir.operation], Optional[int]],
        group_label_fn: Callable[[mlir.operation], Optional[str]],
        is_ptr_fn: Callable[[mlir.type], bool],
        dont_inline: set,
        max_len: int,
        get_var_name_cb: Callable[[int], Optional[str]],
        lines_ref: List[str],
    ):
        self.registry = registry
        self.emit_rhs_cb = emit_rhs_cb
        self.op_name_fn = op_name_fn
        self.read_tag_fn = read_tag_fn
        self.group_label_fn = group_label_fn
        self.is_ptr_fn = is_ptr_fn
        self.dont_inline = dont_inline
        self.max_len = max_len
        self.get_var_name_cb = get_var_name_cb
        self.lines = lines_ref

        self._owner: Dict[int, mlir.operation] = {}
        self._uses: Dict[int, int] = {}
        self._cache: Dict[int, str] = {}
        self._skip_vids: set = set()
        self._current_tag: Optional[int] = None
        self._current_group: Optional[str] = None

    def prepare(self, owner: Dict[int, mlir.operation], uses: Dict[int, int]) -> None:
        self._owner = owner
        self._uses = uses
        self._cache.clear()
        self._skip_vids.clear()
        self._current_tag = None
        self._current_group = None

    def begin_consumer(self, cons_op: mlir.operation) -> None:
        self._current_tag = self.read_tag_fn(cons_op)
        try:
            self._current_group = self.group_label_fn(cons_op)
        except Exception:
            self._current_group = None

    def should_skip(self, op: mlir.operation) -> bool:
        return op.get_num_results() == 1 and int(op.get_result(0).id()) in self._skip_vids

    def _can_inline(self, v: mlir.value) -> bool:
        if self._current_tag is None:
            return False
        vid = int(v.id())
        prod = self._owner.get(vid)
        if prod is None or prod.get_num_results() != 1:
            return False
        if self._uses.get(vid, 0) != 1:
            return False
        if self.op_name_fn(prod) in self.dont_inline:
            return False
        try:
            if self.is_ptr_fn(prod.get_result(0).get_type()):
                return False
        except Exception:
            pass
        # Fine‑grained provenance only:
        # Intentionally restrict inlining to producers whose raise.gradOfTag
        # (fine tag) matches the consumer’s tag. Earlier allowed a coarse
        # fallback (same gradIdx/gradIdxs bucket) and special‑cased constants,
        # which broadened the scope and sometimes folded across human‑meaningful
        # grad branches. Now that the C++ pass tags helper ops (constants/splats)
        # with raise.gradOfTag, we can and should rely solely on the fine tag:
        #   - preserves clear "grads wrt ..." grouping and stable headers;
        #   - prevents accidental cross‑branch folding when multiple inputs share
        #     a coarse bucket;
        #   - keeps inlining deterministic and predictable (single‑use, pure,
        #     non‑pointer, same region, same fine tag).
        # If an op lacks a fine tag, we do not inline it here; the pass should
        # stamp tags on such helpers where appropriate.
        ptag = self.read_tag_fn(prod)
        return (ptag is not None) and (ptag == self._current_tag)

    def get(self, v: mlir.value, fallback_get: Callable[[mlir.value], str]) -> str:
        vid = int(v.id())
        if not self._can_inline(v):
            return fallback_get(v)
        if vid in self._cache:
            return self._cache[vid]

        prod = self._owner[vid]
        # Nested getter stays within the same tag scope
        def nested_get(u: mlir.value) -> str:
            return self.get(u, fallback_get)

        rhs = self.emit_rhs_cb(prod, nested_get)
        if rhs is None:
            return fallback_get(v)

        s = rhs.strip()
        # Wrap only if not already parenthesized/call/name/literal/indexed.
        # Treat bracket-indexed forms like x[:, None] as safe (no extra parens).
        is_parenthesized = s.startswith("(") and s.endswith(")")
        is_call_like = bool(re.match(r'^[A-Za-z_][A-Za-z0-9_\.]*\(.*\)$', s))
        is_simple_name = bool(re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', s))
        is_simple_literal = bool(re.match(r'^(-?\d+(?:\.\d+)?)$', s))
        is_indexed = s.endswith("]") and "[" in s
        need_wrap = not (is_parenthesized or is_call_like or is_simple_name or is_simple_literal or is_indexed)
        rhs_final = f"({rhs})" if need_wrap else rhs
        if len(rhs_final) > self.max_len:
            return fallback_get(v)

        self._cache[vid] = rhs_final
        self._skip_vids.add(vid)
        self._remove_emitted_assignment_for_vid(vid)
        return rhs_final

    def _remove_emitted_assignment_for_vid(self, vid: int) -> None:
        var = self.get_var_name_cb(vid)
        if not isinstance(var, str) or not var:
            return
        needle = f"{var} ="
        for i in range(len(self.lines) - 1, -1, -1):
            s = self.lines[i].strip()
            if s.startswith(needle):
                del self.lines[i]
                break

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
        return f"tl.zeros({_fmt_shape(shp)}, dtype={dty})"
    if dty.startswith("tl.float") or dty == "tl.bfloat16":
        return "0.0"
    if dty == "tl.int1":
        return "False"
    return "0"

def _lit(x):
    if isinstance(x, float):
        if math.isnan(x):  return "float('nan')"
        if math.isinf(x):  return "float('inf')" if x > 0 else "float('-inf')"
        return repr(x)
    if isinstance(x, bool): return "True" if x else "False"
    return str(int(x))


# --- small kwarg builder (avoids None/empty values) ---
def _kw_items(*pairs):
    """Return ['k=v', ...] only for non-empty values. Values should be preformatted strings."""
    return [f"{k}={v}" for k, v in pairs if v not in (None, "", [])]


# ----------------------------- Attr helpers ----------------------------------

def _text_attr(op, name: str) -> Optional[str]:
    """Return textual MLIR attr token or None (quotes stripped)."""
    try:
        a = op.get_attr_text(name)
        if a is None:
            return None
        s = str(a).strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            s = s[1:-1]
        return s or None
    except Exception:
        return None


class Attr:
    @staticmethod
    def int_attr(op, name, default=None):
        v = op.get_int_attr(name)
        return int(v) if v is not None else default

    @staticmethod
    def bool_attr(op, name, default=False):
        v = op.get_bool_attr(name)
        if v is not None:
            return bool(v)
        s = _text_attr(op, name)
        return s.strip().lower() == "true" if isinstance(s, str) else default

    @staticmethod
    def list_int_attr(op, name):
        arr = op.get_i64_array_attr(name)
        return [int(x) for x in arr] if arr is not None else None

    # Common shorthands
    axis        = staticmethod(lambda op, default=0: Attr.int_attr(op, "axis", default))
    order       = staticmethod(lambda op: Attr.list_int_attr(op, "order"))
    boundary_ck = staticmethod(lambda op: Attr.list_int_attr(op, "boundary_check"))
    @staticmethod
    def start_end(op):
        a = Attr.int_attr(op, "start"); b = Attr.int_attr(op, "end")
        if a is not None and b is not None: return (a, b)
        m = re.search(r"\bstart\s*=\s*(-?\d+).*?end\s*=\s*(-?\d+)", op.str_nodebug())
        return (int(m.group(1)), int(m.group(2))) if m else None

    @staticmethod
    def cache_modifier(op):
        t = (_text_attr(op, "cache_modifier") or "").lower()
        return {"ca":".ca","cg":".cg","cs":".cs","wb":".wb","wt":".wt","cv":".cv"}.get(t, "")

    @staticmethod
    def eviction_policy(op):
        t = (_text_attr(op, "eviction_policy") or "").lower()
        return t if t in {"evict_last","evict_first"} else ""

    @staticmethod
    def padding_option(op):
        t = (_text_attr(op, "padding_option") or "").lower()
        return "nan" if t == "nan" else ("zero" if t == "zero" else "")


# ---------------- BlockPtrEmitter: isolated tl.make_block_ptr recovery --------
class BlockPtrEmitter:
    """Pointer-grid recognizer and printer (isolated from the inliner).

    Goals:
    - Reconstruct a compact pointer grid from the canonical broadcast/mul/add tree.
      Canonical TTIR shape we match (2-D tile):
        addptr( splat(base_ptr),
                broadcast( expand_dims(m_idx, axis=1) * splat(stride_m) )
              + broadcast( expand_dims(n_idx, axis=0) * splat(stride_n) ) )
      When possible, we also factor m_idx as (splat(start_m) + ext(make_range)) and
      recognize n_idx as ext(make_range) to recover scalar start offsets.
    - Prefer tl.make_block_ptr when we have explicit shape and scalar starts (shortest
      code; enables tl.advance for subsequent steps).
    - Fall back to a robust _mk_block_ptr when metadata is missing (keeps pointer
      scalar and uses zeros+expand_dims math for the grid).
    - Provide try_emit_advance for later steps and cleanup_offset_tree for pruning
      redundant, single-use intermediates after successful emission.

    Design:
    - Prefer compact, correct reconstruction without depending on inliner state.
    - Use bound env/hints, and inline small index expressions when safe so we
      don’t reference names that haven’t been assigned yet in the Python text.
    - Keep a small cache keyed by SSA id to support tl.advance or a minimal rebuild.
    """

    def __init__(self, raiser: "Raiser"):
        self.r = raiser
        # Cache recognized block-ptr components by SSA id of the produced pointer grid.
        # Purpose: enable compact follow-up steps via tl.advance or a minimal rebuild
        # without re-walking the whole offset tree, and without touching the inliner.
        self._cache: Dict[int, Dict[str, object]] = {}

    # ---- local helpers (no inliner interaction) ----
    def _raw_name(self, v: mlir.value) -> str:
        vid = int(v.id())
        s = self.r.env.get(vid)
        if isinstance(s, str) and s:
            return s
        # Bind on demand to guarantee a usable identifier rather than a hint-only stem.
        try:
            return self.r._get(v)
        except Exception:
            return self.r._hints.get(vid, f"v{vid}")

    def _inline_or_name(self, v: mlir.value) -> str:
        """Prefer a compact RHS expression for small index producers; fallback to name.

        Minimal and safe: handles common index producers (casts, arange/make_range,
        simple adds) to avoid referencing names that may not yet be defined. If the
        producer is complex or unsupported, fall back to a bound/name hint.
        """
        try:
            op = self.r._def.get(int(v.id()))
            if op is not None and op.get_num_results() == 1:
                txt = self.r._emit_rhs_with_get(op, lambda u: self.r._get(u))
                if isinstance(txt, str) and txt and "\n" not in txt and "=" not in txt:
                    return txt
        except Exception:
            pass
        return self._raw_name(v)

    @staticmethod
    def _is_ptr_grid_type(ty: mlir.type) -> bool:
        s = str(ty)
        return s.startswith("tensor<") and "!tt.ptr<" in s

    @staticmethod
    def _grid_block_shape(ty: mlir.type) -> List[int]:
        return _shape_from_tensor_type_string(str(ty)) or []

    def _strip_extsi(self, v: mlir.value) -> mlir.value:
        op = self.r._def.get(int(v.id()))
        if op is not None and op.mnemonic == "arith.extsi" and op.get_num_operands() == 1:
            return op.get_operand(0)
        return v

    def _is_range_vec(self, v: mlir.value) -> bool:
        op = self.r._def.get(int(v.id()))
        if op is None or op.mnemonic != "arith.extsi":
            return False
        inner = op.get_operand(0)
        idef = self.r._def.get(int(inner.id()))
        return idef is not None and idef.mnemonic == "tt.make_range"

    def _parse_broadcast_term(self, v: mlir.value):
        """Return dict with keys: axis, idx, stride, start (optional), range_like(bool) or None."""
        top = self.r._def.get(int(v.id()))
        if top is None or top.mnemonic != "tt.broadcast":
            return None
        mul = self.r._def.get(int(top.get_operand(0).id()))
        if mul is None or mul.mnemonic != "arith.muli":
            return None
        a0_v, a1_v = mul.get_operand(0), mul.get_operand(1)
        a0 = self.r._def.get(int(a0_v.id()))
        a1 = self.r._def.get(int(a1_v.id()))
        if a0 is None or a1 is None:
            return None
        # Accept either order: (expand_dims, splat) or (splat, expand_dims)
        if a0.mnemonic == "tt.expand_dims" and a1.mnemonic == "tt.splat":
            ed, spl = a0, a1
        elif a1.mnemonic == "tt.expand_dims" and a0.mnemonic == "tt.splat":
            ed, spl = a1, a0
        else:
            return None
        axis = Attr.axis(ed, 0)
        idx_v = ed.get_operand(0)
        stride_v = spl.get_operand(0)

        start_v = None
        range_like = False
        idx_def = self.r._def.get(int(idx_v.id()))
        if idx_def is not None and idx_def.mnemonic == "arith.addi":
            x, y = idx_def.get_operand(0), idx_def.get_operand(1)
            xd, yd = self.r._def.get(int(x.id())), self.r._def.get(int(y.id()))
            # Detect: splat(<scalar>) + ext(make_range)
            def splat_scalar(zdef, z):
                return zdef is not None and zdef.mnemonic == "tt.splat" and zdef.get_num_operands() == 1 and "tensor<" not in str(z.get_type())
            if splat_scalar(xd, x) and self._is_range_vec(y):
                start_v = self._strip_extsi(xd.get_operand(0))
                range_like = True
            elif splat_scalar(yd, y) and self._is_range_vec(x):
                start_v = self._strip_extsi(yd.get_operand(0))
                range_like = True
        elif self._is_range_vec(idx_v):
            range_like = True

        return {"axis": axis, "idx": idx_v, "stride": stride_v, "start": start_v, "range_like": range_like}

    # ---- root stem discovery for pointer bases ----
    def _is_ptr(self, ty: mlir.type) -> bool:
        """Shallow pointer check used by _root_stem to follow the pointer-bearing side.

        Note: we intentionally do not parse types structurally here; a textual
        check is sufficient and robust for Triton pointer types in TTIR.
        """
        return "!tt.ptr<" in str(ty)

    def _root_stem(self, v: mlir.value) -> str:
        """Recover a human stem from the original base pointer for grids.

        Algorithm (structure-driven, no regex on names):
        - Start from the value that feeds the grid (usually the base of a splat).
        - Walk backwards through ops that preserve the pointer identity:
          tt.advance, tt.splat, tt.broadcast, bitcast/extend/trunc, and
          pointer+offset adds (choose the pointer-bearing operand).
        - Stop at the first non-pointer-preserving op or when we detect a cycle.
        - Use the raiser-provided stem (hints) of that root pointer as the base.

        Rationale: this avoids brittle suffix chopping and works regardless of
        kernel-specific naming. The allocator will still ensure uniqueness.
        """
        seen = set()
        cur = v
        while True:
            vid = int(cur.id())
            if vid in seen:
                break
            seen.add(vid)
            op = self.r._def.get(vid)
            if op is None:
                break
            m = getattr(op, "mnemonic", op.get_name())
            try:
                if m in ("tt.advance", "tt.splat", "tt.broadcast", "arith.bitcast", "arith.extsi", "arith.extui", "arith.trunci") and op.get_num_operands() >= 1:
                    cur = op.get_operand(0)
                    continue
                if m in ("arith.addi", "arith.addf") and op.get_num_operands() == 2:
                    a, b = op.get_operand(0), op.get_operand(1)
                    if self._is_ptr(a.get_type()) and not self._is_ptr(b.get_type()):
                        cur = a; continue
                    if self._is_ptr(b.get_type()) and not self._is_ptr(a.get_type()):
                        cur = b; continue
            except Exception:
                pass
            break
        return self.r._hints.get(int(cur.id()), f"v{int(cur.id())}")

    def _shape_hint_text(self, op: mlir.operation) -> Optional[str]:
        # Prefer raiser-provided metadata if present (opt-in during lowering)
        shp_syms = _text_attr(op, "raise.shape_syms")
        if isinstance(shp_syms, str):
            syms = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", shp_syms)
            if len(syms) == 2:
                return f"({syms[0]}, {syms[1]})"
        shp_ints = Attr.list_int_attr(op, "raise.shape")
        if isinstance(shp_ints, list) and len(shp_ints) == 2:
            return f"({shp_ints[0]}, {shp_ints[1]})"
        # No shape metadata -> return None to allow fallback to _mk_block_ptr path.
        return None

    # ---- main entry ----
    def try_emit_make_block_ptr(self, op: mlir.operation) -> Optional[str]:
        """Attempt to emit a canonical block pointer construction.

        - Emits tl.make_block_ptr when shape/starts are available (enables
          later tl.advance; shortest code).
        - Otherwise emits a compact _mk_block_ptr(...), robust to missing
          metadata.
        - Never calls Raiser._get, keeping inliner state stable.
        """
        # Expect exactly two operands: pointer grid and offsets tensor
        if op.get_num_operands() != 2:
            return None
        ptr_grid_v = op.get_operand(0)
        offs_v = op.get_operand(1)

        # Pointer grid must be a splat(base_ptr) with tensor<MXN x !tt.ptr<T>>
        if not self._is_ptr_grid_type(ptr_grid_v.get_type()):
            return None
        spl = self.r._def.get(int(ptr_grid_v.id()))
        if spl is None or spl.mnemonic != "tt.splat":
            return None
        base_ptr_v = spl.get_operand(0)
        block_shape = self._grid_block_shape(spl.get_result(0).get_type())
        if len(block_shape) != 2:
            return None

        # Offsets must be addi(bcast(m-term), bcast(n-term)) with expected axes
        add = self.r._def.get(int(offs_v.id()))
        if add is None or add.mnemonic != "arith.addi":
            return None
        A = self._parse_broadcast_term(add.get_operand(0))
        B = self._parse_broadcast_term(add.get_operand(1))
        if not A or not B or {A["axis"], B["axis"]} != {0, 1}:
            return None

        # Place m on axis=1 and n on axis=0 (row-major tile)
        if A["axis"] == 1:
            m_term, n_term = A, B
        else:
            m_term, n_term = B, A

        # Prefer uncluttered stride symbols: strip arith.extsi(i32->i64) if present.
        stride_m_txt = self._raw_name(self._strip_extsi(m_term["stride"]))
        stride_n_txt = self._raw_name(self._strip_extsi(n_term["stride"]))

        base_txt = self._raw_name(base_ptr_v)
        shape_txt = self._shape_hint_text(op)
        order = (1, 0) if m_term["axis"] == 1 else (0, 1)

        s = None
        kind = "mk"
        # Only emit tl.make_block_ptr when we have scalar starts and explicit shape
        if (m_term.get("start") is not None) and (shape_txt is not None):
            m_start_txt = self._raw_name(m_term["start"])
            n_start_txt = self._raw_name(n_term["start"]) if n_term.get("start") is not None else "0"
            s = (
                "tl.make_block_ptr("
                f"base={base_txt}, "
                f"shape={shape_txt}, "
                f"strides=({stride_m_txt}, {stride_n_txt}), "
                f"offsets=({m_start_txt}, {n_start_txt}), "
                f"block_shape={_fmt_shape(block_shape)}, order={order})"
            )
            kind = "make"
        else:
            # Fallback: call a jitted helper with constexpr block dims.
            # Reason: keeps code short, is legal device code, and centralizes the grid math.
            # Context: we didn't have explicit shape/starts to call tl.make_block_ptr safely.
            # The helper expects BM/BN as compile-time ints and internally casts strides to i64.
            BM, BN = block_shape
            m_idx_txt = self._inline_or_name(m_term["idx"]) 
            n_idx_txt = self._inline_or_name(n_term["idx"]) 
            s = (
                f"_mk_block_ptr({base_txt}, {m_idx_txt}, {n_idx_txt}, "
                f"{stride_m_txt}, {stride_n_txt}, {BM}, {BN})"
            )

        # Cache components for potential tl.advance or compact rebuild
        try:
            out_vid = int(op.get_result(0).id())
            # Hint the result stem to a canonical role name derived from the root pointer.
            # We normalize to avoid accidental duplication of "_ptr" or "_block_ptr".
            # This stays a hint only; the allocator enforces uniqueness/reuse.
            try:
                root = self._root_stem(base_ptr_v)
                root_norm = re.sub(r"(_block_ptr|_ptr)$", "", root or "")
                self.r._hints[out_vid] = f"{root_norm}_block_ptr"
            except Exception:
                pass
            self._cache[out_vid] = {
                "kind": kind, "order": order, "block_shape": tuple(block_shape),
                "base": base_ptr_v, "m_idx": m_term["idx"], "n_idx": n_term["idx"],
                "stride_m": m_term["stride"], "stride_n": n_term["stride"],
            }
        except Exception:
            pass
        return s

    def try_emit_advance(self, op: mlir.operation) -> Optional[str]:
        """Emit a compact pointer step when based on a prior block-ptr.

        Prefers tl.advance(base, (dm, dn)) if base was built with tl.make_block_ptr
        and has a simple name; otherwise rebuilds with _mk_block_ptr and updated
        indices. Returns None when the pattern doesn't match a 2D grid step.
        """
        # Expect two operands: base block-ptr and offsets
        if op.get_num_operands() != 2:
            return None
        base_v = op.get_operand(0)
        try:
            info = self._cache.get(int(base_v.id()))
        except Exception:
            info = None
        if not info:
            return None
        add = self.r._def.get(int(op.get_operand(1).id()))
        if add is None or add.mnemonic != "arith.addi":
            return None
        A = self._parse_broadcast_term(add.get_operand(0))
        B = self._parse_broadcast_term(add.get_operand(1))
        if not A or not B or {A["axis"], B["axis"]} != {0, 1}:
            return None
        if A["axis"] == 1:
            dm_v, dn_v = A["idx"], B["idx"]
        else:
            dm_v, dn_v = B["idx"], A["idx"]
        rg = self._raw_name
        bp_name = rg(base_v)
        dm = rg(dm_v); dn = rg(dn_v)
        # Prefer tl.advance when base is a simple identifier
        if info.get("kind") == "make" and re.fullmatch(r"[A-Za-z_]\w*", bp_name or ""):
            # Keep the same human name across advances: if base has a simple identifier,
            # set the produced result's hint to that base name so the allocator binds it
            # consistently. This improves readability of successive tl.advance steps.
            try:
                out_vid = int(op.get_result(0).id())
                self.r._hints[out_vid] = bp_name
            except Exception:
                pass
            return f"tl.advance({bp_name}, ({dm}, {dn}))"
        # Otherwise, rebuild via the jitted helper with updated indices.
        # Reason: avoids inlining a large zeros+broadcast expression and stays legal device code.
        m_idx = f"({rg(info['m_idx'])} + {dm})"
        n_idx = f"({rg(info['n_idx'])} + {dn})"
        base_txt = rg(info['base'])
        stride_m_txt = rg(info['stride_m'])
        stride_n_txt = rg(info['stride_n'])
        BM, BN = info["block_shape"]
        return (
            f"_mk_block_ptr({base_txt}, {m_idx}, {n_idx}, "
            f"{stride_m_txt}, {stride_n_txt}, {BM}, {BN})"
        )

    def cleanup_offset_tree(self, ptr_grid_v, offs_v):
        """Best-effort DCE of single-use broadcast/mul/add feeding addptr.

        Uses the inliner's recorded use counts to safely remove only temps that
        are proven single-use. Never calls Raiser._get or mutates emitter state.
        """
        try:
            inl = getattr(self.r, "_inliner", None)
            if inl is None:
                return
            uses = getattr(inl, "_uses", {}) or {}
            remover = getattr(inl, "_remove_emitted_assignment_for_vid", None)
            if remover is None:
                return
            add = self.r._def.get(int(offs_v.id()))
            if add is None or getattr(add, "mnemonic", "") != "arith.addi":
                return
            # Always remove the head add node: after peephole it's redundant.
            # Reason: we fully replace the offsets tree with a self-contained RHS (make/advance/helper);
            # leaving this add would keep references to temps we intentionally pruned and cause NameError.
            try:
                remover(int(offs_v.id()))
            except Exception:
                pass
            vids = []
            for t in (add.get_operand(0), add.get_operand(1)):
                top = self.r._def.get(int(t.id()))
                if top is None:
                    continue
                if top.mnemonic == "tt.broadcast":
                    if uses.get(int(t.id()), 0) <= 1:
                        vids.append(int(t.id()))
                    mul = self.r._def.get(int(top.get_operand(0).id()))
                    if mul is not None and mul.mnemonic == "arith.muli" and uses.get(int(mul.get_result(0).id()), 0) <= 1:
                        vids.append(int(mul.get_result(0).id()))
                elif top.mnemonic == "arith.muli":
                    if uses.get(int(t.id()), 0) <= 1:
                        vids.append(int(t.id()))
            spl = self.r._def.get(int(ptr_grid_v.id()))
            if spl is not None and spl.mnemonic == "tt.splat" and uses.get(int(ptr_grid_v.id()), 0) <= 1:
                vids.append(int(ptr_grid_v.id()))
            for vid in vids:
                remover(vid)
        except Exception:
            pass

    def maybe_cleanup(self, op: mlir.operation) -> None:
        """Wrapper to safely invoke cleanup after a successful emission."""
        try:
            self.cleanup_offset_tree(op.get_operand(0), op.get_operand(1))
        except Exception:
            pass


# ----------------------------- Raiser ----------------------------------------

class Raiser:
    def __init__(self, module: mlir.module, func_name: Optional[str], *, opts: Optional[RaiserOptions] = None):
        self.m = module
        self.func_name = func_name or self.m.get_entry_func_name() or "raised_kernel"
        self.opts = opts or RaiserOptions()
        self.lines: List[str] = []
        self._n = 0
        # Naming model:
        # - utils provides clean, sanitized stems from NameLocs (no uniquing).
        # - The raiser is the single authority for uniqueness/reuse, because only
        #   here we have def-use/liveness and region context (esp. with backward ops).
        # - This avoids noisy or unstable suffixes introduced too early in the flow.
        self._hints: Dict[int, str] = build_value_name_hints(self.m)
        self.env: Dict[int, str] = {}
        # Pluggable allocator: ensures uniqueness (and optional reuse) at emission time.
        # Why not in codegen: codegen lacks liveness/region context and backward inserts;
        # uniquing there easily leads to noisy or unstable names. The raiser knows
        # last-use and gradient grouping, so it can reuse or suffix only when lifetimes
        # overlap (semantics-preserving).
        self._namer = NameAllocator(
            NameStyle.REUSE_AFTER_LAST_USE if getattr(self.opts, "name_style", "unique") == "reuse" else NameStyle.UNIQUE
        )
        # Remaining use counts per SSA value id (decremented on each _get) for reuse mode.
        self._uses: Dict[int, int] = {}
        self.registry = self._build_registry()
        # Def map for robust chain inspection (e.g., nested casts) without regex over text
        # Key: SSA id (int), Value: defining MLIR operation
        self._def: Dict[int, mlir.operation] = {}
        # Track current gradient group label to reduce noisy headers
        self._last_grad_of: Optional[str] = None
        # Track emitted fine-grained headers by tag id to ensure we print a header
        # for the first actually-printed statement in each backward tag group.
        # This avoids losing headers when earlier statements under the same tag are
        # inlined or DCE'd later.
        self._emitted_local_tags: set[int] = set()
        # Also track the last printed local header label to avoid repeating the
        # exact same "grads wrt <label>" comment back-to-back across
        # consecutive statements mapped to different tags but same label.
        self._last_local_label: Optional[str] = None
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
        # Isolated helper for block‑ptr recovery/advance
        self._bp = BlockPtrEmitter(self)

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
        # Use clean stem from hints; final uniqueness/reuse is handled by the allocator.
        base = name
        py = self._namer.claim(base)
        self.env[vid] = py
        return py

    def _get(self, v: mlir.value, hint: str = "v") -> str:
        vid = self._vid(v)
        if vid not in self.env:
            self._bind(v, self._hints.get(vid, self._fresh(hint)))
        py = self.env[vid]
        # Only in reuse mode: decrement and release at last use to allow safe base reuse.
        # In unique mode, release is a no-op to avoid rebinding the same identifier.
        if vid in self._uses:
            try:
                self._uses[vid] -= 1
                if self._uses[vid] <= 0:
                    self._namer.release(py)
            except Exception:
                pass
        return py

    def _name(self, op: mlir.operation) -> str:
        return op.mnemonic if hasattr(op, "mnemonic") else op.get_name()

    @staticmethod
    def _in_region(op: mlir.operation, rid) -> bool:
        if rid is None:
            return True
        try:
            blk = op.get_block()
            parent_region = blk.get_parent()
            return parent_region.id() == rid
        except Exception:
            return False

    @staticmethod
    def _is_supported(name: str) -> bool:
        return name.startswith(DIALECTS)

    def _collect_ops(self) -> List[mlir.operation]:
        ops: List[mlir.operation] = []
        self.m.walk(lambda o: ops.append(o))
        return ops

    # ---- emission helpers
    def _arith(self, a: str, b: str, sym: str, fn: str) -> str:
        return f"{a} {sym} {b}" if self.opts.infix_arith else f"tl.{fn}({a}, {b})"

    def _cast(self, x: str, dst_ty: mlir.type) -> str:
        dty = _dtype_expr_from_type_string(str(dst_ty)) or "None"
        return f"tl.cast({x}, {dty})"

    def _emit_cast_op(self, op: mlir.operation) -> Optional[str]:
        """Emit a value cast with elision/chain collapse.
        Rules:
        - Identity casts are dropped (bind as alias of source).
        - Nested casts within same family (float->float, int->int) collapse to one.
        - Otherwise keep a single tl.cast.
        Pointer/IO casts elsewhere (addptr/atomics) remain intact.
        """
        dst = op.get_result(0).get_type()
        src_v = op.get_operand(0)
        src = self._get(src_v)
        dst_ty = _dtype_expr_from_type_string(str(dst)) or "None"
        src_ty = _dtype_expr_from_type_string(str(src_v.get_type())) or "None"

        # Identity: drop
        if src_ty == dst_ty:
            return src

        # Collapse nested cast via def-use instead of regex
        is_float = lambda t: t.startswith("tl.float") or t == "tl.bfloat16"
        # Inspect producer op directly; if producer is unknown (e.g., block arg),
        # fall back to emitting a single cast.
        prod = self._def.get(int(src_v.id()))
        if prod is not None:
            pm = self._name(prod)
            if pm in CAST_OPS:
                inner_v = prod.get_operand(0)
                inner = self._get(inner_v)
                inner_ty = _dtype_expr_from_type_string(str(inner_v.get_type())) or "None"
                # Only collapse float-family chains; integer zero/sign extend chains can differ.
                if (is_float(inner_ty) and is_float(dst_ty)):
                    return f"tl.cast({inner}, {dst_ty})"

        return f"tl.cast({src}, {dst_ty})"

    def _bitcast(self, x: str, dst_ty: mlir.type) -> str:
        dty = _dtype_expr_from_type_string(str(dst_ty)) or "None"
        return f"tl.bitcast({x}, {dty})"


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
        # Emit a fine-grained header exactly once per tag id (first printed stmt).
        tag = self._int_attr(op, "raise.gradOfTag")
        if tag is None:
            return
        tid = int(tag)
        if tid in self._emitted_local_tags:
            return
        py_lbl = self._resolve_local_label(op, None)
        if not py_lbl:
            return
        # Avoid emitting the same local header twice in a row (even if separated
        # by blank lines). Find the most recent comment line and compare.
        comment_line = f"    # grads wrt {py_lbl}"
        last_comment = None
        for i in range(len(self.lines) - 1, -1, -1):
            s = self.lines[i]
            if s.strip() == "":
                continue
            if s.lstrip().startswith("#"):
                last_comment = s
            break
        # Also suppress if the last printed local label equals this one.
        if last_comment == comment_line or self._last_local_label == py_lbl:
            # Same logical header was just printed previously → skip duplicate
            self._emitted_local_tags.add(tid)
            return
        # Avoid inserting a blank line before the fine-grained local header to
        # prevent empty lines between successive fine-grained comments.
        self.lines.append(comment_line)
        self._last_local_label = py_lbl
        self._emitted_local_tags.add(tid)

    def _build_tag_to_py_map(self) -> Dict[int, str]:
        tag2name: Dict[int, str] = {}
        ops: List[mlir.operation] = []
        self.m.walk(lambda o: ops.append(o))
        for o in ops:
            tag = self._int_attr(o, "raise.gradOfTag")
            if tag is None:
                continue
            # Deterministic label for this tag:
            # - Prefer a result stem starting with "fwd_" (forward value names)
            # - Else prefer any non-bwd stem (to avoid "bwd_*" in headers)
            # - Else skip (we'll try another op with the same tag or fallback later)
            best_fwd = None
            any_non_bwd = None
            for i in range(o.get_num_results()):
                nm = self._hints.get(self._vid(o.get_result(i)))
                if isinstance(nm, str):
                    if best_fwd is None and nm.startswith("fwd_"):
                        best_fwd = nm
                    if any_non_bwd is None and not nm.startswith("bwd_"):
                        any_non_bwd = nm
            cand = best_fwd if best_fwd is not None else any_non_bwd
            if cand is None:
                continue
            tid = int(tag)
            # Prefer to keep an existing fwd_* choice; only overwrite if upgrading
            # from a non-fwd to a fwd stem.
            prev = tag2name.get(tid)
            if prev is None or (not prev.startswith("fwd_") and cand.startswith("fwd_")):
                tag2name[tid] = cand
        return tag2name

    def _resolve_local_label(self, op, fallback: str) -> str:
        tag = self._int_attr(op, "raise.gradOfTag")
        if tag is None:
            return fallback
        name = self._fwd_tag_to_py.get(int(tag), fallback)
        if not isinstance(name, str):
            return fallback
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

            # Emit a scalar literal and rely on Triton/Python broadcasting for shaped constants.
            return _lit(v)
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

        # --- casts (value-side elision/chain collapse)
        R["arith.extf"]    = self._emit_cast_op
        R["arith.truncf"]  = self._emit_cast_op
        R["arith.fptosi"]  = self._emit_cast_op
        R["arith.fptoui"]  = self._emit_cast_op
        R["arith.sitofp"]  = self._emit_cast_op
        R["arith.uitofp"]  = self._emit_cast_op
        R["arith.bitcast"] = lambda op: self._bitcast(self._get(op.get_operand(0)), op.get_result(0).get_type())
        # integer width casts
        for _k in ("arith.extsi", "arith.extui", "arith.trunci"):
            R[_k] = self._emit_cast_op

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
            # Fail loudly on unknown preds to avoid silently emitting incorrect equality.
            if pred not in CMP_SYMS:
                raise RuntimeError(f"Unsupported/unknown cmp predicate: {pred} in {op.str_nodebug()}")
            return f"{a} {CMP_SYMS[pred]} {b}"

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
            sem = _enum_from_attrs(op, ["sem", "semantics", "memory_semantics", "ordering"], None)
            scope = _enum_from_attrs(op, ["scope", "mem_scope", "memory_scope"], None)
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

            mask_arg = None
            if op.get_num_operands() >= 3:
                mask_val = self._get(op.get_operand(2))
                # Only include mask kwarg if it's not the Python default (None)
                if mask_val != "None":
                    mask_arg = f"mask={mask_val}"

            # Only include sem/scope when explicitly present and not defaults
            extra = []
            if mask_arg:
                extra.append(mask_arg)
            if sem and sem != "acq_rel":
                extra.append(f"sem='{sem}'")
            if scope and scope != "gpu":
                extra.append(f"scope='{scope}'")

            tail = (", " + ", ".join(extra)) if extra else ""
            return f"tl.{fn}({ptr}, {val}{tail})"
        R["tt.atomic_rmw"] = emit_atomic_rmw

        # --- simple shape ops
        def emit_reshape(op: mlir.operation) -> str:
            x = self._get(op.get_operand(0))
            shp = _shape_from_tensor_type_string(str(op.get_result(0).get_type()))
            if shp:
                return f"tl.reshape({x}, {_fmt_shape(shp)})"
            return f"tl.reshape({x}, None)  # TODO: dynamic shape"
        R["tt.reshape"] = emit_reshape

        # Indexing sugar for rank-1 expand_dims: emit [:, None] / [None, :] with parens for precedence
        def emit_expand_dims(op):
            x = self._get(op.get_operand(0))
            axis = Attr.axis(op, 0)
            src = _shape_from_tensor_type_string(str(op.get_operand(0).get_type())) or []
            if len(src) == 1 and axis in (0, 1):
                # Emit indexing sugar without parentheses universally; rely on Python precedence for indexing.
                return f"{x}[None, :]" if axis == 0 else f"{x}[:, None]"
            return f"tl.expand_dims({x}, axis={axis})"
        R["tt.expand_dims"] = emit_expand_dims

        # Value-only broadcast elides to the source. Pointers remain scalar.
        R["tt.broadcast"] = lambda op: self._get(op.get_operand(0))

        # --- tt.addptr
        # Try to reconstruct tl.make_block_ptr for canonical pointer-grid patterns.
        # This logic is fully isolated inside BlockPtrEmitter and never touches inliner state.
        def emit_addptr(op):
            # Delegate to the emitter. It decides between:
            #  - tl.make_block_ptr(...): when explicit shape/starts exist
            #  - _mk_block_ptr(...): robust fallback when metadata is missing
            #  - tl.advance(...)/minimal rebuild for subsequent steps
            s = self._bp.try_emit_make_block_ptr(op)
            if s is not None:
                self._bp.maybe_cleanup(op)
                return s
            s2 = self._bp.try_emit_advance(op)
            if s2 is not None:
                self._bp.maybe_cleanup(op)
                return s2
            # fallback: base + offsets
            base = self._get(op.get_operand(0))
            offs = []
            for i in range(1, op.get_num_operands()):
                oi = op.get_operand(i)
                si = self._get(oi)
                if "i64" not in str(oi.get_type()):
                    si = f"tl.cast({si}, tl.int64)"
                offs.append(si)
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
                acc_v = op.get_operand(2)
                acc_def = self._def.get(int(acc_v.id()))
                is_zero = False
                if acc_def is not None and (hasattr(acc_def, 'mnemonic') and acc_def.mnemonic == 'arith.constant'):
                    try:
                        val = acc_def.get_splat_value('value')
                        if isinstance(val, (int, float)):
                            is_zero = (val == 0 or val == 0.0)
                        elif isinstance(val, bool):
                            is_zero = (val is False)
                    except Exception:
                        is_zero = False
                c = self._get(acc_v)
                if is_zero or c in ("0", "0.0", "False"):
                    return f"tl.dot({a}, {b})"
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
            return None
            # if op.get_num_operands() == 0:
            #     return None
            # vals = ", ".join(self._get(op.get_operand(i)) for i in range(op.get_num_operands()))
            # return f"return {vals}"
        R["tt.return"] = emit_return

        return R

    # ---- emit a single op
    def _emit_op(self, op: mlir.operation):
        name = op.mnemonic
        if name in ("module", "builtin.module", "tt.func", "func.func"):
            return
        if name.startswith(("scf.", "cf.")):
            self.lines.append(f"    # TODO: raise structured control-flow: {name}")
            return

        # Note: headers are emitted only when we actually print a line for this op
        # (see below), to avoid dangling comments when the op gets fully inlined.

        # Bind results AFTER computing RHS to maximize reuse opportunities (previous
        # operands may release at last use). This also prevents binding names for results
        # that end up fully inlined by the local inliner.
        # Delay binding of result names until after RHS emission to maximize base-name reuse
        res_count = op.get_num_results()
        # Track defining op for robust post-inspection (cast chain, etc.)
        for i in range(res_count):
            self._def[int(op.get_result(i).id())] = op
        emit = self.registry.get(name)
        if emit is None:
            # Include attribute names to ease future handler additions
            try:
                attrs = list(op.get_attr_names())
                self.lines.append(f"    # TODO: raise {name} (attrs: {attrs})")
            except Exception:
                self.lines.append(f"    # TODO: raise {name}")
            return

        # Optional tag-local inlining: if enabled, set up consumer context and inline
        # eligible operands. The inliner folds only pure, single-use values. It never
        # folds loads/stores/dots/atomics/addptr/expand_dims or pointers.
        if getattr(self, "_inliner", None) is not None:
            # Skip this op entirely if its single result was fully inlined elsewhere
            if self._inliner.should_skip(op):
                return
            self._inliner.begin_consumer(op)
            saved_get = self._get
            try:
                def _inline_get(v):
                    return self._inliner.get(v, saved_get)
                self._get = _inline_get
                rhs = emit(op)
            finally:
                self._get = saved_get
        else:
            rhs = emit(op)
        if rhs is None:
            return
        # Elide alias-only assignments more aggressively for readability:
        # - Always inline scalar constants and expand-dims sugar (rank-1 -> indexing),
        #   by binding the SSA id directly to the RHS expression and skipping the assignment line.
        # - Keep existing behavior for broadcast/splat when RHS is a simple name.
        if op.get_num_results() == 1:
            rvid2 = int(op.get_result(0).id())
            if name in ("arith.constant", "tt.expand_dims"):
                # Inline constants (incl. float('-inf')) and expand-dims like x[:, None]
                self.env[rvid2] = rhs
                return
            if name in ("tt.broadcast", "tt.splat"):
                # Legacy alias-elision path for broadcasts/splats of names
                if re.fullmatch(r"[A-Za-z_]\w*", rhs or ""):
                    self.env[rvid2] = rhs
                    return
        # Emit headers right before we actually print a statement for this op
        # (after inlining/alias-elision decisions), so they never dangle.
        # This avoids emitting fine-grained comments for ops that get inlined away.
        self._maybe_emit_grad_header(op)
        # Emit fine-grained headers only for backward ops (not cloned forward ones)
        has_tag = (self._int_attr(op, "raise.gradOfTag") is not None)
        is_cloned = Attr.bool_attr(op, "isCloned", False)
        if has_tag and not is_cloned:
            self._maybe_emit_local_gradof(op)

        # Bind result names now, after operand uses may have released prior owners
        res_vars = [self._bind(op.get_result(i)) for i in range(res_count)]
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
        # Jitted helper for pointer-grid construction with constexpr tile sizes.
        # Reason: allows short, readable calls from kernels without violating Triton's rule
        # about non-constexpr globals. BM/BN are compile-time ints (tile shape), so calls
        # like _mk_block_ptr(base, m_idx, n_idx, stride_m, stride_n, 16, 16) are valid.
        self.lines.append("@triton.jit")
        self.lines.append("def _mk_block_ptr(base, m_idx, n_idx, stride_m, stride_n, BM: tl.constexpr, BN: tl.constexpr):")
        self.lines.append("    # Device helper: rebuild a pointer grid without broadcasting the pointer itself.")
        self.lines.append("    # BM/BN are constexpr tile sizes. Casts strides to int64 to satisfy addptr rules.")
        self.lines.append("    stride_m = tl.cast(stride_m, tl.int64)")
        self.lines.append("    stride_n = tl.cast(stride_n, tl.int64)")
        self.lines.append("    grid = base + tl.zeros((BM, BN), dtype=tl.int64)")
        self.lines.append("    return grid + m_idx[:, None] * stride_m + n_idx[None, :] * stride_n")
        self.lines.append("")
        self.lines.append("# Legend:")
        self.lines.append("#    grads wrt <y>                               (fine-grained: backward ops emitted when differentiating a single forward value y)")
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
        ops = self._collect_ops()
        target_region_id = None
        if func:
            try:
                target_region_id = func.get_region(0).id()
            except Exception:
                target_region_id = None
        # Precompute liveness/use counts for safe name reuse
        self._uses = {}
        for op in ops:
            if not self._in_region(op, target_region_id):
                continue
            if not self._is_supported(op.mnemonic):
                continue
            for j in range(op.get_num_operands()):
                try:
                    vid = int(op.get_operand(j).id())
                    self._uses[vid] = self._uses.get(vid, 0) + 1
                except Exception:
                    pass
        # Prepare defs/uses for optional inlining
        if self.opts.collapse_single_use:
            owner, uses = self._compute_owner_and_uses(ops, target_region_id)
            # Prefer the same uses for inliner and reuse to keep consistent
            self._uses = dict(uses)
            self._inliner = GradLocalInliner(
                registry=self.registry,
                emit_rhs_cb=lambda o, get_fn: self._emit_rhs_with_get(o, get_fn),
                op_name_fn=lambda o: (o.mnemonic if hasattr(o, "mnemonic") else o.get_name()),
                read_tag_fn=lambda o: self._int_attr(o, "raise.gradOfTag"),
                group_label_fn=lambda o: self._group_label(o),
                is_ptr_fn=_is_ptr_type,
                dont_inline={"tt.load","tt.store","tt.dot","tt.atomic_rmw","tt.addptr"},
                max_len=self.opts.max_line,
                get_var_name_cb=lambda vid: self.env.get(vid),
                lines_ref=self.lines,
            )
            self._inliner.prepare(owner, uses)
        else:
            self._inliner = None

        body_started = False
        for op in ops:
            if not self._in_region(op, target_region_id):
                continue
            oname = self._name(op)
            if not body_started and self._is_supported(oname):
                body_started = True
            if self._is_supported(oname):
                self._emit_op(op)

        if not body_started:
            self.lines.append("    pass")
        # Final pass: drop assigned-but-never-used temporaries and then sweep orphan headers.
        self._strip_dead_temporaries()
        self._sweep_orphan_headers()
        return "\n".join(self.lines)

    # small helpers for inliner integration
    def _emit_rhs_with_get(self, op: mlir.operation, get_fn: Callable[[mlir.value], str]) -> Optional[str]:
        name = op.mnemonic
        handler = self.registry.get(name)
        if handler is None:
            return None
        saved = self._get
        try:
            self._get = get_fn
            return handler(op)
        finally:
            self._get = saved

    def _compute_owner_and_uses(self, ops: List[mlir.operation], target_region_id) -> (Dict[int, mlir.operation], Dict[int, int]):
        owner: Dict[int, mlir.operation] = {}
        uses: Dict[int, int] = {}
        # Keep _def available for cast-chain inspection and other helpers
        self._def.clear()
        for op in ops:
            if not self._in_region(op, target_region_id):
                continue
            name = op.mnemonic
            if not self._is_supported(name):
                continue
            for i in range(op.get_num_results()):
                vid = int(op.get_result(i).id())
                owner[vid] = op
                self._def[vid] = op
            for j in range(op.get_num_operands()):
                try:
                    vid = int(op.get_operand(j).id())
                    uses[vid] = uses.get(vid, 0) + 1
                except Exception:
                    pass
        return owner, uses

    # the local header is decided before we actually append the statement, but there’s a corner
    # case where the next printed thing is a different header (e.g., a coarse branch header) and
    # no statement for that fine tag ever gets appended at that spot. That leaves a header line by itself
    def _sweep_orphan_headers(self) -> None:
        def is_coarse(s: str) -> bool:
            return s.lstrip().startswith("# ~~~~~~~~~~ grad branch")
        def is_local(s: str) -> bool:
            return s.lstrip().startswith("# grads wrt ")
        n = len(self.lines)
        out: List[str] = []
        i = 0
        while i < n:
            s = self.lines[i]
            if is_coarse(s):
                j = i + 1
                # allow local headers under a coarse header
                while j < n and (self.lines[j].strip() == "" or is_local(self.lines[j])):
                    j += 1
                # drop only if no code before next coarse header/EOF
                if j >= n or is_coarse(self.lines[j]):
                    i += 1
                    continue
            elif is_local(s):
                j = i + 1
                while j < n and self.lines[j].strip() == "":
                    j += 1
                # drop truly orphan local header
                if j >= n or is_coarse(self.lines[j]) or is_local(self.lines[j]):
                    i += 1
                    continue
            out.append(s)
            i += 1
        # collapse blank runs
        cleaned: List[str] = []
        prev_blank = False
        for s in out:
            b = (s.strip() == "")
            if b and prev_blank:
                continue
            cleaned.append(s)
            prev_blank = b
        self.lines = cleaned



# ----------------------------- Convenience API -------------------------------

    def _strip_dead_temporaries(self) -> None:
        """Drop 'x = <expr>' when x is never used later; keep atomics as calls.
        Operates purely on the emitted text to sweep trivial dead assigns at the end.
        Only considers kernel-body lines (4-space indent). Comments/headers are kept verbatim.
        """
        assign_re = re.compile(r"^\s{4}([A-Za-z_]\w*)\s*=\s*(.+)$")
        ident_re  = re.compile(r"\b[A-Za-z_]\w*\b")

        used: set[str] = set()
        out: list[str] = []

        for line in reversed(self.lines):
            # Only touch kernel body lines (4-space indent); keep others unchanged
            if not line.startswith("    "):
                out.append(line)
                continue

            m = assign_re.match(line)
            if not m:
                # propagate uses from non-assignment lines
                used.update(ident_re.findall(line))
                out.append(line)
                continue

            lhs, rhs = m.group(1), m.group(2).strip()

            if lhs not in used:
                # side-effecting atomics: keep the call, drop the assignment
                if re.search(r"\btl\.atomic_[a-z]+", rhs):
                    out.append("    " + rhs)
                    used.update(ident_re.findall(rhs))
                # pure dead temp: drop whole line
                else:
                    continue
            else:
                # keep assignment and propagate tokens
                used.update(ident_re.findall(rhs))
                used.add(lhs)
                out.append(line)

        self.lines = list(reversed(out))

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
    if len(sys.argv) != 2:
        print("Usage: python raise.py <path/to/file.ttir>")
        raise SystemExit(1)
    ttir_path = sys.argv[1]
    print(raise_from_file(ttir_path, options=RaiserOptions(infix_arith=True)))


