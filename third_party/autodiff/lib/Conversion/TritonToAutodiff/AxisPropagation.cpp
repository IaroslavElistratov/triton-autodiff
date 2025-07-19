// #include "mlir/Config/Version.h"  // adds MLIR_VERSION_MAJOR macro
// static_assert(MLIR_VERSION_MAJOR >= 18, "Old MLIR headers detected");


#include "autodiff/include/Conversion/TritonToAutodiff/AxisPropagation.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/StringSet.h"

#include <string>


// ────────────────────────────────────────────────────────────────────────────
// AxisPropagation.cpp  -  propagate logical‑axis names through Triton/TT IR
//
// * Each tensor‑valued SSA value may carry `tt.axis_names = ["N","D",...]`
//   describing the logical meaning of its *rank* dimensions.
// * Every *index‑producer* (PID x/y/z, loop IV, etc.) must carry
//   `tt.axis = "<logical‑name>"`; PIDs are not special – the loop IV that
//   walks SEQ_LEN needs the same tag so `"SEQ_LEN"` survives inside the `scf.for` body.
// * `tt.addptr` now **filters** its incoming `axis_names` list by the set
//   of live tags seen on its dynamic operands (every numeric offset input of the tt.addptr,
//   apart from the first base pointer) -> different offsets keep different names
//      * Every dynamic *offset operand* (PID, loop‑IV math, etc.) carry
//        `tt.axis = "NAME"` - these are the *live* logical axes for *this* ptr
//      * Start from the incoming axis list on the base pointer, e.g. ["SEQ_LEN","HEAD","HEAD_DIM"]
//      * Keep only names present in the live set (pointer that uses
//        `pid.x` only keeps "SEQ_LEN", pointer that also adds `pid.z` keeps both)
//      * Result: two pointers from the same base but with different offsets now
//        carry *different* `axis_names`
// * For every op compute the output axis list from the inputs based on per-op rules.
// * Keep a scratch  AxisMap<Value, ArrayAttr>  (`tmp`) while walking
//   forward once; when exit write every entry back into the IR.
//
// Attribute layout (agreed across Python & C++)
//   function arg / result :  tt.axis_names              (ArrayAttr)
//   single‑result   op     :  tt.axis_names
//   multi‑result    op     :  tt.axis_names_<idx>       (one attr per result)
//
// Grid / loop config
//   tt.grid_axes = ["SEQ_LEN","B","HEAD"]            tag PIDs, and
//   tt.loop_axis = "SEQ_LEN"                         tag the (single) software loop IV
//
// ────────────────────────────────────────────────────────────────────────────

namespace mlir::triton {
namespace {

using AxisMap = llvm::DenseMap<Value, ArrayAttr>;          // scratch cache

static constexpr StringLiteral kAxisNames = "tt.axis_names";
static constexpr StringLiteral kAxisTag   = "tt.axis";     // on index ops
static constexpr StringLiteral kGridAttr  = "tt.grid_axes";

// ──────────────────────────────────────────────────────────────────────
//  Helpers   StringVec <-> ArrayAttr
// ──────────────────────────────────────────────────────────────────────

// build an ArrayAttr from a list of StringRefs
static ArrayAttr makeAttr(MLIRContext *ctx, ArrayRef<StringRef> v) {
  SmallVector<Attribute> out;
  out.reserve(v.size());
  for (StringRef s : v) out.push_back(StringAttr::get(ctx, s));
  return ArrayAttr::get(ctx, out);
}

// convert an ArrayAttr back to a vector of StringRefs
static SmallVector<StringRef> asVec(ArrayAttr a) {
  SmallVector<StringRef> v;
  if (a)
    for (Attribute t : a) v.push_back(cast<StringAttr>(t).getValue());
  return v;
}

// ──────────────────────────────────────────────────────────────────────────
//  Attribute I/O  -  uniformly handle function args/results and op results
// ──────────────────────────────────────────────────────────────────────────

// Key for multi‑result ops is  "tt.axis_names_<idx>"
static StringAttr perResultKey(MLIRContext *ctx, unsigned idx, bool multi) {
  return !multi ? StringAttr::get(ctx, kAxisNames)
                : StringAttr::get(ctx, ("tt.axis_names_" + std::to_string(idx)));
}

// read attribute attached to a Value (arg, result, or temp)
static ArrayAttr readAttr(Value v) {
  MLIRContext *ctx = v.getContext();

  // function argument
  if (auto arg = dyn_cast<BlockArgument>(v))
    if (auto fn = dyn_cast<FunctionOpInterface>(arg.getParentBlock()->getParentOp()))
      return fn.getArgAttrOfType<ArrayAttr>(arg.getArgNumber(), kAxisNames);

  // function result
  if (auto res = dyn_cast<OpResult>(v))
    if (auto fn = dyn_cast<FunctionOpInterface>(res.getOwner()))
      if (auto dict = fn.getResultAttrDict(res.getResultNumber()))
        return dyn_cast_or_null<ArrayAttr>(dict.get(kAxisNames));

  // regular op result
  if (auto res = dyn_cast<OpResult>(v)) {
    bool multi = res.getOwner()->getNumResults() > 1;
    return res.getOwner()->getAttrOfType<ArrayAttr>(
        perResultKey(ctx, res.getResultNumber(), multi));
  }
  return {};
}

// write attribute back to the Value
static void writeAttr(Value v, ArrayAttr a) {
  if (!a) return;
  MLIRContext *ctx = a.getContext();
  auto key         = StringAttr::get(ctx, kAxisNames);

  // function arg
  if (auto arg = dyn_cast<BlockArgument>(v))
    if (auto fn = dyn_cast<FunctionOpInterface>(arg.getParentBlock()->getParentOp())) {
      fn.setArgAttr(arg.getArgNumber(), key, a);
      return;
    }

  // function result
  if (auto res = dyn_cast<OpResult>(v))
    if (auto fn = dyn_cast<FunctionOpInterface>(res.getOwner())) {
      fn.setResultAttr(res.getResultNumber(), key, a);
      return;
    }

  // regular op result
  if (auto res = dyn_cast<OpResult>(v)) {
    bool multi = res.getOwner()->getNumResults() > 1;
    res.getOwner()->setAttr(perResultKey(ctx, res.getResultNumber(), multi), a);
  }
}

// helpers operating on AxisMap
static ArrayAttr get(Value v, AxisMap &m) {
  if (auto it = m.find(v); it != m.end()) return it->second;
  return readAttr(v);
}
static void set(Value v, ArrayAttr a, AxisMap &m) { if (a) m[v] = a; }


// ──────────────────────────────────────────────────────────────────────────
//  *Grid map* - translate pid.x/y/z -> logical axis names
// ──────────────────────────────────────────────────────────────────────────

using GridMap = llvm::StringMap<StringRef>;   // "x"->"SEQ_LEN", etc.

static GridMap buildGridMap(Operation *funcOp) {
  GridMap map;
  auto arr = funcOp->getAttrOfType<ArrayAttr>(kGridAttr);
  if (!arr || arr.size() == 0) return map;

  static const char *keys[3] = {"x", "y", "z"};
  for (unsigned i = 0, e = std::min<unsigned>(arr.size(), 3); i < e; ++i)
    if (auto s = dyn_cast<StringAttr>(arr[i]))
      if (!s.getValue().empty())
        map[keys[i]] = s.getValue();
  return map;
}

// ──────────────────────────────────────────────────────────────────────────
//  Dynamic‑axis utilities  (live‑axis discovery, filtering)
// ──────────────────────────────────────────────────────────────────────────

// collect axis names that appear on any dynamic operand
static llvm::SmallDenseSet<StringRef>
collectLiveAxes(mlir::ValueRange dynOps, AxisMap &m) {
  llvm::SmallDenseSet<StringRef> live;
  for (Value v : dynOps){
    // op result: read tag from producing op
    // dynamic operands that are Op results - have a definingOp(),
    // so can inspect the op itself for tt.axis = "SEQ_LEN"
    if (auto *def = v.getDefiningOp())
      if (auto s = def->getAttrOfType<StringAttr>(kAxisTag))
        live.insert(s.getValue());

    // region argument: tag stored only in AxisMap
    // dynamic operands that are region arguments (e.g. the loop IV %iv)
    // - do not have a defining op. The only place where we recorded
    // the tag ("SEQ_LEN") is the temporary map m that the pass uses
    // to track axis information per Value
    if (auto a = get(v, m))
      for (StringRef n : asVec(a)) live.insert(n);
  }
  return live;
}

// keep only names that are still "live" in this address expression.
static ArrayAttr
filterByLiveAxes(ArrayAttr in, const llvm::SmallDenseSet<StringRef> &live,
                 MLIRContext *ctx) {
  if (!in) return {};
  SmallVector<StringRef> out;
  for (StringRef name : asVec(in))
    if (live.contains(name)) out.push_back(name);
  return makeAttr(ctx, out);
}

// ──────────────────────────────────────────────────────────────────────────
//  Per‑op transfer rules
// ──────────────────────────────────────────────────────────────────────────

// 1. Element‑wise ops - copy axis_names verbatim from first operand
static const llvm::StringSet<> kElt = {
    // triton
    "tt.add", "tt.sub", "tt.mul", "tt.div", "tt.max", "tt.min",
    // arith
    "arith.addf", "arith.subf", "arith.mulf", "arith.divf",
    "arith.addi", "arith.subi", "arith.muli",
    "arith.truncf", "arith.extf", "arith.maxnumf",
    // math
    "math.cos", "math.sin", "math.sqrt",
    "math.log", "math.log2", "math.exp", "math.exp2"};

static bool handleElt(Operation *op, AxisMap &m) {
  if (!kElt.contains(op->getName().getStringRef()) || op->getNumOperands() == 0)
    return false;
  if (auto a = get(op->getOperand(0), m))
    for (Value r : op->getResults()) set(r, a, m);
  return true;
}

// 2. tt.dot - drop the contracting K dim:  (...m k) x (k n...) -> (...m n...)
static bool handleDot(Operation *op, AxisMap &m) {
  if (op->getName().getStringRef() != "tt.dot") return false;

  auto lhs = get(op->getOperand(0), m);
  auto rhs = get(op->getOperand(1), m);
  if (!lhs || !rhs) return false;

  auto L = asVec(lhs), R = asVec(rhs);
  if (L.empty() || R.empty()) return false;

  SmallVector<StringRef> out;
  out.append(L.begin(), L.end() - 1);     // drop trailing K of lhs
  out.append(R.begin() + 1, R.end());     // drop leading  K of rhs
  set(op->getResult(0), makeAttr(op->getContext(), out), m);
  return true;
}

// 3. Transpose (and legacy alias tt.trans) - permute axis names
static bool handleTranspose(Operation *op, AxisMap &m) {
  StringRef name = op->getName().getStringRef();
  if (name != "tt.transpose" && name != "tt.trans") return false;
  auto a = get(op->getOperand(0), m);
  if (!a) return false;

  auto perm = op->getAttrOfType<DenseIntElementsAttr>("perm");
  if (!perm) perm = op->getAttrOfType<DenseIntElementsAttr>("order");
  if (!perm) return false;

  auto v = asVec(a);
  SmallVector<StringRef> dst(v.size());
  unsigned i = 0;
  for (APInt p : perm) dst[i++] = v[p.getZExtValue()];
  set(op->getResult(0), makeAttr(op->getContext(), dst), m);
  return true;
}

// 4. Reduce - erase every axis listed in `axes`
static bool handleReduce(Operation *op, AxisMap &m) {
  if (op->getName().getStringRef() != "tt.reduce") return false;
  auto a = get(op->getOperand(0), m);
  if (!a) return false;

  auto dims = op->getAttrOfType<DenseIntElementsAttr>("axes");
  if (!dims) dims = op->getAttrOfType<DenseIntElementsAttr>("dim");
  if (!dims) return false;

  llvm::SmallBitVector drop(a.size());
  for (APInt d : dims) drop.set(d.getZExtValue());

  SmallVector<StringRef> keep;
  for (auto [idx, s] : llvm::enumerate(asVec(a)))
    if (!drop.test(idx)) keep.push_back(s);

  set(op->getResult(0), makeAttr(op->getContext(), keep), m);
  return true;
}

// 5. Broadcast - pad on the left with fresh placeholders
static bool handleBcast(Operation *op, AxisMap &m) {
  if (op->getName().getStringRef() != "tt.broadcast") return false;
  auto a  = get(op->getOperand(0), m);
  auto rt = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  auto st = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!a || !rt || !st) return false;

  unsigned extra = rt.getRank() - st.getRank();
  SmallVector<StringRef> names;
  for (unsigned i = 0; i < extra; ++i) names.push_back("__b" + std::to_string(i));
  names.append(asVec(a));

  set(op->getResult(0), makeAttr(op->getContext(), names), m);
  return true;
}

// 6. Reshape / expand / collapse - layout‑preserving, so copy
static bool handleReshape(Operation *op, AxisMap &m) {
  static const llvm::StringSet<> kR = {"tensor.expand_shape",
                                       "tensor.collapse_shape",
                                       "tensor.reshape"};
  if (!kR.contains(op->getName().getStringRef())) return false;
  if (auto a = get(op->getOperand(0), m))
    for (Value r : op->getResults()) set(r, a, m);
  return true;
}

// 7. scf.for - propagate iter‑args *and* tag the induction variable
static bool handleLoop(Operation *op, AxisMap &m) {

  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    // (a) loop‑carried tensors: copy axis_names init -> iter‑arg
    for (auto [arg, init] : llvm::zip(forOp.getRegionIterArgs(),
                                      forOp.getInitArgs()))
      set(arg, get(init, m), m);

    // (b) give the induction var the function‑level tt.loop_axis, if any

    // bc set(iv, …) writes directly into the AxisMap, every arithmetic
    // value derived from the IV now shows the "SEQ_LEN" tag to
    // collectLiveAxes, and "SEQ_LEN" survives inside the loop body

    // attach tt.axis to the induction var if user supplied
    if (auto func = op->getParentOfType<FunctionOpInterface>()) {
      if (auto s = func->getAttrOfType<StringAttr>("tt.loop_axis")) {
        auto iv = forOp.getInductionVar();                     // region arg
        set(iv, makeAttr(op->getContext(), {s.getValue()}), m);// tag IV
      }
    }
    return true;   // body will be walked automatically
  }

  // scf.yield: copy iter‑arg axis_names back to loop result
  if (auto y = dyn_cast<scf::YieldOp>(op))
    if (auto parent = dyn_cast<scf::ForOp>(y->getParentOp())) {
      for (auto [res, val] : llvm::zip(parent.getResults(), y.getOperands()))
        set(res, get(val, m), m);
      return true;
    }
  return false;
}

// helpers
static bool cloneAxes(Value src, Value dst, AxisMap &m) {
  if (auto a = get(src, m)) { set(dst, a, m); return true; }
  return false;
}

// tag every tt.get_program_id  - attach  tt.axis = "<logical name>"
static bool handleProgramId(Operation *op, const GridMap &grid) {
  if (op->getName().getStringRef() != "tt.get_program_id") return false;

  StringRef dim;
  if (auto a = op->getAttrOfType<StringAttr>("axis")) dim = a.getValue();
  if (dim.empty()) return false;
  if (auto it = grid.find(dim); it != grid.end()) {
    op->setAttr(kAxisTag, StringAttr::get(op->getContext(), it->second));
  }
  return false;   // continue dispatch
}

//  tt.addptr  - keep only axes that remain "live" after applying offsets
static bool handleAddPtr(Operation *op, AxisMap &m) {
  if (op->getName().getStringRef() != "tt.addptr") return false;

  auto inAxes = get(op->getOperand(0), m);
  if (!inAxes) return false;

  auto live = collectLiveAxes(op->getOperands().drop_front(), m);
  if (live.empty()) {                         // no info -> keep all
    set(op->getResult(0), inAxes, m);
    return true;
  }
  auto out = filterByLiveAxes(inAxes, live, op->getContext());
  set(op->getResult(0), out, m);
  return true;
}

//  tt.load  - result rank  R  -> keep last R names, pad left if needed
static bool handleLoad(Operation *op, AxisMap &m) {
  if (op->getName().getStringRef() != "tt.load") return false;

  auto in  = get(op->getOperand(0), m);
  auto rt  = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!in || !rt) return false;

  auto v = asVec(in);
  unsigned R = rt.getRank();
  if (v.size() > R) v.erase(v.begin(), v.end() - R);              // truncate
  else if (v.size() < R) {                                       // pad left
    unsigned extra = R - v.size();
    static unsigned nextTmp = 0;
    for (unsigned i = 0; i < extra; ++i)
      v.insert(v.begin(), "__b" + std::to_string(nextTmp++));
  }

  set(op->getResult(0), makeAttr(op->getContext(), v), m);
  return true;
}

//  tt.splat - copy trailing names, pad on left until ranks match
static bool handleSplat(Operation *op, AxisMap &m) {
  if (op->getName().getStringRef() != "tt.splat") return false;

  auto a  = get(op->getOperand(0), m);      // copy from scalar/ptr
  auto rt = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!a || !rt) return false;

  auto v  = asVec(a);
  while (v.size() < rt.getRank())           // pad on the left
    v.insert(v.begin(), "__b" + std::to_string(v.size()));

  set(op->getResult(0), makeAttr(op->getContext(), v), m);
  return true;
}

//  arith.select - take axes from the true‑value operand
static bool handleSelect(Operation *op, AxisMap &m) {
  if (op->getName().getStringRef() != "arith.select") return false;
  return cloneAxes(op->getOperand(1), op->getResult(0), m);
}

//  tt.expand_dims - insert length‑1 dimension with fresh placeholder name
static bool handleExpandDims(Operation *op, AxisMap &m) {
  if (op->getName().getStringRef() != "tt.expand_dims") return false;
  auto a = get(op->getOperand(0), m);
  auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
  if (!a || !axisAttr) return false;

  auto v = asVec(a);
  v.insert(v.begin() + axisAttr.getInt(), "__b" + std::to_string(axisAttr.getInt()));
  set(op->getResult(0), makeAttr(op->getContext(), v), m);
  return true;
}

//  tt.make_range - generate fresh placeholders  ["__bN", ...]
static bool handleMakeRange(Operation *op, AxisMap &m) {
  if (op->getName().getStringRef() != "tt.make_range") return false;
  auto rt = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!rt) return false;

  SmallVector<StringRef> names;
  static unsigned nextTmp = 0;
  for (unsigned i = 0; i < rt.getRank(); ++i)
    names.push_back("__b" + std::to_string(nextTmp++));
  set(op->getResult(0), makeAttr(op->getContext(), names), m);
  return true;
}

// ──────────────────────────────────────────────────────────────────────────
//  Dispatcher - try specific handlers first, then generic element‑wise
// ──────────────────────────────────────────────────────────────────────────
static void propagate(Operation *op, AxisMap &m, const GridMap &grid) {
  if (handleProgramId(op, grid)) return;
  if (handleDot(op, m))       return;
  if (handleTranspose(op, m)) return;
  if (handleReduce(op, m))    return;
  if (handleBcast(op, m))     return;
  if (handleMakeRange(op, m)) return;
  if (handleReshape(op, m))   return;
  if (handleLoop(op, m))      return;

  if (handleAddPtr(op, m))    return;
  if (handleLoad(op, m))      return;
  if (handleSplat(op, m))     return;
  if (handleExpandDims(op, m))return;
  if (handleSelect(op, m))    return;

  handleElt(op, m);           // catch‑all element‑wise
}

// ──────────────────────────────────────────────────────────────────────────
//  Public API - called once per kernel
// ──────────────────────────────────────────────────────────────────────────
ArrayAttr getAxisAttr(Value v) { return readAttr(v); }

} // anonymous namespace

// Walk a whole func.func once, filling in every missing tt.axis_names
void propagateAxesInFuncOp(Block *entry) {
  if (!entry) return;

  Operation *top = entry->getParentOp();
  AxisMap tmp;                              // value -> axis_names (scratch)

  GridMap grid = buildGridMap(top);         // pid.x/y/z -> logical name

  // seed scratch map with labels already on entry arguments
  for (Value arg : entry->getArguments())
    if (auto a = readAttr(arg)) tmp[arg] = a;

  // single forward walk
  top->walk([&](Operation *op) { propagate(op, tmp, grid); });

  // persist inferences back to IR
  for (auto &[v, a] : tmp) writeAttr(v, a);
}

} // triton namespace
