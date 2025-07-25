#include <cassert>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Block.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h" // registers "tt" dialect
#include "mlir/IR/Visitors.h"                 // for Block::walk

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"               // zip_equal

using namespace mlir;

namespace {


static constexpr StringLiteral kReuseAttrName = "tt.reuse_counts";
static constexpr int64_t       kUnknown       = -1;

// helpers (shape queries, attr builders, etc.)

// Return static shape or a vector filled with `kUnknown`.
static SmallVector<int64_t> getShapeOrUnknown(Value v) {
  auto st = dyn_cast<ShapedType>(v.getType());
  if (!st) return {};                                // scalars -> rank‑0
  if (st.hasStaticShape()) return llvm::to_vector(st.getShape());
  return SmallVector<int64_t>(st.getRank(), kUnknown);
}

// Wrap raw integers into `ArrayAttr<i64>` so IR prints nicely.
static ArrayAttr makeI64ArrayAttr(OpBuilder &b, ArrayRef<int64_t> vec) {
  SmallVector<Attribute> tmp;
  tmp.reserve(vec.size());
  for (int64_t x : vec) tmp.push_back(b.getI64IntegerAttr(x));
  return b.getArrayAttr(tmp);
}

// Attach `countsAttr` to the *producer* of `v` (OpResult or BlockArgument).
static void setReuseCounts(Value v, ArrayAttr countsAttr) {
  if (auto res = dyn_cast<OpResult>(v)) {
    if (res.getOwner()->hasAttr(kReuseAttrName)) return; // keep first write
    res.getOwner()->setAttr(kReuseAttrName, countsAttr);
    return;
  }
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    Operation *funcOp   = arg.getOwner()->getParentOp();
    ArrayAttr  existing = funcOp->getAttrOfType<ArrayAttr>(kReuseAttrName);

    SmallVector<Attribute> perArg =
        existing ? llvm::to_vector(existing.getValue()) : SmallVector<Attribute>{};

    perArg.resize(std::max<size_t>(perArg.size(), arg.getArgNumber() + 1));
    perArg[arg.getArgNumber()] = countsAttr;

    OpBuilder b(funcOp->getContext());
    funcOp->setAttr(kReuseAttrName, b.getArrayAttr(perArg));
  }
}

// Reuse‑rule primitives

// Element‑wise rule: each axis used **exactly once**.
static ArrayAttr elementwiseRule(Value operand, OpBuilder &b) {
  size_t rank = getShapeOrUnknown(operand).size();
  return makeI64ArrayAttr(b, SmallVector<int64_t>(rank, 1));
}

// Broadcast‑aware rule (align ranks on the **right**).
static ArrayAttr broadcastRule(Operation *op, Value operand, OpBuilder &b) {
  auto outShape = getShapeOrUnknown(op->getResult(0));
  auto inShape  = getShapeOrUnknown(operand);

  unsigned rank = std::max(outShape.size(), inShape.size());
  while (outShape.size() < rank) outShape.insert(outShape.begin(), 1);
  while (inShape .size() < rank) inShape .insert(inShape .begin(), 1);

  SmallVector<int64_t> counts;
  counts.reserve(rank);

  for (auto [o, i] : llvm::zip_equal(outShape, inShape)) {
    if (o == kUnknown || i == kUnknown)      counts.push_back(kUnknown);
    else if (o == i)                         counts.push_back(1);
    else if (i == 1)                         counts.push_back(o); // broadcast
    else                                     counts.push_back(kUnknown);
  }
  return makeI64ArrayAttr(b, counts);
}

// Dot / matmul rule: A(M,K) @ B(K,N)=C(M,N).
// Guard rank assumption with an `assert`
static ArrayAttr dotRule(Operation *dotOp, unsigned operandIdx, OpBuilder &b) {
  auto lhsShape = getShapeOrUnknown(dotOp->getOperand(0)); // A
  auto rhsShape = getShapeOrUnknown(dotOp->getOperand(1)); // B
  assert(lhsShape.size() == 2 && rhsShape.size() == 2 &&
         "dotRule expects rank‑2 operands (no batching support yet)");

  int64_t M = lhsShape[0];
  int64_t N = rhsShape[1];

  SmallVector<int64_t> reuse(2, 1);
  if (operandIdx == 0)          // A : (M,K)
    reuse[1] = (N == kUnknown ? kUnknown : N); // K‑axis reused N×
  else                          // B : (K,N)
    reuse[0] = (M == kUnknown ? kUnknown : M); // K‑axis reused M×

  return makeI64ArrayAttr(b, reuse);
}

// Reduction rule: each reduced axis reused `dim` times.
static ArrayAttr reduceRule(Operation *redOp, Value operand, OpBuilder &b) {
  auto shape = getShapeOrUnknown(operand);

  llvm::DenseSet<int64_t> reduced;
  if (auto ax = redOp->getAttrOfType<DenseIntElementsAttr>("axis"))
    for (auto i : ax) reduced.insert(i.getSExtValue());
  if (auto ax = redOp->getAttrOfType<DenseIntElementsAttr>("axes"))
    for (auto i : ax) reduced.insert(i.getSExtValue());
  if (auto ax = redOp->getAttrOfType<DenseIntElementsAttr>("dim"))
    for (auto i : ax) reduced.insert(i.getSExtValue());

  SmallVector<int64_t> counts;
  counts.reserve(shape.size());
  for (auto [d, dim] : llvm::enumerate(shape)) {
    if (dim == kUnknown)             counts.push_back(kUnknown);
    else if (reduced.contains(d))    counts.push_back(dim);
    else                             counts.push_back(1);
  }
  return makeI64ArrayAttr(b, counts);
}

// Memory rule – pointer is a scalar (*rank‑0*).
static ArrayAttr memRule(OpBuilder &b) { return makeI64ArrayAttr(b, {}); }

// Helper: build a vector `<rank × kUnknown>`.
static ArrayAttr makeUnknown(OpBuilder &b, unsigned rank) {
  return makeI64ArrayAttr(b, SmallVector<int64_t>(rank, kUnknown));
}

// Per‑op handler helpers

// Broadcast op (`tt.broadcast`).
static void handleBroadcastOp(Operation *op, OpBuilder &b) {
  for (Value v : op->getOperands())
    setReuseCounts(v, broadcastRule(op, v, b));
}

// View‑only ops (transpose, reshape, …).
static void handleViewOp(Operation *op, OpBuilder &b) {
  for (Value v : op->getOperands())
    setReuseCounts(v, elementwiseRule(v, b));
}

// Reduction (`tt.reduce`).
static void handleReductionOp(Operation *op, OpBuilder &b) {
  for (Value v : op->getOperands())
    setReuseCounts(v, reduceRule(op, v, b));
}

// Dot / matmul (`tt.dot`).
static void handleDotOp(Operation *op, OpBuilder &b) {
  // First two operands are the matrices.
  for (auto [idx, v] : llvm::enumerate(op->getOperands().take_front(2)))
    setReuseCounts(v, dotRule(op, idx, b));

  // Remaining operands (e.g., bias) are element‑wise.
  for (Value extra : op->getOperands().drop_front(2))
    setReuseCounts(extra, elementwiseRule(extra, b));
}

// Load / Store – pointer operand only.
static void handleLoadStoreOp(Operation *op, OpBuilder &b) {
  setReuseCounts(op->getOperand(0), memRule(b));
}

// Loop‑carried operands (`scf.for`).
static void handleLoopOp(scf::ForOp loop, OpBuilder &b) {
  // Attempt to compute a **static** trip count; else mark unknown.
  auto lb   = loop.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto ub   = loop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto step = loop.getStep().getDefiningOp<arith::ConstantIndexOp>();
  int64_t tripCount =
      (lb && ub && step && step.value() == 1) ? ub.value() - lb.value()
                                              : kUnknown;

  auto *yieldTerm = loop.getBody()->getTerminator();

  for (auto [iterArg, yielded] :
       llvm::zip_equal(loop.getRegionIterArgs(), yieldTerm->getOperands())) {

    ArrayAttr perIter =
        yielded.getDefiningOp()
            ? yielded.getDefiningOp()->getAttrOfType<ArrayAttr>(kReuseAttrName)
            : ArrayAttr();

    if (perIter && tripCount != kUnknown) {
      SmallVector<int64_t> cumulative;
      cumulative.reserve(perIter.size());
      for (IntegerAttr v : perIter.getAsRange<IntegerAttr>()) {
        int64_t x = v.getInt();
        cumulative.push_back(x == kUnknown ? kUnknown : x * tripCount);
      }
      setReuseCounts(iterArg, makeI64ArrayAttr(b, cumulative));
    } else {
      // If the carried value is not a ranked tensor (e.g. pointer) -> rank = 0.
      unsigned rank = 0;
      if (auto shaped = mlir::dyn_cast<ShapedType>(iterArg.getType()))
        rank = shaped.getRank();
      setReuseCounts(iterArg, makeUnknown(b, rank));
    }
  }
}

// Generic element‑wise op (has `Elementwise` trait).
static void handleGenericElementwiseOp(Operation *op, OpBuilder &b) {
  for (Value v : op->getOperands())
    setReuseCounts(v, broadcastRule(op, v, b));
}

// Fallback – give “all unknown”.
static void handleFallbackOp(Operation *op, OpBuilder &b) {
  for (Value v : op->getOperands()) {
    unsigned rank = getShapeOrUnknown(v).size();
    setReuseCounts(v, makeUnknown(b, rank));
  }
}

} // anonymous namespace

//  public
namespace mlir {
namespace triton {

void propagateReuseCounts(Block *block) {
  if (!block)
    return;

  // Use the context attached to this block.
  OpBuilder b(block->getParentOp()->getContext());

  // Helper lambda that applies the reuse rules to a given operation.
  auto processOp = [&](Operation *op) {
    if (op->getNumOperands() == 0 || op->getNumResults() == 0)
      return; // nothing to analyse

    StringRef name = op->getName().getStringRef();

    // 1. Broadcast
    if (name == "tt.broadcast") {
      handleBroadcastOp(op, b);
    }

    // 2. View‑only ops
    else if (name == "tt.transpose"        || name == "tt.trans" ||
             name == "tt.expand_dims"      || name == "tensor.reshape" ||
             name == "tensor.expand_shape" || name == "tensor.collapse_shape") {
      handleViewOp(op, b);
    }

    // 3. Reduction
    else if (name == "tt.reduce") {
      handleReductionOp(op, b);
    }

    // 4. Dot / matmul
    else if (name == "tt.dot") {
      handleDotOp(op, b);
    }

    // 5. Load / Store
    else if (name == "tt.load" || name == "tt.store") {
      handleLoadStoreOp(op, b);
    }

    // 6. Loop (`scf.for`)
    else if (auto loop = dyn_cast<scf::ForOp>(op)) {
      handleLoopOp(loop, b);
    }

    // 7. Generic element‑wise (trait)
    else if (op->hasTrait<OpTrait::Elementwise>()) {
      handleGenericElementwiseOp(op, b);
    }

    // 8. Fallback
    else {
      handleFallbackOp(op, b);
    }
  };

  // walk all operations nested inside the provided block (including the
  // operations in any nested regions)
  for (Operation &topOp : *block) {
    topOp.walk(processOp);
  }
}

} // namespace triton
} // namespace mlir
