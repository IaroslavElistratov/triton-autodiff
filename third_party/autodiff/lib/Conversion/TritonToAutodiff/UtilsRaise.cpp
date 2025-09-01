#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Builders.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "autodiff/include/Conversion/TritonToAutodiff/Utils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

namespace mlir {
namespace triton {

  // Pointer-only backtrace: choose only pointer-typed BlockArguments as bases.
  static Value _findBasePtr(Value anyPtr) {
    if (!anyPtr)
      return Value();
    if (auto ba = dyn_cast<BlockArgument>(anyPtr))
      if (isa<triton::PointerType>(ba.getType()))
        return ba;
    SmallVector<Value, 16> wl{anyPtr};
    DenseSet<Value> seen;
    auto isPtrLike = [](Type t) {
      if (auto rt = dyn_cast<RankedTensorType>(t))
        return isa<triton::PointerType>(rt.getElementType());
      return isa<triton::PointerType>(t);
    };
    while (!wl.empty()) {
      Value cur = wl.pop_back_val();
      if (!seen.insert(cur).second) continue;
      if (auto ba = dyn_cast<BlockArgument>(cur)) {
        if (isPtrLike(ba.getType())) return ba;
        continue;
      }
      if (Operation *def = cur.getDefiningOp())
        for (Value opnd : def->getOperands())
          if (isPtrLike(opnd.getType())) wl.push_back(opnd);
    }
    return Value();
  }

  // From a pointer SSA value, produce both a readable label and the canonical index
  // of the kernel BlockArgument that is the base pointer.
  std::pair<StringAttr, IntegerAttr> labelFromPtr(OpBuilder &builder, Value anyPtr) {
    StringAttr ofAttr = builder.getStringAttr("arg");
    IntegerAttr idxAttr;
    Value base = _findBasePtr(anyPtr);
    if (auto ba = dyn_cast_or_null<BlockArgument>(base)) {
      int64_t idx = ba.getArgNumber();
      idxAttr = builder.getI64IntegerAttr(idx);
      if (auto nl = dyn_cast<NameLoc>(ba.getLoc())) {
        ofAttr = builder.getStringAttr(nl.getName().getValue());
      } else {
        std::string s = (Twine("arg") + Twine(idx)).str();
        ofAttr = builder.getStringAttr(s);
      }
    }
    return {ofAttr, idxAttr};
  }

  // Upstream propagation (Solution C): fill in missing labels on autodiff ops
  static bool isAutodiffOp(Operation *op) {
    if (!op) return false;
    auto ins = op->getAttrOfType<BoolAttr>("isInserted");
    auto reb = op->getAttrOfType<BoolAttr>("isGradPtrRebase");
    return (ins && ins.getValue()) || (reb && reb.getValue());
  }

  // Accumulate into raise.gradIdxs (array of i64) without duplicates
  static void addIdx(Operation *op, IntegerAttr idx, OpBuilder &b) {
    if (!op || !idx) return;
    SmallVector<Attribute, 8> vals;
    if (auto arr = op->getAttrOfType<ArrayAttr>("raise.gradIdxs"))
      vals.append(arr.begin(), arr.end());
    int64_t want = idx.getInt();
    bool present = llvm::any_of(vals, [&](Attribute a){
      if (auto intAttr = llvm::dyn_cast<IntegerAttr>(a))
        return intAttr.getInt() == want;
      return false;
    });
    if (!present) vals.push_back(idx);
    op->setAttr("raise.gradIdxs", b.getArrayAttr(vals));
  }

  void propagateIdxFromSink(Operation *sink, IntegerAttr idx, OpBuilder &b) {
    if (!sink || !idx) return;
    SmallVector<Operation*, 64> wl{sink};
    DenseSet<Operation*> seen;
    while (!wl.empty()) {
      Operation *cur = wl.pop_back_val();
      if (!seen.insert(cur).second) continue;
      if (isAutodiffOp(cur)) addIdx(cur, idx, b);
      // Walk to producers over autodiff-inserted graph only
      for (Value v : cur->getOperands()) {
        if (Operation *def = v.getDefiningOp()) {
          if (isAutodiffOp(def)) wl.push_back(def);
        }
      }
    }
  }

} // namespace triton
} // namespace mlir


