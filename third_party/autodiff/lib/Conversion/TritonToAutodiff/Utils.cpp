#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

// for reverse topo sort
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SetVector.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "autodiff/include/Dialect/Autodiff/IR/Dialect.h"
#include "autodiff/include/Conversion/TritonToAutodiff/Utils.h"
#include "autodiff/include/Conversion/TritonToAutodiff/UtilsIO.h"
#include "llvm/ADT/APSInt.h"
#include <numeric>

#include "llvm/Support/Debug.h"

namespace mlir {
namespace triton {

  void markVisited(OpBuilder &builder, visitedType mode, Operation *op) {
    if (!op) {
      llvm::outs() << "markVisited received null operation pointer\n";
      exit(1);
    }

    NamedAttrList attrs;
    attrs.append("autogradVisited", builder.getBoolAttr(true));
    if (mode == visitedType::Original)
      attrs.append("isOrig", builder.getBoolAttr(true));
    else if (mode == visitedType::Inserted)
      attrs.append("isInserted", builder.getBoolAttr(true));
    else if (mode == visitedType::Cloned)
      attrs.append("isCloned", builder.getBoolAttr(true));
    else {
      llvm::outs() << "markVisited invalid visitedType value\n";
      exit(1);
    }
    op->setAttrs(attrs);
  }

  Value getUpstreamGrad(Value result, const llvm::DenseMap<Value, Value> &gradMap) {
    auto it = gradMap.find(result);
    if (it == gradMap.end()) {
      llvm::outs() << "Expected gradient in the map\n";
      exit(1);
    }
    return it->second;
  }

  triton::SplatOp createConstantTensor(OpBuilder &builder, Location loc, Type tensorType, float value) {
    auto scalarType = builder.getF32Type();
    auto scalarValue = builder.create<arith::ConstantOp>(loc, scalarType, builder.getF32FloatAttr(value));
    auto tensorValue = builder.create<triton::SplatOp>(loc, tensorType, scalarValue);
    markVisited(builder, visitedType::Inserted, scalarValue);
    markVisited(builder, visitedType::Inserted, tensorValue);
    return tensorValue;
  }

  Value cloneSubtree(Operation *targetOp, IRMapping &mapper, OpBuilder &builder) {
    // Null check
    if (!targetOp || targetOp->getNumResults() == 0) {
      llvm::report_fatal_error("Cannot clone operation with no results");
      exit(1);
    }

    // If we've already cloned this result, return it directly
    if (mapper.contains(targetOp->getResult(0)))
      return mapper.lookup(targetOp->getResult(0));

    // Clone all operations we depend on first
    for (Value operand : targetOp->getOperands()) {
      // Skip block arguments and values already in the mapper
      if (mlir::isa<BlockArgument>(operand) || mapper.contains(operand))
        continue;

      if (Operation *defOp = operand.getDefiningOp())
        cloneSubtree(defOp, mapper, builder);
    }

    // Now clone this operation
    Operation *clonedOp = builder.clone(*targetOp, mapper);
    markVisited(builder, visitedType::Cloned, clonedOp);

    // IRMapping used to ensure when you clone operations, the operands of the cloned
    // ops refer to the cloned values, not the original ones.
    // After cloning, manually map the original results to the clone's results:
    // this extracts from the map whatever Value in the map (the copied subgraph)
    // is equivalent to targetOp->getResult(0)

    // returns a value to avoid manually looking up the clone in the mapper after calling the function
    return clonedOp->getResult(0);
  }

} // namespace triton
} // namespace mlir