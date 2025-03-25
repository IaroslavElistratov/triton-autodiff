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



  Operation* cloneSubtree(Operation *targetOp, IRMapping &mapper, OpBuilder &builder) {

    // Null check
    if (!targetOp)
      return nullptr;

    // wt this conditional, same node (e.g. make_range) is duplicated multiple times
    if (auto *clonedOp = mapper.lookupOrNull(targetOp))
      return clonedOp;

    // Clone all operations we depend on first
    for (Value operand : targetOp->getOperands()) {

      // Skip block arguments and values already in the mapper
      if (isa<BlockArgument>(operand))
        continue;

      if (Operation *defOp = operand.getDefiningOp())
        cloneSubtree(defOp, mapper, builder);
    }

    /* currently the order of cloned ops is different from the original
      you can see how cloning left children first in original graph
      below results in this order I'm seeing. The first operand of
      the store is the pointer (%18) -- so the for loop over operands
      visits and clones that pointer arg (%18) first.

    %18 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %19 = tt.addptr %18, %10 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %19, %17 : tensor<4x!tt.ptr<f32>>
    */

    // todo: set the insertion point so that the order of inserted ops doesn't change?
    // Save the current insertion point
    // OpBuilder::InsertionGuard insertGuard(builder);
    // builder.setInsertionPoint(clonedOperand);


    // Now clone this operation
    if (DEBUG_PRINTS) llvm::outs() << "cloning: " << *targetOp << "\n";
    // passing the mapper, maps the results of the original operation
    // to the results of the cloned operation in the IRMapping;
    // IRMapping used to ensure when I clone operations, the operands of the cloned
    // ops refer to the cloned values, not the original ones.
    Operation *clonedOp = builder.clone(*targetOp, mapper);
    markVisited(builder, visitedType::Cloned, clonedOp);

    // if (clonedOp) {
    //   // Find the corresponding result in the cloned operation
    //   unsigned resultIdx = targetValue.cast<OpResult>().getResultNumber();
    //   if (resultIdx < clonedOp->getNumResults())
    //     return clonedOp->getResult(resultIdx);
    // }

    // e.g. tt.store, does not have result
    // Cloned the operation (above) but don't try to return its (non-existent) result
    return clonedOp;
  }


} // namespace triton
} // namespace mlir