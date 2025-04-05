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

// for loop unroll
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"

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

  void maybeAccumulateGrad(Value val, Value grad,
                          llvm::DenseMap<Value, Value> &gradMap,
                          OpBuilder &builder) {
    auto it = gradMap.find(val);

    // no existing grad
    if (it == gradMap.end()) {
      gradMap[val] = grad;

    // found existing grad wrt val
    } else {
      auto existingGrad = it->second;

      // otherwise "does not dominate its use"  err
      builder.setInsertionPointAfterValue(existingGrad);

      // // todo: don't use atomics unless needed (requires some analysis pass
      // // to identify if this accesses the same memory location)
      // auto accumulated_grad = create_atomic_add(existingGrad, grad);

      // keys in my grad map already belong to the re-written part of
      // the bwd graph so don't need map them though origToCloned
      auto accumulatedGrad = builder.create<arith::AddFOp>(existingGrad.getLoc(), existingGrad, grad);
      markVisited(builder, visitedType::Inserted, accumulatedGrad);

      gradMap[val] = accumulatedGrad;
    }
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




  // Helper to get constant integer value if possible
  static std::optional<int64_t> getConstantIntValue(Value value) {
    if (!value)
      return std::nullopt;

    if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
        return intAttr.getInt();
    }
    return std::nullopt;
  }

  void unrollAllForOps(triton::FuncOp func){

    // Find and unroll any for loops in the function body
    SmallVector<scf::ForOp> forOpsToUnroll;
    func.walk([&](scf::ForOp forOp) {
      forOpsToUnroll.push_back(forOp);
    });

    for (auto forOp : forOpsToUnroll) {
      // Check if we can get the upper bound as a constant
      if (auto upperBound = getConstantIntValue(forOp.getUpperBound())) {
        unsigned numIters = *upperBound;
        // Completely unroll
        auto resultLoops = loopUnrollByFactor(forOp, numIters);
        if (DEBUG_PRINTS) llvm::outs() << "Unrolled loop with " << numIters << " iterations\n";
      } else {
        llvm::outs() << "[unrollAllForOps] failed to extract upper bound\n";
        exit(1);
      }
    }

  }


} // namespace triton
} // namespace mlir