#ifndef TRITON_CONVERSION_TRITON_TO_AUTODIFF_PASSES_H
#define TRITON_CONVERSION_TRITON_TO_AUTODIFF_PASSES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/DenseMap.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "autodiff/include/Dialect/Autodiff/IR/Dialect.h"
#include "autodiff/include/Conversion/TritonToAutodiff/Utils.h"


namespace mlir {
namespace triton {

struct ConvertTritonToAutodiff
    : public impl::ConvertTritonToAutodiffBase<ConvertTritonToAutodiff> {

  using ConvertTritonToAutodiffBase::ConvertTritonToAutodiffBase;

  // instead of piping this variable through (which would require changing signature of all the functions) -- instead I set it as a global, so that all handlers can access it
  // To make nodeName accessible across all handler functions without changing their signatures, add it as a member variable to the ConvertTritonToAutodiff struct
  // Member variable to store current node name
  NameLoc currentNodeName;

  // Member variables for the pass state
  llvm::DenseMap<Value, Value> gradMap;
  llvm::DenseMap<Value, Value> ptrToAddedPtrMap;
  IRMapping origToCloned;
  std::optional<OpBuilder> builder;

  // Helper method to create an operation with the current node name
  template <typename OpTy, typename... Args>
  OpTy createGradOp(OpBuilder &builder, Args &&...args) {
    return builder.create<OpTy>(currentNodeName, std::forward<Args>(args)...);
  }

  // main function - declaration only, implementation in .cpp file
  void runOnOperation() override;

private:
  void enableNameLocSSA();
  void rewriteSplatAddOp(triton::FuncOp func);
};

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITON_TO_AUTODIFF_PASSES_H 