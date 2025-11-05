#ifndef TRITON_CONVERSION_TRITON_TO_AUTODIFF_UTILS_H
#define TRITON_CONVERSION_TRITON_TO_AUTODIFF_UTILS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Block.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

namespace mlir {
namespace triton {

  // Helper for compile-time errors with template-dependent context
  template <typename T>
  struct dependent_false : std::false_type {};

  // Check for getOperation method
  template <typename T, typename = void>
  struct has_getOperation : std::false_type {};

  template <typename T>
  struct has_getOperation<T, std::void_t<decltype(std::declval<T>().getOperation())>> : std::true_type {};

  // Check for getDefiningOp method
  template <typename T, typename = void>
  struct has_getDefiningOp : std::false_type {};

  template <typename T>
  struct has_getDefiningOp<T, std::void_t<decltype(std::declval<T>().getDefiningOp())>> : std::true_type {};

  // Get Operation* from any relevant type
  template <typename T>
  Operation* getOperationPtr(T& op) {
    if constexpr (std::is_pointer_v<T>)
      return op;  // Already Operation*
    else if constexpr (has_getOperation<T>::value)
      return op.getOperation();  // Operation-specific class
    else if constexpr (has_getDefiningOp<T>::value)
      return op.getDefiningOp();  // Value type
    else
      // compile time check
      // use dependent_false<T> (instead of just false) bc in templates, static_assert with a direct false
      // would always fail, even for valid cases. Making it dependent on T delays evaluation until the
      // specific template is instantiated
      static_assert(dependent_false<T>::value,
        "Type must be Operation*, have getOperation(), or have getDefiningOp()");
  }

  enum visitedType { Original = 0, Inserted = 1, Cloned = 2, GradPtrRebase = 3 };

  void markVisited(OpBuilder &builder, visitedType mode, Operation *op);

  template <typename... Ops>
  void markAllVisited(OpBuilder &builder, visitedType mode, Ops... ops) {
    (markVisited(builder, mode, getOperationPtr(ops)), ...);
  }

  Value getUpstreamGrad(Value result, const llvm::DenseMap<Value, Value> &gradMap);

  void maybeAccumulateGrad(Value val, Value grad, llvm::DenseMap<Value, Value> &gradMap, OpBuilder &builder);

  Value createConstantTensor(OpBuilder &builder, Location loc, Type tensorType, float value);

  Value createConstantBoolTensor(OpBuilder &builder, Location loc, Type type, bool value);

  Operation* cloneSubtree(Operation *targetOp, IRMapping &mapper, OpBuilder &builder);

  void unrollAllForOps(triton::FuncOp func);

  llvm::DenseMap<Value, Value> addPointerArgsToFunction(triton::FuncOp funcOp);

  Operation* substituteBasePtr(Operation *targetOp, OpBuilder &builder, llvm::DenseMap<Value, Value> ptrToAddedPtrMap);

  void setInsertionPointAfterLastUse(Value val, OpBuilder &builder);

  Value createBroadcastOrSplat(Value input, Type targetType, Location loc, OpBuilder &builder);

  NameLoc createNodeName(Operation *op, std::string prefix);

  // Extract a readable forward-op label from Location (NameLoc through
  // CallSiteLoc/FusedLoc); fallback to op mnemonic
  StringAttr nameFromLoc(Operation *fwd);

  // Derive readable label and kernel-arg index from a pointer Value
  std::pair<StringAttr, IntegerAttr> labelFromPtr(OpBuilder &builder, Value anyPtr);

  // Propagate a branch's canonical arg index upstream from a sink (atomic/store)
  // through autodiff-inserted producers only, unioning into raise.gradIdxs.
  void propagateIdxFromSink(Operation *sink, IntegerAttr idx, OpBuilder &builder);

  // Compute or reuse a stable per-pass tag id for a given cloned forward op
  int64_t getOrAssignGradOfTag(llvm::DenseMap<Operation*, int64_t> &map,
                               int64_t &nextId,
                               Operation *clonedFwd);

  // Provenance tagging control (thread-local)
  // Set the current provenance used by Utils helpers (e.g., createConstantTensor)
  // to stamp raise.gradOfTag / raise.gradIdx / raise.gradIdxs on newly created ops.
  void setCurrentGradProvenance(IntegerAttr gradOfTag,
                                IntegerAttr gradIdx = nullptr,
                                ArrayAttr   gradIdxs = nullptr);

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITON_TO_AUTODIFF_UTILS_H 