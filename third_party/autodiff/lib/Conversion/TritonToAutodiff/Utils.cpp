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
#include "llvm/Support/Signals.h" // report_fatal_error

namespace mlir {
namespace triton {

  void markVisited(OpBuilder &builder, visitedType mode, Operation *op) {
    if (!op) {
      llvm::report_fatal_error("markVisited received null operation pointer\n");
    }

    NamedAttrList attrs;
    attrs.append("autogradVisited", builder.getBoolAttr(true));
    if (mode == visitedType::Original)
      attrs.append("isOrig", builder.getBoolAttr(true));
    else if (mode == visitedType::Inserted)
      attrs.append("isInserted", builder.getBoolAttr(true));
    else if (mode == visitedType::Cloned)
      attrs.append("isCloned", builder.getBoolAttr(true));
    else if (mode == visitedType::GradPtrRebase)
      attrs.append("isGradPtrRebase", builder.getBoolAttr(true));
    else {
      llvm::report_fatal_error("markVisited invalid visitedType value\n");
    }
    op->setAttrs(attrs);
  }

  Value getUpstreamGrad(Value result, const llvm::DenseMap<Value, Value> &gradMap) {
    auto it = gradMap.find(result);
    if (it == gradMap.end()) {
      llvm::errs() << "No grad found for " << result << "\n";
      llvm::report_fatal_error("Expected gradient in gradMap");
      // llvm::report_fatal_error("Expected gradient in the map for " + Twine(result));
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
      // Determine the element type of the tensor
      auto tensorElemType = mlir::cast<ShapedType>(tensorType).getElementType();

      // Ensure the tensor element type is either float16 or float32
      assert((tensorElemType.isF16() || tensorElemType.isF32()) && "Tensor element type must be float16 or float32");

      // Create a constant value of the appropriate type
      auto scalarValue = builder.create<arith::ConstantOp>(loc, builder.getFloatAttr(tensorElemType, value));

      // Splat the scalar value into a tensor of the desired type
      auto tensorValue = builder.create<triton::SplatOp>(loc, tensorType, scalarValue);

      // Mark the operations as visited
      markAllVisited(builder, visitedType::Inserted, scalarValue, tensorValue);

      return tensorValue;
  }



  Operation* cloneSubtree(Operation *targetOp, IRMapping &mapper, OpBuilder &builder) {

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


  Operation* substituteBasePtr(Operation *targetOp, OpBuilder &builder, llvm::DenseMap<Value, Value> ptrToAddedPtrMap) {

    if (!targetOp)
      return nullptr;

    // First identify the base pointer (block argument) we need to replace
    Value basePtr = nullptr;
    {
      SmallVector<Operation*> worklist{targetOp};
      DenseSet<Operation*> visited;

      while (!worklist.empty() && !basePtr) {
        Operation *op = worklist.back();
        worklist.pop_back();

        // If the operation was NOT newly inserted (meaning it was already in the set)
        //  The insert method of a set in C++ (like DenseSet<Operation*> here) returns a std::pair<iterator, bool> where:
        //  The .first element is an iterator pointing to the inserted element or to the existing element if it was already in the set
        //  The .second element is a boolean that is true if the insertion took place or false if the element was already in the set
        if (!visited.insert(op).second)
          continue;

        for (Value operand : op->getOperands()) {
          if (isa<BlockArgument>(operand)) {
            basePtr = operand;
            break;
          }

          if (Operation *defOp = operand.getDefiningOp())
            worklist.push_back(defOp);
        }
      }
    }

    IRMapping mapper;

    // No base pointer found
    if (!basePtr)
      return nullptr;

    // based on the basePtr we found (corresponds to some FuncOp arg), select gradPtr for the basePtr (previsoly in addPointerArgsToFunction I added additional args to the func, corresponding to grad pointers for every arg)
    Value replacementPtr = ptrToAddedPtrMap[basePtr];
    if (DEBUG_PRINTS)
      llvm::outs() << "load's basePtr " << basePtr << " was replaced with replacementPtr " << replacementPtr << ". Dependant nodes cloned to use the new base\n";

    // so the below subgraph will use the new ptr insted of original
    mapper.map(basePtr, replacementPtr);


    /*
    Track operations that depend on the base pointer.
    this is only need for early return below -- need to know what bool return

      if (auto *clonedOp = mapper.lookupOrNull(targetOp))
        return clonedOp;

    */
    DenseSet<Operation*> opsDependendingOnBasePtr;

    // Recursive post-order processing function
    std::function<std::pair<Operation*, bool>(Operation*)> processOp =
        [&](Operation *op) -> std::pair<Operation*, bool> {

      // wt this conditional, same node (e.g. make_range) is duplicated multiple times
      if (auto *clonedOp = mapper.lookupOrNull(op))
        return std::make_pair(clonedOp, opsDependendingOnBasePtr.contains(op));

      // computed as OR of all operands
      // iow if even one of the operands depends, then the current depends as well
      bool dependsOnBasePtr = false;
      Operation *resultOp;

      // Clone all operations we depend on first
      for (Value operand : op->getOperands()) {

        // Skip block arguments and values already in the mapper
        if (isa<BlockArgument>(operand)){

          // todo: I don't think there can be other BlockArgs in the subtree leading to tt:load operand
          //  because I think triton does not supprot adding two pointers, thus there can only be one base (IOW blockArgument)
          assert(operand == basePtr);
          dependsOnBasePtr = true;

          // // there can be other BlockArguments which are not my basePtr
          // if (operand == basePtr)
          //   dependsOnBasePtr = true;
          continue;
        }

        if (Operation *defOp = operand.getDefiningOp()){
          auto [_, isCurrDepends] = processOp(defOp); // , mapper, builder)
          dependsOnBasePtr = dependsOnBasePtr || isCurrDepends;
        }
      }

      if (dependsOnBasePtr){
        if (DEBUG_PRINTS) llvm::outs() << "Cloning (depends on base ptr): " << *op << "\n";
        resultOp = builder.clone(*op, mapper);
        markVisited(builder, visitedType::GradPtrRebase, resultOp);
        opsDependendingOnBasePtr.insert(op);
      } else {
        // Reuse this operation by mapping its results to themselves
        if (DEBUG_PRINTS) llvm::outs() << "Reusing (independent of base ptr): " << *op << "\n";
        // todo: correct?
        for (Value result : op->getResults())
          mapper.map(result, result);
        resultOp = op;
      }

      return std::make_pair(resultOp, dependsOnBasePtr);
    };

    auto [result, _] = processOp(targetOp);

    return result;

  }

  /*
    f = a * z
    k = BASE_PTR + f
    p = k - y

    // ops that don't depend on REPLACED_PTR don't need to be copied
    // ops that depend on it (directly or indirectly need to be copied)

    f = a * z
    k_copy = REPLACED_PTR + f
    p_copy = k_copy - y
  */

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
        llvm::report_fatal_error("[unrollAllForOps] failed to extract upper bound\n");
      }
    }

  }

  // todo-now: simplify this
  llvm::DenseMap<Value, Value> addPointerArgsToFunction(triton::FuncOp funcOp) {

    // If the function has a body, update the entry block arguments
    if (funcOp.isExternal())
      llvm::report_fatal_error("FuncOp body is emtpy\n");

    // Get existing function type
    auto fnType = funcOp.getFunctionType();
    SmallVector<Type> newInputTypes;
    SmallVector<Type> additionalPtrTypes; // Track only the new pointer types

    std::map<unsigned, unsigned> ptrIdxToAddedPtrIdxMap;

    unsigned numOrigInputs = fnType.getInputs().size();

    // For each argument, if it's a pointer, add a corresponding additional pointer
    for (auto [i, inputType] : llvm::enumerate(fnType.getInputs())) {
      newInputTypes.push_back(inputType);

      if (auto ptrType = dyn_cast<triton::PointerType>(inputType)) {
        newInputTypes.push_back(ptrType);
        additionalPtrTypes.push_back(ptrType); // Remember only the new ones

        // origPtrArgIndices.push_back(i);
        auto numAdded = ptrIdxToAddedPtrIdxMap.size();
        ptrIdxToAddedPtrIdxMap[i] = numOrigInputs + numAdded;
      }

    }

    // Create and set the new function type
    // isExternal() checks if the function is just a declaration without a body. If a function is external, it has no implementation block, so we don't need to add arguments to its entry block.
    auto newFnType = FunctionType::get(funcOp.getContext(), newInputTypes,  fnType.getResults());
    funcOp.setType(newFnType);


    Block &entryBlock = funcOp.getBody().front();

    // Only add block arguments for the newly added pointer types
    for (auto ptrType : additionalPtrTypes) {
      entryBlock.addArgument(ptrType, funcOp.getLoc());
    }



    // the below is only needed bc I also want to extract map from orig ptr args to the added ptr args (map from input ptr to its grad ptr)
    // but can't get an SSA value directly from an inputType since types only describe the kind of value, not the actual value
    // so need get Values from entryBlock instead
    llvm::DenseMap<Value, Value> ptrToAddedPtrMap; // map: input ptr -> its grad ptr

    for (auto [origIdx, addedIdx] : ptrIdxToAddedPtrIdxMap){
      // Block.getArgument returns BlockArgument
      // A BlockArgument is already a Value in MLIR. BlockArgument inherits from Value, so you can use it directly wherever a Value is expected.
      // Value origPtrSSA = entryBlock.getArgument(origIdx);
      // Value addedPtrSSA = entryBlock.getArgument(addedIdx);
      Value origPtrSSA = dyn_cast<Value>(entryBlock.getArgument(origIdx));
      Value addedPtrSSA = dyn_cast<Value>(entryBlock.getArgument(addedIdx));
      ptrToAddedPtrMap[origPtrSSA] = addedPtrSSA;

      if (DEBUG_PRINTS)
        // here still see BlockArgument printing overloads due to RTTI
        llvm::outs() << origPtrSSA << "(orig ptr) maps to " << addedPtrSSA << "(added ptr)\n";
    }

    return ptrToAddedPtrMap;


  }

} // namespace triton
} // namespace mlir