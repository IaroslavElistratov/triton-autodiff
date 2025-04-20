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
      result.getParentBlock()->dump();
      llvm::report_fatal_error("Expected gradient in gradMap");
      // llvm::report_fatal_error("Expected gradient in the map for " + Twine(result));
    }
    return it->second;
  }

  void maybeAccumulateGrad(Value val, Value grad,
                          llvm::DenseMap<Value, Value> &gradMap,
                          OpBuilder &builder) {

    if (val.getType() != grad.getType()) {
      llvm::errs() << "val type: " << val.getType() << "\n";
      llvm::errs() << "grad type: " << grad.getType() << "\n";
      llvm::report_fatal_error("[maybeAccumulateGrad] shape of grad does not match shape of Value");
    }
    auto it = gradMap.find(val);

    // no existing grad
    if (it == gradMap.end()) {
      gradMap[val] = grad;

    // found existing grad wrt val
    } else {
      auto existingGrad = it->second;

      // otherwise "does not dominate its use"  err
      // Note: when the grad value was defined by an op which was inserted by
      //  a handler which called setInsertionPointAfterLastUse, that means the
      //  grad Value is already inserted into the graph after last use of the
      //  upstream value (IOW existingGrad value)
      builder.setInsertionPointAfterValue(grad);

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


  Value createConstantTensor(OpBuilder &builder, Location loc, Type type, float value) {
      // Determine the element type of the tensor
      // llvm::errs() << type; // tensor<8192xf32>f32


      // Difference between Type and ShapedType in MLIR:
      // - Type is the base class for all types in MLIR's type system. It represents any kind of type in MLIR, which could be primitive types (like integers, floats), complex types (like tensors, memrefs), or function types.
      // - ShapedType is a subclass of Type that specifically represents types with a shape, such as:
      //    RankedTensorType: Tensors with known dimensions
      //    MemRefType: Memory references with known dimensions
      //    UnrankedTensorType: Tensors with unknown dimensions
      //    UnrankedMemRefType: Memory references with unknown dimensions
      // auto tensorElemType = dyn_cast<ShapedType>(tensorType).getElementType();


      /*
      This needs to handle the case where tensorType is just a Type and not Shaped Type -- e.g. below when divf's operand is just a scalar (and not a tensor)
        but I still need to differentiate it, bc that scalar was computed by taking a tensor and reducing it into a single float (via tt.reduce)

        %18 = "tt.reduce"(%17) <{axis = 0 : i32}> ({
        ^bb0(%arg8: f32, %arg9: f32):
          %43 = arith.addf %arg8, %arg9 : f32
          tt.reduce.return %43 : f32
        }) : (tensor<8192xf32>) -> f32
        %19 = arith.divf %18, %cst_1 : f32
        %20 = arith.addf %19, %arg7 : f32
        %21 = math.sqrt %20 : f32
        %22 = arith.divf %cst_0, %21 : f32

      */

      // Handle all cases: either it's already a ShapedType or we need to extract it
      if (auto tensorType = dyn_cast<ShapedType>(type)) {
        // If it's ShapedType extract type of a single element
        Type scalarType = tensorType.getElementType();
        assert((scalarType.isF16() || scalarType.isF32()) && "Tensor element type must be float16 or float32");

        auto scalarValue = builder.create<arith::ConstantOp>(loc, builder.getFloatAttr(scalarType, value));

        // Splat the scalar value into a tensor of the desired type
        auto tensorValue = builder.create<triton::SplatOp>(loc, tensorType, scalarValue);

        markAllVisited(builder, visitedType::Inserted, scalarValue, tensorValue);
        return tensorValue;

      } else if (auto scalarType = dyn_cast<Type>(type)) {
        // If it's Type, just use it directly
        assert((scalarType.isF16() || scalarType.isF32()) && "Tensor element type must be float16 or float32");

        auto scalarValue = builder.create<arith::ConstantOp>(loc, builder.getFloatAttr(scalarType, value));
        markVisited(builder, visitedType::Inserted, scalarValue);
        return scalarValue;

      } else {
        // can handle other cases in the future,
        // thigh can't hink of any at the moment
        llvm::report_fatal_error("unreachable");
      }

  }


  // todo-low: re-use createConstantTensor?
  Value createConstantBoolTensor(OpBuilder &builder, Location loc, Type type, bool value) {
      // llvm::errs() << type; // tensor<8192xi1>
      //    i1 represents a 1-bit integer (a boolean-like value, typically 0 or 1).

      if (auto tensorType = dyn_cast<ShapedType>(type)) {
        Type elemType = tensorType.getElementType();
        assert(elemType.isInteger(1) && "Tensor element type must be an int with exactly 1 bit");
        auto scalarValue = builder.create<arith::ConstantOp>(loc, builder.getBoolAttr(value));
        // auto scalarValue = builder.create<arith::ConstantOp>(loc, getAttr(elemType, value ? 1 : 0));
        auto tensorValue = builder.create<triton::SplatOp>(loc, tensorType, scalarValue);
        markAllVisited(builder, visitedType::Inserted, scalarValue, tensorValue);
        return tensorValue;
      } else {
        llvm::report_fatal_error("unreachable");
      }
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
          // NOTE: because I explicitly call substituteBasePtr on store / load (targetOp) -- there should be a base pointer (somewhere up in the def-chain, form the perspective of targetOp)
          //  and because, as I understand, triton does not support e.g. adding two pointers -- there can only be one basePtr in that def-chain
          if (isa<BlockArgument>(operand) && isa<triton::PointerType>(operand.getType())) {
            basePtr = operand;
            break;
          }

          if (Operation *defOp = operand.getDefiningOp())
            worklist.push_back(defOp);
        }
      }
    }


    // No base pointer found
    if (!basePtr)
      return nullptr;

    // based on the basePtr we found (corresponds to some FuncOp arg), select gradPtr
    // for the basePtr (in addPointerArgsToFunction I added additional args to the func,
    // corresponding to grad pointers for every arg)
    Value replacementPtr = ptrToAddedPtrMap[basePtr];
    if (DEBUG_PRINTS)
      llvm::errs() << "[substituteBasePtr] load's basePtr " << basePtr << " was replaced with replacementPtr " << replacementPtr << ". Dependant nodes cloned to use the new base\n";

    // don't pass origToCloned as mapper argument (IRMapping) to substituteBasePtr,
    // instead create a completely new IRMapping from inside this function
    // -- using origToCloned here would just always exit early "mapper.lookupOrNull(op)"
    // and not do any copying
    IRMapping mapper;

    // so the below subgraph cloning logic will use the new ptr insted of original
    mapper.map(basePtr, replacementPtr);


    /*
    Track operations that depend on the base pointer. this is only needed for
    early return below "if (auto *clonedOp = mapper.lookupOrNull(targetOp))"
    -- need to know what bool (isOpDependsOnBasePtr) to return
    */
    DenseSet<Operation*> opsDependendingOnBasePtr;

    // Recursive post-order traversal
    std::function<std::pair<Operation*, bool>(Operation*)> processOp =
        [&](Operation *op) -> std::pair<Operation*, bool> {

      // wt this conditional, same node (e.g. make_range) is duplicated multiple times
      if (auto *clonedOp = mapper.lookupOrNull(op))
        return std::make_pair(clonedOp, opsDependendingOnBasePtr.contains(op));

      // computed as OR of all operands
      // iow if even one of the operands depends, then the current depends as well
      bool isOpDependsOnBasePtr = false;
      Operation *resultOp;

      // Clone all operations we depend on first
      for (Value operand : op->getOperands()) {

        // the 2nd condition is needed bc there can other non pointer arguments -- don't treat them as the pointer base that I'm looking for
        if (isa<BlockArgument>(operand) && isa<triton::PointerType>(operand.getType())){

          // NOTE: I don't think there can be other BlockArgs in the subtree leading to tt:load operand
          //  because I think triton does not support adding two pointers, thus there can only be one base (IOW blockArgument)

          // ==> oh yeah, there can be other **NON-POINTER** arguments (e.g. above) -- this was just used as some offset into another BlockArg which was a pointer
          //   - operand: <block argument> of type 'i32' at index: 6; basePtr: <block argument> of type '!tt.ptr<f16>' at index: 1LLVM ERROR: Expected only to find basePtr
          // signature of the associated tl code:
          //    def _layer_norm_fwd_fused(
          //        X,  # pointer to the input
          //        Y,  # pointer to the output
          //        stride,  # how much to increase the pointer when moving by 1 row
          //        N,  # number of columns in X
          //    )

          assert(operand == basePtr);
          isOpDependsOnBasePtr = true;
          continue;
        }

        if (Operation *defOp = operand.getDefiningOp()){
          auto [_, isCurrParentDepends] = processOp(defOp);
          isOpDependsOnBasePtr = isOpDependsOnBasePtr || isCurrParentDepends;
        }
      }

      if (isOpDependsOnBasePtr){
        if (DEBUG_PRINTS) llvm::errs() << "Cloning (depends on base ptr): " << *op << "\n";

        resultOp = builder.clone(*op, mapper);

        markVisited(builder, visitedType::GradPtrRebase, resultOp);
        opsDependendingOnBasePtr.insert(op);
      } else {
        // Reuse this operation by mapping its results to themselves
        if (DEBUG_PRINTS) llvm::errs() << "Reusing (independent of base ptr): " << *op << "\n";
        // todo: correct?
        for (Value result : op->getResults())
          mapper.map(result, result);
        resultOp = op;
      }

      return std::make_pair(resultOp, isOpDependsOnBasePtr);
    };

    auto [result, _] = processOp(targetOp);
    return result;
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
        llvm::report_fatal_error("[unrollAllForOps] failed to extract upper bound\n");
      }
    }

  }

  /* error

  The function's signature expects argument #4 to have the type !tt.ptr<f16> (a pointer to a 16-bit floating-point value).
  However, the entry block of the function defines argument #4 as !tt.ptr<f32> (a pointer to a 32-bit floating-point value).

    the types of arguments in the function's entry block must exactly match the types declared in the function signature


  or: 'tt.func' op type of entry block argument #4('!tt.ptr<f32>') must match the type of the corresponding argument in function signature('!tt.ptr<f16>')
    tt.func public @_layer_norm_fwd_fused(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: f32) attributes {noinline = false} {
    ^

  answer-now:
  ==> currently for block args, I just append newly added arguments AFTER all the original argumenets -- that only works when all

    I think the problem was bc I inserted ptr args to right after original ptr arg (for the fn signature)
    (ptr_1, ADDED_ptr_1, ptr_2, ADDED_ptr_2, **int_1**, ptr_3, ADDED_ptr_3)

    but incorrectly inserted all inserted args after all original args for the blockArgs:
    (ptr_1, ptr_2, **int_1**, ptr_3, ADDED_ptr_1, ADDED_ptr_2, ADDED_ptr_3)

  */
  // todo: simplify this
  llvm::DenseMap<Value, Value> addPointerArgsToFunction(triton::FuncOp funcOp) {

    // isExternal() checks if the function is just a declaration without
    // a body. If a function is external, it has no implementation block
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
        additionalPtrTypes.push_back(ptrType); // Remember only the new ones

        auto numAdded = ptrIdxToAddedPtrIdxMap.size();
        ptrIdxToAddedPtrIdxMap[i] = numOrigInputs + numAdded;
      }

    }

    // append newly added args to the end of ALL original args (including after original non ptr args), iow:
    //  Original inputs: (ptr_1, ptr_2, **int_1**, ptr_3)
    //  Modified: (ptr_1, ptr_2, **int_1**, ptr_3, ADDED_ptr_1, ADDED_ptr_2, ADDED_ptr_3)
    newInputTypes.append(additionalPtrTypes.begin(), additionalPtrTypes.end());

    // Create and set the new function type
    auto newFnType = FunctionType::get(funcOp.getContext(), newInputTypes,  fnType.getResults());
    funcOp.setType(newFnType);


    Block &entryBlock = funcOp.getBody().front();

    // Only add block arguments for the newly added pointer types
    for (auto ptrType : additionalPtrTypes) {
      entryBlock.addArgument(ptrType, funcOp.getLoc());
    }


    // the below is only needed bc I also want to extract map from orig ptr args to the added ptr args (map from input ptr to its grad ptr)
    // but can't get an SSA value directly from an inputType since types only describe the kind of value, not the actual value
    // so need to get Values from entryBlock instead
    llvm::DenseMap<Value, Value> ptrToAddedPtrMap; // map: input ptr -> its grad ptr

    for (auto [origIdx, addedIdx] : ptrIdxToAddedPtrIdxMap){
      // Block.getArgument returns BlockArgument. BlockArgument inherits
      // from Value, so can use it directly wherever a Value is expected
      Value origPtrSSA = entryBlock.getArgument(origIdx);
      Value addedPtrSSA = entryBlock.getArgument(addedIdx);
      ptrToAddedPtrMap[origPtrSSA] = addedPtrSSA;

      if (DEBUG_PRINTS)
        // here still see BlockArgument printing overloads due to RTTI
        llvm::outs() << origPtrSSA << "(orig ptr) maps to " << addedPtrSSA << "(added ptr)\n";
    }

    return ptrToAddedPtrMap;
  }




  // find the last operation with a specific attribute among users of an SSA value
  Operation* findLastNodeWithAttribute(Value val, StringRef attrName) {

    // fallback in case there are no operations in the bwd graph (being built) that use the "val"
    Operation* lastOpWithAttr = val.getDefiningOp();

    for (Operation* user : val.getUsers()) {

      if (user->hasAttr(attrName) && lastOpWithAttr->isBeforeInBlock(user)) {
        lastOpWithAttr = user;
      }

      // Recursively check the users of each result of this operation
      for (Value result : user->getResults()) {
        Operation* childWithAttr = findLastNodeWithAttribute(result, attrName);
        if (lastOpWithAttr->isBeforeInBlock(childWithAttr)) {
          lastOpWithAttr = childWithAttr;
        }
      }
    }

    assert(lastOpWithAttr);

    return lastOpWithAttr;
  }

  // set insertion point after the **LAST USE** (in the bwd graph being re-written) of gradient value they depend on
  // IOW: the last SSA value from the bwd (differentiated so far) that used the gradient value
  void setInsertionPointAfterLastUse(Value val, OpBuilder &builder){
    Operation* lastUseInBwd = findLastNodeWithAttribute(val, "isInserted");
    builder.setInsertionPointAfter(lastUseInBwd);
  }


  // Helper function to create either a SplatOp or ExpandDimsOp+BroadcastOp based on input type
  Value createBroadcastOrSplat(Value input, Type targetType, Location loc, OpBuilder &builder) {
    // Check if input is a tensor or scalar
    if (isa<TensorType>(input.getType())) {
      // For tensor input, use expand_dims + broadcast
      auto inputTensorType = dyn_cast<RankedTensorType>(input.getType());
      auto targetTensorType = dyn_cast<RankedTensorType>(targetType);

      if (inputTensorType && targetTensorType) {
        // First expand dimensions to match the target shape
        Value currentResult = input;

        // If ranks don't match, we need to add dimensions
        int inputRank = inputTensorType.getRank();
        int targetRank = targetTensorType.getRank();

        // Add missing dimensions
        for (int i = inputRank; i < targetRank; i++) {
          auto expandOp = builder.create<triton::ExpandDimsOp>(
              loc,
              currentResult,
              i); // Add dimension at position i
          currentResult = expandOp->getResult(0);
          markVisited(builder, visitedType::Inserted, expandOp);
        }

        // Then broadcast to the target shape
        auto broadcastOp = builder.create<triton::BroadcastOp>(
            loc,
            targetType,
            currentResult);

        markVisited(builder, visitedType::Inserted, broadcastOp);
        return broadcastOp->getResult(0);
      }
    }

    // For scalar input or fallback case, use splat
    auto splatOp = builder.create<triton::SplatOp>(
        loc,
        targetType,
        input);

    markVisited(builder, visitedType::Inserted, splatOp);
    return splatOp->getResult(0);
  }

} // namespace triton
} // namespace mlir