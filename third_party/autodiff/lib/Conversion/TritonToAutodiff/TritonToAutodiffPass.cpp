//#include "triton/Conversion/TritonToAutodiff/TritonToAutodiffPass.h"

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

#define GEN_PASS_DEF_CONVERTTRITONTOAUTODIFF
#include "autodiff/include/Conversion/TritonToAutodiff/Passes.h.inc"

namespace {

struct ConvertTritonToAutodiff
    : public impl::ConvertTritonToAutodiffBase<ConvertTritonToAutodiff> {

  using ConvertTritonToAutodiffBase::ConvertTritonToAutodiffBase;

  // main function
  void runOnOperation() override {
    // grab the module (IOW root) op
    auto mod = getOperation();
    // walk this recursively structred IR, and call rewriteSplatAddOp only on "triton::FuncOp"
    // todo-med: since I'm not using recursive funcs in "rewriteSplatAddOp", I'm not traversing body of the fn recursively (only the upper-most level)
    mod->walk([&](triton::FuncOp func) {
      rewriteSplatAddOp(func);
    });
  }

  // walk the IR backward, rewrite each operation with its corresponding backward function
  void rewriteSplatAddOp(triton::FuncOp func) {

    // todo-now: undo
    unrollAllForOps(func);
    // func.getBody().front().dump();
    // exit(1);
    if (DEBUG_PRINTS) {
      llvm::errs() << "flattening for loop:\n";
      func.getBody().front().print(llvm::errs());
      llvm::errs() << "\n";
    }

    llvm::DenseMap<Value, Value> ptrToAddedPtrMap = addPointerArgsToFunction(func);
    if (DEBUG_PRINTS) {
      llvm::errs() << "adding new pointers:\n" << func.getFunctionType().getInputs() << "\n\n";
    }


    // printOperation(func, true);

    // assimung there's upstream grad only wrt to a single variable initially
    // bool is_first_node

    // error happening because Value (which is the type I'm trying to put into std::map) does not have move interface
    // (< comparitor), which it appers the impl of map is trying ot use to compare eleemtns of the map
    llvm::DenseMap<Value, Value> gradMap;

    // copy entire forward graph once
    IRMapping origToCloned;
    // func.getBody() returns Region, but "setInsertionPointToStart" expects pass Block,
    // .front() gets the first block in that region, which is the entry block
    Block *entryBlock = &func.getBody().front();
    Operation *returnOp = &entryBlock->back();
    // last op before return op
    Operation *beforeReturnOp = returnOp->getPrevNode();
    // because these marked as visited, you will not match
    // them in your loop below, and thus you will not re-write
    // them -- so effectively this cloned is your *Forward* graph

    OpBuilder builder(func.getContext());
    builder.setInsertionPointToStart(entryBlock);

    // copied from: llvm-project/mlir/lib/Dialect/Linalg/Transforms/Hoisting.cpp
    SetVector<Operation *> forwardSlice;
    getForwardSlice(func.getOperation(), &forwardSlice);




    // Clones the entire fwd graph (not just a single subgraph leading from the last fwd op)
    DenseSet<Operation*> visitedLoads;
    // Operation *lastFwdOp = nullptr;
    Operation *lastFwdOp = beforeReturnOp;
    // Operation *currFwdOp;
    for (Operation *op : llvm::reverse(forwardSlice)) {

      if (DEBUG_PRINTS) llvm::errs() << "\n\n\niterating over op " << *op << "\n";

      auto currStoreOp = dyn_cast<triton::StoreOp>(op);
      if (visitedLoads.contains(op) || !currStoreOp || op->getBlock() != entryBlock){
        continue;
      }

      // note: important to pass the same map (origToCloned) -- so that the cloning logic
      //  does not re-clone nodes that are common between the subgraphs of nodes leading to different StoreOps
      cloneSubtree(currStoreOp, origToCloned, builder);
      // currFwdOp = cloneSubtree(currStoreOp, origToCloned, builder);
      // // first condition is for the 1st iter -- to overwrite nullptr at least with some currFwdOp
      // if (!lastFwdOp || currFwdOp->isBeforeInBlock(lastFwdOp)){
      //   lastFwdOp = currFwdOp;
      // }
    }




    // let the ops inserted during rewriting backward be inserted after the forward ops
    builder.setInsertionPointAfter(lastFwdOp);

    if (DEBUG_PRINTS) {
      llvm::errs() << "after cloning:\n";
      func.getBody().front().print(llvm::errs());
    }

    // the above mapping: original nodes -> inserted nodes.
    // To lookup intermideats in the cloned (aka cloned subgraph),
    // when iterating over the original subgraph (and re-writting that original subgraph with derivative formualrs)
    // I think I don't even need to reverse the mapping: can directly use it --
    // bc I'm iterating over the "original nodes" and want to figure out what "cloned" node does an original node refers to


    // // Walk all operations opaquely.
    // // todo: I think, you don't need topo sort if you're iterating in post order traversal (visit children before parent)
    // func->walk<WalkOrder::PostOrder>([&](Operation *op) {           // https://mlir.llvm.org/doxygen/classmlir_1_1Operation.html#a59740592240b950b8c8afcf4a2eb4113




    // separate loop for handle store[s]
    for (Operation *op : llvm::reverse(forwardSlice)) {

      if (DEBUG_PRINTS) llvm::errs() << "\n\n\niterating over op " << *op << "\n";

      if (op->getAttrOfType<BoolAttr>("autogradVisited")) {
          if (DEBUG_PRINTS) llvm::errs() << "Skipping visited" << "\n";
          continue;
      }

      // print only if changed
      std::string initialIR;
      llvm::raw_string_ostream initialStream(initialIR);
      entryBlock->print(initialStream);

      Operation *lastBwdOp = lastFwdOp;
      if (auto storeOp = dyn_cast<triton::StoreOp>(op)){
        lastBwdOp = handleStoreBackward(storeOp, builder, gradMap, origToCloned, lastBwdOp, ptrToAddedPtrMap);
      }

      if (DEBUG_PRINTS) {
        std::string currentIR;
        llvm::raw_string_ostream currentStream(currentIR);
        entryBlock->print(currentStream);
        if (initialIR != currentIR){
          // llvm::errs() << entryBlock->print();
          // dump writes to std err, but I want these be in "sync" with my other prints --
          llvm::raw_ostream &os = llvm::errs();
          entryBlock->print(os);
        }
      }

    }






    // First pass: handle all operations except LoadOp
    for (Operation *op : llvm::reverse(forwardSlice)) {

      if (op->getAttrOfType<BoolAttr>("autogradVisited")) {
          if (DEBUG_PRINTS) llvm::errs() << "Skipping visited" << "\n";
          continue;
      }


      // Note: I want to handle nested ops by dedicated handlers (e.g. reduce handler) -- not by the main loop
      //
      // No grad found for %96 = "arith.addf"(%arg18, %arg19) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      // ^bb0(%arg18: f32, %arg19: f32):
      //   %96 = "arith.addf"(%arg18, %arg19) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      //   "tt.reduce.return"(%96) : (f32) -> ()
      // ERROR: Expected gradient in gradMap
      if (op->getBlock() != entryBlock){
          if (DEBUG_PRINTS) llvm::errs() << "Skipping nested ops" << "\n";
          continue;
      }


      if (DEBUG_PRINTS) llvm::errs() << "\n\n\niterating over op " << *op << "\n";


      // print only if changed
      std::string initialIR;
      llvm::raw_string_ostream initialStream(initialIR);
      entryBlock->print(initialStream);

      // triton ops
      if (auto mmOp = dyn_cast<triton::DotOp>(op)){
        handleMatmulBackward(mmOp, builder, gradMap, origToCloned);

      // } else if (auto rangeOp = dyn_cast<triton::MakeRangeOp>(op)){
      //   if (DEBUG_PRINTS) printIndent() << "visiting tt.make_range op\n";
      // } else if (auto splatOp = dyn_cast<triton::SplatOp>(op)){
      //   if (DEBUG_PRINTS) printIndent() << "visiting tt.splat op\n";
      // } else if (auto addptrOp = dyn_cast<triton::AddPtrOp>(op)){
      //   if (DEBUG_PRINTS) printIndent() << "visiting tt.addptr op\n";

      // arith ops
      } else if (auto addfOp = dyn_cast<arith::AddFOp>(op)){
        handleAddBackward(addfOp, builder, gradMap);
      } else if (auto mulfOp = dyn_cast<arith::MulFOp>(op)){
        handleMulBackward(mulfOp, builder, gradMap, origToCloned);
      } else if (auto divfOp = dyn_cast<arith::DivFOp>(op)){
        handleDivBackward(divfOp, builder, gradMap, origToCloned);
      } else if (auto truncfOp = dyn_cast<arith::TruncFOp>(op)){
        handleTruncfBackward(truncfOp, builder, gradMap);
      } else if (auto constantOp = dyn_cast<arith::ConstantOp>(op)){
        if (DEBUG_PRINTS) printIndent() << "visiting arith.constant op\n";

      // math ops
      } else if (auto cosOp = dyn_cast<math::CosOp>(op)){
        handleCosBackward(cosOp, builder, gradMap, origToCloned);
      } else if (auto sinOp = dyn_cast<math::SinOp>(op)){
        handleSinBackward(sinOp, builder, gradMap, origToCloned);
      } else if (auto sqrtOp = dyn_cast<math::SqrtOp>(op)){
        handleSqrtBackward(sqrtOp, builder, gradMap, origToCloned);
      } else if (auto logOp = dyn_cast<math::LogOp>(op)){
        handleLogBackward(logOp, builder, gradMap, origToCloned);
      } else if (auto expOp = dyn_cast<math::ExpOp>(op)){
        handleExpBackward(expOp, builder, gradMap, origToCloned);
      }

      // todo-high: add else here (catch all) -- and explicitly error if none of the above
      // (otherwise users can have unsorted ops in their programs, and mine will just silently fail)

      // cleaner and more robust than adding printing to each handler
      if (DEBUG_PRINTS) {
        std::string currentIR;
        llvm::raw_string_ostream currentStream(currentIR);
        entryBlock->print(currentStream);
        if (initialIR != currentIR){
          // llvm::errs() << entryBlock->print();
          // dump writes to std err, but I want these be in "sync" with my other prints --
          llvm::raw_ostream &os = llvm::errs();
          entryBlock->print(os);
        }
      }

    } // for loop over ops

    // todo-now:
    //  in the stub I'm using output variable to pass the upstream grad
    //  but now when copying entire fwd graph, and the fwd graph actually
    //  writes output into that variable -- thus overwriting my upstream grad
    //
    // don't overwrite upstream with the fwd output
    lastFwdOp->erase();

    // Second pass: handle LoadOp operations
    // because its derivative (storeOp) destroyaes semantics of input args
    for (Operation *op : llvm::reverse(forwardSlice)) {

      if (op->getAttrOfType<BoolAttr>("autogradVisited")) {
          if (DEBUG_PRINTS) llvm::errs() << "Skipping visited" << "\n";
          continue;
      }

      if (auto loadOp = dyn_cast<triton::LoadOp>(op)){
        handleLoadBackward(loadOp, builder, gradMap, origToCloned, ptrToAddedPtrMap, func);
      }
    } // for loop over loads





    // I think it's bc here i explicitly delete all operations that don't have autogradVisited or isCloned attributes
    //  set -- the problem is this func->wall recursive walks on all IR nodes (including ones contained inside bodies of other
    // ops like reduce in thiscase)
    //
    // So actually I don't want to delete ops (even that doesn't have isCloned  or autogradVisited set) provided that the op
    // they are embedded into (e.g. reduce in this case has these attributes) -- so modify that function that walks that ir and
    // deletes nodes to also check attributes of the outher op -- and to not delete the current op in this case

    // the below modifications make sure I don't delete nested op (in this case %171) even if its unmarked -- provided that the outer op (in this case %18) is marked
    // %18 = "tt.reduce"(%17) <{axis = 0 : i32}> ({
    // ^bb0(%arg20: f32, %arg21: f32):
    //   %171 = "arith.addf"(%arg20, %arg21) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    //   "tt.reduce.return"(%171) : (f32) -> ()
    // }) {autogradVisited = true, isCloned = true} : (tensor<8192xf32>) -> f32


    // Final pass: remove unmarked operations
    func->walk<WalkOrder::PostOrder>([&](Operation *op) {

      auto visitedAttr = op->getAttrOfType<BoolAttr>("autogradVisited");
      // auto isClonedAttr = op->getAttrOfType<BoolAttr>("isCloned");

      if (DEBUG_PRINTS) {
        if (visitedAttr) llvm::errs() << "  Operation is marked as visited: " << *op << "\n";
        else llvm::errs() << "  Operation is NOT marked as visited: " << *op << "\n";
      }

      // Check if this operation or any parent operation has the required attributes
      bool shouldPreserve = visitedAttr || isa<triton::FuncOp, triton::ReturnOp>(op);

      // todo-now: i don't think I want to check for "parent node" but rather for an "outer node"
      // If not, check if any ancestor has the attributes
      if (!shouldPreserve) {
        Operation *parent = op->getParentOp();
        while (parent && !shouldPreserve) {
          shouldPreserve = parent->getAttrOfType<BoolAttr>("autogradVisited") || parent->getAttrOfType<BoolAttr>("isCloned");
          parent = parent->getParentOp();
        }
      }

      // Only delete if neither this op nor any parent has the required attributes
      if (!shouldPreserve) {
        if (DEBUG_PRINTS) llvm::errs() << "Deleting unmarked node" << *op << "\n";
        op->dropAllUses();
        op->erase();

      }
    }); // lambda function for the walk

  } // RewriteSplatOp function



  Operation* handleStoreBackward(triton::StoreOp storeOp, OpBuilder &builder,
                          llvm::DenseMap<Value, Value> &gradMap, IRMapping &origToCloned,
                          Operation *lastBwdOp,
                          llvm::DenseMap<Value, Value> ptrToAddedPtrMap){

    // because this will effectively load the upstream grad, I want to set the insertion point to right after the last node in fwd
    builder.setInsertionPointAfter(lastBwdOp);
    llvm::errs() << "[handleStoreBackward] lastBwdOp: " << lastBwdOp << "\n";

    // see all available constructors in -- triton/include/triton/Dialect/Triton/IR/TritonOps.td -> "def TT_LoadOp"
    // Value ptr = origToCloned.lookup(storeOp->getOperand(0));

    // [see code_comments]
    //   - using substituteBasePtr in handleLoadBackward is needed so that I STORE gradients I computed NOT into the fwd args themselves, but into additional arguments representing grads of these fwd args
    //   - using substituteBasePtr in handleStoreBackward is needed so that I LOAD **UPSTREAM** grads NOT from the "out" fwd arg directly, but from the additional argument representing grad wrt to "out"
    //   - ==> these can be looked at as two separate goals
    Value clonedPtr = origToCloned.lookup(storeOp.getPtr());
    Operation* clonedPtrOpRebased = substituteBasePtr(clonedPtr.getDefiningOp(), builder, ptrToAddedPtrMap);
    Value clonedPtrRebased = clonedPtrOpRebased->getResult(0);

    // remember the semantics:
    // you're iterating over the backward graph (that you're
    // re-writing at the same time), here you matched to StoreOp,
    // StoreOp.getMask() simply returns ssa Value of one of the
    // operands of that op (bc you're iterating over the
    // backward graph, thus that mask Value will come from
    // some node in the backward graph).
    // Because I want to re-use intermideats from the fwd graph instead,
    // here find the same mask Value but from the forward graph
    Value mask = storeOp.getMask();
    // some StoreOps don't have the mask value, in which case
    // mask above is a <<NULL VALUE>>
    Value maskCloned = mask ? origToCloned.lookup(mask) : Value();

    auto load = builder.create<triton::LoadOp>(
        storeOp.getLoc(),
        clonedPtrRebased,
        maskCloned,
        storeOp.getCache(),  // copy cache modifier
        storeOp.getEvict(),  // copy eviction policy
        false  // isVolatile (storeOp doesn't have this, so keep default)
    );

    // grad wrt 1st arg (values) is the output (aka Value) of the newly added op
    // if (DEBUG_PRINTS) llvm::errs() << "should be Value defined by add op: " << storeOp->getOperand(1) << "\n";
    // todo-high: note I'm currently assuming that there's a single Store function and that it's the first one be added to the gradMap
    maybeAccumulateGrad(storeOp->getOperand(1), load, gradMap, builder);

    // mark as visited
    markVisited(builder, visitedType::Inserted, load);

    // todo: (OLD) replace that blockArg with another tensor (corresponding to grad of that original argument)

    // return to use as insertion point for differentiating next soreOp I match to
    return load;
  }

  // todo-now:
  //  Don't just blindly add atomics in all cases, instead have an anylysis pass of what kernel instances actaully confifct and add finer grained atomics (locks) only for them
  // this version of the func adds atomics
  void handleLoadBackward(triton::LoadOp loadOp, OpBuilder &builder,
                          llvm::DenseMap<Value, Value> &gradMap, IRMapping &origToCloned,
                          llvm::DenseMap<Value, Value> ptrToAddedPtrMap,
                          triton::FuncOp func){
    if (DEBUG_PRINTS) llvm::errs() << "visiting tt.load op\n";

    Value upstream = getUpstreamGrad(loadOp, gradMap);

    // Create a builder without setting insertion point at first, then set insertion point
    // Seems no constructor to specify "InsertionPointAfter" at the time of construction

    // set insertion point to before the last operation (before ReturnOp)
    // .front() gets the first block in that region
    Block *entryBlock = &func.getBody().front();
    Operation *lastOp = &entryBlock->back();
    builder.setInsertionPoint(lastOp);

    // TypeRange typically specify types of outputs of an op. Here's it's empty bc this op does not produce any outputs
    //  Unlike e.g. creating LoadOp where I'm passing ptr.getType() because a load operation returns a value of the same type as what it's loading from the pointer
    // auto newOp = builder.create<triton::StoreOp>(loadOp.getLoc(), TypeRange(), operands);
    Value mask = loadOp.getMask();
    Value maskCloned = mask ? origToCloned.lookup(mask) : Value();

    // op with tree (pointer arithmetic) leading to it rooted at the new base (grad ptr)
    //
    // NOTE: use this ptr in the created atomicOp (or StoreOp) would essentially write gradient inplace of the original funcOp argument
    //  but using the opWithNewBase would write the gradient wrt to the argument in the grad ptr for the argument (instead of in the arg ptr itself)
    // Value ptr = origToCloned.lookup(loadOp->getOperand(0));
    //
    // NOTE: use origToCloned.lookup(loadOp->getOperand(0))->getDefiningOp instead of loadOp
    //  directly -- I think the latter would give the op in the backward graph being re-written,
    //  but the latter should give the fwd graph. And since I want my "cloning/or-reusing logic"
    //  (in substituteBasePtr) to re-use intermideats from fwd -- I'm passing the op from the fwd
    //  graph (accessed via origToCloned)

    Value clonedPtr = origToCloned.lookup(loadOp.getPtr());
    Operation* clonedPtrOpRebased = substituteBasePtr(clonedPtr.getDefiningOp(), builder, ptrToAddedPtrMap);
    Value clonedPtrRebased = clonedPtrOpRebased->getResult(0);

    // Create an AtomicRMWOp with FADD operation instead of StoreOp
    // This will atomically add the upstream gradient to the memory location
    //
    // NOTE: atomics are needed bc e.g. tiled matmul accesses same memory locations
    // of input A from different instances of the kernel -- see Done/6_/my.png
    auto atomicOp = builder.create<triton::AtomicRMWOp>(
        loadOp.getLoc(),
        upstream.getType(),  // Result type
        triton::RMWOp::FADD, // Atomic add operation
        clonedPtrRebased,    // Pointer to update
        upstream,            // Value to add
        maskCloned,          // Optional mask
        triton::MemSemantic::ACQUIRE_RELEASE, // Memory semantics
        triton::MemSyncScope::GPU             // Memory scope
    );

    // todo-high: note this op does not add anything to the gradMap, bc here I'm manually traversing
    // inputs to this op (hardcoded for the specific toy graphs I'm working with) and marking them so
    // that they will not be switched on (IOW iterated over) by this loop
    markVisited(builder, visitedType::Inserted, atomicOp);

    // fixes mismatch between the type of the value we're trying to store and the pointee type of the pointer we're storing to.
    // ensure the type of upstream matches what ptr points to.

    // Record original operation for debugging
    // loadOp.getOperation()->getName().getStringRef() -- does not include operands so result value
    // Use the operation's built-in printer
    std::string opStr;
    llvm::raw_string_ostream os(opStr);
    loadOp->print(os);
    atomicOp->setAttr("gradOf", builder.getStringAttr(opStr));
  }

  void handleAddBackward(arith::AddFOp addfOp, OpBuilder &builder,
                        llvm::DenseMap<Value, Value> &gradMap){
    if (DEBUG_PRINTS) llvm::errs() << "visiting arith.addf op\n";

    Value upstream = getUpstreamGrad(addfOp, gradMap);

    // don't insert unnecessary multiply of upstream with 1 (since numerically result is the same as wt multiplying)
    // float local_grad = 1.;

    Value lhs = addfOp.getOperand(0);
    maybeAccumulateGrad(lhs, upstream, gradMap, builder);

    Value rhs = addfOp.getOperand(1);
    maybeAccumulateGrad(rhs, upstream, gradMap, builder);
  }

  void handleTruncfBackward(arith::TruncFOp truncfOp, OpBuilder &builder,
                           llvm::DenseMap<Value, Value> &gradMap) {
    if (DEBUG_PRINTS) llvm::errs() << "visiting arith.truncf op\n";

    /*
    handler solves:
      // NOTE: 97 didn't have gradient in the gradMap bc I was not matching to the "arith.truncf"
      %97 = tt.dot %95, %96, %92
      %100 = arith.truncf %97
      tt.store %109, %100
    */

    Value upstream = getUpstreamGrad(truncfOp, gradMap);
    setInsertionPointAfterLastUse(upstream, builder);

    Value x = truncfOp.getOperand();

    // Create an extension operation to match the input type
    // Since we're going backward, we need to extend from result type to operand type
    auto extOp = builder.create<arith::ExtFOp>(
        truncfOp.getLoc(),
        x.getType(),  // Target type is the original input type
        upstream      // Upstream gradient with the truncated type
    );

    maybeAccumulateGrad(x, extOp, gradMap, builder);

    markAllVisited(builder, visitedType::Inserted, extOp);



    // todo-high: the problem is that I'm casting upstream gradient from float16 to float32 -- but then I'm adding [that upstream] @ [some fwd activation] where the forward activation is float16
    /* my backward graph
      %58 = "tt.load"(%22) tensor<16x16xf16>
      %59 = "arith.extf"(%58) (tensor<16x16xf16>) -> tensor<16x16xf32>
      %61 = "arith.constant"() <{value = 0.000000e+00 : f16}> () -> f16
      %62 = "tt.splat"(%61) (f16) -> tensor<16x16xf16>
      %60 = "tt.trans"(%51) <{order = array<i32: 1, 0>}> (tensor<16x16xf16>) -> tensor<16x16xf16>
      // NOTE: this creates a problem because the first operand is float32, but the second operand is float16
      %63 = "tt.dot"(%59, %60, %62)(tensor<16x16xf32>, tensor<16x16xf16>, tensor<16x16xf16>) -> tensor<16x16xf16>

    */
    // maybeAccumulateGrad(x, upstream, gradMap, builder);

  }

  void handleMulBackward(arith::MulFOp mulfOp, OpBuilder &builder,
                        llvm::DenseMap<Value, Value> &gradMap, IRMapping &origToCloned){
    if (DEBUG_PRINTS) llvm::errs() << "visiting arith.mulf op\n";

    Value upstream = getUpstreamGrad(mulfOp, gradMap);
    // insert operations after the gradient value, they depend on, is defined
    setInsertionPointAfterLastUse(upstream, builder);

    Value lhs = mulfOp.getOperand(0);
    Value rhs = mulfOp.getOperand(1);


    // (1) clone lhs subtree
    // essentially, it's just like std::map but just a mlir specific struct
    Value clonedLhs = origToCloned.lookup(lhs);

    // (2) differentiate rhs
    auto gradRhsOp = builder.create<arith::MulFOp>(mulfOp.getLoc(), clonedLhs, upstream);
    // note: I belive here I want to set grad of the original rhs (not ClonedRhs), because I'd continue differenciating the original path (while cloned will not be differenicated)
    maybeAccumulateGrad(rhs, gradRhsOp, gradMap, builder);
    markVisited(builder, visitedType::Inserted, gradRhsOp);

    // (3) clone rhs subtree
    // prepare for cloning another separate subgraph
    Value clonedRhs = origToCloned.lookup(rhs);

    // (4) differentiate lhs
    auto gradLhsOp = builder.create<arith::MulFOp>(mulfOp.getLoc(), clonedRhs, upstream);
    maybeAccumulateGrad(lhs, gradLhsOp, gradMap, builder);
    markVisited(builder, visitedType::Inserted, gradLhsOp);
  }


  void handleDivBackward(arith::DivFOp divfOp, OpBuilder &builder,
                        llvm::DenseMap<Value, Value> &gradMap, IRMapping &origToCloned){
    if (DEBUG_PRINTS) llvm::errs() << "visiting arith.divf op\n";

    Value upstream = getUpstreamGrad(divfOp, gradMap);
    // insert operations after the gradient value, they depend on, is defined
    setInsertionPointAfterLastUse(upstream, builder);

    Value a = divfOp.getOperand(0);
    Value b = divfOp.getOperand(1);

    // (1) clone lhs subtree
    Value aCloned = origToCloned.lookup(a);

    // (2) clone rhs subtree
    Value bCloned = origToCloned.lookup(b);

    // (3) differentiate lhs

    // a local
    // auto ones = builder.create<arith::ConstantOp>(divfOp->getLoc(), upstream.getType(), builder.getF32FloatAttr(1.0));
    // this creates a scalar and broadcasts it to a shape specificed by "upstream.getType()"
    auto ones = createConstantTensor(builder, divfOp->getLoc(), upstream.getType(), 1.0);
    auto aLocal = builder.create<arith::DivFOp>(divfOp->getLoc(), ones, bCloned);
    auto aDownstream = builder.create<arith::MulFOp>(divfOp->getLoc(), aLocal, upstream);
    maybeAccumulateGrad(a, aDownstream, gradMap, builder);

    markAllVisited(builder, visitedType::Inserted, aDownstream, ones, aLocal);

    // (4) differentiate rhs

    // b local

    // auto two = builder.create<arith::ConstantOp>(divfOp->getLoc(), divfOp.getType(), builder.getF32FloatAttr(2.0));
    auto pow = builder.create<arith::MulFOp>(divfOp.getLoc(), bCloned, bCloned);
    auto div = builder.create<arith::DivFOp>(divfOp.getLoc(), aCloned, pow);
    auto neg = createConstantTensor(builder, divfOp->getLoc(), div.getType(), -1.0);
    auto bLocal = builder.create<arith::MulFOp>(divfOp.getLoc(), neg, div);
    auto bDownstream = builder.create<arith::MulFOp>(divfOp.getLoc(), bLocal, upstream);
    maybeAccumulateGrad(b, bDownstream, gradMap, builder);

    markAllVisited(builder, visitedType::Inserted, bDownstream, bLocal, neg, div, pow);
  }

  void handleCosBackward(math::CosOp cosOp, OpBuilder &builder,
                        llvm::DenseMap<Value, Value> &gradMap, IRMapping &origToCloned) {
    if (DEBUG_PRINTS) llvm::errs() << "visiting math.cos op\n";

    Value upstream = getUpstreamGrad(cosOp, gradMap);
    setInsertionPointAfterLastUse(upstream, builder);

    Value x = cosOp.getOperand();
    Value xCloned = origToCloned.lookup(x);

    // derivative of cos(x) is -sin(x)
    auto sinOp = builder.create<math::SinOp>(cosOp.getLoc(), xCloned);
    auto negOne = createConstantTensor(builder, cosOp->getLoc(), upstream.getType(), -1.0);
    auto negSin = builder.create<arith::MulFOp>(cosOp.getLoc(), negOne, sinOp);
    auto xDownstream = builder.create<arith::MulFOp>(cosOp.getLoc(), negSin, upstream);

    // answer-now:
    //  gardMap seems to map values in OLD graph (which I'm iterating over, but not the cloned)
    //  to values in backward graph which I've already re-written
    maybeAccumulateGrad(x, xDownstream, gradMap, builder);

    markAllVisited(builder, visitedType::Inserted, xDownstream, negSin, negOne, sinOp);
  }

  void handleSinBackward(math::SinOp sinOp, OpBuilder &builder,
                        llvm::DenseMap<Value, Value> &gradMap, IRMapping &origToCloned) {
    if (DEBUG_PRINTS) llvm::errs() << "visiting math.sin op\n";

    Value upstream = getUpstreamGrad(sinOp, gradMap);
    setInsertionPointAfterLastUse(upstream, builder);

    Value x = sinOp.getOperand();
    Value xCloned = origToCloned.lookup(x);

    // derivative of sin(x) is cos(x)
    auto cosOp = builder.create<math::CosOp>(sinOp.getLoc(), xCloned);
    auto xDownstream = builder.create<arith::MulFOp>(sinOp.getLoc(), cosOp, upstream);

    maybeAccumulateGrad(x, xDownstream, gradMap, builder);

    markAllVisited(builder, visitedType::Inserted, xDownstream, cosOp);
  }

  void handleSqrtBackward(math::SqrtOp sqrtOp, OpBuilder &builder,
                          llvm::DenseMap<Value, Value> &gradMap, IRMapping &origToCloned) {
    if (DEBUG_PRINTS) llvm::errs() << "visiting math.sqrt op\n";

    Value upstream = getUpstreamGrad(sqrtOp, gradMap);
    setInsertionPointAfterLastUse(upstream, builder);

    Value x = sqrtOp.getOperand();
    Value sqrtResult = sqrtOp;

    // Value xCloned = origToCloned.lookup(x);
    Value sqrtResultCloned = origToCloned.lookup(sqrtResult);

    // derivative of sqrt(x) is 1/(2*sqrt(x))
    auto two = createConstantTensor(builder, sqrtOp->getLoc(), upstream.getType(), 2.0);
    auto twoSqrtX = builder.create<arith::MulFOp>(sqrtOp.getLoc(), sqrtResultCloned, two);
    auto one = createConstantTensor(builder, sqrtOp->getLoc(), upstream.getType(), 1.0);
    auto localGrad = builder.create<arith::DivFOp>(sqrtOp.getLoc(), one, twoSqrtX);
    auto xDownstream = builder.create<arith::MulFOp>(sqrtOp.getLoc(), localGrad, upstream);

    maybeAccumulateGrad(x, xDownstream, gradMap, builder);

    markAllVisited(builder, visitedType::Inserted, xDownstream, localGrad, one, twoSqrtX, two);
  }

  void handleLogBackward(math::LogOp logOp, OpBuilder &builder,
                        llvm::DenseMap<Value, Value> &gradMap, IRMapping &origToCloned) {
    if (DEBUG_PRINTS) llvm::errs() << "visiting math.log op\n";

    Value upstream = getUpstreamGrad(logOp, gradMap);
    setInsertionPointAfterLastUse(upstream, builder);

    Value x = logOp.getOperand();
    Value xCloned = origToCloned.lookup(x);

    // derivative of log(x) is 1/x
    auto one = createConstantTensor(builder, logOp->getLoc(), upstream.getType(), 1.0);
    auto localGrad = builder.create<arith::DivFOp>(logOp.getLoc(), one, xCloned);
    auto xDownstream = builder.create<arith::MulFOp>(logOp.getLoc(), localGrad, upstream);

    maybeAccumulateGrad(x, xDownstream, gradMap, builder);

    markAllVisited(builder, visitedType::Inserted, xDownstream, localGrad, one);
  }

  void handleExpBackward(math::ExpOp expOp, OpBuilder &builder,
                        llvm::DenseMap<Value, Value> &gradMap, IRMapping &origToCloned) {
    if (DEBUG_PRINTS) llvm::errs() << "visiting math.exp op\n";

    Value upstream = getUpstreamGrad(expOp, gradMap);
    setInsertionPointAfterLastUse(upstream, builder);

    Value x = expOp.getOperand();
    Value expResult = expOp;
    Value expResultCloned = origToCloned.lookup(expResult);

    // derivative of exp(x) is exp(x) itself
    // We already have exp(x) from the forward pass, so use it directly
    auto xDownstream = builder.create<arith::MulFOp>(expOp.getLoc(), expResultCloned, upstream);

    maybeAccumulateGrad(x, xDownstream, gradMap, builder);

    markAllVisited(builder, visitedType::Inserted, xDownstream);
  }



  void handleMatmulBackward(triton::DotOp mmOp, OpBuilder &builder,
                          llvm::DenseMap<Value, Value> &gradMap, IRMapping &origToCloned) {
    if (DEBUG_PRINTS) llvm::errs() << "visiting tt.dot op\n";

    Value upstream = getUpstreamGrad(mmOp, gradMap);
    setInsertionPointAfterLastUse(upstream, builder);

    // Extract matrix multiplication operands
    Value a = mmOp.getA();
    Value b = mmOp.getB();
    Value c = mmOp.getC();  // Accumulator

    // Get cloned operands from forward graph
    Value aCloned = origToCloned.lookup(a);
    Value bCloned = origToCloned.lookup(b);


    // ~~~~~~~~~~~~~~~~~ maybe truncate upstream ~~~~~~~~~~~~~~~~~
    // todo: move this to separate fn; and run it for all handlers -- not just matmul?
    /*
    check dtype of upstream and dtype of a_cloned and b_cloned (operands from the fwd pass which will
    be used for grad computation) -- and if the fwd operands are of lower precision than upstream (grad wrt output buffer for the
    mamtul that we matched to) then add additional operations to cast upstream to the dtype of operands (lower precision)
    */

    // Check element types and truncate upstream if needed
    auto upstreamType = dyn_cast<ShapedType>(upstream.getType());
    auto aType = dyn_cast<ShapedType>(aCloned.getType());
    auto bType = dyn_cast<ShapedType>(bCloned.getType());
    if (!upstreamType || !aType || !bType) {
      llvm::report_fatal_error("Expected shaped types in handleMatmulBackward");
    }

    auto upstreamElemType = dyn_cast<FloatType>(upstreamType.getElementType());
    auto aElemType = dyn_cast<FloatType>(aType.getElementType());
    auto bElemType = dyn_cast<FloatType>(bType.getElementType());

    assert(aElemType == bElemType);

    // If upstream has higher precision than operands, truncate it

    // Potentially truncated dC
    Value processedUpstream = upstream;
    if (upstreamElemType.getWidth() > aElemType.getWidth() ||  upstreamElemType.getWidth() > bElemType.getWidth()) {
      if (DEBUG_PRINTS) llvm::errs() << "Truncating upstream gradient to match operand precision\n";

      // Determine the target type (use the lower precision of a and b)
      FloatType targetElemType = aElemType;

      auto targetType = RankedTensorType::get(upstreamType.getShape(), targetElemType);
      auto processedUpstreamOp = builder.create<arith::TruncFOp>(mmOp.getLoc(), targetType, upstream);
      processedUpstream = processedUpstreamOp->getResult(0);
      markVisited(builder, visitedType::Inserted, processedUpstreamOp);
    }
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    // For matmul C = A * B + acc
    // dA = dC * B^T
    // dB = A^T * dC
    // dacc = dC (gradient flows directly to accumulator)

    std::vector<int32_t> transOrder = {1, 0};
    auto bTrans = builder.create<triton::TransOp>(
        mmOp.getLoc(),
        bCloned.getType(),
        bCloned,
        builder.getDenseI32ArrayAttr(transOrder));

    // Compute gradient for A: dA = dC * B^T
    auto gradA = builder.create<triton::DotOp>(
        mmOp.getLoc(),
        a.getType(),                  // Result type should match A's type
        processedUpstream,                     // dC
        bTrans,                       // B^T
        createConstantTensor(builder, mmOp.getLoc(), a.getType(), 0.0), // zero accumulator
        mmOp.getInputPrecision(),
        mmOp.getMaxNumImpreciseAcc());

    maybeAccumulateGrad(a, gradA, gradMap, builder);


    auto aTrans = builder.create<triton::TransOp>(
        mmOp.getLoc(),
        aCloned.getType(),
        aCloned,
        builder.getDenseI32ArrayAttr(transOrder));

    // Compute gradient for B: dB = A^T * dC
    auto gradB = builder.create<triton::DotOp>(
        mmOp.getLoc(),
        b.getType(),                  // Result type should match B's type
        aTrans,                       // A^T
        processedUpstream,                     // dC
        // todo-high: or accumulate into here (instead of maybeAccumulateGrad)
        createConstantTensor(builder, mmOp.getLoc(), b.getType(), 0.0), // zero accumulator
        mmOp.getInputPrecision(),
        mmOp.getMaxNumImpreciseAcc());

    maybeAccumulateGrad(b, gradB, gradMap, builder);

    // Compute gradient for C (accumulator): dC = dOut
    // The gradient of the accumulator is just the upstream gradient
    maybeAccumulateGrad(c, upstream, gradMap, builder);

    markAllVisited(builder, visitedType::Inserted, gradA, gradB, bTrans, aTrans);
  }


}; // ConvertTritonToAutodiff stuct

} // private namespace
} // namespace triton
} // namespace mlir

