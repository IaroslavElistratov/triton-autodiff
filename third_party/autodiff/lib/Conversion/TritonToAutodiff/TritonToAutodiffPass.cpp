//#include "triton/Conversion/TritonToAutodiff/TritonToAutodiffPass.h"


#include "mlir/IR/AsmState.h"          // registerAsmPrinterCLOptions
#include "llvm/Support/CommandLine.h"


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

  // instead of piping this variable through (which would require changing signature of all the functions) -- instead I set it as a global, so that all handlers can access it
  // To make nodeName accessible across all handler functions without changing their signatures, add it as a member variable to the ConvertTritonToAutodiff struct
  // Member variable to store current node name
  NameLoc currentNodeName;

  // Helper method to create an operation with the current node name
  template <typename OpTy, typename... Args>
  OpTy createGradOp(OpBuilder &builder, Args &&...args) {
    return builder.create<OpTy>(currentNodeName, std::forward<Args>(args)...);
  }

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

  void enableNameLocSSA() {
    mlir::registerAsmPrinterCLOptions();   // registers --mlir-use-nameloc-as-prefix

    static const char *argv[] = {
        "TritonToAutodiffPass",           // argv[0] — file name must be present
        "--mlir-use-nameloc-as-prefix",
        "--mlir-print-debuginfo"
    };
    constexpr int argc = sizeof(argv) / sizeof(argv[0]);

    // If ParseCommandLineOptions might have run before, reset first:
    llvm::cl::ResetAllOptionOccurrences();          // optional safety

    llvm::cl::ParseCommandLineOptions(argc,
                                      const_cast<char **>(argv),
                                      /*Overview=*/"");  // overview text is optional
  }

  // walk the IR backward, rewrite each operation with its corresponding backward function
  void rewriteSplatAddOp(triton::FuncOp func) {

    enableNameLocSSA();

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
    // todo: cleanup
    Operation *lastFwdOp = nullptr;
    Operation *currFwdOp;
    for (Operation *op : llvm::reverse(forwardSlice)) {

      if (DEBUG_PRINTS) llvm::errs() << "\n\n\niterating over op " << *op << "\n";

      auto currStoreOp = dyn_cast<triton::StoreOp>(op);
      if (visitedLoads.contains(op) || !currStoreOp || op->getBlock() != entryBlock){
        continue;
      }

      // note: important to pass the same map (origToCloned) -- so that the cloning logic
      //  does not re-clone nodes that are common between the subgraphs of nodes leading to different StoreOps
      currFwdOp = cloneSubtree(currStoreOp, origToCloned, builder);
      // first condition is for the 1st iter -- to overwrite nullptr at least with some currFwdOp
      if (!lastFwdOp || !currFwdOp->isBeforeInBlock(lastFwdOp)){
        lastFwdOp = currFwdOp;
      }
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

      NameLoc nodeName = createNodeName(op, "bwd_");
      // Store the name in the member variable for use in handlers
      currentNodeName = nodeName;

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

      NameLoc nodeName = createNodeName(op, "bwd_");
      // Store the name in the member variable for use in handlers
      currentNodeName = nodeName;

      // print only if changed
      std::string initialIR;
      llvm::raw_string_ostream initialStream(initialIR);
      entryBlock->print(initialStream);

      // triton ops
      if (auto mmOp = dyn_cast<triton::DotOp>(op)){
        handleMatmulBackward(mmOp, builder, gradMap, origToCloned);
      } else if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)){
        handleReduceBackward(reduceOp, builder, gradMap, origToCloned);
      } else if (auto broadcastOp = dyn_cast<triton::BroadcastOp>(op)){
        handleBroadcastBackward(broadcastOp, builder, gradMap);
      } else if (auto expandDimsOp = dyn_cast<triton::ExpandDimsOp>(op)){
        handleExpandDimsBackward(expandDimsOp, builder, gradMap);
      } else if (auto transOp = dyn_cast<triton::TransOp>(op)){
        handleTransBackward(transOp, builder, gradMap);
      } else if (auto splatOp = dyn_cast<triton::SplatOp>(op)){
        handleSplatBackward(splatOp, builder, gradMap);


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
        if (DEBUG_PRINTS) llvm::errs() << "visiting arith.constant op\n";
      } else if (auto extfOp = dyn_cast<arith::ExtFOp>(op)){
        handleExtFBackward(extfOp, builder, gradMap);
      } else if (auto subfOp = dyn_cast<arith::SubFOp>(op)){
        handleSubfBackward(subfOp, builder, gradMap);
      } else if (auto selectOp = dyn_cast<arith::SelectOp>(op)){
        handleSelectBackward(selectOp, builder, gradMap, origToCloned);
      } else if (auto maxOp = dyn_cast<arith::MaxNumFOp>(op)){
        handleMaxBackward(maxOp, builder, gradMap, origToCloned);

      // math ops
      } else if (auto cosOp = dyn_cast<math::CosOp>(op)){
        handleCosBackward(cosOp, builder, gradMap, origToCloned);
      } else if (auto sinOp = dyn_cast<math::SinOp>(op)){
        handleSinBackward(sinOp, builder, gradMap, origToCloned);
      } else if (auto sqrtOp = dyn_cast<math::SqrtOp>(op)){
        handleSqrtBackward(sqrtOp, builder, gradMap, origToCloned);
      } else if (auto logOp = dyn_cast<math::LogOp>(op)){
        handleLogBackward(logOp, builder, gradMap, origToCloned);  // For natural logarithm (base e): The derivative of ln(x) is 1/x
      } else if (auto log2Op = dyn_cast<math::Log2Op>(op)){
        handleLog2Backward(log2Op, builder, gradMap, origToCloned); // For logarithm base 2: The derivative of log₂(x) is 1/(x·ln(2))
      } else if (auto expOp = dyn_cast<math::ExpOp>(op)){
        handleExpBackward(expOp, builder, gradMap, origToCloned);
      } else if (auto exp2Op = dyn_cast<math::Exp2Op>(op)){
        handleExp2Backward(exp2Op, builder, gradMap, origToCloned);
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
    // lastFwdOp->erase();

    // Second pass: handle LoadOp operations
    // because its derivative (storeOp) destroyaes semantics of input args
    for (Operation *op : llvm::reverse(forwardSlice)) {

      if (op->getAttrOfType<BoolAttr>("autogradVisited")) {
          if (DEBUG_PRINTS) llvm::errs() << "Skipping visited" << "\n";
          continue;
      }

      NameLoc nodeName = createNodeName(op, "bwd_");
      // Store the name in the member variable for use in handlers
      currentNodeName = nodeName;

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

    auto load = createGradOp<triton::LoadOp>(
        builder,
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
    // auto newOp = createGradOp<triton::StoreOp>(builder, TypeRange(), operands);
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
    auto atomicOp = createGradOp<triton::AtomicRMWOp>(
        builder,
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
    auto extOp = createGradOp<arith::ExtFOp>(
        builder,
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
    auto gradRhsOp = createGradOp<arith::MulFOp>(builder, clonedLhs, upstream);
    // note: I belive here I want to set grad of the original rhs (not ClonedRhs), because I'd continue differenciating the original path (while cloned will not be differenicated)
    maybeAccumulateGrad(rhs, gradRhsOp, gradMap, builder);
    markVisited(builder, visitedType::Inserted, gradRhsOp);

    // (3) clone rhs subtree
    // prepare for cloning another separate subgraph
    Value clonedRhs = origToCloned.lookup(rhs);

    // (4) differentiate lhs
    auto gradLhsOp = createGradOp<arith::MulFOp>(builder, clonedRhs, upstream);
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
    // auto ones = createGradOp<arith::ConstantOp>(builder, upstream.getType(), builder.getF32FloatAttr(1.0));
    // this creates a scalar and broadcasts it to a shape specificed by "upstream.getType()"
    auto ones = createConstantTensor(builder, currentNodeName, upstream.getType(), 1.0);
    auto aLocal = createGradOp<arith::DivFOp>(builder, ones, bCloned);
    auto aDownstream = createGradOp<arith::MulFOp>(builder, aLocal, upstream);
    maybeAccumulateGrad(a, aDownstream, gradMap, builder);

    markAllVisited(builder, visitedType::Inserted, aDownstream, ones, aLocal);

    // (4) differentiate rhs

    // b local

    // auto two = createGradOp<arith::ConstantOp>(builder, divfOp.getType(), builder.getF32FloatAttr(2.0));
    auto pow = createGradOp<arith::MulFOp>(builder, bCloned, bCloned);
    auto div = createGradOp<arith::DivFOp>(builder, aCloned, pow);
    auto neg = createConstantTensor(builder, currentNodeName, div.getType(), -1.0);
    auto bLocal = createGradOp<arith::MulFOp>(builder, neg, div);
    auto bDownstream = createGradOp<arith::MulFOp>(builder, bLocal, upstream);
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
    auto sinOp = createGradOp<math::SinOp>(builder, xCloned);
    auto negOne = createConstantTensor(builder, currentNodeName, upstream.getType(), -1.0);
    auto negSin = createGradOp<arith::MulFOp>(builder, negOne, sinOp);
    auto xDownstream = createGradOp<arith::MulFOp>(builder, negSin, upstream);

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
    auto cosOp = createGradOp<math::CosOp>(builder, xCloned);
    auto xDownstream = createGradOp<arith::MulFOp>(builder, cosOp, upstream);

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
    auto two = createConstantTensor(builder, currentNodeName, upstream.getType(), 2.0);
    auto twoSqrtX = createGradOp<arith::MulFOp>(builder, sqrtResultCloned, two);
    auto one = createConstantTensor(builder, currentNodeName, upstream.getType(), 1.0);
    auto localGrad = createGradOp<arith::DivFOp>(builder, one, twoSqrtX);
    auto xDownstream = createGradOp<arith::MulFOp>(builder, localGrad, upstream);

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
    auto one = createConstantTensor(builder, currentNodeName, upstream.getType(), 1.0);
    auto localGrad = createGradOp<arith::DivFOp>(builder, one, xCloned);
    auto xDownstream = createGradOp<arith::MulFOp>(builder, localGrad, upstream);

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
    auto xDownstream = createGradOp<arith::MulFOp>(builder, expResultCloned, upstream);

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
      auto processedUpstreamOp = createGradOp<arith::TruncFOp>(builder, targetType, upstream);
      processedUpstream = processedUpstreamOp->getResult(0);
      markVisited(builder, visitedType::Inserted, processedUpstreamOp);
    }
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    // For matmul C = A * B + acc
    // dA = dC * B^T
    // dB = A^T * dC
    // dacc = dC (gradient flows directly to accumulator)

    std::vector<int32_t> transOrder = {1, 0};
    auto bTrans = createGradOp<triton::TransOp>(
        builder,
        bCloned.getType(),
        bCloned,
        builder.getDenseI32ArrayAttr(transOrder));

    // Compute gradient for A: dA = dC * B^T
    auto gradA = createGradOp<triton::DotOp>(
        builder,
        a.getType(),                  // Result type should match A's type
        processedUpstream,                     // dC
        bTrans,                       // B^T
        createConstantTensor(builder, currentNodeName, a.getType(), 0.0), // zero accumulator
        mmOp.getInputPrecision(),
        mmOp.getMaxNumImpreciseAcc());

    maybeAccumulateGrad(a, gradA, gradMap, builder);


    auto aTrans = createGradOp<triton::TransOp>(
        builder,
        aCloned.getType(),
        aCloned,
        builder.getDenseI32ArrayAttr(transOrder));

    // Compute gradient for B: dB = A^T * dC
    auto gradB = createGradOp<triton::DotOp>(
        builder,
        b.getType(),                  // Result type should match B's type
        aTrans,                       // A^T
        processedUpstream,                     // dC
        // todo-high: or accumulate into here (instead of maybeAccumulateGrad)
        createConstantTensor(builder, currentNodeName, b.getType(), 0.0), // zero accumulator
        mmOp.getInputPrecision(),
        mmOp.getMaxNumImpreciseAcc());

    maybeAccumulateGrad(b, gradB, gradMap, builder);

    // Compute gradient for C (accumulator): dC = dOut
    // The gradient of the accumulator is just the upstream gradient
    maybeAccumulateGrad(c, upstream, gradMap, builder);

    markAllVisited(builder, visitedType::Inserted, gradA, gradB, bTrans, aTrans);
  }

  void handleMaxBackward(arith::MaxNumFOp maxOp, OpBuilder &builder,
                            llvm::DenseMap<Value, Value> &gradMap, IRMapping &origToCloned) {

      Value upstream = getUpstreamGrad(maxOp->getResult(0), gradMap);
      setInsertionPointAfterLastUse(upstream, builder);

      // Get both input tensors
      Value lhs = maxOp->getOperand(0);
      Value rhs = maxOp->getOperand(1);

      // Get the inputs from the forward graph
      Value lhsCloned = origToCloned.lookup(lhs);
      Value rhsCloned = origToCloned.lookup(rhs);
      Value max = origToCloned.lookup(maxOp->getResult(0));

      // For max(a, b), gradient flows only through the maximum element(s)
      // If a > b, all gradient goes to a
      // If b > a, all gradient goes to b
      // If a == b, gradient is split between a and b (here we give it all to both and rely on maybeAccumulateGrad to handle duplicates)

      // Create masks for LHS: mask_lhs = (lhs >= rhs)
      auto cmpLhsOp = createGradOp<arith::CmpFOp>(
          builder,
          arith::CmpFPredicate::OGE,  // ordered greater than or equal
          lhsCloned,
          rhsCloned);

      // For reduce_max, gradient only flows through the maximum element(s)
      // We need to create a mask where elements equal to the maximum get the gradient

      // Create masks for RHS: mask_rhs = (rhs > lhs)
      auto cmpRhsOp = createGradOp<arith::CmpFOp>(
          builder,
          arith::CmpFPredicate::OGT,  // ordered greater than
          rhsCloned,
          lhsCloned);

      // Create a mask where elements equal to the max get 1.0, others get 0.0
      // Compare input with the broadcasted max value

      // Convert boolean masks to float masks (1.0 where true, 0.0 where false)
      auto oneConst = createConstantTensor(builder, currentNodeName, lhs.getType(), 1.0);
      auto zeroConst = createConstantTensor(builder, currentNodeName, lhs.getType(), 0.0);

      auto floatMaskLhs = createGradOp<arith::SelectOp>(
          builder,
          lhs.getType(),
          cmpLhsOp,
          oneConst,
          zeroConst);

      auto floatMaskRhs = createGradOp<arith::SelectOp>(
          builder,
          rhs.getType(),
          cmpRhsOp,
          oneConst,
          zeroConst);

      // Create a broadcast of the upstream gradient if needed
      Value upstreamBroadcast = createBroadcastOrSplat(
          upstream,
          lhs.getType(),
          currentNodeName,
          builder);

      // Compute Downstream grad
      // Multiply the masks by the upstream gradient
      auto maskedGradLhs = createGradOp<arith::MulFOp>(
          builder,
          floatMaskLhs,
          upstreamBroadcast);

      auto maskedGradRhs = createGradOp<arith::MulFOp>(
          builder,
          floatMaskRhs,
          upstreamBroadcast);

      // Propagate the masked gradients to the inputs
      maybeAccumulateGrad(lhs, maskedGradLhs, gradMap, builder);
      maybeAccumulateGrad(rhs, maskedGradRhs, gradMap, builder);

      markAllVisited(builder, visitedType::Inserted, cmpLhsOp, cmpRhsOp,
                     floatMaskLhs, floatMaskRhs, oneConst, zeroConst,
                     maskedGradLhs, maskedGradRhs);
  }


  void handleReduceBackward(triton::ReduceOp reduceOp, OpBuilder &builder,
                            llvm::DenseMap<Value, Value> &gradMap, IRMapping &origToCloned) {
    if (DEBUG_PRINTS) llvm::errs() << "visiting tt.reduce op\n";

    // Get the upstream gradient
    //  getResult retruns result_range
    // answer-now: again I did the same mistake as before. triton::ReduceOp (a specific subclass of Operation) for some reason does not have getReuslt method attached to it -- so what you should instead is use -> syntax on it to dispatch to its parent (generic Operation) which has getResult(1) implemented
    //    same for getOperand(): reduceOp.getOperand(0) -- ERRORS OUT.    reduceOp->getOperand(0) -- works!
    Value upstream = getUpstreamGrad(reduceOp->getResult(0), gradMap);
    setInsertionPointAfterLastUse(upstream, builder);

    // Check that this is a sum reduction (contains only single node, e.g. arith.addf)
    Operation *combiner = reduceOp.getSingleCombiner();
    if (!combiner) {
      llvm::report_fatal_error("Combiner is missing\n");
    }

    // Get the input tensor and its type
    Value input = reduceOp->getOperand(0);

    // Get the input from the forward graph
    Value inputCloned = origToCloned.lookup(input);
    Value reducedValue = origToCloned.lookup(reduceOp->getResult(0));

    if (isa<arith::AddFOp>(combiner)) {

      // Get the axis being reduced
      // int32_t axis = reduceOp.getAxis(); 

      // Create a broadcast of the upstream gradient along the reduced axis
      // For reduce_sum, gradient is uniform broadcast of upstream gradient
      Value downstreamGrad = createBroadcastOrSplat(
          upstream, // castToSameEncoding(upstream, input, builder);
          input.getType(),
          currentNodeName,
          builder);

      // Propagate the gradient to the input
      maybeAccumulateGrad(input, downstreamGrad, gradMap, builder);

    } else if (isa<arith::MaxNumFOp>(combiner)) {
      // For reduce_max, gradient only flows through the maximum element(s)
      // We need to create a mask where elements equal to the maximum get the gradient

      // Create a broadcast of the reduced value (the maximum)
      Value maxBroadcast = createBroadcastOrSplat(
          reducedValue,
          input.getType(),
          currentNodeName,
          builder);

      // Create a mask where elements equal to the max get 1.0, others get 0.0
      // Compare input with the broadcasted max value
      auto cmpOp = createGradOp<arith::CmpFOp>(
          builder,
          arith::CmpFPredicate::OEQ,  // ordered equal
          inputCloned,
          maxBroadcast);

      // Convert boolean mask to float mask (1.0 where true, 0.0 where false)
      auto floatType = cast<ShapedType>(upstream.getType()).getElementType();
      auto oneConst = createConstantTensor(builder, currentNodeName, input.getType(), 1.0);
      auto zeroConst = createConstantTensor(builder, currentNodeName, input.getType(), 0.0);

      auto floatMask = createGradOp<arith::SelectOp>(
          builder,
          input.getType(),
          cmpOp,
          oneConst,
          zeroConst);

      // Create a broadcast of the upstream gradient
      Value upstreamBroadcast = createBroadcastOrSplat(
          upstream,   // castToSameEncoding(upstream, input, builder)
          input.getType(),
          currentNodeName,
          builder);

      // Multiply the mask by the upstream gradient
      auto maskedGrad = createGradOp<arith::MulFOp>(
          builder,
          floatMask,
          upstreamBroadcast);

      // Propagate the masked gradient to the input
      maybeAccumulateGrad(input, maskedGrad, gradMap, builder);

      markAllVisited(builder, visitedType::Inserted, cmpOp, floatMask,
                    oneConst, zeroConst, maskedGrad);

    } else {
        llvm::report_fatal_error("Only sum / add reduction are supported\n");
    }
  }


  void handleExtFBackward(arith::ExtFOp extfOp, OpBuilder &builder,
                        llvm::DenseMap<Value, Value> &gradMap) {
    if (DEBUG_PRINTS) llvm::errs() << "visiting arith.extf op\n";

    Value upstream = getUpstreamGrad(extfOp, gradMap);
    setInsertionPointAfterLastUse(upstream, builder);

    Value x = extfOp.getOperand();

    // For extf, the gradient is simply the truncation of the upstream gradient
    // to the precision of the input
    auto truncOp = createGradOp<arith::TruncFOp>(
        builder,
        x.getType(),  // Result type should match the original input type
        upstream      // Upstream gradient with the extended type
    );

    // Propagate the gradient to the input
    maybeAccumulateGrad(x, truncOp, gradMap, builder);

    markVisited(builder, visitedType::Inserted, truncOp);
  }


  void handleSubfBackward(arith::SubFOp subfOp, OpBuilder &builder,
                        llvm::DenseMap<Value, Value> &gradMap) {
    if (DEBUG_PRINTS) llvm::errs() << "visiting arith.subf op\n";

    Value upstream = getUpstreamGrad(subfOp, gradMap);
    setInsertionPointAfterLastUse(upstream, builder);

    // For subtraction z = x - y
    Value lhs = subfOp.getOperand(0);
    Value rhs = subfOp.getOperand(1);

    // dz/dx = 1, so just pass through the upstream gradient
    maybeAccumulateGrad(lhs, upstream, gradMap, builder);

    // dz/dy = -1, so negate the upstream gradient
    auto negOne = createConstantTensor(builder, currentNodeName, upstream.getType(), -1.0);
    auto negUpstream = createGradOp<arith::MulFOp>(builder, upstream, negOne);
    maybeAccumulateGrad(rhs, negUpstream, gradMap, builder);

    markAllVisited(builder, visitedType::Inserted, negUpstream, negOne);
  }


  // For a select operation like %result = arith.select %condition, %true_value, %false_value, the gradient flows as follows:
  // 1) The condition doesn't receive any gradient since it's a boolean predicate
  // 2) The true value receives the upstream gradient only where the condition is true
  // 3) The false value receives the upstream gradient only where the condition is false
  // The implementation creates two masked gradients:
  //  - For the true value: select(condition, upstream_gradient, zeros)
  //  - For the false value: select(NOT condition, upstream_gradient, zeros)
  void handleSelectBackward(arith::SelectOp selectOp, OpBuilder &builder,
                          llvm::DenseMap<Value, Value> &gradMap, IRMapping &origToCloned) {
    if (DEBUG_PRINTS) llvm::errs() << "visiting arith.select op\n";

    Value upstream = getUpstreamGrad(selectOp, gradMap);
    setInsertionPointAfterLastUse(upstream, builder);

    // Get operands
    Value condition = selectOp.getCondition();
    Value trueValue = selectOp.getTrueValue();
    Value falseValue = selectOp.getFalseValue();

    // Get the cloned condition from the forward pass
    Value conditionCloned = origToCloned.lookup(condition);

    // For select(cond, true_val, false_val):
    // - No gradient for condition (it's boolean/predicate)
    // - For true_val: gradient flows only where condition is true
    // - For false_val: gradient flows only where condition is false

    // Create masked gradients for true value
    auto trueGrad = createGradOp<arith::SelectOp>(
        builder,
        upstream.getType(),
        conditionCloned,    // original condition
        upstream,           // upstream gradient where condition is true
        createConstantTensor(builder, currentNodeName, upstream.getType(), 0.0) // zeros where condition is false
    );

    // Propagate gradient to the true value operand
    maybeAccumulateGrad(trueValue, trueGrad, gradMap, builder);

    // Create masked gradients for false value
    // First, create the negated condition
    auto notCond = createGradOp<arith::SelectOp>(
        builder,
        conditionCloned.getType(),
        conditionCloned,
        createConstantBoolTensor(builder, currentNodeName, conditionCloned.getType(), false),
        createConstantBoolTensor(builder, currentNodeName, conditionCloned.getType(), true)
    );

    auto falseGrad = createGradOp<arith::SelectOp>(
        builder,
        upstream.getType(),
        notCond,           // negated condition
        upstream,          // upstream gradient where condition is false
        createConstantTensor(builder, currentNodeName, upstream.getType(), 0.0) // zeros where condition is true
    );

    // Propagate gradient to the false value operand
    maybeAccumulateGrad(falseValue, falseGrad, gradMap, builder);

    // Mark all created operations as visited
    markAllVisited(builder, visitedType::Inserted, trueGrad, notCond, falseGrad);
  }

  void handleBroadcastBackward(triton::BroadcastOp broadcastOp, OpBuilder &builder,
                            llvm::DenseMap<Value, Value> &gradMap) {
    if (DEBUG_PRINTS) llvm::errs() << "visiting tt.broadcast op\n";

    Value input = broadcastOp.getOperand();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto resultType = dyn_cast<RankedTensorType>(broadcastOp.getType());
    if (!inputType || !resultType) {
      llvm::report_fatal_error("Expected ranked tensor types for broadcast op");
    }

    // my previous kernels also had broadcast ops, but I didn't match on them bc they
    // were using int inputs (computing some pointer offsets) not actual tensor values -- but
    // this attention kernel is different, bc broadcast is used there on the data (rather on the idxs)
    bool isFloat = isa<FloatType>(inputType.getElementType());
    if (!isFloat){
      llvm::errs() << "[handleBroadcastBackward] exiting early, input is not a Float\n";
      return;
    }
    llvm::errs() << "[handleBroadcastBackward] input a Float, adding grad\n";

    Value upstream = getUpstreamGrad(broadcastOp, gradMap);
    // setInsertionPointAfterLastUse(upstream, builder);

    // the gradient is the reduction (sum) of the upstream gradient
    // along the dimensions that were broadcasted
    auto inputShape = inputType.getShape();
    auto resultShape = resultType.getShape();

    // Find dimensions that were broadcasted (where input dim is 1 and result dim is > 1)

    // "broadcast changes one or more dimensions", but reduce (which is the grad
    // of broadcast) only supports one dim at a time -- so need the for loop
    //
    // The implementation reduces each dimension separately due to the design of Triton's ReduceOp, which only supports reducing along a single axis at a time. The ReduceOp constructor takes a single integer parameter for the axis to reduce, not a list of dimensions -- build(..., int axis).
    // While conceptually we're computing the sum across all broadcasted dimensions, in MLIR/Triton couldn't find single operation that can reduce along multiple dimensions at once.
    for (int i = 0; i < inputShape.size(); i++) {

      // if this is the dim that was expanded (during fwd)
      if (inputShape[i] == 1 && resultShape[i] > 1) {

        setInsertionPointAfterLastUse(upstream, builder);

        // Sum along this dimension
        auto reduceOp = createGradOp<triton::ReduceOp>(
            builder,
            upstream,
            i); // axis

        // Add a block to the region first
        auto &combineRegion = reduceOp.getCombineOp();
        auto *combinerBlock = builder.createBlock(&combineRegion);

        // Add arguments
        // the block itself is typically created automatically because it has the OpTrait::SingleBlock trait, but the arguments aren't automatically added.
        Type elemType = dyn_cast<ShapedType>(upstream.getType()).getElementType();
        combinerBlock->addArgument(elemType, broadcastOp.getLoc());
        combinerBlock->addArgument(elemType, broadcastOp.getLoc());

        // insertion point for ops within the reduceOp itself
        // this is kind of inner (builder for the ops inside the reduceOp)
        auto blockBuilder = OpBuilder::atBlockBegin(combinerBlock);
        // note: bc these below don't use createGradOp helper, required to location (currentNodeName) explicitly
        auto sum = blockBuilder.create<arith::AddFOp>(
            currentNodeName,
            combinerBlock->getArgument(0),
            combinerBlock->getArgument(1));

        blockBuilder.create<triton::ReduceReturnOp>(currentNodeName, sum.getResult());


        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // wt the ExpandDimsOp below, we'd completely collapse the 0-th dim, but instead need
        // to preserve it as singleton dim (1) -- bc shape of grad needs to match shape of the original input
        //    iterating over op %150 = tt.broadcast %149 : tensor<16x1xf32> -> tensor<16x16xf32>

        //question-now:
        // setInsertionPointAfterLastUse(upstream, builder);
        // insertion point in the outer graph -- for the expand to be directly after reduceOp
        builder.setInsertionPointAfter(reduceOp);

        // Note: expand_dims needs to outside of the reduce
        // must return the scalar sum directly in the reduce.return op,
        // don't try to use expand_dims inside the reduce region
        auto expand = createGradOp<triton::ExpandDimsOp>(
            builder,
            reduceOp->getResult(0),
            i); // axis

        // expand_dims needs to be outside of the reduce
        //    %130 = "tt.reduce"(%129) <{axis = 1 : i32}> ({
        //    ^bb0(%arg23: f32, %arg24: f32):
        //      %184 = "arith.addf"(%arg23, %arg24) : (f32, f32) -> f32
        //      %185 = "tt.expand_dims"(%130) <{axis = 1 : i32}> : (tensor<16xf32>) -> tensor<16x1xf32>
        //      "tt.reduce.return"(%185) : (f32) -> ()
        //    }): (tensor<16x16xf32>) -> tensor<16xf32>

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        // Update upstream for the next dim: apply expand_dims AFTER the reduce operation, outside its region
        // so that ouput of this entire for loop -- is a sequence of:
        //  reduce(inner_fn=add, dim=0) -> reduce(inner_fn=add, dim=1) -> reduce(inner_fn=add, dim=2)
        upstream = expand->getResult(0);

        // question-now: I think don't need to explicitly mark the inner op (addf) as well?
        markAllVisited(builder, visitedType::Inserted, reduceOp, expand);
      }
    }

    // todo-now: grad accum logic needs to be outside of the reuce
    // %108 = "tt.reduce"(%97) <{axis = 1 : i32}> ({
    // ^bb0(%arg27: f32, %arg28: f32):
    //   %188 = "arith.addf"(%arg27, %arg28) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    //   "tt.reduce.return"(%188) : (f32) -> ()
    //   %189 = "tt.expand_dims"(%108) <{axis = 1 : i32}> : (tensor<16xf32>) -> tensor<16x1xf32>
    //   %190 = "arith.mulf"(%66, %189) <{fastmath = #arith.fastmath<none>}> {autogradVisited = true, isInserted = true} : (tensor<16x1xf32>, tensor<16x1xf32>) -> tensor<16x1xf32>
    //   %191 = "arith.mulf"(%67, %189) <{fastmath = #arith.fastmath<none>}> {autogradVisited = true, isInserted = true} : (tensor<16x1xf32>, tensor<16x1xf32>) -> tensor<16x1xf32>
    // })

    // Reduce along all necessary dimensions
    // Only then propagate the final reduced gradient to the input

    // setInsertionPointAfterLastUse(initialUpstream, builder);
    maybeAccumulateGrad(input, upstream, gradMap, builder);
  }

  void handleLog2Backward(math::Log2Op log2Op, OpBuilder &builder,
                          llvm::DenseMap<Value, Value> &gradMap, IRMapping &origToCloned) {
    if (DEBUG_PRINTS) llvm::errs() << "visiting math.log2 op\n";

    Value upstream = getUpstreamGrad(log2Op, gradMap);
    setInsertionPointAfterLastUse(upstream, builder);

    Value x = log2Op.getOperand();
    Value xCloned = origToCloned.lookup(x);

    // derivative of log2(x) is 1/(x*ln(2))
    // ln(2) ≈ 0.693147
    auto ln2 = createConstantTensor(builder, currentNodeName, upstream.getType(), 0.693147);
    auto xTimesLn2 = createGradOp<arith::MulFOp>(builder, xCloned, ln2);
    auto one = createConstantTensor(builder, currentNodeName, upstream.getType(), 1.0);
    auto localGrad = createGradOp<arith::DivFOp>(builder, one, xTimesLn2);
    auto downstreamGrad = createGradOp<arith::MulFOp>(builder, localGrad, upstream);

    maybeAccumulateGrad(x, downstreamGrad, gradMap, builder);

    markAllVisited(builder, visitedType::Inserted, ln2, xTimesLn2, one, localGrad, downstreamGrad);
  }

  void handleExp2Backward(math::Exp2Op exp2Op, OpBuilder &builder,
                          llvm::DenseMap<Value, Value> &gradMap, IRMapping &origToCloned) {
    if (DEBUG_PRINTS) llvm::errs() << "visiting math.exp2 op\n";

    Value upstream = getUpstreamGrad(exp2Op, gradMap);
    setInsertionPointAfterLastUse(upstream, builder);

    Value x = exp2Op.getOperand();
    Value resultCloned = origToCloned.lookup(exp2Op);

    // derivative of exp2(x) is ln(2) * exp2(x)
    // ln(2) ≈ 0.693147
    auto ln2 = createConstantTensor(builder, currentNodeName, upstream.getType(), 0.693147);
    auto localGrad = createGradOp<arith::MulFOp>(builder, ln2, resultCloned);
    auto downstreamGrad = createGradOp<arith::MulFOp>(builder, localGrad, upstream);

    maybeAccumulateGrad(x, downstreamGrad, gradMap, builder);

    markAllVisited(builder, visitedType::Inserted, ln2, localGrad, downstreamGrad);
  }


  // tt.expand_dims just adds a singleton dimension (1) to the shape without duplicating data, unlike broadcast which replicates values.
  // For the backward pass of expand_dims, we don't need a reduction operation - we just need to reshape by removing the singleton dimension
  void handleExpandDimsBackward(triton::ExpandDimsOp expandDimsOp, OpBuilder &builder,
                                llvm::DenseMap<Value, Value> &gradMap) {
    if (DEBUG_PRINTS) llvm::errs() << "visiting tt.expand_dims op\n";

    // Get the input tensor and axis that was expanded
    Value input = expandDimsOp.getOperand();
    int axis = expandDimsOp.getAxis();

    // my previous kernels also had broadcast ops, but I didn't match on them bc they
    // were using int inputs (computing some pointer offsets) not actual tensor values -- but
    // this attention kernel is different, bc broadcast is used there on the data (rather on the idxs)
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    bool isFloat = isa<FloatType>(inputType.getElementType());
    if (!isFloat){
      llvm::errs() << "[handleExpandDimsBackward] exiting early, input is not a Float\n";
      return;
    }
    llvm::errs() << "[handleExpandDimsBackward] input a Float, adding grad\n";


    Value upstream = getUpstreamGrad(expandDimsOp, gradMap);
    setInsertionPointAfterLastUse(upstream, builder);


    // The backward operation for expand_dims is to remove the singleton dimension
    // This is essentially a reshape operation
    auto reshapeOp = createGradOp<triton::ReshapeOp>(
        builder,
        input.getType(),  // Result type should match the original input type
        upstream,         // Upstream gradient with the expanded dimension
        false,            // allow_reorder: false to maintain the element order
        false             // efficient_layout: false (default value)
    );

    // Propagate the gradient to the input
    maybeAccumulateGrad(input, reshapeOp, gradMap, builder);

    markVisited(builder, visitedType::Inserted, reshapeOp);
  }

  void handleTransBackward(triton::TransOp transOp, OpBuilder &builder,
                          llvm::DenseMap<Value, Value> &gradMap) {
    if (DEBUG_PRINTS) llvm::errs() << "visiting tt.trans op\n";

    // Get the upstream gradient
    Value upstream = getUpstreamGrad(transOp, gradMap);
    setInsertionPointAfterLastUse(upstream, builder);

    // Get the input tensor and permutation order
    Value input = transOp.getSrc();
    ArrayRef<int32_t> order = transOp.getOrder();

    // Check if the tensor contains floating point data
    auto inputType = dyn_cast<RankedTensorType>(input.getType());

    bool isFloat = isa<FloatType>(inputType.getElementType());
    if (!isFloat) {
      llvm::errs() << "[handleTransBackward] exiting early, input is not a Float\n";
      return;
    }
    llvm::errs() << "[handleTransBackward] input is a Float, adding gradient\n";

    // For a transpose operation, the gradient is the transpose of the upstream gradient
    // with the same permutation (which is its own inverse for 2D case)
    // For more complex cases, the permutation is still the same because we're undoing
    // the original permutation

    auto transGrad = createGradOp<triton::TransOp>(
        builder,
        input.getType(),  // Result type should match the original input type
        upstream,         // Upstream gradient
        builder.getDenseI32ArrayAttr(order)  // Same permutation order as the forward pass
    );

    // Propagate the gradient to the input
    maybeAccumulateGrad(input, transGrad, gradMap, builder);

    markVisited(builder, visitedType::Inserted, transGrad);
  }

  void handleSplatBackward(triton::SplatOp splatOp, OpBuilder &builder,
                          llvm::DenseMap<Value, Value> &gradMap) {
    if (DEBUG_PRINTS) llvm::errs() << "visiting tt.splat op\n";


    // Get input scalar value
    Value scalar = splatOp.getSrc();

    // Check if the scalar is a float type
    bool isFloat = isa<FloatType>(scalar.getType());
    if (!isFloat) {
      llvm::errs() << "[handleSplatBackward] exiting early, input is not a Float\n";
      return;
    }

    Value upstream = getUpstreamGrad(splatOp, gradMap);
    setInsertionPointAfterLastUse(upstream, builder);

    // For splat operations, gradient of scalar = sum of all elements in upstream gradient
    // We need to reduce all dimensions to get a scalar

    // Get the result tensor type
    auto resultType = dyn_cast<RankedTensorType>(splatOp.getType());
    // auto resultShape = resultType.getShape();

    // Start with upstream gradient
    Value currentGrad = upstream;

    // Reduce along each dimension to get a scalar
    for (int i = 0; i < resultType.getRank(); i++) {

      builder.setInsertionPointAfterValue(currentGrad);

      // We always reduce dimension 0 since the tensor shape changes after each reduction
      auto reduceOp = createGradOp<triton::ReduceOp>(
          builder,
          currentGrad,
          0); // Always reduce first dimension

      // Add block and arguments
      auto &combineRegion = reduceOp.getCombineOp();
      auto *combinerBlock = builder.createBlock(&combineRegion);

      // Add arguments
      Type elemType = dyn_cast<ShapedType>(currentGrad.getType()).getElementType();
      combinerBlock->addArgument(elemType, splatOp.getLoc());
      combinerBlock->addArgument(elemType, splatOp.getLoc());

      // this is kind of inner (builder for the ops inside the reduceOp)
      auto blockBuilder = OpBuilder::atBlockBegin(combinerBlock);
      auto sum = blockBuilder.create<arith::AddFOp>(
          currentNodeName,
          combinerBlock->getArgument(0),
          combinerBlock->getArgument(1));

      blockBuilder.create<triton::ReduceReturnOp>(currentNodeName, sum.getResult());

      // Update gradient for next reduction
      currentGrad = reduceOp->getResult(0);

      // Mark all operations in the reduction as visited
      // No need to mark operations inside the block as they're contained in the reduce op
      markVisited(builder, visitedType::Inserted, reduceOp);
    }

    maybeAccumulateGrad(scalar, currentGrad, gradMap, builder);
  }


}; // ConvertTritonToAutodiff stuct

} // private namespace
} // namespace triton
} // namespace mlir

