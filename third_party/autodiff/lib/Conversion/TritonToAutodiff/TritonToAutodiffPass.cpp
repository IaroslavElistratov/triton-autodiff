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
#include "autodiff/include/Conversion/TritonToAutodiff/Handlers.h"
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
        lastBwdOp = handleStoreBackward(storeOp, lastBwdOp, this);
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
        handleMatmulBackward(mmOp, this);
      } else if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)){
        handleReduceBackward(reduceOp, this);
      } else if (auto broadcastOp = dyn_cast<triton::BroadcastOp>(op)){
        handleBroadcastBackward(broadcastOp, this);
      } else if (auto expandDimsOp = dyn_cast<triton::ExpandDimsOp>(op)){
        handleExpandDimsBackward(expandDimsOp, this);
      } else if (auto transOp = dyn_cast<triton::TransOp>(op)){
        handleTransBackward(transOp, this);
      } else if (auto splatOp = dyn_cast<triton::SplatOp>(op)){
        handleSplatBackward(splatOp, this);


      // arith ops
      } else if (auto addfOp = dyn_cast<arith::AddFOp>(op)){
        handleAddBackward(addfOp, this);
      } else if (auto mulfOp = dyn_cast<arith::MulFOp>(op)){
        handleMulBackward(mulfOp, this);
      } else if (auto divfOp = dyn_cast<arith::DivFOp>(op)){
        handleDivBackward(divfOp, this);
      } else if (auto truncfOp = dyn_cast<arith::TruncFOp>(op)){
        handleTruncfBackward(truncfOp, this);
      } else if (auto constantOp = dyn_cast<arith::ConstantOp>(op)){
        if (DEBUG_PRINTS) llvm::errs() << "visiting arith.constant op\n";
      } else if (auto extfOp = dyn_cast<arith::ExtFOp>(op)){
        handleExtFBackward(extfOp, this);
      } else if (auto subfOp = dyn_cast<arith::SubFOp>(op)){
        handleSubfBackward(subfOp, this);
      } else if (auto selectOp = dyn_cast<arith::SelectOp>(op)){
        handleSelectBackward(selectOp, this);
      } else if (auto maxOp = dyn_cast<arith::MaxNumFOp>(op)){
        handleMaxBackward(maxOp, this);

      // math ops
      } else if (auto cosOp = dyn_cast<math::CosOp>(op)){
        handleCosBackward(cosOp, this);
      } else if (auto sinOp = dyn_cast<math::SinOp>(op)){
        handleSinBackward(sinOp, this);
      } else if (auto sqrtOp = dyn_cast<math::SqrtOp>(op)){
        handleSqrtBackward(sqrtOp, this);
      } else if (auto logOp = dyn_cast<math::LogOp>(op)){
        handleLogBackward(logOp, this);  // For natural logarithm (base e): The derivative of ln(x) is 1/x
      } else if (auto log2Op = dyn_cast<math::Log2Op>(op)){
        handleLog2Backward(log2Op, this); // For logarithm base 2: The derivative of log₂(x) is 1/(x·ln(2))
      } else if (auto expOp = dyn_cast<math::ExpOp>(op)){
        handleExpBackward(expOp, this);
      } else if (auto exp2Op = dyn_cast<math::Exp2Op>(op)){
        handleExp2Backward(exp2Op, this);
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
        handleLoadBackward(loadOp, func, this);
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





}; // ConvertTritonToAutodiff stuct

} // private namespace
} // namespace triton
} // namespace mlir

