#include "mlir/IR/AsmState.h"          // registerAsmPrinterCLOptions
#include "llvm/Support/CommandLine.h"

#include "mlir/IR/Builders.h"
#include <memory>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

// for reverse topo sort
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Block.h"
#include "llvm/ADT/SetVector.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "autodiff/include/Dialect/Autodiff/IR/Dialect.h"
#include "autodiff/include/Conversion/TritonToAutodiff/Passes.h"
#include "autodiff/include/Conversion/TritonToAutodiff/Handlers.h"
#include "autodiff/include/Conversion/TritonToAutodiff/HandlerForOp.h"
#include "autodiff/include/Conversion/TritonToAutodiff/Utils.h"
#include "autodiff/include/Conversion/TritonToAutodiff/UtilsIO.h"
#include "llvm/ADT/APSInt.h"
#include <numeric>

#include "llvm/Support/Debug.h"

namespace mlir {
namespace triton {


// When GEN_PASS_DEF_CONVERTTRITONTOAUTODIFF is defined, the generated Passes.h.inc file includes the function definition (implementation) of createConvertTritonToAutodiff
// Include the DEFINITIONS in exactly ONE .cpp file
#define GEN_PASS_DEF_CONVERTTRITONTOAUTODIFF
#include "autodiff/include/Conversion/TritonToAutodiff/Passes.h.inc"


  // main function
  void ConvertTritonToAutodiff::runOnOperation() {

    enableNameLocSSA();

    // grab the module (IOW root) op
    auto mod = getOperation();

    auto funcOps = mod.getOps<triton::FuncOp>();
    if (funcOps.empty() || std::next(funcOps.begin()) != funcOps.end()) {
      llvm::report_fatal_error("expect exactly one Triton function in the module");
      return;
    }
    triton::FuncOp func = *funcOps.begin();


    // ----------------
    // init pass state
    // ----------------
    // moved member variables init to outside of rewriteIntoBackward;
    lastFwdOp = nullptr;
    builder.emplace(func.getContext());
    gradMap.clear();     // init as empty
    origToCloned.clear(); // init as empty

    // -------------------------------------------
    // add gradient pointer arguments to the func
    // --------------------------------------------
    if (DEBUG_PRINTS) llvm::errs() << "\n\n\n============ adding grad pointers ============\n\n\n";
    ptrToAddedPtrMap = addPointerArgsToFunction(func);

    // ---------------------------------------
    // rewrite the block (possibly recursive)
    // ---------------------------------------
    // no need for walk, since all my current use cases involve only a single ForOp
    rewriteIntoBackward(func.getBody().front());
  }

  // todo-low: mv to utils, no need to be a pass class member
  void ConvertTritonToAutodiff::enableNameLocSSA() {
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

  // walk the IR backward, rewrite each operation with its corresponding backward function;
  // Accepts an MLIR Block rather than the FuncOp itself to enable
  // recursive application on nested regions such as loop bodies
  void ConvertTritonToAutodiff::rewriteIntoBackward(Block &block) {

    Block *blockPtr = &block;

    // // todo: enable only for "inline" pattern
    // unrollAllForOps(func);
    // // func.getBody().front().dump();
    // // exit(1);
    // if (DEBUG_PRINTS) {
    //   llvm::errs() << "flattening for loop:\n";
    //   func.getBody().front().print(llvm::errs());
    //   llvm::errs() << "\n";
    // }

    // error happening because Value (which is the type I'm trying to put into std::map) does not have move interface
    // (< comparitor), which it appers the impl of map is trying ot use to compare eleemtns of the map

    // copied from: llvm-project/mlir/lib/Dialect/Linalg/Transforms/Hoisting.cpp
    SetVector<Operation *> forwardSlice;
    getForwardSlice(blockPtr->getParentOp(), &forwardSlice);


    // todo-now:
    //  do not need to clone the loop for my "for-loop-no-unroll" -- bc the only ops after ForOp are Store[s];
    //  still do need to clone other nodes outside/before of the for-op though
    // Seems, this is an optimization (can do even w cloning, but less efficient) -- so leave it for later

    if (DEBUG_PRINTS) llvm::errs() << "\n\n\n============ cloning ============\n\n\n";

    // because these marked as visited, you will not match
    // them in your loop below, and thus you will not re-write
    // them -- so effectively this cloned is your *Forward* graph

    builder->setInsertionPointToStart(blockPtr);

    // Clones the entire fwd graph (not just a single subgraph leading from the last fwd op)
    DenseSet<Operation*> visitedLoads;
    Operation *currFwdOp;
    // note: if rewriteIntoBackward was called form HandleForOp then lastFwdOp was populated
    //  and the comparison below will compare currFwdOp (from the current clone call)
    //  to the lastFwdOp (from the HandleForOp's clone that cloned from yeild) -- this is desirable
    for (Operation *op : llvm::reverse(forwardSlice)) {

      auto currStoreOp = dyn_cast<triton::StoreOp>(op);
      // todo: cleanup
      if (visitedLoads.contains(op) || !currStoreOp || op->getBlock() != blockPtr){
        continue;
      }

      if (DEBUG_PRINTS) llvm::errs() << "cloning nodes leading to op: " << *op << "\n";

      // note: important to pass the same map (origToCloned) -- so that the cloning logic
      //  does not re-clone nodes that are common between the subgraphs of nodes leading to different StoreOps
      currFwdOp = cloneSubtree(currStoreOp, origToCloned, *builder);
      // first condition is for the 1st iter -- to overwrite nullptr at least with some currFwdOp
      if (!lastFwdOp || !currFwdOp->isBeforeInBlock(lastFwdOp)){
        lastFwdOp = currFwdOp;
      }
    }

    if (DEBUG_PRINTS) {
      llvm::errs() << "after cloning:\n";
      blockPtr->print(llvm::errs());
    }


    // todo-now: use the lastFwdOp that forOp handler used, bc the above cloning may do nothing (in case there as no additional store[s] in the user fn) bc I laready copied averythign leading to yeild in handleForOp

    // the above mapping: original nodes -> inserted nodes.
    // To lookup intermideats in the cloned (aka cloned subgraph),
    // when iterating over the original subgraph (and re-writting that original subgraph with derivative formualrs)
    // I think I don't even need to reverse the mapping: can directly use it --
    // bc I'm iterating over the "original nodes" and want to figure out what "cloned" node does an original node refers to


    // // Walk all operations opaquely.
    // // todo: I think, you don't need topo sort if you're iterating in post order traversal (visit children before parent)
    // func->walk<WalkOrder::PostOrder>([&](Operation *op) {           // https://mlir.llvm.org/doxygen/classmlir_1_1Operation.html#a59740592240b950b8c8afcf4a2eb4113


    if (DEBUG_PRINTS) llvm::errs() << "\n\n\n============ loop over handleStore ============\n\n\n";

    // separate loop for handle store[s]
    for (Operation *op : llvm::reverse(forwardSlice)) {

      StoreOp storeOp = dyn_cast<triton::StoreOp>(op);
      auto visitedAttr = op->getAttrOfType<BoolAttr>("autogradVisited");
      // only process StoreOps that belong to the block I'm rewriting;
      // otherwise the below will try to handle tt.store from outside
      // the current block being re-written -- undesirable when
      // called rewriteIntoBackward on a loop body from handleForOp
      if (!storeOp || visitedAttr || op->getBlock() != blockPtr) {
          continue;
      }

      if (DEBUG_PRINTS) llvm::errs() << "\niterating over op " << *op << "\n";

      // todo: why currentNodeName
      NameLoc nodeName = createNodeName(op, "bwd_");
      // Store the name in the member variable for use in handlers
      currentNodeName = nodeName;

      // print only if changed
      std::string initialIR;
      if (DEBUG_PRINTS) {
        llvm::raw_string_ostream initialStream(initialIR);
        blockPtr->print(initialStream);
      }


      // todo: cleanup
      Operation *lastBwdOp = lastFwdOp;
      lastBwdOp = handleStoreBackward(storeOp, lastBwdOp, *this);


      if (DEBUG_PRINTS) {
        llvm::errs() << "\nIR after calling handleStoreBackward\n";
        std::string currentIR;
        llvm::raw_string_ostream currentStream(currentIR);
        blockPtr->print(currentStream);
        if (initialIR != currentIR){
          // llvm::errs() << blockPtr->print();
          // dump writes to std err, but I want these be in "sync" with my other prints --
          llvm::raw_ostream &os = llvm::errs();
          blockPtr->print(os);
        }
      }

    }

    if (DEBUG_PRINTS) llvm::errs() << "\n\n\n============ handleAllOps ============\n\n\n";

    // dreference pointer (handleAllOps expects reference)
    handleAllOps(*blockPtr);


    if (DEBUG_PRINTS) llvm::errs() << "\n\n\n============ loop over handleLoad ============\n\n\n";

    // Second pass: handle LoadOp operations
    // because its derivative (storeOp) destroyaes semantics of input args
    // for (Operation *op : llvm::reverse(forwardSlice)) {
    for (Operation &it : llvm::reverse(blockPtr->getOperations())) {
      // todo: temp -- convert reference to pointer (to avoid modifying the the places below -- e.g. dyn_cast)
      Operation *op = &it;

      if (op->getAttrOfType<BoolAttr>("autogradVisited")) {
          if (DEBUG_PRINTS) llvm::errs() << "Skipping visited" << "\n";
          continue;
      }

      NameLoc nodeName = createNodeName(op, "bwd_");
      // Store the name in the member variable for use in handlers
      currentNodeName = nodeName;

      if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
        handleLoadBackward(loadOp, *blockPtr, *this);
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


    if (DEBUG_PRINTS) llvm::errs() << "\n\n\n============ remove unmarked ops ============\n\n\n";

    // Final pass: remove unmarked operations
    blockPtr->walk<WalkOrder::PostOrder>([&](Operation *op) {

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

  } // rewriteIntoBackward function






  // [for-loop over handlers]
  // answer-now: abstracting this into a separate function allows me to call this fn standalone recursively from inside each ForOp handler
  void ConvertTritonToAutodiff::handleAllOps(
      Block &block // SetVector<Operation *> &forwardSlice
    ){


    // First pass: handle all operations except LoadOp
    // for (Operation *op : llvm::reverse(forwardSlice)) {
    for (Operation &it : llvm::reverse(block.getOperations())) {
      // todo: temp conver ref to pointer (alternative is modify all the places below (e.g. dyn_cast) to do this line below)
      // get a pointer to the reference
      Operation *op = &it;


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
      if (op->getBlock() != &block){
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
      block.print(initialStream);

      // triton ops
      if (auto mmOp = dyn_cast<triton::DotOp>(op)){
        handleMatmulBackward(mmOp, *this);
      } else if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)){
        handleReduceBackward(reduceOp, *this);
      } else if (auto broadcastOp = dyn_cast<triton::BroadcastOp>(op)){
        handleBroadcastBackward(broadcastOp, *this);
      } else if (auto expandDimsOp = dyn_cast<triton::ExpandDimsOp>(op)){
        handleExpandDimsBackward(expandDimsOp, *this);
      } else if (auto transOp = dyn_cast<triton::TransOp>(op)){
        handleTransBackward(transOp, *this);
      } else if (auto splatOp = dyn_cast<triton::SplatOp>(op)){
        handleSplatBackward(splatOp, *this);


      // arith ops
      } else if (auto addfOp = dyn_cast<arith::AddFOp>(op)){
        handleAddBackward(addfOp, *this);
      } else if (auto mulfOp = dyn_cast<arith::MulFOp>(op)){
        handleMulBackward(mulfOp, *this);
      } else if (auto divfOp = dyn_cast<arith::DivFOp>(op)){
        handleDivBackward(divfOp, *this);
      } else if (auto truncfOp = dyn_cast<arith::TruncFOp>(op)){
        handleTruncfBackward(truncfOp, *this);
      } else if (auto constantOp = dyn_cast<arith::ConstantOp>(op)){
        if (DEBUG_PRINTS) llvm::errs() << "visiting arith.constant op\n";
      } else if (auto extfOp = dyn_cast<arith::ExtFOp>(op)){
        handleExtFBackward(extfOp, *this);
      } else if (auto subfOp = dyn_cast<arith::SubFOp>(op)){
        handleSubfBackward(subfOp, *this);
      } else if (auto selectOp = dyn_cast<arith::SelectOp>(op)){
        handleSelectBackward(selectOp, *this);
      } else if (auto maxOp = dyn_cast<arith::MaxNumFOp>(op)){
        handleMaxBackward(maxOp, *this);

      // math ops
      } else if (auto cosOp = dyn_cast<math::CosOp>(op)){
        handleCosBackward(cosOp, *this);
      } else if (auto sinOp = dyn_cast<math::SinOp>(op)){
        handleSinBackward(sinOp, *this);
      } else if (auto sqrtOp = dyn_cast<math::SqrtOp>(op)){
        handleSqrtBackward(sqrtOp, *this);
      } else if (auto logOp = dyn_cast<math::LogOp>(op)){
        handleLogBackward(logOp, *this);  // For natural logarithm (base e): The derivative of ln(x) is 1/x
      } else if (auto log2Op = dyn_cast<math::Log2Op>(op)){
        handleLog2Backward(log2Op, *this); // For logarithm base 2: The derivative of log₂(x) is 1/(x·ln(2))
      } else if (auto expOp = dyn_cast<math::ExpOp>(op)){
        handleExpBackward(expOp, *this);
      } else if (auto exp2Op = dyn_cast<math::Exp2Op>(op)){
        handleExp2Backward(exp2Op, *this);

      // scf ops
      } else if (auto forOp = dyn_cast<scf::ForOp>(op)){
        handleForBackward(forOp, *this);
      }

      // todo-high: add else here (catch all) -- and explicitly error if none of the above
      // (otherwise users can have unexpected ops in their programs, and mine will just silently fail)

      // cleaner and more robust than adding printing to each handler
      if (DEBUG_PRINTS) {
        std::string currentIR;
        llvm::raw_string_ostream currentStream(currentIR);
        block.print(currentStream);
        if (initialIR != currentIR){
          // llvm::errs() << block.print();
          // dump writes to std err, but I want these be in "sync" with my other prints --
          llvm::raw_ostream &os = llvm::errs();
          block.print(os);
        }
      }

    } // for loop over ops

  } // handleAllOps

} // namespace triton
} // namespace mlir


//  TableGen emits only a *declaration* of impl::createConvertTritonToAutodiff.
//  Provide the single definition here so the linker can resolve it.

namespace mlir {
namespace triton {
namespace impl {

std::unique_ptr<::mlir::Pass> createConvertTritonToAutodiff() {
  return std::make_unique<ConvertTritonToAutodiff>();
}

} // namespace impl
} // namespace triton
} // namespace mlir

