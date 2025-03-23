//#include "triton/Conversion/TritonToMyArith/TritonToMyArithPass.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

// for reverse topo sort
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SetVector.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "autodiff/include/Dialect/MyArith/IR/Dialect.h"
#include "autodiff/include/Conversion/TritonToMyArith/Utils.h"
#include "llvm/ADT/APSInt.h"
#include <numeric>

#include "llvm/Support/Debug.h"


namespace mlir {
namespace triton {

#define GEN_PASS_DEF_CONVERTTRITONTOMYARITH
#include "autodiff/include/Conversion/TritonToMyArith/Passes.h.inc"

namespace {


struct ConvertTritonToMyArith
    : public impl::ConvertTritonToMyArithBase<ConvertTritonToMyArith> {

  using ConvertTritonToMyArithBase::ConvertTritonToMyArithBase;

  void markVisited(Operation *op, OpBuilder &builder, bool isInserted = false, bool isOriginal = false) {
    op->setAttr("autogradVisited", builder.getBoolAttr(true));
    if (isInserted)
      op->setAttr("isInserted", builder.getBoolAttr(true));
    if (isOriginal)
      op->setAttr("isOrig", builder.getBoolAttr(true));
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
      auto broadcasted = builder.create<triton::SplatOp>(loc, tensorType, scalarValue);

      // mark as visited
      markVisited(broadcasted, builder, true);
      markVisited(scalarValue, builder, true);
      return broadcasted;
  }

  Value cloneSubtree(Operation *targetOp, IRMapping &mapper, OpBuilder &builder) {
    // Null check
    if (!targetOp || targetOp->getNumResults() == 0)
      llvm::report_fatal_error("Cannot clone operation with no results");

    // If we've already cloned this result, return it directly
    if (mapper.contains(targetOp->getResult(0)))
      return mapper.lookup(targetOp->getResult(0));

    // Clone all operations we depend on first
    for (Value operand : targetOp->getOperands()) {
      // Skip block arguments and values already in the mapper
      if (mlir::isa<BlockArgument>(operand) || mapper.contains(operand))
        continue;

      if (Operation *defOp = operand.getDefiningOp())
        cloneSubtree(defOp, mapper, builder);
    }

    // Now clone this operation
    Operation *clonedOp = builder.clone(*targetOp, mapper);
    clonedOp->setAttr("isCloned", builder.getBoolAttr(true));
    markVisited(clonedOp, builder);

    // IRMapping used to ensure when you clone operations, the operands of the cloned
    // ops refer to the cloned values, not the original ones.
    // After cloning, manually map the original results to the clone's results:
    // this extracts from the map whatever Value in the map (the copied subgraph)
    // is equivalent to targetOp->getResult(0)

    // returns a value to avoid manually looking up the clone in the mapper after calling the function
    return clonedOp->getResult(0);
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

  // walk the IR backward, rewrite each operation with its corresponding backward function
  void rewriteSplatAddOp(triton::FuncOp func) {

    // printOperation(func, true);

    // assimung there's upstream grad only wrt to a single variable initially
    // bool is_first_node

    // error happening because Value (which is the type I'm trying to put into std::map) does not have move interface
    // (< comparitor), which it appers the impl of map is trying ot use to compare eleemtns of the map
    llvm::DenseMap<Value, Value> gradMap;


    OpBuilder builder(func.getContext());

    // // Walk all operations opaquely.
    // // todo: I think, you don't need topo sort if you're iterating in post order traversal (visit children before parent)
    // func->walk<WalkOrder::PostOrder>([&](Operation *op) {           // https://mlir.llvm.org/doxygen/classmlir_1_1Operation.html#a59740592240b950b8c8afcf4a2eb4113


    // copied from: llvm-project/mlir/lib/Dialect/Linalg/Transforms/Hoisting.cpp
    SetVector<Operation *> forwardSlice;
    getForwardSlice(func.getOperation(), &forwardSlice);
    for (Operation *op : llvm::reverse(forwardSlice)) {

      if (auto visitedAttr = op->getAttrOfType<BoolAttr>("autogradVisited")) {
          llvm::outs() << "Skipping visited" << "\n";
          continue;
      }

      // triton
      if (auto storeOp = dyn_cast<triton::StoreOp>(op)){
        handleStoreBackward(storeOp, func, builder, gradMap);
      } else if (auto rangeOp = dyn_cast<triton::MakeRangeOp>(op)){
        printIndent() << "visiting tt.make_range op\n";
      } else if (auto splatOp = dyn_cast<triton::SplatOp>(op)){
        printIndent() << "visiting tt.splat op\n";
      } else if (auto addptrOp = dyn_cast<triton::AddPtrOp>(op)){
        printIndent() << "visiting tt.addptr op\n";

      // arith
      } else if (auto addfOp = dyn_cast<arith::AddFOp>(op)){
        handleAddBackward(addfOp, func, builder, gradMap);
      } else if (auto mulfOp = dyn_cast<arith::MulFOp>(op)){
        handleMulBackward(mulfOp, func, builder, gradMap);
      } else if (auto divfOp = dyn_cast<arith::DivFOp>(op)){
        handleDivBackward(divfOp, func, builder, gradMap);
      } else if (auto constantOp = dyn_cast<arith::ConstantOp>(op)){
        printIndent() << "visiting arith.constant op\n";
      }
    } // for loop over ops

    // separate loop for loadop -- because its derivative (storeOp) destroyaes semantics of input args
    for (Operation *op : llvm::reverse(forwardSlice)) {

      if (auto visitedAttr = op->getAttrOfType<BoolAttr>("autogradVisited")) {
          llvm::outs() << "Skipping visited" << "\n";
          continue;
      }

      if (auto loadOp = dyn_cast<triton::LoadOp>(op)){
        handleLoadBackward(loadOp, func, builder, gradMap);
      }
    } // for loop over loads


    // iterate as above but delete ops that are not marked
    func->walk<WalkOrder::PostOrder>([&](Operation *op) {

      auto visitedAttr = op->getAttrOfType<BoolAttr>("autogradVisited");
      if (visitedAttr) llvm::outs() << "  Operation is marked as visited: " << *op << "\n";
      else llvm::outs() << "  Operation is NOT marked as visited: " << *op << "\n";

      // don't delete the fn itself
      if (!op->getAttrOfType<BoolAttr>("autogradVisited")
          && !dyn_cast<triton::FuncOp>(op)
          && !dyn_cast<triton::ReturnOp>(op)) {
        llvm::outs() << "Deleting unmarked node" << *op << "\n";

        op->dropAllUses();
        op->erase();

      }
    }); // lambda function for the walk

  } // RewriteSplatOp function



  void handleStoreBackward(triton::StoreOp storeOp, triton::FuncOp func,
                          OpBuilder &builder, llvm::DenseMap<Value, Value> &gradMap){
    // why .getValue works for StoreOp, but not for AddPtrOp
    //  - getResult() only seems to work on generic Operation -- seems specific subclases (e.g. triton::SplatOp) don't have the attributes of generic Operation (unless use ->)

    // because this will effectively load the upstream grad, I want to set the insertion point to the begining of the module
    // auto moduleOp = func.getBlock()->getParent()->getParentOfType<ModuleOp>();

    // func.getBody() returns Region, but "setInsertionPointToStart" expects pass Block,
    // .front() gets the first block in that region, which is the entry block
    Block *entryBlock = &func.getBody().front();
    builder.setInsertionPointToStart(entryBlock);

    // see all available constructors in -- triton/include/triton/Dialect/Triton/IR/TritonOps.td -> "def TT_LoadOp"
    Value ptr = storeOp->getOperand(0);
    auto newOp = builder.create<triton::LoadOp>(
        storeOp.getLoc(),
        ptr,
        storeOp.getCache(),  // copy cache modifier
        storeOp.getEvict(),  // copy eviction policy
        false  // isVolatile (storeOp doesn't have this, so keep default)
    );

    // grad wrt 1st arg (values) is the output (aka Value) of the newly added op
    llvm::outs() << "should be Value defined by add op: " << storeOp->getOperand(1) << "\n";
    // todo-high: note I'm currently assuming that there's a single Store function and that it's the first one be added to the gradMap
    gradMap[storeOp->getOperand(1)] = newOp.getResult();

    // mark as visited
    markVisited(newOp, builder, true);

    // I want to preserve the of pointer calculations which was leading to the original store, so mark it as keep
    // note: .getDefiningOp wt specifying type, returns a pointer to Operation
    Operation* addptrOp = storeOp.getOperand(0).getDefiningOp();
    markVisited(addptrOp, builder, false, true);
    Operation* splatOp = addptrOp->getOperand(0).getDefiningOp();
    markVisited(splatOp, builder, false, true);
    Operation* makerangeOp = addptrOp->getOperand(1).getDefiningOp();
    markVisited(makerangeOp, builder, false, true);

    // i did set the insertion point above, but becuase I'm not cloning the above ops (i'm just keeping them where they are in the graph).
    // the insertion point only effects the node that I add to the graph (does not effect already existing nodes)
    // here I move them on top of the fucntion body as well
    makerangeOp->moveBefore(newOp);
    splatOp->moveBefore(newOp);
    addptrOp->moveBefore(newOp);

    // todo: (OLD) replace that blockArg with another tensor (corresponding to grad of that original argument)
  }

  void handleLoadBackward(triton::LoadOp loadOp, triton::FuncOp func,
                          OpBuilder &builder, llvm::DenseMap<Value, Value> &gradMap){
    printIndent() << "visiting tt.load op\n";
    // traverse parents to find the initial pointer

    Value upstream = getUpstreamGrad(loadOp.getResult(), gradMap);
    Value ptr = loadOp->getOperand(0);
    // see all available constructors in -- triton/include/triton/Dialect/Triton/IR/TritonOps.td -> "def TT_LoadOp"

    // Create a builder without setting insertion point at first, then set insertion point
    // Seems no constructor to specify "InsertionPointAfter" at the time of construction

    // .front() gets the first block in that region, which is the entry block
    Block *entryBlock = &func.getBody().front();
    // // comment: also changed the insertion point
    // builder.setInsertionPointToEnd(entryBlock);

    // set insertion point to before the last operation (before ReturnOp)
    Operation *lastOp = &entryBlock->back();
    builder.setInsertionPoint(lastOp);
    // printGradMap(gradMap);

    // Create a ValueRange from the operands
    // SmallVector<Value> operands = {ptr, upstream};
    // TypeRange typically specify types of outputs of an op. Here's it's empty bc this op does not produce any outputs
    //  Unlike e.g. creating LoadOp where I'm passing ptr.getType() because a load operation returns a value of the same type as what it's loading from the pointer
    // auto newOp = builder.create<triton::StoreOp>(loadOp.getLoc(), TypeRange(), operands);
    auto newOp = builder.create<triton::StoreOp>(loadOp.getLoc(), ptr, upstream,
                                                triton::CacheModifier::NONE,
                                                triton::EvictionPolicy::NORMAL);

    // todo-high: note this op does not add anything to the gradMap, bc here I'm manually traversing inputs to this op (hardcoded for the specific toy graphs I'm working with) and marking them so that they will not be switched on (IOW iterated over) by this loop
    // fixes mismatch between the type of the value we're trying to store and the pointee type of the pointer we're storing to.
    // ensure the type of upstream matches what ptr points to.

    // mark as visited
    //   get generic Operation, note returns a pointer
    markVisited(newOp, builder, true);


    // loadOp.getOperation()->getName().getStringRef() -- does not include operands so reesult value
    // Use the operation's built-in printer
    std::string opStr;
    llvm::raw_string_ostream os(opStr);
    loadOp->print(os);
    newOp->setAttr("gradOf", builder.getStringAttr(opStr));


    // I want to preserve the of pointer calculations which was leading to the original store, so mark it as keep
    Operation* addptrOp = loadOp.getOperand(0).getDefiningOp();
    markVisited(addptrOp, builder, false, true);
    Operation* splatOp = addptrOp->getOperand(0).getDefiningOp();
    markVisited(splatOp, builder, false, true);
    Operation* makerangeOp = addptrOp->getOperand(1).getDefiningOp();
    // todo: we have already visistd this Operation, fine for my toy example, but keep in mind for the future examples
    markVisited(makerangeOp, builder, false, true);
  }

  void handleAddBackward(arith::AddFOp addfOp, triton::FuncOp func,
                          OpBuilder &builder, llvm::DenseMap<Value, Value> &gradMap){
    printIndent() << "visiting arith.addf op\n";

    Value upstream = getUpstreamGrad(addfOp.getResult(), gradMap);

    // float local_grad = 1.;

    // 0-th arg is not a consant, so compute grad wrt it
    Value lhs = addfOp.getOperand(0);
    assert(!dyn_cast<arith::ConstantOp>(lhs.getDefiningOp()));
    // don't insert unnecessary multiply of upstream with 1 (since numerically result is the same as wt multiplying)
    gradMap[lhs] = upstream;

    // todo-high: hardcoded for my specific graph
    // 1st arg is a constant, so grad wrt it is zero
    Value rhs = addfOp.getOperand(1);
    Operation* rhs_producer = rhs.getDefiningOp();
    assert(dyn_cast<arith::ConstantOp>(rhs_producer));
  }

  void handleMulBackward(arith::MulFOp mulfOp, triton::FuncOp func,
                          OpBuilder &builder, llvm::DenseMap<Value, Value> &gradMap){
    printIndent() << "visiting arith.mulf op\n";

    Value upstream = getUpstreamGrad(mulfOp.getResult(), gradMap);

    Value lhs = mulfOp.getOperand(0);
    Value rhs = mulfOp.getOperand(1);


    // (1) clone lhs subtree
    // essentially, it's just like std::map but just a mlir specific struct
    IRMapping mapper;

    Operation *targetOpLhs = lhs.getDefiningOp();
    builder.setInsertionPoint(targetOpLhs);
    Value clonedResultLhs = cloneSubtree(targetOpLhs, mapper, builder);

    // (2) differentiate rhs
    auto OpGradRhs = builder.create<arith::MulFOp>(mulfOp.getLoc(), clonedResultLhs, upstream);
    // note: I belive here I want to set grad of the original rhs (not ClonedRhs), because I'd continue differenciating the original path (while cloned will not be differenicated)
    gradMap[rhs] = OpGradRhs.getResult();

    markVisited(OpGradRhs, builder, true);

    // (3) clone rhs subtree
    // prepare for cloning another separate subgraph
    mapper.clear();

    Operation *targetOpRhs = rhs.getDefiningOp();
    builder.setInsertionPoint(targetOpRhs);
    Value clonedResultRhs = cloneSubtree(targetOpRhs, mapper, builder);

    // (4) differentiate lhs
    auto OpGradLhs = builder.create<arith::MulFOp>(mulfOp.getLoc(), clonedResultRhs, upstream);
    gradMap[lhs] = OpGradLhs.getResult();

    // mark as visited
    markVisited(OpGradLhs, builder, true);
  }


  void handleDivBackward(arith::DivFOp divfOp, triton::FuncOp func,
                          OpBuilder &builder, llvm::DenseMap<Value, Value> &gradMap){
    printIndent() << "visiting arith.divf op\n";

    Value upstream = getUpstreamGrad(divfOp.getResult(), gradMap);

    Value a = divfOp.getOperand(0);
    Value b = divfOp.getOperand(1);

    IRMapping mapper;

    // (1) clone lhs subtree
    Operation *targetOpLhs = a.getDefiningOp();
    builder.setInsertionPoint(targetOpLhs);
    Value a_cloned = cloneSubtree(targetOpLhs, mapper, builder);

    // (2) clone rhs subtree
    mapper.clear();
    Operation *targetOpRhs = b.getDefiningOp();
    builder.setInsertionPoint(targetOpRhs);
    Value b_cloned = cloneSubtree(targetOpRhs, mapper, builder);

    // (3) differentiate lhs

    // a local
    // auto ones = builder.create<arith::ConstantOp>(divfOp->getLoc(), upstream.getType(), builder.getF32FloatAttr(1.0));
    // this creates a scalar and broadcasts it to a shape specificed by "upstream.getType()"
    auto ones = createConstantTensor(builder, divfOp->getLoc(), upstream.getType(), 1.0);
    auto a_local = builder.create<arith::DivFOp>(divfOp->getLoc(), ones.getResult(), b_cloned);

    auto OpGradLhs = builder.create<arith::MulFOp>(divfOp->getLoc(), a_local.getResult(), upstream);
    gradMap[a] = OpGradLhs.getResult();

    // set attributes
    markVisited(OpGradLhs, builder, true);

    markVisited(ones, builder, true);
    markVisited(a_local, builder, true);

    // (4) differentiate rhs

    // b local

    // auto two = builder.create<arith::ConstantOp>(divfOp->getLoc(), divfOp.getType(), builder.getF32FloatAttr(2.0));
    auto pow = builder.create<arith::MulFOp>(divfOp.getLoc(), b_cloned, b_cloned);
    auto div = builder.create<arith::DivFOp>(divfOp.getLoc(), a_cloned, pow.getResult());
    auto neg = createConstantTensor(builder, divfOp->getLoc(), div.getType(), -1.0);
    auto b_local = builder.create<arith::MulFOp>(divfOp.getLoc(), neg.getResult(), div.getResult());

    auto OpGradRhs = builder.create<arith::MulFOp>(divfOp.getLoc(), b_local.getResult(), upstream);
    gradMap[b] = OpGradRhs.getResult();

    // set attributes
    markVisited(OpGradRhs, builder, true);

    markVisited(b_local, builder, true);
    markVisited(neg, builder, true);
    markVisited(div, builder, true);
    markVisited(pow, builder, true);

  }

}; // ConvertTritonToMyArith stuct

} // private namespace
} // namespace triton
} // namespace mlir

