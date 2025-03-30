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

    // answer-now: the problem because I didn't set the insertion point :)
    OpBuilder builder(func.getContext());
    builder.setInsertionPointToStart(entryBlock);

    Operation *lastFwdOp = cloneSubtree(beforeReturnOp, origToCloned, builder);
    // let the ops inserted during rewriting backward be inserted after the forward ops
    builder.setInsertionPointAfter(lastFwdOp);


    // the above mapping: original nodes -> inserted nodes.
    // To lookup intermideats in the cloned (aka cloned subgraph),
    // when iterating over the original subgraph (and re-writting that original subgraph with derivative formualrs)
    // I think I don't even need to reverse the mapping: can directly use it --
    // bc I'm iterating over the "original nodes" and want to figure out what "cloned" node does an original node refers to


    // // Walk all operations opaquely.
    // // todo: I think, you don't need topo sort if you're iterating in post order traversal (visit children before parent)
    // func->walk<WalkOrder::PostOrder>([&](Operation *op) {           // https://mlir.llvm.org/doxygen/classmlir_1_1Operation.html#a59740592240b950b8c8afcf4a2eb4113


    // copied from: llvm-project/mlir/lib/Dialect/Linalg/Transforms/Hoisting.cpp
    SetVector<Operation *> forwardSlice;
    getForwardSlice(func.getOperation(), &forwardSlice);

    // First pass: handle all operations except LoadOp
    for (Operation *op : llvm::reverse(forwardSlice)) {

      if (op->getAttrOfType<BoolAttr>("autogradVisited")) {
          if (DEBUG_PRINTS) llvm::outs() << "Skipping visited" << "\n";
          continue;
      }

      // triton ops
      if (auto storeOp = dyn_cast<triton::StoreOp>(op)){
        handleStoreBackward(storeOp, builder, gradMap, origToCloned, lastFwdOp);
      } else if (auto mmOp = dyn_cast<triton::DotOp>(op)){
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
          if (DEBUG_PRINTS) llvm::outs() << "Skipping visited" << "\n";
          continue;
      }

      if (auto loadOp = dyn_cast<triton::LoadOp>(op)){
        handleLoadBackward(loadOp, builder, gradMap, origToCloned, func);
      }
    } // for loop over loads

    // Final pass: remove unmarked operations
    func->walk<WalkOrder::PostOrder>([&](Operation *op) {

      auto visitedAttr = op->getAttrOfType<BoolAttr>("autogradVisited");
      if (DEBUG_PRINTS) {
        if (visitedAttr) llvm::outs() << "  Operation is marked as visited: " << *op << "\n";
        else llvm::outs() << "  Operation is NOT marked as visited: " << *op << "\n";
      }

      // don't delete the fn itself
      if (!op->getAttrOfType<BoolAttr>("autogradVisited") &&
          !isa<triton::FuncOp, triton::ReturnOp>(op)) {
        if (DEBUG_PRINTS) llvm::outs() << "Deleting unmarked node" << *op << "\n";
        op->dropAllUses();
        op->erase();

      }
    }); // lambda function for the walk

  } // RewriteSplatOp function



  void handleStoreBackward(triton::StoreOp storeOp, OpBuilder &builder,
                          llvm::DenseMap<Value, Value> &gradMap, IRMapping &origToCloned,
                          Operation *lastFwdOp){
    // why .getValue works for StoreOp, but not for AddPtrOp
    //  - getResult() only seems to work on generic Operation -- seems specific subclases (e.g. triton::SplatOp) don't have the attributes of generic Operation (unless use ->)

    // because this will effectively load the upstream grad, I want to set the insertion point to right after the last node in fwd
    builder.setInsertionPointAfter(lastFwdOp);

    // see all available constructors in -- triton/include/triton/Dialect/Triton/IR/TritonOps.td -> "def TT_LoadOp"
    Value ptr = origToCloned.lookup(storeOp->getOperand(0));

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
        ptr,
        maskCloned,
        storeOp.getCache(),  // copy cache modifier
        storeOp.getEvict(),  // copy eviction policy
        false  // isVolatile (storeOp doesn't have this, so keep default)
    );

    // grad wrt 1st arg (values) is the output (aka Value) of the newly added op
    if (DEBUG_PRINTS) llvm::outs() << "should be Value defined by add op: " << storeOp->getOperand(1) << "\n";
    // todo-high: note I'm currently assuming that there's a single Store function and that it's the first one be added to the gradMap
    maybeAccumulateGrad(storeOp->getOperand(1), load, gradMap, builder);

    // mark as visited
    markVisited(builder, visitedType::Inserted, load);

    // todo: (OLD) replace that blockArg with another tensor (corresponding to grad of that original argument)
  }

  void handleLoadBackward(triton::LoadOp loadOp, OpBuilder &builder,
                          llvm::DenseMap<Value, Value> &gradMap, IRMapping &origToCloned,
                          triton::FuncOp func){
    if (DEBUG_PRINTS) printIndent() << "visiting tt.load op\n";
    // traverse parents to find the initial pointer

    Value upstream = getUpstreamGrad(loadOp, gradMap);
    // question-now:
    //  here and everywhere where you use lookup to get clonedOp (i.e. op from the fwd graph),
    //  explicitly mark subgraph leading to original op (that you use in the lookup) for deletion?
    //  So that you don't match to them in my first loop.
    //  IOW: Basically you're iterating over the bwd graph, and matching to ops, and re-writing these
    //  ops with bwd formulas, and when you need some intermidiate for a bwd formula you use that
    //  intermidiate from fwd (by doing "opFromFwd = origToCloned.lookup(opFromBwd)")
    //  but there is still subgraph of nodes leading to "opFromBwd" -- you don't use opFromBwd,
    //  and subsequently all the nodes that produced that node.
    //  Because you don't set isAutogradVisited attribute on these nodes, they subsequently get
    //  deleted by my 3rd loop. But you're first loop may match to these nodes again
    //  (you just processed them here and thus you don't want 1st loop to match to them again).
    //  At the moment this works fine because these nodes leading to opFromBwd are typically
    //  splat, make_range, etc -- and bc you're first loop done't match to these nodes this kind of
    //  works -- but it would be more robust and future proof to explicitly mark them for deletion
    Value ptr = origToCloned.lookup(loadOp->getOperand(0));
    // see all available constructors in -- triton/include/triton/Dialect/Triton/IR/TritonOps.td -> "def TT_LoadOp"

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
    auto store = builder.create<triton::StoreOp>(
      loadOp.getLoc(),
      ptr,
      upstream,
      maskCloned,
      triton::CacheModifier::NONE,
      triton::EvictionPolicy::NORMAL);

    // todo-high: note this op does not add anything to the gradMap, bc here I'm manually traversing inputs to this op (hardcoded for the specific toy graphs I'm working with) and marking them so that they will not be switched on (IOW iterated over) by this loop
    // fixes mismatch between the type of the value we're trying to store and the pointee type of the pointer we're storing to.
    // ensure the type of upstream matches what ptr points to.

    markVisited(builder, visitedType::Inserted, store);

    // Record original operation for debugging
    // loadOp.getOperation()->getName().getStringRef() -- does not include operands so result value
    // Use the operation's built-in printer
    std::string opStr;
    llvm::raw_string_ostream os(opStr);
    loadOp->print(os);
    store->setAttr("gradOf", builder.getStringAttr(opStr));

  }

  void handleAddBackward(arith::AddFOp addfOp, OpBuilder &builder,
                        llvm::DenseMap<Value, Value> &gradMap){
    if (DEBUG_PRINTS) printIndent() << "visiting arith.addf op\n";

    Value upstream = getUpstreamGrad(addfOp, gradMap);

    // float local_grad = 1.;

    // 0-th arg is not a consant, so compute grad wrt it
    Value lhs = addfOp.getOperand(0);
    assert(!dyn_cast<arith::ConstantOp>(lhs.getDefiningOp()));
    // don't insert unnecessary multiply of upstream with 1 (since numerically result is the same as wt multiplying)
    maybeAccumulateGrad(lhs, upstream, gradMap, builder);

    // todo-high: hardcoded for my specific graph
    // 1st arg is a constant, so grad wrt it is zero
    Value rhs = addfOp.getOperand(1);
    Operation* rhs_producer = rhs.getDefiningOp();
    assert(dyn_cast<arith::ConstantOp>(rhs_producer));
  }

  void handleMulBackward(arith::MulFOp mulfOp, OpBuilder &builder,
                        llvm::DenseMap<Value, Value> &gradMap, IRMapping &origToCloned){
    if (DEBUG_PRINTS) printIndent() << "visiting arith.mulf op\n";

    Value upstream = getUpstreamGrad(mulfOp, gradMap);
    // insert operations after the gradient value, they depend on, is defined
    builder.setInsertionPointAfter(upstream.getDefiningOp());

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
    if (DEBUG_PRINTS) printIndent() << "visiting arith.divf op\n";

    Value upstream = getUpstreamGrad(divfOp, gradMap);
    // insert operations after the gradient value, they depend on, is defined
    builder.setInsertionPointAfter(upstream.getDefiningOp());

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
    if (DEBUG_PRINTS) printIndent() << "visiting math.cos op\n";

    Value upstream = getUpstreamGrad(cosOp, gradMap);
    // insert operations after the gradient value they depend on is defined
    builder.setInsertionPointAfter(upstream.getDefiningOp());

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
    if (DEBUG_PRINTS) printIndent() << "visiting math.sin op\n";

    Value upstream = getUpstreamGrad(sinOp, gradMap);
    // insert operations after the gradient value they depend on is defined
    builder.setInsertionPointAfter(upstream.getDefiningOp());

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
    if (DEBUG_PRINTS) printIndent() << "visiting math.sqrt op\n";

    Value upstream = getUpstreamGrad(sqrtOp, gradMap);
    // insert operations after the gradient value they depend on is defined
    builder.setInsertionPointAfter(upstream.getDefiningOp());

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
    if (DEBUG_PRINTS) printIndent() << "visiting math.log op\n";

    Value upstream = getUpstreamGrad(logOp, gradMap);
    // insert operations after the gradient value they depend on is defined
    builder.setInsertionPointAfter(upstream.getDefiningOp());

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
    if (DEBUG_PRINTS) printIndent() << "visiting math.exp op\n";

    Value upstream = getUpstreamGrad(expOp, gradMap);
    // insert operations after the gradient value they depend on is defined
    builder.setInsertionPointAfter(upstream.getDefiningOp());

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
    if (DEBUG_PRINTS) printIndent() << "visiting tt.dot op\n";

    Value upstream = getUpstreamGrad(mmOp, gradMap);
    builder.setInsertionPointAfter(upstream.getDefiningOp());

    // Extract matrix multiplication operands
    Value a = mmOp.getA();
    Value b = mmOp.getB();
    Value c = mmOp.getC();  // Accumulator

    // Get cloned operands from forward graph
    Value aCloned = origToCloned.lookup(a);
    Value bCloned = origToCloned.lookup(b);

    // For matmul C = A * B + acc
    // dA = dC * B^T
    // dB = A^T * dC

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
        upstream,                     // dC
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
        upstream,                     // dC
        createConstantTensor(builder, mmOp.getLoc(), b.getType(), 0.0), // zero accumulator
        mmOp.getInputPrecision(),
        mmOp.getMaxNumImpreciseAcc());

    maybeAccumulateGrad(b, gradB, gradMap, builder);


    markAllVisited(builder, visitedType::Inserted, gradA, gradB, bTrans, aTrans);
  }

}; // ConvertTritonToAutodiff stuct

} // private namespace
} // namespace triton
} // namespace mlir

