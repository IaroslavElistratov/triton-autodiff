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

// TritonToMyArithPass.cpp
struct ConvertTritonToMyArith
    : public impl::ConvertTritonToMyArithBase<ConvertTritonToMyArith> {

  using ConvertTritonToMyArithBase::ConvertTritonToMyArithBase;

  // main function
  void runOnOperation() override {
    // grab the module (IOW root) op
    auto mod = getOperation();
    // walk this recursively structred IR, and call rewriteSplatAddOp
    // only on "triton::FuncOp" -- which is the function op which encapsulates
    // our entire program

    // note:
    // To traverse, can call my recursive functions (like printOperation, printRegion, printBlock),
    // but docs: "in many cases, unwrapping the recursive structure of the IR is cumbersome and
    // you may be interested in using other helpers. [...] The getOps<OpTy>() is useful to iterate
    // on some Operations immediately listed inside a single block (or a single region), however
    // it is frequently interesting to traverse the IR in a nested fashion. To this end MLIR exposes
    // the walk() helper on Operation, Block, and Region. This helper takes a single argument: a callback
    // method that will be invoked for every operation recursively nested under the provided entity."

    // todo-med: since I'm not using recursive funcs in "rewriteSplatAddOp", I'm not traversing body of the fn recursively (only the upper-most level)
    // answer-now: oh this body in { } is just a lamda function, which specifies that callback, which is called by walk whenerver there's a match 
    mod->walk([&](triton::FuncOp func) {
      rewriteSplatAddOp(func);
    });
  }

  triton::SplatOp createConstantTensor(OpBuilder &builder, Location loc, Type tensorType, float value) {
      auto scalarType = builder.getF32Type();
      auto scalarValue = builder.create<arith::ConstantOp>(loc, scalarType, builder.getF32FloatAttr(value));
      auto broadcasted = builder.create<triton::SplatOp>(loc, tensorType, scalarValue);

      // mark as visited
      broadcasted->setAttr("autogradVisited", builder.getBoolAttr(true));
      broadcasted->setAttr("isInserted", builder.getBoolAttr(true));

      scalarValue->setAttr("autogradVisited", builder.getBoolAttr(true));
      scalarValue->setAttr("isInserted", builder.getBoolAttr(true));
      return broadcasted;
  }


  void cloneSubtree(Operation *targetOp, IRMapping &mapper, OpBuilder &builder) {
    if (mapper.contains(targetOp))
      return; // Operation already cloned

    // Recursively clone operand-defining operations
    for (auto operand : targetOp->getOperands()) {
      if (auto definingOp = operand.getDefiningOp())
        cloneSubtree(definingOp, mapper, builder);
    }

    // Clone the current operation
    Operation *clonedOp = builder.clone(*targetOp, mapper);
    // mark as visited
    clonedOp->setAttr("autogradVisited", builder.getBoolAttr(true));
    // also setting this additional attribute for readability of printed IRs, this attribute is not checked anywehre in the code
    clonedOp->setAttr("isCloned", builder.getBoolAttr(true));
    mapper.map(targetOp->getResults(), clonedOp->getResults());
  }


  // walk the IR backward, rewrite each operation with its corresponding backward function
  void rewriteSplatAddOp(triton::FuncOp func) {

    // printOperation(func, true);

    // assimung there's upstream grad only wrt to a single variable initially
    // bool is_first_node


    // error hapening becuase Value (which is the type I'm trying to put into std::map) does not have move interface (< comparitor), which it appers the impl of map is trying ot use to compare eleemtns of the map
    llvm::DenseMap<Value, Value> grad_map{};



    // // Walk all operations opaquely.
    // // todo: I think, you don't need topo sort if you're iterating in post order traversal (visit children before parent)
    // func->walk<WalkOrder::PostOrder>([&](Operation *op) {           // https://mlir.llvm.org/doxygen/classmlir_1_1Operation.html#a59740592240b950b8c8afcf4a2eb4113

    // todo-high:
    // 1. Init grads of the last variable in the fwd ("Grad Outputs") to all ones
    //   - find out tensor size (get last operation in the function, assume it's a store, grab tensor sizes from there); allocate tensor of ones with that shape
    // I guess the problem is that you don't know the size of input tensors (you only know the size of the slices of these tensors that the current program processes)
    // So, can't allocate grad for each tensor as a single contigious tensor, seems can only allocate grad for each individual slice (probably leads memory fragmentation?)

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

        // why .getValue works for StoreOp, but not for AddPtrOp
        //  - getResult() only seems to work on generic Operation -- seems specific subclases (e.g. triton::SplatOp) don't have the attributes of generic Operation (unless use ->)

        // todo-now: maybe don't even need to replace original pointer (block arg) with grad pointer -- just allocate the grads outside of the kenrel and, for every arg of the kernel, pass grad of taht arg in that argument

        // // extract original ptr (to the begining of the array which fwd STORE'ed the data in)
        // note: I believe op.getPtr() retruns Value of one of the arguments to that op (the argument can be at different index, for different ops, depedining which one of the arugments is of type PointerType)
        // Value ptr = storeOp.getPtr();
        // auto addptrOp = ptr.getDefiningOp<triton::AddPtrOp>();
        // Value addptrPtr = addptrOp.getPtr();
        // auto splatOp = addptrPtr.getDefiningOp<triton::SplatOp>();
        // Value operand = splatOp->getOperand(0);
        // // Operation *producer = operand.getDefiningOp()
        // auto blockArg = cast<BlockArgument>(operand);
        // llvm::outs() << "blockArg: " << blockArg << "\n";

        // // todo: need to replace the above with a new pointer, need to create a new pointer

        // // todo-high: Instead of traversing manually, use getPointerTypeWithShape getPointerTypeSameShape? 
        // // Type new_ptr = getPointerTypeSameShape(ptr);
        // int address_space = 1;
        // //  could not convert 'blockArg' from 'mlir::BlockArgument' to 'mlir::Type'
        // // so try passing not 'blockArg' but operand.getPtr()
        // Type ptrType = getPointerType(operand.getPtr(), address_space);



        // because this will effectively load the upstream grad, I want to set the insertion point to the begining of the module
        OpBuilder builder(func.getContext());
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
        // todo-high: note I'm currently assuming that there's a single Store function and that it's the first one be added to the grad_map
        grad_map[storeOp->getOperand(1)] = newOp.getResult();




        // mark as visited
        newOp->setAttr("autogradVisited", builder.getBoolAttr(true)); // mark as visited
        newOp->setAttr("isInserted", builder.getBoolAttr(true)); // for readability, not used in the code

        // I want to preserve the of pointer calculations which was leading to the original store, so mark it as keep
        // note: .getDefiningOp wt specifying type, returns a pointer to Operation
        Operation* addptrOp = storeOp.getOperand(0).getDefiningOp();
        addptrOp->setAttr("autogradVisited", builder.getBoolAttr(true));
        addptrOp->setAttr("isOrig", builder.getBoolAttr(true));
        Operation* splatOp = addptrOp->getOperand(0).getDefiningOp();
        splatOp->setAttr("autogradVisited", builder.getBoolAttr(true));
        splatOp->setAttr("isOrig", builder.getBoolAttr(true));
        Operation* makerangeOp = addptrOp->getOperand(1).getDefiningOp();
        makerangeOp->setAttr("autogradVisited", builder.getBoolAttr(true));
        makerangeOp->setAttr("isOrig", builder.getBoolAttr(true));

        // i did set the insertion point above, but becuase I'm not cloning the above ops (i'm just keeping them where they are in the graph).
        // the insertion point only effects the node that I add to the graph (does not effect already existing nodes)
        // here I move them on top of the fucntion body as well
        makerangeOp->moveBefore(newOp);
        splatOp->moveBefore(newOp);
        addptrOp->moveBefore(newOp);

        // todo: (OLD) replace that blockArg with another tensor (corresponding to grad of that original argument)

      } else if (auto rangeOp = dyn_cast<triton::MakeRangeOp>(op)){
        printIndent() << "visiting tt.make_range op\n";
      } else if (auto splatOp = dyn_cast<triton::SplatOp>(op)){
        printIndent() << "visiting tt.splat op\n";
      } else if (auto addptrOp = dyn_cast<triton::AddPtrOp>(op)){
        printIndent() << "visiting tt.addptr op\n";

      // arith
      } else if (auto addfOp = dyn_cast<arith::AddFOp>(op)){
        printIndent() << "visiting arith.addf op\n";

        Value upstream = grad_map[addfOp.getResult()];
        if (!upstream){
          llvm::outs() << "expected grad in the map" << "\n";
          exit(1);
        }
        llvm::outs() << "extracted upstream for addf, from the grad map: " << upstream << "\n";


        // float local_grad = 1.;

        // 0-th arg is not a consant, so compute grad wrt it
        Value lhs = addfOp.getOperand(0);
        assert(!dyn_cast<arith::ConstantOp>(lhs.getDefiningOp()));
        // don't insert unnecessary multiply of upstream with 1 (since numerically result is the same as wt multiplying)
        grad_map[lhs] = upstream;

        // todo-now: hardcoded for my specific graph
        // 1st arg is a constant, so grad wrt it is zero
        Value rhs = addfOp.getOperand(1);
        Operation* rhs_producer = rhs.getDefiningOp();
        assert(dyn_cast<arith::ConstantOp>(rhs_producer));

      } else if (auto mulfOp = dyn_cast<arith::MulFOp>(op)){
        printIndent() << "visiting arith.mulf op\n";

        Value upstream = grad_map[mulfOp.getResult()];
        if (!upstream){
          llvm::outs() << "expected grad in the map" << "\n";
          exit(1);
        }

        Value lhs = mulfOp.getOperand(0);
        Value rhs = mulfOp.getOperand(1);


        // todo-high:
        // Don't clone if inputs to an op you matched to are directly inputs to the fn (see issues/1)
        //    I think I need to check if any of the values inputs to each op I match to (in this case MulOp) are intermideats (not inputs to fwd fn, and not ouputs of the fwd fn)
        //    need to do cloning only IF the inputs to this op are not inputs to the function (cout this as direct inputs: fn_input->splat->add_ptr) -- in such cases do not duplicate the subgraph -- its ok if you do duplicate but you'll create a redundant copy of (fn_input->splat->add_ptr) -- duplicating subgprah is not really needed bc there were no intermideates computed in fwd in the first place
        // And if so, copy the subtree leading to these nodes

        // (1) clone lhs subtree
        OpBuilder builder(func.getContext());
        // essentially, it's just like std::map but just a mlir specific struct
        IRMapping mapper;

        Operation *targetOpLhs = lhs.getDefiningOp();
        builder.setInsertionPoint(targetOpLhs);
        cloneSubtree(targetOpLhs, mapper, builder);

        // IRMapping plays a critical role in ensuring that when you clone operations, the operands of the cloned ops refer to the cloned values, not the original ones.
        // After cloning, you manually map the original results to the clone's results:
        // this extracts from the map wahtever Value in the map (the copied subgraph) is equivalent to targetOpLhs->getResult(0)
        Value clonedResultLhs = mapper.lookup(targetOpLhs->getResult(0));
        // ... use clonedResult in a new operation, if needed



        // (2) differentiate rhs

        auto OpGradRhs = builder.create<arith::MulFOp>(mulfOp.getLoc(), clonedResultLhs, upstream);
        // note: I belive here I want to set grad of the original rhs (not ClonedRhs), because I'd continue differenciating the original path (while cloned will not be differenicated)
        grad_map[rhs] = OpGradRhs.getResult();

        OpGradRhs->setAttr("autogradVisited", builder.getBoolAttr(true));
        OpGradRhs->setAttr("isInserted", builder.getBoolAttr(true));

        // (3) clone rhs subtree

        // prepare for cloning another separate subgraph
        mapper.clear();

        Operation *targetOpRhs = rhs.getDefiningOp();
        builder.setInsertionPoint(targetOpRhs);
        cloneSubtree(targetOpRhs, mapper, builder);
        Value clonedResultRhs = mapper.lookup(targetOpRhs->getResult(0));


        // (4) differentiate lhs

        auto OpGradLhs = builder.create<arith::MulFOp>(mulfOp.getLoc(), clonedResultRhs, upstream);
        grad_map[lhs] = OpGradLhs.getResult();

        // mark as visited
        OpGradLhs->setAttr("autogradVisited", builder.getBoolAttr(true));
        OpGradLhs->setAttr("isInserted", builder.getBoolAttr(true));


      } else if (auto divfOp = dyn_cast<arith::DivFOp>(op)){
        printIndent() << "visiting arith.divf op\n";

        Value upstream = grad_map[divfOp.getResult()];
        if (!upstream){
          llvm::outs() << "expected grad in the map" << "\n";
          exit(1);
        }

        Value a = divfOp.getOperand(0);
        Value b = divfOp.getOperand(1);


        OpBuilder builder(func.getContext());
        IRMapping mapper;


        // (1) clone lhs subtree

        Operation *targetOpLhs = a.getDefiningOp();
        builder.setInsertionPoint(targetOpLhs);
        cloneSubtree(targetOpLhs, mapper, builder);
        Value clonedResultLhs = mapper.lookup(targetOpLhs->getResult(0));
        auto a_cloned = clonedResultLhs; // for brevity


        // (2) clone rhs subtree

        mapper.clear();
        Operation *targetOpRhs = b.getDefiningOp();
        builder.setInsertionPoint(targetOpRhs);
        cloneSubtree(targetOpRhs, mapper, builder);
        Value clonedResultRhs = mapper.lookup(targetOpRhs->getResult(0));
        auto b_cloned = clonedResultRhs;


        // todo-now:
        // todo-now:
        // todo-now:
        //  big problem with my "copy the subgraph to recompute the operands" approach is that there's overlap between recompuated values when matching to each op
        //  when matching to Z (div), you gonna insert subgraphs that recompute: x, y. But after that (when matching to the next op), Y (mul) you again will copy the subgraph leading to operands of the match op -- which will recompute x AGAIN
        //   - well, because here (when matching to each op) I'm traversing nodes from the end, I basically need to copy the sugraph (leading to the operands of the last op) once and this will recompute all the intermidate ops; Next when matching to the next ops (further from the end of the graph) all their operands should be already recompuated and you just need to have some kind of mapping (from their %names in the current graph to ?? their names in fwd)
        //   - even simpler: can I just copy entire forward into my backward once (mark all it as visited) before matching and any of the ops?
        //
        // answer-now: before doing any optimizaitons (above), just make the below work



        // (3) differentiate lhs

        // a local
        // auto ones = builder.create<arith::ConstantOp>(divfOp->getLoc(), upstream.getType(), builder.getF32FloatAttr(1.0));
        // this creates a scalar and broadcasts it to a shape specificed by "upstream.getType()"
        auto ones = createConstantTensor(builder, divfOp->getLoc(), upstream.getType(), 1.0);
        auto a_local = builder.create<arith::DivFOp>(divfOp->getLoc(), ones.getResult(), b_cloned);

        auto OpGradLhs = builder.create<arith::MulFOp>(divfOp->getLoc(), a_local.getResult(), upstream);
        grad_map[a] = OpGradLhs.getResult();

        // set attributes
        OpGradLhs->setAttr("autogradVisited", builder.getBoolAttr(true));
        OpGradLhs->setAttr("isInserted", builder.getBoolAttr(true));

        ones->setAttr("autogradVisited", builder.getBoolAttr(true));
        ones->setAttr("isInserted", builder.getBoolAttr(true));

        a_local->setAttr("autogradVisited", builder.getBoolAttr(true));
        a_local->setAttr("isInserted", builder.getBoolAttr(true));



        // (4) differentiate rhs

        // b local

        // auto two = builder.create<arith::ConstantOp>(divfOp->getLoc(), divfOp.getType(), builder.getF32FloatAttr(2.0));
        auto pow = builder.create<arith::MulFOp>(divfOp.getLoc(), b_cloned, b_cloned);
        auto div = builder.create<arith::DivFOp>(divfOp.getLoc(), a_cloned, pow.getResult());
        auto neg = createConstantTensor(builder, divfOp->getLoc(), div.getType(), -1.0);
        auto b_local = builder.create<arith::MulFOp>(divfOp.getLoc(), neg.getResult(), div.getResult());

        auto OpGradRhs = builder.create<arith::MulFOp>(divfOp.getLoc(), b_local.getResult(), upstream);
        grad_map[b] = OpGradRhs.getResult();

        // set attributes
        OpGradRhs->setAttr("autogradVisited", builder.getBoolAttr(true));
        OpGradRhs->setAttr("isInserted", builder.getBoolAttr(true));

        b_local->setAttr("autogradVisited", builder.getBoolAttr(true));
        b_local->setAttr("isInserted", builder.getBoolAttr(true));

        neg->setAttr("autogradVisited", builder.getBoolAttr(true));
        neg->setAttr("isInserted", builder.getBoolAttr(true));

        div->setAttr("autogradVisited", builder.getBoolAttr(true));
        div->setAttr("isInserted", builder.getBoolAttr(true));

        pow->setAttr("autogradVisited", builder.getBoolAttr(true));
        pow->setAttr("isInserted", builder.getBoolAttr(true));


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
        printIndent() << "visiting tt.load op\n";
        // traverse parents to find the initial pointer

        Value upstream = grad_map[loadOp.getResult()];
        if (!upstream){
          llvm::outs() << "expected grad in the map" << "\n";
          exit(1);
        }
        Value ptr = loadOp->getOperand(0);
        // see all available constructors in -- triton/include/triton/Dialect/Triton/IR/TritonOps.td -> "def TT_LoadOp"

        // Create a builder without setting insertion point at first, then set insertion point
        // Seems no constructor to specify "InsertionPointAfter" at the time of construction
        OpBuilder builder(func.getContext());

        // .front() gets the first block in that region, which is the entry block
        Block *entryBlock = &func.getBody().front();
        // // comment: also changed the insertion point
        // builder.setInsertionPointToEnd(entryBlock);

        // set insertion point to before the last operation (before ReturnOp)
        Operation *lastOp = &entryBlock->back();
        builder.setInsertionPoint(lastOp);
        // printGradMap(grad_map);

        // Create a ValueRange from the operands
        SmallVector<Value> operands = {ptr, upstream};
        // TypeRange typically specify types of outputs of an op. Here's it's empty bc this op does not produce any outputs
        //  Unlike e.g. creating LoadOp where I'm passing ptr.getType() because a load operation returns a value of the same type as what it's loading from the pointer
        // auto newOp = builder.create<triton::StoreOp>(loadOp.getLoc(), TypeRange(), operands);
        auto newOp = builder.create<triton::StoreOp>(loadOp.getLoc(), ptr, upstream,
                                                    triton::CacheModifier::NONE,
                                                    triton::EvictionPolicy::NORMAL);

        // todo-now: note this op does not add anything to the grad_map, bc here I'm manually traversing inputs to this op (hardcoded for the specific toy graphs I'm working with) and marking them so that they will not be switched on (IOW iterated over) by this loop
        // fixes mismatch between the type of the value we're trying to store and the pointee type of the pointer we're storing to.
        // ensure the type of upstream matches what ptr points to.

        // // Get the pointee type from the pointer to ensure type compatibility
        // //  1) ensure the pointer type (ptr) is a valid Triton pointer type
        // //  2) The value we're storing (upstream) has a type that matches what the pointer points to
        // auto ptrType = mlir::cast<triton::PointerType>(ptr.getType());
        // // Returns the type of the value that the pointer points to
        // Type pointeeType = ptrType.getPointeeType();

        // example of creating StoreOp:
        // rewriter.create<triton::StoreOp>(loc, newPointer, newData, newMask,
        //                                 op.getBoundaryCheck(), op.getCache(),
        //                                 op.getEvict());


        // mark as visited
        //   get generic Operation, note returns a pointer
        newOp->setAttr("autogradVisited", builder.getBoolAttr(true));
        newOp->setAttr("isInserted", builder.getBoolAttr(true));


        // loadOp.getOperation()->getName().getStringRef() -- does not include operands so reesult value
        // Use the operation's built-in printer
        std::string opStr;
        llvm::raw_string_ostream os(opStr);
        loadOp->print(os);
        newOp->setAttr("gradOf", builder.getStringAttr(opStr));


        // I want to preserve the of pointer calculations which was leading to the original store, so mark it as keep
        Operation* addptrOp = loadOp.getOperand(0).getDefiningOp();
        addptrOp->setAttr("autogradVisited", builder.getBoolAttr(true)); // mark as visited
        addptrOp->setAttr("isOrig", builder.getBoolAttr(true));
        Operation* splatOp = addptrOp->getOperand(0).getDefiningOp();
        splatOp->setAttr("autogradVisited", builder.getBoolAttr(true)); // mark as visited
        splatOp->setAttr("isOrig", builder.getBoolAttr(true));
        Operation* makerangeOp = addptrOp->getOperand(1).getDefiningOp();
        // todo: we have already visistd this Operation, fine for my toy example, but keep in mind for the future examples
        makerangeOp->setAttr("autogradVisited", builder.getBoolAttr(true)); // mark as visited
        makerangeOp->setAttr("isOrig", builder.getBoolAttr(true));

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

  }; // ConvertTritonToMyArith stuct

} // private namespace
} // namespace triton
} // namespace mlir

