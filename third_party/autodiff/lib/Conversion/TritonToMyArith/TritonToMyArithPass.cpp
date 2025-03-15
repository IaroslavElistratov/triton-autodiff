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


  void rewriteAddFOp(arith::AddFOp addFOp, Value tensorSrc, Value scalarSrc) {
    /// Create a builder and set insertion point to the given operation, which
    /// will cause subsequent insertions to go right before it.
    OpBuilder builder(addFOp);

    // figure out which one is the tensor operand,
    // we already have the scalar operand as a parameter to this fn
    auto tensorOpnd = addFOp.getLhs() == tensorSrc ? addFOp.getRhs() : addFOp.getLhs();

    // create AddTensorScalarOp
    auto newOp = builder.create<myarith::AddTensorScalarOp>(addFOp.getLoc(), tensorOpnd.getType(), tensorOpnd, scalarSrc);

    // replace the use of the result of "arith::AddFOp" with the
    // result of "myarith::AddTensorScalarOp"
    addFOp->replaceAllUsesWith(newOp);
  };





  /// The three methods below are mutually recursive and follow the nesting of
  /// the IR: operation->region->block->operation->...

  void printOperation(Operation *op, bool is_recursive) {

    // Print the operation itself and some of its properties
    printIndent() << "\n";
    printIndent() << "visiting op: '" << op->getName() << "' with "
                  << op->getNumOperands() << " operands and "
                  << op->getNumResults() << " results\n";

    // Print the operation attributes
    if (!op->getAttrs().empty()) {
      printIndent() << op->getAttrs().size() << " attributes:\n";
      for (NamedAttribute attr : op->getAttrs())
        printIndent() << " - '" << attr.getName().getValue() << "' : '" << attr.getValue() << "'\n";
    }

    // Print information about the producer of each of the operands.
    if (op->getNumOperands()) {
      printIndent() << op->getNumOperands() << " operands:\n";
      for (Value operand : op->getOperands()) {

        if (Operation *producer = operand.getDefiningOp()) {
          printIndent() << " - Operand produced by operation '" << producer->getName() << "'\n";
        } else {
          // If there is no defining op, the Value is necessarily a Block argument.
          auto blockArg = cast<BlockArgument>(operand); // operand.cast<BlockArgument>();
          printIndent() << " - Operand produced by Block argument, number " << blockArg.getArgNumber() << "\n";
        }
      }
    }

    // Print information about the user of each of the result.
    if (op->getNumResults()) {
      printIndent() << op->getNumResults() << " results:\n";
      for (auto indexedResult : llvm::enumerate(op->getResults())) {
        Value result = indexedResult.value();
        //   >>> %4 = arith.addf %3, %cst : tensor<4xf32>

        printIndent() << "  - Result " << indexedResult.index();

        // Print shape
        Type type = result.getType(); 
        //   >>> tensor<4xf32>
        RankedTensorType tensorTy = dyn_cast<RankedTensorType>(type);
        ArrayRef<int64_t> shape = tensorTy.getShape(); // .getRank();
        llvm::outs() << " shape = [";
        for (unsigned i = 0; i < shape.size(); ++i) {
          if (i > 0) llvm::outs() << ", ";
          llvm::outs() << shape[i];
        }
        llvm::outs() << "]";
        //   >>> [4]


        if (result.use_empty()) {
          llvm::outs() << " has no uses:\n";
          continue;
        } else if (result.hasOneUse()) {
          llvm::outs() << " has 1 use:\n";
        } else {
          llvm::outs() << " has " << std::distance(result.getUses().begin(), result.getUses().end()) << " uses:\n";
        }

        for (Operation *userOp : result.getUsers()) {
          printIndent() << "    - " << userOp->getName() << "\n";
        }
      }
    }

    if (op->getNumRegions()){
      printIndent() << op->getNumRegions() << " nested regions:\n";
    }
    // Recurse into each of the regions attached to the operation.
    if (is_recursive){
      auto indent = pushIndent();
      for (Region &region : op->getRegions())
        printRegion(region);
    }
  }

  void printRegion(Region &region) {
    // A region does not hold anything by itself other than a list of blocks.
    printIndent() << "Region with " << region.getBlocks().size()
                  << " blocks:\n";
    auto indent = pushIndent();
    for (Block &block : region.getBlocks())
      printBlock(block);
  }

  void printBlock(Block &block) {
    // Print the block intrinsics properties (basically: argument list)
    printIndent()
        << "Block with " << block.getNumArguments() << " arguments, "
        << block.getNumSuccessors()
        << " successors, and "
        // Note, this `.size()` is traversing a linked-list and is O(n).
        << block.getOperations().size() << " operations\n";

    // Block main role is to hold a list of Operations: let's recurse.
    auto indent = pushIndent();
    for (Operation &op : block.getOperations())
      printOperation(&op, true);
  }

  /// Manages the indentation as we traverse the IR nesting.
  int indent;
  struct IdentRAII {
    int &indent;
    IdentRAII(int &indent) : indent(indent) {}
    ~IdentRAII() { --indent; }
  };
  void resetIndent() { indent = 0; }
  IdentRAII pushIndent() { return IdentRAII(++indent); }

  llvm::raw_ostream &printIndent() {
    for (int i = 0; i < indent; ++i)
      llvm::outs() << "  ";
    return llvm::outs();
  }



  // for now limit with:
  //  - 1) assume there's only 1 instance of the fn executing (don't worry about computing slices of input/output)
  //  - 2)


  // walk the IR backward, rewrite each operation with its corresponding backward function
  void rewriteSplatAddOp(triton::FuncOp func) {

    printOperation(func, true);


    // error hapening becuase Value (which is the type I'm trying to put into std::map) does not have move interface (< comparitor), which it appers the impl of map is trying ot use to compare eleemtns of the map
    // std::map<Value, Value> grad_map{};
    // grad_map["SSD"] = 30;
    // grad_map.size()
    // print_map("map: ", grad_map);

     llvm::DenseMap<Value, Value> grad_map{};



    // // Walk all operations opaquely.
    // // todo: I think, you don't need topo sort if you're iterating in post order traversal (visit children before parent)
    // func->walk<WalkOrder::PostOrder>([&](Operation *op) {           // https://mlir.llvm.org/doxygen/classmlir_1_1Operation.html#a59740592240b950b8c8afcf4a2eb4113

    // todo-high:
    // I guess the problem is that you don't know the size of input tensors (you only know the size of the slices of these tensors that the current program processes)
    // So, can't allocate grad for each tensor as a single contigious tensor, seems can only allocate grad for each individual slice (probably leads memory fragmentation?)

    // todo-high: 
    // 1. Init grads of the last variable in the fwd ("Grad Outputs") to all ones
    //   - find out tensor size (get last operation in the function, assume it's a store, grab tensor sizes from there); allocate tensor of ones with that shape

    // copied from: llvm-project/mlir/lib/Dialect/Linalg/Transforms/Hoisting.cpp
    SetVector<Operation *> forwardSlice;
    // todo-high: make sure "func.getOperation()" extract Value of root op in the fn, getForwardSlice expects Value
    getForwardSlice(func.getOperation(), &forwardSlice);
    for (Operation *op : llvm::reverse(forwardSlice)) {
    // for (auto *op : llvm::reverse(forwardSlice)) {


      // if (bool is_visited = dyn_cast<bool>(op->getAttr("autogradVisited"))){
    if (auto visitedAttr = op->getAttrOfType<BoolAttr>("autogradVisited")) {
        llvm::outs() << "Skipping visited" << "\n";
        continue;
      }

      // triton
      if (auto storeOp = dyn_cast<triton::StoreOp>(op)){         // python signature: tl.store(pointer, value, ...)

      // The dyn_cast<> operator is a “checking cast” operation. It checks to see if the operand is of the specified type, and if so, returns a pointer to it (this operator does not work with references). If the operand is not of the correct type, a null pointer is returned. Thus, this works very much like the dynamic_cast<> operator in C++, and should be used in the same circumstances. Typically, the dyn_cast<> operator is used in an if statement or some other flow control statement like this -- https://llvm.org/docs/ProgrammersManual.html#dyn_cast
      // if (triton::StoreOp storeOp = dyn_cast<triton::StoreOp>(op)){         // python signature: tl.store(pointer, value, ...)


        // why .getValue works for StoreOp, but not for AddPtrOp
        //  - getResult() only seems to work on generic Operation -- seems specific subclases (e.g. triton::SplatOp) don't have the attributes of generic Operation
        // Operation* addptrOp = storeOp.getOperand(0).getDefiningOp();
        // Operation* splatOp = addptrOp->getOperand(1).getDefiningOp();



        // todo-now: maybe don't even need to replace original pointer (block arg) with grad pointer -- just allocate the grads outside of the kenrel and, for every arg of the kernel, pass grad of taht arg in that argument

        // // extract original ptr (to the begining of the array which fwd STORE'ed the data in)
        // note: I believe op.getPtr() retruns Value of one of the arguments to that op (the argument can be at different index, for different ops, depedining which one of the arugments is of type PointerType)
        // Value ptr = storeOp.getPtr();
        // auto addptrOp = ptr.getDefiningOp<triton::AddPtrOp>();
        // Value addptrPrt = addptrOp.getPtr();
        // auto splatOp = addptrPrt.getDefiningOp<triton::SplatOp>();
        // Value operand = splatOp->getOperand(0);
        // // Operation *producer = operand.getDefiningOp()
        // auto blockArg = cast<BlockArgument>(operand);
        // llvm::outs() << "blockArg: " << blockArg << "\n";

        // // todo: need to replace the above with a new pointer, need to create a new pointer

        // // todo-high: Instead of traversing manually, use getPointerTypeWithShape getPointerTypeSameShape? 
        // // Type new_ptr = getPointerTypeSameShape(ptr);

        // int address_space = 1;
        // //  could not convert ‘blockArg’ from ‘mlir::BlockArgument’ to ‘mlir::Type’
        // // so try passing not ‘blockArg’ but operand.getPtr()
        // Type ptrType = getPointerType(operand.getPtr(), address_space);





        /// Create a builder and set insertion point to the given operation, which
        /// will cause subsequent insertions to go right before it.
        OpBuilder builder(storeOp);

        // see all available constructors in -- triton/include/triton/Dialect/Triton/IR/TritonOps.td -> "def TT_LoadOp"
        Value ptr = storeOp->getOperand(0);
        auto newOp = builder.create<triton::LoadOp>(storeOp.getLoc(), ptr.getType(), ptr);

        // grad wrt 1st arg (values) is the output (aka Value) of the newly added op
        llvm::outs() << "should be Value defined by add op: " << storeOp->getOperand(1) << "\n";
        grad_map[storeOp->getOperand(1)] = newOp.getResult();




        // mark as visited
        //   get generic Operation, note returns a pointer
        Operation* op = newOp.getOperation();
        op->setAttr("autogradVisited", builder.getBoolAttr(true));

        // I want to preserve the of pointer calculations which was leading to the original store, so mark it as keep
        // note: .getDefiningOp wt specifying type, returns a pointer to Operation
        Operation* addptrOp = storeOp.getOperand(0).getDefiningOp();
        addptrOp->setAttr("autogradVisited", builder.getBoolAttr(true)); // mark as visited
        Operation* splatOp = addptrOp->getOperand(0).getDefiningOp();
        splatOp->setAttr("autogradVisited", builder.getBoolAttr(true)); // mark as visited
        Operation* makerangeOp = addptrOp->getOperand(1).getDefiningOp();
        makerangeOp->setAttr("autogradVisited", builder.getBoolAttr(true)); // mark as visited




        // storeOp.dropAllUse();
        storeOp.erase();

        // We only have to rewrite load/stores with tensor pointers
        // if (!triton::isTensorPointerType(ptr.getType())){
        //   return nullptr;
        // }

        // todo-now:
        //  1) replace that blockArg with another tensor (corresponding to grad of that original argument)
        //  2) mark the subgraph I traversed above as "keep" (add that attirbute on every node of that subgraph)
        //    - in each condition check if either "keep" or "delete" attr is set -- if so, continue to loop (but skip the current op)
        //  3) add my manual DCE pass after iterating over all ops, and deletes them if "delete" flag is set on them






      } else if (auto loadOp = dyn_cast<triton::LoadOp>(op)){
        printIndent() << "visiting tt.load op\n";
        // traverse parents to find the initial pointer
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

        // 1st arg is a constant, so grad wrt it is zero
        Value rhs = addfOp.getOperand(1);
        Operation* rhs_producer = rhs.getDefiningOp();
        assert(dyn_cast<arith::ConstantOp>(rhs_producer));

        addfOp.erase();
        // cleaner to delete right after the assert above, but moving it (delete *after* current op),
        // so that don't get err: op deleted but still has uses
        rhs_producer->erase();

      } else if (auto constantOp = dyn_cast<arith::ConstantOp>(op)){
        printIndent() << "visiting arith.constant op\n";
      }


    // // todo-now: do the below but for re-writting for each op above
    // todo: cast to triton op before acessing these?
    //   auto src = op->getSrc();
    //   auto res = op->getResult();

    //   // iterate over the users of the result of the SplatOp
    //   for (auto user : res.getUsers()) {
    //     // for each user, check if the user is the "arith::AddFOp"
    //     if (auto addOp = dyn_cast<arith::AddFOp>(user)) {
    //       // we found the pattern we're looking for, so re-write this op

    //       Value tensorSrc = res;
    //       Value scalarSrc = src;

    //       /// Create a builder and set insertion point to the given operation, which
    //       /// will cause subsequent insertions to go right before it.
    //       OpBuilder builder(addFOp);

    //       // figure out which one is the tensor operand,
    //       // we already have the scalar operand as a parameter to this fn
    //       auto tensorOpnd = addFOp.getLhs() == tensorSrc ? addFOp.getRhs() : addFOp.getLhs();

    //       // create AddTensorScalarOp
    //       auto newOp = builder.create<myarith::AddTensorScalarOp>(addFOp.getLoc(), tensorOpnd.getType(), tensorOpnd, scalarSrc);

    //       // replace the use of the result of "arith::AddFOp" with the
    //       // result of "myarith::AddTensorScalarOp"
    //       addFOp->replaceAllUsesWith(newOp);

    //     }
    //   }



    };
    // });
  }
};

} // namespace
} // namespace triton
} // namespace mlir







/* useful examples:

  // Load/store with tensor pointers implicitly will check the bound while
  // accessing memory, so we should set `mask` and `other` (according to the
  // padding). Also note that load with tensor pointers do not have `mask` and
  // `other` while building IR from Python AST
  std::optional<ArrayRef<int>> boundaryCheck;
  if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
    assert(!loadOp.getMask() && !loadOp.getOther());
    boundaryCheck = loadOp.getBoundaryCheck();
  } else if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
    assert(!storeOp.getMask());
    boundaryCheck = storeOp.getBoundaryCheck();
  }

  // Generate new `ptr`, `mask` and `other`
  auto newPtr = info.generatePtr(builder, op->getLoc());
  auto newMask = info.generateMask(builder, op->getLoc(), boundaryCheck);
  Value newOther;
  if (auto loadOp = dyn_cast<triton::LoadOp>(op))
    newOther = info.generateOther(builder, op->getLoc(), loadOp.getPadding());

  // Create a new operation
  if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
    auto newResult = builder.create<triton::LoadOp>(
        loadOp.getLoc(), newPtr, newMask, newOther, loadOp.getCache(),
        loadOp.getEvict(), loadOp.getIsVolatile());
    op->getResult(0).replaceAllUsesWith(newResult);
  } else if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
    builder.create<triton::StoreOp>(storeOp.getLoc(), newPtr,
                                    storeOp.getValue(), newMask,
                                    storeOp.getCache(), storeOp.getEvict());
  }

  // Erase the original operation
  eraser.push(op);
  return nullptr;
}


Value generatePtr(OpBuilder &builder, const Location &loc) {
  assert(tensorShape.size() == offsets.size() &&
        tensorShape.size() == strides.size());
  auto indexTensorType =
      RankedTensorType::get(tensorShape, builder.getI64Type());
  auto ptrType = cast<triton::PointerType>(base.getType());
  auto ptrTensorType = RankedTensorType::get(tensorShape, ptrType);

  // Generate offsets per dimension
  Value ptr = builder.create<triton::SplatOp>(loc, ptrTensorType, base);
  for (unsigned i = 0; i < tensorShape.size(); ++i) {
    auto offsetWithRange = getExpandedOffsetWithRange(builder, loc, i);

    // We must splat strides into the expanded shape not a row for retaining
    // the divisibility information given by strides
    Value splatStride = builder.create<triton::SplatOp>(
        loc, offsetWithRange.getType(), strides[i]);
    Value offsetWithStride =
        builder.create<arith::MulIOp>(loc, offsetWithRange, splatStride);
    Value broadcasted = builder.create<triton::BroadcastOp>(
        loc, indexTensorType, offsetWithStride);

    // Add to the pointer
    ptr = builder.create<triton::AddPtrOp>(loc, ptrTensorType, ptr,
                                          broadcasted);
  }

  return ptr;
}







// ConvertTritonStoreToBufferStore
auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
Value tensorPtr = addPtrOp.getPtr();   // retruns Value for consumer of the addPtrOp output
Value tensorOffset = addPtrOp.getOffset();
auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
Value basePtr = splatOp.getSrc();

auto ptrType = dyn_cast<RankedTensorType>(ptr.getType());
auto valType = dyn_cast<RankedTensorType>(val.getType());

*/


