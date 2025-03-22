#include "autodiff/include/Conversion/TritonToMyArith/Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/APSInt.h"
#include <numeric>

#include "llvm/Support/Debug.h"

namespace mlir {
namespace triton {

  /// The three methods below are mutually recursive and follow the nesting of
  /// the IR: operation->region->block->operation->...

  /// Manages the indentation as we traverse the IR nesting.
  int indent;

  void resetIndent() { indent = 0; }
  IdentRAII pushIndent() { return IdentRAII(++indent); }

  llvm::raw_ostream &printIndent() {
    for (int i = 0; i < indent; ++i)
      llvm::outs() << "  ";
    return llvm::outs();
  }

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

  // void printGradMap(const llvm::DenseMap<Value, Value>& map) {
  //   llvm::outs() << "=== Printing grad_map contents ===\n";
  //   for (const auto& pair : map) {
  //     llvm::outs() << "Key: " << pair.first << "\n";
  //     llvm::outs() << "Key type: " << pair.first.getType() << "\n";
  //     llvm::outs() << "Value: " << pair.second << "\n";
  //     llvm::outs() << "Value type: " << pair.second.getType() << "\n";
  //     llvm::outs() << "-------------------\n";
  //   }
  //   llvm::outs() << "=== End of grad_map ===\n";
  // }

} // namespace triton
} // namespace mlir