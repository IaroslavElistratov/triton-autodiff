#ifndef TRITON_CONVERSION_TRITON_TO_AUTODIFF_UTILS_IO_H
#define TRITON_CONVERSION_TRITON_TO_AUTODIFF_UTILS_IO_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Block.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace triton {

// Indentation support
extern int indent;
struct IdentRAII {
  int &indent;
  IdentRAII(int &indent) : indent(indent) {}
  ~IdentRAII() { --indent; }
};

// Functions for IR exploration and debugging
void printOperation(Operation *op, bool is_recursive);
void printRegion(Region &region);
void printBlock(Block &block);

// Indentation helpers
void resetIndent();
IdentRAII pushIndent();
llvm::raw_ostream &printIndent();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITON_TO_AUTODIFF_UTILS_IO_H 