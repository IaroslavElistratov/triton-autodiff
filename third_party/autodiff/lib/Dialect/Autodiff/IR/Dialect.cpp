#include "autodiff/include/Dialect/Autodiff/IR/Dialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "autodiff/include/Dialect/Autodiff/IR/Dialect.cpp.inc"

void mlir::triton::autodiff::AutodiffDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "autodiff/include/Dialect/Autodiff/IR/Ops.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "autodiff/include/Dialect/Autodiff/IR/Ops.cpp.inc"