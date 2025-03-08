#include "triton/Dialect/MyArith/IR/Dialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "triton/Dialect/MyArith/IR/Dialect.cpp.inc"

void mlir::triton::myarith::MyArithDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/MyArith/IR/Ops.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "triton/Dialect/MyArith/IR/Ops.cpp.inc"