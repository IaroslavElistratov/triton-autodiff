#ifndef TRITON_DIALECT_AUTODIFF_IR_DIALECT_H_
#define TRITON_DIALECT_AUTODIFF_IR_DIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"   // same as amd
// question-now: AutodiffDialect?
#include "autodiff/include/Dialect/Autodiff/IR/Dialect.h.inc"

#define GET_OP_CLASSES
// question-now: AutodiffOps?
#include "autodiff/include/Dialect/Autodiff/IR/Ops.h.inc"

#endif