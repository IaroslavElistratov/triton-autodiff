#ifndef TRITON_DIALECT_MYARITH_IR_DIALECT_H_
#define TRITON_DIALECT_MYARITH_IR_DIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/MyArith/IR/Dialect.h.inc"

#define GET_OP_CLASSES
#include "triton/Dialect/MyArith/IR/Ops.h.inc"

#endif