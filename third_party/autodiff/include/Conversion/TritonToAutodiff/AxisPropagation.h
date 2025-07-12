#pragma once

#include "mlir/IR/Value.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"

namespace mlir {
// namespace scf { class ForOp; } // forward declaration
namespace triton {

void propagateAxesInFuncOp(mlir::Block *blockPtr);

} // namespace triton
} // namespace mlir