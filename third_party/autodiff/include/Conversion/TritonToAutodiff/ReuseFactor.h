#pragma once

#include "mlir/IR/Value.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"

namespace mlir {
namespace triton {

void propagateReuseCounts(mlir::Block *blockPtr);

} // namespace triton
} // namespace mlir