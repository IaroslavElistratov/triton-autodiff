#ifndef TRITON_CONVERSION_HEADER_FOR_H
#define TRITON_CONVERSION_HEADER_FOR_H

#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"


#include "autodiff/include/Dialect/Autodiff/IR/Dialect.h"

namespace mlir {
namespace triton {

// Forward declaration
struct ConvertTritonToAutodiff;

void handleForBackward(scf::ForOp forOp, ConvertTritonToAutodiff& pass);

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_HEADER_FOR_H 

