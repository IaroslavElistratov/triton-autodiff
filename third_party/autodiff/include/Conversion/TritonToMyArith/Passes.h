#ifndef TRITON_CONVERSION_TRITON_TO_AUTODIFF_PASSES_H
#define TRITON_CONVERSION_TRITON_TO_AUTODIFF_PASSES_H

#include "mlir/Pass/Pass.h"

#include "triton/Dialect/Triton/IR/Dialect.h"   // same as amd

#include "autodiff/include/Dialect/Autodiff/IR/Dialect.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "autodiff/include/Conversion/TritonToAutodiff/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
