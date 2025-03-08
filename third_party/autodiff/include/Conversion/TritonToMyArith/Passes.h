#ifndef TRITON_CONVERSION_TRITON_TO_MYARITH_PASSES_H
#define TRITON_CONVERSION_TRITON_TO_MYARITH_PASSES_H

#include "mlir/Pass/Pass.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/MyArith/IR/Dialect.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "triton/Conversion/TritonToMyArith/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
