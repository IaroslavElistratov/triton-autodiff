#ifndef TRITON_DIALECT_AUTODIFF_IR_AUTODIFFDIALECT_TD
#define TRITON_DIALECT_AUTODIFF_IR_AUTODIFFDIALECT_TD

// autodiff dialect inherits MLIR builtin dialect
include "mlir/IR/OpBase.td"

// AutodiffDialect.td
def Autodiff_Dialect : Dialect {
  let name = "autodiff";

  let cppNamespace = "mlir::triton::autodiff";

  let description = [{ My Arith Dialect. }];
}

#endif
