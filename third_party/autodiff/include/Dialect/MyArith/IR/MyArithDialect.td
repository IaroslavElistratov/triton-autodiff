#ifndef TRITON_DIALECT_MYARITH_IR_MYARITHDIALECT_TD
#define TRITON_DIALECT_MYARITH_IR_MYARITHDIALECT_TD

// myarith dialect inherits MLIR builtin dialect
include "mlir/IR/OpBase.td"

// MyArithDialect.td
def MyArith_Dialect : Dialect {
  let name = "myarith";

  let cppNamespace = "mlir::triton::myarith";

  let description = [{ My Arith Dialect. }];
}

#endif
