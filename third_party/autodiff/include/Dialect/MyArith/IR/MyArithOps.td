#ifndef TRITON_DIALECT_MYARITH_IR_MYARITHOPS_TD
#define TRITON_DIALECT_MYARITH_IR_MYARITHOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "autodiff/include/Dialect/MyArith/IR/MyArithDialect.td"


// MyArithOps.td

// MyArith_Op -- all the operations in MyArith dialect;
// We inherit the base MLIR op, and we passing the mnemonic and the traits through
class MyArith_Op<string mnemonic, list<Trait> traits = []> :
    Op<MyArith_Dialect, mnemonic, traits>;

// actually defines an operation;
// specifies mnemonic to be "add_ts"
def MyArith_AddTensorScalarOp : MyArith_Op<"add_ts"> {
  let summary = "perform tensor-scalar add.";

  // the two operand types are:
  //    - $tensor -- a ranked tensor of Float or Int
  //    - $scalar -- just a scalar Float or Int
  let arguments = (ins RankedTensorOf<[AnyFloat, AnyInteger]> : $tensor,
                       AnyTypeOf<[AnyFloat, AnyInteger]> : $scalar);

  // result is the same as $tensor
  let results = (outs RankedTensorOf<[AnyFloat, AnyInteger]> : $result);

  // td will generate a printer and parser for us
  let assemblyFormat = "operands attr-dict `:` type(operands) `->` type($result)";
}

#endif
