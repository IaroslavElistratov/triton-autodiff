#ifndef TRITON_DIALECT_MYARITH_IR_MYARITHOPS_TD
#define TRITON_DIALECT_MYARITH_IR_MYARITHOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "triton/Dialect/MyArith/IR/MyArithDialect.td"

// MyArithOps.td
class MyArith_Op<string mnemonic, list<Trait> traits = []> :
    Op<MyArith_Dialect, mnemonic, traits>;

def MyArith_AddTensorScalarOp : MyArith_Op<"add_ts"> {
  let summary = "perform tensor-scalar add.";

  let arguments = (ins RankedTensorOf<[AnyFloat, AnyInteger]> : $tensor,
                       AnyTypeOf<[AnyFloat, AnyInteger]> : $scalar);

  let results = (outs RankedTensorOf<[AnyFloat, AnyInteger]> : $result);

  let assemblyFormat = "operands attr-dict `:` type(operands) `->` type($result)";
}

#endif
