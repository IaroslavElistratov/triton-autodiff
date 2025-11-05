#ifndef TRITON_DIALECT_AUTODIFF_IR_AUTODIFFOPS_TD
#define TRITON_DIALECT_AUTODIFF_IR_AUTODIFFOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "autodiff/include/Dialect/Autodiff/IR/AutodiffDialect.td"


// AutodiffOps.td

// Autodiff_Op -- all the operations in Autodiff dialect;
// We inherit the base MLIR op, and we passing the mnemonic and the traits through
class Autodiff_Op<string mnemonic, list<Trait> traits = []> :
    Op<Autodiff_Dialect, mnemonic, traits>;

// actually defines an operation;
// specifies mnemonic to be "add_ts"
def Autodiff_AddTensorScalarOp : Autodiff_Op<"add_ts"> {
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
