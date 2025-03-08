#ifndef TRITON_CONVERSION_TRITON_TO_MYARITH_PASSES
#define TRITON_CONVERSION_TRITON_TO_MYARITH_PASSES

include "mlir/Pass/PassBase.td"

// Passes.td
def ConvertTritonToMyArith: Pass<"convert-triton-to-myarith", "mlir::ModuleOp"> {
    let summary = "Convert Triton to MyArith";
    let dependentDialects = ["mlir::triton::myarith::MyArithDialect"];
}

#endif
