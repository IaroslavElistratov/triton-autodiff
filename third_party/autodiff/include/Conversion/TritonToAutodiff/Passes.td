#ifndef TRITON_CONVERSION_TRITON_TO_AUTODIFF_PASSES
#define TRITON_CONVERSION_TRITON_TO_AUTODIFF_PASSES

include "mlir/Pass/PassBase.td"

// Passes.td
def ConvertTritonToAutodiff: Pass<"convert-triton-to-autodiff", "mlir::ModuleOp"> {
    let summary = "Convert Triton to Autodiff";
    let dependentDialects = ["mlir::triton::autodiff::AutodiffDialect"];
}

#endif
