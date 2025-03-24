#include "autodiff/include/Dialect/Autodiff/IR/Dialect.h"
#include "autodiff/include/Conversion/TritonToAutodiff/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

namespace {

void init_triton_autodiff_passes(py::module &&m) {

  // Add Autodiff conversion passes
  m.def("add_triton_to_autodiff", [](mlir::PassManager &pm) {
    // In MLIR code, there's a naming convention at play:
    // The class that implements the pass is often named with a "Pass" suffix, like ConvertTritonToAutodiffPass
    // But the factory function that creates instances of this pass typically doesn't include the "Pass" suffix, so it's createConvertTritonToAutodiff() instead of createConvertTritonToAutodiffPass()
    // This is why the compiler was suggesting the correct name without the "Pass" suffix.
    pm.addPass(mlir::triton::createConvertTritonToAutodiff());
  });
  
  // Add more passes as needed
  // For example:
  // m.def("add_autodiff_to_llvm", [](mlir::PassManager &pm) {
  //   pm.addPass(createConvertAutodiffToLLVMPass());
  // });
}

} // namespace

void init_triton_autodiff(py::module &&m) {
  m.doc() = "Python bindings to the AutoDiff Triton backend";

  auto passes = m.def_submodule("passes");
  init_triton_autodiff_passes(passes.def_submodule("autodiff"));

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::autodiff::AutodiffDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
} 
