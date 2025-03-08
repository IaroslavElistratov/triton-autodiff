//#include "triton/Conversion/TritonToMyArith/TritonToMyArithPass.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/MyArith/IR/Dialect.h"
#include "llvm/ADT/APSInt.h"
#include <numeric>

#include "llvm/Support/Debug.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_CONVERTTRITONTOMYARITH
#include "triton/Conversion/TritonToMyArith/Passes.h.inc"

namespace {

// TritonToMyArithPass.cpp
struct ConvertTritonToMyArith
    : public impl::ConvertTritonToMyArithBase<ConvertTritonToMyArith> {

  using ConvertTritonToMyArithBase::ConvertTritonToMyArithBase;

  // main function
  void runOnOperation() override {
    // grab the module (IOW root) op
    auto mod = getOperation();
    // walk this recursively structred IR, and call rewriteSplatAddOp
    // only on "triton::FuncOp" -- which is the function op which encapsulates
    // our entire program
    mod->walk([&](triton::FuncOp func) {
      rewriteSplatAddOp(func);
    });
  }


  void rewriteAddFOp(arith::AddFOp addFOp, Value tensorSrc, Value scalarSrc) {
    // instantiate OpBuilder
    OpBuilder b(addFOp);

    // figure out which one is the tensor operand,
    // we already have the scalar operand as a parameter to this fn
    auto tensorOpnd =
        addFOp.getLhs() == tensorSrc ? addFOp.getRhs() : addFOp.getLhs();

    // create AddTensorScalarOp
    auto newOp = b.create<myarith::AddTensorScalarOp>(
        addFOp.getLoc(), tensorOpnd.getType(), tensorOpnd, scalarSrc);

    // replace the use of the result of "arith::AddFOp" with the
    // result of "myarith::AddTensorScalarOp"
    addFOp->replaceAllUsesWith(newOp);
  };

  void rewriteSplatAddOp(triton::FuncOp func) {
    // walk the IR, pattern match SplatOp
    func->walk([&](triton::SplatOp splatOp) {
      auto src = splatOp.getSrc();
      auto res = splatOp.getResult();

      // iterate over the users of the result of the SplatOp
      for (auto user : res.getUsers()) {
        // for each user, check if the user is the "arith::AddFOp"
        if (auto addOp = dyn_cast<arith::AddFOp>(user)) {
          // we found the pattern we're looking for, so re-write this op
          rewriteAddFOp(addOp, res, src);
        }
      }
    });
  }
};

} // namespace
} // namespace triton
} // namespace mlir
