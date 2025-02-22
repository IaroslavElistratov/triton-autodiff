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
  void runOnOperation() override {
    auto mod = getOperation();
    mod->walk([&](triton::FuncOp func) {
      rewriteSplatAddOp(func);
    });
  }


  void rewriteAddFOp(arith::AddFOp addFOp, Value tensorSrc, Value scalarSrc) {
    OpBuilder b(addFOp);

    auto tensorOpnd =
        addFOp.getLhs() == tensorSrc ? addFOp.getRhs() : addFOp.getLhs();
    auto newOp = b.create<myarith::AddTensorScalarOp>(
        addFOp.getLoc(), tensorOpnd.getType(), tensorOpnd, scalarSrc);

    addFOp->replaceAllUsesWith(newOp);
  };

  void rewriteSplatAddOp(triton::FuncOp func) {
    func->walk([&](triton::SplatOp splatOp) {
      auto src = splatOp.getSrc();
      auto res = splatOp.getResult();

      for (auto user : res.getUsers()) {
        if (auto addOp = dyn_cast<arith::AddFOp>(user)) {
          rewriteAddFOp(addOp, res, src);
        }
      }
    });
  }
};

} // namespace
} // namespace triton
} // namespace mlir
