#ifndef TRITON_CONVERSION_TRITON_TO_AUTODIFF_HANDLERS_H
#define TRITON_CONVERSION_TRITON_TO_AUTODIFF_HANDLERS_H

#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "autodiff/include/Dialect/Autodiff/IR/Dialect.h"

namespace mlir {
namespace triton {

  Operation* handleStoreBackward(triton::StoreOp storeOp, Operation *lastBwdOp, ConvertTritonToAutodiff& pass);

  void handleLoadBackward(triton::LoadOp loadOp, triton::FuncOp func, ConvertTritonToAutodiff& pass);

  void handleAddBackward(arith::AddFOp addfOp, ConvertTritonToAutodiff& pass);

  void handleTruncfBackward(arith::TruncFOp truncfOp, ConvertTritonToAutodiff& pass);

  void handleMulBackward(arith::MulFOp mulfOp, ConvertTritonToAutodiff& pass);

  void handleDivBackward(arith::DivFOp divfOp, ConvertTritonToAutodiff& pass);

  void handleCosBackward(math::CosOp cosOp, ConvertTritonToAutodiff& pass);

  void handleSinBackward(math::SinOp sinOp, ConvertTritonToAutodiff& pass);

  void handleSqrtBackward(math::SqrtOp sqrtOp, ConvertTritonToAutodiff& pass);

  void handleLogBackward(math::LogOp logOp, ConvertTritonToAutodiff& pass);

  void handleExpBackward(math::ExpOp expOp, ConvertTritonToAutodiff& pass);

  void handleMatmulBackward(triton::DotOp mmOp, ConvertTritonToAutodiff& pass);

  void handleMaxBackward(arith::MaxNumFOp maxOp, ConvertTritonToAutodiff& pass);

  void handleReduceBackward(triton::ReduceOp reduceOp,ConvertTritonToAutodiff& pass);

  void handleExtFBackward(arith::ExtFOp extfOp, ConvertTritonToAutodiff& pass);

  void handleSubfBackward(arith::SubFOp subfOp, ConvertTritonToAutodiff& pass);

  void handleSelectBackward(arith::SelectOp selectOp, ConvertTritonToAutodiff& pass);

  void handleBroadcastBackward(triton::BroadcastOp broadcastOp, ConvertTritonToAutodiff& pass);

  void handleLog2Backward(math::Log2Op log2Op, ConvertTritonToAutodiff& pass);

  void handleExp2Backward(math::Exp2Op exp2Op, ConvertTritonToAutodiff& pass);

  void handleExpandDimsBackward(triton::ExpandDimsOp expandDimsOp, ConvertTritonToAutodiff& pass);

  void handleTransBackward(triton::TransOp transOp, ConvertTritonToAutodiff& pass);

  void handleSplatBackward(triton::SplatOp splatOp, ConvertTritonToAutodiff& pass);

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITON_TO_AUTODIFF_HANDLERS_H 

