#pragma once

#include "mlir/IR/Block.h"

namespace mlir {
namespace triton {

/// Run the backward-tiling inference starting from the kernel’s entry block.
/// Attaches four attributes (tt.tile_axes, tt.stream_axes, tt.tile_sig,
/// tt.sig_id) on each contraction operand value encountered in the IR rooted
/// at the block’s parent op.
void inferTiling(mlir::Block *entry);

} // namespace triton
} // namespace mlir 