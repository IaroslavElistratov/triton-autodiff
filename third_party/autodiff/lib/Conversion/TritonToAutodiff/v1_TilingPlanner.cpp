#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "autodiff/include/Conversion/TritonToAutodiff/TilingPlanner.h"

using namespace mlir;

namespace mlir {
namespace triton {

// String literals for every attribute key touched
static constexpr StringLiteral kAxisNamesAttr     = "tt.axis_names";
static constexpr StringLiteral kBroadcastAxesAttr = "tt.broadcast_axes";
static constexpr StringLiteral kTiledFwdAttr      = "tt.tiled_axes";
static constexpr StringLiteral kStreamedFwdAttr   = "tt.streamed_axes";
static constexpr StringLiteral kContractionLike   = "tt.contraction_like";

static constexpr StringLiteral kTileAxesAttr   = "tt.tile_axes";
static constexpr StringLiteral kStreamAxesAttr = "tt.stream_axes";
static constexpr StringLiteral kTileSigAttr    = "tt.tile_sig";
static constexpr StringLiteral kSigIdAttr      = "tt.sig_id";

// todo: later unify the helpers between the two files -- seems can unify all except serialiseSig below

// Fetch `tt.axis_names` on any SSA value (op result or func arg)
static ArrayAttr axisAttr(Value v) {
  if (auto res = dyn_cast<OpResult>(v))
    return res.getOwner()->getAttrOfType<ArrayAttr>(kAxisNamesAttr);

  if (auto arg = dyn_cast<BlockArgument>(v)) {
    auto func = dyn_cast<FunctionOpInterface>(
        arg.getOwner()->getParentOp());
    return func.getArgAttrOfType<ArrayAttr>(arg.getArgNumber(), kAxisNamesAttr);
  }
  return {};
}

// Convert `ArrayAttr -> SmallVector<StringRef>`
static SmallVector<StringRef> asVec(ArrayAttr arr) {
  SmallVector<StringRef> v;
  if (arr)
    for (Attribute a : arr)
      v.push_back(cast<StringAttr>(a).getValue());
  return v;
}


static void attachAttr(Value v, StringLiteral key, Attribute a) {
  if (!a)
    return;

  if (auto res = dyn_cast<OpResult>(v)) {
    res.getOwner()->setAttr(key, a);
    return;
  }

  if (auto arg = dyn_cast<BlockArgument>(v)) {
    if (auto fn = dyn_cast<FunctionOpInterface>(
            arg.getOwner()->getParentOp()))
      fn.setArgAttr(arg.getArgNumber(), key, a);
  }
}

// Serialise one signature  {tile, stream}  ->  unique string
// Example:  "T:[SEQ_LEN] S:[HEAD_DIM]"
static StringAttr serialiseSig(MLIRContext *ctx,
                               ArrayRef<StringRef> tile,
                               ArrayRef<StringRef> stream) {
  std::string tmp;
  llvm::raw_string_ostream os(tmp);
  os << "T:[";
  llvm::interleaveComma(tile, os);
  os << "] S:[";
  llvm::interleaveComma(stream, os);
  os << ']';
  return StringAttr::get(ctx, os.str());
}

// turn StringVector -> ArrayAttr
static ArrayAttr makeArr(MLIRContext *ctx, ArrayRef<StringRef> v) {
  SmallVector<Attribute> elts;
  elts.reserve(v.size());
  for (StringRef s : v) elts.push_back(StringAttr::get(ctx, s));
  return ArrayAttr::get(ctx, elts);
}

//  Simple contraction detector (backup if trait is missing)

// todo-now: i think that's the weakest part -- I'm not yet clear in my mind, is it better to encode for each axis, how many times it gets re-used -- as a more general form of what I want to achieve here
static bool looksLikeContraction(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "tt.dot"         ||
         name == "tt.matmul"      ||
         name == "tt.batch_matmul"||
         name == "tt.reduce"      ||
         name == "tt.conv";
}


namespace {

// Optional promotion helper (kept unchanged; fitsRegisters() is still off)
static std::pair<SmallVector<StringRef>, SmallVector<StringRef>>
promoteTileAxes(SmallVector<StringRef> tile,
                SmallVector<StringRef> stream,
                Value sample) {

  if (tile.empty() && !stream.empty()) {
    StringRef cand = stream.front();
    tile.push_back(cand);
    stream.erase(stream.begin());

    // if (!fitsRegisters(cand, sample)) {
    //   // undo – stay purely streaming
    //   stream.insert(stream.begin(), cand);
    //   tile.clear();
    // }
  }
  // keep stable order so signatures are reproducible
  llvm::sort(tile);
  llvm::sort(stream);
  return {std::move(tile), std::move(stream)};
}

// anonymous namespace
}


void inferTiling(Block *entry) {
  if (!entry) return;

  Operation *top = entry->getParentOp();
  MLIRContext *ctx = top->getContext();

  llvm::DenseMap<StringAttr, unsigned> sig2id;
  unsigned nextId = 0;

  top->walk([&](Operation *op) {
    // must be contraction-like either by explicit trait or by backup check
    if (!op->hasAttr(kContractionLike) && !looksLikeContraction(op))
      return;

    // assume contraction ops have *one* result that carries all axes.
    Value out = op->getNumResults() ? op->getResult(0) : Value();
    auto outAxesArr = axisAttr(out);
    if (!outAxesArr) return;                      // safety guard
    SmallVector<StringRef> O = asVec(outAxesArr);

    // iterate every operand X of the op
    for (Value X : op->getOperands()) {
      auto xArr = axisAttr(X);
      if (!xArr) continue;                       // scalars / un-tagged

      SmallVector<StringRef> Xv = asVec(xArr);
      llvm::StringSet<>      Xset;
      for (StringRef s : Xv) Xset.insert(s);

      // one-pass split of O into {tile, stream}
      SmallVector<StringRef> tile, stream;
      tile.reserve(O.size());
      stream.reserve(O.size());

      // stream = O − X
      // tile   = O ∩ X
      for (StringRef ax : O)
        (Xset.contains(ax) ? tile : stream).push_back(ax);

      // optional promotion & canonical ordering
      std::tie(tile, stream) = promoteTileAxes(std::move(tile), std::move(stream), X);

      // build / intern signature 
      StringAttr sigStr = serialiseSig(ctx, tile, stream);

      unsigned id = sig2id.try_emplace(sigStr, nextId).first->second;
      if (id == nextId) ++nextId;                // new signature -> bump id

      auto tileArr   = makeArr(ctx, tile);
      auto streamArr = makeArr(ctx, stream);
      auto sigIdAttr = IntegerAttr::get(IntegerType::get(ctx, 32), id);

      // attach four attrs on the *value* (same style as axis names)
      attachAttr(X, kTileAxesAttr,   tileArr);
      attachAttr(X, kStreamAxesAttr, streamArr);
      attachAttr(X, kTileSigAttr,    sigStr);
      attachAttr(X, kSigIdAttr,      sigIdAttr);
    }
  });
}

} // namespace triton
} // namespace mlir
