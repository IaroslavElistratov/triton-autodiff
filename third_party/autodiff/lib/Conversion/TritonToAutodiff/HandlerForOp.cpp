#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

// for reverse topo sort
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/STLExtras.h"  // enumerate, make_range, zip
#include "mlir/IR/Block.h"
#include <numeric>

#include "triton/Dialect/Triton/IR/Dialect.h"


#include "autodiff/include/Conversion/TritonToAutodiff/Passes.h"
#include "autodiff/include/Dialect/Autodiff/IR/Dialect.h"
#include "autodiff/include/Conversion/TritonToAutodiff/Handlers.h"
#include "autodiff/include/Conversion/TritonToAutodiff/Utils.h"
#include "autodiff/include/Conversion/TritonToAutodiff/UtilsIO.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/Signals.h" // report_fatal_error
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/IR/PatternMatch.h" // IRRewriter for replaceWithAdditionalYields

namespace mlir {
namespace triton {

// todo-high: for this pattern "elementwise" -- overwrite the HandleAllOps -> handleLoad -- to
// use regular tt.store and not atomics -- bc for the elementwise pattern atomics aren't needed


// [REF-7] reverseLoopCounter related

/// create an `arith.constant` whose type matches `ty` (either `index` or iN)
static mlir::Value buildConst(mlir::OpBuilder &b, mlir::Location loc, mlir::Type ty, int64_t v) {
  if (ty.isIndex())
    return b.create<mlir::arith::ConstantIndexOp>(loc, v);
  return b.create<mlir::arith::ConstantOp>(loc, ty, b.getIntegerAttr(ty, v));        // e.g. i32/i64
}

static mlir::scf::ForOp
reverseLoopCounter(mlir::scf::ForOp forOp, mlir::IRMapping *origToCloned, DenseMap<Value, Value> *gradMap = nullptr) {
  using namespace mlir;

  // ─────────────────── 1.  Extract literal bounds & step ──────────────────
  auto getConstInt = [](Value v, int64_t &out) -> bool {
    if (auto cIdx = v.getDefiningOp<arith::ConstantIndexOp>()) {
      out = cIdx.value();
      return true;
    }
    if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = llvm::dyn_cast<IntegerAttr>(cOp.getValue())) {
        out = intAttr.getValue().getSExtValue();
        return true;
      }
    }
    return false;
  };

  int64_t lb, ub, step;
  if (!getConstInt(forOp.getLowerBound(), lb)   ||
      !getConstInt(forOp.getUpperBound(), ub)   ||
      !getConstInt(forOp.getStep(),       step) || step <= 0)
    llvm::report_fatal_error("cannot reverse for-loop: dynamic or negative");

  const int64_t tripCnt = llvm::divideCeil(ub - lb, step);
  if (tripCnt <= 0)
    llvm::report_fatal_error("cannot reverse for-loop: empty loop");

  // ─────────────────── 2.  IR builder setup ───────────────────────────────
  OpBuilder b(forOp);
  b.setInsertionPoint(forOp);                // insert before old loop
  Location loc    = forOp.getLoc();
  Type      ivTy  = forOp.getInductionVar().getType();

  // Re‑usable typed constants
  Value c0    = buildConst(b, loc, ivTy, 0);
  Value c1    = buildConst(b, loc, ivTy, 1);
  Value cTrip = buildConst(b, loc, ivTy, tripCnt);
  Value cStep = buildConst(b, loc, ivTy, step);
  Value cUb   = buildConst(b, loc, ivTy, ub);


  // because my origToCloned maps Values and not Ops, but need to find the fwdOp itself
  //    - take Value ouput of the current (original ForOP)
  //    - map to the cloned Value
  //    - access producer of that cloned Value -- that is the cloned FwdForOp (which corresponds to the current forOp)
  Value fwdForOut = origToCloned->lookupOrNull(forOp.getResult(0));
  if (!fwdForOut)
    llvm::report_fatal_error("didn't find the fwd ForOp");
  scf::ForOp fwdForOp = llvm::dyn_cast<scf::ForOp>(fwdForOut.getDefiningOp());


  // build the new forward trip‑counter loop
  auto newFor = b.create<scf::ForOp>(
      // answer-now: initial values of the backward loop are the final values of the fwd loop
      // loc, c0, cTrip, c1, forOp.getInitArgs(),
      loc, c0, cTrip, c1, fwdForOp.getResults(),
      [&](OpBuilder &bodyBuilder, Location loc, Value t, ValueRange iterArgs) {
        // realIv = ub − step * (t + 1)
        Value tPlus1 = bodyBuilder.create<arith::AddIOp>(loc, t, c1);
        Value scaled = bodyBuilder.create<arith::MulIOp>(loc, cStep, tPlus1);
        Value realIv = bodyBuilder.create<arith::SubIOp>(loc, cUb, scaled);

        // clone original ops, remapping IV & loop‑carried values
        IRMapping map;
        map.map(forOp.getInductionVar(), realIv);
        map.map(forOp.getRegionIterArgs(), iterArgs);

        for (Operation &op : forOp.getBody()->without_terminator())
          bodyBuilder.clone(op, map);

        auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        SmallVector<Value> yieldVals;
        for (Value v : oldYield.getOperands())
          yieldVals.push_back(map.lookupOrDefault(v));

        bodyBuilder.create<scf::YieldOp>(loc, yieldVals);
      });

  // otherwise gets deleted later
  markVisited(b, visitedType::Inserted, newFor);

  if (gradMap) {
    for (auto [oldR, newR] : llvm::zip(forOp.getResults(), newFor.getResults()))
      if (auto it = gradMap->find(oldR); it != gradMap->end())
        (*gradMap)[newR] = it->second;
  }

  // replace & erase original loop
  forOp.replaceAllUsesWith(newFor.getResults());
  forOp.erase();
  return newFor;
}



//  essentially this function computes "accum" -- IOW accumulator at the current iteration BEFORE it was updated like so "accum_new = accum [op] curr"
//    in the original loop-body, "accum_new = accum [op] curr" was computed and the returned from the current iteration of the loop (to the next)
//    but here I return accum (before the update in the current iter) to the next iteration -- which as the effect of gradually undoing the updates as the loop progresses
//  also notcie after this fn executes, original "accum_new" is still present in the loop-body, it just has no op that use its result (I can delete it, but for now i just keep it hoping the cacnonicaltion will prune it -- my cleanup pass will not delete them bc they have isCloned attibute)
//  interpret inverse as fwd op: also note, when I add the "accum" in this fn -- I mark it as Inserted so that later when "RewriteIntoBackward -> handleAllOps" it doesn't try to differentiate the op that constructs the accum (the inverse op i insert here) itself
//    remember you already done the cloning earlier in the handlerForOp so that there's already exists backward part of forOpBody -- and bc it was cloned before reverseAccumulatorUpdates ran -- that copied bwd part of the forOp body uses accum_new (bc that bwd part of the graph was simply created a clone fwd part of the graph -- and the fwd part of the graph used accum_new, so bwd part uses accum_new as well)
//    and I want that bwd part of the body to use the result of my inverted op (accum) instead of the accum_new (which it is after cloning). So substitute all uses of "accum_new" with reconstructed "accum"
//    this allows 2 things:
//      1) replaces the yeild operand (accum_new -> accum) so now the loop yeilds the reversed "accum" to the next iter -- which corresponds to unwining the accumulator as the loop progresses
//      2) the bwd part of the loopBody now uses "accum" -- which allows to natively run the handleAllOps and the local derivatives will be correct bc to differentiate the "accum_new = accum * curr" wrt to "curr" -- need the accum which the mul op now uses (after the below replace all uses with)
Operation* reverseAccumulatorUpdates(scf::ForOp forOp, OpBuilder &builder, IRMapping &origToCloned) {

  Operation *yieldTerm = forOp.getBody()->getTerminator();
  Location loc = yieldTerm->getLoc();

  auto regionArgs = forOp.getRegionIterArgs();

  // rewriteIntoBwd inserts ops after fwd part of the graph but BEFORE that reconstructed
  // accum -- this creates invalid IR (because the ops that it inserts actually use values of the
  // reconstructed accum). Instead rewriteIntoBwd should insert ops AFTER the reconstructed accums,
  // track the *last* inverse operation we insert so that later during backward-sweep, handlers
  // can anchor the nodes they insert nodes after it
  Operation *lastInverseOp = nullptr;


  // I guess there's not only accum for the value but also another accumulator for the idxs (in fwd idxs are shifted
  // in each iter of the loop) -- so need to invert it as well. To generically determine which one of the yield operands is such accum
  // so the heuristic basically: 1) if it's a yeild operand; 2) which uses blockArgs as one of its operands -- that's the accum
  //
  // detecting the fwd-update pattern
  // look for a binary op that
  //  1) has one operand coming from the old accumulator (iterArgs[0]), and
  //  2) feeds the scf.yield that returns the new accumulator
  for (auto [idx, bbArg] : llvm::enumerate(regionArgs)) {
    Value yielded       = yieldTerm->getOperand(idx);
    Operation *updateOp = yielded.getDefiningOp();

    // passthrough, not an updateOp (e.g. directly a BlockAarg)
    if (!updateOp)
        continue;

    // is bbArg one of the operands of updateOp?
    bool usesIterArg = false;
    unsigned blockArgIdx = 0;
    // recognise only binary updates for now
    if (updateOp->getNumOperands() == 2) {
      if (updateOp->getOperand(0) == bbArg) {
        usesIterArg = true; blockArgIdx = 0;
      } else if (updateOp->getOperand(1) == bbArg) {
        usesIterArg = true; blockArgIdx = 1;
      }
    }

    // not an accumulator
    if (!usesIterArg)
      continue;

    unsigned otherPos = (blockArgIdx == 0) ? 1 : 0;
    Value other = updateOp->getOperand(otherPos);
    Value accum = updateOp->getOperand(blockArgIdx);
    Value stateAfter = yielded;
    // accum -- accumulator **after** this fwd lap
    // accPrev -- accumulate **before** this fwd lap (IOW inversed)

    // bc the loop now iterates from last to first, the value that enters the body (%range_18)
    // is the post‑update accumulator of the next forward iteration


    // [REF-8] move the reversed ptr-idx accum right to the begining of the for-loop body (before both fwd-part and the bwd-part),
    //  but keep reversed value accum below (IOW after) the fwd part (isCloned) of the loop body
    //
    // decide where to insert the inverse operation. If *all* operands it
    // needs are already available at the start of the block (i.e. defined
    // outside the body or are block arguments) emit the inverse at
    // the very top of the block so the forward-clone will immediately see
    // the reconstructed value. Otherwise place it right after the original updateOp
    bool operandsDominateHeader = true;
    Block *bodyBlock = forOp.getBody();
    auto dominatesHeader = [&](Value v) {
      if (auto *defOp = v.getDefiningOp())
        return defOp->getBlock() != bodyBlock;   // defined outside
      return true;                               // block argument
    };
    for (Value opVal : updateOp->getOperands()) {
      if (!dominatesHeader(opVal)) { operandsDominateHeader = false; break; }
    }

    OpBuilder::InsertionGuard ipg(builder);
    if (operandsDominateHeader)
      builder.setInsertionPointToStart(bodyBlock);
    else
      builder.setInsertionPointAfter(updateOp);


    // [done] if using [inverse-op](stateAfter, other) below
    //    [ORIG]
    //    %offsets_25 = arith.addi %range_19, %cst_6
    //    [REVERSED]
    //    %range_27 = arith.subi %offsets_25, %cst_6
    //  ==> wrong, it instead should be "subi(range_19, cst_6)"
    // After the fix -- using [inverse-op](accum, other)
    //    [ORIG]
    //     %offsets_25 = arith.addi %range_19, %cst_6
    //    [REVERSED]
    //     %range_27 = arith.subi %range_19, %cst_6

    // compute the value that the *previous* iteration saw
    Type resTy = accum.getType();
    Operation *inverseOp = nullptr;
    if (auto mulf = dyn_cast<arith::MulFOp>(updateOp)) {
      inverseOp = builder.create<arith::DivFOp>(loc, resTy, accum, other).getOperation();
    } else if (auto addf = dyn_cast<arith::AddFOp>(updateOp)) {
      inverseOp = builder.create<arith::SubFOp>(loc, resTy, accum, other).getOperation();
    } else if (auto addi = dyn_cast<arith::AddIOp>(updateOp)) {
      inverseOp = builder.create<arith::SubIOp>(loc, resTy, accum, other).getOperation();
    } else if (auto muli = dyn_cast<arith::MulIOp>(updateOp)) {
      inverseOp = builder.create<arith::DivSIOp>(loc, resTy, accum, other).getOperation();
    } else {
      continue; // unsupported accumulator kind yet
    }

    Value accPrev = inverseOp->getResult(0); // reconstructed pre-update

    // record the last inverse so callers can anchor their backward sweep.
    if (!lastInverseOp || lastInverseOp->isBeforeInBlock(inverseOp))
      lastInverseOp = inverseOp;

    // [REF-9] now ops to be differentiated use the reconstructed buffer (not the original
    // buffer) -- therefore when these "ops to be differentiated" will be matched with the
    // handlers, the differentiation rules will automatically use their operands as local
    // derivatives (these operands are reconstructed buffer) -- so backprop will correctly
    // use reconstructed buffer as local grad
    accum.replaceUsesWithIf(accPrev, [inverseOp](OpOperand &use) {
      Operation *user = use.getOwner();
      // rewrite only uses that are in the same block and appear *after* the inverse operation
      return user->getBlock() == inverseOp->getBlock() &&
             inverseOp->isBeforeInBlock(user) &&
             user != inverseOp; // skip the inverse's own operands
    });

    // todo:
    // updateOp->erase();

    // [REF-10]
    origToCloned.map(accPrev, accPrev);

    // probably should select them from the fwd part of the graph (Iscloned) -- so that the parent
    // nodes that this computation depends on don't get changed during fwd
    markVisited(builder, visitedType::Inserted, inverseOp);

    // crucial -- passes the reversed accum down to the next iter, which will:
    //  1) use it as its block args, and 
    //  2) inturn decrement the accum, and
    //  3) pass that again decremented accum to the iter after that, so on...
    //
    // now ops in the loop body which correspond to the fwd graph -- will use thse
    // reversed block-args (at the next iter) in the same way as they used original
    // block-args -- e.g. load data based on these offsets (so now it loads data
    // based on these reversed offsets -- great!)
    // 
    // %range_16:3 = scf.for %range_17 = %range_11 to %range_13 step %range_12 iter_args(%range_18 = %range#0, %range_19 = %range#1, %bwd_accum_20 = %bwd_accum) : i32 {
    //
    //   %offsets_24 = tt.addptr %offsets_23, %range_19
    //   %b_temp1 = tt.load %offsets_24
    //   ...
    //   //inserted ops that reverse the accum
    //   %range_26 = arith.divf %range_18, %b
    //   %range_27 = arith.subi %range_19, %cst_6
    //   ....
    //   scf.yield %range_26, %range_27, %bwd_accum_20
    //
    // pass stateBefore to the next iter
    yieldTerm->setOperand(idx, accPrev);


  }

  if (DEBUG_PRINTS) {
    llvm::errs() << "[reverseAccumulatorUpdates] reversed accum update:\n";
    forOp.print(llvm::errs());
  }

  return lastInverseOp;

}

  void handleForBackward(scf::ForOp forOp, ConvertTritonToAutodiff& pass) {
    llvm::errs() << "handleForBackward\n";

    // dereference the optional before using
    OpBuilder& builder = *pass.builder;


    // @@@@@@@@@@@@@@@@@@@@ 0. reverse iteration order by swapping loop bounds/step @@@@@@@@@@@@@@@@@@@@
    if (DEBUG_PRINTS) llvm::errs() << "\n\n\n============ [handleForBackward] step 0 -- reverseLoop bounds ============\n\n\n";

    // note reverseLoopCounter is meaningless by itself -- only makes sense as part of
    //  - swap initial/terminal values of an external buffer.
    //  - invert the body's arithmetic (+= -> ‑=, *= -> /=)
    //
    // for the IR i'm testing on, reverseLoopCounter has no effect bc the loop-body in my ir doesn't even use the loop counter for anything
    //  so all the semantics about "what does it mean to reverse a loop" is not controlled at all by swapping the order of induction variable
    //  (my PERF_for-loop-no-unroll test does not use loop counter at all) -- instead, it's controlled by the
    //  1) initial state of the accum which is set-up before the loop starts executing (Start from the final value %buf_final, not from the original zero), and 
    //  2) inverting the update to the accumulator (Undo the body update so that each step walks the accumulator backwards)

    // todo: bc later I run cloning which inserts cloned nodes right to the beginning of the forOp
    //  body -- this causes the cloned nodes to be inserted above even the recreated idxs below
    forOp = reverseLoopCounter(forOp, &pass.origToCloned, &pass.gradMap);
    llvm::errs() << "[reverseLoop] reversed loop bounds:\n";
    forOp.print(llvm::errs());
    /// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    builder.setInsertionPoint(forOp);

    Block *loopBody = &forOp.getRegion().front();
    // Operation *yieldOp = &entryBlock->back();
    Operation *yieldOp = forOp.getBody()->getTerminator();

    SmallVector<Value> origYieldOperands(yieldOp->getOperands());
    // region iter-args exclude the induction variable; they line up 1-to-1 with
    // the init operands and the loop results
    ArrayRef<BlockArgument> origBodyArgs = forOp.getRegionIterArgs();

    // incudes the 3 args for loop bounds + loop-carry args
    SmallVector<Value> origForOpOperands(forOp.getOperands()); // forOp.getRegionIterArgs();

    // reverse the iteration order of the backward loop:
    // computing forward values is problematic when the loop passes
    // intermediates between iterations. For autodiff correctness the
    // backward pass must walk the unrolled forward loop from the last
    // iteration toward the first. For a 5-iteration loop that means
    // starting from iteration 4, then 3, and so on. The problem is that
    // when I start from the end we no longer have access to the forward
    // loop iter-args produced by earlier iterations

    // Extract loop parameters
    // Value lowerBound = forOp.getLowerBound();
    // Value upperBound = forOp.getUpperBound();
    // Value step = forOp.getStep();
    // .getInductionVar();

    // @@@@@@@@@@@@@@@@@@@@ 1. add additional args for the upstream grad @@@@@@@@@@@@@@@@@@@@
    if (DEBUG_PRINTS) llvm::errs() << "\n\n\n============ [handleForBackward] step 1 -- add iter-args for upstream ============\n\n\n";

    // getInitArgs() - Returns the initial values for iteration arguments (values OUTSIDE the loop)
    // getRegionIterArgs() - Returns the BlockArguments for iteration arguments inside the loop body
    // unsigned origNumIterArgs = forOp.getNumRegionIterArgs();

    //  add upstream grads (from outside the loop) as loop-carry initializers

    // the number of upstream grads wrt loop outputs is the same as the number of outputs (one upstream for each ouput)
    SmallVector<Value> upstreamOutsideValues; // Grad values for results that have gradients
    // save the indices, because after the loop is replaced I can
    // no longer iterate over forOp.getResults() and look them up in
    // pass.gradMap. The replacement introduces new SSA values that are not
    // in the map, so record indices instead of the Value objects whose
    // identity will change (after I replace the loop below)
    SmallVector<unsigned> gradResultIdx;     // Indices of the loop results that have gradients
    unsigned resIdxCtr = 0;
    for (Value v : forOp.getResults()) {
        // use pass.gradMap.find(v) instead of operator[], so we no longer create "dummy" entries with a null Value
        auto it = pass.gradMap.find(v);
        if (it == pass.gradMap.end()) {
          // not all forOp results may have grads -- that's ok as long as the ones that don't have grad are basically offset-calculations
          // this will be checked in handleAllOps (if a value turns out not be offset calculation is matched -- the grad lookup fail there)
          // so here safe to skip the assert.
          // IOW: in my for-loop-no-unroll test the second loop-carry value is used only for indexing, so it is correct
          // that it has no gradient. So, no need to assert here that every forOp result must have its grad.
          // Also, not having the assert here doesn't sacrifice robustness, bc you will fail in handleAllOps if
          // you try to fetch a gradient for a value that does not have one.
          // On the other hand, if the value will never be matched in handleAllOps (e.g. if it's an integer offset calculation)
          // then it's totally fine for a Value (corresponding to such loopCarry) to not have grads.
          // So, don't raise error here if grad is not found;
          // e.g. these don't have grads: (indices, integer accumulators, pointers)
          continue;
        }

        Value gradVal = it->second;
        if (gradVal) {
            upstreamOutsideValues.push_back(gradVal);
            gradResultIdx.push_back(resIdxCtr);
        }
        ++resIdxCtr;
    }

    // Add block arguments to the loop body for each new iteration argument
    // Append the upstream gradients as new loop-carried values
    //
    // MLIR requires: num_initOperands == num_regionIterArgs == num_results,
    // but seems cannot mutate the result (`Operation` result seems immutable).
    // So, use and MLIR helper that rebuilds the loop with additional iter-operands.
    // helper expects a RewriterBase, use IRRewriter (simpler than PatternRewriter)
    mlir::IRRewriter rewriter(builder.getContext());
    rewriter.setInsertionPoint(forOp);

    // the below keeps original fwd outputs of yield (I guess needed when original for-loop uses
    //  its own result as input to the next iteration). Wt preserving this arg,
    //  I guess can't correctly re-compute forward itermideats in next loop iter
    auto maybeNewLoop = forOp.replaceWithAdditionalYields(
        rewriter,                            // rewriter
        upstreamOutsideValues,               // new init operands
        // otherwise every iteration of the loop re-reads the initial values of the loop-iter-values in each iteration
        // MLIR doesn't go through the loop body and swap every use of the init operand for the corresponding block argument that is carried from the previous iteration
        //  Init operand – the SSA value that lives outside the loop and seeds the first iteration.
        //  Block argument – the per-iteration version that is produced by the last scf.yield and visible only inside the loop body of the next lap.
        // When the flag is false, any operation in the body that was originally wired to the init operand continues to read that same outside value every time the body executes. The block argument still exists and is threaded through the header/yield machinery, but nothing in the body looks at it unless I rewired those uses myself
        /*replaceInitOperandUsesInLoop*/ true,
        // lambda that tells the helper what the loop must yield for each of the new iter-operands
        [&](OpBuilder &b, Location loc, ValueRange newIterArgs) {
          // forward the iter-args themselves to the next iteration
          // todo:
          // adding iter-args for upstream seems only required when the value must be updated on every
          // iteration -- if not (which is the case for for-loop-no-unroll example)
          // then can just access the upstream value (inside the loop) from outside the loop directly
          // (available read-only in every iteration)
          return SmallVector<Value>(newIterArgs.begin(), newIterArgs.end());
        });

    if (failed(maybeNewLoop))
      llvm::report_fatal_error("[handleForBackward] failed to extend loop with upstream grads");

    // Update our handle to the (replacement) loop and related helpers.
    forOp = cast<scf::ForOp>(maybeNewLoop->getOperation());
    // Recompute commonly used pointers after the replacement.
    loopBody = &forOp.getRegion().front();
    yieldOp = forOp.getBody()->getTerminator();

    // these block arguments correspond 1-to-1 to the newly appended
    // iter-args and represent the same upstream values, but inside the
    // loop body;
    // `take_back` already returns an ArrayRef, so no copy is necessary
    ArrayRef<BlockArgument> upstreamInsideValues =
        forOp.getRegionIterArgs().take_back(upstreamOutsideValues.size());

    // Update the yield operation to yield values for the new iteration arguments
    // Later, when the grad is computed



    // ********  add additional elements to the grad map ********
    llvm::errs() << "add additional elements to the grad map\n";

    // used for later differentiation
    // populate upstream grad otherwise errors handleAllOps errors (expected gradient in the map)
    // from the outer grad, you do have grad wrt %7, but when the inner graph starts iterating from the back
    // and sees %11 (does not see %7) -- it expects grads wrt be present in the map. So re-map
    // grad %11 -> grad %7 -- so that re-writing inner graph can proceed (in handleAllOps)

    // initialise a separate grad map that (to be used
    // in handleAllOps when processing the loop body)
    llvm::DenseMap<Value, Value> localGradMap;

    // can't just iterate over added yeild operands and map these 1:1 to forOp results (at
    // the same idx as the yield operand). Because they are not necessarily 1:1 -- e.g. if
    // forOp has 2 ouput and only 1st ouput has grads wrt to it, and we add 1 more yield
    // operand -- naive that logic would extract 2nd forOp output grad (which is incorrect)

    // Map each ORIGINAL forward yield value that actually has a gradient to the
    // corresponding upstream-gradient block argument. The order of
    // `upstreamInsideValues` matches the order in which we added iter args, i.e.
    // exactly the subset of loop results that have gradients.

    assert(gradResultIdx.size() == upstreamInsideValues.size() && "Mismatch results indices vs inside values");


    // 1. iterate over each original yield operand
    // 2. for each operand, init its grads with the loop-carry
    //    values I added above (these loop-carry[s] represent the
    //    upstream grads wrt corresponding yield outputs)
    // upstreamInsideValues  – the extra iter-args we just added
    // gradResultIdx         – indices of loop results that do have grads
    for (auto [j, idx] : llvm::enumerate(gradResultIdx)) {
      Value yieldOperand = origYieldOperands[idx];
      Value gradArg = upstreamInsideValues[j];
      // Only the results that were present in the outer gradMap received an
      localGradMap[yieldOperand] = gradArg;
      llvm::errs() << "adding grads to localGradMap: " << printName(yieldOperand) << " " << printName(gradArg) << "\n";
    }

    // only iterating over the added yeiled args (not over all yeild args)
    // grad wrt each operand of yeild, is corresponding to the newly added "upstream" loop iter arg
    unsigned upstreamIdx = upstreamInsideValues.size();
    assert(upstreamIdx == upstreamInsideValues.size() &&
           "Did not consume all upstream gradient block arguments");

    // print the contents
    for (const auto &kv : localGradMap) {
      const Value key = kv.first;
      const Value val = kv.second;
      // check nulls
      if (key && val)
        llvm::errs() << "[upstreamInsideValues] key: " << printName(key) << "  ->  value: " << printName(val) << '\n';
    }

    llvm::errs() << "[handleForBackward] added iter-args for upstream:\n";
    forOp.print(llvm::errs());
    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@




    // todo-med: it sets insertion point ABOVE the loop-counter computations added in step 0

    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 2. clone from yield @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    if (DEBUG_PRINTS) llvm::errs() << "\n\n\n============ [handleForBackward] step 2 - clone from yeild ============\n\n\n";

    /* the reason I do cloning here, even when calling rewriteIntoBackward below (which also does cloning), is
      the cloning in rewriteIntoBackward is done from the Store[s] (which makes sense to use store[s] as terminal
      nodes for differentiation in the "inline" pattern) but in the forOp body there can be ops which you may want
      to diff after the Store[s] (which is not the case in the "inline" patter)
      Specifically there can be ops after StoreOp[s] but before yield op.
      I want to make sure that these get cloned as well -- thus do the copying there as well.
      Then rewriteIntoBackward does its, but only copying the subgraphs leading to the StoreOps
      -- both clones (1) from yield op (here) and (2) from store ops (in rewriteIntoBackward),
      are needed bc they clone from different "anchors" sort of speak.

      And then another, independent reason (for cloning here), is that forOp body may simply not have any StoreOp
      (e.g. for-loop-no-unroll example) in which case the logic in rewriteIntoBackward (which does cloning based
      on StoreOps) will clone nothing.

      Also when calling rewriteIntoBackward from inside the forOpHandler, you can't just omit that "clone based on storeOps logic"
      bc in general a forOp body may have the StoreOps.

      Seems fine that there can be overlap in some nodes they might want to clone -- bc when calling rewriteIntoBackward
      I'm passing localOrigToCloned, so no redundant cloning should happen (bc the nodes that appear in both: coming form
      yeild, and coming from store -- will already be in the localOrigToCloned thus they aren't needed to be cloned again there)
    */

    // Clone inner graph
    //  previously (in outer rewriteIntoBackward) I cloned all the ops including this forOp that we matched
    //  to (to be more precise what I cloned semantically bc fwd part, and was iterating over the backward
    //  (ie. original) part) but previously I did not clone the body of that loop -- here bc the for-loop
    //  I matched to -- represents backward op -- its body needs to contain both fwd and bwd so cloning the body
    //  of that loop here

    // note this is a brand new map -- otherwise I guess the values inside
    // the loop body are already in the map "OrigToCloned" (not local) and
    // thus the below loop does nothing otherwise cloneSubtree does nothing
    IRMapping localOrigToCloned;

    // in handleAllOps handlers use "pass.origToCloned.lookup" to grab fwd values of operands
    // of an op that they matched to, because some of my fwd args are block args (and thus weren't cloned when calling clone above)
    // where were not recorded in the origToCloned map, so when handlers to the lookup on them it fails.
    // origToCloned.lookup(v) is used every time a handler needs the forward value of v
    for (BlockArgument ba : loopBody->getArguments())
        localOrigToCloned.map(ba, ba);   // identity

    // don't want to copy yeild itself
    builder.setInsertionPointToStart(loopBody);
    Operation *clonedYield = cloneSubtree(yieldOp, localOrigToCloned, builder);

    // todo: the only place that used lastFwdOp is handleStoreBackward -- get rid of that concept all together

    // MLIR requires scf.yield to be the last operation in a block,
    // cloning the fwd sub-graph including the original scf.yield
    // causes the clone to appear before the still-existing original ops,
    // leaving illegal code where non-terminators follow the block terminator.
    // So, pick the operand that is **latest** in the block to be used as
    // lastFwdOp and delete the cloned yeild [REF-3]
    Operation *lastFwdOp = nullptr;
    for (Value v : clonedYield->getOperands()) {
      if (Operation *def = v.getDefiningOp()) {
        // if an operand is a block-argument it has no defining op, so skip it
        if (!lastFwdOp || lastFwdOp->isBeforeInBlock(def))
          lastFwdOp = def;
      }
    }
    clonedYield->moveAfter(yieldOp);
    yieldOp->erase();
    // this is what later handlers should treat as "the last forward op".
    llvm::errs() << "lastFwdOp: " << *lastFwdOp << "\n";

    llvm::errs() << "[handleForBackward]cloned for-loop body:\n";
    forOp.print(llvm::errs());
    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



    // ################ 3  Recreate acc_before for every accumulator slot ################

    if (DEBUG_PRINTS) llvm::errs() << "\n\n\n============ [handleForBackward] step 3 - reversing accum updates ============\n\n\n";


    // when inverting the accumulate -- I need to change (in the outside graph) what inputs get passed to
    // the inverted-loop -- specifically by default i think it pass the initial (zero) state of the
    // buffer -- but here I instead need to pass the last state of the buffer
    //
    // for reversing accum, note the after cloning in handleForBackward -- the for-op body graph looks like
    // so. I do the cloning to recompute fwd intermideats in bwd (and avoid storing them from fwd) -- so
    // after cloning there semantically 2 parts to this body: fwd-part-cloned and fwd-part, then from handleForBackward
    // I re-write fwd-part into bwd-part -- and that bwd computations uses the fwd intermideats from fwd-part-cloned (see log)
    //
    // [RM]
    // So when reverse the accum I guess only want to modify the fwd-part-cloned? but currently it seems modifying
    // the fwd-part (which will be re-written into backward) when rewriteIntoBackward is called. I think want to use origToCloned
    // to map the yeild operands to the cloned values and then reverse the accum in these cloned part of the subgraph (but not in
    // the part of the subgraph which will be re-writen into bwd)

    Operation *lastInverseOp = reverseAccumulatorUpdates(forOp, builder, localOrigToCloned);
    if (lastFwdOp->isBeforeInBlock(lastInverseOp)){
      lastFwdOp = lastInverseOp;
    }

    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@




    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 4. Diff the inner graph: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    // remember in the outer graph you've duplicated nodes (isClone'ed) nodes represent the forward part of the outer graph,
    //  and you the way you got to the current handler HandleForBackward is by iterating over the original (not cloned) nodes in the outher graph
    //  and one of the ops in that graph was the for loop -- that's how you got here.
    //  Remember also, when you did the cloning, you populated origToCloned -- which specifies, for each value in the original (i.e. not cloned) part of the outer graph (IOW the part of the graph that you iterated over, re-writing each op there with derivative formulas) with all your handlers (called from handleAllOps)
    //  that map specifies for each Value there, what Value in the cloned graph (aka forward part of the outer graph) does it correspond to.
    //  So, all the op-handlers, when they need some intermediate value (to compute derivative) they use origToCloned to get a particular value in the forward part of the outer graph (bc the forward part will not be re-written: so it's safe to use intermediates from there)
    //  ==> But bc you now iterating over

    if (DEBUG_PRINTS) llvm::errs() << "\n\n\n============ [handleForBackward] step 4 -- recursively calling handleForBackward ============\n\n\n";

    // Merge localGradMap into the pass-wide grad map for use inside handleAllOps.
    auto globalGradMap = pass.gradMap;
    auto globalOrigToCloned = pass.origToCloned;

    pass.gradMap = localGradMap;
    pass.origToCloned = localOrigToCloned;

    pass.lastFwdOp = lastFwdOp;
    // 1) clone 2) handleStore 3) handleLoad 4) delete unused
    //  ==> yes, seems need all of these, so call rewriteIntoBackward (not just handleAllOps) from here recursively
    pass.rewriteIntoBackward(*loopBody);
    llvm::errs() << "[handleForBackward] rewriteIntoBackward done\n";


    // NOTE: needed bc "pass.gradMap = localGradMap;"" create a separate independent copy which does not update localGradMap
    // thus in step 5 it failed bc localGradMap was never updated and thus didn't containt grads wrt original buffer (blockArg)
    localGradMap = pass.gradMap;    // pull back the new entries
    // restore original grad map entries after processing the body
    pass.gradMap = globalGradMap;
    pass.origToCloned = globalOrigToCloned;

    loopBody->print(llvm::errs());

    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@




    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 5) for each Block arg, connect grad wrt that arg to the ouput @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    // map outputs of differentiated for-loop as grads of arguments to the for loop

    // grad wrt each of the for-loop inputs have been populated into the localGradMap, as a result of running HandleAllOps above
    // because I think my diff system computes grad wrt to original arguments, here I'm using them (org BlockAgs) to extract the grads from the grad map

    // NOTE: the order of iter-args you added is:
    //    [original iter args ... ] -> [ADDED upstream args ...] -> [todo: ADDED grads wrt outside values]
    // Need to preserve this order for args of the yield op (bc these mapped exactly in the same order to the iter args of the next iteration)


    // [REF-6-EXC]
    // Replace the placeholder upstream iter-arg yields with real gradients
    //
    // when i created the extended loop above i forwarded the
    // upstream iter-args unchanged. That is fine for additive accumulators "accum += curr"
    // (where local grad = 1, so upstream grads wrt the final loop output accum **same as**
    // upstream grad wrt accum **in every iteration**).
    // But wrong for cases such as "accum *= curr" where the grad depends on the values
    // computed in the *current* iteration (local grad for "accum" is "curr")
    // -- therefore differs from iteration to iteration.
    // After differentiating the loop body (rewriteIntoBackward) now have the true gradients of
    // each original yield operand available in "localGradMap". Replace the
    // previously forwarded values in the "scf.yield" with those gradients so
    // that the next backward iteration receives the correct upstream value


    {
      OpBuilder::InsertionGuard guard(builder);

      Operation *currentYield = forOp.getBody()->getTerminator();
      unsigned numOrigYield   = origYieldOperands.size();

      // does not include induction value;
      // 1:1 with loop-carry args, and with loop outputs
      ArrayRef<BlockArgument> bodyArgs = forOp.getRegionIterArgs();


      // gradResultIdx was computed before adding the extra grad-carrying iter-args,
      // but those new iter-args were appended to the end of the list, so the indices
      // of all original loop-carried values stayed the same; therefore the same
      // gradResultIdx can safely index forOp.getRegionIterArgs()
      for (auto [j, idx] : llvm::enumerate(gradResultIdx)) {

        // after  reverseAccumUpdate the block arg (accum_new) is not used as an operand to the ops being differentiated (bc you
        // replace all uses of "accum_new" with the reconsturcted "accum") -- so here when I try to lookup grad wrt block arg
        // (accum_new) I can't find it -- the correct thing is to use grad wrt accum;
        //
        // what happens after I “rewire” the accumulator reverseAccumulatorUpdates
        // creates accPrev (value before this forward lap)
        // rewrites every later use of accIn (%range_9) to accPrev (%range_19)
        // So no operation that the autodiff sweep visits afterwards
        // ever uses the original block argument, so localGradMap.lookup(loopArg) errs out
        //
        // Value loopArg   = bodyArgs[idx];   // e.g. %accum_in
        //
        // I think it lives under grad range_24 (which is bwd_out_23 here) that's why using yeild-arg
        Value yeildArg  = currentYield->getOperand(idx);      // acc_prev
        Value gradVal   = localGradMap.lookup(yeildArg);

        // // must propagate the gradient wrt the iteration argument (value before the update), not wrt the value just yielded
        // llvm::errs() << "checking grad wrt blockArg at idx: " << idx << "\n";

        if (!gradVal){
          // llvm::errs() << "loop body BlockArg " << loopArg << " does not have grad (after running rewriteIntoBackward) \n";
          llvm::errs() << "loop body yeildArg " << yeildArg << " does not have grad (after running rewriteIntoBackward) \n";
          llvm::report_fatal_error("[handleForBackward] gradient for loop-carried value not found in localGradMap");
        }

        // update the corresponding operand in scf.yield
        currentYield->setOperand(numOrigYield + j, gradVal);

        // todo: use maybeAccumulate?
      }
    }

    if (DEBUG_PRINTS) llvm::errs() << "\n\n\n============ [handleForBackward] step 5 -- for each Block arg, connect grad wrt that arg to the ouput ============\n\n\n";
    loopBody->print(llvm::errs());



    // @@@@@@@@@@@@@@@@@@@@ 6. populate outer-graph's gradMap @@@@@@@@@@@@@@@@@@@@

    if (DEBUG_PRINTS) llvm::errs() << "\n\n\n============ [handleForBackward] step 6 -- populate outer-graph's gradMap ============\n\n\n";

    /*
    [exclidraw "REF-1"]
    - the reverse loop carries a gradient accumulator in each extra iter-arg
    - each iter updates that accumulator locally and yields it
    - after the final iter, the accumulator holds the gradient wrt the original loop input, which the scf.for result contains
    */

    // populate grads wrt original inputs
    unsigned origNumIter = origYieldOperands.size();
    auto initArgs = forOp.getInitArgs();
    for (auto [j, idx] : llvm::enumerate(gradResultIdx)) {
      // map the corresponding init operand to the appended gradient result
      pass.gradMap[initArgs[idx]] = forOp.getResult(origNumIter + j);
    }

    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    if (DEBUG_PRINTS) llvm::errs() << "\n\n\n============ [handleForBackward] HANDLER FINISHED ============\n\n\n";

  }

  // comment: temporarily removed for simplicity, for now don't handle grads wrt value accessed from the outside

    // // populate grads wrt values accessed from outside
    // unsigned offsetGradsOutsideValues = origYieldOperands.size() + gradsArgs.size();
    // for (auto [i, value] : llvm::enumerate(accessedOutsideValues)){
    //   // I think this mapping from ouputs to vector of accessedOutsideValues
    //   // is valid bc "gradsOutsideValues" (which was used for creating additional outputs of yeild) is 1:1 with "accessedOutsideValues"
    //   pass.gradMap[value] = forOp.getOutput(offsetGradsOutsideValues + i);
    // }

    // // @@@@@@@@@@@@@@@@@@@@ handle grads wrt values accessed in the loop form ouside the loop @@@@@@@@@@@@@@@@@@@@
    // //                        STEP 1/2: Figure out how many new loop-carry variable to add (for grad wrt outer values)

    // /*
    //   Iterate over each nodes in the body of the for-loop and record
    //   "outside" Value[s] (i.e. the ones that are not: 1. loop args; 2. intermideats in the loop body)
    //   that are accessed by nodes in the loop body
    // */


    // // add them into some kind of map
    // SmallVector<Value> accessedOutsideValues;

    // for (Operation &op : entryBlock->getOperations()) {

    //   // if (isa<scf::YieldOp>(op))
    //   //   continue;

    //   for (Value operand : op.getOperands()) {
    //       if (operand.getParentBlock() != entryBlock){
    //         llvm::errs() << "[handleForBackward] for-loop operand " << operand << "is accessed from outside\n";
    //         accessedOutsideValues.push_back(operand);
    //       }
    //   }

    // }

    // /*
    //   For each one of these Value,
    //     - add an additional loop carry value (and thus the loop output value, bc the number of loop-carry and loop-ouput always should match the the scf.loop definition)
    //     - add to yield in backward loop (to pass between the loop iterations)

    //   Later on will treat them as grads wrt these "outside" Values
    // */

    // // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@





    // // @@@@@@@@@@@@@@@@@@@@ 5. handle grads wrt values accessed in the loop form ouside the loop @@@@@@@@@@@@@@@@@@@@
    // //                            STEP 2/2: add additional outputs

    // // gather all grads wrt outside values (these grads are SSA values inside the loop)
    // SmallVector<Value> gradsOutsideValues;
    // for (auto v : accessedOutsideValues){
    //   Value gradOutsideVal = localGradMap[v];
    //   gradsOutsideValues.push_back(gradOutsideVal);
    // }

    // // [OLD]
    // //    now gradMap contains grads wrt all the Values inside the body of the for-loop
    // //    including grads for "outside" Values -- so now iterate over gradMap,
    // //    checking if a given element is in "outsideMap" -- if so, perform some post processing modifications
    // //      - or just initialize a new origToCloned and pass it gradMap here -- so that every item there is related to body of this loop (not related to the outside graph, that this ops emeded into)
    // //    - 1) remove it from the gradMap (will add back in step 5)
    // //    - 2) connect its gradient Value (some value inside the for-loop body) to yield
    // //    - 3) add an additional loop-cary variable
    // //    - 4) [this more about this step] insert accumulate grad node (before it's connected to yield in step 2 above)
    // //    - 5) add that output from the for loop (same things as the loop cary arg above, but now output of the entire ForOp node) back to the gradMap -- adding the previous key (you removed in step 1 -- in this case %3) to the new Value output of the for loop (from step 2)


    // // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


} // namespace triton
} // namespace mlir
