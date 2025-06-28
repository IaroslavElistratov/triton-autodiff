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

#include "mlir/IR/PatternMatch.h" // IRRewriter for replaceWithAdditionalYields

namespace mlir {
namespace triton {


  void handleForBackward(scf::ForOp forOp, ConvertTritonToAutodiff& pass) {

    // pass.builder is an optional<OpBuilder> -- dereference the optional before using
    OpBuilder& builder = *pass.builder;

    // rm
    builder.setInsertionPoint(forOp);

    llvm::errs() << "handleForBackward\n";
    Block *loopBody = &forOp.getRegion().front();
    // Operation *yieldOp = &entryBlock->back();
    Operation *yieldOp = forOp.getBody()->getTerminator();

    SmallVector<Value> origYieldOperands(yieldOp->getOperands());
    // region iter-args exclude the induction variable; they line up 1-to-1 with
    // the init operands and the loop results
    ArrayRef<BlockArgument> origBodyArgs = forOp.getRegionIterArgs();

    // incudes the 3 args for loop bounds + loop-carry args
    SmallVector<Value> origForOpOperands(forOp.getOperands()); // forOp.getRegionIterArgs();


    // todo-now: consider reversing the iteration order of the backward loop
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
    if (DEBUG_PRINTS) llvm::errs() << "\n\n\n============ [handleForBackward] step 1 ============\n\n\n";

    // getInitArgs() - Returns the initial values for iteration arguments (values OUTSIDE the loop)
    // getRegionIterArgs() - Returns the BlockArguments for iteration arguments inside the loop body
    // unsigned origNumIterArgs = forOp.getNumRegionIterArgs();

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
        /*replaceInitOperandUsesInLoop*/ false,
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
    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@





    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 2. clone form yield: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

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
    //  (ie. original) part) but previously I did not clone the body of that for loop -- here bc the for-loop
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





    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 3. Diff the inner graph: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    // remember in the outer graph you've duplicated nodes (isClone'ed) nodes represent the forward part of the outer graph,
    //  and you the way you got to the current handler HandleForBackward is by iterating over the original (not cloned) nodes in the outher graph
    //  and one of the ops in that graph was the for loop -- that's how you got here.
    //  Remember also, when you did the cloning, you populated origToCloned -- which specifies, for each value in the original (i.e. not cloned) part of the outer graph (IOW the part of the graph that you iterated over, re-writing each op there with derivative formulas) with all your handlers (called from handleAllOps)
    //  that map specifies for each Value there, what Value in the cloned graph (aka forward part of the outer graph) does it correspond to.
    //  So, all the op-handlers, when they need some intermediate value (to compute derivative) they use origToCloned to get a particular value in the forward part of the outer graph (bc the forward part will not be re-written: so it's safe to use intermediates from there)
    //  ==> But bc you now iterating over

    if (DEBUG_PRINTS) llvm::errs() << "\n\n\n============ [handleForBackward] step 3 -- recursively calling handleForBackward ============\n\n\n";

    // Merge localGradMap into the pass-wide grad map for use inside handleAllOps.
    auto globalGradMap = pass.gradMap;
    auto globalOrigToCloned = pass.origToCloned;

    pass.gradMap = localGradMap;
    pass.origToCloned = localOrigToCloned;

    pass.lastFwdOp = lastFwdOp;
    // 1) clone 2) handleStore 3) handleLoad 4) delete unused
    //  ==> yes, seems need all of these, so call rewriteIntoBackward (not just handleAllOps) from here recursively
    pass.rewriteIntoBackward(*loopBody);

    llvm::errs() << "[handleForBackward] done handleAllOps:\n";
    loopBody->print(llvm::errs());


    // Restore original grad map entries after processing the body.
    pass.gradMap = globalGradMap;
    pass.origToCloned = globalOrigToCloned;

    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@




    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 4) for each Block arg, connect grad wrt that arg to the ouput @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    // map outputs of differenciated for-loop as grads of arguments to the for loop

    // grad wrt each of the for-loop inputs have been populated into the localGradMap, as a result of running HandleAllOps above
    // because I think my diff system computes grad wrt to original arguments, here I'm using them (org BlockAgs) to extract the grads from the grad map

    // NOTE: the order of iter-args you added is:
    //    [original iter args ... ] -> [ADDED upstream args ...] -> [todo: ADDED grads wrt outside values]
    // Need to preserve this order for args of the yield op (bc these mapped exactly in the same order to the iter args of the next iteration)

    // // gradsArgs will be some of the added yield operands
    // SmallVector<Value> gradsArgs;
    // for (BlockArgument arg : origBodyArgs){
    //   Value gradArg = localGradMap[arg];
    //   gradsArgs.push_back(gradArg);
    // }

    // @@@@@@@@@@@@@@@@@@@@ 5. populate outer-graph's gradMap @@@@@@@@@@@@@@@@@@@@

    if (DEBUG_PRINTS) llvm::errs() << "\n\n\n============ [handleForBackward] step 5 ============\n\n\n";

    /*
    [see exclidraw "REF-1"]
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

    // todo-now: add upstream grads from outside the loop as loop-carry initializers?

    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    if (DEBUG_PRINTS) llvm::errs() << "\n\n\n============ [handleForBackward] HANDLER FINISHED ============\n\n\n";

    // todo-now: use maybeAccumulate?
  }

  // comment: temporarily removed for simplicity, for now don't handle grads wrt value accessed from the outside-graph

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
