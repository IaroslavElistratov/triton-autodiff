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
    Block::BlockArgListType origBodyArgs = loopBody->getArguments(); // forOp.getRegionIterArgs();

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

    // getInitArgsMutable  // getInitArgs
    // forOp.getInitArgs() returns the initial values for the iteration arguments - these are the values provided when creating the ForOp that initialize the loop-carried variables before the first iteration.
    // Differences between related methods:
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
          // So, don't raise error here if grad is not found
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
    auto maybeNewLoop = forOp.replaceWithAdditionalYields(
        rewriter,                            // rewriter
        upstreamOutsideValues,               // new init operands
        /*replaceInitOperandUsesInLoop*/ false,
        // lambda that tells the helper what the loop must yield for each of the new iter-operands
        [&](OpBuilder &b, Location loc, ValueRange newIterArgs) {
          // forward the iter-args themselves to the next iteration
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

    for (auto [j, idx] : llvm::enumerate(gradResultIdx)) {
      Value fwdVal = origYieldOperands[idx];
      Value gradArg = upstreamInsideValues[j];
      // Only the results that were present in the outer gradMap received an
      localGradMap[fwdVal] = gradArg;
      llvm::errs() << "adding grads to localGradMap: " << printName(fwdVal) << " " << printName(gradArg) << "\n";
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






    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 2. Clone inner graph: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    if (DEBUG_PRINTS) llvm::errs() << "\n\n\n============ [handleForBackward] step 2 ============\n\n\n";

    // previously (in rewriteIntoBackward) I cloned all the ops including this forOp that we matched to (to be more precise what I cloned semantically bc fwd part, and was iterating over the backward (ie. original) part)
    // but previously I did not clone the body of that for loop -- here bc the for-loop I matched to -- represents backward op -- its body needs to contain both fwd and bwd
    // so cloning the body of that loop here

    // answer-now: note this is a brand new map -- otherwise I guess the values inside the loop body are already in the map "OrigToCloned" (not local) and thus the below loop does nothing
    // otherwise cloneSubtree does nothing
    IRMapping localOrigToCloned;

    // don't want to copy yeild itself
    Operation *beforeYieldOp = yieldOp->getPrevNode();
    builder.setInsertionPointToStart(loopBody);
    Operation *lastFwdOp = cloneSubtree(beforeYieldOp, localOrigToCloned, builder);
    // let the ops inserted during rewriting backward be inserted after the forward ops
    builder.setInsertionPointAfter(lastFwdOp);

    llvm::errs() << "[handleForBackward]cloned for-loop body:\n";
    forOp.print(llvm::errs());


    // todo: this logic is basically the same as in rewriteIntoBackward -- can put this logic into handleAllOps and share between these two funcs, instead of re-implementing it here

    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@






    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 3. Diff the inner graph: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    if (DEBUG_PRINTS) llvm::errs() << "\n\n\n============ [handleForBackward] step 3 ============\n\n\n";

    // remember in the outer graph you've duplicated nodes (isClone'ed) nodes represent the forward part of the outer graph,
    //  and you the way you got to the current handler HandleForBackward is by iterating over the original (not cloned) nodes in the outher graph
    //  and one of the ops in that graph was the for loop -- that's how you got here.
    //  Remember also, when you did the cloning, you populated origToCloned -- which specifies, for each value in the original (i.e. not cloned) part of the outer graph (IOW the part of the graph that you iterated over, re-writing each op there with derivative formulas) with all your handlers (called from handleAllOps)
    //  that map specifies for each Value there, what Value in the cloned graph (aka forward part of the outer graph) does it correspond to.
    //  So, all the op-handlers, when they need some intermediate value (to compute derivative) they use origToCloned to get a particular value in the forward part of the outer graph (bc the forward part will not be re-written: so it's safe to use intermediates from there)
    //  ==> But bc you now iterating over

    llvm::errs() << "[handleForBackward] recursively calling handleAllOps (on the body of the for-loop):\n";

    // Merge localGradMap into the pass-wide grad map for use inside handleAllOps.
    auto globalGradMap = pass.gradMap;
    pass.gradMap = localGradMap;
    // pass.gradMap.insert(localGradMap.begin(), localGradMap.end());

    // todo: pass lastFwdOp?
    pass.handleAllOps(*loopBody);

    // todo-now: also, need to add additional logic from rewriteIntoBackward, such as: deleting unmarked nodes, (?) deleting last store 
    //    1) clone
    //    2) handleStore
    //    3) handleLoad
    //    4) delete unused
    //  ==> yes, seems need all of these, so basically you need to call rewriteIntoBackward form here (recursively), and not just handleAllOps
    llvm::errs() << "[handleForBackward] done handleAllOps:\n";
    loopBody->print(llvm::errs());

    // llvm::errs() << "exit\n";
    // exit(1);


    // Restore original grad map entries after processing the body.
    pass.gradMap = globalGradMap;



    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@




    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@     4) for each Block arg, connect grad wrt that arg to the ouput @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    // map outputs of differenciated for-loop as grads of arguments to the for loop

    // grad wrt each of the for-loop inputs have been populated into the localGradMap, as a result of running HandleAllOps above
    // because I think my diff system computes grad wrt to original arguments, here I'm using them (org BlockAgs) to extract the grads from the grad map

    // NOTE: the order of iter-args you added is:
    //    [original iter args ... ] -> [ADDED upstream args ...] -> [todo: ADDED grads wrt outside values]
    // Need to preserve this order for args of the yield op (bc these mapped exactly in the same order to the iter args of the next iteration)

    // gradsArgs will be some of the added yield operands
    SmallVector<Value> gradsArgs;
    for (BlockArgument arg : origBodyArgs){
      Value gradArg = localGradMap[arg];
      gradsArgs.push_back(gradArg);
    }

    // I don't replace the yield right here bc later will do it together with added yield operands for
    //  the grads of outside Values as well -- avoids needing to replace the yield op twice

    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



    // @@@@@@@@@@@@@@@@@@@@ return more Values from each iteration @@@@@@@@@@@@@@@@@@@@

    // replace yield op (note also returns upstream grad and todo: grads wrt outside values)
    // Replace the old yield with a new one that yields all values
    builder.setInsertionPoint(yieldOp);

    // gather all values that you want to output from the new yield
    SmallVector<Value> allYieldOperands;
    // original fwd outputs of yield (I guess needed when original for-loop uses
    //  its own result as input to the next iteration). Wt preserving this arg,
    //  I guess can't correctly re-compute forward itermideats in next loop iter
    allYieldOperands.append(origYieldOperands.begin(), origYieldOperands.end());
    // newly added outputs
    allYieldOperands.append(gradsArgs.begin(), gradsArgs.end());
    // allYieldOperands.append(gradsOutsideValues.begin(), gradsOutsideValues.end());

    // replace original yield
    builder.create<scf::YieldOp>(yieldOp->getLoc(), allYieldOperands);
    yieldOp->erase();
    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@






    // @@@@@@@@@@@@@@@@@@@@ 5. populate outer-graph's gradMap @@@@@@@@@@@@@@@@@@@@


    // figure out which of the ouput idxs represent:
    //    - upstream grads;
    //    - grad wrt outside variables

    // populate grads wrt original inputs
    for (auto [i, value] : llvm::enumerate(origForOpOperands)){
      pass.gradMap[value] = forOp.getResult(i);
    }

    // todo: later
    // // populate grads wrt values accessed from outside
    // unsigned offsetGradsOutsideValues = origYieldOperands.size() + gradsArgs.size();
    // for (auto [i, value] : llvm::enumerate(accessedOutsideValues)){
    //   // I think this mapping from ouputs to vector of accessedOutsideValues
    //   // is valid bc "gradsOutsideValues" (which was used for creating additional outputs of yeild) is 1:1 with "accessedOutsideValues"
    //   pass.gradMap[value] = forOp.getOutput(offsetGradsOutsideValues + i);
    // }

    /*
    > Expected gradient in the map for Value: %23 = "tt.load"(%22) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<4x!tt.ptr<f32>>) -> tensor<4xf32>

        %16 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %17 = "tt.make_range"() <{end = 4 : i32, start = 0 : i32}> : () -> tensor<4xi32>
        %18 = "tt.splat"(%arg0) : (!tt.ptr<f32>) -> tensor<4x!tt.ptr<f32>>
        %19 = "tt.addptr"(%18, %17) : (tensor<4x!tt.ptr<f32>>, tensor<4xi32>) -> tensor<4x!tt.ptr<f32>>
        %20 = "tt.load"(%19) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<4x!tt.ptr<f32>>) -> tensor<4xf32>
        %21 = "tt.splat"(%arg1) : (!tt.ptr<f32>) -> tensor<4x!tt.ptr<f32>>
        %22 = "tt.addptr"(%21, %17) : (tensor<4x!tt.ptr<f32>>, tensor<4xi32>) -> tensor<4x!tt.ptr<f32>>
        // todo: this value is passed as one of the args to the for loop, now since I differenciated my for loop, it contains
        %23 = "tt.load"(%22) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<4x!tt.ptr<f32>>) -> tensor<4xf32>

        // answer-now: the for-loop returns one SSA value (24), but it actually contains 3 results (i32, i32, i32, tensor<4xf32>)
        %24 = "scf.for"(%16, %15, %14, %23) ({
        ^bb0(%arg2: i32, %arg3: tensor<4xf32>):
          %25 = "arith.sitofp"(%arg2) {autogradVisited = true, isCloned = true} : (i32) -> f32
          %26 = "tt.splat"(%25) {autogradVisited = true, isCloned = true} : (f32) -> tensor<4xf32>
          %27 = "arith.mulf"(%20, %26) <{fastmath = #arith.fastmath<none>}> {autogradVisited = true, isCloned = true} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %28 = "arith.addf"(%arg3, %27) <{fastmath = #arith.fastmath<none>}> {autogradVisited = true, isCloned = true} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %29 = "arith.sitofp"(%arg2) : (i32) -> f32
          %30 = "tt.splat"(%29) : (f32) -> tensor<4xf32>
          %31 = "arith.mulf"(%20, %30) <{fastmath = #arith.fastmath<none>}> : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          %32 = "arith.addf"(%arg3, %31) <{fastmath = #arith.fastmath<none>}> : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          "scf.yield"(%32) : (tensor<4xf32>) -> ()
        }) : (i32, i32, i32, tensor<4xf32>) -> tensor<4xf32>
    */

    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

  }

  // comment: temporarily removed for simplicity, for now don't handle grads wrt value accessed from the outside-graph

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
