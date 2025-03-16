
visiting op: 'tt.func' with 0 operands and 0 results
5 attributes:
 - 'arg_attrs' : '[{tt.divisibility = 4 : i32}, {tt.divisibility = 4 : i32}, {tt.divisibility = 4 : i32}]'
 - 'function_type' : '(!tt.ptr<f32>, !tt.ptr<f32>, i32) -> ()'
 - 'noinline' : 'false'
 - 'sym_name' : '"add_kernel"'
 - 'sym_visibility' : '"public"'
1 nested regions:
  Region with 1 blocks:
    Block with 3 arguments, 0 successors, and 10 operations
      
      visiting op: 'arith.constant' with 0 operands and 1 results
      1 attributes:
       - 'value' : 'dense<4.200000e+01> : tensor<4xf32>'
      1 results:
        - Result 0 shape = [4] has 1 use:
          - arith.addf
      
      visiting op: 'tt.make_range' with 0 operands and 1 results
      2 attributes:
       - 'end' : '4 : i32'
       - 'start' : '0 : i32'
      1 results:
        - Result 0 shape = [4] has 2 uses:
          - tt.addptr
          - tt.addptr
      
      visiting op: 'tt.splat' with 1 operands and 1 results
      1 operands:
       - Operand produced by Block argument, number 0
      1 results:
        - Result 0 shape = [4] has 1 use:
          - tt.addptr
      
      visiting op: 'tt.addptr' with 2 operands and 1 results
      2 operands:
       - Operand produced by operation 'tt.splat'
       - Operand produced by operation 'tt.make_range'
      1 results:
        - Result 0 shape = [4] has 1 use:
          - tt.load
      
      visiting op: 'tt.load' with 1 operands and 1 results
      5 attributes:
       - 'boundaryCheck' : 'array<i32>'
       - 'cache' : '1 : i32'
       - 'evict' : '1 : i32'
       - 'isVolatile' : 'false'
       - 'operandSegmentSizes' : 'array<i32: 1, 0, 0>'
      1 operands:
       - Operand produced by operation 'tt.addptr'
      1 results:
        - Result 0 shape = [4] has 1 use:
          - arith.addf
      
      visiting op: 'arith.addf' with 2 operands and 1 results
      1 attributes:
       - 'fastmath' : '#arith.fastmath<none>'
      2 operands:
       - Operand produced by operation 'tt.load'
       - Operand produced by operation 'arith.constant'
      1 results:
        - Result 0 shape = [4] has 1 use:
          - tt.store
      
      visiting op: 'tt.splat' with 1 operands and 1 results
      1 operands:
       - Operand produced by Block argument, number 1
      1 results:
        - Result 0 shape = [4] has 1 use:
          - tt.addptr
      
      visiting op: 'tt.addptr' with 2 operands and 1 results
      2 operands:
       - Operand produced by operation 'tt.splat'
       - Operand produced by operation 'tt.make_range'
      1 results:
        - Result 0 shape = [4] has 1 use:
          - tt.store
      
      visiting op: 'tt.store' with 2 operands and 0 results
      3 attributes:
       - 'boundaryCheck' : 'array<i32>'
       - 'cache' : '1 : i32'
       - 'evict' : '1 : i32'
      2 operands:
       - Operand produced by operation 'tt.addptr'
       - Operand produced by operation 'arith.addf'
      
      visiting op: 'tt.return' with 0 operands and 0 results
should be Value defined by add op: %4 = arith.addf %3, %cst : tensor<4xf32>
visiting arith.addf op
extracted upstream for addf, from the grad map: %7 = tt.load %6 {autogradVisited = true} : tensor<4x!tt.ptr<f32>>
visiting arith.constant op
Skipping visited
visiting tt.load op
Skipping visited
Skipping visited
Skipping visited
Skipping visited
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg2: i32 {tt.divisibility = 4 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {autogradVisited = true, end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.splat %arg0 {autogradVisited = true} : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 {autogradVisited = true} : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %3 = tt.splat %arg1 {autogradVisited = true} : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %4 = tt.addptr %3, %0 {autogradVisited = true} : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %5 = tt.load %4 {autogradVisited = true} : tensor<4x!tt.ptr<f32>>
    tt.store %2, %5 {autogradVisited = true} : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

