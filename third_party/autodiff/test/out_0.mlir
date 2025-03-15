
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
should be Value defined by add op: %5 = "arith.addf"(%4, %0) <{fastmath = #arith.fastmath<none>}> : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
visiting arith.addf op
visiting arith.constant op
visiting tt.addptr op
visiting tt.load op
visiting tt.addptr op
visiting tt.make_range op
visiting tt.splat op
visiting tt.splat op
"builtin.module"() ({
  "tt.func"() <{arg_attrs = [{tt.divisibility = 4 : i32}, {tt.divisibility = 4 : i32}, {tt.divisibility = 4 : i32}], function_type = (!tt.ptr<f32>, !tt.ptr<f32>, i32) -> (), sym_name = "add_kernel", sym_visibility = "public"}> ({
  ^bb0(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32):
    %0 = "arith.constant"() <{value = dense<4.200000e+01> : tensor<4xf32>}> : () -> tensor<4xf32>
    %1 = "tt.make_range"() <{end = 4 : i32, start = 0 : i32}> : () -> tensor<4xi32>
    %2 = "tt.splat"(%arg0) : (!tt.ptr<f32>) -> tensor<4x!tt.ptr<f32>>
    %3 = "tt.addptr"(%2, %1) : (tensor<4x!tt.ptr<f32>>, tensor<4xi32>) -> tensor<4x!tt.ptr<f32>>
    %4 = "tt.load"(%3) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<4x!tt.ptr<f32>>) -> tensor<4xf32>
    %5 = "arith.addf"(%4, %0) <{fastmath = #arith.fastmath<none>}> : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %6 = "tt.splat"(%arg1) : (!tt.ptr<f32>) -> tensor<4x!tt.ptr<f32>>
    %7 = "tt.addptr"(%6, %1) : (tensor<4x!tt.ptr<f32>>, tensor<4xi32>) -> tensor<4x!tt.ptr<f32>>
    %8 = "tt.load"(%7) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 0, 0, 0>}> : (tensor<4x!tt.ptr<f32>>) -> tensor<4x!tt.ptr<f32>>
    "tt.return"() : () -> ()
  }) {noinline = false} : () -> ()
}) : () -> ()

