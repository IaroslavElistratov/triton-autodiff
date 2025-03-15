<!-- cmake -G Ninja ../triton -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON -DLLVM_CCACHE_BUILD=ON -->


rm -rf python/build && pip install -e python



- # minimal (aka run):
  python/build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt --convert-triton-to-myarith third_party/autodiff/test/inp.mlir > third_party/autodiff/test/out_0.mlir 

- # + plot:
  python/build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt --convert-triton-to-myarith --view-op-graph third_party/autodiff/test/inp.mlir > third_party/autodiff/test/out_0.mlir 2> third_party/autodiff/test/out_0.dot && dot -Tsvg third_party/autodiff/test/out_0.dot > third_party/autodiff/test/out_0.svg && rm third_party/autodiff/test/out_0.dot

- # dissable checks (useful to see transformed IR even thought it's not valid)
  python/build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt --no-implicit-module --verify-each=0  --convert-triton-to-myarith third_party/autodiff/test/inp.mlir > third_party/autodiff/test/out_0.mlir

# Canonicalize the output to remove values with no uses
python/build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt --convert-triton-to-myarith --canonicalize third_party/autodiff/test/out_0.mlir > third_party/autodiff/test/out_1.mlir



- original:
  module {
    tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg2: i32 {tt.divisibility = 4 : i32}) attributes {noinline = false} {
      %cst = arith.constant dense<4.200000e+01> : tensor<4xf32>
      %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
      %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
      %2 = tt.addptr %1, %0 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %3 = tt.load %2 : tensor<4x!tt.ptr<f32>>
      %4 = arith.addf %3, %cst : tensor<4xf32>
      %5 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
      %6 = tt.addptr %5, %0 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      tt.store %6, %4 : tensor<4x!tt.ptr<f32>>
      tt.return
    }
  }




arith.constant -> 1.
arith.addf -> 1 for each input

tt.load -> atomic add into that location of memory
tt.make_range -> treat as input constant?
tt.splat(scalar_arg) -> grad for scalar_arg, sum that range
tt.addptr -> ?
tt.store -> load op loading upstream grad for these vlaues (note requires a different memory location though)

// tt.func
// tt.return





--------------------------


// not defined?
    tt.func
    tt.get_program_id
    tt.return

// tt
    tt.addptr
    tt.splat -> sum over that range
    tt.load
    tt.store
    tt.make_range -> treat as inputting a constant?

// arith
    arith.addf
    arith.addi
    arith.muli
    arith.constant




- # use "mlir-opt -test-print-nesting" to get this printed

  ```mlir
  visiting op: 'tt.func' with 0 operands and 0 results
  5 attributes:
  - 'arg_attrs' : '[{tt.divisibility = 4 : i32}, {tt.divisibility = 4 : i32}, {tt.divisibility = 4 : i32}]'
  - 'function_type' : '(!tt.ptr<f32>, !tt.ptr<f32>, i32) -> ()'
  - 'noinline' : 'false'
  - 'sym_name' : '"add_kernel"'
  - 'sym_visibility' : '"public"'
  1 nested regions:

    // comment: regions don't contain anything other than blocks
    Region with 1 blocks:

      // comment: a Block has a list of arguments, and holds a list of Operations
      Block with 3 arguments, 0 successors, and 10 operations

        visiting op: 'arith.constant' with 0 operands and 1 results
        1 attributes:
        - 'value' : 'dense<4.200000e+01> : tensor<4xf32>'
        1 results:
          - Result 0 has 1 use:
            - arith.addf
        
        visiting op: 'tt.make_range' with 0 operands and 1 results
        2 attributes:
        - 'end' : '4 : i32'
        - 'start' : '0 : i32'
        1 results:
          - Result 0 has 2 uses:
            - tt.addptr
            - tt.addptr
        
        visiting op: 'tt.splat' with 1 operands and 1 results
        1 operands:
        - Operand produced by Block argument, number 0
        1 results:
          - Result 0 has 1 use:
            - tt.addptr
        
        visiting op: 'tt.addptr' with 2 operands and 1 results
        2 operands:
        - Operand produced by operation 'tt.splat'
        - Operand produced by operation 'tt.make_range'
        1 results:
          - Result 0 has 1 use:
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
          - Result 0 has 1 use:
            - arith.addf
        
        visiting op: 'arith.addf' with 2 operands and 1 results
        1 attributes:
        - 'fastmath' : '#arith.fastmath<none>'
        2 operands:
        - Operand produced by operation 'tt.load'
        - Operand produced by operation 'arith.constant'
        1 results:
          - Result 0 has 1 use:
            - tt.store
        
        visiting op: 'tt.splat' with 1 operands and 1 results
        1 operands:
        - Operand produced by Block argument, number 1
        1 results:
          - Result 0 has 1 use:
            - tt.addptr
        
        visiting op: 'tt.addptr' with 2 operands and 1 results
        2 operands:
        - Operand produced by operation 'tt.splat'
        - Operand produced by operation 'tt.make_range'
        1 results:
          - Result 0 has 1 use:
            - tt.store
        
        visiting op: 'tt.store' with 2 operands and 0 results
        3 attributes:
        - 'boundaryCheck' : 'array<i32>'
        - 'cache' : '1 : i32'
        - 'evict' : '1 : i32'
        2 operands:
        - Operand produced by operation 'tt.addptr'
        - Operand produced by operation 'arith.addf'
  ```


