
--------------------------


- dump reuslts on internal trtion optmizaitons
    / TRITON_ALWAYS_COMPILE forces to ingore cache wt needing to manualy delete it (rm -rf /home/iaro/.triton/cache/)
  TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=1 _rpy 2> ./third_party/autodiff/test/dump.log
- dissable some otpimzaitons
  DISABLE_LLVM_OPT=1 DISABLE_PTXAS_OPT=1

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
  ```
