```
<!-- cmake -G Ninja ../triton -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON -DLLVM_CCACHE_BUILD=ON -->


rm -rf python/build && pip install -e python



- # minimal (aka run):
  python/build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt --convert-triton-to-autodiff third_party/autodiff/test/inp.mlir > third_party/autodiff/test/out_0.mlir 

- # + plot:
  python/build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt --convert-triton-to-autodiff --view-op-graph third_party/autodiff/test/inp.mlir > third_party/autodiff/test/out_0.mlir 2> third_party/autodiff/test/out_0.dot && dot -Tsvg third_party/autodiff/test/out_0.dot > third_party/autodiff/test/out_0.svg && rm third_party/autodiff/test/out_0.dot

- # dissable checks (useful to see transformed IR even thought it's not valid)
  python/build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt --no-implicit-module --verify-each=0  --convert-triton-to-autodiff third_party/autodiff/test/inp.mlir > third_party/autodiff/test/out_0.mlir

# Canonicalize the output to remove values with no uses
python/build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt --convert-triton-to-autodiff --canonicalize third_party/autodiff/test/out_0.mlir > third_party/autodiff/test/out_1.mlir


--------------------------


- dump reuslts on internal trtion optmizaitons
    / TRITON_ALWAYS_COMPILE forces to ingore cache wt needing to manualy delete it (rm -rf /home/iaro/.triton/cache/)
  TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=1 _rpy 2> ./third_party/autodiff/test/dump.log
- dissable some otpimzaitons
  DISABLE_LLVM_OPT=1 DISABLE_PTXAS_OPT=1

```

- use "mlir-opt -test-print-nesting" to get this printed

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
