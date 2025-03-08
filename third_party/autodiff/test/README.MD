python/build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt --convert-triton-to-myarith third_party/autodiff/test/inp.mlir > third_party/autodiff/test/out_0.mlir

# Canonicalize the output to remove values with no uses
python/build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt --convert-triton-to-myarith --canonicalize third_party/autodiff/test/inp.mlir > third_party/autodiff/test/out_1.mlir

