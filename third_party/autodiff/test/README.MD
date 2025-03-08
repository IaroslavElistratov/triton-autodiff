python/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt --convert-triton-to-myarith custom_passes/inp.mlir > custom_passes/out_0.mlir

# Canonicalize the output to remove values with no uses
python/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt --convert-triton-to-myarith --canonicalize custom_passes/inp.mlir > custom_passes/out_1.mlir