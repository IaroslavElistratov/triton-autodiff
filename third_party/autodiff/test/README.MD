# Reproduce results

Please see instructions in the [main readme](../../../README.md#reproduce-results).

# Development commands

```shell
# "triton-opt" is e.g. python/build/cmake.linux-x86_64-cpython-3.12/bin/

# run
triton-opt --convert-triton-to-autodiff inp.ttir > out.ttir

# run, disable checks
triton-opt --no-implicit-module --verify-each=0 --convert-triton-to-autodiff inp.ttir > out.ttir

# run, plot
triton-opt --convert-triton-to-autodiff --view-op-graph inp.ttir > out.ttir 2> out.dot
dot -Tsvg out.dot > out.svg && rm out.dot
```