

- # original:
    ```mlir
    module {
      tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg2: i32 {tt.divisibility = 4 : i32}) attributes {noinline = false} {
        %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>

        // Load A
        %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
        %2 = tt.addptr %1, %0 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
        %3 = tt.load %2 : tensor<4x!tt.ptr<f32>>

        // add
        %cst = arith.constant dense<4.200000e+01> : tensor<4xf32>
        %4 = arith.addf %3, %cst : tensor<4xf32>

        // Store Out
        %5 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
        %6 = tt.addptr %5, %0 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
        tt.store %6, %4 : tensor<4x!tt.ptr<f32>>
        tt.return
      }
    }
    ```


- # after autodiff
    ```mlir
    tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg2: i32 {tt.divisibility = 4 : i32}) attributes {noinline = false} {
      %0 = tt.make_range {autogradVisited = true, end = 4 : i32, start = 0 : i32} : tensor<4xi32>

      // grad Out value -- initialized outside of the kernel to ONES
      %3 = tt.splat %arg1 {autogradVisited = true} : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
      %4 = tt.addptr %3, %0 {autogradVisited = true} : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %5 = tt.load %4 {autogradVisited = true} : tensor<4x!tt.ptr<f32>>

      // grad A pointer -- initialized outside of the kernel to ZEROS
      %1 = tt.splat %arg0 {autogradVisited = true} : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
      %2 = tt.addptr %1, %0 {autogradVisited = true} : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      // grad A computed
      tt.store %2, %5 {autogradVisited = true} : tensor<4x!tt.ptr<f32>>
      tt.return
    }
    ```

