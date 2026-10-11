builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<1x32x64xf32>, %1: tensor<1x64x64xf32>, %2: tensor<1x64x64xf32>) -> tensor<1x32x64xf32> {
    %3 = arith.constant {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %4 = tensor.splat %3 {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<1x32x64xf32>
    %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%0, %1 : tensor<1x32x64xf32>, tensor<1x64x64xf32>) outs(%4 : tensor<1x32x64xf32>) attrs =  {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
    ^bb0(%6: f32, %7: f32, %8: f32):
      %9 = arith.mulf %6, %7 : f32
      %10 = arith.addf %8, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<1x32x64xf32>
    %11 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.250000e-01 : f32
    %12 = tensor.splat %11 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x32x64xf32>
    %13 = tensor.empty() : tensor<1x32x64xf32>
    %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %12 : tensor<1x32x64xf32>, tensor<1x32x64xf32>) outs(%13 : tensor<1x32x64xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb1(%15: f32, %16: f32, %17: f32):
      %18 = arith.mulf %15, %16 : f32
      linalg.yield %18 : f32
    } -> tensor<1x32x64xf32>
    %19 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %20 = tensor.splat %19 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x32xf32>
    %21 = linalg.reduce ins(%14:tensor<1x32x64xf32>) outs(%20:tensor<1x32xf32>) dimensions = [2]
    (%22: f32, %23: f32) {
      %24 = arith.maximumf %22, %23 : f32
      linalg.yield %24 : f32
    }
    %25 = tensor.collapse_shape %21 [[0 : i64, 1 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x32xf32> into tensor<32xf32>
    %26 = tensor.expand_shape %25 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x32x1xf32>
    %27 = tensor.empty() : tensor<1x32x64xf32>
    %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14, %26 : tensor<1x32x64xf32>, tensor<1x32x1xf32>) outs(%27 : tensor<1x32x64xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb2(%29: f32, %30: f32, %31: f32):
      %32 = arith.subf %29, %30 : f32
      linalg.yield %32 : f32
    } -> tensor<1x32x64xf32>
    %33 = tensor.empty() : tensor<1x32x64xf32>
    %34 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%28 : tensor<1x32x64xf32>) outs(%33 : tensor<1x32x64xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb3(%35: f32, %36: f32):
      %37 = math.exp %35 : f32
      linalg.yield %37 : f32
    } -> tensor<1x32x64xf32>
    %38 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %39 = tensor.splat %38 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x32xf32>
    %40 = linalg.reduce ins(%34:tensor<1x32x64xf32>) outs(%39:tensor<1x32xf32>) dimensions = [2]
    (%41: f32, %42: f32) {
      %43 = arith.addf %41, %42 : f32
      linalg.yield %43 : f32
    }
    %44 = tensor.collapse_shape %40 [[0 : i64, 1 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x32xf32> into tensor<32xf32>
    %45 = tensor.expand_shape %44 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x32x1xf32>
    %46 = tensor.empty() : tensor<1x32x64xf32>
    %47 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %45 : tensor<1x32x64xf32>, tensor<1x32x1xf32>) outs(%46 : tensor<1x32x64xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb4(%48: f32, %49: f32, %50: f32):
      %51 = arith.divf %48, %49 : f32
      linalg.yield %51 : f32
    } -> tensor<1x32x64xf32>
    %52 = arith.constant {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %53 = tensor.splat %52 {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<1x32x64xf32>
    %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%47, %2 : tensor<1x32x64xf32>, tensor<1x64x64xf32>) outs(%53 : tensor<1x32x64xf32>) attrs =  {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
    ^bb5(%55: f32, %56: f32, %57: f32):
      %58 = arith.mulf %55, %56 : f32
      %59 = arith.addf %57, %58 : f32
      linalg.yield %59 : f32
    } -> tensor<1x32x64xf32>
    func.return %54 : tensor<1x32x64xf32>
  }
}
