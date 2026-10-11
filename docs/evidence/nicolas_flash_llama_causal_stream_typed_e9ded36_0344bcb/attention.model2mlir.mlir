builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<1x64x64xf32>, %1: tensor<1x64x2048xf32>, %2: tensor<1x64x2048xf32>, %3: tensor<1x2048x64xf32>) -> tensor<1x64x64xf32> {
    %4 = arith.constant {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %5 = tensor.splat %4 {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<1x64x2048xf32>
    %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%0, %1 : tensor<1x64x64xf32>, tensor<1x64x2048xf32>) outs(%5 : tensor<1x64x2048xf32>) attrs =  {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
    ^bb0(%7: f32, %8: f32, %9: f32):
      %10 = arith.mulf %7, %8 : f32
      %11 = arith.addf %9, %10 : f32
      linalg.yield %11 : f32
    } -> tensor<1x64x2048xf32>
    %12 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.250000e-01 : f32
    %13 = tensor.splat %12 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x64x2048xf32>
    %14 = tensor.empty() : tensor<1x64x2048xf32>
    %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %13 : tensor<1x64x2048xf32>, tensor<1x64x2048xf32>) outs(%14 : tensor<1x64x2048xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb1(%16: f32, %17: f32, %18: f32):
      %19 = arith.mulf %16, %17 : f32
      linalg.yield %19 : f32
    } -> tensor<1x64x2048xf32>
    %20 = tensor.empty() : tensor<1x64x2048xf32>
    %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %2 : tensor<1x64x2048xf32>, tensor<1x64x2048xf32>) outs(%20 : tensor<1x64x2048xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb2(%22: f32, %23: f32, %24: f32):
      %25 = arith.addf %22, %23 : f32
      linalg.yield %25 : f32
    } -> tensor<1x64x2048xf32>
    %26 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %27 = tensor.splat %26 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x64xf32>
    %28 = linalg.reduce ins(%21:tensor<1x64x2048xf32>) outs(%27:tensor<1x64xf32>) dimensions = [2]
    (%29: f32, %30: f32) {
      %31 = arith.maximumf %29, %30 : f32
      linalg.yield %31 : f32
    }
    %32 = tensor.collapse_shape %28 [[0 : i64, 1 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x64xf32> into tensor<64xf32>
    %33 = tensor.expand_shape %32 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<64xf32> into tensor<1x64x1xf32>
    %34 = tensor.empty() : tensor<1x64x2048xf32>
    %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%21, %33 : tensor<1x64x2048xf32>, tensor<1x64x1xf32>) outs(%34 : tensor<1x64x2048xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb3(%36: f32, %37: f32, %38: f32):
      %39 = arith.subf %36, %37 : f32
      linalg.yield %39 : f32
    } -> tensor<1x64x2048xf32>
    %40 = tensor.empty() : tensor<1x64x2048xf32>
    %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35 : tensor<1x64x2048xf32>) outs(%40 : tensor<1x64x2048xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb4(%42: f32, %43: f32):
      %44 = math.exp %42 : f32
      linalg.yield %44 : f32
    } -> tensor<1x64x2048xf32>
    %45 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %46 = tensor.splat %45 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x64xf32>
    %47 = linalg.reduce ins(%41:tensor<1x64x2048xf32>) outs(%46:tensor<1x64xf32>) dimensions = [2]
    (%48: f32, %49: f32) {
      %50 = arith.addf %48, %49 : f32
      linalg.yield %50 : f32
    }
    %51 = tensor.collapse_shape %47 [[0 : i64, 1 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x64xf32> into tensor<64xf32>
    %52 = tensor.expand_shape %51 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<64xf32> into tensor<1x64x1xf32>
    %53 = tensor.empty() : tensor<1x64x2048xf32>
    %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %52 : tensor<1x64x2048xf32>, tensor<1x64x1xf32>) outs(%53 : tensor<1x64x2048xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb5(%55: f32, %56: f32, %57: f32):
      %58 = arith.divf %55, %56 : f32
      linalg.yield %58 : f32
    } -> tensor<1x64x2048xf32>
    %59 = arith.constant {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %60 = tensor.splat %59 {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<1x64x64xf32>
    %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%54, %3 : tensor<1x64x2048xf32>, tensor<1x2048x64xf32>) outs(%60 : tensor<1x64x64xf32>) attrs =  {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
    ^bb6(%62: f32, %63: f32, %64: f32):
      %65 = arith.mulf %62, %63 : f32
      %66 = arith.addf %64, %65 : f32
      linalg.yield %66 : f32
    } -> tensor<1x64x64xf32>
    func.return %61 : tensor<1x64x64xf32>
  }
}
