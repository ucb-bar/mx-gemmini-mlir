builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<64x8xbf16>, %1: tensor<64x8xbf16>, %2: tensor<64x8xbf16>) -> tensor<16xbf16> {
    %3 = tensor.empty() : tensor<64x8xbf16>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %2 : tensor<64x8xbf16>, tensor<64x8xbf16>) outs(%3 : tensor<64x8xbf16>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb0(%5: bf16, %6: bf16, %7: bf16):
      %8 = arith.addf %5, %6 : bf16
      linalg.yield %8 : bf16
    } -> tensor<64x8xbf16>
    %9 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} 5.000000e-01 : f32
    %10 = tensor.splat %9 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} : tensor<64x8xf32>
    %11 = tensor.empty() : tensor<64x8xf32>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4, %10 : tensor<64x8xbf16>, tensor<64x8xf32>) outs(%11 : tensor<64x8xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb1(%13: bf16, %14: f32, %15: f32):
      %16 = arith.extf %13 : bf16 to f32
      %17 = arith.mulf %16, %14 : f32
      linalg.yield %17 : f32
    } -> tensor<64x8xf32>
    %18 = tensor.empty() : tensor<64x8xbf16>
    %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<64x8xf32>) outs(%18 : tensor<64x8xbf16>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb2(%20: f32, %21: bf16):
      %22 = arith.truncf %20 : f32 to bf16
      linalg.yield %22 : bf16
    } -> tensor<64x8xbf16>
    %23 = tensor.collapse_shape %19 [[0 : i64, 1 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<64x8xbf16> into tensor<512xbf16>
    %24 = tensor.expand_shape %23 [[0 : i64, 1 : i64, 2 : i64]] output_shape [16, 4, 8] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<512xbf16> into tensor<16x4x8xbf16>
    %25 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "bfloat16"} 0xff80 : bf16
    %26 = tensor.splat %25 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "bfloat16"} : tensor<16xbf16>
    %27 = linalg.reduce ins(%24:tensor<16x4x8xbf16>) outs(%26:tensor<16xbf16>) dimensions = [1, 2]
    (%28: bf16, %29: bf16) {
      %30 = arith.maximumf %28, %29 : bf16
      linalg.yield %30 : bf16
    }
    func.return %27 : tensor<16xbf16>
  }
}
