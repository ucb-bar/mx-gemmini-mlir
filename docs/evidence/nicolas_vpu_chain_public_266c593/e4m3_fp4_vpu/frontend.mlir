builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<64x8xbf16>, %1: tensor<64x8xbf16>) -> tensor<16xbf16> {
    %2 = tensor.empty() : tensor<64x8xbf16>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1, %1 : tensor<64x8xbf16>, tensor<64x8xbf16>) outs(%2 : tensor<64x8xbf16>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb0(%4: bf16, %5: bf16, %6: bf16):
      %7 = arith.addf %4, %5 : bf16
      linalg.yield %7 : bf16
    } -> tensor<64x8xbf16>
    %8 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} 5.000000e-01 : f32
    %9 = tensor.splat %8 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} : tensor<64x8xf32>
    %10 = tensor.empty() : tensor<64x8xf32>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3, %9 : tensor<64x8xbf16>, tensor<64x8xf32>) outs(%10 : tensor<64x8xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb1(%12: bf16, %13: f32, %14: f32):
      %15 = arith.extf %12 : bf16 to f32
      %16 = arith.mulf %15, %13 : f32
      linalg.yield %16 : f32
    } -> tensor<64x8xf32>
    %17 = tensor.empty() : tensor<64x8xbf16>
    %18 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%11 : tensor<64x8xf32>) outs(%17 : tensor<64x8xbf16>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb2(%19: f32, %20: bf16):
      %21 = arith.truncf %19 : f32 to bf16
      linalg.yield %21 : bf16
    } -> tensor<64x8xbf16>
    %22 = tensor.collapse_shape %18 [[0 : i64, 1 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<64x8xbf16> into tensor<512xbf16>
    %23 = tensor.expand_shape %22 [[0 : i64, 1 : i64, 2 : i64]] output_shape [16, 4, 8] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<512xbf16> into tensor<16x4x8xbf16>
    %24 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "bfloat16"} 0xff80 : bf16
    %25 = tensor.splat %24 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "bfloat16"} : tensor<16xbf16>
    %26 = linalg.reduce ins(%23:tensor<16x4x8xbf16>) outs(%25:tensor<16xbf16>) dimensions = [1, 2]
    (%27: bf16, %28: bf16) {
      %29 = arith.maximumf %27, %28 : bf16
      linalg.yield %29 : bf16
    }
    func.return %26 : tensor<16xbf16>
  }
}
