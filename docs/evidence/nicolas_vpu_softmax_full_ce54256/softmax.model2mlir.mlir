builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x32xbf16>) -> tensor<16x32xbf16> {
    %1 = tensor.empty() : tensor<16x32xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<16x32xbf16>) outs(%1 : tensor<16x32xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} {
    ^bb0(%3: bf16, %4: f32):
      %5 = arith.extf %3 : bf16 to f32
      linalg.yield %5 : f32
    } -> tensor<16x32xf32>
    %6 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} 0xff800000 : f32
    %7 = tensor.splat %6 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} : tensor<16xf32>
    %8 = linalg.reduce ins(%2:tensor<16x32xf32>) outs(%7:tensor<16xf32>) dimensions = [1]
    (%9: f32, %10: f32) {
      %11 = arith.maximumf %9, %10 : f32
      linalg.yield %11 : f32
    }
    %12 = tensor.expand_shape %8 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} : tensor<16xf32> into tensor<16x1xf32>
    %13 = tensor.empty() : tensor<16x32xf32>
    %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %12 : tensor<16x32xf32>, tensor<16x1xf32>) outs(%13 : tensor<16x32xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} {
    ^bb1(%15: f32, %16: f32, %17: f32):
      %18 = arith.subf %15, %16 : f32
      linalg.yield %18 : f32
    } -> tensor<16x32xf32>
    %19 = tensor.empty() : tensor<16x32xf32>
    %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<16x32xf32>) outs(%19 : tensor<16x32xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} {
    ^bb2(%21: f32, %22: f32):
      %23 = math.exp %21 : f32
      linalg.yield %23 : f32
    } -> tensor<16x32xf32>
    %24 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} 0.000000e+00 : f32
    %25 = tensor.splat %24 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} : tensor<16xf32>
    %26 = linalg.reduce ins(%20:tensor<16x32xf32>) outs(%25:tensor<16xf32>) dimensions = [1]
    (%27: f32, %28: f32) {
      %29 = arith.addf %27, %28 : f32
      linalg.yield %29 : f32
    }
    %30 = tensor.expand_shape %26 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} : tensor<16xf32> into tensor<16x1xf32>
    %31 = tensor.empty() : tensor<16x32xf32>
    %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %30 : tensor<16x32xf32>, tensor<16x1xf32>) outs(%31 : tensor<16x32xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} {
    ^bb3(%33: f32, %34: f32, %35: f32):
      %36 = arith.divf %33, %34 : f32
      linalg.yield %36 : f32
    } -> tensor<16x32xf32>
    %37 = tensor.empty() : tensor<16x32xbf16>
    %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%32 : tensor<16x32xf32>) outs(%37 : tensor<16x32xbf16>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} {
    ^bb4(%39: f32, %40: bf16):
      %41 = arith.truncf %39 : f32 to bf16
      linalg.yield %41 : bf16
    } -> tensor<16x32xbf16>
    func.return %38 : tensor<16x32xbf16>
  }
}
