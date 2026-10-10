builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<64x8xbf16>, %1: tensor<16x8xbf16>) -> tensor<64x8xbf16> {
    %2 = tensor.collapse_shape %0 [[0 : i64, 1 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<64x8xbf16> into tensor<512xbf16>
    %3 = tensor.expand_shape %2 [[0 : i64, 1 : i64, 2 : i64]] output_shape [16, 4, 8] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<512xbf16> into tensor<16x4x8xbf16>
    %4 = tensor.collapse_shape %1 [[0 : i64, 1 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<16x8xbf16> into tensor<128xbf16>
    %5 = tensor.expand_shape %4 [[0 : i64, 1 : i64, 2 : i64]] output_shape [16, 1, 8] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<128xbf16> into tensor<16x1x8xbf16>
    %6 = tensor.empty() : tensor<16x4x8xbf16>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %5 : tensor<16x4x8xbf16>, tensor<16x1x8xbf16>) outs(%6 : tensor<16x4x8xbf16>) attrs =  {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb0(%8: bf16, %9: bf16, %10: bf16):
      %11 = arith.subf %8, %9 : bf16
      linalg.yield %11 : bf16
    } -> tensor<16x4x8xbf16>
    %12 = tensor.empty() : tensor<16x4x8xbf16>
    %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7 : tensor<16x4x8xbf16>) outs(%12 : tensor<16x4x8xbf16>) attrs =  {prov.region_id = "exp_0", prov._pattern_hint = "exp", prov.op = "exp", prov.family = "elementwise", prov.aten = "aten.exp.default", prov.orig_dtype = "bfloat16"} {
    ^bb1(%14: bf16, %15: bf16):
      %16 = math.exp %14 : bf16
      linalg.yield %16 : bf16
    } -> tensor<16x4x8xbf16>
    %17 = tensor.collapse_shape %13 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<16x4x8xbf16> into tensor<512xbf16>
    %18 = tensor.expand_shape %17 [[0 : i64, 1 : i64]] output_shape [64, 8] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<512xbf16> into tensor<64x8xbf16>
    func.return %18 : tensor<64x8xbf16>
  }
}
