builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<64x8xbf16>, %1: tensor<16x8xbf16>) -> tensor<64x8xbf16> {
    %2 = tensor.collapse_shape %0 [[0 : i64, 1 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<64x8xbf16> into tensor<512xbf16>
    %3 = tensor.expand_shape %2 [[0 : i64, 1 : i64, 2 : i64]] output_shape [16, 4, 8] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<512xbf16> into tensor<16x4x8xbf16>
    %4 = tensor.collapse_shape %1 [[0 : i64, 1 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<16x8xbf16> into tensor<128xbf16>
    %5 = tensor.expand_shape %4 [[0 : i64, 1 : i64, 2 : i64]] output_shape [16, 1, 8] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<128xbf16> into tensor<16x1x8xbf16>
    %6 = tensor.empty() : tensor<16x4x8xbf16>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %5 : tensor<16x4x8xbf16>, tensor<16x1x8xbf16>) outs(%6 : tensor<16x4x8xbf16>) attrs =  {prov.region_id = "minmax_0", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.maximum.default", prov.orig_dtype = "bfloat16"} {
    ^bb0(%8: bf16, %9: bf16, %10: bf16):
      %11 = arith.maximumf %8, %9 : bf16
      linalg.yield %11 : bf16
    } -> tensor<16x4x8xbf16>
    %12 = tensor.collapse_shape %7 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<16x4x8xbf16> into tensor<512xbf16>
    %13 = tensor.expand_shape %12 [[0 : i64, 1 : i64]] output_shape [64, 8] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<512xbf16> into tensor<64x8xbf16>
    func.return %13 : tensor<64x8xbf16>
  }
}
