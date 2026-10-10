builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<64x8xbf16>, %1: tensor<64x8xbf16>) -> tensor<16xbf16> {
    %2 = tensor.empty() : tensor<64x8xbf16>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<64x8xbf16>) outs(%2 : tensor<64x8xbf16>) attrs =  {prov.region_id = "abs_0", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "bfloat16"} {
    ^bb0(%4: bf16, %5: bf16):
      %6 = math.absf %4 : bf16
      linalg.yield %6 : bf16
    } -> tensor<64x8xbf16>
    %7 = tensor.collapse_shape %3 [[0 : i64, 1 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<64x8xbf16> into tensor<512xbf16>
    %8 = tensor.expand_shape %7 [[0 : i64, 1 : i64, 2 : i64]] output_shape [16, 4, 8] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<512xbf16> into tensor<16x4x8xbf16>
    %9 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "bfloat16"} 0xff80 : bf16
    %10 = tensor.splat %9 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "bfloat16"} : tensor<16xbf16>
    %11 = linalg.reduce ins(%8:tensor<16x4x8xbf16>) outs(%10:tensor<16xbf16>) dimensions = [1, 2]
    (%12: bf16, %13: bf16) {
      %14 = arith.maximumf %12, %13 : bf16
      linalg.yield %14 : bf16
    }
    func.return %11 : tensor<16xbf16>
  }
}
