builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<64x8xbf16>, %1: tensor<64x8xbf16>) -> tensor<16xbf16> {
    %2 = tensor.collapse_shape %1 [[0 : i64, 1 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<64x8xbf16> into tensor<512xbf16>
    %3 = tensor.expand_shape %2 [[0 : i64, 1 : i64, 2 : i64]] output_shape [16, 4, 8] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<512xbf16> into tensor<16x4x8xbf16>
    %4 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "bfloat16"} 0.000000e+00 : bf16
    %5 = tensor.splat %4 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "bfloat16"} : tensor<16xbf16>
    %6 = linalg.reduce ins(%3:tensor<16x4x8xbf16>) outs(%5:tensor<16xbf16>) dimensions = [1, 2]
    (%7: bf16, %8: bf16) {
      %9 = arith.addf %7, %8 : bf16
      linalg.yield %9 : bf16
    }
    func.return %6 : tensor<16xbf16>
  }
}
