builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<64x8xbf16>, %1: tensor<64x8xbf16>) -> (tensor<64x8xbf16>, tensor<16x8xbf16>) {
    %2 = tensor.empty() : tensor<64x8xbf16>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1, %1 : tensor<64x8xbf16>, tensor<64x8xbf16>) outs(%2 : tensor<64x8xbf16>) attrs =  {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb0(%4: bf16, %5: bf16, %6: bf16):
      %7 = arith.subf %4, %5 : bf16
      linalg.yield %7 : bf16
    } -> tensor<64x8xbf16>
    %8 = tensor.empty() : tensor<64x8xbf16>
    %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<64x8xbf16>) outs(%8 : tensor<64x8xbf16>) attrs =  {prov.region_id = "exp_0", prov._pattern_hint = "exp", prov.op = "exp", prov.family = "elementwise", prov.aten = "aten.exp.default", prov.orig_dtype = "bfloat16"} {
    ^bb1(%10: bf16, %11: bf16):
      %12 = math.exp %10 : bf16
      linalg.yield %12 : bf16
    } -> tensor<64x8xbf16>
    %13 = tensor.collapse_shape %9 [[0 : i64, 1 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<64x8xbf16> into tensor<512xbf16>
    %14 = tensor.expand_shape %13 [[0 : i64, 1 : i64, 2 : i64]] output_shape [16, 4, 8] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<512xbf16> into tensor<16x4x8xbf16>
    %15 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "bfloat16"} 0.000000e+00 : bf16
    %16 = tensor.splat %15 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "bfloat16"} : tensor<16x8xbf16>
    %17 = linalg.reduce ins(%14:tensor<16x4x8xbf16>) outs(%16:tensor<16x8xbf16>) dimensions = [1]
    (%18: bf16, %19: bf16) {
      %20 = arith.addf %18, %19 : bf16
      linalg.yield %20 : bf16
    }
    func.return %9, %17 : tensor<64x8xbf16>, tensor<16x8xbf16>
  }
}
