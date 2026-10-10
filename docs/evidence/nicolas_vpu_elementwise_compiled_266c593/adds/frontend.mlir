builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<64x8xbf16>, %1: tensor<64x8xbf16>) -> tensor<64x8xbf16> {
    %2 = arith.constant {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} 3.000000e+00 : bf16
    %3 = tensor.splat %2 {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} : tensor<64x8xbf16>
    %4 = tensor.empty() : tensor<64x8xbf16>
    %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1, %3 : tensor<64x8xbf16>, tensor<64x8xbf16>) outs(%4 : tensor<64x8xbf16>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb0(%6: bf16, %7: bf16, %8: bf16):
      %9 = arith.addf %6, %7 : bf16
      linalg.yield %9 : bf16
    } -> tensor<64x8xbf16>
    func.return %5 : tensor<64x8xbf16>
  }
}
