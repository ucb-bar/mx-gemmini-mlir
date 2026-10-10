builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<64x8xbf16>, %1: tensor<64x8xbf16>) -> tensor<64x8xbf16> {
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
    func.return %9 : tensor<64x8xbf16>
  }
}
