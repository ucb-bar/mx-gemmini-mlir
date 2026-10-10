builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<64x8xbf16>, %1: tensor<64x8xbf16>, %2: tensor<64x8xbf16>) -> tensor<64x8xbf16> {
    %3 = tensor.empty() : tensor<64x8xbf16>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %2 : tensor<64x8xbf16>, tensor<64x8xbf16>) outs(%3 : tensor<64x8xbf16>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb0(%5: bf16, %6: bf16, %7: bf16):
      %8 = arith.addf %5, %6 : bf16
      linalg.yield %8 : bf16
    } -> tensor<64x8xbf16>
    func.return %4 : tensor<64x8xbf16>
  }
}
