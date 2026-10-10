builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<64x8xbf16>, %1: tensor<64x8xbf16>) -> tensor<64x8xbf16> {
    %2 = tensor.empty() : tensor<64x8xbf16>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<64x8xbf16>) outs(%2 : tensor<64x8xbf16>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "bfloat16"} {
    ^bb0(%4: bf16, %5: bf16):
      %6 = math.rsqrt %4 : bf16
      linalg.yield %6 : bf16
    } -> tensor<64x8xbf16>
    func.return %3 : tensor<64x8xbf16>
  }
}
