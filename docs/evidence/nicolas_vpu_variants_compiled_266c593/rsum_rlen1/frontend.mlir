builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<64x8xbf16>, %1: tensor<64x8xbf16>) -> tensor<64xbf16> {
    %2 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "bfloat16"} 0.000000e+00 : bf16
    %3 = tensor.splat %2 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "bfloat16"} : tensor<64xbf16>
    %4 = linalg.reduce ins(%0:tensor<64x8xbf16>) outs(%3:tensor<64xbf16>) dimensions = [1]
    (%5: bf16, %6: bf16) {
      %7 = arith.addf %5, %6 : bf16
      linalg.yield %7 : bf16
    }
    func.return %4 : tensor<64xbf16>
  }
}
