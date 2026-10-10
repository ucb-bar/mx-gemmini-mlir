builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<64x8xbf16>, %1: tensor<64x8xbf16>) -> tensor<64x8xbf16> {
    %2 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} 5.000000e-01 : f32
    %3 = tensor.splat %2 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} : tensor<64x8xf32>
    %4 = tensor.empty() : tensor<64x8xf32>
    %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1, %3 : tensor<64x8xbf16>, tensor<64x8xf32>) outs(%4 : tensor<64x8xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb0(%6: bf16, %7: f32, %8: f32):
      %9 = arith.extf %6 : bf16 to f32
      %10 = arith.mulf %9, %7 : f32
      linalg.yield %10 : f32
    } -> tensor<64x8xf32>
    %11 = tensor.empty() : tensor<64x8xbf16>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<64x8xf32>) outs(%11 : tensor<64x8xbf16>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb1(%13: f32, %14: bf16):
      %15 = arith.truncf %13 : f32 to bf16
      linalg.yield %15 : bf16
    } -> tensor<64x8xbf16>
    func.return %12 : tensor<64x8xbf16>
  }
}
