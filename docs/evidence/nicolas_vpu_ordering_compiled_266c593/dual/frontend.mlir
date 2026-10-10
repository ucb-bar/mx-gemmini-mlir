builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<64x8xbf16>, %1: tensor<64x8xbf16>, %2: tensor<64x8xbf16>) -> (tensor<64x8xbf16>, tensor<64x8xbf16>, tensor<64x8xbf16>) {
    %3 = arith.constant {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} 3.000000e+00 : bf16
    %4 = tensor.splat %3 {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} : tensor<64x8xbf16>
    %5 = tensor.empty() : tensor<64x8xbf16>
    %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %4 : tensor<64x8xbf16>, tensor<64x8xbf16>) outs(%5 : tensor<64x8xbf16>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb0(%7: bf16, %8: bf16, %9: bf16):
      %10 = arith.addf %7, %8 : bf16
      linalg.yield %10 : bf16
    } -> tensor<64x8xbf16>
    %11 = tensor.empty() : tensor<64x8xbf16>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<64x8xbf16>) outs(%11 : tensor<64x8xbf16>) attrs =  {prov.region_id = "exp_0", prov._pattern_hint = "exp", prov.op = "exp", prov.family = "elementwise", prov.aten = "aten.exp.default", prov.orig_dtype = "bfloat16"} {
    ^bb1(%13: bf16, %14: bf16):
      %15 = math.exp %13 : bf16
      linalg.yield %15 : bf16
    } -> tensor<64x8xbf16>
    %16 = tensor.empty() : tensor<64x8xbf16>
    %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%6, %2 : tensor<64x8xbf16>, tensor<64x8xbf16>) outs(%16 : tensor<64x8xbf16>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb2(%18: bf16, %19: bf16, %20: bf16):
      %21 = arith.mulf %18, %19 : bf16
      linalg.yield %21 : bf16
    } -> tensor<64x8xbf16>
    func.return %6, %12, %17 : tensor<64x8xbf16>, tensor<64x8xbf16>, tensor<64x8xbf16>
  }
}
