builtin.module attributes {prov.weights_file = "../.tmp/mx_model_goal_20261018/rotary_probe/capture/weights.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<1x32x1xf32>, %1: tensor<1x1x160xf32>) -> tensor<1x32x160xf32> {
    %2 = tensor.empty() : tensor<1x32x1xf32>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<1x32x1xf32>) outs(%2 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb0(%4: f32, %5: f32):
      linalg.yield %4 : f32
    } -> tensor<1x32x1xf32>
    %6 = tensor.empty() : tensor<1x1x160xf32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x1x160xf32>) outs(%6 : tensor<1x1x160xf32>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb1(%8: f32, %9: f32):
      linalg.yield %8 : f32
    } -> tensor<1x1x160xf32>
    %10 = arith.constant {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %11 = tensor.splat %10 {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<1x32x160xf32>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3, %7 : tensor<1x32x1xf32>, tensor<1x1x160xf32>) outs(%11 : tensor<1x32x160xf32>) attrs =  {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
    ^bb2(%13: f32, %14: f32, %15: f32):
      %16 = arith.mulf %13, %14 : f32
      %17 = arith.addf %15, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<1x32x160xf32>
    func.return %12 : tensor<1x32x160xf32>
  }
}