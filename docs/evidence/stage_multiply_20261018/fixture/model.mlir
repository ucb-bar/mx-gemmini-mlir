builtin.module attributes {prov.weights_file = "/scratch/agustin/merlin/tasks/mx-vpu-four-model-e2e/.tmp/tiny_vpu_opt_20261018/capture_linear/weights.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<32x32xf32>, %1: tensor<32x32xf32>, %2: tensor<32x32xf32>, %3: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %4 = tensor.empty() : tensor<32x32xf32>
    %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1, %2 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%4 : tensor<32x32xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb0(%6: f32, %7: f32, %8: f32):
      %9 = arith.mulf %6, %7 : f32
      linalg.yield %9 : f32
    } -> tensor<32x32xf32>
    %10 = tensor.empty() : tensor<32x32xf32>
    %11 = linalg.transpose ins(%0:tensor<32x32xf32>) outs(%10:tensor<32x32xf32>) permutation = [1, 0]
    %12 = arith.constant {prov.region_id = "matmul_0", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "up", prov.fqn = "up"} 0.000000e+00 : f32
    %13 = tensor.splat %12 {prov.region_id = "matmul_0", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "up", prov.fqn = "up"} : tensor<32x32xf32>
    %14 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "up", prov.fqn = "up", prov.transposed_b = "true"} ins(%3, %11 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%13 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %15 = tensor.empty() : tensor<32x32xf32>
    %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5, %14 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb1(%17: f32, %18: f32, %19: f32):
      %20 = arith.mulf %17, %18 : f32
      linalg.yield %20 : f32
    } -> tensor<32x32xf32>
    func.return %16 : tensor<32x32xf32>
  }
}