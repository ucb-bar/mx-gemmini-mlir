builtin.module attributes {prov.weights_file = "/scratch/agustin/merlin/tasks/mx-vpu-four-model-e2e/.tmp/mx_model_goal_20261018/projected_session_varied/capture/stages/prefill/weights.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @merlin_stage_0_prefill(%0: tensor<32x32xf32>, %1: tensor<32xf32>, %2: tensor<16x32xf32>, %3: tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xf32>) {
    %4 = tensor.empty() : tensor<32x32xf32>
    %5 = linalg.transpose ins(%0:tensor<32x32xf32>) outs(%4:tensor<32x32xf32>) permutation = [1, 0]
    %6 = tensor.empty() : tensor<16x32xf32>
    %7 = arith.constant {prov.module = "linear"} 0.000000e+00 : f32
    %8 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "linear"} ins(%7 : f32) outs(%6 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %9 = linalg.matmul {prov.region_id = "matmul_0", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "linear", prov.fqn = "linear", prov.transposed_b = "true"} ins(%2, %5 : tensor<16x32xf32>, tensor<32x32xf32>) outs(%8 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %10 = tensor.empty() : tensor<16x32xf32>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9, %1 : tensor<16x32xf32>, tensor<32xf32>) outs(%10 : tensor<16x32xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "linear", prov.fqn = "linear"} {
    ^bb0(%12: f32, %13: f32, %14: f32):
      %15 = arith.addf %12, %13 : f32
      linalg.yield %15 : f32
    } -> tensor<16x32xf32>
    %16 = tensor.empty() : tensor<16x32xf32>
    %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%11, %3 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%16 : tensor<16x32xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb1(%18: f32, %19: f32, %20: f32):
      %21 = arith.addf %18, %19 : f32
      linalg.yield %21 : f32
    } -> tensor<16x32xf32>
    %22 = arith.constant {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 5.000000e-01 : f32
    %23 = tensor.splat %22 {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<16x32xf32>
    %24 = tensor.empty() : tensor<16x32xf32>
    %25 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %23 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%24 : tensor<16x32xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb2(%26: f32, %27: f32, %28: f32):
      %29 = arith.addf %26, %27 : f32
      linalg.yield %29 : f32
    } -> tensor<16x32xf32>
    func.return %17, %25 : tensor<16x32xf32>, tensor<16x32xf32>
  }
}