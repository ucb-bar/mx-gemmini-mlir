builtin.module {
  func.func @forward(%0: tensor<17x8xbf16>) -> tensor<17x8xbf16> {
    %1 = func.call @forward$kernel_0__rmul_0(%0) : (tensor<17x8xbf16>) -> tensor<17x8xbf16>
    %2 = func.call @forward$kernel_1__radd_0(%1) : (tensor<17x8xbf16>) -> tensor<17x8xbf16>
    func.return %2 : tensor<17x8xbf16>
  }
  func.func private @forward$kernel_0__rmul_0(%0: tensor<17x8xbf16>) -> tensor<17x8xbf16> attributes {merlin.group = 0 : i64, merlin.placement = "vpu"} {
    %1 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.source_node_ids = ["g:prepared:root:n1"], prov.origin_node_ids = ["g:original:root:n1", "g:quantized:root:n1"], prov.trace_role = "lowering", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} 1.1015625 : f32
    %2 = tensor.splat %1 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.source_node_ids = ["g:prepared:root:n1"], prov.origin_node_ids = ["g:original:root:n1", "g:quantized:root:n1"], prov.trace_role = "lowering", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} : tensor<17x8xf32>
    %3 = tensor.empty() : tensor<17x8xf32>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %2 : tensor<17x8xbf16>, tensor<17x8xf32>) outs(%3 : tensor<17x8xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.source_node_ids = ["g:prepared:root:n1"], prov.origin_node_ids = ["g:original:root:n1", "g:quantized:root:n1"], prov.trace_role = "lowering", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb0(%5: bf16, %6: f32, %7: f32):
      %8 = arith.extf %5 {prov.source_node_ids = ["g:prepared:root:n1"], prov.origin_node_ids = ["g:original:root:n1", "g:quantized:root:n1"], prov.trace_role = "lowering"} : bf16 to f32
      %9 = arith.mulf %8, %6 {prov.source_node_ids = ["g:prepared:root:n1"], prov.origin_node_ids = ["g:original:root:n1", "g:quantized:root:n1"], prov.trace_role = "lowering"} : f32
      linalg.yield {prov.source_node_ids = ["g:prepared:root:n1"], prov.origin_node_ids = ["g:original:root:n1", "g:quantized:root:n1"], prov.trace_role = "lowering"} %9 : f32
    } -> tensor<17x8xf32>
    %10 = tensor.empty() : tensor<17x8xbf16>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<17x8xf32>) outs(%10 : tensor<17x8xbf16>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.source_node_ids = ["g:prepared:root:n1"], prov.origin_node_ids = ["g:original:root:n1", "g:quantized:root:n1"], prov.trace_role = "lowering", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb1(%12: f32, %13: bf16):
      %14 = arith.truncf %12 {prov.source_node_ids = ["g:prepared:root:n1"], prov.origin_node_ids = ["g:original:root:n1", "g:quantized:root:n1"], prov.trace_role = "lowering"} : f32 to bf16
      linalg.yield {prov.source_node_ids = ["g:prepared:root:n1"], prov.origin_node_ids = ["g:original:root:n1", "g:quantized:root:n1"], prov.trace_role = "lowering"} %14 : bf16
    } -> tensor<17x8xbf16>
    func.return %11 : tensor<17x8xbf16>
  }
  func.func private @forward$kernel_1__radd_0(%0: tensor<17x8xbf16>) -> tensor<17x8xbf16> attributes {merlin.group = 1 : i64, merlin.placement = "vpu"} {
    %1 = arith.constant {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} 3.339840e-01 : bf16
    %2 = tensor.splat %1 {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} : tensor<17x8xbf16>
    %3 = tensor.empty() : tensor<17x8xbf16>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %2 : tensor<17x8xbf16>, tensor<17x8xbf16>) outs(%3 : tensor<17x8xbf16>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb0(%5: bf16, %6: bf16, %7: bf16):
      %8 = arith.addf %5, %6 {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : bf16
      linalg.yield {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} %8 : bf16
    } -> tensor<17x8xbf16>
    func.return %4 : tensor<17x8xbf16>
  }
}