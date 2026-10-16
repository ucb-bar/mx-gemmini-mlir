builtin.module {
  func.func @forward(%0: tensor<137xf32>, %1: tensor<137xf32>, %2: tensor<137xf32>) -> tensor<137xf32> {
    %3 = tensor.empty() : tensor<137xf32>
    %4 = func.call @forward$kernel_0__rmul_0(%0, %1) : (tensor<137xf32>, tensor<137xf32>) -> tensor<137xf32>
    %5 = tensor.empty() : tensor<137xf32>
    %6 = func.call @forward$kernel_1__rmul_1(%4, %2) : (tensor<137xf32>, tensor<137xf32>) -> tensor<137xf32>
    func.return %6 : tensor<137xf32>
  }
  func.func private @forward$kernel_0__rmul_0(%0: tensor<137xf32>, %1: tensor<137xf32>) -> tensor<137xf32> {
    %2 = tensor.empty() : tensor<137xf32>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : tensor<137xf32>, tensor<137xf32>) outs(%2 : tensor<137xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.source_node_ids = ["g:prepared:root:n3"], prov.origin_node_ids = ["g:original:root:n3", "g:quantized:root:n3"], prov.trace_role = "lowering", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb0(%4: f32, %5: f32, %6: f32):
      %7 = arith.mulf %4, %5 {prov.source_node_ids = ["g:prepared:root:n3"], prov.origin_node_ids = ["g:original:root:n3", "g:quantized:root:n3"], prov.trace_role = "lowering"} : f32
      linalg.yield {prov.source_node_ids = ["g:prepared:root:n3"], prov.origin_node_ids = ["g:original:root:n3", "g:quantized:root:n3"], prov.trace_role = "lowering"} %7 : f32
    } -> tensor<137xf32>
    func.return %3 : tensor<137xf32>
  }
  func.func private @forward$kernel_1__rmul_1(%0: tensor<137xf32>, %1: tensor<137xf32>) -> tensor<137xf32> {
    %2 = tensor.empty() : tensor<137xf32>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : tensor<137xf32>, tensor<137xf32>) outs(%2 : tensor<137xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.source_node_ids = ["g:prepared:root:n4"], prov.origin_node_ids = ["g:original:root:n4", "g:quantized:root:n4"], prov.trace_role = "lowering", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb0(%4: f32, %5: f32, %6: f32):
      %7 = arith.mulf %4, %5 {prov.source_node_ids = ["g:prepared:root:n4"], prov.origin_node_ids = ["g:original:root:n4", "g:quantized:root:n4"], prov.trace_role = "lowering"} : f32
      linalg.yield {prov.source_node_ids = ["g:prepared:root:n4"], prov.origin_node_ids = ["g:original:root:n4", "g:quantized:root:n4"], prov.trace_role = "lowering"} %7 : f32
    } -> tensor<137xf32>
    func.return %3 : tensor<137xf32>
  }
}