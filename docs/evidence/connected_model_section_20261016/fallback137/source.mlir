"builtin.module"() ({
  "func.func"() <{sym_name = "forward", function_type = (tensor<137xf32>, tensor<137xf32>, tensor<137xf32>) -> tensor<137xf32>}> ({
  ^bb0(%0: tensor<137xf32>, %1: tensor<137xf32>, %2: tensor<137xf32>):
    %3 = "tensor.empty"() {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.source_node_ids = ["g:prepared:root:n3"], prov.origin_node_ids = ["g:original:root:n3", "g:quantized:root:n3"], prov.trace_role = "lowering", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : () -> tensor<137xf32>
    %4 = "linalg.generic"(%0, %1, %3) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb1(%5: f32, %6: f32, %7: f32):
      %8 = "arith.mulf"(%5, %6) <{fastmath = #arith.fastmath<none>}> {prov.source_node_ids = ["g:prepared:root:n3"], prov.origin_node_ids = ["g:original:root:n3", "g:quantized:root:n3"], prov.trace_role = "lowering"} : (f32, f32) -> f32
      "linalg.yield"(%8) {prov.source_node_ids = ["g:prepared:root:n3"], prov.origin_node_ids = ["g:original:root:n3", "g:quantized:root:n3"], prov.trace_role = "lowering"} : (f32) -> ()
    }) {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.source_node_ids = ["g:prepared:root:n3"], prov.origin_node_ids = ["g:original:root:n3", "g:quantized:root:n3"], prov.trace_role = "lowering", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : (tensor<137xf32>, tensor<137xf32>, tensor<137xf32>) -> tensor<137xf32>
    %9 = "tensor.empty"() {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.source_node_ids = ["g:prepared:root:n4"], prov.origin_node_ids = ["g:original:root:n4", "g:quantized:root:n4"], prov.trace_role = "lowering", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : () -> tensor<137xf32>
    %10 = "linalg.generic"(%4, %2, %9) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb2(%11: f32, %12: f32, %13: f32):
      %14 = "arith.mulf"(%11, %12) <{fastmath = #arith.fastmath<none>}> {prov.source_node_ids = ["g:prepared:root:n4"], prov.origin_node_ids = ["g:original:root:n4", "g:quantized:root:n4"], prov.trace_role = "lowering"} : (f32, f32) -> f32
      "linalg.yield"(%14) {prov.source_node_ids = ["g:prepared:root:n4"], prov.origin_node_ids = ["g:original:root:n4", "g:quantized:root:n4"], prov.trace_role = "lowering"} : (f32) -> ()
    }) {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.source_node_ids = ["g:prepared:root:n4"], prov.origin_node_ids = ["g:original:root:n4", "g:quantized:root:n4"], prov.trace_role = "lowering", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : (tensor<137xf32>, tensor<137xf32>, tensor<137xf32>) -> tensor<137xf32>
    "func.return"(%10) : (tensor<137xf32>) -> ()
  }) : () -> ()
}) {prov.level = "linalg-on-tensors"} : () -> ()