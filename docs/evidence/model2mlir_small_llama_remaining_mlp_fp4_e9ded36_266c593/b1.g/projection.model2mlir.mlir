builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<32x128xf32>, %1: tensor<128x352xf32>) -> tensor<32x352xf32> {
    %2 = arith.constant {prov.region_id = "matmul_0", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %3 = tensor.splat %2 {prov.region_id = "matmul_0", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} : tensor<32x352xf32>
    %4 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%0, %1 : tensor<32x128xf32>, tensor<128x352xf32>) outs(%3 : tensor<32x352xf32>) -> tensor<32x352xf32>
    func.return %4 : tensor<32x352xf32>
  }
}
