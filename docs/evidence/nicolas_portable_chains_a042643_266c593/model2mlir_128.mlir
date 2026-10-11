builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<128x128xf32>, %1: tensor<128x128xf32>, %2: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %3 = tensor.empty() : tensor<128x128xf32>
    %4 = arith.constant 0.000000e+00 : f32
    %5 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%4 : f32) outs(%3 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %6 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%0, %1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%5 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %7 = tensor.empty() : tensor<128x128xf32>
    %8 = arith.constant 0.000000e+00 : f32
    %9 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%8 : f32) outs(%7 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %10 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%6, %2 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%9 : tensor<128x128xf32>) -> tensor<128x128xf32>
    func.return %10 : tensor<128x128xf32>
  }
}
