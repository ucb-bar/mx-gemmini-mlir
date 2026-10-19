builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<1x64x5632xf32>, %1: tensor<1x64x5632xf32>, %2: tensor<1x64x5632xf32>) -> tensor<1x64x5632xf32> {
    %3 = tensor.empty() : tensor<1x64x5632xf32>
    %4 = tensor.empty() : tensor<1x64x5632xf32>
    %5 = tensor.empty() : tensor<1x64x5632xf32>
    %6 = func.call @mx_vpu_mul_chain_f32_1_64_5632(%0, %1, %2, %5) : (tensor<1x64x5632xf32>, tensor<1x64x5632xf32>, tensor<1x64x5632xf32>, tensor<1x64x5632xf32>) -> tensor<1x64x5632xf32>
    func.return %6 : tensor<1x64x5632xf32>
  }
  "func.func"() <{sym_name = "mx_vpu_mul_chain_f32_1_64_5632", function_type = (tensor<1x64x5632xf32>, tensor<1x64x5632xf32>, tensor<1x64x5632xf32>, tensor<1x64x5632xf32>) -> tensor<1x64x5632xf32>, sym_visibility = "private", arg_attrs = [{bufferization.access = "read"}, {bufferization.access = "read"}, {bufferization.access = "read"}, {bufferization.access = "write"}]}> ({
  }) : () -> ()
}