builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<137xf32>, %1: tensor<137xf32>, %2: tensor<137xf32>) -> tensor<137xf32> {
    %3 = tensor.empty() : tensor<137xf32>
    %4 = tensor.empty() : tensor<137xf32>
    %5 = tensor.empty() : tensor<137xf32>
    %6 = func.call @mx_vpu_mul_chain_f32_137(%0, %1, %2, %5) : (tensor<137xf32>, tensor<137xf32>, tensor<137xf32>, tensor<137xf32>) -> tensor<137xf32>
    func.return %6 : tensor<137xf32>
  }
  "func.func"() <{sym_name = "mx_vpu_mul_chain_f32_137", function_type = (tensor<137xf32>, tensor<137xf32>, tensor<137xf32>, tensor<137xf32>) -> tensor<137xf32>, sym_visibility = "private", arg_attrs = [{bufferization.access = "read"}, {bufferization.access = "read"}, {bufferization.access = "read"}, {bufferization.access = "write"}]}> ({
  }) : () -> ()
}