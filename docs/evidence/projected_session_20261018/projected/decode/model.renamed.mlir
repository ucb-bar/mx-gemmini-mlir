"builtin.module"() ({
  "func.func"() <{sym_name = "merlin_stage_1_decode", function_type = (tensor<32xf32>, tensor<16x32xf32>, tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xf32>)}> ({
  ^bb0(%0: tensor<32xf32>, %1: tensor<16x32xf32>, %2: tensor<16x32xf32>):
    %3 = "tensor.empty"() : () -> tensor<16x32xf32>
    %4 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.module = "linear"} : () -> f32
    %5 = "linalg.fill"(%4, %3) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb1(%6: f32, %7: f32):
      "linalg.yield"(%6) : (f32) -> ()
    }) {prov.op = "fill", prov.family = "fill", prov.module = "linear"} : (f32, tensor<16x32xf32>) -> tensor<16x32xf32>
    %8 = "tensor.empty"() : () -> tensor<32x32xf32>
    %9 = "func.call"(%1, %8, %5) <{callee = @merlin_dev_mx_gemmini_0}> : (tensor<16x32xf32>, tensor<32x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
    %10 = "tensor.empty"() : () -> tensor<16x32xf32>
    %11 = "func.call"(%9, %0, %10) <{callee = @mx_vpu_add_f32_bias_16_32}> : (tensor<16x32xf32>, tensor<32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
    %12 = "tensor.empty"() : () -> tensor<16x32xf32>
    %13 = "func.call"(%11, %2, %12) <{callee = @mx_vpu_add_f32_same_16_32}> : (tensor<16x32xf32>, tensor<16x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
    %14 = "arith.constant"() <{value = 5.000000e-01 : f32}> {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : () -> f32
    %15 = "tensor.splat"(%14) : (f32) -> tensor<f32>
    %16 = "tensor.empty"() : () -> tensor<16x32xf32>
    %17 = "func.call"(%13, %15, %16) <{callee = @mx_vpu_add_f32_lf32_rf32_16_32_lhs_16_32_laxes_0_1_rhs_scalar_axes_}> : (tensor<16x32xf32>, tensor<f32>, tensor<16x32xf32>) -> tensor<16x32xf32>
    "func.return"(%13, %17) : (tensor<16x32xf32>, tensor<16x32xf32>) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "merlin_dev_mx_gemmini_0", function_type = (tensor<16x32xf32>, tensor<32x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>, sym_visibility = "private", arg_attrs = [{bufferization.access = "read"}, {bufferization.access = "read"}, {bufferization.access = "write"}]}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "mx_vpu_add_f32_bias_16_32", function_type = (tensor<16x32xf32>, tensor<32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>, sym_visibility = "private", arg_attrs = [{bufferization.access = "read"}, {bufferization.access = "read"}, {bufferization.access = "write"}]}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "mx_vpu_add_f32_same_16_32", function_type = (tensor<16x32xf32>, tensor<16x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>, sym_visibility = "private", arg_attrs = [{bufferization.access = "read"}, {bufferization.access = "read"}, {bufferization.access = "write"}]}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "mx_vpu_add_f32_lf32_rf32_16_32_lhs_16_32_laxes_0_1_rhs_scalar_axes_", function_type = (tensor<16x32xf32>, tensor<f32>, tensor<16x32xf32>) -> tensor<16x32xf32>, sym_visibility = "private", arg_attrs = [{bufferization.access = "read"}, {bufferization.access = "read"}, {bufferization.access = "write"}]}> ({
  }) : () -> ()
}) {prov.weights_file = "/scratch/agustin/merlin/tasks/mx-vpu-four-model-e2e/.tmp/mx_model_goal_20261018/projected_session_varied/capture/stages/decode/weights.safetensors", prov.level = "linalg-on-tensors"} : () -> ()