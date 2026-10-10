module attributes {mx.profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487",
  mx.contract_sha256 = "641d60769c10d5ee6bcaa8e76b502266899be1b291bb2b42407a41712a5a8fbe", mx.policy_sha256 = "207a28d2083dce6f5ab7c59088f5e1d65cb0b482487e633064b868a155e765af",
  prov.quantization_manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351",
  mx.frontend_mlir_sha256 = "636980213f08b27b2737718e32cbab5e652427eb0428c070d7dc3f405f7198c9",
  mx.original_graph_sha256 = "4581e83379f6b3d345e303a47ca823143e355ed30a7542f1c1fe3921770df3aa",
  mx.first_input_pair_sha256 = "b34988847883e359b3d077ffa62cb43c0a76b57eb32c24b0cf590b9869b65c9c",
  mx.runtime_resources_sha256 = "30ce66dd215e851cbfff43f327657cb515e6dbda4a095158dd384545bce4f30f"} {
  func.func @nicolas_chain_pipelined(
      %a1: tensor<64x64xi8>, %a1s: tensor<2x64xi8>,
      %b1: tensor<64x64xi8>, %b1s: tensor<2x64xi8>,
      %b2: tensor<64x64xi8>, %b2s: tensor<2x64xi8>)
      -> (tensor<64x64xi8>, tensor<64x2xi8>,
          tensor<64x64xi8>, tensor<64x2xi8>) {
    %acc = "mx_gemmini.contract"(%a1, %a1s, %b1, %b1s) {
      site_id = "functional:matmul", activation_format = "fp8_e4m3",
      weight_format = "fp8_e4m3", activation_projection = "direct",
      weight_projection = "direct", pe_mode = 8 : i32, contract_sha256 = "641d60769c10d5ee6bcaa8e76b502266899be1b291bb2b42407a41712a5a8fbe", policy_sha256 = "207a28d2083dce6f5ab7c59088f5e1d65cb0b482487e633064b868a155e765af", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"}
      : (tensor<64x64xi8>, tensor<2x64xi8>, tensor<64x64xi8>, tensor<2x64xi8>)
      -> tensor<64x64xbf16>
    %bf16 = "mx_gemmini.readout_bf16"(%acc) {
      site_id = "functional:matmul", contract_sha256 = "641d60769c10d5ee6bcaa8e76b502266899be1b291bb2b42407a41712a5a8fbe", policy_sha256 = "207a28d2083dce6f5ab7c59088f5e1d65cb0b482487e633064b868a155e765af", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"}
      : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %v0 = "mx_gemmini.vpu_execute"(%bf16) {
      site_id = "functional:matmul_1", kind = "muls", src1_row = 4096 : i32,
      src2_row = 0 : i32, dst_row = 4096 : i32, rows = 512 : i32,
      reduction_length = 1 : i32, broadcast = false,
      immediate_bf16 = 16384 : i32, contract_sha256 = "641d60769c10d5ee6bcaa8e76b502266899be1b291bb2b42407a41712a5a8fbe", policy_sha256 = "207a28d2083dce6f5ab7c59088f5e1d65cb0b482487e633064b868a155e765af", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"}
      : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %c1_0, %c1s_0 = "mx_gemmini.spad_requant"(%v0) {
      site_id = "functional:matmul_1", source_row = 4096 : i32,
      destination_row = 128 : i32, m = 64 : i32, n = 64 : i32,
      output_format = "fp8_e4m3", tiled = true, resident = true,
      scale_dram_address = 0 : i64, scale_buffer = "c1_scales_0", contract_sha256 = "641d60769c10d5ee6bcaa8e76b502266899be1b291bb2b42407a41712a5a8fbe", policy_sha256 = "207a28d2083dce6f5ab7c59088f5e1d65cb0b482487e633064b868a155e765af", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"}
      : (tensor<64x64xbf16>) -> (tensor<64x64xi8>, tensor<64x2xi8>)
    %c2_0, %c2s_0 = "mx_gemmini.resident_contract"(
      %c1_0, %c1s_0, %b2, %b2s) {
      site_id = "functional:matmul_1", activation_row = 128 : i32,
      weight_row = 16128 : i32, output_row = 512 : i32,
      m = 64 : i32, n = 64 : i32, k = 64 : i32,
      activation_format = "fp8_e4m3", weight_format = "fp8_e4m3",
      output_format = "fp8_e4m3", weight_buffer = "b2_weight",
      weight_scales_buffer = "b2_scales",
      output_scales_buffer = "c2_scales_0", contract_sha256 = "641d60769c10d5ee6bcaa8e76b502266899be1b291bb2b42407a41712a5a8fbe", policy_sha256 = "207a28d2083dce6f5ab7c59088f5e1d65cb0b482487e633064b868a155e765af", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"}
      : (tensor<64x64xi8>, tensor<64x2xi8>, tensor<64x64xi8>, tensor<2x64xi8>)
      -> (tensor<64x64xi8>, tensor<64x2xi8>)
    %v1 = "mx_gemmini.vpu_execute"(%bf16) {
      site_id = "functional:matmul_2", kind = "muls", src1_row = 4608 : i32,
      src2_row = 0 : i32, dst_row = 4608 : i32, rows = 512 : i32,
      reduction_length = 1 : i32, broadcast = false,
      immediate_bf16 = 16512 : i32, contract_sha256 = "641d60769c10d5ee6bcaa8e76b502266899be1b291bb2b42407a41712a5a8fbe", policy_sha256 = "207a28d2083dce6f5ab7c59088f5e1d65cb0b482487e633064b868a155e765af", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"}
      : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %c1_1, %c1s_1 = "mx_gemmini.spad_requant"(%v1) {
      site_id = "functional:matmul_2", source_row = 4608 : i32,
      destination_row = 1024 : i32, m = 64 : i32, n = 64 : i32,
      output_format = "fp8_e4m3", tiled = true, resident = true,
      scale_dram_address = 0 : i64, scale_buffer = "c1_scales_1", contract_sha256 = "641d60769c10d5ee6bcaa8e76b502266899be1b291bb2b42407a41712a5a8fbe", policy_sha256 = "207a28d2083dce6f5ab7c59088f5e1d65cb0b482487e633064b868a155e765af", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"}
      : (tensor<64x64xbf16>) -> (tensor<64x64xi8>, tensor<64x2xi8>)
    %c2_1, %c2s_1 = "mx_gemmini.resident_contract"(
      %c1_1, %c1s_1, %b2, %b2s) {
      site_id = "functional:matmul_2", activation_row = 1024 : i32,
      weight_row = 16128 : i32, output_row = 1536 : i32,
      m = 64 : i32, n = 64 : i32, k = 64 : i32,
      activation_format = "fp8_e4m3", weight_format = "fp8_e4m3",
      output_format = "fp8_e4m3", weight_buffer = "b2_weight",
      weight_scales_buffer = "b2_scales",
      output_scales_buffer = "c2_scales_1", contract_sha256 = "641d60769c10d5ee6bcaa8e76b502266899be1b291bb2b42407a41712a5a8fbe", policy_sha256 = "207a28d2083dce6f5ab7c59088f5e1d65cb0b482487e633064b868a155e765af", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"}
      : (tensor<64x64xi8>, tensor<64x2xi8>, tensor<64x64xi8>, tensor<2x64xi8>)
      -> (tensor<64x64xi8>, tensor<64x2xi8>)
    func.return %c2_0, %c2s_0, %c2_1, %c2s_1
      : tensor<64x64xi8>, tensor<64x2xi8>, tensor<64x64xi8>, tensor<64x2xi8>
  }
}
