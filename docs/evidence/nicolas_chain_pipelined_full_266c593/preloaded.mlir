module attributes {mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d",
  mx.contract_sha256 = "e87136b4a341e95ae84d1455bc85428e625e92728db538820f936f3053a57ea2", mx.policy_sha256 = "207a28d2083dce6f5ab7c59088f5e1d65cb0b482487e633064b868a155e765af",
  prov.quantization_manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351",
  mx.frontend_mlir_sha256 = "046c964f2437cf8ff31c6ac155687516815959af13d02d792152013b438b18a7",
  mx.original_graph_sha256 = "4581e83379f6b3d345e303a47ca823143e355ed30a7542f1c1fe3921770df3aa",
  mx.runtime_resources_sha256 = "ceb14ac77659697442db425bd841d0bfca5a2ab40e3666d0c85b7feed1a55a8a"} {
  func.func @nicolas_chain_pipelined(
      %bf16: tensor<64x64xbf16>, %b2: tensor<64x64xi8>,
      %b2s: tensor<2x64xi8>)
      -> (tensor<64x64xi8>, tensor<64x2xi8>,
          tensor<64x64xi8>, tensor<64x2xi8>) {
    %v0 = "mx_gemmini.vpu_execute"(%bf16) {
      site_id = "functional:matmul_1", kind = "muls", src1_row = 4096 : i32,
      src2_row = 0 : i32, dst_row = 4096 : i32, rows = 512 : i32,
      reduction_length = 1 : i32, broadcast = false,
      immediate_bf16 = 16384 : i32, contract_sha256 = "e87136b4a341e95ae84d1455bc85428e625e92728db538820f936f3053a57ea2", policy_sha256 = "207a28d2083dce6f5ab7c59088f5e1d65cb0b482487e633064b868a155e765af", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %c1_0, %c1s_0 = "mx_gemmini.spad_requant"(%v0) {
      site_id = "functional:matmul_1", source_row = 4096 : i32,
      destination_row = 128 : i32, m = 64 : i32, n = 64 : i32,
      output_format = "fp8_e4m3", tiled = true, resident = true,
      scale_dram_address = 0 : i64, scale_buffer = "c1_scales_0", contract_sha256 = "e87136b4a341e95ae84d1455bc85428e625e92728db538820f936f3053a57ea2", policy_sha256 = "207a28d2083dce6f5ab7c59088f5e1d65cb0b482487e633064b868a155e765af", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<64x64xbf16>) -> (tensor<64x64xi8>, tensor<64x2xi8>)
    %c2_0, %c2s_0 = "mx_gemmini.resident_contract"(
      %c1_0, %c1s_0, %b2, %b2s) {
      site_id = "functional:matmul_1", activation_row = 128 : i32,
      weight_row = 16128 : i32, output_row = 512 : i32,
      m = 64 : i32, n = 64 : i32, k = 64 : i32,
      activation_format = "fp8_e4m3", weight_format = "fp8_e4m3",
      output_format = "fp8_e4m3", weight_buffer = "b2_weight",
      weight_scales_buffer = "b2_scales",
      output_scales_buffer = "c2_scales_0", contract_sha256 = "e87136b4a341e95ae84d1455bc85428e625e92728db538820f936f3053a57ea2", policy_sha256 = "207a28d2083dce6f5ab7c59088f5e1d65cb0b482487e633064b868a155e765af", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<64x64xi8>, tensor<64x2xi8>, tensor<64x64xi8>, tensor<2x64xi8>)
      -> (tensor<64x64xi8>, tensor<64x2xi8>)
    %v1 = "mx_gemmini.vpu_execute"(%bf16) {
      site_id = "functional:matmul_2", kind = "muls", src1_row = 4608 : i32,
      src2_row = 0 : i32, dst_row = 4608 : i32, rows = 512 : i32,
      reduction_length = 1 : i32, broadcast = false,
      immediate_bf16 = 16512 : i32, contract_sha256 = "e87136b4a341e95ae84d1455bc85428e625e92728db538820f936f3053a57ea2", policy_sha256 = "207a28d2083dce6f5ab7c59088f5e1d65cb0b482487e633064b868a155e765af", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %c1_1, %c1s_1 = "mx_gemmini.spad_requant"(%v1) {
      site_id = "functional:matmul_2", source_row = 4608 : i32,
      destination_row = 1024 : i32, m = 64 : i32, n = 64 : i32,
      output_format = "fp8_e4m3", tiled = true, resident = true,
      scale_dram_address = 0 : i64, scale_buffer = "c1_scales_1", contract_sha256 = "e87136b4a341e95ae84d1455bc85428e625e92728db538820f936f3053a57ea2", policy_sha256 = "207a28d2083dce6f5ab7c59088f5e1d65cb0b482487e633064b868a155e765af", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<64x64xbf16>) -> (tensor<64x64xi8>, tensor<64x2xi8>)
    %c2_1, %c2s_1 = "mx_gemmini.resident_contract"(
      %c1_1, %c1s_1, %b2, %b2s) {
      site_id = "functional:matmul_2", activation_row = 1024 : i32,
      weight_row = 16128 : i32, output_row = 1536 : i32,
      m = 64 : i32, n = 64 : i32, k = 64 : i32,
      activation_format = "fp8_e4m3", weight_format = "fp8_e4m3",
      output_format = "fp8_e4m3", weight_buffer = "b2_weight",
      weight_scales_buffer = "b2_scales",
      output_scales_buffer = "c2_scales_1", contract_sha256 = "e87136b4a341e95ae84d1455bc85428e625e92728db538820f936f3053a57ea2", policy_sha256 = "207a28d2083dce6f5ab7c59088f5e1d65cb0b482487e633064b868a155e765af", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<64x64xi8>, tensor<64x2xi8>, tensor<64x64xi8>, tensor<2x64xi8>)
      -> (tensor<64x64xi8>, tensor<64x2xi8>)
    func.return %c2_0, %c2s_0, %c2_1, %c2s_1
      : tensor<64x64xi8>, tensor<64x2xi8>, tensor<64x64xi8>, tensor<64x2xi8>
  }
}
