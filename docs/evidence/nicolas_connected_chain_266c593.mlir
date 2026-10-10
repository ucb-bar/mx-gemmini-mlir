module attributes {mx.contract_sha256 = "ad450350be2f3e0cbcf6ca511b0e6ae3813eed8fc6d5205eb5f0e70739474696",
  mx.policy_sha256 = "04dc8f73257310f916ed8d6b0d48e49d57d32c928a561a08d42da1594317b186", prov.quantization_manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351",
  mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d",
  mx.frontend_mlir_sha256 = "828e8a5cee173b181d080a6a71e2f0e91316a39d644d93249803efa8a38ef5bd",
  mx.seam_mlir_sha256 = "a86cea56dcfbe7ea1191c0e0d66dd2442031d509ee81f2c29de3cbb9b9fa3623",
  mx.source_facts_sha256 = "d42a6fe3f79905084ec7e86604d0095806dd9f6544bd8fc4220d7b44b3e2b1fd",
  mx.runtime_resources_sha256 = "30ce66dd215e851cbfff43f327657cb515e6dbda4a095158dd384545bce4f30f"} {
  func.func @nicolas_connected_chain(
      %a1: tensor<64x64xi8>, %a1s: tensor<2x64xi8>,
      %b1: tensor<64x64xi8>, %b1s: tensor<2x64xi8>,
      %b2: tensor<64x64xi8>, %b2s: tensor<2x64xi8>)
      -> (tensor<64x64xi8>, tensor<64x2xi8>) {
    %acc = "mx_gemmini.contract"(%a1, %a1s, %b1, %b1s) {
      site_id = "functional:matmul", activation_format = "fp8_e4m3",
      weight_format = "fp8_e4m3", activation_projection = "direct",
      weight_projection = "direct", pe_mode = 8 : i32, contract_sha256 = "ad450350be2f3e0cbcf6ca511b0e6ae3813eed8fc6d5205eb5f0e70739474696", policy_sha256 = "04dc8f73257310f916ed8d6b0d48e49d57d32c928a561a08d42da1594317b186", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<64x64xi8>, tensor<2x64xi8>, tensor<64x64xi8>, tensor<2x64xi8>)
      -> tensor<64x64xbf16>
    %bf16 = "mx_gemmini.readout_bf16"(%acc) {
      site_id = "functional:matmul", contract_sha256 = "ad450350be2f3e0cbcf6ca511b0e6ae3813eed8fc6d5205eb5f0e70739474696", policy_sha256 = "04dc8f73257310f916ed8d6b0d48e49d57d32c928a561a08d42da1594317b186", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %vpu = "mx_gemmini.vpu_execute"(%bf16) {
      site_id = "functional:matmul", kind = "muls",
      src1_row = 4096 : i32, src2_row = 0 : i32,
      dst_row = 4096 : i32, rows = 512 : i32,
      reduction_length = 1 : i32, broadcast = false,
      immediate_bf16 = 16384 : i32, contract_sha256 = "ad450350be2f3e0cbcf6ca511b0e6ae3813eed8fc6d5205eb5f0e70739474696", policy_sha256 = "04dc8f73257310f916ed8d6b0d48e49d57d32c928a561a08d42da1594317b186", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %c1, %c1s = "mx_gemmini.spad_requant"(%vpu) {
      site_id = "functional:matmul", source_row = 4096 : i32,
      destination_row = 128 : i32, m = 64 : i32, n = 64 : i32,
      output_format = "fp8_e4m3", tiled = true, resident = true,
      scale_dram_address = 0 : i64, scale_buffer = "c1_scales", contract_sha256 = "ad450350be2f3e0cbcf6ca511b0e6ae3813eed8fc6d5205eb5f0e70739474696", policy_sha256 = "04dc8f73257310f916ed8d6b0d48e49d57d32c928a561a08d42da1594317b186", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<64x64xbf16>) -> (tensor<64x64xi8>, tensor<64x2xi8>)
    %c2, %c2s = "mx_gemmini.resident_contract"(%c1, %c1s, %b2, %b2s) {
      site_id = "functional:matmul_1", activation_row = 128 : i32,
      weight_row = 16128 : i32, output_row = 512 : i32,
      m = 64 : i32, n = 64 : i32, k = 64 : i32,
      activation_format = "fp8_e4m3", weight_format = "fp8_e4m3",
      output_format = "fp8_e4m3", weight_buffer = "b2_weight",
      weight_scales_buffer = "b2_scales", output_scales_buffer = "c2_scales",
      contract_sha256 = "ad450350be2f3e0cbcf6ca511b0e6ae3813eed8fc6d5205eb5f0e70739474696", policy_sha256 = "04dc8f73257310f916ed8d6b0d48e49d57d32c928a561a08d42da1594317b186", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<64x64xi8>, tensor<64x2xi8>, tensor<64x64xi8>, tensor<2x64xi8>)
      -> (tensor<64x64xi8>, tensor<64x2xi8>)
    func.return %c2, %c2s : tensor<64x64xi8>, tensor<64x2xi8>
  }
}
