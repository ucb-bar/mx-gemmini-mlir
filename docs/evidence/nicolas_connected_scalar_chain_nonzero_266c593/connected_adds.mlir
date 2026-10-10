module attributes {mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d",
  mx.contract_sha256 = "7004650bd9cd355822bd8f03adfa6cbcafd9102c587a810a9dfcc20c573e8530",
  mx.policy_sha256 = "04dc8f73257310f916ed8d6b0d48e49d57d32c928a561a08d42da1594317b186",
  prov.quantization_manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351",
  mx.frontend_mlir_sha256 = "850021bb47cf0104c740642bdb2f19c7e83706a687a7d01ce746b53c28090955",
  mx.runtime_resources_sha256 = "01609c1895f9f0d2d628e8068996d4a6c13331b4d2debdba6a81b6cd850ea77a"} {
  func.func @nicolas_narrow_vpu_pair(
      %a1: tensor<64x64xi8>, %a1s: tensor<2x64xi8>,
      %b1: tensor<64x64xi8>, %b1s: tensor<2x64xi8>,
      %b2: tensor<64x32xi8>, %b2s: tensor<2x32xi8>)
      -> (tensor<64x32xi8>, tensor<64x1xi8>) {
    %acc = "mx_gemmini.contract"(%a1, %a1s, %b1, %b1s) {
      site_id = "functional:matmul", activation_format = "fp8_e4m3",
      weight_format = "fp8_e4m3", activation_projection = "direct",
      weight_projection = "direct", pe_mode = 8 : i32, contract_sha256 = "7004650bd9cd355822bd8f03adfa6cbcafd9102c587a810a9dfcc20c573e8530", policy_sha256 = "04dc8f73257310f916ed8d6b0d48e49d57d32c928a561a08d42da1594317b186", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<64x64xi8>, tensor<2x64xi8>, tensor<64x64xi8>, tensor<2x64xi8>)
      -> tensor<64x64xbf16>
    %bf16 = "mx_gemmini.readout_bf16"(%acc) {
      site_id = "functional:matmul", contract_sha256 = "7004650bd9cd355822bd8f03adfa6cbcafd9102c587a810a9dfcc20c573e8530", policy_sha256 = "04dc8f73257310f916ed8d6b0d48e49d57d32c928a561a08d42da1594317b186", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %vpu = "mx_gemmini.vpu_execute"(%bf16) {
      site_id = "functional:matmul", kind = "muls",
      src1_row = 4096 : i32, src2_row = 0 : i32,
      dst_row = 4096 : i32, rows = 512 : i32,
      reduction_length = 1 : i32, broadcast = false,
      immediate_bf16 = 16384 : i32, contract_sha256 = "7004650bd9cd355822bd8f03adfa6cbcafd9102c587a810a9dfcc20c573e8530", policy_sha256 = "04dc8f73257310f916ed8d6b0d48e49d57d32c928a561a08d42da1594317b186", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %vpu2 = "mx_gemmini.vpu_execute"(%vpu) {
      site_id = "functional:matmul", kind = "adds",
      src1_row = 4096 : i32, src2_row = 0 : i32,
      dst_row = 4096 : i32, rows = 512 : i32,
      reduction_length = 1 : i32, broadcast = false,
      immediate_bf16 = 16320 : i32, contract_sha256 = "7004650bd9cd355822bd8f03adfa6cbcafd9102c587a810a9dfcc20c573e8530", policy_sha256 = "04dc8f73257310f916ed8d6b0d48e49d57d32c928a561a08d42da1594317b186", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %c1, %c1s = "mx_gemmini.spad_requant"(%vpu2) {
      site_id = "functional:matmul", source_row = 4096 : i32,
      destination_row = 128 : i32, m = 64 : i32, n = 64 : i32,
      output_format = "fp8_e4m3", tiled = true, resident = true,
      scale_dram_address = 0 : i64, scale_buffer = "c1_scales", contract_sha256 = "7004650bd9cd355822bd8f03adfa6cbcafd9102c587a810a9dfcc20c573e8530", policy_sha256 = "04dc8f73257310f916ed8d6b0d48e49d57d32c928a561a08d42da1594317b186", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<64x64xbf16>) -> (tensor<64x64xi8>, tensor<64x2xi8>)
    %c2, %c2s = "mx_gemmini.resident_contract"(%c1, %c1s, %b2, %b2s) {
      site_id = "functional:matmul_1", activation_row = 128 : i32,
      weight_row = 16256 : i32, output_row = 512 : i32,
      m = 64 : i32, n = 32 : i32, k = 64 : i32,
      activation_format = "fp8_e4m3", weight_format = "fp8_e4m3",
      output_format = "fp8_e4m3", weight_buffer = "b2_weight",
      weight_scales_buffer = "b2_scales", output_scales_buffer = "c2_scales",
      contract_sha256 = "7004650bd9cd355822bd8f03adfa6cbcafd9102c587a810a9dfcc20c573e8530", policy_sha256 = "04dc8f73257310f916ed8d6b0d48e49d57d32c928a561a08d42da1594317b186", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<64x64xi8>, tensor<64x2xi8>, tensor<64x32xi8>, tensor<2x32xi8>)
      -> (tensor<64x32xi8>, tensor<64x1xi8>)
    func.return %c2, %c2s : tensor<64x32xi8>, tensor<64x1xi8>
  }
}
