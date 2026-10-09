module attributes {mx.contract_sha256 = "ad450350be2f3e0cbcf6ca511b0e6ae3813eed8fc6d5205eb5f0e70739474696",
  mx.policy_sha256 = "04dc8f73257310f916ed8d6b0d48e49d57d32c928a561a08d42da1594317b186", prov.quantization_manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351",
  mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"} {
  func.func @nicolas_c1_vpu_requant() {
    "mx_gemmini.vpu_execute"() {site_id = "functional:matmul", kind = "muls",
      src1_row = 4096 : i32, src2_row = 0 : i32, dst_row = 4096 : i32,
      rows = 512 : i32, reduction_length = 1 : i32,
      broadcast = false, immediate_bf16 = 16384 : i32, contract_sha256 = "ad450350be2f3e0cbcf6ca511b0e6ae3813eed8fc6d5205eb5f0e70739474696", policy_sha256 = "04dc8f73257310f916ed8d6b0d48e49d57d32c928a561a08d42da1594317b186", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"} : () -> ()
    "mx_gemmini.spad_requant"() {site_id = "functional:matmul",
      source_row = 4096 : i32, destination_row = 128 : i32,
      m = 64 : i32, n = 64 : i32, output_format = "fp8_e4m3",
      tiled = true, resident = true, scale_dram_address = 0 : i64,
      scale_buffer = "c1_scales", contract_sha256 = "ad450350be2f3e0cbcf6ca511b0e6ae3813eed8fc6d5205eb5f0e70739474696", policy_sha256 = "04dc8f73257310f916ed8d6b0d48e49d57d32c928a561a08d42da1594317b186", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"} : () -> ()
    "mx_gemmini.resident_contract"() {site_id = "functional:matmul_1",
      activation_row = 128 : i32,
      weight_row = 16128 : i32, output_row = 512 : i32,
      m = 64 : i32, n = 64 : i32, k = 64 : i32,
      activation_format = "fp8_e4m3", weight_format = "fp8_e4m3",
      output_format = "fp8_e4m3", weight_buffer = "b2_weight",
      weight_scales_buffer = "b2_scales", output_scales_buffer = "c2_scales",
      contract_sha256 = "ad450350be2f3e0cbcf6ca511b0e6ae3813eed8fc6d5205eb5f0e70739474696", policy_sha256 = "04dc8f73257310f916ed8d6b0d48e49d57d32c928a561a08d42da1594317b186", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"} : () -> ()
    func.return
  }
}
