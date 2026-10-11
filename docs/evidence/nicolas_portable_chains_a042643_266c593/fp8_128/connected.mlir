module attributes {mx.profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c",
  mx.contract_sha256 = "043b8140516ce726c389d6c9c26c131faac7ef00dcb24e6d5dbf2e00591db944", mx.policy_sha256 = "4dbfb74c66877071bfde688d4ba117a3ff668b4107209dbe4f1cc46f63d0f547",
  prov.quantization_manifest_sha256 = "875a0889c489b4f213e15b6f7e55aa6e49b1e7ba46de44c574a33e6b97487af0",
  mx.frontend_mlir_sha256 = "e78ddaebf3385984f2c5cf6a7ed5201fe679ed9a6c748624b4837a30b7048bff",
  mx.runtime_resources_sha256 = "a0a593b19ed951bc7a36d6206f2c0294f2b24b1c5ff24da8f1fe704aacfc6bdf"} {
  func.func @nicolas_plain_chain_128(
      %a1: tensor<128x128xi8>, %a1s: tensor<4x128xi8>,
      %b1: tensor<128x128xi8>, %b1s: tensor<4x128xi8>,
      %b2: tensor<128x128xi8>, %b2s: tensor<4x128xi8>)
      -> (tensor<128x128xi8>, tensor<128x4xi8>) {
    %acc = "mx_gemmini.contract"(%a1, %a1s, %b1, %b1s) {
      site_id = "functional:matmul", activation_format = "fp8_e4m3",
      weight_format = "fp8_e4m3", activation_projection = "direct",
      weight_projection = "direct", pe_mode = 8 : i32, contract_sha256 = "043b8140516ce726c389d6c9c26c131faac7ef00dcb24e6d5dbf2e00591db944", policy_sha256 = "4dbfb74c66877071bfde688d4ba117a3ff668b4107209dbe4f1cc46f63d0f547", manifest_sha256 = "875a0889c489b4f213e15b6f7e55aa6e49b1e7ba46de44c574a33e6b97487af0", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"}
      : (tensor<128x128xi8>, tensor<4x128xi8>, tensor<128x128xi8>, tensor<4x128xi8>)
      -> tensor<128x128xbf16>
    %c1, %c1s = "mx_gemmini.readout_quantized"(%acc) {
      site_id = "functional:matmul", output_format = "fp8_e4m3", contract_sha256 = "043b8140516ce726c389d6c9c26c131faac7ef00dcb24e6d5dbf2e00591db944", policy_sha256 = "4dbfb74c66877071bfde688d4ba117a3ff668b4107209dbe4f1cc46f63d0f547", manifest_sha256 = "875a0889c489b4f213e15b6f7e55aa6e49b1e7ba46de44c574a33e6b97487af0", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"}
      : (tensor<128x128xbf16>) -> (tensor<128x128xi8>, tensor<128x4xi8>)
    %c2, %c2s = "mx_gemmini.resident_contract"(%c1, %c1s, %b2, %b2s) {
      site_id = "functional:matmul_1", activation_row = 2048 : i32,
      weight_row = 15360 : i32, output_row = 4096 : i32,
      m = 128 : i32, n = 128 : i32, k = 128 : i32,
      activation_format = "fp8_e4m3", weight_format = "fp8_e4m3",
      output_format = "fp8_e4m3", weight_buffer = "b2_weight",
      weight_scales_buffer = "b2_scales", output_scales_buffer = "c2_scales",
      contract_sha256 = "043b8140516ce726c389d6c9c26c131faac7ef00dcb24e6d5dbf2e00591db944", policy_sha256 = "4dbfb74c66877071bfde688d4ba117a3ff668b4107209dbe4f1cc46f63d0f547", manifest_sha256 = "875a0889c489b4f213e15b6f7e55aa6e49b1e7ba46de44c574a33e6b97487af0", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"}
      : (tensor<128x128xi8>, tensor<128x4xi8>, tensor<128x128xi8>, tensor<4x128xi8>)
      -> (tensor<128x128xi8>, tensor<128x4xi8>)
    func.return %c2, %c2s : tensor<128x128xi8>, tensor<128x4xi8>
  }
}
