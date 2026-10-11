module attributes {mx.profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c",
  mx.contract_sha256 = "79d924969bc431427f3afecbc11b7895d1e6314cd7019d8b8743ea8281682219", mx.policy_sha256 = "074720a79a433fddf2658b6ba48e30538371ef6a44886e5de9c70c6a3ce8516e",
  prov.quantization_manifest_sha256 = "a7825848f2a6f52cbbdae241bd5c710b376db48e61d6a199d1369fd816ae12e0",
  mx.frontend_mlir_sha256 = "e78ddaebf3385984f2c5cf6a7ed5201fe679ed9a6c748624b4837a30b7048bff",
  mx.frontend_manifest_sha256 = "f5aa869d14343a0b2922d95e0e9b7dbc81c6800828dc807451599eb57dd731d5",
  mx.runtime_resources_sha256 = "5c4d5289efc0dd21a171dfcf866212eec4a53744996b409f6f87650ec270af1c"} {
  func.func @nicolas_plain_fp4_chain(
      %a1: tensor<64x128xi8>, %a1s: tensor<4x128xi8>,
      %b1: tensor<128x64xi8>, %b1s: tensor<4x128xi8>,
      %b2: tensor<128x64xi8>, %b2s: tensor<4x128xi8>)
      -> (tensor<64x128xi8>, tensor<128x4xi8>) {
    %acc = "mx_gemmini.contract"(%a1, %a1s, %b1, %b1s) {
      site_id = "functional:matmul", activation_format = "fp4_e2m1",
      weight_format = "fp4_e2m1", activation_projection = "direct",
      weight_projection = "direct", pe_mode = 0 : i32, contract_sha256 = "79d924969bc431427f3afecbc11b7895d1e6314cd7019d8b8743ea8281682219", policy_sha256 = "074720a79a433fddf2658b6ba48e30538371ef6a44886e5de9c70c6a3ce8516e", manifest_sha256 = "a7825848f2a6f52cbbdae241bd5c710b376db48e61d6a199d1369fd816ae12e0", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"}
      : (tensor<64x128xi8>, tensor<4x128xi8>, tensor<128x64xi8>, tensor<4x128xi8>)
      -> tensor<128x128xbf16>
    %c1, %c1s = "mx_gemmini.readout_quantized"(%acc) {
      site_id = "functional:matmul", output_format = "fp4_e2m1", contract_sha256 = "79d924969bc431427f3afecbc11b7895d1e6314cd7019d8b8743ea8281682219", policy_sha256 = "074720a79a433fddf2658b6ba48e30538371ef6a44886e5de9c70c6a3ce8516e", manifest_sha256 = "a7825848f2a6f52cbbdae241bd5c710b376db48e61d6a199d1369fd816ae12e0", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"}
      : (tensor<128x128xbf16>) -> (tensor<64x128xi8>, tensor<128x4xi8>)
    %c2, %c2s = "mx_gemmini.resident_contract"(%c1, %c1s, %b2, %b2s) {
      site_id = "functional:matmul_1", activation_row = 2048 : i32,
      weight_row = 15872 : i32, output_row = 4096 : i32,
      m = 128 : i32, n = 128 : i32, k = 128 : i32,
      activation_format = "fp4_e2m1", weight_format = "fp4_e2m1",
      output_format = "fp4_e2m1", weight_buffer = "b2_weight",
      weight_scales_buffer = "b2_scales", output_scales_buffer = "c2_scales",
      contract_sha256 = "79d924969bc431427f3afecbc11b7895d1e6314cd7019d8b8743ea8281682219", policy_sha256 = "074720a79a433fddf2658b6ba48e30538371ef6a44886e5de9c70c6a3ce8516e", manifest_sha256 = "a7825848f2a6f52cbbdae241bd5c710b376db48e61d6a199d1369fd816ae12e0", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"}
      : (tensor<64x128xi8>, tensor<128x4xi8>, tensor<128x64xi8>, tensor<4x128xi8>)
      -> (tensor<64x128xi8>, tensor<128x4xi8>)
    func.return %c2, %c2s : tensor<64x128xi8>, tensor<128x4xi8>
  }
}
