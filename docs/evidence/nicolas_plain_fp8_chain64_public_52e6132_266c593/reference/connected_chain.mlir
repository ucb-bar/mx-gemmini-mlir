module attributes {mx.profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c",
  mx.contract_sha256 = "54ecee5e342f8ae62cfb0b64313019a5ab20a740dc6259f77b67808c2fc816bf", mx.policy_sha256 = "1e5e1d5295f46be815411a941646f84f6acf2cdb53507f923861312dd4123d78",
  prov.quantization_manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351",
  mx.frontend_mlir_sha256 = "4e022769011c6492404d83e32bcfce2264f35d116307d7cbe2dedd7eff30de4a",
  mx.runtime_resources_sha256 = "30ce66dd215e851cbfff43f327657cb515e6dbda4a095158dd384545bce4f30f"} {
  func.func @nicolas_plain_chain_64(
      %a1: tensor<64x64xi8>, %a1s: tensor<2x64xi8>,
      %b1: tensor<64x64xi8>, %b1s: tensor<2x64xi8>,
      %b2: tensor<64x64xi8>, %b2s: tensor<2x64xi8>)
      -> (tensor<64x64xi8>, tensor<64x2xi8>) {
    %acc = "mx_gemmini.contract"(%a1, %a1s, %b1, %b1s) {
      site_id = "functional:matmul", activation_format = "fp8_e4m3",
      weight_format = "fp8_e4m3", activation_projection = "direct",
      weight_projection = "direct", pe_mode = 8 : i32, contract_sha256 = "54ecee5e342f8ae62cfb0b64313019a5ab20a740dc6259f77b67808c2fc816bf", policy_sha256 = "1e5e1d5295f46be815411a941646f84f6acf2cdb53507f923861312dd4123d78", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"}
      : (tensor<64x64xi8>, tensor<2x64xi8>, tensor<64x64xi8>, tensor<2x64xi8>)
      -> tensor<64x64xbf16>
    %c1, %c1s = "mx_gemmini.readout_quantized"(%acc) {
      site_id = "functional:matmul", output_format = "fp8_e4m3", contract_sha256 = "54ecee5e342f8ae62cfb0b64313019a5ab20a740dc6259f77b67808c2fc816bf", policy_sha256 = "1e5e1d5295f46be815411a941646f84f6acf2cdb53507f923861312dd4123d78", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"}
      : (tensor<64x64xbf16>) -> (tensor<64x64xi8>, tensor<64x2xi8>)
    %c2, %c2s = "mx_gemmini.resident_contract"(%c1, %c1s, %b2, %b2s) {
      site_id = "functional:matmul_1", activation_row = 128 : i32,
      weight_row = 16128 : i32, output_row = 512 : i32,
      m = 64 : i32, n = 64 : i32, k = 64 : i32,
      activation_format = "fp8_e4m3", weight_format = "fp8_e4m3",
      output_format = "fp8_e4m3", weight_buffer = "b2_weight",
      weight_scales_buffer = "b2_scales", output_scales_buffer = "c2_scales",
      contract_sha256 = "54ecee5e342f8ae62cfb0b64313019a5ab20a740dc6259f77b67808c2fc816bf", policy_sha256 = "1e5e1d5295f46be815411a941646f84f6acf2cdb53507f923861312dd4123d78", manifest_sha256 = "8425226b26ee6c6153de9e0f6969575399ed7216efd72a1f7c441444c43fc351", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"}
      : (tensor<64x64xi8>, tensor<64x2xi8>, tensor<64x64xi8>, tensor<2x64xi8>)
      -> (tensor<64x64xi8>, tensor<64x2xi8>)
    func.return %c2, %c2s : tensor<64x64xi8>, tensor<64x2xi8>
  }
}
