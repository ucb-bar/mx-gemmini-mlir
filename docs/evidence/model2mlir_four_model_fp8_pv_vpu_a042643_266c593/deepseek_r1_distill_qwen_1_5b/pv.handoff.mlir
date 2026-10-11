module attributes {mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d",
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", mx.policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3",
  prov.quantization_manifest_sha256 = "58e83ec1aa8a13954869a6a51ab2fa220707a6ec9530f1741963ca124729d5a3",
  mx.frontend_mlir_sha256 = "45ea42d40f34b476a8c042858a3e03b0259c93a0457de3bb0d4ed1230b2f4a63"} {
  func.func @model_attention_pv_head(%a: tensor<16x32xi8>,
      %as: tensor<1x16xi8>, %b: tensor<32x128xi8>,
      %bs: tensor<1x128xi8>) -> tensor<16x128xbf16> {
    %acc = "mx_gemmini.contract"(%a, %as, %b, %bs) {
      site_id = "functional:matmul", activation_format = "fp8_e4m3",
      weight_format = "fp8_e4m3", activation_projection = "direct",
      weight_projection = "direct", pe_mode = 8 : i32, contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3", manifest_sha256 = "58e83ec1aa8a13954869a6a51ab2fa220707a6ec9530f1741963ca124729d5a3", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<16x32xi8>, tensor<1x16xi8>, tensor<32x128xi8>,
         tensor<1x128xi8>) -> tensor<16x128xbf16>
    %out = "mx_gemmini.readout_bf16"(%acc) {site_id = "functional:matmul", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3", manifest_sha256 = "58e83ec1aa8a13954869a6a51ab2fa220707a6ec9530f1741963ca124729d5a3", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<16x128xbf16>) -> tensor<16x128xbf16>
    func.return %out : tensor<16x128xbf16>
  }
}
