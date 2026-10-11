module attributes {mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d",
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", mx.policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3",
  prov.quantization_manifest_sha256 = "3cf9d96c52914369db0a772637be7f3f3e718f66f5e435d7f75cd22103d25a56",
  mx.frontend_mlir_sha256 = "4e16e387f86410812e9061c6dd3cf21765b0928af86c454fff8af4e6a57573da"} {
  func.func @model_attention_qk_head(%a: tensor<16x64xi8>,
      %as: tensor<2x16xi8>, %b: tensor<64x32xi8>,
      %bs: tensor<2x32xi8>) -> tensor<16x32xbf16> {
    %acc = "mx_gemmini.contract"(%a, %as, %b, %bs) {
      site_id = "functional:matmul", activation_format = "fp8_e4m3",
      weight_format = "fp8_e4m3", activation_projection = "direct",
      weight_projection = "direct", pe_mode = 8 : i32, contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3", manifest_sha256 = "3cf9d96c52914369db0a772637be7f3f3e718f66f5e435d7f75cd22103d25a56", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<16x64xi8>, tensor<2x16xi8>, tensor<64x32xi8>,
         tensor<2x32xi8>) -> tensor<16x32xbf16>
    %out = "mx_gemmini.readout_bf16"(%acc) {site_id = "functional:matmul", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3", manifest_sha256 = "3cf9d96c52914369db0a772637be7f3f3e718f66f5e435d7f75cd22103d25a56", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<16x32xbf16>) -> tensor<16x32xbf16>
    func.return %out : tensor<16x32xbf16>
  }
}
