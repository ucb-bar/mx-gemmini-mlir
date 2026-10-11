module attributes {mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d",
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", mx.policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3",
  prov.quantization_manifest_sha256 = "92c8b37ba64219f719f01b3159d9adb1644cca97a9b3e89a806b7cab90cc9c50",
  mx.frontend_mlir_sha256 = "0a2e3b026f503cd5b24336d13fcd91bcb714a1315b0f3ac1f914ab9eb46ab58c"} {
  func.func @model_projection_slice(%a: tensor<32x768xi8>,
      %as: tensor<24x32xi8>, %b: tensor<768x768xi8>,
      %bs: tensor<24x768xi8>) -> tensor<32x768xbf16> {
    %acc = "mx_gemmini.contract"(%a, %as, %b, %bs) {
      site_id = "functional:matmul", activation_format = "fp4_e2m1",
      weight_format = "fp4_e2m1", activation_projection = "direct",
      weight_projection = "direct", pe_mode = 0 : i32, contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3", manifest_sha256 = "92c8b37ba64219f719f01b3159d9adb1644cca97a9b3e89a806b7cab90cc9c50", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<32x768xi8>, tensor<24x32xi8>, tensor<768x768xi8>,
         tensor<24x768xi8>) -> tensor<32x768xbf16>
    %out = "mx_gemmini.readout_bf16"(%acc) {site_id = "functional:matmul", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3", manifest_sha256 = "92c8b37ba64219f719f01b3159d9adb1644cca97a9b3e89a806b7cab90cc9c50", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<32x768xbf16>) -> tensor<32x768xbf16>
    func.return %out : tensor<32x768xbf16>
  }
}
