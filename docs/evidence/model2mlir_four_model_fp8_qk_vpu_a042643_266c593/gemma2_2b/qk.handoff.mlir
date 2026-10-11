module attributes {mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d",
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", mx.policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3",
  prov.quantization_manifest_sha256 = "106f949900d3840414f5fbe6245db023139f3bd7cd88ced6b7467da65edd38f9",
  mx.frontend_mlir_sha256 = "f5da6f7d9881b67fb64e1082a71e7bfdbe2d2427738ca70bd3e09921fd40767c"} {
  func.func @model_attention_qk_head(%a: tensor<16x256xi8>,
      %as: tensor<8x16xi8>, %b: tensor<256x32xi8>,
      %bs: tensor<8x32xi8>) -> tensor<16x32xbf16> {
    %acc = "mx_gemmini.contract"(%a, %as, %b, %bs) {
      site_id = "functional:matmul", activation_format = "fp8_e4m3",
      weight_format = "fp8_e4m3", activation_projection = "direct",
      weight_projection = "direct", pe_mode = 8 : i32, contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3", manifest_sha256 = "106f949900d3840414f5fbe6245db023139f3bd7cd88ced6b7467da65edd38f9", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<16x256xi8>, tensor<8x16xi8>, tensor<256x32xi8>,
         tensor<8x32xi8>) -> tensor<16x32xbf16>
    %out = "mx_gemmini.readout_bf16"(%acc) {site_id = "functional:matmul", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3", manifest_sha256 = "106f949900d3840414f5fbe6245db023139f3bd7cd88ced6b7467da65edd38f9", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<16x32xbf16>) -> tensor<16x32xbf16>
    func.return %out : tensor<16x32xbf16>
  }
}
