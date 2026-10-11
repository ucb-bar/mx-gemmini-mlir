module attributes {mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d",
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", mx.policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3",
  prov.quantization_manifest_sha256 = "cef599637d5a46162d15ebc4c15ce5149ecc9a9caaca3f5390ae89c3d5864795",
  mx.frontend_mlir_sha256 = "a775fe2dd058052918fac410be41b978bc8d1fe574b5780ccfa065613eecebb5"} {
  func.func @model_attention_pv_head(%a: tensor<16x32xi8>,
      %as: tensor<1x16xi8>, %b: tensor<32x256xi8>,
      %bs: tensor<1x256xi8>) -> tensor<16x256xbf16> {
    %acc = "mx_gemmini.contract"(%a, %as, %b, %bs) {
      site_id = "functional:matmul", activation_format = "fp8_e4m3",
      weight_format = "fp8_e4m3", activation_projection = "direct",
      weight_projection = "direct", pe_mode = 8 : i32, contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3", manifest_sha256 = "cef599637d5a46162d15ebc4c15ce5149ecc9a9caaca3f5390ae89c3d5864795", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<16x32xi8>, tensor<1x16xi8>, tensor<32x256xi8>,
         tensor<1x256xi8>) -> tensor<16x256xbf16>
    %out = "mx_gemmini.readout_bf16"(%acc) {site_id = "functional:matmul", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3", manifest_sha256 = "cef599637d5a46162d15ebc4c15ce5149ecc9a9caaca3f5390ae89c3d5864795", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<16x256xbf16>) -> tensor<16x256xbf16>
    func.return %out : tensor<16x256xbf16>
  }
}
