module attributes {mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d",
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", mx.policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3",
  prov.quantization_manifest_sha256 = "feab2c324460c50b7c51042cb3dfbd5a97bc38303077c34406b986a76c5ea3c3",
  mx.frontend_mlir_sha256 = "8ae7b189276c2a2b10fba8e5c3af51278470d989a9973c92b48d41d71747f99f"} {
  func.func @model_attention_qk_head(%a: tensor<16x128xi8>,
      %as: tensor<4x16xi8>, %b: tensor<128x32xi8>,
      %bs: tensor<4x32xi8>) -> tensor<16x32xbf16> {
    %acc = "mx_gemmini.contract"(%a, %as, %b, %bs) {
      site_id = "functional:matmul", activation_format = "fp8_e4m3",
      weight_format = "fp8_e4m3", activation_projection = "direct",
      weight_projection = "direct", pe_mode = 8 : i32, contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3", manifest_sha256 = "feab2c324460c50b7c51042cb3dfbd5a97bc38303077c34406b986a76c5ea3c3", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<16x128xi8>, tensor<4x16xi8>, tensor<128x32xi8>,
         tensor<4x32xi8>) -> tensor<16x32xbf16>
    %out = "mx_gemmini.readout_bf16"(%acc) {site_id = "functional:matmul", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3", manifest_sha256 = "feab2c324460c50b7c51042cb3dfbd5a97bc38303077c34406b986a76c5ea3c3", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<16x32xbf16>) -> tensor<16x32xbf16>
    func.return %out : tensor<16x32xbf16>
  }
}
