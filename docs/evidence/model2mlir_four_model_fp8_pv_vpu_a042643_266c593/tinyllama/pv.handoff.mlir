module attributes {mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d",
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", mx.policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3",
  prov.quantization_manifest_sha256 = "1858aaa2ca16ca6bdea3c8a97ce44828ad335c297d20f4a06c1a58122f8c3cb2",
  mx.frontend_mlir_sha256 = "d54716a0ec705a1cb8d4f5204f84f19be0b4fc03e101d5ab637fd7cdbfc9fd34"} {
  func.func @model_attention_pv_head(%a: tensor<16x32xi8>,
      %as: tensor<1x16xi8>, %b: tensor<32x64xi8>,
      %bs: tensor<1x64xi8>) -> tensor<16x64xbf16> {
    %acc = "mx_gemmini.contract"(%a, %as, %b, %bs) {
      site_id = "functional:matmul", activation_format = "fp8_e4m3",
      weight_format = "fp8_e4m3", activation_projection = "direct",
      weight_projection = "direct", pe_mode = 8 : i32, contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3", manifest_sha256 = "1858aaa2ca16ca6bdea3c8a97ce44828ad335c297d20f4a06c1a58122f8c3cb2", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<16x32xi8>, tensor<1x16xi8>, tensor<32x64xi8>,
         tensor<1x64xi8>) -> tensor<16x64xbf16>
    %out = "mx_gemmini.readout_bf16"(%acc) {site_id = "functional:matmul", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3", manifest_sha256 = "1858aaa2ca16ca6bdea3c8a97ce44828ad335c297d20f4a06c1a58122f8c3cb2", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<16x64xbf16>) -> tensor<16x64xbf16>
    func.return %out : tensor<16x64xbf16>
  }
}
