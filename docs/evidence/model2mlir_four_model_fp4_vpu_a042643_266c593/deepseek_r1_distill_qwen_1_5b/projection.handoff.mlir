module attributes {mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d",
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", mx.policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3",
  prov.quantization_manifest_sha256 = "cf10d3812493c30dbebb9bb2913547e2b12bfd82dad511a8e0745169073b8ccf",
  mx.frontend_mlir_sha256 = "c8962bf2f276d4ed3f21adbfb4c167febdcf3e8c50b51c275825f7397de9a352"} {
  func.func @model_projection_slice(%a: tensor<32x1536xi8>,
      %as: tensor<48x32xi8>, %b: tensor<1536x1536xi8>,
      %bs: tensor<48x1536xi8>) -> tensor<32x1536xbf16> {
    %acc = "mx_gemmini.contract"(%a, %as, %b, %bs) {
      site_id = "functional:matmul", activation_format = "fp4_e2m1",
      weight_format = "fp4_e2m1", activation_projection = "direct",
      weight_projection = "direct", pe_mode = 0 : i32, contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3", manifest_sha256 = "cf10d3812493c30dbebb9bb2913547e2b12bfd82dad511a8e0745169073b8ccf", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<32x1536xi8>, tensor<48x32xi8>, tensor<1536x1536xi8>,
         tensor<48x1536xi8>) -> tensor<32x1536xbf16>
    %out = "mx_gemmini.readout_bf16"(%acc) {site_id = "functional:matmul", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3", manifest_sha256 = "cf10d3812493c30dbebb9bb2913547e2b12bfd82dad511a8e0745169073b8ccf", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<32x1536xbf16>) -> tensor<32x1536xbf16>
    func.return %out : tensor<32x1536xbf16>
  }
}
