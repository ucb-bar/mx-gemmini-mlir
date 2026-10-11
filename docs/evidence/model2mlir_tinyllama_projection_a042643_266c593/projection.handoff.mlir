module attributes {mx.profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c",
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", mx.policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3",
  prov.quantization_manifest_sha256 = "6e742f124b0141713e4efcd3a967cad49a3fd6e927bd1b0ebfafc77f262878ca",
  mx.frontend_mlir_sha256 = "4dd4648807bb3b147bb0bfb0e66a034865b69a9e64956afb7900280aa16038b3"} {
  func.func @model_projection_slice(%a: tensor<16x2048xi8>,
      %as: tensor<64x16xi8>, %b: tensor<2048x32xi8>,
      %bs: tensor<64x32xi8>) -> tensor<16x32xbf16> {
    %acc = "mx_gemmini.contract"(%a, %as, %b, %bs) {
      site_id = "functional:matmul", activation_format = "fp8_e4m3",
      weight_format = "fp8_e4m3", activation_projection = "direct",
      weight_projection = "direct", pe_mode = 8 : i32, contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3", manifest_sha256 = "6e742f124b0141713e4efcd3a967cad49a3fd6e927bd1b0ebfafc77f262878ca", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"}
      : (tensor<16x2048xi8>, tensor<64x16xi8>, tensor<2048x32xi8>,
         tensor<64x32xi8>) -> tensor<16x32xbf16>
    %out = "mx_gemmini.readout_bf16"(%acc) {site_id = "functional:matmul", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3", manifest_sha256 = "6e742f124b0141713e4efcd3a967cad49a3fd6e927bd1b0ebfafc77f262878ca", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"}
      : (tensor<16x32xbf16>) -> tensor<16x32xbf16>
    func.return %out : tensor<16x32xbf16>
  }
}
