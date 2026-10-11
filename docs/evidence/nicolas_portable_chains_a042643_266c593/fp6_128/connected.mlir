module attributes {mx.profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c",
  mx.contract_sha256 = "236dd96dc3385c83755112e75453927a746d49a01b7407c4265f00e00ecab0a0", mx.policy_sha256 = "25f828f236ae064668e90784bad6dccc2788bd334eab5c0bb98ea71aab397e4d",
  prov.quantization_manifest_sha256 = "50dadea0cb83cc416a501a347ba3716de514e5aeedeec5a11fedeea9d1abd0c8",
  mx.frontend_mlir_sha256 = "e78ddaebf3385984f2c5cf6a7ed5201fe679ed9a6c748624b4837a30b7048bff",
  mx.frontend_manifest_sha256 = "b90008af024325e3346e1e3570e182062687b20e7cde718fb2b3a99c867f9406",
  mx.runtime_resources_sha256 = "9a71f70169d0deca1970376b80e77e713bce954ee51a84ee75a3e2b94662fa2b"} {
  func.func @nicolas_plain_fp6_chain(
      %a1: tensor<64x128xi8>, %a1s: tensor<4x128xi8>,
      %b1: tensor<128x64xi8>, %b1s: tensor<4x128xi8>,
      %b2: tensor<128x64xi8>, %b2s: tensor<4x128xi8>,
      %a1lut: tensor<64x3xi32>, %b1lut: tensor<64x3xi32>,
      %b2lut: tensor<64x3xi32>, %c1lut: tensor<64x3xi32>,
      %c2lut: tensor<64x3xi32>)
      -> (tensor<64x128xi8>, tensor<128x4xi8>) {
    "mx_gemmini.runtime_lut"(%b1lut) {site_id = "functional:matmul", lut_target = "weight", runtime_buffer = "b1_lut", groups = 64 : i32, entry_bits = 6 : i32, contract_sha256 = "236dd96dc3385c83755112e75453927a746d49a01b7407c4265f00e00ecab0a0", policy_sha256 = "25f828f236ae064668e90784bad6dccc2788bd334eab5c0bb98ea71aab397e4d", manifest_sha256 = "50dadea0cb83cc416a501a347ba3716de514e5aeedeec5a11fedeea9d1abd0c8", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"} : (tensor<64x3xi32>) -> ()
    "mx_gemmini.runtime_lut"(%a1lut) {site_id = "functional:matmul", lut_target = "activation", runtime_buffer = "a1_lut", groups = 64 : i32, entry_bits = 6 : i32, contract_sha256 = "236dd96dc3385c83755112e75453927a746d49a01b7407c4265f00e00ecab0a0", policy_sha256 = "25f828f236ae064668e90784bad6dccc2788bd334eab5c0bb98ea71aab397e4d", manifest_sha256 = "50dadea0cb83cc416a501a347ba3716de514e5aeedeec5a11fedeea9d1abd0c8", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"} : (tensor<64x3xi32>) -> ()
    "mx_gemmini.runtime_lut"(%c1lut) {site_id = "functional:matmul", lut_target = "output", runtime_buffer = "c1_lut", groups = 64 : i32, entry_bits = 6 : i32, contract_sha256 = "236dd96dc3385c83755112e75453927a746d49a01b7407c4265f00e00ecab0a0", policy_sha256 = "25f828f236ae064668e90784bad6dccc2788bd334eab5c0bb98ea71aab397e4d", manifest_sha256 = "50dadea0cb83cc416a501a347ba3716de514e5aeedeec5a11fedeea9d1abd0c8", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"} : (tensor<64x3xi32>) -> ()
    %acc = "mx_gemmini.contract"(%a1, %a1s, %b1, %b1s) {
      site_id = "functional:matmul", activation_format = "fp6_e3m2",
      weight_format = "fp6_e3m2", activation_projection = "lut",
      weight_projection = "lut", pe_mode = 4 : i32, contract_sha256 = "236dd96dc3385c83755112e75453927a746d49a01b7407c4265f00e00ecab0a0", policy_sha256 = "25f828f236ae064668e90784bad6dccc2788bd334eab5c0bb98ea71aab397e4d", manifest_sha256 = "50dadea0cb83cc416a501a347ba3716de514e5aeedeec5a11fedeea9d1abd0c8", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"}
      : (tensor<64x128xi8>, tensor<4x128xi8>, tensor<128x64xi8>, tensor<4x128xi8>)
      -> tensor<128x128xbf16>
    %c1, %c1s = "mx_gemmini.readout_quantized"(%acc) {
      site_id = "functional:matmul", output_format = "fp6_e3m2", contract_sha256 = "236dd96dc3385c83755112e75453927a746d49a01b7407c4265f00e00ecab0a0", policy_sha256 = "25f828f236ae064668e90784bad6dccc2788bd334eab5c0bb98ea71aab397e4d", manifest_sha256 = "50dadea0cb83cc416a501a347ba3716de514e5aeedeec5a11fedeea9d1abd0c8", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"}
      : (tensor<128x128xbf16>) -> (tensor<64x128xi8>, tensor<128x4xi8>)
    "mx_gemmini.runtime_lut"(%b2lut) {site_id = "functional:matmul_1", lut_target = "weight", runtime_buffer = "b2_lut", groups = 64 : i32, entry_bits = 6 : i32, contract_sha256 = "236dd96dc3385c83755112e75453927a746d49a01b7407c4265f00e00ecab0a0", policy_sha256 = "25f828f236ae064668e90784bad6dccc2788bd334eab5c0bb98ea71aab397e4d", manifest_sha256 = "50dadea0cb83cc416a501a347ba3716de514e5aeedeec5a11fedeea9d1abd0c8", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"} : (tensor<64x3xi32>) -> ()
    "mx_gemmini.runtime_lut"(%c1lut) {site_id = "functional:matmul_1", lut_target = "activation", runtime_buffer = "c1_lut", groups = 64 : i32, entry_bits = 6 : i32, contract_sha256 = "236dd96dc3385c83755112e75453927a746d49a01b7407c4265f00e00ecab0a0", policy_sha256 = "25f828f236ae064668e90784bad6dccc2788bd334eab5c0bb98ea71aab397e4d", manifest_sha256 = "50dadea0cb83cc416a501a347ba3716de514e5aeedeec5a11fedeea9d1abd0c8", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"} : (tensor<64x3xi32>) -> ()
    "mx_gemmini.runtime_lut"(%c2lut) {site_id = "functional:matmul_1", lut_target = "output", runtime_buffer = "c2_lut", groups = 64 : i32, entry_bits = 6 : i32, contract_sha256 = "236dd96dc3385c83755112e75453927a746d49a01b7407c4265f00e00ecab0a0", policy_sha256 = "25f828f236ae064668e90784bad6dccc2788bd334eab5c0bb98ea71aab397e4d", manifest_sha256 = "50dadea0cb83cc416a501a347ba3716de514e5aeedeec5a11fedeea9d1abd0c8", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"} : (tensor<64x3xi32>) -> ()
    %c2, %c2s = "mx_gemmini.resident_contract"(%c1, %c1s, %b2, %b2s) {
      site_id = "functional:matmul_1", activation_row = 2048 : i32,
      weight_row = 15872 : i32, output_row = 4096 : i32,
      m = 128 : i32, n = 128 : i32, k = 128 : i32,
      activation_format = "fp6_e3m2", weight_format = "fp6_e3m2",
      output_format = "fp6_e3m2", weight_buffer = "b2_weight",
      weight_scales_buffer = "b2_scales", output_scales_buffer = "c2_scales",
      weight_lut_buffer = "b2_lut", activation_lut_buffer = "c1_lut",
      output_lut_buffer = "c2_lut", lut_groups = 64 : i32,
      contract_sha256 = "236dd96dc3385c83755112e75453927a746d49a01b7407c4265f00e00ecab0a0", policy_sha256 = "25f828f236ae064668e90784bad6dccc2788bd334eab5c0bb98ea71aab397e4d", manifest_sha256 = "50dadea0cb83cc416a501a347ba3716de514e5aeedeec5a11fedeea9d1abd0c8", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"}
      : (tensor<64x128xi8>, tensor<128x4xi8>, tensor<128x64xi8>, tensor<4x128xi8>)
      -> (tensor<64x128xi8>, tensor<128x4xi8>)
    func.return %c2, %c2s : tensor<64x128xi8>, tensor<128x4xi8>
  }
}
