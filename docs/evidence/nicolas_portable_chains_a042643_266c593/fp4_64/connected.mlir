module attributes {mx.profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c",
  mx.contract_sha256 = "83a9ba1742da993d263bfcae1f9da9d1640e8ee1bfb556dfa05d30e612324257", mx.policy_sha256 = "b6043e4a008d9955d717add33b28b87e8c9db39896f69fce5e9263584a29f472",
  prov.quantization_manifest_sha256 = "2df2bf78912754631985f8d8803251125dc52a2b7e772c41eab65c24b79b278e",
  mx.frontend_mlir_sha256 = "42d31a797eb3de71db50b58d86327f229e62fd9bc0df524ae5fb8d4e7783592a",
  mx.frontend_manifest_sha256 = "bc94516fb3b64e6cc60997c69b3de7bc8eba7ebd3773212b8079558b48db9f6f",
  mx.runtime_resources_sha256 = "336d4dc5fabe74fe60c32c93db43444b36f2b22c02abba8e23144af85d15f9f0"} {
  func.func @nicolas_plain_fp4_chain(
      %a1: tensor<32x64xi8>, %a1s: tensor<2x64xi8>,
      %b1: tensor<64x32xi8>, %b1s: tensor<2x64xi8>,
      %b2: tensor<64x32xi8>, %b2s: tensor<2x64xi8>)
      -> (tensor<32x64xi8>, tensor<64x2xi8>) {
    %acc = "mx_gemmini.contract"(%a1, %a1s, %b1, %b1s) {
      site_id = "functional:matmul", activation_format = "fp4_e2m1",
      weight_format = "fp4_e2m1", activation_projection = "direct",
      weight_projection = "direct", pe_mode = 0 : i32, contract_sha256 = "83a9ba1742da993d263bfcae1f9da9d1640e8ee1bfb556dfa05d30e612324257", policy_sha256 = "b6043e4a008d9955d717add33b28b87e8c9db39896f69fce5e9263584a29f472", manifest_sha256 = "2df2bf78912754631985f8d8803251125dc52a2b7e772c41eab65c24b79b278e", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"}
      : (tensor<32x64xi8>, tensor<2x64xi8>, tensor<64x32xi8>, tensor<2x64xi8>)
      -> tensor<64x64xbf16>
    %c1, %c1s = "mx_gemmini.readout_quantized"(%acc) {
      site_id = "functional:matmul", output_format = "fp4_e2m1", contract_sha256 = "83a9ba1742da993d263bfcae1f9da9d1640e8ee1bfb556dfa05d30e612324257", policy_sha256 = "b6043e4a008d9955d717add33b28b87e8c9db39896f69fce5e9263584a29f472", manifest_sha256 = "2df2bf78912754631985f8d8803251125dc52a2b7e772c41eab65c24b79b278e", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"}
      : (tensor<64x64xbf16>) -> (tensor<32x64xi8>, tensor<64x2xi8>)
    %c2, %c2s = "mx_gemmini.resident_contract"(%c1, %c1s, %b2, %b2s) {
      site_id = "functional:matmul_1", activation_row = 2048 : i32,
      weight_row = 16256 : i32, output_row = 4096 : i32,
      m = 64 : i32, n = 64 : i32, k = 64 : i32,
      activation_format = "fp4_e2m1", weight_format = "fp4_e2m1",
      output_format = "fp4_e2m1", weight_buffer = "b2_weight",
      weight_scales_buffer = "b2_scales", output_scales_buffer = "c2_scales",
      contract_sha256 = "83a9ba1742da993d263bfcae1f9da9d1640e8ee1bfb556dfa05d30e612324257", policy_sha256 = "b6043e4a008d9955d717add33b28b87e8c9db39896f69fce5e9263584a29f472", manifest_sha256 = "2df2bf78912754631985f8d8803251125dc52a2b7e772c41eab65c24b79b278e", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"}
      : (tensor<32x64xi8>, tensor<64x2xi8>, tensor<64x32xi8>, tensor<2x64xi8>)
      -> (tensor<32x64xi8>, tensor<64x2xi8>)
    func.return %c2, %c2s : tensor<32x64xi8>, tensor<64x2xi8>
  }
}
