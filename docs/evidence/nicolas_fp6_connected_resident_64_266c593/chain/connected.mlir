module attributes {mx.profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c",
  mx.contract_sha256 = "fc9cb61cd311d36183027158970e1c247840704dc467e9c13522ef3883d869be", mx.policy_sha256 = "d0cd87951d071d21053365f21df6a25559d4fc9d94fab0edb066eb83b5b277de",
  prov.quantization_manifest_sha256 = "b06444d3e4c0e47499d2638522ea209dd85aa13446c254fa81cbce69954dc621",
  mx.frontend_mlir_sha256 = "4a370cba79f947edae338a8a191afb4612f017a9d57446b9fbc0b0ba8b75dec8",
  mx.frontend_manifest_sha256 = "2d3bdaab54f3751a5b403d81cee0b9439ac5fcf8878632219da5f06650cda1dc",
  mx.runtime_resources_sha256 = "47b3fa37fdd38ed41daa6060eb9b2459f481440edd606812649acb54695c587d"} {
  func.func @nicolas_plain_fp6_chain(
      %a1: tensor<32x64xi8>, %a1s: tensor<2x64xi8>,
      %b1: tensor<64x32xi8>, %b1s: tensor<2x64xi8>,
      %b2: tensor<64x32xi8>, %b2s: tensor<2x64xi8>,
      %a1lut: tensor<32x3xi32>, %b1lut: tensor<32x3xi32>,
      %b2lut: tensor<32x3xi32>, %c1lut: tensor<32x3xi32>,
      %c2lut: tensor<32x3xi32>)
      -> (tensor<32x64xi8>, tensor<64x2xi8>) {
    "mx_gemmini.runtime_lut"(%b1lut) {site_id = "functional:matmul", lut_target = "weight", runtime_buffer = "b1_lut", groups = 32 : i32, entry_bits = 6 : i32, contract_sha256 = "fc9cb61cd311d36183027158970e1c247840704dc467e9c13522ef3883d869be", policy_sha256 = "d0cd87951d071d21053365f21df6a25559d4fc9d94fab0edb066eb83b5b277de", manifest_sha256 = "b06444d3e4c0e47499d2638522ea209dd85aa13446c254fa81cbce69954dc621", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"} : (tensor<32x3xi32>) -> ()
    "mx_gemmini.runtime_lut"(%a1lut) {site_id = "functional:matmul", lut_target = "activation", runtime_buffer = "a1_lut", groups = 32 : i32, entry_bits = 6 : i32, contract_sha256 = "fc9cb61cd311d36183027158970e1c247840704dc467e9c13522ef3883d869be", policy_sha256 = "d0cd87951d071d21053365f21df6a25559d4fc9d94fab0edb066eb83b5b277de", manifest_sha256 = "b06444d3e4c0e47499d2638522ea209dd85aa13446c254fa81cbce69954dc621", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"} : (tensor<32x3xi32>) -> ()
    "mx_gemmini.runtime_lut"(%c1lut) {site_id = "functional:matmul", lut_target = "output", runtime_buffer = "c1_lut", groups = 32 : i32, entry_bits = 6 : i32, contract_sha256 = "fc9cb61cd311d36183027158970e1c247840704dc467e9c13522ef3883d869be", policy_sha256 = "d0cd87951d071d21053365f21df6a25559d4fc9d94fab0edb066eb83b5b277de", manifest_sha256 = "b06444d3e4c0e47499d2638522ea209dd85aa13446c254fa81cbce69954dc621", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"} : (tensor<32x3xi32>) -> ()
    %acc = "mx_gemmini.contract"(%a1, %a1s, %b1, %b1s) {
      site_id = "functional:matmul", activation_format = "fp6_e3m2",
      weight_format = "fp6_e3m2", activation_projection = "lut",
      weight_projection = "lut", pe_mode = 4 : i32, contract_sha256 = "fc9cb61cd311d36183027158970e1c247840704dc467e9c13522ef3883d869be", policy_sha256 = "d0cd87951d071d21053365f21df6a25559d4fc9d94fab0edb066eb83b5b277de", manifest_sha256 = "b06444d3e4c0e47499d2638522ea209dd85aa13446c254fa81cbce69954dc621", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"}
      : (tensor<32x64xi8>, tensor<2x64xi8>, tensor<64x32xi8>, tensor<2x64xi8>)
      -> tensor<64x64xbf16>
    %c1, %c1s = "mx_gemmini.readout_quantized"(%acc) {
      site_id = "functional:matmul", output_format = "fp6_e3m2", contract_sha256 = "fc9cb61cd311d36183027158970e1c247840704dc467e9c13522ef3883d869be", policy_sha256 = "d0cd87951d071d21053365f21df6a25559d4fc9d94fab0edb066eb83b5b277de", manifest_sha256 = "b06444d3e4c0e47499d2638522ea209dd85aa13446c254fa81cbce69954dc621", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"}
      : (tensor<64x64xbf16>) -> (tensor<32x64xi8>, tensor<64x2xi8>)
    "mx_gemmini.runtime_lut"(%b2lut) {site_id = "functional:matmul_1", lut_target = "weight", runtime_buffer = "b2_lut", groups = 32 : i32, entry_bits = 6 : i32, contract_sha256 = "fc9cb61cd311d36183027158970e1c247840704dc467e9c13522ef3883d869be", policy_sha256 = "d0cd87951d071d21053365f21df6a25559d4fc9d94fab0edb066eb83b5b277de", manifest_sha256 = "b06444d3e4c0e47499d2638522ea209dd85aa13446c254fa81cbce69954dc621", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"} : (tensor<32x3xi32>) -> ()
    "mx_gemmini.runtime_lut"(%c1lut) {site_id = "functional:matmul_1", lut_target = "activation", runtime_buffer = "c1_lut", groups = 32 : i32, entry_bits = 6 : i32, contract_sha256 = "fc9cb61cd311d36183027158970e1c247840704dc467e9c13522ef3883d869be", policy_sha256 = "d0cd87951d071d21053365f21df6a25559d4fc9d94fab0edb066eb83b5b277de", manifest_sha256 = "b06444d3e4c0e47499d2638522ea209dd85aa13446c254fa81cbce69954dc621", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"} : (tensor<32x3xi32>) -> ()
    "mx_gemmini.runtime_lut"(%c2lut) {site_id = "functional:matmul_1", lut_target = "output", runtime_buffer = "c2_lut", groups = 32 : i32, entry_bits = 6 : i32, contract_sha256 = "fc9cb61cd311d36183027158970e1c247840704dc467e9c13522ef3883d869be", policy_sha256 = "d0cd87951d071d21053365f21df6a25559d4fc9d94fab0edb066eb83b5b277de", manifest_sha256 = "b06444d3e4c0e47499d2638522ea209dd85aa13446c254fa81cbce69954dc621", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"} : (tensor<32x3xi32>) -> ()
    %c2, %c2s = "mx_gemmini.resident_contract"(%c1, %c1s, %b2, %b2s) {
      site_id = "functional:matmul_1", activation_row = 2048 : i32,
      weight_row = 16256 : i32, output_row = 4096 : i32,
      m = 64 : i32, n = 64 : i32, k = 64 : i32,
      activation_format = "fp6_e3m2", weight_format = "fp6_e3m2",
      output_format = "fp6_e3m2", weight_buffer = "b2_weight",
      weight_scales_buffer = "b2_scales", output_scales_buffer = "c2_scales",
      weight_lut_buffer = "b2_lut", activation_lut_buffer = "c1_lut",
      output_lut_buffer = "c2_lut", lut_groups = 32 : i32,
      contract_sha256 = "fc9cb61cd311d36183027158970e1c247840704dc467e9c13522ef3883d869be", policy_sha256 = "d0cd87951d071d21053365f21df6a25559d4fc9d94fab0edb066eb83b5b277de", manifest_sha256 = "b06444d3e4c0e47499d2638522ea209dd85aa13446c254fa81cbce69954dc621", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c"}
      : (tensor<32x64xi8>, tensor<64x2xi8>, tensor<64x32xi8>, tensor<2x64xi8>)
      -> (tensor<32x64xi8>, tensor<64x2xi8>)
    func.return %c2, %c2s : tensor<32x64xi8>, tensor<64x2xi8>
  }
}
