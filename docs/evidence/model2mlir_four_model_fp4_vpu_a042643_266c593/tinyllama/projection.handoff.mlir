module attributes {mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d",
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", mx.policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3",
  prov.quantization_manifest_sha256 = "894e8f83b8c41ace2fe5d67e284d15e11579b247d08b62a438fa403ad5d55679",
  mx.frontend_mlir_sha256 = "6959e71a0b275d780bc16a523f8446ba6d7e37c9c9c4434bc3ed8ef45d6c359a"} {
  func.func @model_projection_slice(%a: tensor<32x2048xi8>,
      %as: tensor<64x32xi8>, %b: tensor<2048x2048xi8>,
      %bs: tensor<64x2048xi8>) -> tensor<32x2048xbf16> {
    %acc = "mx_gemmini.contract"(%a, %as, %b, %bs) {
      site_id = "functional:matmul", activation_format = "fp4_e2m1",
      weight_format = "fp4_e2m1", activation_projection = "direct",
      weight_projection = "direct", pe_mode = 0 : i32, contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3", manifest_sha256 = "894e8f83b8c41ace2fe5d67e284d15e11579b247d08b62a438fa403ad5d55679", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<32x2048xi8>, tensor<64x32xi8>, tensor<2048x2048xi8>,
         tensor<64x2048xi8>) -> tensor<32x2048xbf16>
    %out = "mx_gemmini.readout_bf16"(%acc) {site_id = "functional:matmul", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "dc71c668056aac512c42f80f8da1bd92e06c4e3e6460ed1d4fc540e6dcc7f7a3", manifest_sha256 = "894e8f83b8c41ace2fe5d67e284d15e11579b247d08b62a438fa403ad5d55679", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"}
      : (tensor<32x2048xbf16>) -> tensor<32x2048xbf16>
    func.return %out : tensor<32x2048xbf16>
  }
}
