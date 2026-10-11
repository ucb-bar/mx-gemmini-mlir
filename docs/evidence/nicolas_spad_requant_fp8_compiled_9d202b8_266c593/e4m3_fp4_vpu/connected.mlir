module attributes {mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d",
  mx.contract_sha256 = "51af741fcfd14b031daa03b12ce6a3f03ee12f9d1f79bad5cabf3089087da508",
  mx.policy_sha256 = "ddc0b86bc0e50f6daf938624da237f84e9b705984a7d16c77dee1c41fe1b2b30",
  prov.quantization_manifest_sha256 = "72ed54217ac242f78b24399fb1620c59b5842110290d0ee768e1fbdc84011c3e"} {
  func.func @nicolas_spad_requant(%x: tensor<32x64xbf16>)
      -> (tensor<32x64xi8>, tensor<32x2xi8>,
          tensor<32x64xi8>, tensor<32x2xi8>) {
    %flat, %flats = "mx_gemmini.spad_requant"(%x) {
      source_row = 0 : i32, destination_row = 4096 : i32,
      m = 32 : i32, n = 64 : i32, output_format = "fp8_e4m3",
      tiled = false, resident = false,
      scale_dram_address = 0 : i64, scale_buffer = "scales_hw",
      site_id = "nicolas:spad_requant", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d", contract_sha256 = "51af741fcfd14b031daa03b12ce6a3f03ee12f9d1f79bad5cabf3089087da508", policy_sha256 = "ddc0b86bc0e50f6daf938624da237f84e9b705984a7d16c77dee1c41fe1b2b30", manifest_sha256 = "72ed54217ac242f78b24399fb1620c59b5842110290d0ee768e1fbdc84011c3e"} : (tensor<32x64xbf16>)
      -> (tensor<32x64xi8>, tensor<32x2xi8>)
    %tiled, %tileds = "mx_gemmini.spad_requant"(%x) {
      source_row = 0 : i32, destination_row = 8192 : i32,
      m = 32 : i32, n = 64 : i32, output_format = "fp8_e4m3",
      tiled = true, resident = false,
      scale_dram_address = 0 : i64, scale_buffer = "scales_hw2",
      site_id = "nicolas:spad_requant", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d", contract_sha256 = "51af741fcfd14b031daa03b12ce6a3f03ee12f9d1f79bad5cabf3089087da508", policy_sha256 = "ddc0b86bc0e50f6daf938624da237f84e9b705984a7d16c77dee1c41fe1b2b30", manifest_sha256 = "72ed54217ac242f78b24399fb1620c59b5842110290d0ee768e1fbdc84011c3e"} : (tensor<32x64xbf16>)
      -> (tensor<32x64xi8>, tensor<32x2xi8>)
    func.return %flat, %flats, %tiled, %tileds
      : tensor<32x64xi8>, tensor<32x2xi8>,
        tensor<32x64xi8>, tensor<32x2xi8>
  }
}
