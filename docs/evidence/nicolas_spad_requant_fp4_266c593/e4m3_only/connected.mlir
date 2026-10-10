module attributes {mx.profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487",
  mx.contract_sha256 = "812842fd751d7fa0d8fb98a150df29424bdaba65613531335001f9f2260eb884",
  mx.policy_sha256 = "3eee3fce11de8c7e9fe7f556a4dec8ab3df25f436a8bb29b520b6fa3d406a494",
  prov.quantization_manifest_sha256 = "72ed54217ac242f78b24399fb1620c59b5842110290d0ee768e1fbdc84011c3e"} {
  func.func @nicolas_spad_requant_fp4(%x: tensor<64x128xbf16>)
      -> (tensor<32x128xi8>, tensor<64x4xi8>,
          tensor<32x128xi8>, tensor<64x4xi8>) {
    %flat, %flats = "mx_gemmini.spad_requant"(%x) {
      source_row = 0 : i32, destination_row = 4096 : i32,
      m = 64 : i32, n = 128 : i32, output_format = "fp4_e2m1",
      tiled = false, resident = false,
      scale_dram_address = 0 : i64, scale_buffer = "scales_hw",
      site_id = "nicolas:spad_requant_fp4", profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "812842fd751d7fa0d8fb98a150df29424bdaba65613531335001f9f2260eb884", policy_sha256 = "3eee3fce11de8c7e9fe7f556a4dec8ab3df25f436a8bb29b520b6fa3d406a494", manifest_sha256 = "72ed54217ac242f78b24399fb1620c59b5842110290d0ee768e1fbdc84011c3e"} : (tensor<64x128xbf16>)
      -> (tensor<32x128xi8>, tensor<64x4xi8>)
    %tiled, %tileds = "mx_gemmini.spad_requant"(%x) {
      source_row = 0 : i32, destination_row = 8192 : i32,
      m = 64 : i32, n = 128 : i32, output_format = "fp4_e2m1",
      tiled = true, resident = false,
      scale_dram_address = 0 : i64, scale_buffer = "scales_hw2",
      site_id = "nicolas:spad_requant_fp4", profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "812842fd751d7fa0d8fb98a150df29424bdaba65613531335001f9f2260eb884", policy_sha256 = "3eee3fce11de8c7e9fe7f556a4dec8ab3df25f436a8bb29b520b6fa3d406a494", manifest_sha256 = "72ed54217ac242f78b24399fb1620c59b5842110290d0ee768e1fbdc84011c3e"} : (tensor<64x128xbf16>)
      -> (tensor<32x128xi8>, tensor<64x4xi8>)
    func.return %flat, %flats, %tiled, %tileds
      : tensor<32x128xi8>, tensor<64x4xi8>,
        tensor<32x128xi8>, tensor<64x4xi8>
  }
}
