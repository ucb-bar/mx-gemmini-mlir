builtin.module attributes {
  mx.attention_program_schema = "mx_gemmini.nicolas_dense_attention_program.v1",
  mx.attention_phase = "setup",
  mx.source_sha256 = "537fcde270abb982b58e6777487d7ea3bc14e5e81da05fb48cb2ac31077a5cd8",
  mx.source_mlir_sha256 = "074066324401345c377e77140f084096fbabbedcae80321595320ccd2abaa5bf",
  mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d",
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "7ce8f6c363d1a1bc43ff6965df4ff5422b09c072f8dce232b7cfa573617fe11d",
  prov.quantization_manifest_sha256 = "972868f8ae048d2a4698929fee764ca3d3e2491a47e9e0e7e20ef325de4968a3"
} {
  func.func @mx_issue(%Q_SCALES: memref<256xi8>, %KT_SCALES: memref<1024xi8>, %V_SCALES: memref<1024xi8>, %Q_IN: memref<8192xi8>, %KT_IN: memref<32768xi8>, %V_IN: memref<32768xi8>, %scale_sink: memref<2048xi8>, %S_hw: memref<32768xi8>, %p_scales: memref<512xi8>, %O_hw: memref<16384xi8>, %P_hw: memref<32768xi8>, %p_codes_hw: memref<16384xi8>) {
    "mx_gemmini.load_scales"(%Q_SCALES) {site_id = "scales:q", payload_bytes = 256 : i32, scale_target = "activation", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d", source_sha256 = "537fcde270abb982b58e6777487d7ea3bc14e5e81da05fb48cb2ac31077a5cd8"} : (memref<256xi8>) -> ()
    "mx_gemmini.load_scales"(%KT_SCALES) {site_id = "scales:kt", payload_bytes = 1024 : i32, scale_target = "weight", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d", source_sha256 = "537fcde270abb982b58e6777487d7ea3bc14e5e81da05fb48cb2ac31077a5cd8"} : (memref<1024xi8>) -> ()
    "mx_gemmini.scale_dma"(%V_SCALES) {site_id = "scales:v", row_bytes = 128 : i32, rows = 8 : i32, pitch_bytes = 128 : i32, destination_byte = 4096 : i32, selector = 1 : i32, profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d", source_sha256 = "537fcde270abb982b58e6777487d7ea3bc14e5e81da05fb48cb2ac31077a5cd8"} : (memref<1024xi8>) -> ()
    func.return
  }
}
