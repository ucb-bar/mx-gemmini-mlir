builtin.module attributes {
  mx.attention_program_schema = "mx_gemmini.nicolas_flash_attention_program.v1",
  mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d", mx.source_sha256 = "004459aa0e95def61427d5a6e9d8f71be59fcbc281a93a35745c1f33591bf7c0",
  mx.source_mlir_sha256 = "cce8457e2fe88ac3f35dc4815258557a2eb5bdd52ce950d8f6c877fd4daefd84",
  mx.source_header_sha256 = "690bb592e6469b129f79552981828c57cc737928b631190483f0351dc9f7e9ee",
  mx.contract_sha256 = "cce8457e2fe88ac3f35dc4815258557a2eb5bdd52ce950d8f6c877fd4daefd84",
  mx.policy_sha256 = "5c605b64af07f08d0e16fb92a627e40bde32fdc92c7af4b143b624fb7985da3b",
  prov.quantization_manifest_sha256 = "ffc9a7babab354ef77162334ebf061ada7e32d661fed0fde61f86b0346be7faf"
} {
  func.func @mx_issue(%Q_SCALES_FOLDED: memref<128xi8>, %KT_SCALES: memref<256xi8>, %Q_IN: memref<4096xi8>, %KT_IN: memref<8192xi8>, %scale_sink: memref<2048xi8>, %S0_HW: memref<4096xi8>) {
    "mx_gemmini.scale_dma"(%Q_SCALES_FOLDED) {site_id = "scales:q_folded", row_bytes = 64 : i32, rows = 2 : i32, pitch_bytes = 64 : i32, destination_byte = 4096 : i32, selector = 0 : i32, profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d", source_sha256 = "004459aa0e95def61427d5a6e9d8f71be59fcbc281a93a35745c1f33591bf7c0"} : (memref<128xi8>) -> ()
    "mx_gemmini.wait"() {site_id = "scales:q_fence", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d", contract_sha256 = "cce8457e2fe88ac3f35dc4815258557a2eb5bdd52ce950d8f6c877fd4daefd84", policy_sha256 = "5c605b64af07f08d0e16fb92a627e40bde32fdc92c7af4b143b624fb7985da3b", manifest_sha256 = "ffc9a7babab354ef77162334ebf061ada7e32d661fed0fde61f86b0346be7faf"} : () -> ()
    "mx_gemmini.dma_matrix"(%Q_IN) {site_id = "operand:q", matrix_rows = 64 : i32, matrix_cols = 64 : i32, burst_cols = 64 : i32, spad_row = 0 : i32, profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d", source_sha256 = "004459aa0e95def61427d5a6e9d8f71be59fcbc281a93a35745c1f33591bf7c0"} : (memref<4096xi8>) -> ()
    "mx_gemmini.dma_matrix"(%KT_IN) {site_id = "operand:kt0", matrix_rows = 64 : i32, matrix_cols = 32 : i32, burst_cols = 32 : i32, spad_row = 4096 : i32, source_stride_bytes = 128 : i32, source_offset_bytes = 0 : i32, profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d", source_sha256 = "004459aa0e95def61427d5a6e9d8f71be59fcbc281a93a35745c1f33591bf7c0"} : (memref<8192xi8>) -> ()
    "mx_gemmini.flash_setup"() {site_id = "qk0:setup", store_stride_bytes = 64 : i32, profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d", source_sha256 = "004459aa0e95def61427d5a6e9d8f71be59fcbc281a93a35745c1f33591bf7c0"} : () -> ()
    "mx_gemmini.gated_scale_dma"(%KT_SCALES) {site_id = "qk0:weight_scales", row_bytes = 32 : i32, rows = 2 : i32, pitch_bytes = 128 : i32, destination_byte = 0 : i32, selector = 1 : i32, profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d", source_sha256 = "004459aa0e95def61427d5a6e9d8f71be59fcbc281a93a35745c1f33591bf7c0"} : (memref<256xi8>) -> ()
    "mx_gemmini.managed_matmul_spad"() {site_id = "qk0:matmul", m = 64 : i32, n = 32 : i32, k = 64 : i32, activation_row = 0 : i32, weight_end_row = 4224 : i32, output_row = 8192 : i32, weight_half = 0 : i32, activation_half = 1 : i32, scale_sink_buffer = "scale_sink", profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d", source_sha256 = "004459aa0e95def61427d5a6e9d8f71be59fcbc281a93a35745c1f33591bf7c0"} : () -> ()
    "mx_gemmini.spad_mvout_linear"(%S0_HW) {site_id = "qk0:readout", total_bytes = 4096 : i32, tile_cols = 16 : i32, spad_row = 8192 : i32, profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d", source_sha256 = "004459aa0e95def61427d5a6e9d8f71be59fcbc281a93a35745c1f33591bf7c0"} : (memref<4096xi8>) -> ()
    func.return
  }
}
