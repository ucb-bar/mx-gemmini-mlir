module attributes {mx.profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c", mx.source_sha256 = "d23c749184ede0db1cb0b4ee67f3e29d8d80ddee4ddf589dd949fc7c40a3b922"} {
  func.func @mx_issue_setup() {
    "mx_gemmini.memory_setup"() {site_id = "setup", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c", source_sha256 = "d23c749184ede0db1cb0b4ee67f3e29d8d80ddee4ddf589dd949fc7c40a3b922"} : () -> ()
    return
  }
  func.func @mx_issue_a64(%arg0: memref<16384xi8>) {
    "mx_gemmini.dma_matrix"(%arg0) {site_id = "a64", matrix_rows = 128 : i32, matrix_cols = 128 : i32, burst_cols = 64 : i32, spad_row = 0 : i32, profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c", source_sha256 = "d23c749184ede0db1cb0b4ee67f3e29d8d80ddee4ddf589dd949fc7c40a3b922"} : (memref<16384xi8>) -> ()
    return
  }
  func.func @mx_issue_b16(%arg0: memref<16384xi8>) {
    "mx_gemmini.dma_matrix"(%arg0) {site_id = "b16", matrix_rows = 128 : i32, matrix_cols = 128 : i32, burst_cols = 16 : i32, spad_row = 1024 : i32, profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c", source_sha256 = "d23c749184ede0db1cb0b4ee67f3e29d8d80ddee4ddf589dd949fc7c40a3b922"} : (memref<16384xi8>) -> ()
    return
  }
  func.func @mx_issue_scale(%arg0: memref<512xi8>) {
    "mx_gemmini.load_scales"(%arg0) {site_id = "scale", payload_bytes = 512 : i32, scale_target = "activation", profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c", source_sha256 = "d23c749184ede0db1cb0b4ee67f3e29d8d80ddee4ddf589dd949fc7c40a3b922"} : (memref<512xi8>) -> ()
    return
  }
  func.func @mx_issue_mvout(%arg0: memref<16384xi8>) {
    "mx_gemmini.spad_mvout_linear"(%arg0) {site_id = "mvout", total_bytes = 16384 : i32, tile_cols = 16 : i32, spad_row = 0 : i32, profile_sha256 = "bf54ac354c298cc71db23598f569fda73b9b2e46df5f7683bdac17a94fac0e3c", source_sha256 = "d23c749184ede0db1cb0b4ee67f3e29d8d80ddee4ddf589dd949fc7c40a3b922"} : (memref<16384xi8>) -> ()
    return
  }
}
