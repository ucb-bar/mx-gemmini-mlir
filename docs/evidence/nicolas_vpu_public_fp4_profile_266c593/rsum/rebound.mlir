builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "c65b31296098c1e4846d8a36c21b031219409de59480e0bcfb0b486f26d4a414",
  prov.quantization_manifest_sha256 = "d69ede7157a8c391dcd4f4737ce8ba469abb14fb416d45b74ffd50b39ad8adb6",
  mx.source_mlir_sha256 = "d8982d684e422c5cb0231b43d83187ecaf4171a9d44f5359249e9f4fcab9bfa6",
  mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"
} {
  func.func @vpu_rsum() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:rsum", kind = "rsum",
      src1_row = 0 : i32, src2_row = 0 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 4 : i32,
      broadcast = false, immediate_bf16 = 0 : i32,
      profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "c65b31296098c1e4846d8a36c21b031219409de59480e0bcfb0b486f26d4a414", manifest_sha256 = "d69ede7157a8c391dcd4f4737ce8ba469abb14fb416d45b74ffd50b39ad8adb6"} : () -> ()
    func.return
  }
}
