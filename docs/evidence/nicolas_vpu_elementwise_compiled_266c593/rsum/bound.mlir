builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "c65b31296098c1e4846d8a36c21b031219409de59480e0bcfb0b486f26d4a414",
  prov.quantization_manifest_sha256 = "691154c837afa2bf07d0153eea8988b19d01a36f57346149c06da0d977d0d59c",
  mx.source_mlir_sha256 = "d8982d684e422c5cb0231b43d83187ecaf4171a9d44f5359249e9f4fcab9bfa6",
  mx.profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"
} {
  func.func @vpu_rsum() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:rsum", kind = "rsum",
      src1_row = 0 : i32, src2_row = 0 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 4 : i32,
      broadcast = false, immediate_bf16 = 0 : i32,
      profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "c65b31296098c1e4846d8a36c21b031219409de59480e0bcfb0b486f26d4a414", manifest_sha256 = "691154c837afa2bf07d0153eea8988b19d01a36f57346149c06da0d977d0d59c"} : () -> ()
    func.return
  }
}
