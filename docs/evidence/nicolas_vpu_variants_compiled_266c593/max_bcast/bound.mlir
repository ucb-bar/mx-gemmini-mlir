builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "10246866fbb0b40db9910238b18a67534f3dcf92d616460e5ba6ed14d4ac9802",
  prov.quantization_manifest_sha256 = "29d17943af224289222131de6f38c244c09795cd58df5eb4e425a817bb5ec53d",
  mx.source_mlir_sha256 = "70d50f855e473830c73e2cc453cdd017fecd56536b644bb2313a7f128a2b75ef",
  mx.profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"
} {
  func.func @vpu_max_bcast() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:max_bcast", kind = "max",
      src1_row = 0 : i32, src2_row = 4096 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 4 : i32,
      broadcast = true, immediate_bf16 = 0 : i32,
      profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "10246866fbb0b40db9910238b18a67534f3dcf92d616460e5ba6ed14d4ac9802", manifest_sha256 = "29d17943af224289222131de6f38c244c09795cd58df5eb4e425a817bb5ec53d"} : () -> ()
    func.return
  }
}
