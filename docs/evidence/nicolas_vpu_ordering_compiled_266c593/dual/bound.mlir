builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "f4a896f367a305eed03b16296a98dcb6508f05da5f3111b748b43251878047af",
  prov.quantization_manifest_sha256 = "980978562ac35efa1d8726ac3bea9575ae7d36cb570f12439572e2eda3eddeaf",
  mx.source_mlir_sha256 = "6a6b9261669b26863b93bc1331ed2ad3a04e32defaf28f66a319af165f3c4eb8",
  mx.profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"
} {
  func.func @vpu_dual() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:dual:0", kind = "adds",
      src1_row = 0 : i32, src2_row = 0 : i32,
      dst_row = 1024 : i32, rows = 64 : i32,
      reduction_length = 1 : i32, broadcast = false,
      immediate_bf16 = 16448 : i32, profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "f4a896f367a305eed03b16296a98dcb6508f05da5f3111b748b43251878047af", manifest_sha256 = "980978562ac35efa1d8726ac3bea9575ae7d36cb570f12439572e2eda3eddeaf"} : () -> ()
    "mx_gemmini.vpu_execute"() {site_id = "vpu:dual:1", kind = "exp",
      src1_row = 13312 : i32, src2_row = 0 : i32,
      dst_row = 12288 : i32, rows = 64 : i32,
      reduction_length = 1 : i32, broadcast = false,
      immediate_bf16 = 0 : i32, profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "f4a896f367a305eed03b16296a98dcb6508f05da5f3111b748b43251878047af", manifest_sha256 = "980978562ac35efa1d8726ac3bea9575ae7d36cb570f12439572e2eda3eddeaf"} : () -> ()
    "mx_gemmini.vpu_execute"() {site_id = "vpu:dual:2", kind = "mul",
      src1_row = 1024 : i32, src2_row = 4096 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 1 : i32, broadcast = false,
      immediate_bf16 = 0 : i32, profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "f4a896f367a305eed03b16296a98dcb6508f05da5f3111b748b43251878047af", manifest_sha256 = "980978562ac35efa1d8726ac3bea9575ae7d36cb570f12439572e2eda3eddeaf"} : () -> ()
    func.return
  }
}
