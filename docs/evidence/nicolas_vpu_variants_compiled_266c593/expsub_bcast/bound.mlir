builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "0c634145f9525753a9c92c41fa30ebbd898958e35876a69bd1ecdd01cb87c48e",
  prov.quantization_manifest_sha256 = "a5a0daf5f543ff90b6dc46cccb01b8a342f9ec31eca0780fe33f4abe1b49b2b3",
  mx.source_mlir_sha256 = "f6b68f1716e66d3f3f20db2f43bc18f5a40206596494a749ba38e6947e6133ce",
  mx.profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"
} {
  func.func @vpu_expsub_bcast() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:expsub_bcast", kind = "expsub",
      src1_row = 0 : i32, src2_row = 4096 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 4 : i32,
      broadcast = true, immediate_bf16 = 0 : i32,
      profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "0c634145f9525753a9c92c41fa30ebbd898958e35876a69bd1ecdd01cb87c48e", manifest_sha256 = "a5a0daf5f543ff90b6dc46cccb01b8a342f9ec31eca0780fe33f4abe1b49b2b3"} : () -> ()
    func.return
  }
}
