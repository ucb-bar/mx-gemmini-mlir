builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "1cbb5de2c782e66b5524e7280292e9b93afbe01ba578ad46bf893417b0809c3b",
  prov.quantization_manifest_sha256 = "de2a3e37f4ca74c059946ee09997ed2da1844622f350ccc088fc464dcdd3eab9",
  mx.source_mlir_sha256 = "f6b68f1716e66d3f3f20db2f43bc18f5a40206596494a749ba38e6947e6133ce",
  mx.profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"
} {
  func.func @vpu_expsub_same_bank() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:expsub_same_bank", kind = "expsub",
      src1_row = 0 : i32, src2_row = 2048 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 4 : i32,
      broadcast = true, immediate_bf16 = 0 : i32,
      profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "1cbb5de2c782e66b5524e7280292e9b93afbe01ba578ad46bf893417b0809c3b", manifest_sha256 = "de2a3e37f4ca74c059946ee09997ed2da1844622f350ccc088fc464dcdd3eab9"} : () -> ()
    func.return
  }
}
