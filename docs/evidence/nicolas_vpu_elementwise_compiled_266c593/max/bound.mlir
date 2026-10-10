builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "d35df8efd3174c82a4ae9d8e377f247d99e263296cd986bd2e58f54fc3edf4f7",
  prov.quantization_manifest_sha256 = "bcf80e58beeae3ea5ebd02515ad9429011f9eaaf84283fb68b6a2d89287c2870",
  mx.source_mlir_sha256 = "ba142e9804a8d2bccd352a3f2b074ccd745e6934e99834b42754037691b68d46",
  mx.profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"
} {
  func.func @vpu_max() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:max", kind = "max",
      src1_row = 0 : i32, src2_row = 4096 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 1 : i32,
      broadcast = false, immediate_bf16 = 0 : i32,
      profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "d35df8efd3174c82a4ae9d8e377f247d99e263296cd986bd2e58f54fc3edf4f7", manifest_sha256 = "bcf80e58beeae3ea5ebd02515ad9429011f9eaaf84283fb68b6a2d89287c2870"} : () -> ()
    func.return
  }
}
