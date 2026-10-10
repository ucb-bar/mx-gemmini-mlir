builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "46f4ea3bd13d70f683cdcbc64695f82291d0ec640437f15963a142e91e848c85",
  prov.quantization_manifest_sha256 = "679ac3c0249533b23637c640f3b6553a5be4c53cbfaef97d038f1deb184d622e",
  mx.source_mlir_sha256 = "ba142e9804a8d2bccd352a3f2b074ccd745e6934e99834b42754037691b68d46",
  mx.profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"
} {
  func.func @vpu_max_same_bank() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:max_same_bank", kind = "max",
      src1_row = 0 : i32, src2_row = 2048 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 1 : i32,
      broadcast = false, immediate_bf16 = 0 : i32,
      profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "46f4ea3bd13d70f683cdcbc64695f82291d0ec640437f15963a142e91e848c85", manifest_sha256 = "679ac3c0249533b23637c640f3b6553a5be4c53cbfaef97d038f1deb184d622e"} : () -> ()
    func.return
  }
}
