builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "e6d73eddb08617c719166aa17d64b934fb62c2b04c40418c9a350478c0652966",
  prov.quantization_manifest_sha256 = "de5b63915c1c00a6d45e9f4b35ff4a683e9ae70c3995a683678e8f4fb421e86b",
  mx.source_mlir_sha256 = "3cfae7502b0a92d65bf296837533bd700546054206d574984e8a2f321dd87a82",
  mx.profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"
} {
  func.func @vpu_muls() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:muls", kind = "muls",
      src1_row = 0 : i32, src2_row = 0 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 1 : i32,
      broadcast = false, immediate_bf16 = 16128 : i32,
      profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "e6d73eddb08617c719166aa17d64b934fb62c2b04c40418c9a350478c0652966", manifest_sha256 = "de5b63915c1c00a6d45e9f4b35ff4a683e9ae70c3995a683678e8f4fb421e86b"} : () -> ()
    func.return
  }
}
