builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "a537cb18674df637883feb87e31c15f6ac675ad7485ab1c45843ad4071c7d6d0",
  prov.quantization_manifest_sha256 = "78adc599b8c0046084cc652393414576b1421bce0e5ab8fc0f8602f6f8eae219",
  mx.source_mlir_sha256 = "639b4257f36d66f60390ea2a1ae0831d724d0308bae3f23bdfc91bddf5857257",
  mx.profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"
} {
  func.func @vpu_expsum_bcast() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:expsum_bcast", kind = "expsum",
      src1_row = 0 : i32, src2_row = 4096 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 4 : i32,
      broadcast = true, immediate_bf16 = 0 : i32, second_dst_row = 12288 : i32,
      profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "a537cb18674df637883feb87e31c15f6ac675ad7485ab1c45843ad4071c7d6d0", manifest_sha256 = "78adc599b8c0046084cc652393414576b1421bce0e5ab8fc0f8602f6f8eae219"} : () -> ()
    func.return
  }
}
