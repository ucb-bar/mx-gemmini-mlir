builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "7ca8be6703ba9fe5e7d53585a1c0f73f75c4aac3d0d11bda61f9a4d5eb726637",
  prov.quantization_manifest_sha256 = "c565d3189032abe2755ed426f29efa86fb9fef189cde35f13e72747e4fef963b",
  mx.source_mlir_sha256 = "4d7458acc5019efc9104bfca0c967639cd8368659a09978a0e56a8009766fdb7",
  mx.profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"
} {
  func.func @vpu_adds() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:adds", kind = "adds",
      src1_row = 0 : i32, src2_row = 0 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 1 : i32,
      broadcast = false, immediate_bf16 = 16448 : i32,
      profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "7ca8be6703ba9fe5e7d53585a1c0f73f75c4aac3d0d11bda61f9a4d5eb726637", manifest_sha256 = "c565d3189032abe2755ed426f29efa86fb9fef189cde35f13e72747e4fef963b"} : () -> ()
    func.return
  }
}
