builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "0ffb6b16a2c30364abb67476ca5e6f3a35402b28b6f38f98716eba40f04ff7bf",
  prov.quantization_manifest_sha256 = "c7b6cf097a65d6510fa825dde6ab0db1c5cd1d8f8ee64f8f03affeacd837aa48",
  mx.source_mlir_sha256 = "f7239e65be3154b17b8a08368b008f63d9695616ea76a622585a550bd58c1a0a",
  mx.profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"
} {
  func.func @vpu_mul_bcast() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:mul_bcast", kind = "mul",
      src1_row = 0 : i32, src2_row = 4096 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 4 : i32,
      broadcast = true, immediate_bf16 = 0 : i32,
      profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "0ffb6b16a2c30364abb67476ca5e6f3a35402b28b6f38f98716eba40f04ff7bf", manifest_sha256 = "c7b6cf097a65d6510fa825dde6ab0db1c5cd1d8f8ee64f8f03affeacd837aa48"} : () -> ()
    func.return
  }
}
