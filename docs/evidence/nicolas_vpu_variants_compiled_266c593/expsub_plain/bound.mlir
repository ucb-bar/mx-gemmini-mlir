builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "64af01da90345633830cfc9ec1f87d9242f811ff9f1b819a46ed1be6c7cfc04f",
  prov.quantization_manifest_sha256 = "d673f841d7772c6472f2a0cbdb7c16b9cd5dfdb81fd4c3f25fae5de14539320f",
  mx.source_mlir_sha256 = "7413ea2c198b8c6ab87bdfb016c3d85bdf5080e0ed71293ea739bbbfc251bbba",
  mx.profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"
} {
  func.func @vpu_expsub_plain() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:expsub_plain", kind = "expsub",
      src1_row = 0 : i32, src2_row = 4096 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 1 : i32,
      broadcast = false, immediate_bf16 = 0 : i32,
      profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "64af01da90345633830cfc9ec1f87d9242f811ff9f1b819a46ed1be6c7cfc04f", manifest_sha256 = "d673f841d7772c6472f2a0cbdb7c16b9cd5dfdb81fd4c3f25fae5de14539320f"} : () -> ()
    func.return
  }
}
