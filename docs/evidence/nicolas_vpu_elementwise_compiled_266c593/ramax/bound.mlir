builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "408fed154b35193dd05412ec434279d48f209679e95307862bfb158dbbd63109",
  prov.quantization_manifest_sha256 = "da4bf0c41a003e99bfd00ae49a3efbef08a3f852c0c0056b8abf93143846655e",
  mx.source_mlir_sha256 = "925096f1dd1c7698197ca402faa3ed1a84ff0b05ba8e8c31be9a2e4b7249da73",
  mx.profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"
} {
  func.func @vpu_ramax() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:ramax", kind = "ramax",
      src1_row = 0 : i32, src2_row = 0 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 4 : i32,
      broadcast = false, immediate_bf16 = 0 : i32,
      profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "408fed154b35193dd05412ec434279d48f209679e95307862bfb158dbbd63109", manifest_sha256 = "da4bf0c41a003e99bfd00ae49a3efbef08a3f852c0c0056b8abf93143846655e"} : () -> ()
    func.return
  }
}
