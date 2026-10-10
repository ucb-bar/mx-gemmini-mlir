builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "53a79f7012bfe4a9d159928d8b4702cf7252e9d3d3d624607853c6686a24e1f1",
  prov.quantization_manifest_sha256 = "0c6aed56fa30042070f7459fb155e69b4223d5461435b6f0ed0ea33e290ac0da",
  mx.source_mlir_sha256 = "360405273be245c4f62743de43bb4acafecbc8ebe73e983e1bd0bafd495a41e1",
  mx.profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"
} {
  func.func @vpu_exp() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:exp", kind = "exp",
      src1_row = 0 : i32, src2_row = 0 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 1 : i32,
      broadcast = false, immediate_bf16 = 0 : i32,
      profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "53a79f7012bfe4a9d159928d8b4702cf7252e9d3d3d624607853c6686a24e1f1", manifest_sha256 = "0c6aed56fa30042070f7459fb155e69b4223d5461435b6f0ed0ea33e290ac0da"} : () -> ()
    func.return
  }
}
