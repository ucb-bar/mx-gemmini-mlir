builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "c9a6106dd4525e6c2fd997ab8df11cb1667313560dc818a26fbc9e1bfaaba795",
  prov.quantization_manifest_sha256 = "4190ef75d0f9332c6d45c48c063148151301188a42a01b71fc0bdfd8327b1a6f",
  mx.source_mlir_sha256 = "b0023616f7cdd46793c4879e107984d20c27288cf4f79f1ea7781cac102a1a1d",
  mx.profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"
} {
  func.func @vpu_war() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:war:0", kind = "add",
      src1_row = 0 : i32, src2_row = 4096 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 1 : i32, broadcast = false,
      immediate_bf16 = 0 : i32, profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "c9a6106dd4525e6c2fd997ab8df11cb1667313560dc818a26fbc9e1bfaaba795", manifest_sha256 = "4190ef75d0f9332c6d45c48c063148151301188a42a01b71fc0bdfd8327b1a6f"} : () -> ()
    func.return
  }
}
