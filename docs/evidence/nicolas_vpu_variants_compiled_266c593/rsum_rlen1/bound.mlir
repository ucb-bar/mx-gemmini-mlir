builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "b4ed686f0575a8b18385f65e397ef07853eb3ed8f98c6e2ca3dff55330dbdde1",
  prov.quantization_manifest_sha256 = "40a2270f48b7139c06e908d7437dce5740786a8d591b942e11dd0ec893e83b06",
  mx.source_mlir_sha256 = "d486efb94b6910afed666ff9e38eedaae9cb4b91ab8219d3e9d92f4b3b71e96c",
  mx.profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"
} {
  func.func @vpu_rsum_rlen1() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:rsum_rlen1", kind = "rsum",
      src1_row = 0 : i32, src2_row = 0 : i32,
      dst_row = 12288 : i32, rows = 64 : i32,
      reduction_length = 1 : i32,
      broadcast = false, immediate_bf16 = 0 : i32,
      profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "b4ed686f0575a8b18385f65e397ef07853eb3ed8f98c6e2ca3dff55330dbdde1", manifest_sha256 = "40a2270f48b7139c06e908d7437dce5740786a8d591b942e11dd0ec893e83b06"} : () -> ()
    func.return
  }
}
