builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "0419016a7c3d1fa1fc5282a76832007393b488425bf426c6e14bff868721a7c9",
  prov.quantization_manifest_sha256 = "d072df623c02c1b748ed32ba40a6a9e0b6412e7e008b19f21aa72da418175a68",
  mx.source_mlir_sha256 = "7413ea2c198b8c6ab87bdfb016c3d85bdf5080e0ed71293ea739bbbfc251bbba",
  mx.profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"
} {
  func.func @fused_vpu() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:expsub", kind = "expsub",
      src1_row = 0 : i32, src2_row = 4096 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 4 : i32, broadcast = true,
      immediate_bf16 = 0 : i32, profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "0419016a7c3d1fa1fc5282a76832007393b488425bf426c6e14bff868721a7c9", manifest_sha256 = "d072df623c02c1b748ed32ba40a6a9e0b6412e7e008b19f21aa72da418175a68"} : () -> ()
    func.return
  }
}
