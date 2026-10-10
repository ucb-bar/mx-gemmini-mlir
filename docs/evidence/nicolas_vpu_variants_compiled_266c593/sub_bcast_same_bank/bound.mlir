builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "64cf5eb55e4ce05e0995a37f4b7c968d7b22540f8a3755d0b791b98dad47232f",
  prov.quantization_manifest_sha256 = "80d2a0972e47cdd05c59520912b42c5c448296ba86e373521fc32e7c68ff0189",
  mx.source_mlir_sha256 = "7ea383cd153711b977cd5f2f6132dd8c084abb84ec5417305b6f29433928d424",
  mx.profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"
} {
  func.func @vpu_sub_bcast_same_bank() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:sub_bcast_same_bank", kind = "sub",
      src1_row = 0 : i32, src2_row = 2048 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 4 : i32,
      broadcast = true, immediate_bf16 = 0 : i32,
      profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "64cf5eb55e4ce05e0995a37f4b7c968d7b22540f8a3755d0b791b98dad47232f", manifest_sha256 = "80d2a0972e47cdd05c59520912b42c5c448296ba86e373521fc32e7c68ff0189"} : () -> ()
    func.return
  }
}
