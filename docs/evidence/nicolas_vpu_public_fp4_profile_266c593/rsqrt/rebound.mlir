builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "62069112a7f860a501d7dedd95188480479e639adce53c22e1b3086f6550ae1d",
  prov.quantization_manifest_sha256 = "b99f88703ea1fb2c68e222c81823eb451bf6a9bfdb46b83a78e204e9933f72c4",
  mx.source_mlir_sha256 = "c18a1ebeef5b1d91c8c543f4ad79c4a6df052959db6ad473433ee4d5a8559722",
  mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"
} {
  func.func @vpu_rsqrt() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:rsqrt", kind = "rsqrt",
      src1_row = 0 : i32, src2_row = 0 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 1 : i32,
      broadcast = false, immediate_bf16 = 0 : i32,
      profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "62069112a7f860a501d7dedd95188480479e639adce53c22e1b3086f6550ae1d", manifest_sha256 = "b99f88703ea1fb2c68e222c81823eb451bf6a9bfdb46b83a78e204e9933f72c4"} : () -> ()
    func.return
  }
}
