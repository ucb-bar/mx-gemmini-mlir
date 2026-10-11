builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "84e77180444838c3ec6ea7a7cfc13575de3a1cb4cf44b5313b4fe50d0920374b",
  prov.quantization_manifest_sha256 = "daeb439e7be9801ad0655fed50545b9fd4f4d8eae47352a02ff912959565f0f7",
  mx.source_mlir_sha256 = "71b807f88fdfb37e80b17349da045a4321018c1da44d864a6d24873a1a285ce4",
  mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"
} {
  func.func @vpu_rcp() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:rcp", kind = "rcp",
      src1_row = 0 : i32, src2_row = 0 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 1 : i32,
      broadcast = false, immediate_bf16 = 0 : i32,
      profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "84e77180444838c3ec6ea7a7cfc13575de3a1cb4cf44b5313b4fe50d0920374b", manifest_sha256 = "daeb439e7be9801ad0655fed50545b9fd4f4d8eae47352a02ff912959565f0f7"} : () -> ()
    func.return
  }
}
