builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "e6d73eddb08617c719166aa17d64b934fb62c2b04c40418c9a350478c0652966",
  prov.quantization_manifest_sha256 = "30f369f1ecc8ce5e8ac1d5998af3f1a9725b4d95f430f3b38684691f249dfb75",
  mx.source_mlir_sha256 = "3cfae7502b0a92d65bf296837533bd700546054206d574984e8a2f321dd87a82",
  mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"
} {
  func.func @vpu_muls() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:muls", kind = "muls",
      src1_row = 0 : i32, src2_row = 0 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 1 : i32,
      broadcast = false, immediate_bf16 = 16128 : i32,
      profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "e6d73eddb08617c719166aa17d64b934fb62c2b04c40418c9a350478c0652966", manifest_sha256 = "30f369f1ecc8ce5e8ac1d5998af3f1a9725b4d95f430f3b38684691f249dfb75"} : () -> ()
    func.return
  }
}
