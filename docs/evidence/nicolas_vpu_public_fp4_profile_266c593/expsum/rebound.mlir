builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "90b7b25704bfa7a1715f1bf00751fc366e2d9fc480125583e341de5f993e48e3",
  prov.quantization_manifest_sha256 = "ac46aa881e4ca6420d4bbc02205e44010bbdd22f0916d1d4d7b26cb91df2f957",
  mx.source_mlir_sha256 = "efa899b913e586ad65617ba55456263763d2e5c33f09ef23e13de19e5d6fc000",
  mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"
} {
  func.func @fused_vpu() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:expsum", kind = "expsum",
      src1_row = 0 : i32, src2_row = 4096 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 4 : i32, broadcast = true,
      immediate_bf16 = 0 : i32, second_dst_row = 12288 : i32, profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "90b7b25704bfa7a1715f1bf00751fc366e2d9fc480125583e341de5f993e48e3", manifest_sha256 = "ac46aa881e4ca6420d4bbc02205e44010bbdd22f0916d1d4d7b26cb91df2f957"} : () -> ()
    func.return
  }
}
