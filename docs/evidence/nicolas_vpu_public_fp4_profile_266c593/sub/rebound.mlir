builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "c107cc19da50e7d399d38bb8be327567a0bc3da3e655eb06b8af6362d7e2001e",
  prov.quantization_manifest_sha256 = "643700e5f5ac332acdbd45e5a3698ef3d39cd11b30ed830a87e509eef3653173",
  mx.source_mlir_sha256 = "62718e1e6f7ead9ad1371296aa6a0f7bcd94a7e8ee11a557354d3c4eabbccf30",
  mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"
} {
  func.func @vpu_sub() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:sub", kind = "sub",
      src1_row = 0 : i32, src2_row = 4096 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 1 : i32,
      broadcast = false, immediate_bf16 = 0 : i32,
      profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "c107cc19da50e7d399d38bb8be327567a0bc3da3e655eb06b8af6362d7e2001e", manifest_sha256 = "643700e5f5ac332acdbd45e5a3698ef3d39cd11b30ed830a87e509eef3653173"} : () -> ()
    func.return
  }
}
