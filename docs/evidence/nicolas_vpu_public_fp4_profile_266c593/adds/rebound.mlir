builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "7ca8be6703ba9fe5e7d53585a1c0f73f75c4aac3d0d11bda61f9a4d5eb726637",
  prov.quantization_manifest_sha256 = "7bac1c2f2d883b2eb8891a795f5e820eaffffe20c6c6d76d7a2df96073361f54",
  mx.source_mlir_sha256 = "4d7458acc5019efc9104bfca0c967639cd8368659a09978a0e56a8009766fdb7",
  mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"
} {
  func.func @vpu_adds() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:adds", kind = "adds",
      src1_row = 0 : i32, src2_row = 0 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 1 : i32,
      broadcast = false, immediate_bf16 = 16448 : i32,
      profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "7ca8be6703ba9fe5e7d53585a1c0f73f75c4aac3d0d11bda61f9a4d5eb726637", manifest_sha256 = "7bac1c2f2d883b2eb8891a795f5e820eaffffe20c6c6d76d7a2df96073361f54"} : () -> ()
    func.return
  }
}
