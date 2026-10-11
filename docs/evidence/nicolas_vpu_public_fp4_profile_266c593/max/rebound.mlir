builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "d35df8efd3174c82a4ae9d8e377f247d99e263296cd986bd2e58f54fc3edf4f7",
  prov.quantization_manifest_sha256 = "f9570506236430f880ebc5b7c98f66ea29487ee6b1ed40426b5f0ca91bc2d2b6",
  mx.source_mlir_sha256 = "ba142e9804a8d2bccd352a3f2b074ccd745e6934e99834b42754037691b68d46",
  mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"
} {
  func.func @vpu_max() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:max", kind = "max",
      src1_row = 0 : i32, src2_row = 4096 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 1 : i32,
      broadcast = false, immediate_bf16 = 0 : i32,
      profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "d35df8efd3174c82a4ae9d8e377f247d99e263296cd986bd2e58f54fc3edf4f7", manifest_sha256 = "f9570506236430f880ebc5b7c98f66ea29487ee6b1ed40426b5f0ca91bc2d2b6"} : () -> ()
    func.return
  }
}
