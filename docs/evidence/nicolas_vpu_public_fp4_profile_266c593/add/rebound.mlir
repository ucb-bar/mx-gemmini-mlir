builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "efb17537ad376fb1fa68d6a7fda68776ab6041ad6a08a03997bb5e9508d9d313",
  prov.quantization_manifest_sha256 = "56d0758889fe5dabb36ef22db9d522caa592633c0aa362ae1ef33e0ec280f68a",
  mx.source_mlir_sha256 = "35093cdbc99b575f8c606da6237cb893beee86c8f09254184378a45409b378ca",
  mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"
} {
  func.func @vpu_add() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:add", kind = "add",
      src1_row = 0 : i32, src2_row = 4096 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 1 : i32,
      broadcast = false, immediate_bf16 = 0 : i32,
      profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "efb17537ad376fb1fa68d6a7fda68776ab6041ad6a08a03997bb5e9508d9d313", manifest_sha256 = "56d0758889fe5dabb36ef22db9d522caa592633c0aa362ae1ef33e0ec280f68a"} : () -> ()
    func.return
  }
}
