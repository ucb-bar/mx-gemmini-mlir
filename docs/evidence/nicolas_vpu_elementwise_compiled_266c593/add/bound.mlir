builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "efb17537ad376fb1fa68d6a7fda68776ab6041ad6a08a03997bb5e9508d9d313",
  prov.quantization_manifest_sha256 = "ee563eb1357e9a9541b952ee96b58f3586fd6afaf7649151a9f9aaa5e48ef50b",
  mx.source_mlir_sha256 = "35093cdbc99b575f8c606da6237cb893beee86c8f09254184378a45409b378ca",
  mx.profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"
} {
  func.func @vpu_add() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:add", kind = "add",
      src1_row = 0 : i32, src2_row = 4096 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 1 : i32,
      broadcast = false, immediate_bf16 = 0 : i32,
      profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "efb17537ad376fb1fa68d6a7fda68776ab6041ad6a08a03997bb5e9508d9d313", manifest_sha256 = "ee563eb1357e9a9541b952ee96b58f3586fd6afaf7649151a9f9aaa5e48ef50b"} : () -> ()
    func.return
  }
}
