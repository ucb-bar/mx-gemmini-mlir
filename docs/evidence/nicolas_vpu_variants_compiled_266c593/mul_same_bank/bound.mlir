builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "ff2dd6fdb11f3e58e5ef88fb9a4547da0b9c2d09dc70c2e43890f46023244222",
  prov.quantization_manifest_sha256 = "156a4d1dab5deed0d19b5ac5756b25e4b81227e45305218c72c59d9da53dc426",
  mx.source_mlir_sha256 = "934b5b3e349b3c2d478dc6b409cfd05a52d161bf35aaa39cc17711e940812d0e",
  mx.profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"
} {
  func.func @vpu_mul_same_bank() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:mul_same_bank", kind = "mul",
      src1_row = 0 : i32, src2_row = 2048 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 1 : i32,
      broadcast = false, immediate_bf16 = 0 : i32,
      profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "ff2dd6fdb11f3e58e5ef88fb9a4547da0b9c2d09dc70c2e43890f46023244222", manifest_sha256 = "156a4d1dab5deed0d19b5ac5756b25e4b81227e45305218c72c59d9da53dc426"} : () -> ()
    func.return
  }
}
