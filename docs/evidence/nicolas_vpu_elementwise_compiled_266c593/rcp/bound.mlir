builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "84e77180444838c3ec6ea7a7cfc13575de3a1cb4cf44b5313b4fe50d0920374b",
  prov.quantization_manifest_sha256 = "f06880d73d7423591dba533ff8e1d71fb5aad2ad5555d0ea3562d3fe59ef625a",
  mx.source_mlir_sha256 = "71b807f88fdfb37e80b17349da045a4321018c1da44d864a6d24873a1a285ce4",
  mx.profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"
} {
  func.func @vpu_rcp() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:rcp", kind = "rcp",
      src1_row = 0 : i32, src2_row = 0 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 1 : i32,
      broadcast = false, immediate_bf16 = 0 : i32,
      profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "84e77180444838c3ec6ea7a7cfc13575de3a1cb4cf44b5313b4fe50d0920374b", manifest_sha256 = "f06880d73d7423591dba533ff8e1d71fb5aad2ad5555d0ea3562d3fe59ef625a"} : () -> ()
    func.return
  }
}
