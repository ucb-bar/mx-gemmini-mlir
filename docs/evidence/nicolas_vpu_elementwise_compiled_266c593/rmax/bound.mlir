builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "f7787c0963fc1256a68458e2f11b9ed23ee550063a57cc8ac6b4d7f609a163d1",
  prov.quantization_manifest_sha256 = "120446b6f4b15301b577479cb9592adef65925c90e44a23d0b84737c8ddfc31d",
  mx.source_mlir_sha256 = "2365a7e2ae6c4bb772cc5330d5a4c2f6c1ecdd28ae67131a3d0a9133955f9154",
  mx.profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"
} {
  func.func @vpu_rmax() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:rmax", kind = "rmax",
      src1_row = 0 : i32, src2_row = 0 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 4 : i32,
      broadcast = false, immediate_bf16 = 0 : i32,
      profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "f7787c0963fc1256a68458e2f11b9ed23ee550063a57cc8ac6b4d7f609a163d1", manifest_sha256 = "120446b6f4b15301b577479cb9592adef65925c90e44a23d0b84737c8ddfc31d"} : () -> ()
    func.return
  }
}
