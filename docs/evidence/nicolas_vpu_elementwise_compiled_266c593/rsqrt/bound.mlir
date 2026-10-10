builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "62069112a7f860a501d7dedd95188480479e639adce53c22e1b3086f6550ae1d",
  prov.quantization_manifest_sha256 = "6863c1a954bc4d551e5879c66159b5e4f4bb385885a8fd6a45569b24bcc7ecfe",
  mx.source_mlir_sha256 = "c18a1ebeef5b1d91c8c543f4ad79c4a6df052959db6ad473433ee4d5a8559722",
  mx.profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"
} {
  func.func @vpu_rsqrt() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:rsqrt", kind = "rsqrt",
      src1_row = 0 : i32, src2_row = 0 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 1 : i32,
      broadcast = false, immediate_bf16 = 0 : i32,
      profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "62069112a7f860a501d7dedd95188480479e639adce53c22e1b3086f6550ae1d", manifest_sha256 = "6863c1a954bc4d551e5879c66159b5e4f4bb385885a8fd6a45569b24bcc7ecfe"} : () -> ()
    func.return
  }
}
