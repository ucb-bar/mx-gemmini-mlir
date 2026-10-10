builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "9b92855e636b80bb86016c60d3ccdea6627f3e686238a6b45c172f3e258ec6c2",
  prov.quantization_manifest_sha256 = "10c4543d8725bf41e6ea8fd3bf3b927457df8ad363c2d9ebbeea6709f98ef314",
  mx.source_mlir_sha256 = "2342f2710aae86db0ec67acb4c22a6f42f0843ca0722cd306c44a9e476b73143",
  mx.profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487"
} {
  func.func @vpu_chain() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:chain:0", kind = "add",
      src1_row = 0 : i32, src2_row = 4096 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 1 : i32, broadcast = false,
      immediate_bf16 = 0 : i32, profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "9b92855e636b80bb86016c60d3ccdea6627f3e686238a6b45c172f3e258ec6c2", manifest_sha256 = "10c4543d8725bf41e6ea8fd3bf3b927457df8ad363c2d9ebbeea6709f98ef314"} : () -> ()
    "mx_gemmini.vpu_execute"() {site_id = "vpu:chain:1", kind = "muls",
      src1_row = 8192 : i32, src2_row = 0 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 1 : i32, broadcast = false,
      immediate_bf16 = 16128 : i32, profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "9b92855e636b80bb86016c60d3ccdea6627f3e686238a6b45c172f3e258ec6c2", manifest_sha256 = "10c4543d8725bf41e6ea8fd3bf3b927457df8ad363c2d9ebbeea6709f98ef314"} : () -> ()
    "mx_gemmini.vpu_execute"() {site_id = "vpu:chain:2", kind = "rmax",
      src1_row = 8192 : i32, src2_row = 0 : i32,
      dst_row = 12288 : i32, rows = 64 : i32,
      reduction_length = 4 : i32, broadcast = false,
      immediate_bf16 = 0 : i32, profile_sha256 = "8674d57c7a133eae83baa14385d912a2c1c623fe02d2ea7f98826a09ed783487", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "9b92855e636b80bb86016c60d3ccdea6627f3e686238a6b45c172f3e258ec6c2", manifest_sha256 = "10c4543d8725bf41e6ea8fd3bf3b927457df8ad363c2d9ebbeea6709f98ef314"} : () -> ()
    func.return
  }
}
