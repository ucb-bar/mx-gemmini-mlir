builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "408fed154b35193dd05412ec434279d48f209679e95307862bfb158dbbd63109",
  prov.quantization_manifest_sha256 = "8b97b025f73c69875ecd36c06e83642f4c9cf7bc3fc58fd9de1dce77b7c458ba",
  mx.source_mlir_sha256 = "925096f1dd1c7698197ca402faa3ed1a84ff0b05ba8e8c31be9a2e4b7249da73",
  mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"
} {
  func.func @vpu_ramax() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:ramax", kind = "ramax",
      src1_row = 0 : i32, src2_row = 0 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 4 : i32,
      broadcast = false, immediate_bf16 = 0 : i32,
      profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "408fed154b35193dd05412ec434279d48f209679e95307862bfb158dbbd63109", manifest_sha256 = "8b97b025f73c69875ecd36c06e83642f4c9cf7bc3fc58fd9de1dce77b7c458ba"} : () -> ()
    func.return
  }
}
