builtin.module attributes {
  mx.contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e",
  mx.policy_sha256 = "d38fa1053d4b62efc5001fc5941481eafbd19cb1afc121c00f9c3a3fd0810b24",
  prov.quantization_manifest_sha256 = "ced00f8b87eb12d5c4b637f212cea1076b369a6d3a774b5d5bd84c2fc7806015",
  mx.source_mlir_sha256 = "934b5b3e349b3c2d478dc6b409cfd05a52d161bf35aaa39cc17711e940812d0e",
  mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"
} {
  func.func @vpu_mul() {
    "mx_gemmini.vpu_execute"() {site_id = "vpu:mul", kind = "mul",
      src1_row = 0 : i32, src2_row = 4096 : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = 1 : i32,
      broadcast = false, immediate_bf16 = 0 : i32,
      profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d", contract_sha256 = "b3416a8a7040c4dbd22a7f03840fd819a0b4ec984064482dbeebcd160749f80e", policy_sha256 = "d38fa1053d4b62efc5001fc5941481eafbd19cb1afc121c00f9c3a3fd0810b24", manifest_sha256 = "ced00f8b87eb12d5c4b637f212cea1076b369a6d3a774b5d5bd84c2fc7806015"} : () -> ()
    func.return
  }
}
