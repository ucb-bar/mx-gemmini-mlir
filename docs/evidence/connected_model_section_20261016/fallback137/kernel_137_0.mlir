builtin.module attributes {mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d",
              mx.contract_sha256 = "6e901cdfb635c9c2834e3d32fe9e677db587abba04c33fdd0c2e7bb34eb55a99", mx.policy_sha256 = "6e901cdfb635c9c2834e3d32fe9e677db587abba04c33fdd0c2e7bb34eb55a99",
              prov.quantization_manifest_sha256 = "6e901cdfb635c9c2834e3d32fe9e677db587abba04c33fdd0c2e7bb34eb55a99"} {
              func.func @multiply() {
                "mx_gemmini.vpu_execute"() {site_id = "forward$kernel_0__rmul_0", kind = "mul",
                  src1_row = 0 : i32, src2_row = 4096 : i32, dst_row = 8192 : i32,
                  rows = 18 : i32, reduction_length = 1 : i32, broadcast = false,
                  immediate_bf16 = 0 : i32, profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d",
                  contract_sha256 = "6e901cdfb635c9c2834e3d32fe9e677db587abba04c33fdd0c2e7bb34eb55a99", policy_sha256 = "6e901cdfb635c9c2834e3d32fe9e677db587abba04c33fdd0c2e7bb34eb55a99",
                  manifest_sha256 = "6e901cdfb635c9c2834e3d32fe9e677db587abba04c33fdd0c2e7bb34eb55a99"} : () -> ()
                func.return
              }
            }