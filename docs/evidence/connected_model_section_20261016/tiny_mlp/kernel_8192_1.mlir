builtin.module attributes {mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d",
              mx.contract_sha256 = "26714fc6e9e1d870ad845bbd747dbd7e5a0ef1d1ebefef65f1a34f5893338329", mx.policy_sha256 = "26714fc6e9e1d870ad845bbd747dbd7e5a0ef1d1ebefef65f1a34f5893338329",
              prov.quantization_manifest_sha256 = "26714fc6e9e1d870ad845bbd747dbd7e5a0ef1d1ebefef65f1a34f5893338329"} {
              func.func @multiply() {
                "mx_gemmini.vpu_execute"() {site_id = "forward$kernel_1__rmul_1", kind = "mul",
                  src1_row = 8192 : i32, src2_row = 4096 : i32, dst_row = 0 : i32,
                  rows = 1024 : i32, reduction_length = 1 : i32, broadcast = false,
                  immediate_bf16 = 0 : i32, profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d",
                  contract_sha256 = "26714fc6e9e1d870ad845bbd747dbd7e5a0ef1d1ebefef65f1a34f5893338329", policy_sha256 = "26714fc6e9e1d870ad845bbd747dbd7e5a0ef1d1ebefef65f1a34f5893338329",
                  manifest_sha256 = "26714fc6e9e1d870ad845bbd747dbd7e5a0ef1d1ebefef65f1a34f5893338329"} : () -> ()
                func.return
              }
            }