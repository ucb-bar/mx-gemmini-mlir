builtin.module attributes {mx.profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d",
          mx.contract_sha256 = "677f3dcdc53519b465d986cc31e4e02fddc50ccb4829f1a65b19b369d35be24d", mx.policy_sha256 = "87c8c1c5027871bea80ef6c2aa2f19477f5944c4431a4c5fd1b7da2958200767",
          prov.quantization_manifest_sha256 = "677f3dcdc53519b465d986cc31e4e02fddc50ccb4829f1a65b19b369d35be24d"} {
          func.func @kernel() {
            "mx_gemmini.vpu_execute"() {site_id = "forward$kernel_1__radd_0", kind = "adds",
              src1_row = 64 : i32, src2_row = 0 : i32, dst_row = 0 : i32,
              rows = 64 : i32, reduction_length = 1 : i32, broadcast = false,
              immediate_bf16 = 16043 : i32, profile_sha256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d",
              contract_sha256 = "677f3dcdc53519b465d986cc31e4e02fddc50ccb4829f1a65b19b369d35be24d", policy_sha256 = "87c8c1c5027871bea80ef6c2aa2f19477f5944c4431a4c5fd1b7da2958200767",
              manifest_sha256 = "677f3dcdc53519b465d986cc31e4e02fddc50ccb4829f1a65b19b369d35be24d"} : () -> ()
            func.return
          }
        }
