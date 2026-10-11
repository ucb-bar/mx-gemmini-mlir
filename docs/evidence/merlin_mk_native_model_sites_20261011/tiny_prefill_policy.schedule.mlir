builtin.module {
  "schedule.plan"() <{sym_name = "mx_dense_kernel_0_plan", source = #schedule<policy_source kernel_mining>}> ({
    %0 = "schedule.bind"() <{target = "@mx_dense_kernel_0"}> : () -> !schedule.handle
    "schedule.apply_policy"(%0) <{policy = "mx_native_dram_loop"}> : (!schedule.handle) -> ()
  }) : () -> ()
}