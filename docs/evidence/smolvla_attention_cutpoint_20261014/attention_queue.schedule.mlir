builtin.module {
  "schedule.plan"() <{sym_name = "mx_issue_plan", source = #schedule<policy_source kernel_mining>}> ({
    %0 = "schedule.bind"() <{target = "@mx_issue"}> : () -> !schedule.handle
    "schedule.apply_policy"(%0) <{policy = "mx_queued_commands"}> : (!schedule.handle) -> ()
  }) : () -> ()
}