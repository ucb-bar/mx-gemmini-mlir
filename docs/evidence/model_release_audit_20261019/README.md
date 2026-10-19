# Published model audit

The [audit report](audit.json) records a successful recheck of all three
completed model workloads, including the actual original large executables
and stage objects. The model readers' logs are retained beside it. See the
[human-readable audit](../../model_results_audit.md) for scope and findings.

Additional checks retain:

- `check-*.json`: complete output checker results for all three original runs.
- `reproduction-input-checks.json`: actual checksum-verified input exports and checked build-command plans; compilation was not rerun.
- `targeted-tests-final.log`: nine passing packaging/deadlock refusal and Radiance retained-artifact tests.
- `firesim-parser-check.json`: the actual packaged cached TinyLlama executable resolved by the pinned FireSim parser.
- `firesim-requirements.json`: target profile, memory map, and deployment requirements for that package.

`files.json` hashes these report files. The checkpoint files and large
executables remain external; their sizes and SHA-256 values are recorded in
the model archives and release registry. A recorded hash is distinguished
from a byte recheck by `actual_external_images_checked` in `audit.json`.
No matching FPGA execution is asserted.

Reproduce the portable audit with:

```sh
python -m tools.audit_published_models --out /path/to/new-audit
```

Add `--build-root /path/to/retained-builds` for the original large images and
`--radiance-root /path/to/radiance-kernels` for source-driver byte checks.
