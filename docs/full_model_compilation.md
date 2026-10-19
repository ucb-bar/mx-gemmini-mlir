# Complete-model support

The compiler captures complete graphs through model2MLIR, lowers portable
host work through Merlin, and emits MX matrix catalogs plus selected BF16 VPU
operations. Separate stage programs are linked with persistent-state storage
into a standalone RV64 executable.

| Model | Capture and link | Completed workload | Published performance scope |
| --- | --- | --- | --- |
| TinyLlama 1.1B Chat | Yes | Full 160-token forward; 128-token prefill + 32 cached decodes | Independent-program forward comparison; compiler-to-compiler cached comparison |
| SmolVLA base | Yes | Prefix + ten flow steps + action decode | Compiler-to-compiler complete-session comparison |
| DeepSeek-R1-Distill-Qwen-1.5B | Yes | No qualified complete execution in this release | Compilation/linking only |
| Gemma 2 2B | Yes | No qualified complete execution in this release | Compilation/linking only |

The completed runs use all-FP8 offloaded matrices and explicit host/VPU
placement policies. Graph coverage and structural ELF admission do not by
themselves establish numerical quality. The target contract remains
`structural_unqualified`, with `phase0_admitted: false`.

Start with [compile and run complete models](compile_e2e_models.md).
The [comparison guide](fair_full_model_comparisons.md) gives current cycles,
speedups, source-kernel coverage, and numerical limits. The
[artifact audit](model_results_audit.md) rechecks the retained execution and
compilation evidence. The [release registry](../examples/models/release.json)
pins the inputs, policies, repositories, tool hashes, and measured executable
identities. The [four-session link record](evidence/four_vpu_session_links_20261012/README.md)
preserves the earlier compilation-only evidence.

FireSim packaging is available for linked sessions. The package's workload
format, ELF identity, load segments, and memory requirement are checked;
a matching FPGA execution is not yet measured.
