# Evaluate matched cached execution

`tools.evaluate_matched_tinyllama_session` evaluates the fresh 128-token prefill
and 32 cached decode capture using explicit, source-bound placement policies.
It does not assume every operation runs on the VPU. The older all-FP8 evaluator
retains its original all-VPU contract.

The evaluator reopens the bundle inventory, matched token IDs and checkpoint,
policies, source coverage, compiled objects and entry-projection proofs. For a
completed run, it checks the actual console, observed simulator environment,
loaded image, final admission and performance report. Merlin rereads all 33
complete native logit rows and retained dumps. The evaluator reconciles step
and preparation records and reports their combined execution total. Reset and
cross-program state handoff remain inside that total; console and phase
bookkeeping retain the harness's original exclusion.

The result reports per-row cosine, error, KL divergence and top-token agreement
against the captured FP32 recurrent trajectory. There is no independent cached
reference execution, so this tool reports neither a reference-relative numerical
gate nor an independent-source speedup. The independent 160-token forward has
its separate source-relative gate and timing boundary.

Run from the repository root with Merlin and model2MLIR available on
`PYTHONPATH`. Use fresh output paths and the actual receipt-bound build and
policy files:

```sh
"$PYTHON" -m tools.evaluate_matched_tinyllama_session \
  --run-receipt "$RUN/mx_session_run.json" \
  --link-receipt "$LINK/mx_session_link.json" \
  --session-root "$CAPTURE/capture" \
  --reference-archive docs/evidence/nicolas_tinyllama_onepass160_same_input_20261014 \
  --prefill-build "$PREFILL_BUILD" --decode-build "$DECODE_BUILD" \
  --prefill-policy "$CAPTURE/prefill_policy.json" \
  --decode-policy "$CAPTURE/decode_policy.json" \
  --profile profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json \
  --out "$NEW_REPORT"
```

The link must request `--measure-preparation`, all of `prefill:0` and `decode:0`,
and `--dump-limit-bytes 128000`. Incomplete runs refuse evaluation. Simulation
observation timeouts do not justify restarting an engine that remains live.

After evaluating both runs, use `tools.compare_linked_session_cycles` with
`--reference-link`, `--reference-run`, `--candidate-link`, `--candidate-run`,
and a fresh `--out`. It requires identical output digests, capture and timing
settings. For preparation-measured runs it reparses the native session records
and compares execution totals including reset and handoff, reporting those
costs separately. Legacy runs retain their step-only boundary; comparisons
between the two timing selections refuse.
