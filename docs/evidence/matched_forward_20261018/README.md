# Complete quality-qualified TinyLlama forward

Both compiler programs execute all22 layers and the LM head on the same160 token
IDs and checkpoint, with complete5,120,000 FP32 output words retained. The
candidate preserves singleton-reduction arithmetic on the host, keeps sensitive
softmax/mask paths on the host, and connects22 private VPU multiply pairs.
Both prepack immutable weights and project unused checkpoint entry arguments.

| Token-to-logits execution | Modeled cycles | Mean KL, late33 | Worst KL | Top matches, late33 |
| --- | ---: | ---: | ---: | ---: |
| Retained independent reference source | 12,762,360,525 | 0.0641713 | 0.1896031 | 27/33 |
| Fresh all-VPU compiler baseline | 15,171,236,821 | 0.2962464 | 0.7839974 | 20/33 |
| Numerical-placement + connected-multiply candidate | 15,984,649,026 | 0.0637956 | 0.1699798 | 27/33 |

The candidate passes the existing unchanged5% relative mean/worst KL and
top-agreement gates. These gates cover the33 full source logit rows that were
retained; no full160-row independent-source logit claim is made. Complete
compiler outputs are retained. FP32 top agreement across160 positions improves
from103 to133; the retained source matches131. Passing numerical policy does
not mean bit identity to the independent source.

The candidate remains25.248% slower than the independent source and5.362%
slower than the inaccurate compiler baseline. It reduces VPU commands44872→29340
and extension load bytes4407640384→4259846464. Matrix work stays essentially
unchanged at40970240 mesh tiles. These counters include the extension's wider
execution scope and do not measure host-memory traffic with tracking disabled.
Fewer commands and transfers have not yet produced an end-to-end speedup.
The shared-activation-packing whole-model variant is a separate running job;
this archive contains no result for it or for private matrix/BF16 transport.

The cycle ratio uses the retained token-lookup source control, which includes
embedding lookup inside the original stage timer. The earlier12,703,713,784-cycle
precomputed-embedding control has a different logical input boundary and must
not supply this ratio. An initial exploratory comparison selected that older
timer; the corrected tool requires an explicit `--boundary-archive` and reopens
the source log and receipt chain. No simulation was rerun for that reader fix.

Both old in-flight builders completed Spike successfully but refused their
large output streams at the old parser's1,048,576-element budget. Independent
validators checked the exact captured5,120,000 count, every word, both output
digests and build identity. Their original terminal receipts/refusals remain
unchanged. This changes an evidence reader, not a numerical gate.

The compressed archive retains complete compiler logs, original receipts,
independent output-validation records, observed live simulator settings, target
catalogs, perf footers, capture IR/input/checkpoint identities, completed recipes,
accepted structural admission records and the corrected comparison. It omits
the gigabyte-scale ELFs and checkpoint data. `verify.py` checks all retained
words, digests, quality metrics and the equal source timer using NumPy. With
`--build-root TASK_TMP/mx_model_goal_20261018`, it additionally reopens both
actual compiler executables. Structural admission provides image/declared-ISA
custody; it is not an instruction policy or hardware qualification.

From the configured editable task worktree:

```sh
python -m tools.validate_completed_merlin_stage --stage "$CAPTURE" \
  --build "$BASELINE" --out "$BASELINE/mx_stage_output_validation.json"
python -m tools.validate_completed_merlin_stage --stage "$CAPTURE" \
  --build "$CANDIDATE" --out "$CANDIDATE/mx_stage_output_validation.json"
python -m tools.compare_matched_tinyllama_forward --stage "$CAPTURE" \
  --archive docs/evidence/nicolas_tinyllama_onepass160_same_input_20261014 \
  --boundary-archive docs/evidence/nicolas_tinyllama_lookup_equal_boundary_20261015 \
  --baseline "$BASELINE" --candidate "$CANDIDATE" --out-dir "$FRESH_COMPARISON"
```

Validators require fresh receipt paths and run only after the original simulator
terminates. Existing validated logs need no rerun. The complete8-area compiler
optimization goal, cached128+32 trajectory and SmolVLA evaluation remain open.
