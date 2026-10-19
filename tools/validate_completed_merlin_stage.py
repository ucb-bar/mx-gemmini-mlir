"""Independently validate complete outputs without rerunning or altering a stage.

Older in-flight builders reject streams above their default parser budget.
This reader binds an explicit complete capture shape and retains the original
terminal receipt, including its refusal. It never changes a numerical gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.build_merlin_stage import _observe_spike_evidence
from tools.capture_matched_tinyllama_onepass import sha


def validate(stage: Path, build: Path, out: Path) -> dict:
    from merlin.common.mlir_query import forward_signature
    from merlin.runtime.backends.spike_model import parse_console

    if out.exists():
        raise ValueError("output validation receipt must be fresh")
    original = build / "mx_stage_probe.json"
    receipt = json.loads(original.read_text())
    run = receipt["spike"]
    if (run["exit_code"] != 0 or not (run["done"] or
            run.get("output_evidence_error") == "invalid chunked model output type or count")):
        raise ValueError("stage did not finish successfully or has an unrelated validation failure")
    source = stage / "model.mlir"
    capture = json.loads((stage / "capture_receipt.json").read_text())
    if sha(source) != receipt["capture_model_sha256"] or sha(source) != capture["artifacts"]["model.mlir"]["sha256"]:
        raise ValueError("capture differs from the terminal stage receipt")
    _, outputs = forward_signature(source)
    log = build / "spike.log"
    console = log.read_text()
    parsed = parse_console(console, expected_output_elements=int(np.prod(outputs[0][0])))
    observed = _observe_spike_evidence(console, source, receipt["build_hash"],
                                       first_typed=receipt["first_output_sha256_enabled"], require_complete=True)
    little = parsed["outputs"].astype(parsed["outputs"].dtype.newbyteorder("<"), copy=False)
    digest = hashlib.sha256(little.tobytes()).hexdigest()
    if digest != observed["indexed"][0]["sha256"]:
        raise ValueError("complete output words differ from the runtime digest")
    result = {"schema": "mx_gemmini.complete_stage_output_validation.v1",
              "scope": "completed_process_and_full_output_protocol_not_numerical_qualification",
              "original_stage_receipt_sha256": sha(original),
              "capture_model_sha256": sha(source), "console_sha256": sha(log),
              "complete_output_elements": parsed["outputs"].size,
              "output_digests": observed,
              "original_reader_refused": not run["done"], "validated": True}
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage", "build", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.stage, args.build, args.out), indent=2))


if __name__ == "__main__":
    main()
