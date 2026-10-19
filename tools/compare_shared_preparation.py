"""Compare complete matched stage outputs and costs for preparation experiments.

Bit equality to a compiled baseline establishes unchanged output for this stage,
not independent model numerical quality or a whole-model performance result.
The first output requires complete console words. Every additional output needs
explicit independently derived full bytes checked against its complete ABI hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.build_merlin_stage import _observe_spike_evidence
from tools.capture_matched_tinyllama_onepass import sha
from tools.report_stage_profile import parse_gemmini_perf


def compare(stage: Path, builds: dict[str, Path], out: Path, *, additional_outputs: dict[int, Path] | None = None):
    from merlin.common.mlir_query import forward_signature
    from merlin.runtime.backends.spike_model import parse_console

    if out.exists() or len(builds) < 2:
        raise ValueError("comparison needs a fresh output and at least two stages")
    capture = json.loads((stage / "capture_receipt.json").read_text())
    for name in ("model.mlir", "weights.safetensors", "inputs.npz", "golden.npy"):
        if sha(stage / name) != capture["artifacts"][name]["sha256"]:
            raise ValueError("comparison capture changed: " + name)
    source = stage / "model.mlir"
    _, outputs = forward_signature(source)
    if not outputs or outputs[0][1] != "f32":
        raise ValueError("preparation qualifier requires a complete first f32 stage output")
    additional_outputs = {} if additional_outputs is None else additional_outputs
    if set(additional_outputs) != set(range(1, len(outputs))):
        raise ValueError("every additional output requires an explicit complete independent byte reference")
    references = {index: {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}
                  for index, path in additional_outputs.items()}
    count = int(np.prod(outputs[0][0]))
    records, arrays, identity = {}, {}, None
    for label, build in builds.items():
        receipt_path = build / "mx_stage_probe.json"
        receipt = json.loads(receipt_path.read_text())
        run = receipt["spike"]
        if (not run["done"] or run["exit_code"] != 0 or receipt["op_profile"]
                or receipt["capture_model_sha256"] != sha(source)
                or receipt["checkpoint_weights_sha256"] != sha(stage / "weights.safetensors")):
            raise ValueError("stage is incomplete, instrumented, or differs from capture: " + label)
        current = {"profile_sha256": receipt["profile_sha256"],
                   "extension_sha256": run["extension_sha256"], "environment": run["environment"],
                   "execution_policy": receipt["execution_policy"],
                   "spike_isa": receipt["spike_isa"]}
        if identity is None:
            identity = current
        elif identity != current:
            raise ValueError("stage profile, simulator settings or numeric policy differs: " + label)
        extensions = [arg.removeprefix("--extlib=") for arg in run["command"]
                      if arg.startswith("--extlib=")]
        if len(extensions) != 1 or sha(Path(extensions[0])) != run["extension_sha256"]:
            raise ValueError("loaded extension changed: " + label)
        recipe = json.loads((build / "compilation_recipe.json").read_text())
        elf = build / "model.elf"
        if recipe["status"] != "completed" or recipe["executable"]["sha256"] != sha(elf):
            raise ValueError("completed recipe differs from executable: " + label)
        admissions = list(build.glob("invocations/*/policy/admission.json"))
        if len(admissions) != 1:
            raise ValueError("stage lacks one final-image admission: " + label)
        admission = json.loads(admissions[0].read_text())
        if admission["status"] != "accepted" or admission["elf_sha256"] != sha(elf):
            raise ValueError("stage admission differs from executable: " + label)
        console = (build / "spike.log").read_text()
        parsed = parse_console(console, expected_output_elements=count)
        evidence = _observe_spike_evidence(console, source, receipt["build_hash"],
                                           first_typed=receipt["first_output_sha256_enabled"],
                                           require_complete=True)
        bits = parsed["outputs"].astype("<f4", copy=False).view("<u4")
        if bits.size != count or hashlib.sha256(bits.tobytes()).hexdigest() != evidence["indexed"][0]["sha256"]:
            raise ValueError("complete console words differ from digest: " + label)
        for index, reference in references.items():
            observed = evidence["indexed"][index]
            if observed["bytes"] != reference["bytes"] or observed["sha256"] != reference["sha256"]:
                raise ValueError("complete additional output differs from independent reference: " + label)
        arrays[label] = bits
        rewrite = build / "device_host_abi/mx_shared_activation_rewrite.json"
        groups = json.loads(rewrite.read_text())["groups"] if rewrite.exists() else []
        transport = build / "device_host_abi/mx_matrix_transport_rewrite.json"
        edges = json.loads(transport.read_text())["edges"] if transport.exists() else []
        records[label] = {
            "cycles": parsed["metrics"]["cycles"], "perf": parse_gemmini_perf(console),
            "build_hash": receipt["build_hash"], "elf_bytes": elf.stat().st_size,
            "elf_sha256": sha(elf), "admission_sha256": sha(admissions[0]),
            "stage_receipt_sha256": sha(receipt_path), "console_sha256": sha(build / "spike.log"),
            "output_sha256": evidence["indexed"][0]["sha256"], "output_elements": count,
            "arena_estimate": receipt["arena_estimate"], "groups": groups,
            "private_transport_edges": edges,
            "retained_unused_rhs": receipt.get("retain_shared_activation_rhs", False),
            **({"additional_output_observations": evidence["indexed"][1:]} if references else {}),
        }
    baseline = next(iter(builds))
    for label, bits in arrays.items():
        if not np.array_equal(bits, arrays[baseline]):
            raise ValueError("stage output changed: " + label)
        records[label]["cycles_saved_percent"] = 100 * (1 - records[label]["cycles"] / records[baseline]["cycles"])
    report = {"schema": "mx_gemmini.shared_preparation_comparison.v1",
              "scope": "complete_stage_bit_equality_and_matched_forward_interval_not_whole_model",
              "baseline": baseline, "all_words_exact": True, "identity": identity,
              "capture_receipt_sha256": sha(stage / "capture_receipt.json"), "results": records}
    if references:
        report.update(additional_output_references=references,
                      observation_scope="first_output_full_console_words_and_all_additional_full_reference_hashes")
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--build", action="append", required=True, help="LABEL=BUILD; baseline first")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--additional-output-reference", action="append", default=[], metavar="INDEX=PATH",
                        help="independently derived complete bytes for every additional observable output")
    args = parser.parse_args()
    builds = {}
    for value in args.build:
        label, separator, path = value.partition("=")
        if not separator or not label or not path or label in builds:
            parser.error("each build needs a unique LABEL=BUILD")
        builds[label] = Path(path)
    references = {}
    for value in args.additional_output_reference:
        index, separator, path = value.partition("=")
        if not separator or not index.isdecimal() or int(index) <= 0 or not path or int(index) in references:
            parser.error("each additional output needs a distinct positive INDEX=PATH")
        references[int(index)] = Path(path)
    result = compare(args.stage, builds, args.out, additional_outputs=references)
    print(json.dumps({label: row["cycles"] for label, row in result["results"].items()}))


if __name__ == "__main__":
    main()
