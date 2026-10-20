"""Compare a fresh compiler forward with the completed Radiance Rocket port.

Reopen executable/build observations, every logit, input identities and matrix
work before applying the unchanged late-33 and complete-160 quality gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from mx_gemmini_support.spike_timing import reject_deadlocked_timing
from tools.compare_radiance_rocket_forward import (
    ROOT,
    compare,
    load_module,
    quality_and_ratio,
)
from tools.qualify_radiance_rocket_llama import EXTENSION_SHA256, PROFILE_SHA256, sha
from tools.report_stage_profile import parse_gemmini_perf


def expected_matrix_tiles(catalog):
    """Derive complete all-FP8 matrix work from the source-bound catalog."""
    kernels = {row["symbol"]: row for row in catalog["kernels"]}
    total = 0
    for binding in catalog["bindings"]:
        row = kernels[binding["symbol"]]
        if row["precision"] != "mxfp8":
            raise ValueError("matched forward requires all-FP8 matrix precision")
        m, n, k = row["padded_shape_mnk"]
        batch = row["dimensions"]["batch"]
        if min(m, n, k, batch) <= 0 or any(value % 16 for value in (m, n, k)):
            raise ValueError("invalid compiled matrix extent")
        total += batch * (m // 16) * (n // 16) * (k // 16)
    if not total:
        raise ValueError("compiled forward has no matrix work")
    return total


def compare_candidate(port, capture, build):
    # Reuse the complete source and frozen compiler archive identity checks.
    control = compare(port, capture)
    report = json.loads((build / "mx_stage_probe.json").read_text())
    source = json.loads((port / "receipt.json").read_text())
    run = report["spike"]
    if (
        report.get("op_profile") is not False
        or not run["done"]
        or run["exit_code"] != 0
        or report["capture_model_sha256"]
        != source["capture_files_sha256"]["model.mlir"]
        or report["checkpoint_weights_sha256"]
        != source["capture_files_sha256"]["weights.safetensors"]
        or report["profile_sha256"] != PROFILE_SHA256
        or run["extension_sha256"] != EXTENSION_SHA256
        or report["unbound_precision_proposals"] != 0
        or report["raw_selected_source_coverage_complete"] is not True
    ):
        raise ValueError("candidate completion, input, precision or target differs")
    for key, value in source["environment"].items():
        if run["environment"].get(key) != value:
            raise ValueError("candidate execution settings differ")
    command = run["command"]
    if (
        sha(command[0]) != source["tools_sha256"]["spike"]
        or "--isa=rv64gc_zicntr" not in command
        or "--extension=gemmini" not in command
        or "-m0x80000000:0x1c0000000" not in command
        or Path(command[-1]).resolve() != (build / "model.elf").resolve()
    ):
        raise ValueError("candidate simulator or executable differs")
    extension = next(
        arg.split("=", 1)[1] for arg in command if arg.startswith("--extlib=")
    )
    if sha(extension) != EXTENSION_SHA256:
        raise ValueError("candidate extension bytes differ")
    # Reopen every directly observed compiler/link input and the actual ELF.
    from merlin.llvmlower.compilation_recipe import verify_completed_recipe

    recipe = build / "compilation_recipe.json"
    verify_completed_recipe(
        recipe, executable=build / "model.elf", expected_recipe_sha256=sha(recipe)
    )
    admission = list((build / "invocations").glob("*/policy/admission.json"))
    if len(admission) != 1:
        raise ValueError("candidate lacks one final image admission")
    admitted = json.loads(admission[0].read_text())
    if admitted.get("status") != "accepted" or admitted.get("elf_sha256") != sha(
        build / "model.elf"
    ):
        raise ValueError("candidate image differs from structural admission")
    ids = re.findall(
        r"static long merlin_in_[0-9]+\[\] = \{([^}]+)\};",
        (build / "cgen/model_io.h").read_text(),
    )
    if (
        len(ids) != 1
        or np.array([int(x) for x in ids[0].split(",")], dtype="<i8").tobytes()
        != (port / "token_ids.bin").read_bytes()
    ):
        raise ValueError("candidate embedded token IDs differ")
    console = (build / "spike.log").read_bytes()
    reject_deadlocked_timing(console.decode())
    reader = load_module(
        "candidate_forward_output_reader",
        ROOT / "docs/evidence/matched_forward_20261018/verify.py",
    )
    compiler, metrics = reader.parse(console)
    if metrics["build_hash"] != report["build_hash"]:
        raise ValueError("candidate log belongs to a different build")
    digest = hashlib.sha256(compiler.astype("<f4").tobytes()).hexdigest()
    if digest != run["output_digests"]["first_typed"]["sha256"]:
        raise ValueError("candidate complete words differ from the execution receipt")
    catalog = json.loads((build / "device_catalog/mx_catalog.json").read_text())
    work = parse_gemmini_perf(console.decode())
    tiles = expected_matrix_tiles(catalog)
    if (
        work["mode"] != "e4m3_vpu"
        or work["mesh_tiles"] != tiles
        or work["mesh_busy_cycles"] != 16 * tiles
    ):
        raise ValueError("candidate modeled matrix work is incomplete")
    reference = np.load(capture / "golden.npy", allow_pickle=False).reshape(160, 32000)
    observed_source = np.fromfile(port / "observed0.bin", "<f4").reshape(160, 32000)
    result = quality_and_ratio(
        reference,
        observed_source,
        compiler,
        control["source_cycles"],
        int(metrics["cycles"]),
    )
    result.update(
        {
            "schema": "mx_gemmini.radiance_rocket_candidate.v1",
            "scope": "complete uninstrumented compiler forward vs Radiance-derived Rocket port",
            "source_cycles": control["source_cycles"],
            "compiler_cycles": int(metrics["cycles"]),
            "source_revision": control["source_revision"],
            "source_receipt_sha256": sha(port / "receipt.json"),
            "source_logits_sha256": control["source_logits_sha256"],
            "compiler_logits_sha256": digest,
            "compiler_receipt_sha256": sha(build / "mx_stage_probe.json"),
            "compilation_recipe_sha256": sha(recipe),
            "elf_sha256": sha(build / "model.elf"),
            "compiler_console_sha256": hashlib.sha256(console).hexdigest(),
            "capture_files_sha256": control["capture_files_sha256"],
            "matrix_work": work,
            "expected_matrix_tiles": tiles,
            "execution_policy": report["execution_policy"],
            "vpu_softmax_implementation": report.get(
                "vpu_softmax_implementation", "bf16"
            ),
            "original_muon_execution": False,
            "faster_and_numerically_qualified": bool(
                result["numerical_gate_passed"]
                and int(metrics["cycles"]) < control["source_cycles"]
            ),
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("port-build", "capture", "compiler-build", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be fresh")
    result = compare_candidate(args.port_build, args.capture, args.compiler_build)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "source_cycles",
                    "compiler_cycles",
                    "numerical_gate_passed",
                    "qualified_speedup",
                    "faster_and_numerically_qualified",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
