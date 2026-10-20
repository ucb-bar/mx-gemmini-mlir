"""Compare a completed Radiance Rocket forward with the published compiler run.

Reopen complete output streams and their identity records before computing
quality or cycle ratios. Timing is qualified only when both the original
33-position numerical gate and the full 160-position gate pass.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from tools.compare_mx_model_quality import compare_tinyllama
from tools.qualify_radiance_rocket_llama import (
    EXTENSION_SHA256,
    PROFILE_SHA256,
    REVISION,
    check_accelerator_work,
    parse_console,
    sha,
)

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quality_and_ratio(reference, source, compiler, source_cycles, compiler_cycles):
    if (
        np.shape(reference) != (160, 32000)
        or np.shape(source) != np.shape(reference)
        or np.shape(compiler) != np.shape(reference)
    ):
        raise ValueError("matched forward requires every 160x32000 logit")
    if any(type(n) is not int or n <= 0 for n in (source_cycles, compiler_cycles)):
        raise ValueError("completed positive modeled cycle counts are required")
    late = compare_tinyllama(reference[-33:], source[-33:], compiler[-33:])
    full = compare_tinyllama(reference, source, compiler)
    passed = late["passed"] and full["passed"]
    return {
        "quality_late_33": late,
        "quality_all_160": full,
        "numerical_gate_passed": passed,
        "unqualified_source_to_compiler_cycle_ratio": source_cycles / compiler_cycles,
        "qualified_speedup": source_cycles / compiler_cycles if passed else None,
        "qualified_cycle_reduction": 1 - compiler_cycles / source_cycles
        if passed
        else None,
    }


def compare(port, capture):
    report = json.loads((port / "receipt.json").read_text())
    if (
        report.get("status") != "completed_requires_matched_comparison"
        or report.get("mode") != "recorded_tinyllama_forward"
        or report.get("source_revision") != REVISION
        or report.get("profile_sha256") != PROFILE_SHA256
        or report.get("extension_sha256") != EXTENSION_SHA256
        or report.get("phase_count") != 266
        or report.get("phase_timing") is not False
        or report.get("original_muon_timing") is not False
    ):
        raise ValueError(
            "port is incomplete or uses a different source, target or timing scope"
        )
    if (
        report["admission"].get("status") != "accepted"
        or report["admission"].get("elf_sha256") != report["elf_sha256"]
        or sha(port / "session.elf") != report["elf_sha256"]
        or sha(port / "driver.c") != report["port_source_sha256"]
    ):
        raise ValueError("port executable or source admission differs")
    for name, digest in report["files_sha256"].items():
        if Path(name).name != name or sha(port / name) != digest:
            raise ValueError("port artifact changed: " + name)
    for region, row in report["regions"].items():
        if sha(port / f"data/llama.region{region}.bin") != row["blob_sha256"]:
            raise ValueError("source model data changed")
    header = (port / "rocket_data.h").read_text()
    if any(
        flag not in header
        for flag in (
            "#define RK_PROFILE_PHASES 0",
            "#define RK_EMBEDDING_LOOKUP 1",
            "#define RK_ROUND_QUANT_INPUTS 1",
        )
    ):
        raise ValueError("port logical boundary or live quantization policy differs")
    checks = json.loads((port / "checks.json").read_text())
    cycles, _, outputs = parse_console(
        (port / "spike.log").read_text(), checks, 266, False
    )
    accelerator_work = check_accelerator_work(
        np.fromfile(port / "phases.bin", dtype="<u4").reshape(-1, 32),
        (port / "perf.txt").read_text(),
    )
    if cycles != report["cycles"] or checks[0]["name"] != "logits":
        raise ValueError("port observation differs from its receipt")
    if checks[0]["kind"] != "f32" or checks[0]["bytes"] != 160 * 32000 * 4:
        raise ValueError("source did not return the complete model-visible F32 tensor")
    source = np.frombuffer(outputs[0], "<f4").reshape(160, 32000)
    if not np.array_equal(
        source.view(np.uint32),
        np.load(port / "logits.npy", allow_pickle=False).view(np.uint32),
    ):
        raise ValueError("saved port logits differ from complete observed words")
    identities = report["capture_files_sha256"]
    for name in ("inputs.npz", "golden.npy", "model.mlir"):
        if sha(capture / name) != identities[name]:
            raise ValueError("floating reference or recorded input changed")
    with np.load(capture / "inputs.npz", allow_pickle=False) as inputs:
        ids = inputs["in0"].astype("<i8").tobytes()
    if (port / "token_ids.bin").read_bytes() != ids:
        raise ValueError("port token IDs differ from the compiler capture")
    reference = np.load(capture / "golden.npy", allow_pickle=False).reshape(160, 32000)

    release = json.loads((ROOT / "examples/models/release.json").read_text())
    current = release["models"]["tinyllama-forward"]
    archive = ROOT / current["archive"]
    # The published reader binds all dependency archives and all member bytes.
    loader = load_module("radiance_compiler_archive", archive / "verify.py")
    manifest = json.loads((archive / "artifacts.json").read_text())
    data = loader.composed_evidence(archive, manifest, set())
    parser_path = archive.parent / "matched_forward_20261018/verify.py"
    if sha(parser_path) != manifest["reader_sha256"]:
        raise ValueError("published compiler output parser changed")
    reader = load_module("radiance_compiler_output_parser", parser_path)
    compiler, metrics = reader.parse(data["candidate/spike.log"])
    prior = json.loads(data["comparison.json"])["records"]["candidate"]
    build = json.loads(data["candidate/mx_stage_probe.json"])
    environment = json.loads(data["candidate/live_environment.json"])["environment"]
    captured = json.loads(data["capture/capture_receipt.json"])
    historical_tools = json.loads(
        (
            ROOT
            / "docs/evidence/nicolas_tinyllama_onepass160_same_input_20261014/compiler_run.json"
        ).read_text()
    )
    compiled_cycles = int(metrics["cycles"])
    compiler_sha = hashlib.sha256(compiler.astype("<f4").tobytes()).hexdigest()
    if (
        compiled_cycles != current["cycles"]
        or compiled_cycles != prior["cycles"]
        or compiler_sha != prior["output_words_sha256"]
        or prior["timing_instrumented"]
        or build["profile_sha256"] != PROFILE_SHA256
        or build["spike"]["extension_sha256"] != EXTENSION_SHA256
        or historical_tools["spike_sha256"] != report["tools_sha256"]["spike"]
        or build["checkpoint_weights_sha256"] != identities["weights.safetensors"]
    ):
        raise ValueError(
            "compiler timing, outputs, checkpoint or execution identity differs"
        )
    for name in identities:
        if captured["artifacts"][name]["sha256"] != identities[name]:
            raise ValueError("source and compiler capture identities differ: " + name)
    for key, value in report["environment"].items():
        if environment.get(key) != value:
            raise ValueError("source and compiler simulator settings differ")
    result = quality_and_ratio(reference, source, compiler, cycles, compiled_cycles)
    result.update(
        {
            "schema": "mx_gemmini.radiance_rocket_forward_comparison.v1",
            "scope": "Radiance-derived Rocket port vs complete compiler forward",
            "source_revision": REVISION,
            "original_muon_execution": False,
            "source_cycles": cycles,
            "source_accelerator_work": accelerator_work,
            "compiler_accelerator_work": prior["performance"],
            "compiler_cycles": compiled_cycles,
            "source_adaptations": report["adaptations"],
            "source_receipt_sha256": sha(port / "receipt.json"),
            "source_logits_sha256": hashlib.sha256(
                source.astype("<f4").tobytes()
            ).hexdigest(),
            "compiler_logits_sha256": compiler_sha,
            "compiler_archive_manifest_sha256": sha(archive / "artifacts.json"),
            "compiler_console_sha256": hashlib.sha256(
                data["candidate/spike.log"]
            ).hexdigest(),
            "capture_files_sha256": identities,
            "port_limits": "native Rocket transport, FP32 host row arithmetic, BF16 VPU exp; not a Muon timing or a best-kernel claim",
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port-build", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be fresh")
    result = compare(args.port_build, args.capture)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["numerical_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
