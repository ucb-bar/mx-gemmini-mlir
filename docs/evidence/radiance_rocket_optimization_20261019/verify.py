"""Recompute full-forward quality and timing from retained complete observations."""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), "output must be fresh"
    manifest = json.loads((HERE / "artifacts.json").read_text())
    for name, record in manifest["supplements"].items():
        raw = (HERE / name).read_bytes()
        assert len(raw) == record["bytes"] and sha(raw) == record["sha256"]
    for name, expected in manifest["support_sha256"].items():
        assert sha((ROOT / name).read_bytes()) == expected
    source_dir = HERE.parent / "radiance_rocket_forward_20261019"
    source_manifest_raw = (source_dir / "artifacts.json").read_bytes()
    assert sha(source_manifest_raw) == manifest["source_control_manifest_sha256"]
    baseline = load("source_forward_reader", source_dir / "verify.py")
    # Recheck the complete source/control and diagnostic profiles first.
    with contextlib.redirect_stdout(io.StringIO()):
        baseline.main()
    source_manifest = json.loads(source_manifest_raw)
    data = baseline.unpack(
        "forward.tar.xz", source_manifest["archives"]["forward.tar.xz"]
    )
    source_receipt, source_outputs, _ = baseline.port_outputs(data, False)
    reference = np.load(
        io.BytesIO(data["capture/golden.npy"]), allow_pickle=False
    ).reshape(160, 32000)
    source = np.frombuffer(source_outputs[0], "<f4").reshape(160, 32000)
    assert source_receipt["cycles"] == 7082331012
    parser_module = load(
        "complete_forward_parser", HERE.parent / "matched_forward_20261018/verify.py"
    )
    perf_reader = load("candidate_perf_reader", ROOT / "tools/report_stage_profile.py")
    # Use the same checked archive decoder with this evidence directory.
    archive_reader = load("candidate_archive_reader", source_dir / "verify.py")
    archive_reader.HERE = HERE
    report = {}
    for name, case in manifest["cases"].items():
        files = archive_reader.unpack(case["archive"], case)
        comparison = json.loads(files["comparison.json"])
        probe = json.loads(files["build/mx_stage_probe.json"])
        recipe = json.loads(files["build/compilation_recipe.json"])
        admission = json.loads(files["build/admission.json"])
        audit = json.loads(files["linked_payload_audit.json"])
        catalog_raw = files["build/device_catalog/mx_catalog.json"]
        catalog = json.loads(catalog_raw)
        raw = files["build/spike.log"]
        assert "MODEL DEADLOCK" not in raw.decode()
        values, metrics = parser_module.parse(raw)
        assert values.shape == (160, 32000) and np.isfinite(values).all()
        digest = sha(values.astype("<f4").tobytes())
        assert (
            digest == comparison["compiler_logits_sha256"] == case["full_logits_sha256"]
        )
        assert sha(raw) == comparison["compiler_console_sha256"]
        assert (
            sha(files["build/mx_stage_probe.json"])
            == comparison["compiler_receipt_sha256"]
        )
        assert (
            sha(files["build/compilation_recipe.json"])
            == comparison["compilation_recipe_sha256"]
        )
        assert (
            probe["op_profile"] is False
            and probe["raw_selected_source_coverage_complete"] is True
        )
        assert probe["unbound_precision_proposals"] == 0
        assert (
            probe["capture_model_sha256"]
            == source_receipt["capture_files_sha256"]["model.mlir"]
        )
        assert (
            probe["checkpoint_weights_sha256"]
            == source_receipt["capture_files_sha256"]["weights.safetensors"]
        )
        assert probe["profile_sha256"] == source_receipt["profile_sha256"]
        assert probe["execution_policy"] == comparison["execution_policy"]
        run = probe["spike"]
        assert run["done"] and run["exit_code"] == 0
        assert run["output_digests"]["first_typed"]["sha256"] == digest
        assert run["extension_sha256"] == source_receipt["extension_sha256"]
        assert all(
            run["environment"][key] == value
            for key, value in source_receipt["environment"].items()
        )
        assert {
            "--isa=rv64gc_zicntr",
            "--extension=gemmini",
            "-m0x80000000:0x1c0000000",
        } <= set(run["command"])
        assert int(metrics["memref_rank_mismatch"]) == 0
        assert metrics["build_hash"] == probe["build_hash"]
        assert recipe["status"] == "completed" and admission["status"] == "accepted"
        assert (
            recipe["executable"]["sha256"]
            == admission["elf_sha256"]
            == audit["elf_sha256"]
            == comparison["elf_sha256"]
        )
        assert (
            audit["status"] == "linked_payloads_match"
            and audit["packed_matrices_checked"] == 155
        )
        assert audit["catalog_sha256"] == sha(catalog_raw)
        assert audit["checkpoint_sha256"] == probe["checkpoint_weights_sha256"]
        ids = re.findall(
            r"static long merlin_in_[0-9]+\[\] = \{([^}]+)\};",
            files["build/cgen/model_io.h"].decode(),
        )
        assert len(ids) == 1
        assert (
            np.array([int(v) for v in ids[0].split(",")], dtype="<i8").tobytes()
            == data["source/token_ids.bin"]
        )
        kernels = {row["symbol"]: row for row in catalog["kernels"]}
        tiles = 0
        assert len(catalog["bindings"]) == 199
        for binding in catalog["bindings"]:
            kernel = kernels[binding["symbol"]]
            assert kernel["precision"] == "mxfp8"
            m, n, k = kernel["padded_shape_mnk"]
            assert min(m, n, k) > 0 and all(d % 16 == 0 for d in (m, n, k))
            tiles += kernel["dimensions"]["batch"] * (m // 16) * (n // 16) * (k // 16)
        work = perf_reader.parse_gemmini_perf(raw.decode())
        assert work == comparison["matrix_work"]
        assert (
            work["mesh_tiles"]
            == tiles
            == comparison["expected_matrix_tiles"]
            == 40970240
        )
        assert work["mesh_busy_cycles"] == tiles * 16
        cycles = int(metrics["cycles"])
        assert cycles == case["cycles"] == comparison["compiler_cycles"] > 0
        assert comparison["source_cycles"] == source_receipt["cycles"]
        assert (
            comparison["source_revision"]
            == source_receipt["source_revision"]
            == manifest["source_revision"]
        )
        assert comparison["source_logits_sha256"] == sha(source_outputs[0])
        assert comparison["source_receipt_sha256"] == sha(data["source/receipt.json"])
        gates = []
        for label, rows in (
            ("quality_late_33", slice(-33, None)),
            ("quality_all_160", slice(None)),
        ):
            expected = comparison[label]
            source_quality = baseline.quality(reference[rows], source[rows])
            candidate_quality = baseline.quality(reference[rows], values[rows])
            for role, quality in (
                ("source", source_quality),
                ("compiler", candidate_quality),
            ):
                assert all(
                    abs(value - expected[role][key]) < 1e-12
                    for key, value in quality.items()
                )
            checks = {
                "mean_kl_nats": candidate_quality["mean_kl_nats"]
                <= 1.05 * source_quality["mean_kl_nats"],
                "worst_kl_nats": candidate_quality["worst_kl_nats"]
                <= 1.05 * source_quality["worst_kl_nats"],
                "top1_agreement": candidate_quality["top1_agreement"]
                >= source_quality["top1_agreement"],
            }
            assert (
                checks == expected["checks"]
                and all(checks.values()) == expected["passed"]
            )
            assert expected["margin"] == 1.05
            gates.append(all(checks.values()))
        passed = all(gates)
        assert (
            passed
            == case["numerical_gate_passed"]
            == comparison["numerical_gate_passed"]
        )
        ratio = source_receipt["cycles"] / cycles
        assert (
            abs(ratio - comparison["unqualified_source_to_compiler_cycle_ratio"])
            < 1e-12
        )
        assert (
            comparison["qualified_speedup"]
            == case["qualified_speedup"]
            == (ratio if passed else None)
        )
        assert comparison["qualified_cycle_reduction"] == (
            1 - cycles / source_receipt["cycles"] if passed else None
        )
        assert comparison["faster_and_numerically_qualified"] == (passed and ratio > 1)
        fused = (
            len(probe["gated_projection_rewrite"]["fusions"])
            if probe.get("gated_projection_rewrite")
            else 0
        )
        assert fused == case["gated_fusions"]
        report[name] = {
            "cycles": cycles,
            "numerical_gate_passed": passed,
            "qualified_speedup": ratio if passed else None,
            "full_logits_sha256": digest,
            "gated_fusions": fused,
            "full_160": comparison["quality_all_160"]["compiler"],
            "late_33": comparison["quality_late_33"]["compiler"],
        }
    original = json.loads(data["comparison.json"])["compiler_logits_sha256"]
    for name in manifest["exact_original_output_cases"]:
        if name in report:
            assert report[name]["full_logits_sha256"] == original
    proof = json.loads((HERE / "clean-exp-rebuild-identity.json").read_text())
    assert proof["identical_elf"] is True and proof["old"] == proof["new"]
    if "exp_vpu" in manifest["cases"]:
        case = manifest["cases"]["exp_vpu"]
        files = archive_reader.unpack(case["archive"], case)
        assert (
            proof["old"]["elf_sha256"]
            == json.loads(files["comparison.json"])["elf_sha256"]
        )
    for candidate, control in manifest["exact_output_relations"].items():
        if candidate in report and control in report:
            assert (
                report[candidate]["full_logits_sha256"]
                == report[control]["full_logits_sha256"]
            )
    result = {
        "passed": True,
        "source_cycles": source_receipt["cycles"],
        "all_5120000_logits_checked_per_candidate": True,
        "source_control_and_profiles_rechecked": True,
        "actual_large_ELFs_reopened_by_this_portable_reader": False,
        "cases": report,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(args.out.read_text())


if __name__ == "__main__":
    main()
