"""Reopen a measured compaction regression, ordinary fallback and retained win."""

import hashlib
import importlib.util
import io
import json
import struct
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / "artifacts.json").read_text())
    raw = (here / manifest["archive"]).read_bytes()
    assert len(raw) == manifest["archive_bytes"] and sha(raw) == manifest["archive_sha256"]
    data = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts and member.name not in data
            data[member.name] = archive.extractfile(member).read()
    assert set(data) == set(manifest["members"])
    for name, raw in data.items():
        assert manifest["members"][name] == {"sha256": sha(raw), "bytes": len(raw)}
    reader_path = here.parent / "broadcast_multiply_20261019/verify.py"
    assert sha(reader_path.read_bytes()) == manifest["elf_reader_sha256"]
    spec = importlib.util.spec_from_file_location("retained_elf_reader", reader_path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)

    def load(name):
        return json.loads(data[name])

    def counters(console):
        result = reader.counters(console)
        summary, = [line for line in console.splitlines() if line.startswith("gemmini perf [e4m3_vpu]:")]
        result["loops"] = int(summary.split(": ", 1)[1].split(", ")[1].split()[0])
        mesh, = [line.split() for line in console.splitlines() if line.strip().startswith("mesh:")]
        result.update(mesh_tiles=int(mesh[1]), mesh_busy_cycles=int(mesh[3]), vpu_commands=int(mesh[7]))
        return result

    profile = load("profile.json")
    profile_sha = sha((json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n").encode())
    reports = {name: load("comparisons/" + name + ".json") for name in ["regression", "live_guarded", "private_guarded"]}
    for label in ["live", "private"]:
        prefix = "captures/" + label + "/capture/"
        for name, identity in load(prefix + "capture_receipt.json")["artifacts"].items():
            assert identity == {"bytes": len(data[prefix + name]), "sha256": sha(data[prefix + name])}
    with np.load(io.BytesIO(data["captures/live/capture/inputs.npz"]), allow_pickle=False) as inputs:
        rhs = inputs["in1"].astype("<f4", copy=False).view("<u4")
        assert rhs.shape == (2, 23, 37)
        expected_rhs = np.repeat(rhs, 3, axis=0).transpose(0, 2, 1).tobytes()
    assert expected_rhs == data["reference/full_rhs.bin"] and len(expected_rhs) == 20424
    bindings = {"live_reuse": ("regression", "reused"), "live_unguarded": ("regression", "unguarded"),
                "live_guarded": ("live_guarded", "guarded"), "private_reuse": ("private_guarded", "reused"),
                "private_guarded": ("private_guarded", "guarded")}
    outputs, device_work, stages = {}, {}, {}
    for label, (report_name, role) in bindings.items():
        report = reports[report_name]
        row = report["results"][role]
        prefix = "stages/" + label + "/"
        stage = stages[label] = load(prefix + "mx_stage_probe.json")
        capture_prefix = "captures/" + ("live" if label.startswith("live") else "private") + "/capture/"
        assert report["capture_receipt_sha256"] == sha(data[capture_prefix + "capture_receipt.json"])
        assert report["identity"]["profile_sha256"] == profile_sha and report["all_words_exact"]
        assert stage["execution_policy"] == report["identity"]["execution_policy"]
        assert stage["spike"]["exit_code"] == 0 and stage["spike"]["done"] and not stage["op_profile"]
        assert stage["spike"]["environment"] == report["identity"]["environment"]
        assert stage["capture_model_sha256"] == sha(data[capture_prefix + "model.mlir"])
        assert stage["checkpoint_weights_sha256"] == sha(data[capture_prefix + "weights.safetensors"])
        assert sha(data[prefix + "mx_stage_probe.json"]) == row["stage_receipt_sha256"]
        image = data[prefix + "model.elf"]
        reader.sections(image)
        assert sha(image) == row["elf_sha256"] and len(image) == row["elf_bytes"]
        recipe = load(prefix + "compilation_recipe.json")
        assert recipe["status"] == "completed" and recipe["executable"]["sha256"] == sha(image)
        admissions = [name for name in data if name.startswith(prefix) and name.endswith("/policy/admission.json")]
        assert len(admissions) == 1 and sha(data[admissions[0]]) == row["admission_sha256"]
        assert load(admissions[0])["status"] == "accepted" and load(admissions[0])["elf_sha256"] == sha(image)
        console = data[prefix + "spike.log"]
        assert sha(console) == row["console_sha256"]
        lines = console.decode().splitlines()
        observed, = [line.split() for line in lines if line.startswith("OUT ")]
        assert int(observed[1]) == len(observed[2:]) == row["output_elements"] == 2622
        words = struct.pack("<" + "I" * 2622, *map(int, observed[2:]))
        assert sha(words) == row["output_sha256"]
        typed, = [line.split() for line in lines if line.startswith("OUT_SHA256 ")]
        assert typed[1:] == ["f32le", "2622", "10488", sha(words)]
        indexed = [line.split() for line in lines if line.startswith("OUT_SHA256_INDEX ")]
        expected = [["0", "rawbytes", "2622", "10488", sha(words)]]
        if label.startswith("live"):
            expected.append(["1", "rawbytes", "5106", "20424", sha(expected_rhs)])
            reference = report["additional_output_references"]["1"]
            assert reference["sha256"] == sha(expected_rhs) and reference["bytes"] == len(expected_rhs)
            assert row["additional_output_observations"][0]["sha256"] == sha(expected_rhs)
            assert report["observation_scope"] == "first_output_full_console_words_and_all_additional_full_reference_hashes"
        assert [fields[1:] for fields in indexed] == expected
        metrics = dict(line.split()[1:] for line in lines if line.startswith("METRIC "))
        assert int(metrics["cycles"]) == row["cycles"] and metrics["build_hash"] == stage["build_hash"] == row["build_hash"]
        assert metrics["memref_rank_mismatch"] == "0" and lines.count("DONE") == 1
        assert stage["arena_estimate"] == row["arena_estimate"]
        catalog = load(prefix + "device_catalog/mx_catalog.json")
        assert sha(data[prefix + "device_catalog/mx_catalog.o"]) == catalog["compilation"]["object_sha256"]
        for name, digest in catalog["compilation"]["source_sha256"].items():
            filename = {"runtime_group_prepare": "runtime_group_prepare.h", "runtime_quant_header": "runtime_quant.h",
                        "runtime_quant_lut": "runtime_quant_lut.h", "checkpoint_weights": "checkpoint_weights.S"}.get(
                            name, name + ".c")
            assert sha(data[prefix + "device_catalog/" + filename]) == digest
        if label.endswith("guarded") and "unguarded" not in label:
            rewrite = load(prefix + "device_host_abi/mx_batch_rhs_reuse_rewrite.json")
            assert catalog["compact_profitability_guard"] == rewrite["compaction_cost_guard"] == "private_materialization"
            assert rewrite["compact_calls"] == int(label == "private_guarded")
            assert rewrite["ordinary_reuse_fallback_calls"] == int(label == "live_guarded")
            assert rewrite["selected_calls"] == 1
            assert sha(data[prefix + "device_host_abi/mx_batch_rhs_reuse.mlir"]) == rewrite["output_sha256"]
            assert sha(data[prefix + "device_catalog/mx_catalog.json"]) == rewrite["catalog_sha256"]
            assert sha(data[prefix + "device_signatures.json"]) == rewrite["sidecar_sha256"]
        outputs[label] = words
        device_work[label] = counters(console.decode())
        assert all(row["perf"][key] == value for key, value in device_work[label].items() if key in row["perf"])
        print(label, row["cycles"], "cycles")
    assert len(set(outputs.values())) == 1
    assert device_work["live_reuse"] == device_work["live_unguarded"] == device_work["live_guarded"]
    assert device_work["private_reuse"] == device_work["private_guarded"]
    assert reports["regression"]["results"]["unguarded"]["cycles"] > reports["regression"]["results"]["reused"]["cycles"]
    assert reports["live_guarded"]["results"]["guarded"]["cycles"] == reports["regression"]["results"]["reused"]["cycles"]
    assert reports["private_guarded"]["results"]["guarded"]["cycles"] < reports["private_guarded"]["results"]["reused"]["cycles"]
    for name, result in load("negative_checks/results.json").items():
        assert result == {"exit_code": 1, "output_absent": True}
        message = "every additional output" if name == "missing" else "complete additional output"
        assert message in data["negative_checks/" + name + ".log"].decode()
    changed = data["negative_checks/changed_rhs_reference.bin"]
    assert len(changed) == len(expected_rhs) and sha(changed) != sha(expected_rhs)
    for name, row in load("diagnostics/private_guard.json").items():
        assert row["source_sha256"] == sha(data["diagnostics/" + name + ".mlir"])
        assert row["proved_calls"] == row["private_compact_calls"] == len(row["proofs"]) == 44
        assert row["scope"] == "current_source_use_closure_only_not_new_compiled_or_timing_evidence"
    print("PASS:", len(data), "files, five actual ELFs, 13110 first-output words and three complete live-RHS hash checks")


if __name__ == "__main__":
    main()
