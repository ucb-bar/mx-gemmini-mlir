"""A live secondary output needs its complete independent byte reference."""

import hashlib
import json

import numpy as np
import pytest

from tools.compare_shared_preparation import compare


def fixture(tmp_path, monkeypatch, *, changed_secondary=False):
    from merlin.common import mlir_query

    monkeypatch.setattr(mlir_query, "forward_signature", lambda _source: ([], [([2], "f32"), ([2], "f32")]))
    stage = tmp_path / "capture"
    stage.mkdir()
    for name in ["model.mlir", "weights.safetensors", "inputs.npz"]:
        (stage / name).write_bytes(name.encode())
    np.save(stage / "golden.npy", np.asarray([1, 2], dtype="<f4"))

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    capture = {"artifacts": {name: {"sha256": sha(stage / name)}
                             for name in ["model.mlir", "weights.safetensors", "inputs.npz", "golden.npy"]}}
    (stage / "capture_receipt.json").write_text(json.dumps(capture))
    extension = tmp_path / "extension"
    extension.write_bytes(b"fixture extension identity")
    first = hashlib.sha256(np.asarray([1, 2], dtype="<f4").tobytes()).hexdigest()
    reference = tmp_path / "secondary.bin"
    reference.write_bytes(np.asarray([3, 4], dtype="<f4").tobytes())
    builds = {}
    for label in ["baseline", "candidate"]:
        build = tmp_path / label
        build.mkdir()
        image = build / "model.elf"
        image.write_bytes(b"fixture executable identity")
        (build / "compilation_recipe.json").write_text(json.dumps({
            "status": "completed", "executable": {"sha256": sha(image)}}))
        admission = build / "invocations/unit/policy/admission.json"
        admission.parent.mkdir(parents=True)
        admission.write_text(json.dumps({"status": "accepted", "elf_sha256": sha(image)}))
        receipt = {"spike": {"done": True, "exit_code": 0, "environment": {}, "command": ["spike", "--extlib=" + str(extension)],
                             "extension_sha256": sha(extension)},
                   "profile_sha256": "fixture-profile", "execution_policy": {}, "spike_isa": "rv64gc_zicntr",
                   "op_profile": False, "build_hash": "123456789abc", "first_output_sha256_enabled": True,
                   "capture_model_sha256": sha(stage / "model.mlir"),
                   "checkpoint_weights_sha256": sha(stage / "weights.safetensors"), "arena_estimate": {}}
        (build / "mx_stage_probe.json").write_text(json.dumps(receipt))
        observed = np.asarray([3, 5 if changed_secondary and label == "candidate" else 4], dtype="<f4").tobytes()
        second = hashlib.sha256(observed).hexdigest()
        (build / "spike.log").write_text(
            f"OUT 2 1065353216 1073741824\nOUT_SHA256 f32le 2 8 {first}\n"
            f"OUT_SHA256_INDEX 0 rawbytes 2 8 {first}\nOUT_SHA256_INDEX 1 rawbytes 2 8 {second}\n"
            "METRIC cycles 100\nMETRIC build_hash 123456789abc\nMETRIC memref_rank_mismatch 0\nDONE\n"
            "gemmini perf [e4m3_vpu]: 1 commands, 1 loops, 1 fences;\n"
            "  mesh: 1 tiles, 16 busy cycles; VPU: 0 commands\n  load: 64 bytes; scales: 2 bytes\n")
        builds[label] = build
    return stage, builds, reference


def test_secondary_output_cannot_be_silently_ignored(tmp_path, monkeypatch):
    stage, builds, _ = fixture(tmp_path, monkeypatch)
    output = tmp_path / "comparison.json"
    with pytest.raises(ValueError, match="every additional output"):
        compare(stage, builds, output)
    assert not output.exists()


@pytest.mark.parametrize("change", ["candidate", "reference", "extent"])
def test_secondary_output_or_reference_change_refuses_before_report(tmp_path, monkeypatch, change):
    stage, builds, reference = fixture(tmp_path, monkeypatch, changed_secondary=change == "candidate")
    if change in {"reference", "extent"}:
        reference.write_bytes(b"changed!" if change == "reference" else b"tiny")
    output = tmp_path / "comparison.json"
    with pytest.raises(ValueError, match="complete additional output"):
        compare(stage, builds, output, additional_outputs={1: reference})
    assert not output.exists()


def test_complete_first_words_and_complete_secondary_reference_are_both_checked(tmp_path, monkeypatch):
    stage, builds, reference = fixture(tmp_path, monkeypatch)
    report = compare(stage, builds, tmp_path / "comparison.json", additional_outputs={1: reference})
    assert report["all_words_exact"] and report["additional_output_references"][1]["bytes"] == 8
    assert all(row["additional_output_observations"][0]["index"] == 1 for row in report["results"].values())
