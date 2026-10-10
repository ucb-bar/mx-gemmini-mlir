"""Bind scalar epilogues from digest-checked model2MLIR graphs."""

from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mx_gemmini_support.capture_epilogue import append_captured_tilewise_vpu_muls
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence"
CASES = (
    "radiance_tilewise_vpu_x2_266c593",
    "radiance_tilewise_vpu_scalar_266c593",
    "radiance_fp4_generated_tilewise_vpu_266c593",
    "radiance_fp4_generated_tilewise_vpu_scalar_266c593",
)


def _read(root: Path, name: str) -> bytes:
    path = root / name
    return path.read_bytes() if path.exists() else gzip.decompress(
        (root / f"{name}.gz").read_bytes())


def _case(name: str):
    root = EVIDENCE / name
    manifest, _ = load_bundle(root / "bundle")
    capture = SimpleNamespace(
        ok=True,
        capture_trace=json.loads(_read(root, "capture_trace.json")),
        quantization_manifest=json.loads(_read(root, "quantization_manifest.json")),
        mlir_text=_read(root, "frontend.mlir").decode())
    return root, manifest, capture


@pytest.mark.parametrize("name", CASES)
def test_captured_epilogue_recovers_archived_typed_mlir(name: str):
    root, manifest, capture = _case(name)
    profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                           "MxE4M3Fp4VpuGemminiRocketConfig.json")
    bound = append_captured_tilewise_vpu_muls(
        _read(root, "payload_bound.mlir").decode(), profile, manifest, capture)
    assert bound == _read(root, "tilewise_bound.mlir").decode()


def test_captured_epilogue_rejects_graph_and_frontend_drift():
    root, manifest, capture = _case("radiance_tilewise_vpu_scalar_266c593")
    profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                           "MxE4M3Fp4VpuGemminiRocketConfig.json")
    payload = _read(root, "payload_bound.mlir").decode()

    changed = deepcopy(capture)
    changed.capture_trace["graphs"]["original"]["nodes"][3]["args"][1] = 2.0
    with pytest.raises(ValueError):
        append_captured_tilewise_vpu_muls(payload, profile, manifest, changed)

    changed = deepcopy(capture)
    changed.capture_trace["graphs"]["original"]["nodes"][4]["args"] = []
    with pytest.raises(ValueError):
        append_captured_tilewise_vpu_muls(payload, profile, manifest, changed)

    changed = deepcopy(capture)
    changed.mlir_text += "\n"
    with pytest.raises(ValueError):
        append_captured_tilewise_vpu_muls(payload, profile, manifest, changed)

    changed = deepcopy(capture)
    changed.quantization_manifest["sites"][0]["format"] = "mxfp4"
    with pytest.raises(ValueError):
        append_captured_tilewise_vpu_muls(payload, profile, manifest, changed)


def test_general_source_cli_two_run_spike_receipts():
    root = EVIDENCE / "radiance_tilewise_vpu_scalar_266c593"
    archived = json.loads((root / "index.json").read_text())
    index = json.loads((root / "cli_capture_index.json").read_text())
    assert index["schema"] == "mx_gemmini.capture_derived_source_cli_spike.v1"
    assert index["status"] == "two_capture_derived_cli_runs_matched_on_pinned_spike"
    assert index["compiler_revision"] == "b5637a3425236134709955a624bcc2a86c13762b"
    assert index["capture_scalar_bf16"] == 0x3fc0
    assert index["compared_bf16_outputs"] == 65536
    assert index["bound_mlir_sha256"] == archived["files_sha256"]["tilewise_bound.mlir"]
    assert index["physical_program_sha256"] == archived[
        "files_sha256"]["build/physical_program.json"]
    assert index["generated_issue_sha256"] == archived["files_sha256"]["build/mx_issue.c"]
    assert index["elf_sha256"] == archived["elf_sha256"]
    assert index["spike_log_sha256"] == archived["files_sha256"]["build/spike.log"]
    assert index["source_index_sha256"] == hashlib.sha256(
        (root / "index.json").read_bytes()).hexdigest()
    receipts = []
    for name, digest in zip(("cli_capture_artifact_manifest.json",
                             "cli_capture_artifact_manifest_repro.json"),
                            index["artifact_manifest_sha256"]):
        data = (root / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        receipt = json.loads(data)
        assert receipt["compiler_revision"] == index["compiler_revision"]
        assert receipt["compiler_source_closure_sha256"] == index[
            "compiler_source_closure_sha256"]
        assert receipt["bound_mlir_sha256"] == index["bound_mlir_sha256"]
        assert receipt["elf_sha256"] == index["elf_sha256"]
        assert receipt["spike_log_sha256"] == index["spike_log_sha256"]
        assert receipt["status"] == "derived_vpu_golden_matched_on_pinned_spike"
        assert receipt["compared_bf16_outputs"] == 65536
        receipt.pop("build_log_sha256")
        receipts.append(receipt)
    assert receipts[0] == receipts[1]
