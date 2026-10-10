"""Check the latest 31-driver model2MLIR-to-Spike source parity archive."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


EVIDENCE = (Path(__file__).resolve().parents[1] / "docs/evidence/"
            "radiance_mx_gemm_latest_e9ded36_ee22")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_latest_full_radiance_mx_roster_reproduces() -> None:
    front = [_read(EVIDENCE / name / "index.json") for name in
             ("frontend", "frontend_repro")]
    spike = [_read(EVIDENCE / name / "index.json") for name in
             ("spike", "spike_repro")]
    for index in front + spike:
        assert index["source_revision"] == (
            "ee22e0b87180436cd0fa1583c411a3cf49d7586a")
        assert index["model2mlir_revision"] == (
            "e9ded36eb85abf2d9097ac4dc11457c825853388")
        assert index["rtl_revision"] == (
            "266c593f2cb51d7e3fe83fc0317072b585ac3c52")
        assert (index["fullout_drivers"], index["requant_drivers"]) == (23, 8)
        assert len(index["rows"]) == 31
    assert spike[0]["compiler_revision"] == spike[1]["compiler_revision"]
    headers = _read(EVIDENCE / "header_materialization.json")
    assert headers["schema"] == "mx_gemmini.radiance_header_materialization.v1"
    assert headers["drivers"] == 31
    assert len(headers["generated_headers"]) == 19
    assert len(headers["rows"]) == 31
    for name, qualification, capture in zip(
            ("spike", "spike_repro"), spike,
            ("frontend", "frontend_repro")):
        assert qualification["frontend_index_sha256"] == _sha(
            EVIDENCE / capture / "index.json")

    for position in range(31):
        captures = [index["rows"][position] for index in front]
        qualifications = [index["rows"][position] for index in spike]
        name = captures[0]["driver"]
        assert all(row["driver"] == name for row in captures + qualifications)
        assert headers["rows"][position]["driver"] == name
        assert headers["rows"][position]["driver_sha256"] == (
            captures[0]["driver_sha256"])
        assert headers["rows"][position]["header_sha256"] == (
            captures[0]["header_sha256"])
        assert all(row["site_id"] == "functional:matmul" for row in captures)
        for key in ("driver_sha256", "header_sha256", "source_mlir_sha256",
                    "bound_mlir_sha256", "profile_sha256", "shape_mnk",
                    "tile_mnk", "precision", "quant_output"):
            assert captures[0][key] == captures[1][key]
        case = Path(name).stem
        receipts = []
        for folder, row, qualified in zip(("frontend", "frontend_repro"),
                                          captures, qualifications):
            base = EVIDENCE / folder
            receipt_path = base / row["receipt"]
            assert _sha(receipt_path) == row["receipt_sha256"]
            assert qualified["frontend_receipt_sha256"] == row["receipt_sha256"]
            receipt = _read(receipt_path)
            assert receipt["status"] == "source_shape_frontend_handoff_only"
            assert receipt["mx_support_revision"] == (
                front[0 if folder == "frontend" else 1]["compiler_revision"])
            assert receipt["source_driver_sha256"] == row["driver_sha256"]
            assert receipt["source_data_header_sha256"] == row["header_sha256"]
            assert not receipt["opaque_calls"]
            receipts.append(receipt)
            if folder == "frontend":
                assert _sha(base / case / "mx_gemm.model2mlir.mlir") == (
                    row["source_mlir_sha256"])
                assert _sha(base / case / "mx_gemm.handoff.mlir") == (
                    receipt["handoff_mlir_sha256"])
                assert _sha(base / case / "mx_gemm.profile_bound.mlir") == (
                    row["bound_mlir_sha256"])
                assert _sha(base / case / "quantization_manifest.json") == (
                    receipt["quantization_manifest_sha256"])
        for receipt in receipts:
            receipt.pop("mx_support_revision")
        assert receipts[0] == receipts[1]

        manifests = []
        for folder, row in zip(("spike", "spike_repro"), qualifications):
            base = EVIDENCE / folder
            receipt_path = base / row["receipt"]
            assert _sha(receipt_path) == row["receipt_sha256"]
            manifest = _read(receipt_path)
            manifests.append(manifest)
            assert manifest["status"] == row["status"]
            assert manifest["spike_exit_code"] == 0
            assert manifest["rtl_revision"] == spike[0]["rtl_revision"]
            assert manifest["source_driver_sha256"] == captures[0]["driver_sha256"]
            assert manifest["source_header_sha256"] == captures[0]["header_sha256"]
            assert manifest["profile_sha256"] == captures[0]["profile_sha256"]
            assert manifest["shape_mnk"] == captures[0]["shape_mnk"]
            assert manifest[row["comparison"]] == row["compared_count"]
            m, n, _ = row["shape_mnk"]
            assert row["compared_count"] == m * n // (
                2 if row["comparison"] == "compared_source_fp6_packed_bytes" else 1)
            if captures[0]["quant_output"]:
                assert manifest["compared_source_e8m0_scales"] == m * n // 32
        for key in ("bound_mlir_sha256", "files_sha256", "object_sha256",
                    "extension_sha256", "elf_sha256", "spike_log_sha256"):
            assert manifests[0][key] == manifests[1][key]
        base = EVIDENCE / "spike" / case
        assert _sha(base / "payload_bound.mlir") == manifests[0]["bound_mlir_sha256"]
        assert _sha(base / "build/spike.log") == manifests[0]["spike_log_sha256"]
        for filename in ("physical_program.json", "mx_issue.c", "mx_driver.c",
                         "mx_data.S"):
            assert _sha(base / "build" / filename) == (
                manifests[0]["files_sha256"][filename])


def test_one_command_reproduces_archived_radiance_mx_roster() -> None:
    run = EVIDENCE / "one_command_66df445"
    report = _read(run / "reproduction.json")
    assert report["schema"] == "mx_gemmini.radiance_mx_gemm_reproduction.v1"
    assert report["status"] == "all_31_source_goldens_reproduced_on_pinned_spike"
    assert report["compiler_revision"] == "66df445cfa0d1fb1e4a6fdc9cca3721681e6af5a"
    assert (report["covered_drivers"], report["fullout_drivers"],
            report["requant_drivers"]) == (31, 23, 8)
    assert report["baseline_frontend_index_sha256"] == _sha(
        EVIDENCE / "frontend/index.json")
    assert report["baseline_spike_index_sha256"] == _sha(
        EVIDENCE / "spike/index.json")
    assert report["frontend_index_sha256"] == _sha(run / "frontend/index.json")
    assert report["spike_index_sha256"] == _sha(run / "spike/index.json")
    headers = _read(run / "headers.json")
    assert headers["drivers"] == 31
    assert len(headers["rows"]) == 31
    assert headers["generated_headers"] == []
    frontend, spike = (_read(run / f"{section}/index.json") for section in
                       ("frontend", "spike"))
    expected_front, expected_spike = (_read(EVIDENCE / f"{section}/index.json")
                                      for section in ("frontend", "spike"))
    assert len(frontend["rows"]) == len(spike["rows"]) == 31
    assert frontend["compiler_revision"] == spike["compiler_revision"] == (
        report["compiler_revision"])
    assert spike["frontend_index_sha256"] == report["frontend_index_sha256"]
    for row, expected, header in zip(frontend["rows"], expected_front["rows"],
                                     headers["rows"]):
        assert row["driver"] == expected["driver"] == header["driver"]
        assert header["driver_sha256"] == row["driver_sha256"]
        assert header["header_sha256"] == row["header_sha256"]
        for key in ("source_mlir_sha256", "bound_mlir_sha256", "profile_sha256",
                    "shape_mnk", "tile_mnk", "quant_output"):
            assert row[key] == expected[key]
        receipt_path = run / "frontend" / row["receipt"]
        assert _sha(receipt_path) == row["receipt_sha256"]
        assert _read(receipt_path)["mx_support_revision"] == report["compiler_revision"]
    for row, expected in zip(spike["rows"], expected_spike["rows"]):
        assert row["driver"] == expected["driver"]
        for key in ("profile_bound_mlir_sha256", "payload_bound_mlir_sha256",
                    "elf_sha256", "spike_log_sha256", "comparison",
                    "compared_count", "status"):
            assert row[key] == expected[key]
        receipt_path = run / "spike" / row["receipt"]
        assert _sha(receipt_path) == row["receipt_sha256"]
        receipt = _read(receipt_path)
        baseline = _read(EVIDENCE / "spike" / expected["receipt"])
        assert receipt["compiler_revision"] == report["compiler_revision"]
        for key in ("files_sha256", "object_sha256", "extension_sha256",
                    "elf_sha256", "spike_log_sha256", "bound_mlir_sha256"):
            assert receipt[key] == baseline[key]
