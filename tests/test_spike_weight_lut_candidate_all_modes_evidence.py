"""Validate isolated patched-Spike results without upgrading stock qualification."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_spike_weight_lut_candidate_all_modes_266c593"
STOCK_EXTENSION_CLOSURE = (
    "dbbc62171a9393691c575ec278dfdd75ca20fd6a2c9b52825380dad4e5d0af3f")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_isolated_spike_weight_lut_candidate_all_legal_modes() -> None:
    index = _read(EVIDENCE / "qualification.json")
    assert index["schema"] == "mx_gemmini.spike_weight_lut_candidate_all_modes.v1"
    assert "isolated patched Spike" in index["scope"]
    assert index["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert index["model2mlir_revision"] == (
        "e9ded36eb85abf2d9097ac4dc11457c825853388")
    assert index["compiler_revision"] == "0d7e31be2dbea83b06d2d700bdcb199fdbbc6e3a"
    assert index["spike_extension_revision"] == (
        "876631072ecc7cc8197df5dc69c4d26bd4cd5f54")
    assert index["extension_source_closure_sha256"] != STOCK_EXTENSION_CLOSURE
    assert _sha(ROOT / index["patch"]) == index["patch_sha256"]
    assert (index["candidate_passes"], index["stock_passes_unchanged"],
            index["stock_failures_repaired_in_candidate"]) == (108, 105, 3)
    rows = index["rows"]
    assert len(rows) == 108
    assert Counter(row["mesh_dim"] for row in rows) == {8: 36, 16: 36, 32: 36}
    assert len({(row["mesh_dim"], row["source_suffix"]) for row in rows}) == 108
    assert {(row["mesh_dim"], row["source_suffix"]) for row in rows
            if row["comparison"] == "stock_failure_repaired"} == {
                (8, "e4m3s_e4m3"), (16, "e4m3s_e4m3"), (32, "e4m3s_e4m3")}
    for row in rows:
        candidate_path = EVIDENCE / row["candidate_receipt"]
        log_path = EVIDENCE / row["candidate_log"]
        stock_path = ROOT / row["stock_receipt"]
        assert _sha(candidate_path) == row["candidate_receipt_sha256"]
        assert _sha(log_path) == row["candidate_log_sha256"]
        assert _sha(stock_path) == row["stock_receipt_sha256"]
        candidate, stock = _read(candidate_path), _read(stock_path)
        assert candidate["status"] == "source_golden_matched_on_pinned_spike"
        assert candidate["spike_exit_code"] == 0
        assert candidate["compared_bf16_outputs"] == 4096
        assert candidate["rtl_revision"] == index["rtl_revision"]
        assert candidate["compiler_revision"] == index["compiler_revision"]
        assert candidate["source_header_sha256"] == stock["source_header_sha256"]
        assert candidate.get("source_driver_sha256") == stock.get("source_driver_sha256")
        assert candidate.get("source_generation_manifest_sha256") == (
            stock.get("source_generation_manifest_sha256"))
        assert candidate["profile_sha256"] == stock["profile_sha256"]
        assert candidate["elf_sha256"] == stock["elf_sha256"] == row["elf_sha256"]
        assert candidate["spike_log_sha256"] == row["candidate_log_sha256"]
        assert candidate["extension_sha256"] != stock["extension_sha256"]
        closure = candidate.get("extension_source_closure_sha256",
                                candidate.get("gemmini_extension_source_closure_sha256"))
        assert closure == index["extension_source_closure_sha256"]
        stock_closure = stock.get("extension_source_closure_sha256",
                                  stock.get("gemmini_extension_source_closure_sha256"))
        assert stock_closure == STOCK_EXTENSION_CLOSURE
        if row["comparison"] == "stock_failure_repaired":
            assert stock["status"] == "source_golden_failed_on_pinned_spike"
            assert candidate["spike_log_sha256"] != stock["spike_log_sha256"]
        else:
            assert row["comparison"] == "stock_pass_unchanged"
            assert stock["status"] == "source_golden_matched_on_pinned_spike"
            assert candidate["spike_log_sha256"] == stock["spike_log_sha256"]
    for dim in (8, 16, 32):
        matrix = _read(EVIDENCE / f"dim{dim}/matrix_receipt.json")
        assert matrix["legal_mode_count"] == 36
        assert matrix["selected_modes"] == matrix["passed_modes"] == (
            35 if dim == 16 else 36)
        assert not matrix["selected_but_failed_compute"]
        assert matrix["profile_complete"] == (dim != 16)
        for mode in matrix["rows"]:
            assert _sha(EVIDENCE / f"dim{dim}" / mode["source_suffix"] /
                        "receipt.json") == mode["receipt_sha256"]
        if dim == 16:
            assert len(matrix["uncovered_legal_compute"]) == 1
            bound = EVIDENCE / "dim16/direct_e4m3/profile_bound.mlir"
            capture = _read(EVIDENCE / "dim16/direct_e4m3/capture_receipt.json")
            assert capture["model2mlir_revision"] == index["model2mlir_revision"]
            assert capture["target_binding"]["bound_mlir_sha256"] == _sha(bound)
            assert 'pe_mode = 8 : i32' in bound.read_text()
        else:
            assert matrix["uncovered_legal_compute"] == []
