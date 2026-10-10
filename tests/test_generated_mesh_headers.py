"""Pin the DIM8/DIM32 generated headers to Nicolas's model and profile gaps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FORMAT = {
    "fp4": ("fp4_e2m1", "direct"),
    "e2m3": ("fp6_e2m3", "lut"),
    "e3m2": ("fp6_e3m2", "lut"),
    "e4m3": ("fp8_e4m3", "lut"),
    "e4m3s": ("fp8_e4m3", "direct"),
    "e5m2": ("fp8_e5m2", "lut"),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("dim,baseline", [
    (8, "0ae643c6be0e288d9d9b3cb9c163ac8c669b0fb8941cee85cd1a7daa35c94b83"),
    (32, "13730bd97ec1306bb93d11f7cb7464dd5ce5b1b0a111f524c5679524c9a507b5"),
])
def test_generated_mesh_headers_cover_exact_missing_source_modes(
        dim: int, baseline: str) -> None:
    folder = ROOT / f"docs/evidence/nicolas_generated_mesh_dim{dim}_266c593"
    manifest = json.loads((folder / "generation_manifest.json").read_text())
    assert manifest["schema"] == "mx_gemmini.nicolas_generated_mesh_headers.v1"
    assert manifest["mesh_dim"] == dim
    assert manifest["rtl_revision"].startswith("266c593")
    assert manifest["software_revision"].startswith("350547f")
    assert manifest["microxcaling_revision"] == (
        "7bc41952de394f5cc5e782baf132e7c7542eb4e4")
    assert manifest["baseline_sha256"] == baseline
    assert manifest["wrapper_sha256"] == _sha(
        ROOT / "tools/generate_nicolas_mesh_headers.py")
    assert len(manifest["headers_sha256"]) == 15
    assert set(manifest["header_origin"]) == set(manifest["headers_sha256"])
    for name, digest in manifest["headers_sha256"].items():
        assert _sha(folder / f"matmul_data_asym_{name}_dim{dim}.h") == digest
        assert manifest["header_origin"][name] in {"checked_in", "generated"}

    profile = json.loads((ROOT / "profiles/gemmini-mx-cleanup-266c593" /
                          f"MxDim{dim}AllAsymGemminiRocketConfig.json").read_text())
    previous = json.loads((ROOT / "docs/evidence" /
                           f"nicolas_asym_matrix_dim{dim}_266c593/matrix_first.json").read_text())
    missing = {json.dumps(row["compute"], sort_keys=True)
               for row in previous["uncovered_legal_compute"]}
    assert len(missing) == 15
    selected = set()
    for name in manifest["headers_sha256"]:
        left, right = name.split("_")
        af, ap = FORMAT[left]
        wf, wp = FORMAT[right]
        cells = [cell for cell in profile["legal_compute"] if
                 (cell["activation_format"], cell["activation_projection"],
                  cell["weight_format"], cell["weight_projection"]) ==
                 (af, ap, wf, wp)]
        assert len(cells) == 1
        selected.add(json.dumps(cells[0], sort_keys=True))
    assert selected == missing


@pytest.mark.parametrize("dim,stock_mismatches", [(8, 4095), (32, 4096)])
def test_generated_mesh_spike_qualification_evidence(
        dim: int, stock_mismatches: int) -> None:
    folder = ROOT / f"docs/evidence/nicolas_generated_mesh_dim{dim}_266c593"
    header_manifest = json.loads((folder / "generation_manifest.json").read_text())
    profile = json.loads((ROOT / "profiles/gemmini-mx-cleanup-266c593" /
                          f"MxDim{dim}AllAsymGemminiRocketConfig.json").read_text())
    index = json.loads((folder / "qualification.json").read_text())
    first = json.loads((folder / "matrix_first.json").read_text())
    repro = json.loads((folder / "matrix_generated_repro.json").read_text())
    failure = "e4m3s_e4m3"

    assert index["schema"] == "mx_gemmini.nicolas_generated_mesh_mode_matrix.v1"
    assert (index["mesh_dim"], index["legal_mode_count"],
            index["checked_in_source_passes"],
            index["generated_mode_stock_spike_passes"],
            index["generated_mode_stock_spike_failures"],
            index["distinct_stock_spike_passes"]) == (dim, 36, 21, 14, 1, 35)
    assert index["generation_manifest_sha256"] == _sha(folder / "generation_manifest.json")
    assert index["full_matrix_sha256"] == _sha(folder / "matrix_first.json")
    assert index["generated_repro_matrix_sha256"] == _sha(folder / "matrix_generated_repro.json")
    assert (first["selected_modes"], first["passed_modes"],
            first["profile_complete"]) == (36, 35, False)
    assert (repro["selected_modes"], repro["passed_modes"],
            repro["profile_complete"]) == (15, 14, False)
    assert {json.dumps(row["compute"], sort_keys=True) for row in first["rows"]} == {
        json.dumps(cell, sort_keys=True) for cell in profile["legal_compute"]}
    assert {row["source_suffix"] for row in repro["rows"]} == set(
        header_manifest["headers_sha256"])
    assert len(index["rows"]) == 36
    assert [row["mode"] for row in index["rows"] if row["stock_status"] != "passed"] == [failure]
    assert index["unqualified_legal_compute"] == [next(
        row["compute"] for row in first["rows"] if row["source_suffix"] == failure)]

    first_rows = {row["source_suffix"]: row for row in first["rows"]}
    repro_rows = {row["source_suffix"]: row for row in repro["rows"]}
    for row in index["rows"]:
        name = row["mode"]
        first_path = folder / "first" / f"{name}.json"
        receipt = json.loads(first_path.read_text())
        assert _sha(first_path) == row["first_receipt_sha256"] == first_rows[name]["receipt_sha256"]
        assert receipt["compared_bf16_outputs"] == row["compared_bf16_outputs"] == 4096
        assert receipt["compiler_revision"] == index["compiler_revision"]
        assert receipt["rtl_revision"] == index["rtl_revision"]
        if name in header_manifest["headers_sha256"]:
            assert row["source_driver"] == "generated_mode"
            assert row["header_origin"] == header_manifest["header_origin"][name]
            assert receipt["source_header_sha256"] == header_manifest["headers_sha256"][name]
            assert receipt["source_generation_manifest_sha256"] == index[
                "generation_manifest_sha256"]
            repro_path = folder / "repro" / f"{name}.json"
            again = json.loads(repro_path.read_text())
            assert _sha(repro_path) == row["repro_receipt_sha256"] == repro_rows[name]["receipt_sha256"]
            receipt["build_log_sha256"].pop("link.log")
            again["build_log_sha256"].pop("link.log")
            assert receipt == again
            assert row["reproduction_exact_except_link_log"] is True
        else:
            assert row["source_driver"] == "checked_in_nicolas_test"
            assert row["repro_receipt_sha256"] is None
            assert row["reproduction_exact_except_link_log"] is None

    diagnostic_folder = folder / failure
    diagnostic = json.loads((diagnostic_folder / "patch_diagnostic.json").read_text())
    stock = json.loads((folder / "first" / f"{failure}.json").read_text())
    assert index["patch_diagnostic_sha256"] == _sha(
        diagnostic_folder / "patch_diagnostic.json")
    assert diagnostic["elf_sha256"] == stock["elf_sha256"]
    assert diagnostic["patch_sha256"] == _sha(ROOT / "docs/evidence/"
        "nicolas_generated_modes_266c593/spike_weight_lut_quad_candidate.patch")
    assert diagnostic["stock_spike_log_sha256"] == stock["spike_log_sha256"] == _sha(
        diagnostic_folder / "stock_spike.log")
    assert diagnostic["patched_spike_log_sha256"] == _sha(
        diagnostic_folder / "patched_spike.log")
    assert (diagnostic["stock_mismatches"], diagnostic["patched_mismatches"],
            diagnostic["compared_bf16_outputs"],
            diagnostic["stock_qualification"]) == (stock_mismatches, 0, 4096, "failed")
