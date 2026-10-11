"""Pin executable public-object evidence for Nicolas's VPU operation classes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.qualify_nicolas_vpu_elementwise import KINDS


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_vpu_public_objects_266c593"
BASE = ROOT / "docs/evidence/nicolas_vpu_elementwise_compiled_266c593"
FUSED = ROOT / "docs/evidence/nicolas_vpu_fused_compiled_266c593"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_fourteen_public_vpu_objects_have_full_spike_checks() -> None:
    index = json.loads((EVIDENCE / "index.json").read_text())
    assert index["schema"] == "mx_gemmini.nicolas_vpu_public_object_replay.v1"
    assert index["status"] == "all_base_and_fused_vpu_public_objects_match_source_reference"
    assert index["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert index["compiler_revision"] == "4fa78deda76edda974f46d745bb3c6f632ef5786"
    assert index["baseline_index_sha256"] == _sha(BASE / "index.json")
    assert index["fused_baseline_index_sha256"] == _sha(FUSED / "index.json")
    assert index["total_output_bf16_compared"] == 6016
    assert index["total_sum_bf16_compared"] == 128
    assert {row["kind"] for row in index["rows"]} == set(KINDS) | {
        "expsub", "expsum"}
    assert len(index["rows"]) == 14

    for row in index["rows"]:
        kind = row["kind"]
        case = EVIDENCE / kind
        baseline = (FUSED if kind in {"expsub", "expsum"} else BASE) / kind
        manifest = json.loads((case / "object_manifest.json").read_text())
        dispatch = json.loads((case / "compile_manifest.json").read_text())
        assert row["status"] == "public_object_source_vpu_reference_matched_on_pinned_spike"
        assert row["bound_mlir_sha256"] == _sha(baseline / "bound.mlir")
        assert row["driver_sha256"] == _sha(baseline / "mx_driver.c")
        assert row["baseline_issuer_sha256"] == _sha(baseline / "mx_issue.c")
        assert row["issuer_c_sha256"] == _sha(case / "mx_issue.c")
        assert row["object_sha256"] == _sha(case / "mx_issue.o")
        assert row["object_manifest_sha256"] == _sha(case / "object_manifest.json")
        assert row["physical_program_sha256"] == _sha(case / "physical_program.json")
        assert row["elf_sha256"] == _sha(case / "vpu.elf")
        assert row["spike_log_sha256"] == _sha(case / "spike.log")
        assert row["mismatches"] == 0
        assert manifest["vpu_kind"] == kind
        assert manifest["transport"] == "rocket_rocc"
        assert manifest["allocated_data_section_bytes"] == 0
        assert manifest["embedded_operand_bytes"] == 0
        assert manifest["embedded_golden_bytes"] == 0
        assert manifest["object_sha256"] == row["object_sha256"]
        assert dispatch["lowering_family"] == "vpu_elementwise"
        assert dispatch["object_sha256"] == row["object_sha256"]
        assert len(dispatch["native_verifier_sha256"]) == 64
        if kind in {"expsub", "expsum"}:
            assert row["issuer_equal_baseline"] is False
            assert row["compared_output_bf16"] == 512
            assert (f"compiled fused {kind}: 0 output mismatches, "
                    "0 sum mismatches") in (case / "spike.log").read_text()
        else:
            assert row["issuer_equal_baseline"] is True
            assert row["compared_output_bf16"] in (128, 512)
            assert f"compiled VPU {kind}: 0 mismatches" in (
                case / "spike.log").read_text()
        assert row["compared_sum_bf16"] == (128 if kind == "expsum" else 0)
    expsub = next(row for row in index["rows"] if row["kind"] == "expsub")
    assert expsub["linked_driver_sha256"] == _sha(
        EVIDENCE / "expsub/linked_driver.c")
    assert expsub["linked_driver_sha256"] != expsub["driver_sha256"]
