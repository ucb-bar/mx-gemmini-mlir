"""Keep the candidate Spike replay separate from RTL or stock-Spike claims."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.compile_nicolas_accumulator_readout import CASES_HARDWARE
from tools.qualify_nicolas_plain_matrix_object import CASES, RTL_REVISION


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_accumulator_candidate_spike_266c593"
OBJECTS = ROOT / "docs/evidence/nicolas_accumulator_readout_objects_266c593"
PATCH = ROOT / "tools/patches/nicolas_spike_mx_accumulator_candidate_266c593.patch"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_candidate_replay_has_full_output_proof_without_hardware_claim() -> None:
    index = json.loads((EVIDENCE / "index.json").read_text())
    assert index["schema"] == "mx_gemmini.nicolas_accumulator_candidate_index.v1"
    assert index["rtl_revision"] == RTL_REVISION
    assert index["patch_sha256"] == _sha(PATCH)
    assert index["total_bf16_values_checked"] == 24_576
    assert index["status"] == "experimental_accumulator_spike_full_bf16_matched"
    assert index["rtl_or_fpga_qualified"] is False
    assert index["source_hardware_scale_semantics_matched"] is False
    assert {row["case"] for row in index["cases"]} == set(CASES_HARDWARE)

    for row in index["cases"]:
        case = row["case"]
        source = CASES[case]
        log = EVIDENCE / case / "spike.log"
        object_file = OBJECTS / case / "object/mx_issue.o"
        manifest = OBJECTS / case / "object/object_manifest.json"
        assert row["status"] == index["status"]
        assert row["source_driver_sha256"] == source.source_sha256
        assert row["source_header_sha256"] == source.header_sha256
        assert row["object_sha256"] == _sha(object_file)
        assert row["object_manifest_sha256"] == _sha(manifest)
        assert row["extension_sha256"] == index["extension_sha256"]
        assert row["spike_log_sha256"] == _sha(log)
        assert row["bf16_values_checked"] == source.shape[0] * source.shape[1]
        assert row["mismatches"] == 0
        assert f"0 mismatches / {row['bf16_values_checked']} BF16 values" in log.read_text()
        assert row["rtl_or_fpga_qualified"] is False
        assert row["source_hardware_scale_semantics_matched"] is False
