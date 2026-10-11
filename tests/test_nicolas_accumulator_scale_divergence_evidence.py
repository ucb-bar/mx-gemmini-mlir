"""Pin the source-header versus hardware-constant scale comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATCHED = ROOT / "docs/evidence/nicolas_accumulator_candidate_spike_266c593/index.json"
DIVERGED = ROOT / "docs/evidence/nicolas_accumulator_constant_scale_divergence_266c593"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_constant_scales_diverge_under_same_candidate_spike_model() -> None:
    matched = json.loads(MATCHED.read_text())
    divergence = json.loads((DIVERGED / "index.json").read_text())
    assert divergence["schema"] == "mx_gemmini.nicolas_accumulator_scale_divergence_index.v1"
    assert divergence["scale_mode"] == "hardware_constant_0x7f"
    assert divergence["total_bf16_values_checked"] == 24_576
    assert divergence["total_bf16_mismatches"] == 24_516
    assert divergence["patch_sha256"] == matched["patch_sha256"]
    assert divergence["extension_sha256"] == matched["extension_sha256"]
    assert divergence["rtl_or_fpga_qualified"] is False
    assert divergence["source_hardware_scale_values_matched"] is True
    assert divergence["source_hardware_scale_transport_matched"] is False
    expected = {
        "fp8_64x64x64_dram_mvout_spike": 4096,
        "fp4_64x64x64_dram_mvout_spike": 4036,
        "fp8_128x128x256_dram_mvout_spike": 16_384,
    }
    by_case = {row["case"]: row for row in matched["cases"]}
    assert {row["case"] for row in divergence["cases"]} == set(expected)
    for row in divergence["cases"]:
        case = row["case"]
        baseline = by_case[case]
        log = DIVERGED / case / "spike.log"
        assert row["object_sha256"] == baseline["object_sha256"]
        assert row["source_driver_sha256"] == baseline["source_driver_sha256"]
        assert row["source_header_sha256"] == baseline["source_header_sha256"]
        assert row["extension_sha256"] == baseline["extension_sha256"]
        assert row["bf16_values_checked"] == baseline["bf16_values_checked"]
        assert baseline["mismatches"] == 0
        assert row["mismatches"] == expected[case]
        assert row["spike_exit_code"] != 0
        assert row["spike_log_sha256"] == _sha(log)
        assert f"{expected[case]} mismatches / {row['bf16_values_checked']} BF16 values" in log.read_text()
        assert row["rtl_or_fpga_qualified"] is False
