"""Summarize pinned Radiance MX kernel coverage without mixing simulators."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "docs/evidence"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(relative: str) -> tuple[dict, str]:
    path = EVIDENCE / relative
    return json.loads(path.read_text()), digest(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radiance-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ws, ws_hash = load("radiance_ws_read_once_1a7bc9c/index.json")
    restream, restream_hash = load("radiance_restream_1df2c5f/index.json")
    roster, roster_hash = load("radiance_mx_vpu_legal_roster_266c593/index.json")
    gemv, gemv_hash = load("radiance_batched_gemv_552a16b/index.json")
    gqa, gqa_hash = load("radiance_gqa_pv_roster_80f84ca/index.json")
    assert ws["schema"] == "mx_gemmini.radiance_ws_read_once_roster.v1"
    assert restream["schema"] == "mx_gemmini.radiance_restream_source_parity.v1"
    assert roster["schema"] == "mx_gemmini.radiance_vpu_legal_roster_spike_matrix.v1"
    assert roster["profile_name"] == "MxE4M3Fp4VpuGemminiRocketConfig"
    assert roster["legal_precisions"] == ["FP4", "FP8"]
    assert roster["case_count_per_run"] == 24
    assert len(gemv["rows"]) == 4
    assert gqa["compared_bf16_outputs"] == 65536

    source_root = args.radiance_root.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    rows = []
    for entry in ws["rows"]:
        source = source_root / "kernels" / entry["directory"] / "kernel.cpp"
        assert digest(source) == entry["driver_sha256"], source
        assert entry["compared_bf16_outputs"] == 16384
        assert entry["weight_bytes_covered_once"] > 0
        rows.append({
            "source": f"kernels/{entry['directory']}/kernel.cpp",
            "source_sha256": entry["driver_sha256"],
            "precision": entry["precision"],
            "shape_mnk": entry["shape_mnk"],
            "compiler_spike_bf16_values_matching_source_golden":
                entry["compared_bf16_outputs"],
            "compiler_weight_tile_reads": entry["weight_tile_reads"],
            "compiler_weight_bytes": entry["weight_bytes_covered_once"],
            "radiance_source_schedule": "read_once_weight_stationary",
        })

    source = source_root / "kernels/gemm_mxgemmini_ws_restream/kernel.cpp"
    assert digest(source) == restream["driver_sha256"], source
    assert restream["shape_mnk"] == rows[0]["shape_mnk"]
    assert restream["m_tiles"] == 4
    assert restream["weight_bytes_transferred"] == 4 * rows[0]["compiler_weight_bytes"]
    assert restream["weight_tile_reads"] == 4 * rows[0]["compiler_weight_tile_reads"]
    rows.append({
        "source": "kernels/gemm_mxgemmini_ws_restream/kernel.cpp",
        "source_sha256": restream["driver_sha256"],
        "precision": "fp8",
        "shape_mnk": restream["shape_mnk"],
        "compiler_spike_bf16_values_matching_source_golden":
            restream["compared_bf16_outputs"],
        "compiler_weight_tile_reads": restream["weight_tile_reads"],
        "compiler_weight_bytes": restream["weight_bytes_transferred"],
        "radiance_source_schedule": "four_pass_weight_restream",
    })
    gemv_rows = []
    for entry in gemv["rows"]:
        source = source_root / "kernels" / entry["directory"] / "kernel.cpp"
        assert digest(source) == entry["driver_sha256"], source
        gemv_rows.append({
            "source": f"kernels/{entry['directory']}/kernel.cpp",
            "source_sha256": entry["driver_sha256"],
            "shape_mnk": entry["shape_mnk"],
            "precision": entry["precision"],
            "compiler_spike_bf16_values_matching_source_golden":
                entry["compared_bf16_outputs"],
        })
    gqa_source = source_root / "kernels/flash_attention_mx_gqa/kernel.cpp"
    assert digest(gqa_source) == gqa["source_kernel_sha256"], gqa_source

    result = {
        "schema": "mx_gemmini.radiance_kernel_comparison.v1",
        "radiance_revision": revision,
        "profile": roster["profile_name"],
        "source_execution": "Radiance Muon plus MX gateway",
        "compiler_execution": "standalone Rocket RoCC on Nicolas Spike",
        "paired_source_compiler_cycles": False,
        "cycle_ratio_valid": False,
        "radiance_vpu_profile_roster_cases_per_run": roster["case_count_per_run"],
        "radiance_vpu_profile_roster_runs": 2,
        "evidence": {
            "read_once_index_sha256": ws_hash,
            "restream_index_sha256": restream_hash,
            "legal_roster_index_sha256": roster_hash,
            "batched_gemv_index_sha256": gemv_hash,
            "gqa_pv_index_sha256": gqa_hash,
        },
        "rows": rows,
        "batched_gemv_rows": gemv_rows,
        "gqa_source_sha256": gqa["source_kernel_sha256"],
        "gqa_compiler_spike_pv_bf16_values_matched":
            gqa["compared_bf16_outputs"],
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
