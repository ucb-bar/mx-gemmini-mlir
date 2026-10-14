"""Verify the archived compiler/source SmolVLA attention Spike comparison."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", choices=("compiled_pair", "compiled_pair_queued",
                                              "compiled_pair_pipeline"),
                        default="compiled_pair")
    args = parser.parse_args()
    archive = HERE / args.archive
    receipt = json.loads((archive / "receipt.json").read_text())
    schedule = {"compiled_pair": "fenced", "compiled_pair_queued": "queued",
                "compiled_pair_pipeline": "pipelined"}[args.archive]
    overlapped = schedule != "fenced"
    assert receipt["schema"] == (
        f"mx_gemmini.smolvla_masked_attention_paired_{schedule}.v1" if overlapped else
        "mx_gemmini.smolvla_masked_attention_paired.v1")
    assert receipt.get("compiler_schedule", "fenced") == schedule
    if overlapped:
        assert receipt["schedule_plan_sha256"] == sha(
            HERE / ("attention_queue.schedule.mlir" if schedule == "queued" else
                    "attention_pipeline.schedule.mlir"))
    assert receipt["scope"] == "first 15-head call of each key shape; head 0 only"
    assert receipt["is_stitched_stage_or_full_model"] is False
    assert receipt["spike_mode"] == "GEMMINI_MODE=both; no performance override"
    assert receipt["source_revision"] == "0344bcba70b899039ccd0ac0801ca5605d46d3eb"
    assert receipt["source_patch_sha256"] == sha(HERE / "attn_flash_smol.patch")
    assert receipt["driver_sha256"] == sha(
        HERE.parents[2] / "tools/fixtures" /
        ("smol_attention_compiler_first_driver.c" if schedule == "pipelined" else
         "smol_attention_paired_driver.c"))
    for name, digest in receipt["artifact_sha256"].items():
        assert sha(archive / name) == digest, name
    assert [row["logical_keys"] for row in receipt["rows"]] == [113, 163]
    for row in receipt["rows"]:
        keys = row["logical_keys"]
        folder = archive / f"keys{keys}"
        reference = json.loads((HERE / ("source_113_all_heads.json"
                                         if keys == 113 else
                                         "source_all_heads.json")).read_text())
        cutpoint = HERE / ("cutpoint113.npz" if keys == 113 else "cutpoint.npz")
        header = HERE / ("smol_head113.h" if keys == 113 else "smol_head0.h")
        frontend = HERE / f"attention{keys}.model2mlir.mlir"
        assert row["physical_shape"] == [64, 128 if keys == 113 else 192, 64]
        assert row["cutpoint_sha256"] == reference["cutpoint_sha256"] == sha(cutpoint)
        assert row["header_sha256"] == reference["rows"][0]["generated_header_sha256"] == sha(header)
        assert row["model2mlir_frontend_sha256"] == sha(frontend)
        assert row["typed_mlir_sha256"] == sha(folder / "attention.mx.mlir")
        assert row["issuer_source_sha256"] == sha(folder / "mx_issue.c")
        assert row["issuer_object_sha256"] == sha(folder / "mx_issue.o")
        assert row["expected_output_hash"] == reference["rows"][0]["output_hash"]
        assert row["same_output"] is True
        assert row["command_count"] == (167 if keys == 113 else 232)
        assert row["fence_count"] == ((0 if overlapped else 11) if keys == 113 else
                                      (0 if overlapped else 16))
        log = gzip.decompress((folder / "spike.log.gz").read_bytes()).decode()
        assert "attn_flash PASSED" in log
        assert "paired full BF16 output match 4096 values" in log
        if schedule == "pipelined":
            assert f"compiler first fresh guest hash {row['expected_output_hash']}" in log
            assert "compiler first full BF16 output match 4096 values" in log
        assert f"check O hash vs Spike: match ({row['expected_output_hash']})" in log
        matches = re.findall(
            r"^paired (pipeline|compiler) ([0-3]) cycles (\d+) hash ([0-9a-f]{16})$",
            log, re.M)
        assert [(kind, int(index)) for kind, index, _, _ in matches] == [
            ("pipeline", 0), ("compiler", 1),
            ("pipeline", 2), ("compiler", 3)]
        assert all(digest == row["expected_output_hash"]
                   for _, _, _, digest in matches)
        source = [int(matches[i][2]) for i in (0, 2)]
        compiler = [int(matches[i][2]) for i in (1, 3)]
        assert source == row["source_pipelined_cycles"]
        assert compiler == row[f"compiler_{schedule}_cycles"]
        assert abs((sum(compiler) / sum(source) - 1) -
                   row["compiler_over_source_fraction"]) < 1e-12
    print(f"SmolVLA masked attention {receipt.get('compiler_schedule', 'fenced')} "
          "compiler/source pairs verified")


if __name__ == "__main__":
    main()
