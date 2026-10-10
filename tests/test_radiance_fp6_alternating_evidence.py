"""Check the archived source-bound 16-wave FP6 Spike qualification."""

from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_fp6_alternating_80f84ca"
CAPTURE = (ROOT / "docs/evidence/radiance_mx_gemm_latest_e9ded36_ee22/frontend/"
           "mxgemm.fp6.m128n128k2048.tm128tn128tk128.fullout/"
           "mx_gemm.profile_bound.mlir")
SERIAL = (ROOT / "docs/evidence/radiance_mx_gemm_latest_e9ded36_ee22/spike/"
          "mxgemm.fp6.m128n128k2048.tm128tn128tk128.fullout/build/"
          "physical_program.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_bound_alternating_fp6_evidence():
    index = json.loads((EVIDENCE / "index.json").read_text())
    receipt = json.loads((EVIDENCE / "artifact_manifest.json").read_text())
    physical = json.loads((EVIDENCE / "physical_program.json").read_text())
    assert index["status"] == "same_elf_failed_stock_and_matched_corrected_spike"
    assert index["source_kind"] == "radiance_checked_in_fp6_driver"
    assert index["radiance_revision"] == "80f84caedbabc663a7433c1da4455b936cca41f3"
    assert receipt["source_driver_sha256"] == index["radiance_driver_sha256"]
    assert receipt["source_header_sha256"] == index["radiance_header_sha256"]
    assert sha(CAPTURE) == index["model2mlir_profile_bound_capture_sha256"]
    assert sha(EVIDENCE / "payload_bound.mlir") == index["source_bound_mlir_sha256"]
    assert sha(EVIDENCE / "bundle_manifest.json") == index["source_bundle_manifest_sha256"]
    assert sha(EVIDENCE / "physical_program.json") == index["physical_program_sha256"]
    assert sha(EVIDENCE / "spike_fp6_scale_selector.patch") == index["model_patch_sha256"]
    assert sha(EVIDENCE / "mx_issue.c") == receipt["files_sha256"]["mx_issue.c"]
    assert sha(EVIDENCE / "stock_spike.log") == index["stock_spike_log_sha256"]
    assert sha(EVIDENCE / "corrected_spike.log") == index["corrected_spike_log_sha256"]
    assert receipt["status"] == "source_golden_matched_on_experimental_spike"
    assert receipt["experimental_spike_extension"] is True
    assert receipt["fp6_spike_scale_selector_workaround"] is False
    assert receipt["elf_sha256"] == index["elf_sha256"]
    assert receipt["extension_sha256"] == index["corrected_extension_sha256"]
    assert receipt["mode"] == physical["mode"] == "rtl_alternating"
    assert receipt["shape_mnk"] == physical["shape_mnk"] == [128, 128, 2048]
    assert receipt["compared_bf16_outputs"] == index["compared_bf16_outputs"] == 16384
    assert index["stock_bf16_mismatches"] == 16368
    assert index["corrected_bf16_mismatches"] == 0
    assert "lowered MX 128x128x2048: 16368 BF16 mismatches" in (
        EVIDENCE / "stock_spike.log").read_text()
    assert "lowered MX 128x128x2048: 0 BF16 mismatches" in (
        EVIDENCE / "corrected_spike.log").read_text()

    waves = physical["plan"]["waves"]
    assert index["k_waves"] == len(waves) == 16
    assert [wave["k_start"] for wave in waves] == list(range(0, 2048, 128))
    steps = physical["steps"]
    assert receipt["command_count"] == sum("funct" in step["command"] for step in steps)
    for wave in range(16):
        uploads = [step["command"] for step in steps if step["wave"] == wave and
                   step["phase"] == "upload_scales" and "funct" in step["command"]]
        assert {command["rs1"]["buffer"] for command in uploads} == {
            "activation_scales", "weight_scales"}
        assert len(uploads) == 2
        assert {((command["rs2"]["immediate"] >> 33) & 0x1fff) for command in uploads} == {
            (wave & 1) * 4096}
        selectors = [step["command"] for step in steps if step["wave"] == wave and
                     step["phase"] == "select_scales" and "funct" in step["command"]]
        assert len(selectors) == 1 and selectors[0]["funct"] == 26
        assert ((selectors[0]["rs1"]["or_bits"] >> 60) & 3) == (3 if wave & 1 else 0)

    serial = json.loads(SERIAL.read_text())
    assert len(serial["steps"]) == len(steps) == 1340
    changed = Counter((a["phase"], a["wave"]) for a, b in zip(steps, serial["steps"])
                      if a != b)
    assert changed == Counter({(phase, wave): count
                              for wave in range(1, 16, 2)
                              for phase, count in (("upload_scales", 2),
                                                   ("select_scales", 1))})
