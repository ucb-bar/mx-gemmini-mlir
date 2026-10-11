"""The complete generated command replay must retain its Spike parity receipt."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_attn_vpu_generated_io_e9ded36_0344bcb"
SOURCE = ROOT / "docs/evidence/nicolas_attn_vpu_source_baseline_e9ded36_0344bcb"
COMPUTE = ROOT / "docs/evidence/nicolas_attn_vpu_generated_compute_e9ded36_0344bcb"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generated_attention_io_matches_source_on_spike() -> None:
    receipt = json.loads((ARCHIVE / "receipt.json").read_text())
    source = json.loads((SOURCE / "receipt.json").read_text())
    compute = json.loads((COMPUTE / "receipt.json").read_text())
    vector = json.loads((ARCHIVE / "vector_compile_manifest.json").read_text())
    assert receipt["stage_hashes"] == source["source_stage_hashes"]
    assert receipt["source_sha256"] == source["npu_source_sha256"]
    assert receipt["frontend_mlir_sha256"] == source["frontend_mlir_sha256"]
    assert receipt["header_sha256"] == source["generated_header_sha256"]
    assert receipt["matrix_object_sha256"] == compute["matrix_object_sha256"]
    assert receipt["vector_object_sha256"] == compute["vector_object_sha256"]
    assert vector["lowering_family"] == "attention_vector"
    assert vector["object_sha256"] == receipt["vector_object_sha256"]
    assert receipt["driver_sha256"] == _sha(ARCHIVE / "attn_vpu_generated_io.c")
    assert receipt["elf_sha256"] == _sha(ARCHIVE / "attn_vpu_generated_io.elf")
    assert receipt["spike_log_sha256"] == _sha(ARCHIVE / "spike.log")
    assert receipt["spike_reproduced_runs"] == 2
    assert set(receipt["io_object_sha256"]) == {
        "scales", "operands", "score", "probability", "probability_codes", "output"}
    for name, digest in receipt["io_object_sha256"].items():
        assert digest == _sha(ARCHIVE / f"{name}_issue.o")
        assert f"void mx_issue_{name}(" in (ARCHIVE / f"{name}_issue.c").read_text()
    driver = (ARCHIVE / "attn_vpu_generated_io.c").read_text()
    for source_call in ("static void mm(", "mvin_A(", "mvin_B(", "mvout_rows(",
                        "gemmini_vpu_", "gemmini_spad_requant(",
                        "gemmini_mx_load_scales(", "gemmini_mx_load_scales_2d("):
        assert source_call not in driver
    assert "gemmini_flush(0);" in driver and "gemmini_fence();" in driver
    log = (ARCHIVE / "spike.log").read_text()
    assert "check S vs mesh golden: 0/512 4-element words differ" in log
    assert "check hashes vs Spike: match (0 of 4 differ)" in log
    assert "attn_vpu PASSED" in log
