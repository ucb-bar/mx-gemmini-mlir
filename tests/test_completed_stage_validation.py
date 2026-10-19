"""A reader-budget refusal can be rechecked only with complete source-bound words."""

import hashlib
import json

import pytest

from tools.validate_completed_merlin_stage import validate


def fixture(tmp_path, *, count=5120, done=True):
    stage, build = tmp_path / "stage", tmp_path / "build"
    stage.mkdir()
    build.mkdir()
    raw = (f"builtin.module {{func.func @forward(%a:tensor<1x{count}xf32>)"
           f"->tensor<1x{count}xf32>{{func.return %a:tensor<1x{count}xf32>}}}}").encode()
    digest = hashlib.sha256(raw).hexdigest()
    (stage / "model.mlir").write_bytes(raw)
    (stage / "capture_receipt.json").write_text(json.dumps({
        "artifacts": {"model.mlir": {"sha256": digest}}}))
    word_hash = hashlib.sha256(bytes(count * 4)).hexdigest()
    lines = [f"OUT_BEGIN f32 {count}"]
    for start in range(0, count, 1024):
        length = min(1024, count - start)
        lines.append(f"OUT_CHUNK f32 {start} {length} " + " ".join(["0"] * length))
    lines += [f"OUT_SHA256 f32le {count} {count * 4} {word_hash}",
              f"OUT_SHA256_INDEX 0 rawbytes {count} {count * 4} {word_hash}",
              "METRIC memref_rank_mismatch 0", "METRIC build_hash abcdef123456", "DONE"]
    (build / "spike.log").write_text("\n".join(lines) + "\n")
    receipt = {"capture_model_sha256": digest, "build_hash": "abcdef123456",
               "first_output_sha256_enabled": True,
               "spike": {"exit_code": 0, "done": done}}
    if not done:
        receipt["spike"]["output_evidence_error"] = "invalid chunked model output type or count"
    (build / "mx_stage_probe.json").write_text(json.dumps(receipt))
    return stage, build, tmp_path / "validation.json"


def test_complete_large_output_revalidation_preserves_original_refusal(tmp_path):
    stage, build, out = fixture(tmp_path, count=1_048_577, done=False)
    original = (build / "mx_stage_probe.json").read_bytes()
    result = validate(stage, build, out)
    assert result["validated"] and result["original_reader_refused"]
    assert result["complete_output_elements"] == 1_048_577
    assert result["output_digests"]["numerical_qualification"] == "not_established"
    assert (build / "mx_stage_probe.json").read_bytes() == original


@pytest.mark.parametrize("change", ["partial", "digest", "build_hash", "exit_code", "other_refusal"])
def test_revalidation_never_substitutes_for_failed_evidence(tmp_path, change):
    stage, build, out = fixture(tmp_path)
    log = build / "spike.log"
    if change == "partial":
        log.write_text(log.read_text().replace("OUT_CHUNK f32 4096 1024", "DISCARDED 4096 1024"))
    elif change == "digest":
        log.write_text(log.read_text().replace(hashlib.sha256(bytes(5120 * 4)).hexdigest(), "0" * 64))
    elif change == "build_hash":
        log.write_text(log.read_text().replace("abcdef123456", "fedcba654321"))
    else:
        receipt_path = build / "mx_stage_probe.json"
        receipt = json.loads(receipt_path.read_text())
        if change == "exit_code":
            receipt["spike"]["exit_code"] = 1
        else:
            receipt["spike"].update(done=False, output_evidence_error="unrelated failure")
        receipt_path.write_text(json.dumps(receipt))
    with pytest.raises((ValueError, RuntimeError)):
        validate(stage, build, out)
    assert not out.exists()
