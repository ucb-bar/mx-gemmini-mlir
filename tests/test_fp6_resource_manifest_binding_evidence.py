"""Pin the source-bound FP6 LUT resource contract to compiler/Spike evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mx_gemmini_support.source_payload import manifest_json, manifest_sha256
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/fp6_resource_manifest_binding_266c593"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fp6_resource_manifest_is_bound_to_reproduced_spike_program() -> None:
    index = json.loads((EVIDENCE / "qualification.json").read_text())
    manifest = json.loads((EVIDENCE / "payload_manifest.json").read_text())
    first = json.loads((EVIDENCE / "first.json").read_text())
    repro = json.loads((EVIDENCE / "repro.json").read_text())
    mlir = (EVIDENCE / "payload_bound.mlir").read_text()
    profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                           "MxE3M2OnlyGemminiRocketConfig.json")

    assert index["schema"] == "mx_gemmini.fp6_resource_manifest_binding_qualification.v1"
    assert index["compiler_revision"].startswith("c1bf581")
    assert index["reproduction_exact_except_link_log"] is True
    assert index["serial_scale_selector_workaround"] is True
    assert index["shape_mnk"] == [128, 128, 2048]
    assert (index["compared_bf16_outputs"], index["spike_mismatches"],
            index["command_count"]) == (16384, 0, 1289)
    assert index["payload_manifest_sha256"] == manifest_sha256(manifest)
    for name, digest in index["files_sha256"].items():
        assert _sha(EVIDENCE / name) == digest
    assert verify_ir(mlir, profile)["contracts"] == 1
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir).parse_module()
    assert module.attributes["mx.payload_manifest_json"].data == manifest_json(manifest)
    for name, layout in (("activation_lut", "row_pair_lut_6bit"),
                         ("weight_lut", "column_pair_lut_6bit"),
                         ("output_lut", "output_pair_lut_6bit")):
        assert manifest["resources"][name]["shape"] == [64, 3]
        assert manifest["resources"][name]["element_bits"] == 32
        assert manifest["resources"][name]["layout"] == layout

    for receipt in (first, repro):
        assert receipt["status"] == "source_golden_matched_on_pinned_spike"
        assert receipt["compiler_revision"] == index["compiler_revision"]
        assert receipt["spike_exit_code"] == 0
        assert receipt["compared_bf16_outputs"] == 16384
        assert receipt["payload_manifest_sha256"] == index["payload_manifest_sha256"]
        assert receipt["bound_mlir_sha256"] == _sha(EVIDENCE / "payload_bound.mlir")
        assert receipt["files_sha256"]["physical_program.json"] == _sha(
            EVIDENCE / "physical_program.json")
        assert receipt["files_sha256"]["mx_issue.c"] == _sha(EVIDENCE / "mx_issue.c")
        assert receipt["spike_log_sha256"] == _sha(EVIDENCE / "spike.log")
        receipt["build_log_sha256"].pop("link.log")
    assert first == repro
    assert "0 BF16 mismatches" in (EVIDENCE / "spike.log").read_text()

    previous = json.loads((ROOT / "docs/evidence/"
                           "compiled_mx_fp6_128x128x2048_20261009.json").read_text())
    assert previous["files_sha256"]["physical_program.json"] == _sha(
        EVIDENCE / "physical_program.json")
