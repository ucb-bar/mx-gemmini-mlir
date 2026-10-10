"""Check clean Spike evidence for source resources connected through SSA."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mx_gemmini_support.source_payload import manifest_json, manifest_sha256
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/ssa_source_binding_266c593"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ssa_source_resources_preserve_numerically_qualified_commands() -> None:
    index = json.loads((EVIDENCE / "qualification.json").read_text())
    assert index["schema"] == "mx_gemmini.ssa_source_binding_spike_matrix.v1"
    assert index["compiler_revision"].startswith("0d6460c")
    assert len(index["cases"]) == 6
    historical = {
        "fp6": ("fp6_resource_manifest_binding_266c593/qualification.json",
                ("files_sha256", "physical_program.json")),
        "fp8_vpu": ("compiled_matrix_vpu_requant_fp8_64x64x128_20261009.json",
                    ("files_sha256", "physical_program.json")),
        "fp4": ("compiled_mx_fp4_64x64x128_20261009.json",
                ("files_sha256", "physical_program.json")),
        "asym_fp6_e3m2_e3m2": ("nicolas_generated_modes_266c593/"
                               "e3m2_e3m2/first.json", ("physical_program_sha256",)),
        "asym_dim8_fp4_e4m3": ("nicolas_generated_mesh_dim8_266c593/"
                               "first/fp4_e4m3.json", ("physical_program_sha256",)),
        "asym_dim32_fp4_e4m3": ("nicolas_generated_mesh_dim32_266c593/"
                                "first/fp4_e4m3.json", ("physical_program_sha256",)),
    }
    expected_comparisons = {
        "fp6": {"bf16": 16384},
        "fp8_vpu": {"fp8_codes": 4096, "e8m0_scales": 128},
        "fp4": {"bf16": 4096},
        "asym_fp6_e3m2_e3m2": {"bf16": 4096},
        "asym_dim8_fp4_e4m3": {"bf16": 4096},
        "asym_dim32_fp4_e4m3": {"bf16": 4096},
    }
    for row in index["cases"]:
        name = row["case"]
        folder = EVIDENCE / name
        assert row["compiler_revision"] == index["compiler_revision"]
        assert row["rtl_revision"] == index["rtl_revision"]
        assert row["comparisons"] == expected_comparisons[name]
        assert row["reproduced_except_link_log"] is True
        for filename, digest in row["files_sha256"].items():
            assert _sha(folder / filename) == digest
        first = json.loads((folder / "first.json").read_text())
        repro = json.loads((folder / "repro.json").read_text())
        for receipt in (first, repro):
            assert receipt["status"] == row["status"]
            assert receipt["compiler_revision"] == row["compiler_revision"]
            assert receipt["elf_sha256"] == row["elf_sha256"]
            assert receipt["spike_exit_code"] == 0
            assert receipt["spike_log_sha256"] == _sha(folder / "spike.log")
            receipt["build_log_sha256"].pop("link.log")
        assert first == repro

        manifest = json.loads((folder / "resource_manifest.json").read_text())
        assert manifest_sha256(manifest) == row["payload_manifest_sha256"]
        bound = (folder / "bound.mlir").read_text()
        profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593" /
                               f"{row['profile_name']}.json")
        report = verify_ir(bound, profile)
        assert report["source_resources"] == (7 if name in {
            "fp6", "asym_fp6_e3m2_e3m2", "asym_dim8_fp4_e4m3",
            "asym_dim32_fp4_e4m3"} else 4)
        assert report["lut_uploads"] == (3 if name in {
            "fp6", "asym_fp6_e3m2_e3m2", "asym_dim8_fp4_e4m3",
            "asym_dim32_fp4_e4m3"} else 0)
        from xdsl.context import Context
        from xdsl.dialects.builtin import Builtin
        from xdsl.dialects.func import Func
        from xdsl.parser import Parser
        context = Context(allow_unregistered=True)
        context.load_dialect(Builtin)
        context.load_dialect(Func)
        module = Parser(context, bound).parse_module()
        assert module.attributes["mx.payload_binding_schema"].data == (
            "source_resources_ssa_v1")
        assert module.attributes["mx.payload_manifest_json"].data == manifest_json(manifest)
        assert "func.func @site_0()" in bound
        if name == "fp8_vpu":
            assert (report["vpu_commands"], report["spad_requants"]) == (1, 1)
            assert "0 FP8 code mismatches, 0 E8M0 scale mismatches" in (
                folder / "spike.log").read_text()
        else:
            assert "0 BF16 mismatches" in (folder / "spike.log").read_text()

        path, keys = historical[name]
        prior = json.loads((ROOT / "docs/evidence" / path).read_text())
        for key in keys:
            prior = prior[key]
        assert prior == _sha(folder / "physical_program.json")
