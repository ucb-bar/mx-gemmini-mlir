"""Audit the archived eight-case linkable host-output Spike qualification."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from mx_gemmini_support import radiance_fp6_host, radiance_fp8_host


EVIDENCE = (Path(__file__).resolve().parents[1] /
            "docs/evidence/radiance_host_requant_objects_80f84ca")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path) -> bytes:
    content = path.read_bytes()
    return gzip.decompress(content) if path.suffix == ".gz" else content


def test_all_source_host_requant_objects_match_on_pinned_spike():
    index_bytes = (EVIDENCE / "index.json").read_bytes()
    index = json.loads(index_bytes)
    archive = json.loads((EVIDENCE / "evidence.json").read_text())
    assert index["schema"] == "mx_gemmini.radiance_host_object_roster.v1"
    assert index["status"] == "all_8_requant_objects_matched_source_goldens_on_pinned_spike"
    assert len(index["cases"]) == 8
    assert index["compared_codes"] == 90112
    assert index["compared_scales"] == 3328
    assert archive["status"] == "fresh_checkout_reproduced_all_8_host_objects_and_spike_logs"
    assert archive["index_sha256"] == archive["fresh_index_sha256"] == _sha(index_bytes)
    clone = json.loads((EVIDENCE / "published_clone_replay.json").read_text())
    assert clone["status"] == "published_clone_rebuilt_and_reproduced_all_8_objects"
    assert clone["branch"] == "handwritten-implementation"
    assert clone["archived_index_sha256"] == clone["new_clone_index_sha256"] == _sha(index_bytes)
    assert clone["native_verifier_sha256"] == index["native_verifier_sha256"]
    assert clone["cases"] == 8
    assert clone["compared_codes"] == 90112
    assert clone["compared_scales"] == 3328
    for key, file in (("configure_log_sha256", "published_clone_configure.log"),
                      ("build_log_sha256", "published_clone_build.log")):
        assert _sha((EVIDENCE / file).read_bytes()) == clone[key]
    assert b"Linking CXX executable tools/mx-gemmini-opt" in (
        EVIDENCE / "published_clone_build.log").read_bytes()
    assert len(archive["files"]) == 8 * 12
    for file in archive["files"]:
        assert _sha(_read(EVIDENCE / file["path"])) == file["raw_sha256"]

    precisions = {"fp4": 0, "fp6": 0, "fp8": 0}
    for row in index["cases"]:
        name = row["case"]
        precision = name.split(".")[1]
        precisions[precision] += 1
        prefix = EVIDENCE / "cases" / name
        source = EVIDENCE / "source" / name
        obj = json.loads((prefix / "object/object_manifest.json").read_text())
        built = json.loads((prefix / "object/compile_manifest.json").read_text())
        spike = json.loads((prefix / "spike/index.json").read_text())
        assert obj["schema"] == "mx_gemmini.linkable_object.v1"
        assert obj["status"] == "rv64_rocc_composed_object_built"
        assert obj["defined_symbol"] == "mx_issue"
        assert obj["undefined_symbols"] == []
        assert obj["allocated_data_section_bytes"] == 0
        assert obj["embedded_operand_bytes"] == obj["embedded_golden_bytes"] == 0
        assert obj["host_output_format"] == (
            "radiance_header_fp6" if precision == "fp6" else "radiance_header_fp8")
        abi = {entry["name"]: entry for entry in obj["buffer_abi"]}
        assert {"output_bf16", "output_quantized", "scratch_output_scales"} <= abi.keys()
        assert abi["output_quantized"]["role"] == "write"
        assert abi["scratch_output_scales"]["role"] == "write"
        assert abi["output_quantized"]["minimum_bytes"] == row["compared_codes"]
        assert abi["scratch_output_scales"]["minimum_bytes"] == row["compared_scales"]
        if precision == "fp6":
            assert abi["output_lut"]["minimum_bytes"] == 768
            assert abi["output_quantized"]["layout"] == "pair_major_nibble_codes"
        assert built["status"] == "rv64_rocc_object_built"
        assert built["compiler_revision"] == archive["compiler_revision"]
        assert built["native_verifier_sha256"] == index["native_verifier_sha256"]
        assert spike["status"] == row["status"] == "source_quantized_output_matched_on_pinned_spike"
        assert spike["spike_exit_code"] == 0
        assert spike["compared_codes"] == row["compared_codes"]
        assert spike["compared_scales"] == row["compared_scales"]
        assert _sha((prefix / "spike/index.json").read_bytes()) == row["qualifier_index_sha256"]
        assert _sha(_read(prefix / "object/mx_issue.o.gz")) == row["object_sha256"]
        assert _sha(_read(prefix / "spike/mx_host_object.elf.gz")) == row["elf_sha256"]
        assert _sha((prefix / "spike/spike.log").read_bytes()) == row["spike_log_sha256"]
        assert _sha(_read(source / "payload_bound.mlir.gz")) == row["bound_mlir_sha256"]
        assert _sha((source / "bundle/manifest.json").read_bytes()) == row["bundle_manifest_sha256"]
        command_c = _read(prefix / "object/mx_issue.c.gz")
        driver_c = _read(prefix / "spike/mx_driver.c.gz")
        m, n, _ = spike["shape_mnk"]
        generator = radiance_fp6_host if precision == "fp6" else radiance_fp8_host
        legacy = (Path(__file__).resolve().parents[1] /
                  "docs/evidence/radiance_mx_gemm_latest_e9ded36_ee22/spike" /
                  name / "build/mx_driver.c").read_text()
        first = ("static uint32_t float_bits(float value) {" if precision == "fp6"
                 else "static float bf16_value(uint16_t bits) {")
        legacy_kernel = first + legacy.split(first, 1)[1].split("\nint main(void) {", 1)[0]
        assert generator.emit_kernel(m, n) == legacy_kernel
        assert generator.emit_kernel(m, n, runtime_pointers=True).encode() in command_c
        assert b"mx_issue_commands(" in command_c
        assert b"radiance_header_requantize(" in command_c
        assert b"radiance_header_requantize(" not in driver_c
        assert driver_c.count(b"mx_issue(") == 1
        assert b".incbin" not in command_c
        expected = (f"runtime host MX: 0/{row['compared_codes']} code mismatches, "
                    f"0/{row['compared_scales']} scale mismatches").encode()
        assert expected in (prefix / "spike/spike.log").read_bytes()
    assert precisions == {"fp4": 3, "fp6": 2, "fp8": 3}
