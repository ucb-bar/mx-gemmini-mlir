"""Classify Nicolas MX programs using validated generated-object Spike receipts.

The selected matrix result is narrower than rebuilding every instruction and
check in the source C program. Other receipt statuses stay separate so a
source hash or a source-only Spike run never becomes a regeneration claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs/evidence/nicolas_mx_source_inventory_266c593/index.json"
OUTPUT = ROOT / "docs/evidence/nicolas_mx_regeneration_audit_266c593/index.json"
NUMERIC_FIELDS = ("outputs_checked", "compared_bf16_outputs",
                  "compared_packed_lut_bytes", "compared_e8m0_scales",
                  "codes_checked", "packed_bytes_checked", "scales_checked")
MATRIX_FAMILIES = {"asymmetric_matrix", "other_tiled_matrix"}
HEX256 = re.compile(r"[0-9a-f]{64}\Z")
HEX160 = re.compile(r"[0-9a-f]{40}\Z")
CHAIN_NAME = re.compile(r"matmul_tiled_(fp4|fp6|fp8)_(64x64|128x128)_chain\Z")
SPECIALIZED_RESULTS = {
    "vpu_softmax": ("mx_gemmini.nicolas_vpu_softmax_spike.v1",
                    "source_vpu_softmax_matched_on_pinned_spike",
                    {"compared_bf16_values": 512}),
    "spad_requant": ("mx_gemmini.nicolas_fp8_spad_requant_compiler_spike.v1",
                     "source_and_compiled_flat_tiled_fp8_matched_on_pinned_spike",
                     {"compared_fp8_codes": 4096, "compared_e8m0_scales": 128}),
    "spad_requant_fp4": ("mx_gemmini.nicolas_fp4_dual_public_object_spike.v1",
                         "source_and_public_object_matched_on_pinned_spike",
                         {"compared_fp4_codes": 16384, "compared_e8m0_scales": 512}),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_digest(value: object, label: str) -> None:
    if isinstance(value, str) and HEX256.fullmatch(value):
        return
    if isinstance(value, dict) and value and all(
            isinstance(key, str) and isinstance(digest, str) and
            HEX256.fullmatch(digest) for key, digest in value.items()):
        return
    raise ValueError(f"{label} is not a SHA-256 digest or digest map")


def _archived_sha(directory: Path, name: str) -> str:
    """Hash an archived artifact, including archives saved as gzip."""
    plain, compressed = directory / name, directory / f"{name}.gz"
    if plain.is_file():
        return _sha(plain)
    if compressed.is_file():
        with gzip.open(compressed, "rb") as stream:
            return hashlib.sha256(stream.read()).hexdigest()
    raise ValueError(f"missing archived chain artifact: {plain}")


def _full_chain_result(entry: dict) -> dict | None:
    """Validate both outputs of a named connected matrix source program."""
    match = CHAIN_NAME.fullmatch(entry["name"])
    if entry["family"] != "other_tiled_matrix" or match is None:
        return None
    precision, shape = match.groups()
    n = int(shape.split("x")[0])
    expected_codes, expected_scales = n * n, n * n // 32
    status = ("source_connected_chain_matched_on_pinned_spike"
              if precision == "fp8" else
              "source_and_compiler_matched_on_pinned_spike")
    schema = (f"mx_gemmini.nicolas_connected_plain_chain_"
              f"{'64x64' if n == 64 else '128'}.v1"
              if precision == "fp8" else
              f"mx_gemmini.nicolas_{precision}_connected_resident_spike.v1")
    for ref in entry["evidence_references"]:
        if ref["status"] != status or ref["schema"] != schema:
            continue
        path = ROOT / ref["path"]
        if _sha(path) != ref["sha256"]:
            raise ValueError(f"changed Nicolas chain receipt: {ref['path']}")
        receipt = json.loads(path.read_text())
        if (receipt.get("schema") != schema or
                receipt.get("status") != status or
                receipt.get(ref["field"]) != entry["source_sha256"] or
                receipt.get("rtl_revision") !=
                "266c593f2cb51d7e3fe83fc0317072b585ac3c52"):
            raise ValueError(f"stale chain source binding in {ref['path']}")
        for field in ("profile_sha256", "object_sha256",
                      "frontend_mlir_sha256", "bound_mlir_sha256"):
            _require_digest(receipt.get(field), f"{ref['path']} {field}")
        revision = receipt.get("compiler_revision")
        if not isinstance(revision, str) or not HEX160.fullmatch(revision):
            raise ValueError(f"{ref['path']} has no pinned compiler revision")
        code1 = f"compared_c1_{precision}_codes"
        code2 = (f"compared_{precision}_codes" if precision == "fp8"
                 else f"compared_c2_{precision}_codes")
        scale2 = ("compared_e8m0_scales" if precision == "fp8"
                  else "compared_c2_e8m0_scales")
        metrics = {field: receipt.get(field) for field in (
            code1, "compared_c1_e8m0_scales", code2, scale2)}
        if metrics != {code1: expected_codes,
                       "compared_c1_e8m0_scales": expected_scales,
                       code2: expected_codes, scale2: expected_scales}:
            raise ValueError(f"incomplete two-stage chain comparison in {ref['path']}")
        if precision == "fp8":
            if receipt.get("spike_exit_code") != 0:
                raise ValueError(f"failed compiler Spike run in {ref['path']}")
            objects = receipt["object_sha256"]
            archived = {name: ((path.parent / name).is_file() or
                               (path.parent / f"{name}.gz").is_file())
                        for name in objects}
            if any(archived.values()) and not all(archived.values()):
                raise ValueError(f"partially archived chain objects in {ref['path']}")
            if all(archived.values()):
                for name in objects:
                    if _archived_sha(path.parent, name) != objects[name]:
                        raise ValueError(f"changed chain object {name} in {ref['path']}")
            object_archive_verified = all(archived.values())
            for field, name in (("elf_sha256", "mx_program.elf"),
                                ("spike_log_sha256", "spike.log")):
                _require_digest(receipt.get(field), f"{ref['path']} {field}")
                if _archived_sha(path.parent, name) != receipt[field]:
                    raise ValueError(f"changed chain {name} in {ref['path']}")
            files = receipt.get("files_sha256", {})
            for name in ("c1_codes_ref.bin", "c1_scales_ref.bin",
                         "c2_codes_ref.bin", "c2_scales_ref.bin",
                         "mx_issue.c", "physical_program.json"):
                _require_digest(files.get(name), f"{ref['path']} {name}")
                if _archived_sha(path.parent, name) != files[name]:
                    raise ValueError(f"changed chain {name} in {ref['path']}")
            baseline = receipt.get("source_baseline")
            if baseline is not None and (
                    baseline.get("source_spike_exit_code") != 0 or
                    baseline.get("source_c1_codes_checked") != expected_codes or
                    baseline.get("source_c2_codes_checked") != expected_codes or
                    baseline.get("source_c1_scales_checked") != expected_scales or
                    baseline.get("source_c2_scales_checked") != expected_scales):
                raise ValueError(f"incomplete source chain baseline in {ref['path']}")
            elf_sha = receipt["elf_sha256"]
            log_sha = receipt["spike_log_sha256"]
        else:
            source = receipt.get("source_spike", {})
            compiler = receipt.get("compiler_spike", {})
            if (source.get("exit_code") != 0 or compiler.get("exit_code") != 0 or
                    source.get("matched") is not True or
                    compiler.get("matched") is not True or
                    receipt.get("allocated_data_section_bytes") != 0):
                raise ValueError(f"failed source or compiler chain in {ref['path']}")
            for result, fields in ((source, (("elf_sha256", "source.elf"),
                                            ("spike_log_sha256", "source_spike.log"))),
                                   (compiler, (("elf_sha256", "compiled.elf"),
                                               ("spike_log_sha256", "compiled_spike.log")))):
                for field, name in fields:
                    _require_digest(result.get(field), f"{ref['path']} {field}")
                    if _archived_sha(path.parent, name) != result[field]:
                        raise ValueError(f"changed chain {name} in {ref['path']}")
            if _archived_sha(path.parent, "mx_issue.o") != receipt["object_sha256"]:
                raise ValueError(f"changed chain object in {ref['path']}")
            object_archive_verified = True
            elf_sha = compiler["elf_sha256"]
            log_sha = compiler["spike_log_sha256"]
        return {"receipt": ref["path"], "receipt_sha256": ref["sha256"],
                "receipt_schema": schema, "profile_sha256": receipt["profile_sha256"],
                "compiler_revision": revision,
                "frontend_mlir_sha256": receipt["frontend_mlir_sha256"],
                "bound_mlir_sha256": receipt["bound_mlir_sha256"],
                "object_sha256": receipt["object_sha256"],
                "object_archive_verified": object_archive_verified,
                "elf_sha256": elf_sha, "spike_log_sha256": log_sha,
                "checked_output_metrics": metrics}
    return None


def _full_matrix_result(entry: dict) -> dict | None:
    if entry["family"] not in MATRIX_FAMILIES:
        return None
    candidates = [ref for ref in entry["evidence_references"]
                  if ref["status"] == "source_golden_matched_on_pinned_spike"]
    for ref in candidates:
        path = ROOT / ref["path"]
        if _sha(path) != ref["sha256"]:
            raise ValueError(f"changed Nicolas receipt: {ref['path']}")
        receipt = json.loads(path.read_text())
        if (receipt.get("schema") != ref["schema"] or
                receipt.get("status") != ref["status"] or
                receipt.get(ref["field"]) != entry["source_sha256"]):
            raise ValueError(f"stale source binding in {ref['path']}")
        for field in ("profile_sha256", "object_sha256", "elf_sha256",
                      "spike_log_sha256"):
            _require_digest(receipt.get(field), f"{ref['path']} {field}")
        for field in ("compiler_revision", "model2mlir_revision"):
            if not isinstance(receipt.get(field), str) or not HEX160.fullmatch(receipt[field]):
                raise ValueError(f"{ref['path']} {field} is not a pinned git revision")
        if receipt.get("mismatches") != 0 and receipt.get("spike_exit_code") != 0:
            raise ValueError(f"no zero-error Spike outcome in {ref['path']}")
        metrics = {field: receipt[field] for field in NUMERIC_FIELDS
                   if isinstance(receipt.get(field), int) and receipt[field] > 0}
        if not metrics:
            raise ValueError(f"no full-output comparison count in {ref['path']}")
        return {"receipt": ref["path"], "receipt_sha256": ref["sha256"],
                "receipt_schema": ref["schema"],
                "profile_sha256": receipt["profile_sha256"],
                "compiler_revision": receipt["compiler_revision"],
                "model2mlir_revision": receipt["model2mlir_revision"],
                "object_sha256": receipt["object_sha256"],
                "elf_sha256": receipt["elf_sha256"],
                "spike_log_sha256": receipt["spike_log_sha256"],
                "checked_output_metrics": metrics}
    return None


def _full_specialized_result(entry: dict) -> dict | None:
    """Validate complete selected outputs for non-matrix MX sources."""
    if entry["name"] == "matmul_ws_mx_generic":
        return _full_ws_generic_result(entry)
    if entry["name"] == "vpu_ops":
        return _full_vpu_program_result(entry)
    if entry["name"] == "chain_vpu_spad_requant":
        return _full_resident_vpu_result(entry)
    if entry["name"] in ("chain_pipelined", "mx_mem_bw"):
        return _full_compound_result(entry)
    specification = SPECIALIZED_RESULTS.get(entry["name"])
    if specification is None:
        return None
    schema, status, metrics = specification
    for ref in entry["evidence_references"]:
        if ref["schema"] != schema or ref["status"] != status:
            continue
        path = ROOT / ref["path"]
        if _sha(path) != ref["sha256"]:
            raise ValueError(f"changed Nicolas specialized receipt: {ref['path']}")
        receipt = json.loads(path.read_text())
        if (receipt.get("schema") != schema or receipt.get("status") != status or
                receipt.get(ref["field"]) != entry["source_sha256"] or
                receipt.get("rtl_revision") !=
                "266c593f2cb51d7e3fe83fc0317072b585ac3c52"):
            raise ValueError(f"stale specialized source binding in {ref['path']}")
        _require_digest(receipt.get("profile_sha256"), f"{ref['path']} profile")
        revision = receipt.get("compiler_revision")
        if not isinstance(revision, str) or not HEX160.fullmatch(revision):
            raise ValueError(f"{ref['path']} has no pinned compiler revision")
        if entry["name"] == "vpu_softmax":
            if (receipt.get("spike_exit_code") != 0 or
                    receipt.get("compared_bf16_values") != metrics["compared_bf16_values"]):
                raise ValueError(f"incomplete softmax result in {ref['path']}")
            object_sha = receipt["object_sha256"]["mx_issue.o"]
            object_path = path.parent / "object/mx_issue.o"
            log_sha = receipt.get("spike_log_sha256")
            log_path = path.parent / "spike.log"
            elf_sha = receipt.get("elf_sha256")
            model2mlir_revision = receipt.get("model2mlir_revision")
            if (not isinstance(model2mlir_revision, str) or
                    not HEX160.fullmatch(model2mlir_revision)):
                raise ValueError(f"{ref['path']} has no pinned model2MLIR revision")
        else:
            source, compiler = receipt.get("source_spike", {}), receipt.get("compiler_spike", {})
            if (receipt.get("allocated_data_section_bytes") != 0 or
                    source.get("exit_code") != 0 or compiler.get("exit_code") != 0 or
                    any(source.get(key) != value or compiler.get(key) != value
                        for key, value in metrics.items())):
                raise ValueError(f"incomplete requant result in {ref['path']}")
            for result, directory in ((source, "source"), (compiler, "compiled")):
                for field, name in (("elf_sha256", "program.elf"),
                                    ("spike_log_sha256", "spike.log")):
                    _require_digest(result.get(field), f"{ref['path']} {directory} {field}")
                    if _sha(path.parent / directory / name) != result[field]:
                        raise ValueError(f"changed {directory} {name} in {ref['path']}")
            object_sha = receipt.get("object_sha256")
            object_path = (path.parent / "object/mx_issue.o" if entry["name"] ==
                           "spad_requant_fp4" else path.parent / "mx_issue.o")
            log_sha = compiler["spike_log_sha256"]
            log_path = path.parent / "compiled/spike.log"
            elf_sha = compiler["elf_sha256"]
            model2mlir_revision = None
        _require_digest(object_sha, f"{ref['path']} object")
        _require_digest(log_sha, f"{ref['path']} Spike log")
        _require_digest(elf_sha, f"{ref['path']} ELF")
        if _sha(object_path) != object_sha or _sha(log_path) != log_sha:
            raise ValueError(f"changed specialized object or log in {ref['path']}")
        result = {"receipt": ref["path"], "receipt_sha256": ref["sha256"],
                  "receipt_schema": schema, "profile_sha256": receipt["profile_sha256"],
                  "compiler_revision": revision, "object_sha256": object_sha,
                  "elf_sha256": elf_sha, "spike_log_sha256": log_sha,
                  "checked_output_metrics": metrics}
        if model2mlir_revision is not None:
            result["model2mlir_revision"] = model2mlir_revision
        return result
    return None


def _full_ws_generic_result(entry: dict) -> dict:
    """Validate the Rocket numerical replay without claiming MMIO parity."""
    directory = ROOT / "docs/evidence/nicolas_ws_generic_portable_a042643_266c593"
    index = json.loads((directory / "index.json").read_text())
    if (index.get("schema") != "mx_gemmini.nicolas_ws_generic_portable_archive.v1" or
            index.get("status") != "current_model2mlir_rocket_numerical_replay_matched" or
            index.get("source_sha256") != entry["source_sha256"] or
            index.get("rtl_revision") !=
            "266c593f2cb51d7e3fe83fc0317072b585ac3c52" or
            index.get("source_transport") != "fixed_mmio" or
            index.get("compiled_transport") != "rocket_rocc" or
            index.get("source_mmio_issue_qualification") != "not_tested" or
            index.get("same_profile_reproduction_exact") is not True or
            (index.get("bf16_values_checked"), index.get("packed_bytes_checked"),
             index.get("e8m0_scales_checked")) != (16384, 8192, 512)):
        raise ValueError("incomplete Nicolas generic FP6 numerical archive")
    required = {"model2mlir.mlir", "frontend_binding.json", "profile_bound.mlir",
                "payload_bound.mlir", "bundle/manifest.json", "bundle/activation_lut.bin",
                "bundle/weight_lut.bin", "bundle/output_lut.bin",
                "object/mx_issue.o", "object/object_manifest.json",
                "object/physical_program.json", "run/mx_driver.c",
                "run/mx_program.elf", "run/spike.log", "receipt.json"}
    files = index.get("files_sha256", {})
    if not required <= set(files):
        raise ValueError("Nicolas generic FP6 archive omits compiler evidence")
    for name, digest in files.items():
        _require_digest(digest, f"Nicolas generic FP6 {name}")
        if _sha(directory / name) != digest:
            raise ValueError(f"changed Nicolas generic FP6 artifact {name}")
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != "mx_gemmini.nicolas_ws_generic_portable_spike.v1" or
            receipt.get("status") !=
            "source_header_selected_outputs_matched_on_pinned_spike" or
            receipt.get("source_driver_sha256") != entry["source_sha256"] or
            receipt.get("source_header_sha256") != index["source_header_sha256"] or
            receipt.get("model2mlir_revision") != index["model2mlir_revision"] or
            receipt.get("compiler_revision") != index["compiler_revision"] or
            receipt.get("source_transport") != "fixed_mmio" or
            receipt.get("compiled_transport") != "rocket_rocc" or
            receipt.get("source_mmio_issue_qualification") != "not_tested" or
            receipt.get("source_header_projection_policy") !=
            "radiance_header_fp6_lut_v1" or
            (receipt.get("bf16_values_checked"), receipt.get("packed_bytes_checked"),
             receipt.get("e8m0_scales_checked"), receipt.get("mismatches")) !=
            (16384, 8192, 512, 0) or
            json.loads((directory / "reproduction_receipt.json").read_text()) != receipt or
            _sha(directory / "reproduction_receipt.json") !=
            index.get("reproduction_receipt_sha256")):
        raise ValueError("Nicolas generic FP6 source or reproduction receipt differs")
    for name, field in (("model2mlir.mlir", "frontend_mlir_sha256"),
                        ("frontend_binding.json", "frontend_binding_sha256"),
                        ("payload_bound.mlir", "bound_mlir_sha256"),
                        ("object/mx_issue.o", "object_sha256"),
                        ("run/mx_program.elf", "elf_sha256"),
                        ("run/spike.log", "spike_log_sha256")):
        if _sha(directory / name) != receipt.get(field):
            raise ValueError(f"changed Nicolas generic FP6 {name}")
    for field in ("profile_sha256", "model2mlir_source_closure_sha256",
                  "spike_extension_sha256", "spike_binary_sha256",
                  "riscv_gcc_sha256", "native_mx_opt_sha256"):
        _require_digest(receipt.get(field), f"Nicolas generic FP6 {field}")
    revision = receipt.get("compiler_revision")
    if not isinstance(revision, str) or not HEX160.fullmatch(revision):
        raise ValueError("Nicolas generic FP6 compiler revision is not pinned")
    return {"receipt": str(receipt_path.relative_to(ROOT)),
            "receipt_sha256": _sha(receipt_path),
            "receipt_schema": receipt["schema"],
            "profile_sha256": receipt["profile_sha256"],
            "compiler_revision": revision,
            "model2mlir_revision": receipt["model2mlir_revision"],
            "object_sha256": receipt["object_sha256"],
            "elf_sha256": receipt["elf_sha256"],
            "spike_log_sha256": receipt["spike_log_sha256"],
            "source_transport": "fixed_mmio", "compiled_transport": "rocket_rocc",
            "source_mmio_issue_qualification": "not_tested",
            "checked_output_metrics": {"bf16_values_checked": 16384,
                                       "packed_bytes_checked": 8192,
                                       "e8m0_scales_checked": 512}}


def _full_vpu_program_result(entry: dict) -> dict:
    """Validate the single compiler-issued Nicolas VPU test executable."""
    directory = ROOT / "docs/evidence/nicolas_vpu_ops_program_public_266c593"
    index = json.loads((directory / "index.json").read_text())
    if (index.get("schema") != "mx_gemmini.nicolas_vpu_ops_public_program_archive.v1" or
            index.get("status") != "two_profile_full_vpu_program_source_reference_matched" or
            index.get("source_sha256") != entry["source_sha256"] or
            index.get("rtl_revision") !=
            "266c593f2cb51d7e3fe83fc0317072b585ac3c52" or
            index.get("operation_count") != 30 or
            index.get("snapshot_count") != 30 or
            index.get("source_check_count") != 29 or
            index.get("bf16_values_checked_per_profile") != 13056 or
            index.get("same_profile_reproduction_exact") is not True or
            index.get("cross_profile_object_elf_log_exact") is not True or
            len(index.get("rows", [])) != 2):
        raise ValueError("incomplete Nicolas VPU program archive")
    receipts = []
    required_files = {"abi.json", "compiler.log", "mx_driver.c",
                      "object/compile_manifest.json", "object/mx_issue.c",
                      "object/mx_issue.h", "object/mx_issue.o",
                      "object/object_manifest.json", "object/physical_program.json",
                      "receipt.json", "spike.log", "vpu_ops.elf", "vpu_ops.mlir"}
    seen_profiles = set()
    for row in index["rows"]:
        if row.get("name") not in ("e4m3_vpu", "e4m3_fp4_vpu") or row["name"] in seen_profiles:
            raise ValueError("unknown Nicolas VPU program profile")
        seen_profiles.add(row["name"])
        if set(row.get("files_sha256", {})) != required_files:
            raise ValueError("incomplete Nicolas VPU program artifact set")
        base = directory / row["name"]
        for name, digest in row.get("files_sha256", {}).items():
            _require_digest(digest, f"VPU program {name}")
            if _sha(base / name) != digest:
                raise ValueError(f"changed VPU program artifact {name}")
        receipt_path = base / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        if (row.get("receipt_sha256") != _sha(receipt_path) or
                receipt.get("schema") != "mx_gemmini.nicolas_vpu_program_spike.v1" or
                receipt.get("status") !=
                "compiler_vpu_program_matched_source_reference_on_pinned_spike" or
                receipt.get("source_sha256") != entry["source_sha256"] or
                receipt.get("profile_sha256") != row.get("profile_sha256") or
                receipt.get("mismatches") != 0 or
                (receipt.get("operation_count"), receipt.get("snapshot_count"),
                 receipt.get("source_check_count"), receipt.get("bf16_values_checked")) !=
                (30, 30, 29, 13056)):
            raise ValueError("VPU program receipt or source binding differs")
        for field in ("profile_sha256", "object_sha256", "elf_sha256",
                      "spike_log_sha256"):
            _require_digest(receipt.get(field), f"VPU program {field}")
        revision = receipt.get("compiler_revision")
        if not isinstance(revision, str) or not HEX160.fullmatch(revision):
            raise ValueError("VPU program has no pinned compiler revision")
        for name, field in (("object/mx_issue.o", "object_sha256"),
                            ("vpu_ops.elf", "elf_sha256"),
                            ("spike.log", "spike_log_sha256")):
            if _sha(base / name) != receipt[field]:
                raise ValueError(f"changed VPU program {name}")
        receipts.append((receipt, receipt_path))
    if seen_profiles != {"e4m3_vpu", "e4m3_fp4_vpu"}:
        raise ValueError("missing Nicolas VPU program profile")
    first, second = receipts[0][0], receipts[1][0]
    if (first["profile_sha256"] == second["profile_sha256"] or
            any(first[field] != second[field] for field in
                ("object_sha256", "elf_sha256", "spike_log_sha256")) or
            _sha(directory / "reproduction_receipt.json") !=
            index.get("same_profile_second_receipt_sha256") or
            json.loads((directory / "reproduction_receipt.json").read_text()) !=
            first):
        raise ValueError("VPU program profile or reproduction evidence differs")
    return {"receipt": str(receipts[0][1].relative_to(ROOT)),
            "receipt_sha256": _sha(receipts[0][1]),
            "receipt_schema": first["schema"],
            "profile_sha256": first["profile_sha256"],
            "compiler_revision": first["compiler_revision"],
            "object_sha256": first["object_sha256"],
            "elf_sha256": first["elf_sha256"],
            "spike_log_sha256": first["spike_log_sha256"],
            "checked_output_metrics": {"bf16_values_checked": 13056,
                                       "source_check_count": 29,
                                       "snapshot_count": 30},
            "qualified_profile_count": 2}


def _full_resident_vpu_result(entry: dict) -> dict:
    """Validate the full 64-cubed connected object against Nicolas's chain."""
    directory = ROOT / "docs/evidence/nicolas_resident_vpu_object_6c9ed40"
    path = directory / "qualification_manifest.json"
    receipt = json.loads(path.read_text())
    index = json.loads((directory / "index.json").read_text())
    manifest = json.loads((directory / "object_manifest.json").read_text())
    metrics = {"compared_bf16_values": 4096,
               "compared_fp8_codes": 8192, "compared_e8m0_scales": 256}
    if (receipt.get("schema") != "mx_gemmini.connected_full_chain_sources.v1" or
            receipt.get("status") != "source_connected_full_chain_matched_on_pinned_spike" or
            receipt.get("source_facts", {}).get("source_sha256") !=
            entry["source_sha256"] or
            receipt.get("rtl_revision") !=
            "266c593f2cb51d7e3fe83fc0317072b585ac3c52" or
            receipt.get("spike_exit_code") != 0 or
            any(receipt.get(key) != value for key, value in metrics.items()) or
            manifest.get("allocated_data_section_bytes") != 0 or
            manifest.get("object_sha256") != receipt.get("object_sha256", {}).get(
                "mx_issue.o") or
            index.get("files_sha256", {}).get("qualification_manifest.json") != _sha(path) or
            index.get("status") != "fresh_checkout_spike_full_output_match"):
        raise ValueError("incomplete or stale resident VPU source qualification")
    profile_sha = receipt["source_facts"]["profile_sha256"]
    _require_digest(profile_sha, "resident VPU profile")
    if (manifest.get("profile_sha256") != profile_sha or
            index.get("profile_sha256") != profile_sha):
        raise ValueError("resident VPU profile binding differs")
    for field in ("elf_sha256", "spike_log_sha256"):
        _require_digest(receipt.get(field), f"resident VPU {field}")
    revision = receipt.get("compiler_revision")
    if not isinstance(revision, str) or not HEX160.fullmatch(revision):
        raise ValueError("resident VPU has no pinned compiler revision")
    model2mlir_revision = index.get("model2mlir_revision")
    if (not isinstance(model2mlir_revision, str) or
            not HEX160.fullmatch(model2mlir_revision)):
        raise ValueError("resident VPU has no pinned model2MLIR revision")
    object_sha = receipt["object_sha256"]["mx_issue.o"]
    _require_digest(object_sha, "resident VPU object")
    for name, digest in (("mx_issue.o", object_sha),
                         ("mx_program.elf", receipt["elf_sha256"]),
                         ("spike.log", receipt["spike_log_sha256"])):
        if _sha(directory / name) != digest:
            raise ValueError(f"changed resident VPU {name}")
    return {"receipt": str(path.relative_to(ROOT)), "receipt_sha256": _sha(path),
            "receipt_schema": receipt["schema"],
            "profile_sha256": profile_sha,
            "compiler_revision": revision,
            "model2mlir_revision": model2mlir_revision,
            "object_sha256": object_sha,
            "elf_sha256": receipt["elf_sha256"],
            "spike_log_sha256": receipt["spike_log_sha256"],
            "checked_output_metrics": metrics,
            "source_binding_field": "source_facts.source_sha256"}


def _full_compound_result(entry: dict) -> dict | None:
    """Check archived complete outputs while keeping timing outside the claim."""
    definitions = {
        "chain_pipelined": (
            "docs/evidence/nicolas_chain_pipelined_full_266c593/object_manifest.json",
            "mx_gemmini.full_chain_pipelined_linkable_object.v1",
            "rv64_rocc_two_tile_object_built"),
        "mx_mem_bw": (
            "docs/evidence/nicolas_mem_bw_typed_public_9cb0e4a_266c593/receipt.json",
            "mx_gemmini.nicolas_mem_bw_typed_object_spike.v1",
            "source_memory_phases_and_full_mvout_matched_on_pinned_spike"),
    }
    receipt_name, schema, status = definitions[entry["name"]]
    ref = next((item for item in entry["evidence_references"]
                if item["path"] == receipt_name and item["schema"] == schema and
                item["status"] == status), None)
    if ref is None:
        return None
    path = ROOT / receipt_name
    if _sha(path) != ref["sha256"]:
        raise ValueError(f"changed Nicolas compound receipt: {receipt_name}")
    receipt = json.loads(path.read_text())
    if (receipt.get("schema") != schema or receipt.get("status") != status or
            receipt.get(ref["field"]) != entry["source_sha256"] or
            receipt.get("rtl_revision") !=
            "266c593f2cb51d7e3fe83fc0317072b585ac3c52"):
        raise ValueError(f"stale compound source binding in {receipt_name}")
    for field in ("profile_sha256", "object_sha256"):
        _require_digest(receipt.get(field), f"{receipt_name} {field}")
    revision = receipt.get("compiler_revision")
    if not isinstance(revision, str) or not HEX160.fullmatch(revision):
        raise ValueError(f"{receipt_name} has no pinned compiler revision")
    if entry["name"] == "chain_pipelined":
        execution = receipt.get("spike_qualification", {})
        metrics = {"compared_c1_bf16_values": 4096,
                   "compared_fp8_codes": 16384, "compared_e8m0_scales": 512}
        if (receipt.get("allocated_data_section_bytes") != 0 or
                receipt.get("embedded_operand_bytes") != 0 or
                receipt.get("embedded_golden_bytes") != 0 or
                execution.get("status") != "full_three_site_chain_matched_on_pinned_spike" or
                execution.get("spike_exit_code") != 0 or
                any(execution.get(key) != value for key, value in metrics.items())):
            raise ValueError(f"incomplete three-site chain in {receipt_name}")
        files = (("object_sha256", "mx_issue.o", receipt),
                 ("elf_sha256", "mx_program.elf", execution),
                 ("spike_log_sha256", "spike.log", execution))
        elf_sha, log_sha = execution["elf_sha256"], execution["spike_log_sha256"]
    else:
        metrics = {"compared_mvout_bytes": 16384, "phase_count": 7}
        if (receipt.get("compiled_exit_code") != 0 or
                receipt.get("source_full_exit_code") != 0 or
                any(receipt.get(key) != value for key, value in metrics.items()) or
                receipt.get("compiled_output_sha256") !=
                receipt.get("source_full_output_sha256")):
            raise ValueError(f"incomplete memory result in {receipt_name}")
        _require_digest(receipt.get("compiled_output_sha256"),
                        f"{receipt_name} full readout")
        files = (("object_sha256", "mx_issue.o", receipt),
                 ("compiled_elf_sha256", "compiled/program.elf", receipt),
                 ("compiled_spike_log_sha256", "compiled/spike.log", receipt),
                 ("source_full_elf_sha256", "source_full/program.elf", receipt),
                 ("source_full_spike_log_sha256", "source_full/spike.log", receipt))
        elf_sha, log_sha = (receipt["compiled_elf_sha256"],
                            receipt["compiled_spike_log_sha256"])
    for field, name, owner in files:
        _require_digest(owner.get(field), f"{receipt_name} {field}")
        if _sha(path.parent / name) != owner[field]:
            raise ValueError(f"changed {name} in {receipt_name}")
    return {"receipt": receipt_name, "receipt_sha256": ref["sha256"],
            "receipt_schema": schema, "profile_sha256": receipt["profile_sha256"],
            "compiler_revision": revision, "object_sha256": receipt["object_sha256"],
            "elf_sha256": elf_sha, "spike_log_sha256": log_sha,
            "checked_output_metrics": metrics}


def build_report(inventory: dict) -> dict:
    if (inventory.get("schema") != "mx_gemmini.nicolas_mx_source_inventory.v1" or
            inventory.get("programs") != len(inventory.get("entries", [])) or
            inventory.get("rtl_revision") !=
            "266c593f2cb51d7e3fe83fc0317072b585ac3c52"):
        raise ValueError("Nicolas regeneration needs the pinned source inventory")
    rows = []
    for entry in inventory["entries"]:
        proof = _full_matrix_result(entry)
        chain_proof = _full_chain_result(entry) if proof is None else None
        specialized_proof = (_full_specialized_result(entry)
                             if proof is None and chain_proof is None else None)
        diagnostic = None
        if entry["name"] == "matmul_single_tile_test":
            refs = [ref for ref in entry["evidence_references"] if
                    ref["schema"] == "mx_gemmini.nicolas_single_tile_source_audit.v1"]
            if len(refs) != 1:
                raise ValueError("single-tile debug source audit is missing or ambiguous")
            ref = refs[0]
            path = ROOT / ref["path"]
            if _sha(path) != ref["sha256"]:
                raise ValueError("single-tile debug source audit changed")
            diagnostic = json.loads(path.read_text())
            if (diagnostic.get("source_sha256") != entry["source_sha256"] or
                    diagnostic.get("rtl_revision") != inventory["rtl_revision"] or
                    diagnostic.get("status") !=
                    "debug_source_has_no_active_numerical_oracle" or
                    diagnostic.get("active_output_comparisons") != 0 or
                    diagnostic.get("printed_pass_is_numerical_evidence") is not False or
                    diagnostic.get("compiler_selected_output_qualification") != "not_tested"):
                raise ValueError("single-tile audit does not prove its limited scope")
        if proof is not None:
            status = "generated_object_selected_spike_result_matched"
        elif chain_proof is not None:
            status = "generated_connected_chain_spike_result_matched"
        elif specialized_proof is not None:
            status = "generated_specialized_selected_spike_result_matched"
        elif diagnostic is not None:
            status = "diagnostic_only_no_numerical_oracle"
        elif entry["evidence_references"]:
            status = "separate_evidence_requires_scope_review"
        else:
            status = "no_direct_source_receipt"
        row = {"name": entry["name"], "family": entry["family"],
               "source_sha256": entry["source_sha256"], "status": status,
               "source_reference_count": len(entry["evidence_references"])}
        if proof is not None or chain_proof is not None or specialized_proof is not None:
            row["selected_spike_result"] = proof or chain_proof or specialized_proof
        else:
            row["other_receipt_statuses"] = sorted({
                ref["status"] for ref in entry["evidence_references"]
                if isinstance(ref["status"], str)})
            if diagnostic is not None:
                row["source_diagnostic"] = {
                    "receipt": ref["path"], "receipt_sha256": ref["sha256"],
                    "active_output_comparisons": diagnostic["active_output_comparisons"],
                    "compiler_selected_output_qualification": diagnostic[
                        "compiler_selected_output_qualification"]}
        rows.append(row)
    counts = Counter(row["status"] for row in rows)
    return {
        "schema": "mx_gemmini.nicolas_mx_regeneration_audit.v3",
        "scope": ("matched generated matrix, connected-chain, and specialized results cover selected "
                  "Spike output paths, not complete C control flow, RTL timing, or FPGA parity"),
        "rtl_revision": inventory["rtl_revision"],
        "source_inventory_sha256": _sha(INVENTORY),
        "programs": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "entries": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.inventory.resolve() != INVENTORY.resolve():
        parser.error("regeneration audit currently requires the archived pinned inventory")
    report = build_report(json.loads(args.inventory.read_text()))
    content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.out.is_file() or args.out.read_text() != content:
            raise SystemExit("Nicolas regeneration audit differs from current receipts")
    else:
        if args.out.exists():
            parser.error(f"refusing to overwrite {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(content)
    print(args.out)


if __name__ == "__main__":
    main()
