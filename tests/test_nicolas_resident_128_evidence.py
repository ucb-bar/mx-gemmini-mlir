"""Check source-bound 128³ resident MM2 lowering and direct Spike parity."""

from __future__ import annotations

from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.command_ir import Command, emit_c
from mx_gemmini_support.plain_chain_128 import lower_plain_chain_128
from mx_gemmini_support.resident_pair_graph import INPUTS, lower_connected_fp8_pair
from mx_gemmini_support.resident_pair_plan import (lower_first_fp8_resident,
                                                    plan_fp8_resident_pair)
from mx_gemmini_support.resident_lowering import (lower_single_resident_contract,
                                                  validate_resident_contract)
from mx_gemmini_support.target_profile import load_profile, profile_sha256


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_resident_mm2_128_266c593"
FRONTEND = ROOT / "docs/evidence/nicolas_plain_chain_128_model2mlir_e9ded36"
CONNECTED = ROOT / "docs/evidence/nicolas_connected_plain_chain_128_266c593"
FRESH = ROOT / "docs/evidence/nicolas_connected_plain_chain_128_fresh_checkout_ae945d0"
PAIR_PLAN = ROOT / "docs/evidence/nicolas_resident_pair_plan_b1b5882"
ROW_PREFIX = ROOT / "docs/evidence/nicolas_connected_plain_chain_64x128_d512fc2"
ROW_PREFIX_FRESH = ROOT / "docs/evidence/nicolas_connected_plain_chain_64x128_fresh_5cf6e1a"
PREFIX_LADDER = ROOT / "docs/evidence/nicolas_plain_chain_prefix_ladder_4cf23ef"
GENERIC_PAIR = ROOT / "docs/evidence/nicolas_generic_pair_1f8c6ab"
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(name: str) -> bytes:
    return gzip.decompress((EVIDENCE / f"{name}.gz").read_bytes())


def _connected(name: str) -> bytes:
    return gzip.decompress((CONNECTED / f"{name}.gz").read_bytes())


def _row_prefix(name: str) -> bytes:
    return gzip.decompress((ROW_PREFIX / f"{name}.gz").read_bytes())


@pytest.mark.parametrize("m", (16, 32, 48, 80, 96, 112))
def test_complete_source_row_prefix_ladder_matches_spike(m: int) -> None:
    index = json.loads((PREFIX_LADDER / "index.json").read_text())
    case = index["cases"][str(m)]
    directory = PREFIX_LADDER / f"m{m}"
    first = json.loads((directory / "artifact_manifest.json").read_text())
    second = json.loads((directory / "reproduction_manifest.json").read_text())
    capture = json.loads((directory / "receipt.json").read_text())

    def read(name: str) -> bytes:
        plain = directory / name
        return plain.read_bytes() if plain.is_file() else gzip.decompress(
            (directory / f"{name}.gz").read_bytes())

    assert first == second
    assert first["compiler_revision"] == index["compiler_revision"] == (
        "4cf23ef89cbd587910751d60a56f6211a4eb569b")
    assert capture["model2mlir_revision"] == index["model2mlir_revision"]
    assert capture["output_rows"] == m
    assert [site["shape"] for site in capture["sites"]] == [
        [m, 128, 128], [m, 128, 128]]
    assert first["status"] == case["status"] == (
        "source_prefix_connected_chain_matched_on_pinned_spike")
    assert first["spike_exit_code"] == 0
    assert (first["compared_c1_fp8_codes"], first["compared_c1_e8m0_scales"],
            first["compared_fp8_codes"], first["compared_e8m0_scales"]) == (
            m * 128, m * 4, m * 128, m * 4)
    for name, digest in case["files_sha256"].items():
        assert _sha(read(name)) == digest, (m, name)
    assert (f"lowered connected {m}x128: C1 0 codes 0 scales; "
            "C2 0 codes 0 scales").encode() in read("spike.log")
    assert read("a1_activation.bin") == _connected("a1_activation.bin")[:m * 128]
    full_scales = _connected("a1_scales.bin")
    assert read("a1_scales.bin") == b"".join(
        full_scales[group * 128:group * 128 + m] for group in range(4))
    for name in ("c1_codes_ref", "c2_codes_ref"):
        assert read(f"{name}.bin") == _connected(f"{name}.bin")[:m * 128]
    for name in ("c1_scales_ref", "c2_scales_ref"):
        assert read(f"{name}.bin") == _connected(f"{name}.bin")[:m * 4]
    resources = {path.name.removesuffix(".bin.gz"):
                 read(path.name.removesuffix(".gz"))
                 for path in directory.glob("*.bin.gz")}
    commands = lower_plain_chain_128(
        read("connected_chain.mlir").decode(),
        read("nicolas_chain.profile_bound.mlir").decode(),
        json.loads(read("quantization_manifest.json")),
        load_profile(PROFILE), resources,
        source_sha256=index["source_sha256"],
        header_sha256=index["header_sha256"])
    buffers = tuple(sorted({operand.buffer for command in commands
                            if isinstance(command, Command)
                            for operand in (command.rs1, command.rs2)
                            if operand.buffer is not None}))
    assert emit_c(commands, transport="rocket_rocc", buffers=buffers).encode() == (
        read("mx_issue.c"))
    uploads = [command.rs1.buffer for command in commands
               if isinstance(command, Command) and command.funct == 2]
    assert uploads.count("a1_activation") == m // 2
    assert uploads.count("b1_weight") == uploads.count("b2_weight") == 64
    readouts = [command.rs1.buffer for command in commands
                if isinstance(command, Command) and command.funct == 3]
    assert readouts == ["c1_tiled_observed"] * (m // 2) + ["c2_tiled"] * (m // 2)


def test_connected_pair_lowerer_accepts_other_sites_and_runtime_symbols() -> None:
    directory = PREFIX_LADDER / "m96"

    def archived(name: str) -> bytes:
        return gzip.decompress((directory / f"{name}.gz").read_bytes())

    mlir = archived("connected_chain.mlir").decode()
    mlir = (mlir.replace("@nicolas_plain_chain_128", "@user_pair")
            .replace("functional:matmul_1", "user:mm2")
            .replace("functional:matmul", "user:mm1")
            .replace('weight_buffer = "b2_weight"',
                     'weight_buffer = "runtime_b2_weight"')
            .replace('weight_scales_buffer = "b2_scales"',
                     'weight_scales_buffer = "runtime_b2_scales"')
            .replace('output_scales_buffer = "c2_scales"',
                     'output_scales_buffer = "runtime_c2_scales"'))
    buffers = {name: f"runtime_{name}" for name in INPUTS}
    resources = {buffers[name]: archived(f"{name}.bin") for name in INPUTS}
    profile = load_profile(PROFILE)
    pair = lower_connected_fp8_pair(
        mlir, profile, resources, buffers=buffers,
        c1_scales="runtime_c1_scales",
        c1_tiled_observed="runtime_c1_observed", c2_tiled="runtime_c2_tiled")
    assert (pair.first_site, pair.second_site) == ("user:mm1", "user:mm2")
    assert (pair.plan.m, pair.plan.n, pair.plan.k) == (96, 128, 128)
    assert pair.plan.b_row == 15360
    command_buffers = {operand.buffer for command in pair.commands
                       if isinstance(command, Command)
                       for operand in (command.rs1, command.rs2)
                       if operand.buffer is not None}
    assert command_buffers == set(buffers.values()) | {
        "runtime_c1_scales", "runtime_c1_observed",
        "runtime_c2_scales", "runtime_c2_tiled"}
    with pytest.raises(ValueError, match="SSA tensor edges"):
        lower_connected_fp8_pair(
            mlir.replace('"mx_gemmini.contract"(%a1, %a1s, %b1, %b1s)',
                         '"mx_gemmini.contract"(%a1, %a1s, %b2, %b1s)'),
            profile, resources, buffers=buffers,
            c1_scales="runtime_c1_scales",
            c1_tiled_observed="runtime_c1_observed", c2_tiled="runtime_c2_tiled")
    with pytest.raises(ValueError, match="MM2 buffers differ"):
        lower_connected_fp8_pair(
            mlir.replace('output_scales_buffer = "runtime_c2_scales"',
                         'output_scales_buffer = "runtime_a1_scales"'),
            profile, resources, buffers=buffers,
            c1_scales="runtime_c1_scales",
            c1_tiled_observed="runtime_c1_observed", c2_tiled="runtime_c2_tiled")


def test_published_reusable_pair_lowerer_preserves_spike_programs() -> None:
    index = json.loads((GENERIC_PAIR / "index.json").read_text())
    assert index["schema"] == "mx_gemmini.generic_resident_pair_fresh_reproduction.v1"
    assert index["compiler_revision"] == (
        "1f8c6ab33e19a39ae5d5ef8eb93bb2414ce7a763")
    assert index["fresh_checkout"] == "fresh_m96"
    assert set(index["cases"]) == {"m16", "m96", "m128", "fresh_m96"}
    for name, case in index["cases"].items():
        baseline = json.loads((ROOT / case["baseline"]).read_text())
        actual = json.loads((GENERIC_PAIR / name / "artifact_manifest.json").read_text())
        assert actual["compiler_revision"] == index["compiler_revision"]
        assert actual["status"] == case["status"]
        assert actual["spike_exit_code"] == 0
        for field in case["stable_fields_equal_to_baseline"]:
            assert actual[field] == baseline[field], (name, field)
        for file, digest in case["files_sha256"].items():
            assert _sha((GENERIC_PAIR / name / file).read_bytes()) == digest
def test_source_derived_64x128_connected_chain_matches_stock_spike() -> None:
    index = json.loads((ROW_PREFIX / "index.json").read_text())
    first = json.loads((ROW_PREFIX / "artifact_manifest.json").read_text())
    second = json.loads((ROW_PREFIX / "reproduction_manifest.json").read_text())
    capture = json.loads((ROW_PREFIX / "receipt.json").read_text())
    assert first == second
    assert first["status"] == index["status"] == (
        "source_prefix_connected_chain_matched_on_pinned_spike")
    assert first["schema"] == "mx_gemmini.nicolas_connected_plain_chain_64x128.v1"
    assert first["spike_exit_code"] == 0
    assert first["compiler_revision"] == index["compiler_revision"] == (
        "d512fc2491ab67ed2b822c39d7897fb0d2976bcf")
    assert capture["model2mlir_revision"] == index["model2mlir_revision"] == (
        "e9ded36eb85abf2d9097ac4dc11457c825853388")
    assert capture["output_rows"] == 64
    assert [site["shape"] for site in capture["sites"]] == [
        [64, 128, 128], [64, 128, 128]]
    assert (first["compared_c1_fp8_codes"], first["compared_c1_e8m0_scales"],
            first["compared_fp8_codes"], first["compared_e8m0_scales"]) == (
            8192, 256, 8192, 256)
    for name, digest in index["files_sha256"].items():
        data = ((ROW_PREFIX / name).read_bytes() if (ROW_PREFIX / name).exists()
                else _row_prefix(name))
        assert _sha(data) == digest, name
    assert (b"lowered connected 64x128: C1 0 codes 0 scales; "
            b"C2 0 codes 0 scales") in _row_prefix("spike.log")
    assert _row_prefix("a1_activation.bin") == _connected("a1_activation.bin")[:8192]
    full_scales = _connected("a1_scales.bin")
    assert _row_prefix("a1_scales.bin") == b"".join(
        full_scales[group * 128:group * 128 + 64] for group in range(4))
    for name in ("c1_codes_ref", "c2_codes_ref"):
        assert _row_prefix(f"{name}.bin") == _connected(f"{name}.bin")[:8192]
    for name in ("c1_scales_ref", "c2_scales_ref"):
        assert _row_prefix(f"{name}.bin") == _connected(f"{name}.bin")[:256]
    frontend = _row_prefix("nicolas_chain.profile_bound.mlir").decode()
    manifest = json.loads(_row_prefix("quantization_manifest.json"))
    resources = {path.name.removesuffix(".bin.gz"):
                 _row_prefix(path.name.removesuffix(".gz"))
                 for path in ROW_PREFIX.glob("*.bin.gz")}
    commands = lower_plain_chain_128(
        _row_prefix("connected_chain.mlir").decode(), frontend, manifest,
        load_profile(PROFILE), resources,
        source_sha256=index["source_sha256"],
        header_sha256=index["header_sha256"])
    buffers = tuple(sorted({operand.buffer for command in commands
                            if isinstance(command, Command)
                            for operand in (command.rs1, command.rs2)
                            if operand.buffer is not None}))
    assert emit_c(commands, transport="rocket_rocc", buffers=buffers).encode() == (
        _row_prefix("mx_issue.c"))
    uploads = [command.rs1.buffer for command in commands
               if isinstance(command, Command) and command.funct == 2]
    assert uploads.count("a1_activation") == 32
    assert uploads.count("b1_weight") == uploads.count("b2_weight") == 64
    readouts = [command.rs1.buffer for command in commands
                if isinstance(command, Command) and command.funct == 3]
    assert readouts == ["c1_tiled_observed"] * 32 + ["c2_tiled"] * 32


def test_fresh_published_checkout_replays_rectangular_spike_program() -> None:
    index = json.loads((ROW_PREFIX_FRESH / "index.json").read_text())
    actual = json.loads((ROW_PREFIX_FRESH / "artifact_manifest.json").read_text())
    baseline = json.loads((ROW_PREFIX / "artifact_manifest.json").read_text())
    assert index["schema"] == (
        "mx_gemmini.nicolas_connected_plain_chain_64x128_fresh_reproduction.v1")
    assert actual["compiler_revision"] == index["compiler_revision"] == (
        "5cf6e1a1953fdcebe5d7d1819d0a045ece4b8a63")
    assert index["baseline_compiler_revision"] == baseline["compiler_revision"]
    assert actual["status"] == index["status"] == (
        "source_prefix_connected_chain_matched_on_pinned_spike")
    assert actual["spike_exit_code"] == 0
    for key in index["stable_fields_equal_to_baseline"]:
        assert actual[key] == baseline[key], key
    for name, digest in index["files_sha256"].items():
        assert _sha((ROW_PREFIX_FRESH / name).read_bytes()) == digest


def test_archived_typed_mm2_and_spike_result_match_source() -> None:
    index = json.loads((EVIDENCE / "index.json").read_text())
    first = json.loads((EVIDENCE / "artifact_manifest.json").read_text())
    reproduced = json.loads((EVIDENCE / "reproduction_manifest.json").read_text())
    profile = load_profile(PROFILE)
    assert index["schema"] == "mx_gemmini.nicolas_resident_mm2_128_archive.v1"
    assert index["profile_sha256"] == profile_sha256(profile)
    assert index["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert profile["resources"]["requantizer"] is True
    assert profile["resources"]["spad_requant"] is False
    assert index["compared_fp8_codes"] == 16384
    assert index["compared_e8m0_scales"] == 512
    assert first["status"] == reproduced["status"] == (
        "source_resident_mm2_matched_on_pinned_spike")
    for key in ("source_sha256", "header_sha256", "profile_sha256",
                "bound_mlir_sha256", "elf_sha256", "extension_sha256",
                "spike_log_sha256", "files_sha256", "object_sha256"):
        assert first[key] == reproduced[key], key
    assert first["spike_exit_code"] == reproduced["spike_exit_code"] == 0
    for name, digest in index["files_sha256"].items():
        assert _sha(_read(name)) == digest
    assert _sha(_read("resident_mm2.mlir")) == first["bound_mlir_sha256"]
    assert _sha(_read("mx_program.elf")) == first["elf_sha256"]
    assert _sha(_read("spike.log")) == first["spike_log_sha256"]
    assert (b"lowered resident MM2 128x128: 0 FP8 code mismatches, "
            b"0 E8M0 scale mismatches") in _read("spike.log")


def test_latest_model2mlir_captures_both_plain_chain_sites() -> None:
    index = json.loads((FRONTEND / "index.json").read_text())
    first = json.loads((FRONTEND / "receipt.json").read_text())
    second = json.loads((FRONTEND / "reproduction_receipt.json").read_text())
    assert first == second
    assert first["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert first["matrix_dim"] == 128
    assert first["profile_sha256"] == index["profile_sha256"]
    assert first["source_sha256"] == json.loads(
        (EVIDENCE / "index.json").read_text())["source_sha256"]
    assert first["opaque_calls"] == {}
    assert [(site["site_id"], site["status"], site["shape"])
            for site in first["sites"]] == [
                ("functional:matmul", "quantized", [128, 128, 128]),
                ("functional:matmul_1", "quantized", [128, 128, 128]),
            ]
    for name, digest in index["files_sha256"].items():
        assert _sha(gzip.decompress((FRONTEND / f"{name}.gz").read_bytes())) == digest
    bound = gzip.decompress((FRONTEND / "nicolas_chain.profile_bound.mlir.gz").read_bytes())
    assert bound.count(b'"mx_gemmini.contract"') == 2


def test_connected_plain_chain_executes_both_sites_without_c1_reload() -> None:
    index = json.loads((CONNECTED / "index.json").read_text())
    first = json.loads((CONNECTED / "artifact_manifest.json").read_text())
    second = json.loads((CONNECTED / "reproduction_manifest.json").read_text())
    assert index["schema"] == "mx_gemmini.nicolas_connected_plain_chain_128_archive.v1"
    assert first["status"] == second["status"] == (
        "source_connected_chain_matched_on_pinned_spike")
    assert first["spike_exit_code"] == second["spike_exit_code"] == 0
    assert first["compiler_revision"] == index["compiler_revision"]
    assert first["frontend_mlir_sha256"] == index["frontend_mlir_sha256"]
    for key in ("bound_mlir_sha256", "elf_sha256", "extension_sha256",
                "spike_log_sha256", "files_sha256", "object_sha256"):
        assert first[key] == second[key], key
    assert (index["compared_c1_fp8_codes"], index["compared_c1_e8m0_scales"],
            index["compared_fp8_codes"], index["compared_e8m0_scales"]) == (
            16384, 512, 16384, 512)
    for name, digest in index["files_sha256"].items():
        assert _sha(_connected(name)) == digest
    assert (b"lowered connected 128x128: C1 0 codes 0 scales; "
            b"C2 0 codes 0 scales") in _connected("spike.log")
    resources = {path.name.removesuffix(".bin.gz"):
                 _connected(path.name.removesuffix(".gz"))
                 for path in CONNECTED.glob("*.bin.gz")}
    assert "c1_tiled" not in resources
    assert "c1_act_scales" not in resources
    frontend = gzip.decompress(
        (FRONTEND / "nicolas_chain.profile_bound.mlir.gz").read_bytes()).decode()
    manifest = json.loads(gzip.decompress(
        (FRONTEND / "quantization_manifest.json.gz").read_bytes()))
    mlir = _connected("connected_chain.mlir").decode()
    profile = load_profile(PROFILE)
    args = {"source_sha256": index["source_sha256"],
            "header_sha256": index["header_sha256"]}
    commands = lower_plain_chain_128(mlir, frontend, manifest, profile, resources,
                                      **args)
    buffers = tuple(sorted({operand.buffer for command in commands
                            if isinstance(command, Command)
                            for operand in (command.rs1, command.rs2)
                            if operand.buffer is not None}))
    assert emit_c(commands, transport="rocket_rocc", buffers=buffers).encode() == (
        _connected("mx_issue.c"))
    uploads = [command.rs1.buffer for command in commands
               if isinstance(command, Command) and command.funct == 2]
    assert uploads.count("a1_activation") == 64
    assert uploads.count("b1_weight") == 64
    assert uploads.count("b2_weight") == 64
    assert set(uploads) == {"a1_activation", "b1_weight", "b2_weight"}
    readouts = [command.rs1.buffer for command in commands
                if isinstance(command, Command) and command.funct == 3]
    assert readouts == ["c1_tiled_observed"] * 64 + ["c2_tiled"] * 64
    with pytest.raises(ValueError, match="SSA tensor edges"):
        lower_plain_chain_128(
            mlir.replace('"mx_gemmini.resident_contract"(%c1, %c1s, %b2, %b2s)',
                         '"mx_gemmini.resident_contract"(%b2, %c1s, %c1, %b2s)'),
            frontend, manifest, profile, resources, **args)
    changed = dict(resources)
    changed["a1_activation"] = bytes([resources["a1_activation"][0] ^ 1]) + (
        resources["a1_activation"][1:])
    with pytest.raises(ValueError, match="source binding differs"):
        lower_plain_chain_128(mlir, frontend, manifest, profile, changed, **args)


def test_resident_pair_planner_derives_counts_and_rejects_overlapping_lifetimes() -> None:
    profile = load_profile(PROFILE)
    small = plan_fp8_resident_pair(
        profile, shape=(64, 64, 64), a_row=0, c1_row=512, c2_row=1024)
    assert (small.a_rows, small.b_rows, small.c_rows, small.b_row) == (
        256, 256, 256, 16128)
    assert (small.m_tiles, small.n_tiles, small.k_tiles,
            small.a_scale_bytes, small.output_scale_bytes) == (4, 4, 4, 128, 128)
    commands = lower_first_fp8_resident(
        small, activation_buffer="a1_activation",
        activation_scales_buffer="a1_scales", weight_buffer="b1_weight",
        weight_scales_buffer="b1_scales", output_scales_buffer="c1_scales")
    assert sum(isinstance(item, Command) and item.funct == 2 for item in commands) == 32
    launch = next(item for item in commands if isinstance(item, Command) and
                  item.funct == 8)
    assert launch.rs2.immediate >> 32 == small.c1_row
    non_square = plan_fp8_resident_pair(
        profile, shape=(64, 128, 64), a_row=0, c1_row=1024, c2_row=2048)
    assert non_square.b_row == 15872
    with pytest.raises(ValueError, match="needs square N/K"):
        lower_first_fp8_resident(
            non_square, activation_buffer="a1_activation",
            activation_scales_buffer="a1_scales", weight_buffer="b1_weight",
            weight_scales_buffer="b1_scales", output_scales_buffer="c1_scales")
    row_prefix = plan_fp8_resident_pair(
        profile, shape=(64, 128, 128), a_row=0, c1_row=2048, c2_row=4096)
    assert (row_prefix.a_rows, row_prefix.b_rows, row_prefix.c_rows,
            row_prefix.a_scale_bytes, row_prefix.output_scale_bytes) == (
            512, 1024, 512, 256, 256)
    row_commands = lower_first_fp8_resident(
        row_prefix, activation_buffer="a1_activation",
        activation_scales_buffer="a1_scales", weight_buffer="b1_weight",
        weight_scales_buffer="b1_scales", output_scales_buffer="c1_scales")
    assert sum(isinstance(item, Command) and item.funct == 2
               for item in row_commands) == 96
    with pytest.raises(ValueError, match="row lifetimes overlap"):
        plan_fp8_resident_pair(
            profile, shape=(128, 128, 128), a_row=0, c1_row=512, c2_row=4096)
    with pytest.raises(ValueError, match="DIM16 alignment"):
        plan_fp8_resident_pair(
            profile, shape=(128, 128, 128), a_row=0, c1_row=2049, c2_row=4096)
    with pytest.raises(ValueError, match="accumulator capacity"):
        plan_fp8_resident_pair(
            profile, shape=(256, 256, 256), a_row=0, c1_row=4096, c2_row=8192)
    profile["resources"]["requantizer"] = False
    with pytest.raises(ValueError, match="requantizer profile"):
        plan_fp8_resident_pair(
            profile, shape=(128, 128, 128), a_row=0, c1_row=2048, c2_row=4096)


def test_published_pair_planner_preserves_full_spike_program() -> None:
    index = json.loads((PAIR_PLAN / "index.json").read_text())
    actual = json.loads((PAIR_PLAN / "artifact_manifest.json").read_text())
    baseline = json.loads((CONNECTED / "artifact_manifest.json").read_text())
    saved = json.loads((PAIR_PLAN / "plan.json").read_text())
    profile = load_profile(PROFILE)
    plan = plan_fp8_resident_pair(
        profile, shape=(128, 128, 128), a_row=0, c1_row=2048, c2_row=4096)
    assert saved == asdict(plan)
    assert (saved["b_row"], saved["a_rows"], saved["c_rows"],
            saved["a_scale_bytes"]) == (15360, 1024, 1024, 512)
    assert index["schema"] == "mx_gemmini.nicolas_resident_pair_plan_reproduction.v1"
    assert index["compiler_revision"] == (
        "b1b58821deacd3e660baf8605c170f2c99a01005")
    assert index["status"] == actual["status"] == (
        "source_connected_chain_matched_on_pinned_spike")
    assert index["baseline_compiler_revision"] == baseline["compiler_revision"]
    assert actual["spike_exit_code"] == 0
    for key in index["stable_fields_equal_to_baseline"]:
        assert actual[key] == baseline[key], key
    for name, digest in index["files_sha256"].items():
        assert _sha((PAIR_PLAN / name).read_bytes()) == digest


def test_fresh_published_checkout_rebuilds_and_reproduces_connected_chain() -> None:
    index = json.loads((FRESH / "index.json").read_text())
    actual = json.loads((FRESH / "artifact_manifest.json").read_text())
    baseline = json.loads((CONNECTED / "artifact_manifest.json").read_text())
    assert index["schema"] == "mx_gemmini.nicolas_connected_plain_chain_128_fresh_checkout.v1"
    assert index["status"] == (
        "fresh_checkout_built_and_connected_chain_reproduced_on_pinned_spike")
    assert index["compiler_revision"] == (
        "ae945d0d10e1234945b2f7bb68242fed54a9295b")
    assert index["baseline_compiler_revision"] == baseline["compiler_revision"]
    assert actual["compiler_revision"] == index["compiler_revision"]
    assert actual["spike_exit_code"] == 0
    assert (index["compared_c1_fp8_codes"], index["compared_c1_e8m0_scales"],
            index["compared_c2_fp8_codes"], index["compared_c2_e8m0_scales"]) == (
            16384, 512, 16384, 512)
    for key in index["stable_fields_equal_to_baseline"]:
        assert actual[key] == baseline[key], key
    for name, digest in index["files_sha256"].items():
        assert _sha((FRESH / name).read_bytes()) == digest
    assert index["spike_log_sha256"] == _sha((FRESH / "spike.log").read_bytes())


def test_typed_mm2_uses_resident_c1_and_source_placement() -> None:
    profile = load_profile(PROFILE)
    mlir = _read("resident_mm2.mlir").decode()
    commands = lower_single_resident_contract(mlir, profile)
    transfers = [item for item in commands if isinstance(item, Command) and
                 item.funct == 2]
    assert len(transfers) == 64
    assert all(item.rs1.buffer == "b2_weight" for item in transfers)
    assert transfers[0].rs2.immediate & 0x3fff == 15360
    assert transfers[-1].rs2.immediate & 0x3fff == 16368
    execute = next(item for item in commands if isinstance(item, Command) and
                   item.funct == 24)
    assert (execute.rs1.immediate, execute.rs2.immediate) == (2048, 16384)
    compute = next(item for item in commands if isinstance(item, Command) and
                   item.funct == 8)
    assert compute.rs2.immediate >> 32 == 4096
    assert (b'"mx_gemmini.resident_contract"(%c1, %c1s, %b2, %b2s)'
            in _read("resident_mm2.mlir"))


def test_wrong_lifetime_or_ssa_handoff_fails_closed() -> None:
    profile = load_profile(PROFILE)
    mlir = _read("resident_mm2.mlir").decode()
    with pytest.raises(ValueError, match="scratchpad tile placement"):
        lower_single_resident_contract(
            mlir.replace("output_row = 4096 : i32", "output_row = 15300 : i32"),
            profile)
    with pytest.raises(ValueError, match="scratchpad tile placement"):
        lower_single_resident_contract(
            mlir.replace("output_row = 4096 : i32", "output_row = 4097 : i32"),
            profile)
    with pytest.raises(ValueError, match="SSA tensor edges"):
        lower_single_resident_contract(
            mlir.replace('"mx_gemmini.resident_contract"(%c1, %c1s, %b2, %b2s)',
                         '"mx_gemmini.resident_contract"(%b2, %c1s, %c1, %b2s)'),
            profile)
    attrs = {"activation_row": 2048, "weight_row": 15360, "output_row": 4096,
             "m": 128, "n": 128, "k": 128,
             "activation_format": "fp8_e4m3", "weight_format": "fp8_e4m3",
             "output_format": "fp8_e4m3", "weight_buffer": "b2_weight",
             "weight_scales_buffer": "b2_scales", "output_scales_buffer": "c2_scales"}
    profile["resources"]["requantizer"] = False
    with pytest.raises(ValueError, match="requantizer profile"):
        validate_resident_contract(profile, attrs)
    profile = load_profile(PROFILE)
    with pytest.raises(ValueError, match="SPAD_REQUANT profile"):
        validate_resident_contract(profile, attrs | {"m": 64, "n": 64, "k": 64})
    vpu_profile = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3VpuGemminiRocketConfig.json")
    with pytest.raises(ValueError, match="plain MX profile"):
        validate_resident_contract(vpu_profile, attrs)
