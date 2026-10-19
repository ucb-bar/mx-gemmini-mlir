"""The full-model native issuer is compiled from Merlin's checked loop IR."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from merlin.sched.ir import parse_kernel

from mx_gemmini_support.dense_contraction import (
    emit_dense_batch_contraction,
    emit_dense_contraction,
)
from mx_gemmini_support.merlin_schedule_policy import build_policy_plan, selected_policy
from mx_gemmini_support.scheduled_native_issuer import emit_scheduled_native_issuer
from mx_gemmini_support.target_profile import load_profile

ROOT = Path(__file__).resolve().parents[1]
PROFILE = load_profile(
    ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json")


@pytest.mark.parametrize("shape,precision,tiles", [
    ((32, 352, 128), "FP4", (32, 32, 128)),
    ((32, 128, 352), "FP4", (32, 128, 32)),
    ((128, 2048, 2048), "FP4", (128, 128, 512)),
    ((128, 160, 64), "FP8", (128, 32, 64)),
    ((1024, 768, 768), "FP4", (128, 128, 256)),
])
def test_checked_schedule_round_trips_and_drives_native_loops(shape, precision, tiles):
    result = emit_scheduled_native_issuer(PROFILE, shape, precision=precision)
    kernel = parse_kernel(result.text)
    assert kernel.digest() == result.digest
    assert result.issuer.tile_mnk == tiles
    assert result.chunks == tiles
    assert "for row in 0.." in result.text
    assert "for chunk in 0.." in result.text
    assert "for wave in 0.." in result.text
    assert "mod(" in result.text
    source = result.issuer.source
    assert "MX_CMD(31," in source  # loop-managed scales
    assert "MX_CMD(9," in source and "MX_CMD(8," in source
    assert "MX_CMD(27," not in source  # no manual scale uploads
    assert "MX_CMD(2," not in source and "MX_CMD(3," not in source
    assert source.count("for (int ") == 3


def test_tiling_changes_schedule_digest_and_generated_c():
    a = emit_scheduled_native_issuer(PROFILE, (128, 128, 512), precision="FP4")
    b = emit_scheduled_native_issuer(PROFILE, (128, 256, 512), precision="FP4")
    assert a.digest != b.digest
    assert a.issuer.source != b.issuer.source


def test_explicit_non_power_of_two_row_partition_reaches_actual_schedule():
    legacy = emit_scheduled_native_issuer(PROFILE, (160, 2048, 2048), precision="FP8")
    selected = emit_scheduled_native_issuer(PROFILE, (160, 2048, 2048), precision="FP8", tile_mnk=(80, 128, 512))
    assert legacy.chunks == (160, 64, 512)
    assert selected.chunks == (80, 128, 512)
    assert parse_kernel(selected.text).digest() == selected.digest != legacy.digest
    assert "for row in 0..2" in selected.text and "for chunk in 0..16" in selected.text


@pytest.mark.parametrize("tiles", [(80, 128, 512), (160, 128, 512), (160, 64, 1024),
                                  (32, 512, 512), (32, 32, 96), (True, 64, 512), (0, 64, 512)])
def test_explicit_native_partition_cannot_bypass_capacity_or_alignment(tiles):
    # FP4 rejects 80 rows (not a full 32-row mesh tile); other cases exceed
    # accumulator/scale capacity, fail divisibility, or have invalid extents.
    with pytest.raises(ValueError, match="native tiles"):
        emit_scheduled_native_issuer(PROFILE, (160, 2048, 2048), precision="FP4", tile_mnk=tiles)


@pytest.mark.parametrize("shape,tiles", [((160, 2048, 2048), (80, 128, 512)),
                                        ((160, 160, 64), (160, 32, 64)),
                                        ((192, 384, 1024), (96, 128, 512)),
                                        ((128, 2048, 2048), (128, 128, 512))])
def test_movement_search_uses_source_divisors_and_preserves_accumulation_partition(shape, tiles):
    result = emit_scheduled_native_issuer(PROFILE, shape, precision="FP8", tile_policy="movement")
    selected = result.issuer.plan["tile_selection"]
    assert result.chunks == tiles and result.chunks[2] == selected["baseline"][2]
    assert selected["complete"]
    assert all(a <= b for a, b in zip(selected["selected_work"], selected["baseline_work"]))
    explicit = emit_scheduled_native_issuer(PROFILE, shape, precision="FP8", tile_mnk=tiles)
    assert result.issuer.source == explicit.issuer.source and result.digest == explicit.digest


def test_movement_search_does_not_promote_unqualified_precision():
    legacy = emit_scheduled_native_issuer(PROFILE, (160, 5632, 2048), precision="FP4")
    selected = emit_scheduled_native_issuer(PROFILE, (160, 5632, 2048), precision="FP4", tile_policy="movement")
    assert legacy.issuer.source == selected.issuer.source and legacy.digest == selected.digest
    assert selected.issuer.plan["tile_selection"]["selection"] == "legacy_unqualified_precision"


def test_dense_and_batch_catalog_entries_select_checked_schedule():
    dense = emit_dense_contraction(
        PROFILE, (8, 344, 128), precision="FP4", symbol="dense",
        issuer_symbol="dense_issue", schedule="native")
    batch = emit_dense_batch_contraction(
        PROFILE, (32, 128, 160, 64), precision="FP8", symbol="batch",
        issuer_symbol="batch_issue", schedule="native")
    for result in (dense, batch):
        assert result.issuer_mode == "merlin_mk_native_dram_loop"
        assert result.schedule_text is not None
        assert parse_kernel(result.schedule_text).digest() == result.schedule_digest
        assert result.instruction_set_digest is not None
        assert result.shape_plan.tile_mnk == result.issuer.tile_mnk


def test_profile_and_schedule_selection_fail_closed():
    fp8_only = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3VpuGemminiRocketConfig.json")
    with pytest.raises(ValueError, match="compute mode"):
        emit_scheduled_native_issuer(fp8_only, (32, 32, 32), precision="FP4")
    with pytest.raises(ValueError, match="aligned"):
        emit_scheduled_native_issuer(PROFILE, (33, 32, 64), precision="FP4")
    with pytest.raises(ValueError, match="unknown MX dense schedule"):
        emit_dense_contraction(PROFILE, (32, 32, 64), precision="FP4",
                               symbol="dense", issuer_symbol="issue", schedule="unknown")


def test_schedule_dialect_policy_is_consumed_by_issuer_selection():
    native = build_policy_plan("dense", schedule="native")
    serial = build_policy_plan("dense", schedule="serial")
    assert selected_policy(native, symbol="dense") == "native"
    assert selected_policy(serial, symbol="dense") == "serial"
    assert emit_dense_contraction(
        PROFILE, (32, 32, 64), precision="FP4", symbol="dense",
        issuer_symbol="issue", schedule=selected_policy(native, symbol="dense")
    ).issuer_mode == "merlin_mk_native_dram_loop"
    assert emit_dense_contraction(
        PROFILE, (32, 32, 64), precision="FP4", symbol="dense",
        issuer_symbol="issue", schedule=selected_policy(serial, symbol="dense")
    ).issuer_mode == "unrolled"
    with pytest.raises(ValueError, match="exact catalog binding"):
        selected_policy(native, symbol="other")
    with pytest.raises(ValueError, match="unsupported policy"):
        selected_policy(native.replace("mx_native_dram_loop", "invented"), symbol="dense")


def test_movement_policy_dialect_reaches_dense_and_batched_catalog_lowering():
    policy = build_policy_plan("dense", schedule="movement")
    selected = selected_policy(policy, symbol="dense")
    dense = emit_dense_contraction(PROFILE, (160, 2048, 2048), precision="FP8", symbol="dense",
                                   issuer_symbol="issue", schedule=selected)
    assert dense.shape_plan.tile_mnk == (80, 128, 512)
    batch = emit_dense_batch_contraction(PROFILE, (2, 160, 160, 64), precision="FP8", symbol="dense",
                                         issuer_symbol="issue", schedule=selected)
    assert batch.shape_plan.tile_mnk == (160, 32, 64)
    assert batch.issuer.plan["tile_selection"]["selected"] == batch.issuer.plan["tile_selection"]["baseline"]


def test_archived_native_model_sites_retain_spike_goldens():
    archive = ROOT / "docs/evidence/merlin_mk_native_model_sites_20261011"
    receipt = json.loads((archive / "receipt.json").read_text())
    assert receipt["status"] == "three_model_derived_native_sites_matched_on_pinned_spike"
    assert len(receipt["cases"]) == 3
    for case in receipt["cases"]:
        folder = archive / case["case"]
        for name, digest in case["artifacts"].items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
        assert case["expected_log"] in (folder / "spike.log").read_text()
        assert parse_kernel((folder / "schedule.mk").read_text()).text() == (
            folder / "schedule.mk").read_text()
    full = receipt["full_tiny_prefill_catalog"]
    assert full["matched_contractions"] == 199
    assert full["unique_kernels"] == 7
    assert full["status"] == "guest_done_output_matches_serial_on_pinned_spike"
    assert len(full["kernels"]) == 7
    policy_path = archive / "tiny_prefill_policy.schedule.mlir"
    assert hashlib.sha256(policy_path.read_bytes()).hexdigest() == full["policy_artifact_sha256"]
    assert selected_policy(policy_path.read_text(), symbol="mx_dense_kernel_0") == "native"
    execution = receipt["full_tiny_native_prefill_execution"]
    log = (archive / "tiny_native_prefill_spike.log").read_bytes()
    assert hashlib.sha256(log).hexdigest() == execution["spike_log_sha256"]
    assert execution["catalog_object_sha256"] == full["object_sha256"]
    assert execution["output_f32le_sha256"] == execution["serial_output_f32le_sha256"]
    assert b"OUT_SHA256 f32le 32000 128000 " + execution["output_f32le_sha256"].encode() in log
    assert b"METRIC memref_rank_mismatch 0\nDONE\n" in log
    assert execution["native_simulated_cycles"] < execution["serial_simulated_cycles"]
    row = receipt["row_tile_parity"]
    assert row["status"] == "native_equals_serial_on_pinned_spike"
    assert row["compared_bf16_outputs"] == row["nonzero_outputs"] == 32768
    for name, key in (("schedule.mk", "schedule_sha256"), ("spike.log", "log_sha256")):
        assert hashlib.sha256((archive / "fp4_m1024_serial_parity" / name).read_bytes()).hexdigest() == row[key]
