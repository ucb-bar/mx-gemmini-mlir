"""Spike operation timing must be tied to the exact profiled build."""

from __future__ import annotations

import json

import pytest

from tools.report_stage_profile import report_stage_profile


def test_stage_profile_reconciles_guest_ticks_and_build(tmp_path) -> None:
    build = tmp_path / "build"
    build.mkdir()
    (build / "mx_stage_probe.json").write_text(json.dumps({
        "op_profile": True, "build_hash": "abc123"}))
    (build / "op_profile_table.json").write_text(json.dumps([
        {"id": 0, "mlir_op": "func.call", "callee": "@merlin_dev_mx_gemmini_0"},
        {"id": 1, "mlir_op": "func.call", "callee": "@mx_vpu_add_f32"},
        {"id": 2, "mlir_op": "linalg.generic", "family": "elementwise"},
    ]))
    (build / "device").mkdir()
    (build / "device_catalog").mkdir()
    (build / "device/catalog_binding.json").write_text(json.dumps({
        "entry_to_kernel": {"merlin_dev_mx_gemmini_0": "mx_dense_kernel_0"}}))
    (build / "device_catalog/mx_catalog.json").write_text(json.dumps({
        "bindings": [{"symbol": "mx_dense_kernel_0", "region": "matmul_7",
                      "source_operation_ordinal": 17}]}))
    log = tmp_path / "spike.log"
    log.write_text("\n".join((
        "PROF 0 30 1", "PROF 1 20 1", "PROF 2 40 1",
        "METRIC prof_total_ticks 90", "METRIC cycles 100",
        "METRIC build_hash abc123", "METRIC memref_rank_mismatch 0", "DONE", "")))
    report = report_stage_profile(build, log)
    assert report["attribution_coverage"] == 0.9
    assert report["by_owner"] == {"host": 40, "mx": 30, "vpu": 20}
    assert report["by_source_region"] == {"matmul_7": 30}
    assert next(row for row in report["top_ops"] if row["owner"] == "mx")["region"] == "matmul_7"
    (build / "device_catalog/mx_catalog.json").write_text(json.dumps({"bindings": []}))
    with pytest.raises(ValueError, match="source bindings"):
        report_stage_profile(build, log)
    log.write_text(log.read_text().replace("build_hash abc123", "build_hash stale"))
    with pytest.raises(ValueError, match="not for this stage"):
        report_stage_profile(build, log)


def test_stage_profile_accepts_receipt_bound_capacity_above_8192(tmp_path) -> None:
    build = tmp_path / "build"
    build.mkdir()
    table = [{"id": i, "mlir_op": "arith.constant"} for i in range(8193)]
    (build / "op_profile_table.json").write_text(json.dumps(table))
    receipt = {"op_profile": True, "build_hash": "wide",
               "op_profile_operations": len(table), "op_profile_capacity": 16384}
    (build / "mx_stage_probe.json").write_text(json.dumps(receipt))
    log = tmp_path / "spike.log"
    log.write_text("PROF 8192 10 1\nMETRIC prof_total_ticks 10\n"
                   "METRIC cycles 10\nMETRIC build_hash wide\n"
                   "METRIC memref_rank_mismatch 0\nDONE\n")
    assert report_stage_profile(build, log)["observed_operations"] == 1
    receipt["op_profile_capacity"] = 8192
    (build / "mx_stage_probe.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="linked profiler capacity"):
        report_stage_profile(build, log)
