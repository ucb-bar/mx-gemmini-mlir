"""The session audit separates shape plans from executable placements."""

from pathlib import Path

from mx_gemmini_support.target_profile import load_profile
from tools.audit_model2mlir_session import summarize_dense_shape_plans


ROOT = Path(__file__).resolve().parents[1]
PROFILE = load_profile(
    ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json")


def test_dense_shape_audit_reports_rank2_and_batched_slice_separately() -> None:
    proposal = {"sites": [
        {"region_id": "dense", "selected_format": "mxfp4",
         "shape_mnk": [8, 344, 128], "element_type": "f32"},
        {"region_id": "attention", "selected_format": "mxfp8",
         "shape_bmnk": [4, 8, 127, 128]},
        {"region_id": "host", "selected_format": "host"},
    ]}
    audit = summarize_dense_shape_plans(proposal, PROFILE)
    assert audit["status"] == "shape_schedule_only_no_executable_placement"
    assert audit["counts"] == {
        "rank2_f32_dense_plan": 1,
        "rank3_slice_shape_plan": 1,
    }
    assert audit["max_output_tiles_per_call"] >= 8
    assert audit["first_failures"] == []
