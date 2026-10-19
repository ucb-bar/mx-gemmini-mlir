"""The session audit separates shape plans from executable placements."""

import hashlib
from pathlib import Path

import pytest

from mx_gemmini_support.target_profile import load_profile
from tools.audit_model2mlir_session import (
    read_inventory_source,
    summarize_dense_shape_plans,
)

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


CUSTOM = '''module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%x: tensor<1x8xi64>) -> tensor<1x1xi64> {
    %r = tensor.extract_slice %x[0, 0] [1, 1] [1, 1] : tensor<1x8xi64> to tensor<1x1xi64>
    func.return %r : tensor<1x1xi64>
  }
}'''


def test_custom_syntax_requires_explicit_printer_work_and_preserves_capture(tmp_path):
    from merlin.llvmlower.toolchain import compiler_python
    from merlin.targetgen.contract.linalg_iface import LinalgParseError

    if not compiler_python().is_file():
        pytest.skip("generic printer needs the explicitly selected MLIR interpreter")
    capture = tmp_path / "capture"
    capture.mkdir()
    source = capture / "model.mlir"
    source.write_text(CUSTOM)
    with pytest.raises(LinalgParseError):
        read_inventory_source(source, PROFILE)
    work = tmp_path / "view"
    inventory, facts, view = read_inventory_source(source, PROFILE, work)
    assert inventory["entry"] == "forward"
    assert view["source_sha256"] == hashlib.sha256(CUSTOM.encode()).hexdigest()
    assert view["view_sha256"] == facts["source_mlir_sha256"]
    assert view["view_sha256"] != view["source_sha256"]
    assert source.read_text() == CUSTOM and list(capture.iterdir()) == [source]
    assert Path(view["path"]).parent == work.resolve()


def test_parseable_source_retains_identity_and_has_no_printer_side_effect(tmp_path, monkeypatch):
    import merlin.llvmlower.generic_form as printer

    def forbidden(*args, **kwargs):
        pytest.fail("parseable input must not launch a printer")

    monkeypatch.setattr(printer, "to_generic_form", forbidden)
    source = tmp_path / "plain.mlir"
    text = '''module attributes {prov.level = "linalg-on-tensors"} {
      func.func @forward(%x:tensor<2xf32>)->tensor<2xf32> {func.return %x:tensor<2xf32>}
    }'''
    source.write_text(text)
    _, facts, view = read_inventory_source(source, PROFILE, tmp_path / "unused")
    assert view is None and facts["source_mlir_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert not (tmp_path / "unused").exists()


def test_changed_capture_cannot_supply_a_generic_view(tmp_path, monkeypatch):
    import merlin.llvmlower.generic_form as printer

    source = tmp_path / "model.mlir"
    source.write_text(CUSTOM)

    def changed(path, work, **kwargs):
        path.write_text(CUSTOM + "\n// changed during read\n")
        output = tmp_path / "normalized.mlir"
        output.write_text('''module attributes {prov.level="linalg-on-tensors"} {
          func.func @forward(%x:tensor<2xf32>)->tensor<2xf32> {func.return %x:tensor<2xf32>}
        }''')
        return output

    monkeypatch.setattr(printer, "to_generic_form", changed)
    with pytest.raises(ValueError, match="source changed"):
        read_inventory_source(source, PROFILE, tmp_path / "view")


def test_native_printer_refusal_is_not_an_empty_inventory(tmp_path):
    from merlin.llvmlower.generic_form import GenericFormError
    from merlin.llvmlower.toolchain import compiler_python

    if not compiler_python().is_file():
        pytest.skip("generic printer needs the explicitly selected MLIR interpreter")
    source = tmp_path / "invalid.mlir"
    text = CUSTOM.replace("%x[0, 0]", "%undefined[0, 0]")
    source.write_text(text)
    with pytest.raises(GenericFormError):
        read_inventory_source(source, PROFILE, tmp_path / "view")
    assert source.read_text() == text
