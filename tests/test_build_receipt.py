"""A declared Scala census must match the current selected source tree."""

import hashlib

import pytest

from mx_gemmini_support.build_receipt import _mesh_compute_hierarchy, _mesh_grid, _source_census


def _receipt_census(root):
    recorded = {}
    for path in sorted(root.rglob("*.scala")):
        recorded[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = hashlib.sha256("".join(
        f"{name} {digest}\n" for name, digest in recorded.items()
    ).encode()).hexdigest()
    return {"sha256_by_relative_path": recorded, "scala_files": len(recorded),
            "manifest_sha256": manifest}


def test_source_census_refuses_changed_and_extra_scala_files(tmp_path):
    source = tmp_path / "src/main/scala/gemmini/Config.scala"
    source.parent.mkdir(parents=True)
    source.write_text("object Config {}\n")
    mx_source = tmp_path / "mxgen/src/main/scala/mxgen/Mx.scala"
    mx_source.parent.mkdir(parents=True)
    mx_source.write_text("object Mx {}\n")
    census = _receipt_census(tmp_path)
    assert _source_census(tmp_path, census) == census["manifest_sha256"]
    mx_source.write_text("object Mx { val changed = true }\n")
    with pytest.raises(ValueError, match="Scala source differs"):
        _source_census(tmp_path, census)
    mx_source.write_text("object Mx {}\n")
    (source.parent / "Added.scala").write_text("object Added {}\n")
    with pytest.raises(ValueError, match="omits or adds"):
        _source_census(tmp_path, census)


def test_elaborated_mesh_requires_dense_tile_coordinates(tmp_path):
    fir = tmp_path / "design.fir"
    tiles = """  module Mesh : @[generators/gemmini/src/main/scala/gemmini/Mesh.scala 18:7]
    inst mesh_0_0 of Tile
    inst mesh_0_1 of Tile_1
    inst mesh_1_0 of Tile_2
    inst mesh_1_1 of Tile_3
  module Host :
    inst mesh_9_9 of Unrelated
"""
    fir.write_text(tiles)
    assert _mesh_grid(fir)["rows"] == 2
    assert _mesh_grid(fir)["columns"] == 2
    fir.write_text(tiles.replace("    inst mesh_1_0 of Tile_2\n", ""))
    with pytest.raises(ValueError, match="not a dense grid"):
        _mesh_grid(fir)
    fir.write_text(tiles.replace("mesh_1_0", "mesh_1_1"))
    with pytest.raises(ValueError, match="duplicate"):
        _mesh_grid(fir)


def test_selected_mesh_multiplier_hierarchy_refuses_missing_link(tmp_path):
    fir = tmp_path / "design.fir"
    text = """  module Mesh :
    inst mesh_0_0 of Tile
  module Tile :
    inst tile_0_0 of PE_1
  module PE_1 :
    inst mac_unit of MacUnit
    connect mac_unit.io.in_a.bits, io.in_a.bits
    connect mac_unit.io.in_b.bits, io.in_b.bits
    connect mac_unit.io.in_c.bits, io.in_c.bits
    connect io.out_b, mac_unit.io.out_d
  module MacUnit :
    inst io_out_d_macc of MxFpMul
    connect io_out_d_macc.io.in_activation, io.in_a.bits
    connect io_out_d_macc.io.in_weights, io.in_b.bits
    connect io_out_d_macc.io.rec_c, io.in_c.bits
    connect result, io_out_d_macc.io.out
    connect io.out_d, result
  module MxFpMul :
    inst core of MxFpMulCore
    inst fma of MxMulAddRecFN
    connect core.io.in_activation, io.in_activation
    connect core.io.in_weights, io.in_weights
    connect outputs[0], fma.io.out
    connect io.out, outputs[0]
  module MxMulAddRecFN :
"""
    fir.write_text(text)
    mesh = _mesh_grid(fir)
    hierarchy = _mesh_compute_hierarchy(fir, mesh)
    assert hierarchy["tiles_with_hierarchy"] == 1
    assert hierarchy["fused_units_per_tile"] == 1
    assert hierarchy["port_wiring"]["tiles_with_witnesses"] == 1
    assert hierarchy["connected_arithmetic_verified"] is False
    fir.write_text(text.replace("inst fma of MxMulAddRecFN", "inst fma of Other"))
    with pytest.raises(ValueError, match="no elaborated fused"):
        _mesh_compute_hierarchy(fir, mesh)
    fir.write_text(text.replace("connect core.io.in_weights, io.in_weights", "connect core.io.in_weights, UInt<8>(0)"))
    with pytest.raises(ValueError, match="weight input port wiring"):
        _mesh_compute_hierarchy(fir, mesh)
