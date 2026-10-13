"""A fixed MX RHS may drop host preparation only with its catalog proof."""

from __future__ import annotations

import json

import pytest

from mx_gemmini_support.merlin_fixed_weight import MXFixedWeightPostOffload


def test_fixed_rhs_rewrite_removes_dead_cast(tmp_path) -> None:
    pytest.importorskip("merlin.frontends.linalg_mlir")
    prepared = tmp_path / "prepared.mlir"
    prepared.write_text("""module {
      func.func @forward(%weight: tensor<2x2xbf16>,
                         %lhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
        %out = tensor.empty() : tensor<2x2xf32>
        %rhs_out = tensor.empty() : tensor<2x2xf32>
        %rhs = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%weight : tensor<2x2xbf16>) outs(%rhs_out : tensor<2x2xf32>) {
          ^bb0(%in: bf16, %old: f32):
            %wide = arith.extf %in : bf16 to f32
            linalg.yield %wide : f32
        } -> tensor<2x2xf32>
        %result = func.call @merlin_dev_mx_gemmini_0(%lhs, %rhs, %out) :
          (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        return %result : tensor<2x2xf32>
      }
      func.func private @merlin_dev_mx_gemmini_0(
        tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    }""")
    (tmp_path / "device_catalog").mkdir()
    catalog = tmp_path / "device_catalog/catalog.json"
    catalog.write_text(json.dumps({
        "source_sha256": "source",
        "kernels": [{"symbol": "fixed_0", "batched": False,
                     "prepacked_checkpoint_weight": True,
                     "dimensions": {"m": 2, "n": 2, "k": 2},
                     "checkpoint_parameter_arg": 0,
                     "checkpoint_parameter_name": "weight",
                     "checkpoint_parameter_shape": [2, 2],
                     "checkpoint_parameter_transforms": ["linalg.generic"]}],
        "bindings": [{"symbol": "fixed_0", "region": "matmul_0",
                      "source_operation_ordinal": 5,
                      "tensor_types": ["tensor<2x2xf32>"] * 3}],
    }))
    sidecar = tmp_path / "device_signatures.json"
    sidecar.write_text(json.dumps({"catalog_manifest": str(catalog),
                                   "model_sha256": "source",
                                   "routed": [{"symbol": "merlin_dev_mx_gemmini_0",
                                               "source_region": "matmul_0",
                                               "source_operation_ordinal": 5,
                                               "tensor_types": ["tensor<2x2xf32>"] * 3}]}))
    result = MXFixedWeightPostOffload(catalog)(prepared, sidecar, tmp_path / "out")
    lowered = result.read_text()
    assert "linalg.generic" not in lowered
    assert "func.call @merlin_dev_mx_gemmini_0" in lowered
    assert json.loads((tmp_path / "out/mx_fixed_rhs_rewrite.json").read_text())[
        "rewritten_calls"] == 1

    sidecar.write_text(sidecar.read_text().replace("matmul_0", "stale"))
    with pytest.raises(ValueError, match="source binding"):
        MXFixedWeightPostOffload(catalog)(prepared, sidecar, tmp_path / "out2")
