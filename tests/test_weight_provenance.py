"""Only layout-preserving checkpoint casts may be packed at build time."""

from __future__ import annotations

import pytest

from mx_gemmini_support.weight_provenance import source_parameter_arg


@pytest.mark.parametrize("input_map,expected_arg", [
    ("(d0, d1)", 0),
    ("(d1, d0)", None),
])
def test_bf16_cast_requires_identity_indexing(input_map, expected_arg) -> None:
    pytest.importorskip("merlin.frontends.linalg_mlir")
    from merlin.frontends.linalg_mlir import parse_mlir_text

    module = parse_mlir_text(f"""module {{
      func.func @forward(%arg0: tensor<2x2xbf16>) -> tensor<2x2xf32> {{
        %0 = tensor.empty() : tensor<2x2xf32>
        %1 = linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> {input_map}>,
                         affine_map<(d0, d1) -> (d0, d1)>],
                         iterator_types = ["parallel", "parallel"]}}
            ins(%arg0 : tensor<2x2xbf16>) outs(%0 : tensor<2x2xf32>) {{
          ^bb0(%in: bf16, %out: f32):
            %2 = arith.extf %in : bf16 to f32
            linalg.yield %2 : f32
        }} -> tensor<2x2xf32>
        return %1 : tensor<2x2xf32>
      }}
    }}""")
    cast = next(op for op in module.walk() if str(op.name) == "linalg.generic")
    assert source_parameter_arg(cast.results[0])[0] == expected_arg
