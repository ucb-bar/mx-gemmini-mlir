"""Only independent, effect-checked packed-input projections may execute together."""

import pytest
from merlin.frontends.linalg_mlir import parse_mlir_text

from mx_gemmini_support.merlin_projection_group import rewrite_projection_groups


def fixture(*, barrier=False, different_input=False, bad_access=False):
    packed, out = "tensor<528xi8>", "tensor<16x32xf32>"
    source = f'''builtin.module {{
    func.func private @a(%x: {packed} {{bufferization.access = "read"}},
       %out: {out} {{bufferization.access = "write"}}) -> {out}
    func.func private @b(%x: {packed} {{bufferization.access = "{"write" if bad_access else "read"}"}},
       %out: {out} {{bufferization.access = "write"}}) -> {out}
    func.func private @unknown()
    func.func @forward(%x: {packed}, %y: {packed}) -> ({out}, {out}) {{
      %d = tensor.empty() : {out}
      %a = func.call @a(%x, %d) : ({packed}, {out}) -> {out}
      {"func.call @unknown() : () -> ()" if barrier else ""}
      %e = tensor.empty() : {out}
      %b = func.call @b(%{"y" if different_input else "x"}, %e) : ({packed}, {out}) -> {out}
      func.return %a, %b : {out}, {out}
    }}
    }}'''
    catalog = {
        "projection_groups": [
            {
                "symbol": "grouped",
                "consumer_symbols": ["a", "b"],
                "kernel_symbols": ["ka", "kb"],
                "source_regions": ["ra", "rb"],
                "packed_bytes": 528,
                "output_shapes": [[16, 32], [16, 32]],
            }
        ]
    }
    return parse_mlir_text(source), catalog


def calls(module):
    return [
        op.callee.root_reference.data for op in module.walk() if op.name == "func.call"
    ]


def test_preserves_both_observable_outputs_with_fresh_destinations():
    original, catalog = fixture()
    result, records = rewrite_projection_groups(original, catalog)
    assert calls(original) == ["a", "b"]
    assert calls(result) == ["grouped"]
    assert len(records) == 1
    call = next(op for op in result.walk() if op.name == "func.call")
    assert len(call.results) == 2 and len(call.arguments) == 3
    assert call.arguments[1] is not call.arguments[2]
    declaration = next(
        op
        for op in result.body.block.ops
        if getattr(getattr(op, "sym_name", None), "data", None) == "grouped"
    )
    assert [a.data["bufferization.access"].data for a in declaration.arg_attrs] == [
        "read",
        "write",
        "write",
    ]


@pytest.mark.parametrize(
    "options", [{"barrier": True}, {"different_input": True}, {"bad_access": True}]
)
def test_unknown_effects_unmatched_packed_input_or_unproved_access_keep_calls(options):
    original, catalog = fixture(**options)
    result, records = rewrite_projection_groups(original, catalog)
    assert calls(result) == calls(original) and not records


def test_shape_mismatch_or_repeated_consumer_keeps_calls():
    for kind in ("shape", "repeat"):
        original, catalog = fixture()
        if kind == "shape":
            catalog["projection_groups"][0]["output_shapes"][1] = [32, 16]
        else:
            catalog["projection_groups"][0]["consumer_symbols"][1] = "a"
        result, records = rewrite_projection_groups(original, catalog)
        assert calls(result) == calls(original) and not records
