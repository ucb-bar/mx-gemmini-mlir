"""Softmax packing requires a private probability tensor and exact batch layouts."""

import pytest
from merlin.frontends.linalg_mlir import parse_mlir_text

from mx_gemmini_support.merlin_softmax_packing import rewrite_softmax_packing


def fixture(
    *, escape=False, barrier=False, bad_access=False, identity=False, negate=False
):
    scores = "tensor<1x2x16x32xf32>"
    a = "tensor<2x16x32xf32>"
    b = "tensor<1x32x16xf32>"
    out = "tensor<2x16x16xf32>"
    copy = (
        f"""
        %copy_out=tensor.empty() : {scores}
        %copy=linalg.generic {{indexing_maps=[affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>],iterator_types=["parallel","parallel","parallel","parallel"]}}
          ins(%p : {scores}) outs(%copy_out : {scores}) {{
          ^bb0(%in: f32,%out: f32):
            {"%neg=arith.negf %in : f32" if negate else ""}
            linalg.yield %{"neg" if negate else "in"} : f32
          }} -> {scores}
    """
        if identity
        else ""
    )
    source = f'''builtin.module {{
      func.func private @softmax(%x: {scores} {{bufferization.access="read"}},
        %out: {scores} {{bufferization.access="write"}}) -> {scores}
      func.func private @matrix(%a: {a} {{bufferization.access="read"}},
        %b: {b} {{bufferization.access="{"write" if bad_access else "read"}"}},
        %out: {out} {{bufferization.access="write"}}) -> {out}
      func.func private @unknown()
      func.func @forward(%x: {scores},%b: {b}) -> ({out}{"," + scores if escape else ""}) {{
        %d=tensor.empty() : {scores}
        %p=func.call @softmax(%x,%d) : ({scores},{scores}) -> {scores}
        {copy}
        %v=tensor.collapse_shape %{"copy" if identity else "p"} [[0,1],[2],[3]] : {scores} into {a}
        {"func.call @unknown() : () -> ()" if barrier else ""}
        %o=tensor.empty() : {out}
        %r=func.call @matrix(%v,%b,%o) : ({a},{b},{out}) -> {out}
        func.return %r{",%p" if escape else ""} : {out}{"," + scores if escape else ""}
      }}
    }}'''
    endpoint = {
        "symbol": "packed",
        "original_softmax": "softmax",
        "original_consumers": ["matrix"],
        "source_kernel": "source_matrix",
        "shape": [1, 2, 16, 32],
        "matrix_shape_bmnk": [2, 16, 16, 32],
        "rhs_repeat": 2,
        "packed_bytes": 1056,
    }
    return parse_mlir_text(source), {"softmax_packing": [endpoint]}


def calls(module):
    return [
        o.callee.root_reference.data for o in module.walk() if o.name == "func.call"
    ]


def test_private_probability_tensor_becomes_an_explicit_packed_edge():
    original, catalog = fixture()
    result, records = rewrite_softmax_packing(original, catalog)
    assert calls(original) == ["softmax", "matrix"]
    assert calls(result) == ["packed"] and len(records) == 1
    call = next(o for o in result.walk() if o.name == "func.call")
    assert str(call.arguments[0].type) == "tensor<1x2x16x32xf32>"
    assert str(call.results[0].type) == "tensor<2x16x16xf32>"
    assert not any(o.name == "tensor.collapse_shape" for o in result.walk())


@pytest.mark.parametrize(
    "options", [{"escape": True}, {"barrier": True}, {"bad_access": True}]
)
def test_observable_probability_or_unknown_effect_keeps_original(options):
    original, catalog = fixture(**options)
    result, records = rewrite_softmax_packing(original, catalog)
    assert calls(result) == calls(original) and not records


def test_changed_matrix_layout_cannot_reuse_old_packed_bytes():
    original, catalog = fixture()
    catalog["softmax_packing"][0]["matrix_shape_bmnk"] = [2, 32, 16, 16]
    result, records = rewrite_softmax_packing(original, catalog)
    assert calls(result) == calls(original) and not records


def test_only_exact_identity_copy_may_disappear_with_private_probability():
    for negate in (False, True):
        original, catalog = fixture(identity=True, negate=negate)
        result, records = rewrite_softmax_packing(original, catalog)
        if negate:
            assert not records and calls(result) == calls(original)
        else:
            assert len(records) == 1 and calls(result) == ["packed"]


def test_registered_endpoint_keeps_rhs_refresh_local_to_each_forward_call():
    from pathlib import Path

    from mx_gemmini_support.merlin_softmax_packing import register_softmax_packing
    from mx_gemmini_support.target_profile import load_profile

    profile = load_profile(
        Path(__file__).resolve().parents[1]
        / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
    )
    softmax = [
        {
            "symbol": "s",
            "shape": [1, 2, 16, 32],
            "logical_rows": 32,
            "row_width": 32,
            "implementation": "f32_reduce",
        }
    ]
    kernel = {
        "symbol": "matrix",
        "precision": "mxfp8",
        "batched": True,
        "dimensions": {"batch": 2, "m": 16, "n": 16, "k": 32},
        "padded_shape_mnk": [16, 16, 32],
        "tile_mnk": [16, 16, 32],
        "schedule_file": "issue.mk",
    }
    bindings = [
        {
            "symbol": "matrix",
            "batch_rhs_compact": {
                "entry_symbol": "old",
                "repeat": 2,
                "workspace_bytes": 100000,
            },
        }
    ]
    compiled = {}
    abi = {}
    records = register_softmax_packing(
        softmax, [kernel], bindings, profile, lambda n, s: compiled.update({n: s}), abi
    )
    assert len(records) == len(compiled) == len(abi) == 1
    source = next(iter(compiled.values()))
    assert "batch%2u==0u" in source and "rhs+(batch/2u)*512u" in source
    assert records[0]["packed_lifetime"] == "one_serial_matrix"
    assert records[0]["packed_bytes"] == 16 * 32 * 33 // 32
    assert [b.dtype for b in next(iter(abi.values()))[1]] == ["f32", "f32", "f32"]
    assert not register_softmax_packing(
        softmax,
        [dict(kernel, precision="mxfp4")],
        bindings,
        profile,
        lambda *args: pytest.fail("unsupported FP4 endpoint compiled"),
        {},
    )
