"""Gated projection fusion needs private tensors and checked source bindings."""

import copy
from pathlib import Path

import pytest
from merlin.frontends.linalg_mlir import parse_mlir_text

from mx_gemmini_support.merlin_gated_projection import (
    register_gated_projections,
    rewrite_gated_projections,
    source_matrix_callees,
)
from mx_gemmini_support.target_profile import load_profile


def fixture(
    *,
    matrix=False,
    escape=False,
    barrier=False,
    different_input=False,
    bad_access=False,
    commute=False,
    bf16_sigmoid=False,
    reused_destination=False,
):
    packed, out = "tensor<528xi8>", "tensor<16x32xf32>"
    maps = "[affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>]"

    def mul(name, a, b):
        return f"""%{name}_d=tensor.empty() : {out}
        %{name}=linalg.generic {{indexing_maps={maps},iterator_types=["parallel","parallel"]}}
          ins(%{a},%{b} : {out},{out}) outs(%{name}_d : {out}) {{
          ^bb0(%a: f32,%b: f32,%d: f32):
            %v=arith.mulf %{"b, %a" if commute else "a, %b"} : f32
            linalg.yield %v : f32
          }} -> {out}"""

    terminal = (
        f"""%w=tensor.empty() : tensor<32x32xf32>
        %dest=tensor.empty() : {out}
        %r=func.call @down(%product,%w,%dest) : ({out},tensor<32x32xf32>,{out})->{out}"""
        if matrix
        else f"""%dest=tensor.empty() : {packed}
        %r=func.call @prepare(%product,%dest) : ({out},{packed})->{packed}"""
    )
    result_type = out if matrix else packed
    source = f'''builtin.module {{
      func.func private @gate(%x: {packed} {{bufferization.access="read"}},%d: {out} {{bufferization.access="write"}})->{out}
      func.func private @up(%x: {packed} {{bufferization.access="{"write" if bad_access else "read"}"}},%d: {out} {{bufferization.access="write"}})->{out}
      func.func private @sigmoid(%x: {out} {{bufferization.access="read"}},%d: {out} {{bufferization.access="write"}})->{out}
      func.func private @prepare(%x: {out} {{bufferization.access="read"}},%d: {packed} {{bufferization.access="write"}})->{packed}
      func.func private @down(%x: {out} {{bufferization.access="read"}},%w: tensor<32x32xf32> {{bufferization.access="read"}},%d: {out} {{bufferization.access="write"}})->{out}
      func.func private @unknown()
      func.func @forward(%x: {packed},%y: {packed},%existing: {out})->({result_type}{"," + out if escape else ""}) {{
        %gd=tensor.empty() : {out}
        %g=func.call @gate(%x,%{"existing" if reused_destination else "gd"}) : ({packed},{out})->{out}
        %sd=tensor.empty() : {out}
        %s=func.call @sigmoid(%g,%sd) : ({out},{out})->{out}
        {mul("first", "g", "s")}
        {"func.call @unknown() : ()->()" if barrier else ""}
        %ud=tensor.empty() : {out}
        %u=func.call @up(%{"y" if different_input else "x"},%ud) : ({packed},{out})->{out}
        {mul("product", "first", "u")}
        {terminal}
        func.return %r{",%g" if escape else ""} : {result_type}{"," + out if escape else ""}
      }}
    }}'''
    preparation = {
        "consumer_arguments": [0, 2],
        "consumer_symbol": "down_prepared",
        "prepare_symbol": "prepare",
        "packed_bytes": 528,
    }
    catalog = {
        "kernels": [
            {
                "symbol": "down_kernel",
                "batched": False,
                "precision": "mxfp8",
                "prepacked_checkpoint_weight": True,
                "dimensions": {"m": 16, "n": 32, "k": 32},
                "padded_shape_mnk": [16, 32, 32],
                "activation_preparation": preparation,
            }
        ],
        "vpu_unary": {
            "kernels": [
                {
                    "symbol": "sigmoid",
                    "operation": "sigmoid",
                    "dtype": "f32",
                    "implementation": "bf16" if bf16_sigmoid else "f32_reciprocal",
                }
            ]
        },
        "gated_projection_packing": [
            {
                "symbol": "gated",
                "shape_mnk": [16, 32, 32],
                "input_packed_bytes": 528,
                "output_packed_bytes": 528,
                "prepare_symbols": ["prepare"],
                "gate_consumer": "gate",
                "up_consumer": "up",
                "source_kernels": ["gk", "uk"],
                "source_regions": ["gr", "ur"],
            }
        ],
    }
    return parse_mlir_text(source), catalog


def calls(module):
    return [
        o.callee.root_reference.data for o in module.walk() if o.name == "func.call"
    ]


@pytest.mark.parametrize("matrix", [False, True])
def test_both_explicit_packing_and_internal_down_packing_keep_public_result(matrix):
    original, catalog = fixture(matrix=matrix)
    result, records = rewrite_gated_projections(
        original, catalog, source_callees={"down": "down_kernel"}
    )
    assert calls(original) == ["gate", "sigmoid", "up", "down" if matrix else "prepare"]
    assert (
        calls(result) == ["gated", "down_prepared"]
        if matrix
        else calls(result) == ["gated"]
    )
    assert len(records) == 1
    assert records[0]["original_prepare"] == (None if matrix else "prepare")
    result.verify()


@pytest.mark.parametrize(
    "options",
    [
        {"escape": True},
        {"barrier": True},
        {"different_input": True},
        {"bad_access": True},
        {"commute": True},
        {"bf16_sigmoid": True},
        {"reused_destination": True},
    ],
)
def test_escaping_or_unproved_arithmetic_and_effects_keep_original(options):
    original, catalog = fixture(**options)
    result, records = rewrite_gated_projections(original, catalog)
    assert not records and calls(result) == calls(original)


def test_down_preparation_requires_a_bound_source_and_unchanged_logical_packing():
    for mutate in (
        lambda c: None,
        lambda c: c["kernels"][0].update(precision="mxfp4"),
        lambda c: c["kernels"][0].update(padded_shape_mnk=[32, 32, 32]),
    ):
        original, catalog = fixture(matrix=True)
        mutate(catalog)
        mapping = (
            {}
            if catalog["kernels"][0]["precision"] == "mxfp8"
            and catalog["kernels"][0]["padded_shape_mnk"] == [16, 32, 32]
            else {"down": "down_kernel"}
        )
        result, records = rewrite_gated_projections(
            original, catalog, source_callees=mapping
        )
        assert not records and calls(result) == calls(original)


def test_routing_checks_ordinals_tensor_types_and_ambiguous_callees():
    binding = {
        "region": "matmul_1",
        "symbol": "kernel",
        "source_operation_ordinal": 4,
        "tensor_types": ["a", "b", "c"],
    }
    route = {
        "source_region": "matmul_1",
        "symbol": "callee",
        "source_operation_ordinal": 4,
        "tensor_types": ["a", "b", "c"],
    }
    catalog = {"bindings": [binding]}
    assert source_matrix_callees(catalog, {"routed": [route]}) == {"callee": "kernel"}
    for field, value in (
        ("source_operation_ordinal", 5),
        ("tensor_types", ["a", "b", "d"]),
    ):
        with pytest.raises(ValueError, match="binding differs"):
            source_matrix_callees(catalog, {"routed": [dict(route, **{field: value})]})
    catalog["bindings"].append(dict(binding, region="matmul_2", symbol="other"))
    with pytest.raises(ValueError, match="ambiguous"):
        source_matrix_callees(
            catalog, {"routed": [route, dict(route, source_region="matmul_2")]}
        )


def test_registration_shares_shape_code_without_extending_serial_workspace():
    _, catalog = fixture()
    down = catalog["kernels"][0]
    gate = dict(
        copy.deepcopy(down),
        symbol="gk",
        tile_mnk=[16, 16, 32],
        schedule_file="issue.mk",
    )
    gate["activation_preparation"]["consumer_symbol"] = "gate"
    up = copy.deepcopy(gate)
    up["symbol"] = "uk"
    up["activation_preparation"]["consumer_symbol"] = "up"
    groups = [[{"symbol": "gk", "region": "gr"}, {"symbol": "uk", "region": "ur"}]] * 2
    profile = load_profile(
        Path(__file__).resolve().parents[1]
        / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
    )
    compiled = {}
    abi = {}
    records = register_gated_projections(
        groups,
        [gate, up, down],
        profile,
        100000,
        lambda n, s: compiled.update({n: s}),
        abi,
    )
    assert len(records) == 2 and len(compiled) == 3 and len(abi) == 2
    assert records[0]["generic_symbol"] == records[1]["generic_symbol"]
    assert all(r["workspace_bytes"] <= 100000 for r in records)
    assert not register_gated_projections(
        groups,
        [dict(gate, precision="mxfp4"), up, down],
        profile,
        100000,
        lambda *args: pytest.fail("compiled FP4"),
        {},
    )
    assert not register_gated_projections(
        groups,
        [gate, up, down],
        profile,
        64,
        lambda *args: pytest.fail("exceeded workspace"),
        {},
    )


def test_endpoint_declaration_collision_is_refused_before_replacing_calls():
    from xdsl.dialects import func
    from xdsl.dialects.builtin import f32
    from xdsl.ir import Region

    original, catalog = fixture()
    original.body.block.add_op(
        func.FuncOp("gated", ([f32], [f32]), Region(), visibility="private")
    )
    with pytest.raises(ValueError, match="incompatible declaration"):
        rewrite_gated_projections(original, catalog)
