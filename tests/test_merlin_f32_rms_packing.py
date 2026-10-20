"""FP32 RMS fusion preserves observable tensors and exact arithmetic boundaries."""

import pytest
from merlin.frontends.linalg_mlir import parse_mlir_text

from mx_gemmini_support.merlin_f32_rms_packing import rewrite_f32_rms_packing


def fixture(*, matrix=False, escape=False, barrier=False, dead_view=False):
    full, row, packed = "tensor<16x32xf32>", "tensor<16x1xf32>", "tensor<528xi8>"
    identity, scalar = "affine_map<(d0,d1)->(d0,d1)>", "affine_map<(d0,d1)->()>"

    def generic(name, inputs, output, maps, body, rank=2):
        operands = ",".join("%" + value for value, _ in inputs)
        types = ",".join(dtype for _, dtype in inputs)
        args = ",".join(f"%a{i}: f32" for i in range(len(inputs) + 1))
        return f"""%{name}_d=tensor.empty() : {output}
          %{name}=linalg.generic {{indexing_maps=[{",".join(maps)}],iterator_types=[{",".join(['"parallel"'] * rank)}]}}
          ins({operands} : {types}) outs(%{name}_d : {output}) {{
          ^bb0({args}):
            %v={body} : f32
            linalg.yield %v : f32
          }} -> {output}"""

    final_type = full if matrix else packed
    terminal = (
        f"""%w=tensor.empty() : tensor<32x32xf32>
        %dest=tensor.empty() : {full}
        %result=func.call @matrix(%product,%w,%dest) : ({full},tensor<32x32xf32>,{full})->{full}"""
        if matrix
        else f"""%dest=tensor.empty() : {packed}
        %result=func.call @prepare(%product,%dest) : ({full},{packed})->{packed}"""
    )
    source = f"""builtin.module {{
      func.func private @prepare(%x: {full} {{bufferization.access="read"}},%d: {packed} {{bufferization.access="write"}})->{packed}
      func.func private @matrix(%x: {full} {{bufferization.access="read"}},%w: tensor<32x32xf32> {{bufferization.access="read"}},%d: {full} {{bufferization.access="write"}})->{full}
      func.func private @unknown()
      func.func @forward(%x: {full},%gamma: tensor<32xf32>)->({final_type}{"," + full if escape else ""}) {{
        {generic("square", [("x", full)], full, [identity, identity], "arith.mulf %a0, %a0")}
        %zero=arith.constant 0.0 : f32
        %init=tensor.splat %zero : tensor<16xf32>
        %sum=linalg.reduce ins(%square:{full}) outs(%init:tensor<16xf32>) dimensions=[1]
        (%value: f32,%acc: f32) {{
          %v=arith.addf %value,%acc : f32
          linalg.yield %v : f32
        }}
        %divisor=arith.constant 32.0 : f32
        %ds=tensor.splat %divisor : tensor<f32>
        {generic("mean", [("sum", "tensor<16xf32>"), ("ds", "tensor<f32>")], "tensor<16xf32>", ["affine_map<(d0)->(d0)>", "affine_map<(d0)->()>", "affine_map<(d0)->(d0)>"], "arith.divf %a0, %a1", rank=1)}
        %mean_view=tensor.expand_shape %mean [[0 : i64,1 : i64]] output_shape [16,1] : tensor<16xf32> into {row}
        %epsilon=arith.constant 1.0e-5 : f32
        %eps=tensor.splat %epsilon : tensor<f32>
        {generic("added", [("mean_view", row), ("eps", "tensor<f32>")], row, [identity, scalar, identity], "arith.addf %a0, %a1")}
        {generic("inverse", [("added", row)], row, [identity, identity], "math.rsqrt %a0")}
        {generic("scaled", [("x", full), ("inverse", row)], full, [identity, "affine_map<(d0,d1)->(d0,0)>", identity], "arith.mulf %a0, %a1")}
        {"func.call @unknown() : ()->()" if barrier else ""}
        {generic("product", [("gamma", "tensor<32xf32>"), ("scaled", full)], full, ["affine_map<(d0,d1)->(d1)>", identity, identity], "arith.mulf %a0, %a1")}
        {"%dead=tensor.collapse_shape %product [[0 : i64,1 : i64]] : " + full + " into tensor<512xf32>" if dead_view else ""}
        {terminal}
        func.return %result{",%scaled" if escape else ""} : {final_type}{"," + full if escape else ""}
      }}
    }}"""
    endpoint = {
        "symbol": "fused",
        "shape": [16, 32],
        "rows": 16,
        "width": 32,
        "packed_bytes": 528,
        "prepare_symbols": ["prepare"],
        "reduction": "ordered",
    }
    kernel = {
        "symbol": "matrix_kernel",
        "batched": False,
        "precision": "mxfp8",
        "prepacked_checkpoint_weight": True,
        "dimensions": {"m": 16, "n": 32, "k": 32},
        "padded_shape_mnk": [16, 32, 32],
        "activation_preparation": {
            "consumer_arguments": [0, 2],
            "consumer_symbol": "matrix_prepared",
            "prepare_symbol": "prepare",
            "packed_bytes": 528,
        },
    }
    return source, {"f32_rms_packing": [endpoint], "kernels": [kernel]}


@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("dead_view", [False, True])
def test_closed_normalization_and_matrix_terminal(matrix, dead_view):
    source, catalog = fixture(matrix=matrix, dead_view=dead_view)
    original = parse_mlir_text(source)
    result, records = rewrite_f32_rms_packing(
        original, catalog, source_callees={"matrix": "matrix_kernel"}
    )
    assert len(records) == 1
    assert records[0]["original_prepare"] == (None if matrix else "prepare")
    assert [
        o.callee.root_reference.data for o in result.walk() if o.name == "func.call"
    ] == (["fused", "matrix_prepared"] if matrix else ["fused"])
    assert any(o.name == "linalg.reduce" for o in original.walk())
    assert not any(o.name == "linalg.reduce" for o in result.walk())
    result.verify()


@pytest.mark.parametrize("options", [{"escape": True}, {"barrier": True}])
def test_observable_intermediates_and_unknown_effects_are_barriers(options):
    source, catalog = fixture(**options)
    result, records = rewrite_f32_rms_packing(parse_mlir_text(source), catalog)
    assert records == []
    result.verify()


@pytest.mark.parametrize(
    "old,new",
    [
        ("32.0 : f32", "16.0 : f32"),
        ("1.0e-5 : f32", "0.0 : f32"),
        ("arith.addf %value,%acc", "arith.mulf %value,%acc"),
        ("arith.mulf %a0, %a0", "arith.addf %a0, %a0"),
        ("arith.divf %a0, %a1", "arith.divf %a1, %a0"),
        ("affine_map<(d0,d1)->(d0,0)>", "affine_map<(d0,d1)->(d0,d1)>"),
        ("arith.constant 0.0 : f32", "arith.constant 1.0 : f32"),
    ],
)
def test_different_arithmetic_is_not_reassociated(old, new):
    source, catalog = fixture()
    _, records = rewrite_f32_rms_packing(
        parse_mlir_text(source.replace(old, new)), catalog
    )
    assert records == []


def test_registration_pins_contraction_and_respects_shared_workspace(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from mx_gemmini_support.merlin_f32_rms_packing import register_f32_rms_packing
    from mx_gemmini_support.target_profile import load_profile

    source, catalog = fixture()
    prepared = tmp_path / "source.mlir"
    prepared.write_text(source)
    monkeypatch.setattr(
        "mx_gemmini_support.merlin_vpu_reduce.selected_row_reduce_sites",
        lambda *_: [{"operation": "rsum", "shape": [16, 32]}],
    )
    profile = load_profile(
        Path(__file__).parents[1]
        / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
    )
    compiled, abi = [], {}
    compile_source = lambda name, source, **kwargs: compiled.append(
        (name, source, kwargs)
    )
    assert (
        register_f32_rms_packing(
            prepared, catalog["kernels"], profile, 64, compile_source, abi
        )
        == []
    )
    assert compiled == [] and abi == {}
    records = register_f32_rms_packing(
        prepared, catalog["kernels"], profile, 128, compile_source, abi
    )
    assert len(records) == len(compiled) == len(abi) == 1
    assert compiled[0][2] == {"extra_flags": ("-ffp-contract=off", "-fno-math-errno")}


def test_existing_endpoint_declaration_cannot_change_access_contract():
    source, catalog = fixture()
    declaration = """func.func private @fused(%x: tensor<16x32xf32>,%g: tensor<32xf32>,%d: tensor<f32>,%e: tensor<f32>,%out: tensor<528xi8>)->tensor<528xi8>"""
    source = source.replace("builtin.module {", "builtin.module {" + declaration)
    with pytest.raises(ValueError, match="incompatible declaration"):
        rewrite_f32_rms_packing(parse_mlir_text(source), catalog)
