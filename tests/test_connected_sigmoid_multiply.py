"""Resident sigmoid chains require source arithmetic and private graph contracts."""

from copy import deepcopy
from pathlib import Path

import pytest

from mx_gemmini_support.connected_sigmoid_multiply import (
    check_sigmoid_multiply,
    emit_connected_sigmoid_multiply,
)
from mx_gemmini_support.target_profile import load_profile

ROOT = Path(__file__).resolve().parents[1]


def source(elements=137):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects.lowering.dispatch_program import (
        lower_model_to_dispatch_program,
    )

    typ = f"tensor<{elements}xf32>"
    unary = '''%one=arith.constant 1.0:f32
      %neg=arith.negf %a:f32
      %exp=math.exp %neg:f32
      %denom=arith.addf %one,%exp:f32
      %sig=arith.divf %one,%denom:f32
      linalg.yield %sig:f32'''
    operations = []
    for i, (inputs, body) in enumerate((("%x", unary),
            ("%x,%r0", "%m=arith.mulf %a,%b:f32\n linalg.yield %m:f32"),
            ("%r1,%u", "%m=arith.mulf %a,%b:f32\n linalg.yield %m:f32"))):
        count = 1 if i == 0 else 2
        maps = ",".join(["affine_map<(d0)->(d0)>"] * (count + 1))
        args = "%a:f32,%z:f32" if i == 0 else "%a:f32,%b:f32,%z:f32"
        types = ",".join([typ] * count)
        operations.append(f'''%e{i}=tensor.empty():{typ}
          %r{i}=linalg.generic {{indexing_maps=[{maps}],iterator_types=["parallel"]}}
          ins({inputs}:{types}) outs(%e{i}:{typ}) {{^bb0({args}): {body}}}-> {typ}''')
    body = "\n".join(operations)
    module = parse_mlir_text(f'''module {{func.func @forward(%x:{typ},%u:{typ})->{typ} {{
      {body}
      func.return %r2:{typ}
    }}}}''')
    outlined, program = lower_model_to_dispatch_program(module)
    return outlined.module, program.to_dict()


@pytest.mark.parametrize("elements", [137, 8192, 8311])
def test_six_step_chain_preserves_boundaries_and_has_bounded_resources(elements):
    module, program = source(elements)
    profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json")
    code = emit_connected_sigmoid_multiply(module, program, profile, symbol="checked",
                                           boundary_policy="bf16_rne_each_vpu_call")
    assert code.shape == (elements,)
    assert code.resident_bytes <= profile["resources"]["scratchpad_bytes"]
    assert code.removed_transfer_bytes == (elements + 7) // 8 * 8 * 8


@pytest.mark.parametrize("mutation", ["observable", "repeated", "regions", "not_common_gate"])
def test_invalid_dataflow_refuses_before_emitting_a_kernel(mutation):
    module, program = source()
    program = deepcopy(program)
    if mutation == "observable":
        program["results"].append(program["nodes"][0]["outputs"][0])
    elif mutation == "repeated":
        program["nodes"][2]["inputs"][1] = program["nodes"][0]["outputs"][0]
    elif mutation == "regions":
        program["nodes"][0]["regions"] = 1
    else:
        program["nodes"][1]["inputs"][0] = program["nodes"][2]["inputs"][1]
    with pytest.raises(ValueError):
        check_sigmoid_multiply(module, program)


def test_same_symbol_names_cannot_qualify_changed_sigmoid_arithmetic():
    module, program = source()
    from xdsl.dialects.builtin import FloatAttr, f32
    constant = next(op for op in module.walk() if op.name == "arith.constant")
    constant.properties["value"] = FloatAttr(2.0, f32)
    with pytest.raises(ValueError, match="strict"):
        check_sigmoid_multiply(module, program)
