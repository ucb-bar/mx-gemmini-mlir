"""Typed arithmetic, numerical policy and graph ownership gate f32 residency."""

from copy import deepcopy
from pathlib import Path

import pytest

from mx_gemmini_support.connected_elementwise import F32MultiplyAdapter
from mx_gemmini_support.target_profile import load_profile
from test_connected_vpu import _captured_module


PROFILE = load_profile(Path(__file__).parents[1] /
                       "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json")


def case(elements=137):
    functions = []
    for index in range(2):
        functions.append(f'''func.func @multiply{index}(%a: tensor<{elements}xf32>,
          %b: tensor<{elements}xf32>) -> tensor<{elements}xf32> {{
          %empty = tensor.empty() : tensor<{elements}xf32>
          %r = linalg.generic {{indexing_maps = [affine_map<(i)->(i)>,
            affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]}}
            ins(%a, %b: tensor<{elements}xf32>, tensor<{elements}xf32>)
            outs(%empty: tensor<{elements}xf32>) {{
              ^bb0(%x: f32, %y: f32, %unused: f32):
                %p = arith.mulf %x, %y : f32
                linalg.yield %p : f32
            }} -> tensor<{elements}xf32>
          func.return %r : tensor<{elements}xf32>
        }}''')
    module = _captured_module("builtin.module {" + "\n".join(functions) + "}")
    graph = dict(nodes=[
        dict(kind="dispatch", op="multiply0", inputs=["a", "b"], outputs=["m"]),
        dict(kind="dispatch", op="multiply1", inputs=["m", "c"], outputs=["o"])],
        buffers={name: dict(shape=[elements], dtype="f32", kind="arg" if name in "abc" else "intermediate")
                 for name in ("a", "b", "c", "m", "o")}, results=["o"])
    return module, graph


def adapter(module, graph, **kwargs):
    return F32MultiplyAdapter(module, graph, PROFILE,
                              boundary_policy="bf16_rne_each_vpu_call", **kwargs)


@pytest.mark.parametrize("elements", [137, 8193])
def test_private_chain_derives_bounded_chunks_and_partial_rows(elements):
    module, graph = case(elements)
    target = adapter(module, graph)
    before, after = target.lower(set()), target.lower({(0, 1, "m")})
    assert not before.realized
    assert after.realized == {(0, 1, "m")}
    assert after.removed_transfer_bytes == ((elements + 7) // 8) * 32
    assert target.handoff_facts(0, 1, "m")["equivalent_to_materialized"]
    assert "if(!safe) { materialized_chain" in after.source
    assert "chain_pack(c[base+i]" in after.source


def test_unselected_f32_policy_cannot_claim_lossless_bf16_lowering():
    module, graph = case()
    with pytest.raises(ValueError, match="explicitly selected"):
        F32MultiplyAdapter(module, graph, PROFILE, boundary_policy="strict_f32")


@pytest.mark.parametrize("change", ["fanout", "live_out", "argument"])
def test_selected_edge_cannot_hide_observable_intermediate(change):
    module, graph = case()
    if change == "fanout":
        graph["nodes"][1]["inputs"][1] = "m"
    elif change == "live_out":
        graph["results"].append("m")
    else:
        graph["buffers"]["m"]["kind"] = "arg"
    with pytest.raises(ValueError, match="no legal selected handoff|buffer map differs"):
        adapter(module, graph).lower({(0, 1, "m")})


@pytest.mark.parametrize("change", ["arithmetic", "operand", "type", "fastmath"])
def test_no_names_or_graph_labels_can_override_actual_typed_function(change):
    module, graph = case()
    if change == "type":
        graph["buffers"]["b"]["dtype"] = "bf16"
    else:
        multiply = next(op for op in module.walk() if op.name == "arith.mulf")
        if change == "operand":
            multiply.operands = [multiply.operands[0], multiply.parent.args[2]]
        elif change == "fastmath":
            from xdsl.dialects.arith import FastMathFlagsAttr
            multiply.properties["fastmath"] = FastMathFlagsAttr("fast")
        else:
            text = str(module).replace("arith.mulf", "arith.addf")
            module = _captured_module(text)
    with pytest.raises(ValueError, match="scalar body|strict floating|equal static f32"):
        adapter(module, deepcopy(graph))
