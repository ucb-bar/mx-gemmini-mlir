"""A branch must keep its aliased matrix output live until the merge consumes it."""

from copy import deepcopy
from pathlib import Path

import pytest

from merlin.xdsl_dialects.lowering.dispatch_program import Buffer, DispatchProgram, Node
from mx_gemmini_support.dispatch_graph_memory import dispatch_graph_memory_plan
from mx_gemmini_support.physical_memory_plan import PhysicalMemoryError
from mx_gemmini_support.target_profile import load_profile


PROFILE = load_profile(Path(__file__).resolve().parents[2] /
                       "npu-exploration/examples/spike_model_slice/target_profile.v3.json")
H = "a" * 64


def branch() -> DispatchProgram:
    return DispatchProgram("forward", [0, 1], {
        "a": Buffer("a", [8, 128], "f32", "arg", 0),
        "b": Buffer("b", [128, 344], "f32", "arg", 1),
        "m": Buffer("m", [8, 344], "f32", "intermediate"),
        "v0": Buffer("v0", [2752], "f32", "intermediate"),
        "v1": Buffer("v1", [1, 8, 344], "f32", "intermediate"),
        "s": Buffer("s", [1, 8, 344], "f32", "intermediate"),
        "z": Buffer("z", [1, 8, 344], "f32", "intermediate"),
    }, [
        Node("dispatch", "matmul_6", ["a", "b"], ["m"], captures=[]),
        Node("view", "tensor.collapse_shape", ["m"], ["v0"], captures=[]),
        Node("view", "tensor.expand_shape", ["v0"], ["v1"], captures=[]),
        Node("dispatch", "sigmoid_0", ["v1"], ["s"], captures=[]),
        Node("dispatch", "mul_12", ["v1", "s"], ["z"], captures=[]),
    ], ["z"])


def proof(program: DispatchProgram):
    return dispatch_graph_memory_plan(
        program, PROFILE, source_mlir_sha256=H, recipe_sha256=H,
        driver_sha256=H,
        dispatch_source_sha256={"matmul_6": H, "sigmoid_0": H, "mul_12": H})


def test_branch_keeps_alias_root_and_sigmoid_live_through_merge():
    result = proof(branch())
    placement = result["placement"]
    assert result["status"] == "verified_merlin_dag_host_buffer_liveness_only"
    assert placement["m"]["last_use_completed_at"] == 4
    assert placement["v1"]["alias_of"] == "m"
    assert placement["s"]["last_use_completed_at"] == 4
    assert placement["z"]["last_use_completed_at"] == 5


@pytest.mark.parametrize("change", [
    lambda p: (setattr(p.nodes[3], "regions", 1), setattr(p.nodes[3], "captures", None)),
    lambda p: (setattr(p.nodes[3], "regions", 1), setattr(p.nodes[3], "captures", ["a"])),
    lambda p: p.nodes[4].inputs.append("unknown"),
    lambda p: p.nodes[4].outputs.__setitem__(0, "s"),
    lambda p: p.buffers["v0"].shape.__setitem__(0, 2751),
])
def test_incomplete_or_changed_graph_refuses(change):
    program = deepcopy(branch())
    change(program)
    with pytest.raises(PhysicalMemoryError):
        proof(program)
