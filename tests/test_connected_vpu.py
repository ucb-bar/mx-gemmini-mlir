"""Resident handoffs must remove only private, lossless BF16 DMA pairs."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.connected_vpu import VpuDispatchAdapter, scalar_vpu_fragments
from mx_gemmini_support.target_profile import load_profile, profile_sha256


ROOT = Path(__file__).resolve().parents[1]
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json")


def _case(rows=64, consumer_row=4096):
    digest = profile_sha256(PROFILE)
    fragments = []
    for index, (source, target) in enumerate(((0, 4096), (consumer_row, 8192))):
        binding = (f'profile_sha256 = "{digest}", contract_sha256 = "{"a" * 64}", '
                   f'policy_sha256 = "{"b" * 64}", manifest_sha256 = "{"c" * 64}"')
        mlir = f'''builtin.module attributes {{mx.profile_sha256 = "{digest}",
          mx.contract_sha256 = "{"a" * 64}", mx.policy_sha256 = "{"b" * 64}",
          prov.quantization_manifest_sha256 = "{"c" * 64}"}} {{
          func.func @kernel() {{
            "mx_gemmini.vpu_execute"() {{site_id = "op:{index}", kind = "muls",
              src1_row = {source} : i32, src2_row = 0 : i32, dst_row = {target} : i32,
              rows = {rows} : i32, reduction_length = 1 : i32, broadcast = false,
              immediate_bf16 = 16384 : i32, {binding}}} : () -> ()
            func.return
          }}
        }}'''
        fragments.append(dict(mlir=mlir, buffers=dict(
            src1="input" if index == 0 else "middle", src2=f"unused{index}",
            output="middle" if index == 0 else "result")))
    program = dict(nodes=[
        dict(kind="dispatch", inputs=["input"], outputs=["middle"]),
        dict(kind="dispatch", inputs=["middle"], outputs=["result"])], results=["result"], buffers={
            name: dict(shape=[rows, 8], dtype="bf16", kind="arg" if name == "input" else "intermediate")
            for name in ("input", "middle", "result")})
    return program, fragments


@pytest.mark.parametrize("rows", [1, 17, 64])
def test_direct_handoff_preserves_arithmetic_and_removes_exact_dma_bytes(rows):
    graph, fragments = _case(rows)
    adapter = VpuDispatchAdapter(graph, PROFILE, fragments)
    before, after = adapter.lower(set()), adapter.lower({(0, 1, "middle")})
    assert after.removed_transfer_bytes == rows * 16 * 2
    assert len(before.commands) - len(after.commands) == 2 * ((rows + 15) // 16)
    assert not any(getattr(c, "rs1", None) and c.rs1.buffer == "middle" for c in after.commands)


@pytest.mark.parametrize("change", ["fanout", "live_out", "different_row"])
def test_backend_refuses_an_illegal_selected_edge_even_if_planner_is_bypassed(change):
    graph, fragments = _case(consumer_row=3000 if change == "different_row" else 4096)
    if change == "fanout":
        # A duplicate input slot is another observable use, even in the same dispatch.
        graph["nodes"].append(deepcopy(graph["nodes"][1]))
        fragments.append(deepcopy(fragments[1]))
    elif change == "live_out":
        graph["results"].append("middle")
    adapter = VpuDispatchAdapter(graph, PROFILE, fragments)
    assert adapter.handoff_facts(0, 1, "middle") is None
    with pytest.raises(ValueError, match="no legal direct path"):
        adapter.lower({(0, 1, "middle")})


def test_f32_boundaries_cannot_claim_lossless_bf16_handoff():
    graph, fragments = _case()
    graph["buffers"]["middle"]["dtype"] = "f32"
    with pytest.raises(ValueError, match="contiguous BF16"):
        VpuDispatchAdapter(graph, PROFILE, fragments)


def _captured_module(text):
    from xdsl.context import Context
    from xdsl.dialects import arith, builtin, func, linalg, tensor
    from xdsl.parser import Parser

    context = Context()
    for dialect in (arith.Arith, builtin.Builtin, func.Func, linalg.Linalg, tensor.Tensor):
        context.load_dialect(dialect)
    return Parser(context, text).parse_module()


@pytest.mark.parametrize("rows", [17, 64])
def test_captured_pytorch_arithmetic_rebuilds_the_spike_qualified_commands(rows):
    directory = ROOT / f"docs/evidence/connected_dispatch_20261016/rows{rows}"
    source_sha = hashlib.sha256((directory / "source.mlir").read_bytes()).hexdigest()
    program = json.loads((directory / "dispatch_program.json").read_text())
    fragments = scalar_vpu_fragments(_captured_module((directory / "outlined.mlir").read_text()),
                                     program, PROFILE, source_sha256=source_sha)
    adapter = VpuDispatchAdapter(program, PROFILE, fragments)
    intermediate = program["nodes"][0]["outputs"][0]
    lowered = adapter.lower({(0, 1, intermediate)})
    assert [asdict(command) for command in lowered.commands] == json.loads(
        (directory / "connected/commands.json").read_text())


@pytest.mark.parametrize("old,new", [("1.1015625", "1.1"), ("arith.mulf", "arith.divf")])
def test_source_changes_cannot_silently_reuse_scalar_vpu_lowering(old, new):
    directory = ROOT / "docs/evidence/connected_dispatch_20261016/rows17"
    text = (directory / "outlined.mlir").read_text()
    assert old in text
    changed = _captured_module(text.replace(old, new))
    with pytest.raises(ValueError, match="not exactly BF16|unsupported operation"):
        scalar_vpu_fragments(changed, json.loads((directory / "dispatch_program.json").read_text()),
                             PROFILE, source_sha256="a" * 64)
