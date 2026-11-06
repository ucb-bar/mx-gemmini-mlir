"""Typed, source-bound pass trace for one captured matrix/VPU branch.

This module records checks already performed by the captured-branch compiler.
It deliberately accepts only op169..173 and never treats a host-buffer proof
as an on-chip proof. The static trace is certified before the final branch
ELF; a separately verified prefix guest may have run earlier. Spike
verification is appended only after both final runs match independent oracles.
"""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

from .physical_memory_plan import source_bound_capacities
from .target_profile import profile_sha256


SCHEMA = "mx_gemmini.connected_pass_trace.v1"
PRE_ELF_STATUS = "verified_pre_final_branch_elf_pass_sequence"
SPIKE_STATUS = "verified_selected_spike_captured_branch_pass_sequence"
PASS_NAMES = (
    "capture_recipe_binding",
    "legal_placement",
    "host_buffer_liveness",
    "on_chip_row_planning",
    "target_command_emission",
    "selected_spike_verification",
)
_OPS = (
    ("dispatch", "matmul_6"),
    ("view", "tensor.collapse_shape"),
    ("view", "tensor.expand_shape"),
    ("dispatch", "sigmoid_0"),
    ("dispatch", "mul_12"),
)
_PLACEMENT = {
    "matmul_6": "mesh",
    "view_19": "row_major_layout_alias",
    "sigmoid_0#sigmoid0": "vpu",
    "mul_12#mul0": "vpu",
    "matrix_to_vpu_bridge": "guest_host_unpack_padded_bf16_and_widen_to_f32",
}
_BRIDGE = {
    "kind": "guest_host_unpack_padded_bf16_and_widen_to_f32",
    "logical_elements": 2752,
    "fp32_staging_bytes": 11008,
}
_QUALIFIED_GEOMETRY = {
    "MxDim4Fp4E3M2VpuGemminiRocketConfig": (4, 2),
    "MxDim8Fp4E3M2E4M3VpuGemminiRocketConfig": (8, 4),
    "MxFp4E3M2E4M3VpuGemminiRocketConfig": (16, 8),
    "MxFp4E3M2E4M3VpuAcc32GemminiRocketConfig": (16, 8),
    "MxDim32E4M3VpuWideMemGemminiRocketConfig": (32, 16),
}
_GEMMINI_REVISION = "12912323f1aa66d7b28687f01faf0a33af455f6f"


class ConnectedPassRefusal(ValueError):
    """Pre-ELF or Spike gate failure with an actionable source-site binding."""

    def __init__(self, pass_name: str, recipe_sha256: str, reason: str):
        self.pass_name = pass_name
        self.source_site = "matmul_6..mul_12 (captured op169..173)"
        self.recipe_sha256 = recipe_sha256
        self.reason = reason
        super().__init__(f"{pass_name} rejects {self.source_site} for recipe "
                         f"{recipe_sha256}: {reason}")

    def to_dict(self) -> dict[str, str]:
        return {"pass": self.pass_name, "source_site": self.source_site,
                "recipe_sha256": self.recipe_sha256, "reason": self.reason}


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _file_digest(value: object) -> str:
    """Hash the existing deterministic JSON artifact serialization."""
    return hashlib.sha256((json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n").encode()).hexdigest()


def _hash(value: object) -> bool:
    return (type(value) is str and len(value) == 64 and
            all(ch in "0123456789abcdef" for ch in value))


def _need(ok: bool, stage: str, recipe: str, reason: str) -> None:
    if not ok:
        raise ConnectedPassRefusal(stage, recipe, reason)


def _record(name: str, input_kind: str, input_sha: str,
            output_kind: str, output_sha: str, *, facts: Mapping[str, object],
            invariants: tuple[str, ...], refusal_policy: tuple[str, ...]) -> dict:
    record = {
        "name": name,
        "input": {"kind": input_kind, "sha256": input_sha},
        "output": {"kind": output_kind, "sha256": output_sha},
        "required_target_facts": dict(facts),
        "preserved_invariants": list(invariants),
        "refusal_policy": list(refusal_policy),
        "refusals": [],
        "status": "verified",
    }
    record["digest"] = _digest(record)
    return record


def _seal(trace: dict) -> dict:
    trace["digest"] = _digest(trace)
    return trace


def certify_connected_passes(*, source_binding: Mapping[str, object],
                             profile: dict, recipe_sha256: str,
                             graph_plan: dict, graph_plan_sha256: str,
                             onchip_plan: dict, onchip_plan_sha256: str,
                             driver_sha256: str, prefix_receipt_sha256: str,
                             placement: Mapping[str, str],
                             bridge: Mapping[str, object]) -> dict:
    """Certify five ordered passes before emitting the final branch ELF.

    The caller writes each plan with ``json.dumps(..., indent=2,
    sort_keys=True) + '\\n'`` and supplies that file's SHA-256. No output
    directory or host path participates in the trace identity.
    """
    recipe = recipe_sha256 if _hash(recipe_sha256) else "<invalid-recipe-sha256>"
    _need(_hash(recipe_sha256), PASS_NAMES[0], recipe,
          "recipe needs an exact SHA-256")
    _need(_hash(prefix_receipt_sha256), PASS_NAMES[0], recipe,
          "verified prefix receipt SHA-256 is required")
    source_sha = source_binding.get("source_mlir_sha256")
    _need(_hash(source_sha) and _hash(source_binding.get("selected_ops_sha256")) and
          source_binding.get("selected_op_ids") == [169, 170, 171, 172, 173] and
          source_binding.get("selected_regions") ==
          ["matmul_6", "view_19", "view_19", "sigmoid_0", "mul_12"],
          PASS_NAMES[0], recipe, "captured source sites or source hash differ")
    profile_hash = profile_sha256(profile)
    dim = profile.get("geometry", {}).get("mesh_rows")
    lanes = profile.get("resources", {}).get("vpu_config", {}).get("lanes")
    _need(type(dim) is int and type(lanes) is int and
          profile.get("schema") == "mx_gemmini.target_profile.v3" and
          profile.get("geometry", {}).get("mesh_columns") == dim and
          _QUALIFIED_GEOMETRY.get(profile.get("name")) == (dim, lanes) and
          profile.get("resources", {}).get("vpu") is True and
          profile.get("resources", {}).get("scratchpad_row_bits") == dim * 8 and
          profile.get("source", {}).get("gemmini_revision") == _GEMMINI_REVISION,
          PASS_NAMES[1], recipe,
          "selected branch needs a source-bound coupled mesh/VPU profile")
    _need(dict(placement) == _PLACEMENT and dict(bridge) == _BRIDGE,
          PASS_NAMES[1], recipe, "placement or explicit host bridge differs")
    program = graph_plan.get("program")
    nodes = program.get("nodes") if isinstance(program, dict) else None
    _need(graph_plan.get("schema") == "mx_gemmini.dispatch_graph_memory.v1" and
          graph_plan.get("status") == "verified_merlin_dag_host_buffer_liveness_only" and
          isinstance(nodes, list) and
          [(node.get("kind"), node.get("op")) for node in nodes
           if isinstance(node, dict)] == list(_OPS) and len(nodes) == len(_OPS) and
          graph_plan.get("program_sha256") == _digest(program) and
          graph_plan.get("source_binding") == dict(source_binding),
          PASS_NAMES[2], recipe, "Merlin dispatch graph identity or host proof differs")
    bindings = graph_plan.get("bindings", {})
    _need(isinstance(bindings, dict) and
          bindings.get("profile_sha256") == profile_hash and
          bindings.get("source_mlir_sha256") == source_sha and
          bindings.get("recipe_sha256") == recipe_sha256 and
          bindings.get("driver_sha256") == driver_sha256 and
          bindings.get("dispatch_program_sha256") == graph_plan.get("program_sha256") and
          _hash(bindings.get("dispatch_sources_sha256")) and
          _hash(graph_plan_sha256) and
          graph_plan_sha256 == _file_digest(graph_plan),
          PASS_NAMES[2], recipe, "host liveness binding or serialized plan hash differs")
    dispatch_sources = graph_plan.get("dispatch_source_sha256", {})
    _need(isinstance(dispatch_sources, dict) and
          set(dispatch_sources) == {"matmul_6", "sigmoid_0", "mul_12"} and
          all(_hash(value) for value in dispatch_sources.values()) and
          bindings["dispatch_sources_sha256"] == _digest(dispatch_sources),
          PASS_NAMES[2], recipe, "emitted dispatch source hashes are incomplete")
    onchip_bindings = onchip_plan.get("bindings", {})
    onchip_status = onchip_plan.get("status")
    onchip_status_ok = (onchip_status == "sequential_prefix_and_mul_rows_certified" or
                        (dim == 16 and onchip_status ==
                         "sequential_prefix_and_mul_rows_certified_dim16"))
    _need(onchip_plan.get("schema") == "npu_exploration.connected_branch_onchip.v1" and
          onchip_status_ok and
          onchip_plan.get("prefix_physical_status") ==
          "fixed_and_dynamic_rows_verified_no_live_overlap" and
          onchip_plan.get("graph_program_sha256") == graph_plan.get("program_sha256") and
          onchip_plan.get("profile_sha256") == profile_hash and
          isinstance(onchip_bindings, dict) and
          onchip_bindings.get("graph_program_sha256") == graph_plan.get("program_sha256") and
          onchip_bindings.get("driver_sha256") == driver_sha256 and
          onchip_bindings.get("profile_sha256") == profile_hash and
          _hash(onchip_bindings.get("prefix_physical_plan_sha256")) and
          _hash(onchip_bindings.get("multiply_source_sha256")) and
          _hash(onchip_plan_sha256) and
          onchip_plan_sha256 == _file_digest(onchip_plan),
          PASS_NAMES[3], recipe, "on-chip effect or source binding differs")
    events = onchip_plan.get("events")
    try:
        scratchpad_capacity = source_bound_capacities(profile)["scratchpad"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ConnectedPassRefusal(PASS_NAMES[3], recipe,
                                   "selected source has no proved row capacity") from exc
    expected_mul_commands = (2752 + 1024 * lanes - 1) // (1024 * lanes)
    _need(isinstance(events, list) and len(events) > 2 and
          all(isinstance(event, dict) and event.get("kind") in {"command", "fence"}
              for event in events) and
          events[-1].get("kind") == "fence" and
          events[-1].get("completes") == ["gemmini"] and
          type(onchip_plan.get("prefix_final_fence_index")) is int and
          onchip_plan["prefix_final_fence_index"] >= 0 and
          type(onchip_plan.get("scratchpad_peak_live_rows")) is int and
          onchip_plan["scratchpad_peak_live_rows"] <= scratchpad_capacity and
          onchip_plan.get("emitted_vpu_mul_commands") == expected_mul_commands,
          PASS_NAMES[3], recipe, "row capacity, typed multiply effect or completion fence differs")
    _need(_hash(driver_sha256) and
          onchip_plan["bindings"]["driver_sha256"] == driver_sha256,
          PASS_NAMES[4], recipe, "emitted guest driver differs from certified effects")

    placement_hash = _digest({"placement": dict(placement), "bridge": dict(bridge)})
    facts = {"profile_sha256": profile_hash, "mesh_rows": dim,
             "mesh_columns": dim, "vpu_lanes": lanes,
             "multiply_chunks": expected_mul_commands,
             "scratchpad_capacity_rows": scratchpad_capacity,
             "prefix_receipt_sha256": prefix_receipt_sha256,
             "prefix_physical_plan_sha256":
                 onchip_bindings["prefix_physical_plan_sha256"]}
    passes = [
        _record(PASS_NAMES[0], "captured_mlir", source_sha,
                "dispatch_program", graph_plan["program_sha256"], facts=facts,
                invariants=("exact captured op169..173 source identity",
                            "user recipe hash is unchanged"),
                refusal_policy=("unknown or changed source operation", "unbound recipe")),
        _record(PASS_NAMES[1], "dispatch_program", graph_plan["program_sha256"],
                "placement_policy", placement_hash, facts=facts,
                invariants=("source-selected matrix/VPU placement",
                            "explicit host BF16/FP32 bridge"),
                refusal_policy=("unsupported placement", "hidden host bridge")),
        _record(PASS_NAMES[2], "placement_policy", placement_hash,
                "host_liveness_plan", graph_plan_sha256, facts=facts,
                invariants=("view aliases and branch uses remain live",
                            "every dispatch source hash is bound"),
                refusal_policy=("unknown dispatch", "unproved alias or host effect")),
        _record(PASS_NAMES[3], "host_liveness_plan", graph_plan_sha256,
                "on_chip_row_plan", onchip_plan_sha256, facts=facts,
                invariants=("typed row effects precede final branch ELF",
                            "prefix fence completes before multiply"),
                refusal_policy=("unmodeled command", "overlapping or over-capacity rows")),
        _record(PASS_NAMES[4], "on_chip_row_plan", onchip_plan_sha256,
                "guest_driver", driver_sha256, facts=facts,
                invariants=("emitted driver matches certified source and effects",
                            "verified prefix receipt is an explicit dependency"),
                refusal_policy=("driver or emitter hash drift",)),
    ]
    return _seal({
        "schema": SCHEMA, "status": PRE_ELF_STATUS,
        "source_site": "matmul_6..mul_12 (captured op169..173)",
        "source_mlir_sha256": source_sha,
        "graph_program_sha256": graph_plan["program_sha256"],
        "profile_sha256": profile_hash, "recipe_sha256": recipe_sha256,
        "prefix_receipt_sha256": prefix_receipt_sha256,
        "placement_sha256": placement_hash,
        "bridge": dict(bridge), "passes": passes,
    })


def verify_connected_pass_trace(trace: dict, *, require_spike: bool = True) -> None:
    """Reject a changed pass record, ordering, digest chain, or claimed status."""
    recipe = trace.get("recipe_sha256", "<invalid-recipe-sha256>")
    names = PASS_NAMES if require_spike else PASS_NAMES[:-1]
    status = SPIKE_STATUS if require_spike else PRE_ELF_STATUS
    passes = trace.get("passes")
    _need(trace.get("schema") == SCHEMA and trace.get("status") == status and
          isinstance(passes, list) and len(passes) == len(names) and
          [record.get("name") for record in passes if isinstance(record, dict)] ==
          list(names) and _hash(trace.get("digest")) and
          trace["digest"] == _digest({k: v for k, v in trace.items() if k != "digest"}),
          "pass_trace_verification", recipe, "trace schema, order or digest differs")
    for record in passes:
        _need(record.get("status") == "verified" and
              record.get("refusals") == [] and
              _hash(record.get("digest")) and
              record["digest"] == _digest({k: v for k, v in record.items()
                                            if k != "digest"}) and
              all(isinstance(record.get(side), dict) and
                  _hash(record[side].get("sha256")) and
                  isinstance(record[side].get("kind"), str)
                  for side in ("input", "output")),
              "pass_trace_verification", recipe, "pass record or typed artifact differs")
    for before, after in zip(passes, passes[1:]):
        _need(before["output"] == after["input"], "pass_trace_verification", recipe,
              "typed artifact chain is broken")
    _need(passes[0]["input"]["sha256"] == trace.get("source_mlir_sha256") and
          passes[0]["output"]["sha256"] == trace.get("graph_program_sha256") and
          passes[1]["output"]["sha256"] == trace.get("placement_sha256") and
          all(_hash(trace.get(key)) for key in
              ("source_mlir_sha256", "graph_program_sha256", "profile_sha256",
               "recipe_sha256", "placement_sha256", "prefix_receipt_sha256")),
          "pass_trace_verification", recipe, "trace source/profile/recipe bindings differ")
    facts = passes[0].get("required_target_facts")
    _need(isinstance(facts, dict) and
          all(record.get("required_target_facts") == facts for record in passes[:-1]) and
          facts.get("profile_sha256") == trace.get("profile_sha256") and
          type(facts.get("mesh_rows")) is int and
          type(facts.get("vpu_lanes")) is int and
          type(facts.get("multiply_chunks")) is int and
          facts.get("mesh_rows") in (4, 8, 16, 32) and
          facts.get("mesh_columns") == facts.get("mesh_rows") and
          facts.get("vpu_lanes") == facts.get("mesh_rows") // 2 and
          facts.get("multiply_chunks") ==
          (2752 + 1024 * facts["vpu_lanes"] - 1) // (1024 * facts["vpu_lanes"]),
          "pass_trace_verification", recipe,
          "pass target facts differ from the selected coupled geometry")


def finalize_selected_spike_pass(trace: dict, *, functional_stdout_sha256: str,
                                 timed_stdout_sha256: str, extension_sha256: str,
                                 oracle_results: dict, timing_scope: str) -> dict:
    """Append selected Spike verification after both full-output oracle checks."""
    verify_connected_pass_trace(trace, require_spike=False)
    recipe = trace["recipe_sha256"]
    _need(all(_hash(value) for value in
              (functional_stdout_sha256, timed_stdout_sha256, extension_sha256)),
          PASS_NAMES[-1], recipe, "selected extension or Spike output hash is absent")
    _need(timing_scope == "Spike model prediction; no RTL or FireSim calibration",
          PASS_NAMES[-1], recipe, "timing scope must name uncalibrated Spike prediction")
    _need(isinstance(oracle_results, dict) and
          set(oracle_results) == {"functional", "timed"},
          PASS_NAMES[-1], recipe, "both functional and timed oracle results are required")
    dim = trace["passes"][0]["required_target_facts"]["mesh_rows"]
    matrix_outputs = ((8 + dim - 1) // dim * dim) * ((344 + dim - 1) // dim * dim)
    for mode in ("functional", "timed"):
        result = oracle_results[mode]
        matrix = result.get("matrix", {}) if isinstance(result, dict) else {}
        multiply = result.get("multiply", {}) if isinstance(result, dict) else {}
        _need(matrix.get("elements") == matrix_outputs and matrix.get("mismatches") == 0 and
              _hash(matrix.get("actual_sha256")) and
              matrix.get("actual_sha256") == matrix.get("expected_sha256") and
              multiply.get("elements") == 2752 and
              _hash(multiply.get("actual_sha256")) and
              multiply.get("actual_sha256") == multiply.get("expected_sha256") and
              _hash(multiply.get("f32_actual_sha256")) and
              multiply.get("f32_actual_sha256") == multiply.get("f32_expected_sha256") and
              all(multiply.get(key) == 0 for key in
                  ("branch_input_mismatches", "sigmoid_edge_mismatches",
                   "multiply_mismatches", "f32_bit_mismatches")),
              PASS_NAMES[-1], recipe, f"{mode} full-output independent oracle differs")
    _need(oracle_results["functional"]["matrix"]["actual_sha256"] ==
          oracle_results["timed"]["matrix"]["actual_sha256"] and
          oracle_results["functional"]["multiply"]["actual_sha256"] ==
          oracle_results["timed"]["multiply"]["actual_sha256"] and
          oracle_results["functional"]["multiply"]["f32_actual_sha256"] ==
          oracle_results["timed"]["multiply"]["f32_actual_sha256"],
          PASS_NAMES[-1], recipe, "functional and timed selected Spike outputs differ")
    output = _digest({"functional_stdout_sha256": functional_stdout_sha256,
                      "timed_stdout_sha256": timed_stdout_sha256,
                      "extension_sha256": extension_sha256,
                      "oracle_results": oracle_results,
                      "timing_scope": timing_scope})
    final = {key: value for key, value in trace.items() if key != "digest"}
    final["status"] = SPIKE_STATUS
    final["pre_elf_digest"] = trace["digest"]
    final["passes"] = list(trace["passes"]) + [
        _record(PASS_NAMES[-1], "guest_driver",
                trace["passes"][-1]["output"]["sha256"],
                "selected_spike_oracle_results", output,
                facts={"profile_sha256": trace["profile_sha256"],
                       "extension_sha256": extension_sha256,
                       "timing_scope": timing_scope},
                invariants=("functional and timed full-output oracle parity",
                            "FP32-bit multiply parity"),
                refusal_policy=("missing selected extension", "any output mismatch")),
    ]
    _seal(final)
    verify_connected_pass_trace(final)
    return final
