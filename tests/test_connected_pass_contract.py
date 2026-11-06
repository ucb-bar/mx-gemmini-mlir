"""Focused integrity and refusal checks for the captured branch pass trace."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import unittest

from mx_gemmini_support.command_ir import Command, Fence
from mx_gemmini_support.connected_pass_contract import (
    ConnectedPassRefusal, _digest, _file_digest, _host_tile_event,
    certify_connected_passes,
    finalize_selected_spike_pass, verify_connected_pass_trace,
)
from mx_gemmini_support.target_profile import profile_sha256
from mx_gemmini_support.vpu_add_kernel import elementwise_issue_commands


H = "a" * 64
R = "b" * 64
D = "c" * 64
P = "d" * 64
M = "e" * 64
PLACEMENT = {
    "matmul_6": "mesh",
    "view_19": "row_major_layout_alias",
    "sigmoid_0#sigmoid0": "vpu",
    "mul_12#mul0": "vpu",
    "matrix_to_vpu_bridge": "guest_host_unpack_padded_bf16_and_widen_to_f32",
}
BRIDGE = {"kind": "guest_host_unpack_padded_bf16_and_widen_to_f32",
          "logical_elements": 2752, "fp32_staging_bytes": 11008}


def _events(profile: dict) -> tuple[list[dict], list[int], int]:
    lanes = profile["resources"]["vpu_config"]["lanes"]
    chunks = [min(1024 * lanes, 2752 - offset)
              for offset in range(0, 2752, 1024 * lanes)]
    events = []
    offset = 0
    command_count = 0
    for i, count in enumerate(chunks):
        events.append(_host_tile_event("narrow", i, offset, count))
        for j, command in enumerate(elementwise_issue_commands(
                profile, elements=count, operation="mul")):
            if isinstance(command, Command):
                events.append({"id": f"mul_command_{i}_{j}",
                               "kind": "command", "source": "emitted typed multiply command",
                               "queue": "gemmini", "completes": [],
                               "command": asdict(command), "reads": [], "writes": []})
            else:
                assert isinstance(command, Fence)
                events.append({"id": f"mul_fence_{i}_{j}",
                               "kind": "fence", "source": "emitted multiply fence",
                               "queue": None, "completes": ["gemmini"],
                               "command": "fence", "reads": [], "writes": []})
            command_count += 1
        events.append(_host_tile_event("widen", i, offset, count))
        offset += count
    return events, chunks, command_count


def _inputs() -> dict:
    profile = {
        "schema": "mx_gemmini.target_profile.v3",
        "name": "MxFp4E3M2E4M3VpuGemminiRocketConfig",
        "transport": "rocket_rocc",
        "geometry": {"mesh_rows": 16, "mesh_columns": 16},
        "resources": {"scratchpad_bytes": 262144, "scratchpad_row_bits": 128,
                      "scratchpad_banks": 4, "scratchpad_bank_rows": 4096,
                      "accumulator_bytes": 65536, "accumulator_row_bits": 1024,
                      "accumulator_rows": 512,
                      "scale_mem_config": {"size_bytes": 16384,
                                           "subbank_line_bytes": 16},
                      "vpu": True, "vpu_config": {"lanes": 8}},
        "source": {"gemmini_revision":
                   "12912323f1aa66d7b28687f01faf0a33af455f6f"},
    }
    profile_hash = profile_sha256(profile)
    source = {"selected_op_ids": [169, 170, 171, 172, 173],
              "selected_regions": ["matmul_6", "view_19", "view_19",
                                   "sigmoid_0", "mul_12"],
              "selected_ops_sha256": H, "source_mlir_sha256": H}
    program = {"nodes": [
        {"kind": "dispatch", "op": "matmul_6"},
        {"kind": "view", "op": "tensor.collapse_shape"},
        {"kind": "view", "op": "tensor.expand_shape"},
        {"kind": "dispatch", "op": "sigmoid_0"},
        {"kind": "dispatch", "op": "mul_12"},
    ]}
    dispatch_sources = {"matmul_6": H, "sigmoid_0": H, "mul_12": H}
    program_hash = _digest(program)
    graph = {"schema": "mx_gemmini.dispatch_graph_memory.v1",
             "status": "verified_merlin_dag_host_buffer_liveness_only",
             "program": program, "program_sha256": program_hash,
             "source_binding": source,
             "dispatch_source_sha256": dispatch_sources,
             "bindings": {"profile_sha256": profile_hash,
                          "source_mlir_sha256": H, "recipe_sha256": R,
                          "driver_sha256": D,
                          "dispatch_program_sha256": program_hash,
                          "dispatch_sources_sha256": _digest(dispatch_sources)}}
    events, chunks, command_count = _events(profile)
    onchip = {"schema": "npu_exploration.connected_branch_onchip.v1",
              "status": "sequential_prefix_and_mul_rows_certified_dim16",
              "prefix_physical_status":
                  "fixed_and_dynamic_rows_verified_no_live_overlap",
              "graph_program_sha256": program_hash,
              "profile_sha256": profile_hash,
              "bindings": {"graph_program_sha256": program_hash,
                           "driver_sha256": D, "profile_sha256": profile_hash,
                           "prefix_physical_plan_sha256": P,
                           "multiply_source_sha256": M},
              "events": events,
              "vpu_chunk_elements": chunks,
              "emitted_multiply_command_count": command_count,
              "prefix_final_fence_index": 58,
              "scratchpad_peak_live_rows": 3648,
              "emitted_vpu_mul_commands": 1}
    return {"source_binding": source, "profile": profile,
            "recipe_sha256": R, "graph_plan": graph,
            "graph_plan_sha256": _file_digest(graph),
            "onchip_plan": onchip, "onchip_plan_sha256": _file_digest(onchip),
            "driver_sha256": D, "prefix_receipt_sha256": P,
            "placement": PLACEMENT, "bridge": BRIDGE}


def _oracle() -> dict:
    matrix = {"elements": 5632, "mismatches": 0,
              "actual_sha256": H, "expected_sha256": H}
    multiply = {"elements": 2752, "branch_input_mismatches": 0,
                "sigmoid_edge_mismatches": 0, "multiply_mismatches": 0,
                "f32_bit_mismatches": 0, "actual_sha256": M,
                "expected_sha256": M, "f32_actual_sha256": D,
                "f32_expected_sha256": D}
    return {mode: {"matrix": deepcopy(matrix), "multiply": deepcopy(multiply)}
            for mode in ("functional", "timed")}


class ConnectedPassContractTest(unittest.TestCase):
    def test_pre_final_elf_trace_is_stable_and_typed(self):
        args = _inputs()
        first = certify_connected_passes(**args)
        second = certify_connected_passes(**deepcopy(args))
        self.assertEqual(first, second)
        self.assertEqual(first["status"],
                         "verified_pre_final_branch_elf_pass_sequence")
        self.assertEqual(len(first["passes"]), 5)
        self.assertEqual(first["passes"][0]["input"]["kind"], "captured_mlir")
        self.assertEqual(first["passes"][-1]["output"]["kind"], "guest_driver")
        verify_connected_pass_trace(first, require_spike=False)

    def test_unknown_captured_operation_refuses_with_site_and_recipe(self):
        args = _inputs()
        args["graph_plan"]["program"]["nodes"][3]["op"] = "unknown_0"
        args["graph_plan"]["program_sha256"] = _digest(args["graph_plan"]["program"])
        args["graph_plan"]["bindings"]["dispatch_program_sha256"] = args["graph_plan"]["program_sha256"]
        args["graph_plan_sha256"] = _file_digest(args["graph_plan"])
        with self.assertRaises(ConnectedPassRefusal) as caught:
            certify_connected_passes(**args)
        self.assertEqual(caught.exception.pass_name, "host_buffer_liveness")
        self.assertIn("matmul_6..mul_12", str(caught.exception))
        self.assertIn(R, str(caught.exception))

    def test_changed_placement_or_host_bridge_refuses(self):
        args = _inputs()
        args["placement"] = {**PLACEMENT, "mul_12#mul0": "host"}
        with self.assertRaisesRegex(ConnectedPassRefusal, "legal_placement"):
            certify_connected_passes(**args)
        args = _inputs()
        args["bridge"] = {**BRIDGE, "fp32_staging_bytes": 0}
        with self.assertRaisesRegex(ConnectedPassRefusal, "host bridge"):
            certify_connected_passes(**args)

    def test_unselected_source_revision_refuses_before_driver(self):
        args = _inputs()
        args["profile"]["source"]["gemmini_revision"] = "0" * 40
        with self.assertRaisesRegex(ConnectedPassRefusal, "source-bound coupled"):
            certify_connected_passes(**args)

    def test_row_over_capacity_and_changed_plan_hash_refuse(self):
        args = _inputs()
        args["onchip_plan"]["scratchpad_peak_live_rows"] = 16385
        args["onchip_plan_sha256"] = _file_digest(args["onchip_plan"])
        with self.assertRaisesRegex(ConnectedPassRefusal, "on_chip_row_planning"):
            certify_connected_passes(**args)
        args = _inputs()
        args["graph_plan_sha256"] = H
        with self.assertRaisesRegex(ConnectedPassRefusal, "serialized plan hash"):
            certify_connected_passes(**args)

    def test_changed_host_tile_effect_and_completion_fence_refuse(self):
        args = _inputs()
        args["onchip_plan"]["events"][0]["reads"][0]["length"] -= 4
        args["onchip_plan_sha256"] = _file_digest(args["onchip_plan"])
        with self.assertRaisesRegex(ConnectedPassRefusal, "host narrow effect"):
            certify_connected_passes(**args)
        args = _inputs()
        args["onchip_plan"]["events"][-2]["completes"] = []
        args["onchip_plan_sha256"] = _file_digest(args["onchip_plan"])
        with self.assertRaisesRegex(ConnectedPassRefusal, "fence"):
            certify_connected_passes(**args)
        args = _inputs()
        args["onchip_plan"]["events"].append(
            _host_tile_event("widen", 1, 2752, 0))
        args["onchip_plan_sha256"] = _file_digest(args["onchip_plan"])
        with self.assertRaisesRegex(ConnectedPassRefusal, "extra or missing"):
            certify_connected_passes(**args)

    def test_selected_spike_pass_requires_both_full_oracles(self):
        trace = certify_connected_passes(**_inputs())
        kwargs = {"functional_stdout_sha256": H,
                  "timed_stdout_sha256": R,
                  "extension_sha256": P,
                  "oracle_results": _oracle(),
                  "timing_scope": "Spike model prediction; no RTL or FireSim calibration"}
        final = finalize_selected_spike_pass(trace, **kwargs)
        self.assertEqual(len(final["passes"]), 6)
        self.assertEqual(final["pre_elf_digest"], trace["digest"])
        verify_connected_pass_trace(final)
        bad = deepcopy(kwargs)
        bad["oracle_results"]["timed"]["multiply"]["f32_bit_mismatches"] = 1
        with self.assertRaisesRegex(ConnectedPassRefusal, "timed full-output"):
            finalize_selected_spike_pass(trace, **bad)

    def test_tampered_trace_refuses_even_with_rehashed_top_level(self):
        trace = certify_connected_passes(**_inputs())
        trace["passes"][2]["output"]["sha256"] = H
        trace["digest"] = _digest({k: v for k, v in trace.items() if k != "digest"})
        with self.assertRaisesRegex(ConnectedPassRefusal, "pass record"):
            verify_connected_pass_trace(trace, require_spike=False)

    def test_all_named_coupled_profiles_bind_expected_matrix_counts(self):
        variants = (
            (4, 2, "MxDim4Fp4E3M2VpuGemminiRocketConfig", 2752),
            (8, 4, "MxDim8Fp4E3M2E4M3VpuGemminiRocketConfig", 2752),
            (16, 8, "MxFp4E3M2E4M3VpuGemminiRocketConfig", 5632),
            (32, 16, "MxDim32E4M3VpuWideMemGemminiRocketConfig", 11264),
        )
        for dim, lanes, name, outputs in variants:
            with self.subTest(dim=dim):
                args = _inputs()
                p = args["profile"]
                p["name"] = name
                p["geometry"] = {"mesh_rows": dim, "mesh_columns": dim}
                p["resources"].update(scratchpad_bytes=131072 if dim == 4 else 262144,
                                      scratchpad_row_bits=dim * 8,
                                      scratchpad_bank_rows=(131072 if dim == 4 else 262144) // (4 * dim),
                                      accumulator_row_bits=dim * 64,
                                      accumulator_rows=8192 // dim,
                                      vpu_config={"lanes": lanes})
                ph = profile_sha256(p)
                args["graph_plan"]["bindings"]["profile_sha256"] = ph
                args["onchip_plan"]["profile_sha256"] = ph
                args["onchip_plan"]["bindings"]["profile_sha256"] = ph
                args["onchip_plan"]["status"] = "sequential_prefix_and_mul_rows_certified"
                events, chunks, command_count = _events(p)
                args["onchip_plan"].update(
                    events=events, vpu_chunk_elements=chunks,
                    emitted_multiply_command_count=command_count,
                    emitted_vpu_mul_commands=len(chunks))
                args["graph_plan_sha256"] = _file_digest(args["graph_plan"])
                args["onchip_plan_sha256"] = _file_digest(args["onchip_plan"])
                trace = certify_connected_passes(**args)
                self.assertEqual(trace["passes"][0]["required_target_facts"]["mesh_rows"], dim)
                oracle = _oracle()
                for mode in oracle.values():
                    mode["matrix"]["elements"] = outputs
                final = finalize_selected_spike_pass(
                    trace, functional_stdout_sha256=H, timed_stdout_sha256=R,
                    extension_sha256=P, oracle_results=oracle,
                    timing_scope="Spike model prediction; no RTL or FireSim calibration")
                verify_connected_pass_trace(final)
                p["resources"]["vpu_config"]["lanes"] = lanes + 1
                with self.assertRaisesRegex(ConnectedPassRefusal, "coupled mesh/VPU"):
                    certify_connected_passes(**args)


if __name__ == "__main__":
    unittest.main()
