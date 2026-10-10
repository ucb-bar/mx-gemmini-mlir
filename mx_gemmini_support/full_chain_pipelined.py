"""Add the captured MM1 to Nicolas's source-bound two-branch MX+VPU chain."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .chain_pipelined_graph import (
    INPUTS, OUTPUTS, TwoTileChain, lower_chain_pipelined,
    render_chain_pipelined)
from .chain_pipelined_source import audit_chain_pipelined
from .command_ir import Command, Fence, Operand
from .first_matrix_lowering import emit_verified_first_matrix_commands
from .source_vector_chain import capture_nicolas_vpu_requant
from .target_profile import profile_sha256
from .verify_profile_ir import _operation_name, _text_attr, verify_ir


FULL_INPUTS = ("a1_activation", "a1_scales", "b1_weight", "b1_scales",
               "b2_weight", "b2_scales")
FULL_OUTPUTS = ("c1_scales_mm1", "c1_bf16_observed", *OUTPUTS)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(resources: dict[str, bytes]) -> str:
    return _sha(json.dumps({name: _sha(resources[name]) for name in FULL_INPUTS},
                           sort_keys=True, separators=(",", ":")).encode())


def audit_full_chain_pipelined(source_path: Path, header_path: Path,
                                first_source_path: Path, seam_source_path: Path,
                                profile: dict) -> tuple[dict[str, bytes], dict]:
    """Audit all six packed MM1/B2 inputs against Nicolas's source files."""
    resources, facts = audit_chain_pipelined(source_path, header_path, profile)
    _, first_resources, first_facts = capture_nicolas_vpu_requant(
        seam_source_path, header_path, profile, include_resident_matmul=True,
        first_source_path=first_source_path)
    for name in ("c1_bf16", "b2_weight", "b2_scales"):
        if resources[name] != first_resources[name]:
            raise ValueError("Nicolas MM1 source and two-tile C1/B2 data differ")
    for name in FULL_INPUTS[:4]:
        resources[name] = first_resources[name]
    facts = {
        **facts,
        "source_scope": "captured MM1 from checked packed A1/B1; two VPU scalar branches and resident MM2 tiles",
        "first_source_sha256": first_facts["first_source_sha256"],
        "seam_source_sha256": first_facts["source_sha256"],
        "resource_sha256": {name: _sha(data) for name, data in sorted(resources.items())},
    }
    return resources, facts


def _inject_mm1(preloaded_mlir: str, profile: dict,
                resources: dict[str, bytes]) -> str:
    """Replace the BF16 preload argument with a typed MM1/readout SSA edge."""
    from .chain_pipelined_graph import _module

    module = _module(preloaded_mlir)
    attrs = (f'contract_sha256 = "{_text_attr(module, "mx.contract_sha256")}", '
             f'policy_sha256 = "{_text_attr(module, "mx.policy_sha256")}", '
             f'manifest_sha256 = "{_text_attr(module, "prov.quantization_manifest_sha256")}", '
             f'profile_sha256 = "{profile_sha256(profile)}"')
    original_signature = (
        "      %bf16: tensor<64x64xbf16>, %b2: tensor<64x64xi8>,\n"
        "      %b2s: tensor<2x64xi8>)")
    full_signature = (
        "      %a1: tensor<64x64xi8>, %a1s: tensor<2x64xi8>,\n"
        "      %b1: tensor<64x64xi8>, %b1s: tensor<2x64xi8>,\n"
        "      %b2: tensor<64x64xi8>, %b2s: tensor<2x64xi8>)")
    first_ops = f'''    %acc = "mx_gemmini.contract"(%a1, %a1s, %b1, %b1s) {{
      site_id = "functional:matmul", activation_format = "fp8_e4m3",
      weight_format = "fp8_e4m3", activation_projection = "direct",
      weight_projection = "direct", pe_mode = 8 : i32, {attrs}}}
      : (tensor<64x64xi8>, tensor<2x64xi8>, tensor<64x64xi8>, tensor<2x64xi8>)
      -> tensor<64x64xbf16>
    %bf16 = "mx_gemmini.readout_bf16"(%acc) {{
      site_id = "functional:matmul", {attrs}}}
      : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
'''
    if (preloaded_mlir.count(original_signature) != 1 or
            preloaded_mlir.count('    %v0 = "mx_gemmini.vpu_execute"') != 1):
        raise ValueError("preloaded two-tile graph signature differs")
    full = preloaded_mlir.replace(original_signature, full_signature, 1)
    full = full.replace('    %v0 = "mx_gemmini.vpu_execute"',
                        first_ops + '    %v0 = "mx_gemmini.vpu_execute"', 1)
    old_digest = _text_attr(module, "mx.runtime_resources_sha256")
    full = full.replace(f'mx.runtime_resources_sha256 = "{old_digest}"',
                        f'mx.runtime_resources_sha256 = "{_digest(resources)}"', 1)
    full = full.replace('  mx.runtime_resources_sha256 = ',
                        '  mx.first_input_pair_sha256 = "' +
                        _sha(resources["a1_activation"] + resources["b1_weight"]) +
                        '",\n  mx.runtime_resources_sha256 = ', 1)
    return full


def render_full_chain_pipelined(frontend: str, trace: dict, manifest: dict,
                                profile: dict, resources: dict[str, bytes],
                                facts: dict) -> tuple[str, str]:
    """Render both source-preloaded and complete three-site typed graphs."""
    if (facts.get("profile_sha256") != profile_sha256(profile) or
            any(facts.get("resource_sha256", {}).get(name) != _sha(data)
                for name, data in resources.items()) or
            any(len(resources.get(name, b"")) != size for name, size in
                (("a1_activation", 4096), ("a1_scales", 128),
                 ("b1_weight", 4096), ("b1_scales", 128)))):
        raise ValueError("full two-tile MM1 resources differ from checked source")
    preloaded = render_chain_pipelined(frontend, trace, manifest, profile,
                                       {name: resources[name] for name in resources
                                        if name not in FULL_INPUTS[:4]}, facts)
    full = _inject_mm1(preloaded, profile, resources)
    report = verify_ir(full, profile)
    if (report["contracts"], report["vpu_commands"],
            report["spad_requants"], report["resident_contracts"]) != (1, 2, 2, 2):
        raise ValueError("full two-tile graph lost MM1 or a branch")
    lower_full_chain_pipelined(full, preloaded, profile, resources)
    return full, preloaded


def lower_full_chain_pipelined(full_mlir: str, preloaded_mlir: str,
                               profile: dict, resources: dict[str, bytes]
                               ) -> TwoTileChain:
    """Replace C1 preload with compiler-issued MM1 and preserve both branches."""
    from .chain_pipelined_graph import _module
    from xdsl.dialects.func import FuncOp, ReturnOp

    if full_mlir != _inject_mm1(preloaded_mlir, profile, resources):
        raise ValueError("full two-tile graph differs from captured MM1 binding")
    report = verify_ir(full_mlir, profile)
    if (report["contracts"], report["vpu_commands"],
            report["spad_requants"], report["resident_contracts"]) != (1, 2, 2, 2):
        raise ValueError("full two-tile graph operation counts differ")
    module = _module(full_mlir)
    funcs = [op for op in module.walk() if isinstance(op, FuncOp)]
    if len(funcs) != 1:
        raise ValueError("full two-tile chain needs one function")
    args = list(funcs[0].body.block.args)
    ops = list(funcs[0].body.block.ops)
    if (tuple(str(arg.type) for arg in args) !=
            ("tensor<64x64xi8>", "tensor<2x64xi8>",
             "tensor<64x64xi8>", "tensor<2x64xi8>",
             "tensor<64x64xi8>", "tensor<2x64xi8>") or
            [_operation_name(op) for op in ops] != [
                "mx_gemmini.contract", "mx_gemmini.readout_bf16",
                "mx_gemmini.vpu_execute", "mx_gemmini.spad_requant",
                "mx_gemmini.resident_contract", "mx_gemmini.vpu_execute",
                "mx_gemmini.spad_requant", "mx_gemmini.resident_contract",
                "func.return"] or
            list(ops[0].operands) != args[:4] or
            list(ops[1].operands) != list(ops[0].results) or
            list(ops[2].operands) != list(ops[1].results) or
            list(ops[5].operands) != list(ops[1].results) or
            list(ops[4].operands) != [*ops[3].results, *args[4:]] or
            list(ops[7].operands) != [*ops[6].results, *args[4:]] or
            not isinstance(ops[8], ReturnOp) or
            list(ops[8].operands) != [*ops[4].results, *ops[7].results]):
        raise ValueError("full two-tile MM1 or branch SSA edges differ")
    if (_text_attr(ops[0], "site_id") != "functional:matmul" or
            _text_attr(ops[1], "site_id") != "functional:matmul" or
            _text_attr(module, "mx.runtime_resources_sha256") != _digest(resources)):
        raise ValueError("full two-tile MM1 provenance or resources differ")
    baseline = lower_chain_pipelined(
        preloaded_mlir, profile,
        {name: resources[name] for name in resources if name not in FULL_INPUTS[:4]})
    first_buffers = {name: name for name in FULL_INPUTS[:4]}
    first_buffers.update({"c1_scales": "c1_scales_mm1",
                          "c1_bf16_observed": "c1_bf16_observed"})
    first = emit_verified_first_matrix_commands(
        profile, resources, buffers=first_buffers)
    commands = baseline.commands
    first_c1 = next(i for i, item in enumerate(commands)
                    if isinstance(item, Command) and item.funct == 2 and
                    item.rs1.buffer == "c1_bf16")
    preload_start = max(i for i in range(first_c1)
                        if isinstance(commands[i], Command) and
                        commands[i].funct == 0 and
                        commands[i].rs1.immediate == (16 << 16) | (1 << 8) | 1 and
                        commands[i].rs2.immediate == 16)
    first_vpu = next(i for i, item in enumerate(commands)
                     if isinstance(item, Command) and item.funct == 33)
    if (first_c1 - preload_start != 2 or
            sum(isinstance(item, Command) and item.rs1.buffer == "c1_bf16"
                for item in commands[preload_start:first_vpu]) != 32):
        raise ValueError("full two-tile C1 preload schedule differs")
    body: list[Command | Fence] = []
    for item in commands[first_vpu:]:
        if isinstance(item, Command) and item.rs1.buffer == "c1_bf16":
            body.append(Command(
                item.funct, Operand(buffer="c1_bf16_observed",
                                    byte_offset=item.rs1.byte_offset), item.rs2))
        else:
            body.append(item)
    if sum(isinstance(item, Command) and item.rs1.buffer == "c1_bf16_observed"
           for item in body) != 32:
        raise ValueError("full two-tile second BF16 branch did not reload MM1 result")
    return TwoTileChain(
        (*first, *commands[1:preload_start], *body), baseline.sites)
