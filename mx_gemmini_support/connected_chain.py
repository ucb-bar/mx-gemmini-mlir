"""One SSA-connected MLIR chain for Nicolas's qualified MM1/VPU/MM2 mode.

The source audit supplies the packed operands and goldens. The model2MLIR
two-site capture supplies the contraction sites. This module combines their
checked facts into one executable dataflow and reuses the existing physical
command emitters. Other modes remain fail-closed.
"""

from __future__ import annotations

import hashlib
import json

from .command_ir import Command, Fence, spad_requant_command, vpu_command
from .first_matrix_lowering import (emit_verified_first_matrix_commands,
                                    lower_first_matrix_commands)
from .resident_lowering import (lower_resident_chain_commands,
                                lower_resident_contract)
from .target_profile import profile_sha256
from .verify_profile_ir import (_bool_attr, _int_attr, _operation_name,
                                _text_attr, verify_ir)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _facts_digest(facts: dict) -> str:
    return _sha(json.dumps(facts, sort_keys=True, separators=(",", ":")).encode())


def render_connected_chain(frontend_mlir: str, seam_mlir: str, profile: dict,
                           resources: dict[str, bytes], facts: dict) -> str:
    """Join two captured contraction sites and the audited resident seam."""
    lower_first_matrix_commands(frontend_mlir, profile, resources)
    lower_resident_chain_commands(
        seam_mlir, profile,
        expected_sites=("functional:matmul", "functional:matmul_1"))
    if (facts.get("typed_mlir_sha256") != _sha(seam_mlir.encode()) or
            facts.get("profile_sha256") != profile_sha256(profile)):
        raise ValueError("connected MX chain source seam or profile digest differs")
    selected = ("a1_activation", "a1_scales", "b1_weight", "b1_scales",
                "b2_weight", "b2_scales")
    if any(_sha(resources.get(name, b"")) != facts["resource_sha256"].get(name)
           for name in selected):
        raise ValueError("connected MX chain operands differ from audited source")
    digest = facts["contract_sha256"]
    policy = facts["policy_sha256"]
    manifest = facts["header_sha256"]
    profile_digest = facts["profile_sha256"]
    binding = (f'contract_sha256 = "{digest}", policy_sha256 = "{policy}", '
               f'manifest_sha256 = "{manifest}", profile_sha256 = "{profile_digest}"')
    resource_digest = _sha(json.dumps(
        {name: facts["resource_sha256"][name] for name in selected},
        sort_keys=True, separators=(",", ":")).encode())
    sp_bf16, sp_c1, two = facts["sp_bf16"], facts["sp_c1"], facts["bf16_scalar"]
    b_row = profile["resources"]["scratchpad_bytes"] // 16 - 256
    mlir = f'''module attributes {{mx.contract_sha256 = "{digest}",
  mx.policy_sha256 = "{policy}", prov.quantization_manifest_sha256 = "{manifest}",
  mx.profile_sha256 = "{profile_digest}",
  mx.frontend_mlir_sha256 = "{_sha(frontend_mlir.encode())}",
  mx.seam_mlir_sha256 = "{_sha(seam_mlir.encode())}",
  mx.source_facts_sha256 = "{_facts_digest(facts)}",
  mx.runtime_resources_sha256 = "{resource_digest}"}} {{
  func.func @nicolas_connected_chain(
      %a1: tensor<64x64xi8>, %a1s: tensor<2x64xi8>,
      %b1: tensor<64x64xi8>, %b1s: tensor<2x64xi8>,
      %b2: tensor<64x64xi8>, %b2s: tensor<2x64xi8>)
      -> (tensor<64x64xi8>, tensor<64x2xi8>) {{
    %acc = "mx_gemmini.contract"(%a1, %a1s, %b1, %b1s) {{
      site_id = "functional:matmul", activation_format = "fp8_e4m3",
      weight_format = "fp8_e4m3", activation_projection = "direct",
      weight_projection = "direct", pe_mode = 8 : i32, {binding}}}
      : (tensor<64x64xi8>, tensor<2x64xi8>, tensor<64x64xi8>, tensor<2x64xi8>)
      -> tensor<64x64xbf16>
    %bf16 = "mx_gemmini.readout_bf16"(%acc) {{
      site_id = "functional:matmul", {binding}}}
      : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %vpu = "mx_gemmini.vpu_execute"(%bf16) {{
      site_id = "functional:matmul", kind = "muls",
      src1_row = {sp_bf16} : i32, src2_row = 0 : i32,
      dst_row = {sp_bf16} : i32, rows = 512 : i32,
      reduction_length = 1 : i32, broadcast = false,
      immediate_bf16 = {two} : i32, {binding}}}
      : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %c1, %c1s = "mx_gemmini.spad_requant"(%vpu) {{
      site_id = "functional:matmul", source_row = {sp_bf16} : i32,
      destination_row = {sp_c1} : i32, m = 64 : i32, n = 64 : i32,
      output_format = "fp8_e4m3", tiled = true, resident = true,
      scale_dram_address = 0 : i64, scale_buffer = "c1_scales", {binding}}}
      : (tensor<64x64xbf16>) -> (tensor<64x64xi8>, tensor<64x2xi8>)
    %c2, %c2s = "mx_gemmini.resident_contract"(%c1, %c1s, %b2, %b2s) {{
      site_id = "functional:matmul_1", activation_row = {sp_c1} : i32,
      weight_row = {b_row} : i32, output_row = 512 : i32,
      m = 64 : i32, n = 64 : i32, k = 64 : i32,
      activation_format = "fp8_e4m3", weight_format = "fp8_e4m3",
      output_format = "fp8_e4m3", weight_buffer = "b2_weight",
      weight_scales_buffer = "b2_scales", output_scales_buffer = "c2_scales",
      {binding}}}
      : (tensor<64x64xi8>, tensor<64x2xi8>, tensor<64x64xi8>, tensor<2x64xi8>)
      -> (tensor<64x64xi8>, tensor<64x2xi8>)
    func.return %c2, %c2s : tensor<64x64xi8>, tensor<64x2xi8>
  }}
}}
'''
    lower_connected_chain_commands(mlir, profile, resources, facts,
                                   frontend_mlir=frontend_mlir,
                                   seam_mlir=seam_mlir)
    return mlir


def lower_connected_chain_commands(mlir_text: str, profile: dict,
                                   resources: dict[str, bytes], facts: dict, *,
                                   frontend_mlir: str, seam_mlir: str
                                   ) -> tuple[Command | Fence, ...]:
    """Verify the SSA edges, then lower the qualified resident command path."""
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func, FuncOp, ReturnOp
    from xdsl.parser import Parser

    report = verify_ir(mlir_text, profile)
    if (report["contracts"], report["vpu_commands"], report["spad_requants"],
            report["resident_contracts"]) != (1, 1, 1, 1):
        raise ValueError("connected MX chain requires MM1, VPU, requant, and MM2")
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    if (_text_attr(module, "mx.profile_sha256") != profile_sha256(profile) or
            _text_attr(module, "mx.source_facts_sha256") != _facts_digest(facts) or
            _text_attr(module, "mx.frontend_mlir_sha256") !=
            _sha(frontend_mlir.encode()) or
            _text_attr(module, "mx.seam_mlir_sha256") != _sha(seam_mlir.encode())):
        raise ValueError("connected MX chain source facts or profile differ")
    if (facts.get("typed_mlir_sha256") != _sha(seam_mlir.encode()) or
            facts.get("profile_sha256") != profile_sha256(profile)):
        raise ValueError("connected MX chain seam source or profile differs")
    lower_first_matrix_commands(frontend_mlir, profile, resources)
    source_body = lower_resident_chain_commands(
        seam_mlir, profile,
        expected_sites=("functional:matmul", "functional:matmul_1"))
    selected = ("a1_activation", "a1_scales", "b1_weight", "b1_scales",
                "b2_weight", "b2_scales")
    hashes = {name: _sha(resources.get(name, b"")) for name in selected}
    if (any(hashes[name] != facts["resource_sha256"].get(name) for name in selected) or
            _text_attr(module, "mx.runtime_resources_sha256") != _sha(json.dumps(
                hashes, sort_keys=True, separators=(",", ":")).encode())):
        raise ValueError("connected MX chain source operands differ")
    functions = [op for op in module.walk() if isinstance(op, FuncOp)]
    if len(functions) != 1 or functions[0].sym_name.data != "nicolas_connected_chain":
        raise ValueError("connected MX chain needs one checked function")
    function = functions[0]
    arguments = list(function.body.block.args)
    expected_args = ("tensor<64x64xi8>", "tensor<2x64xi8>",
                     "tensor<64x64xi8>", "tensor<2x64xi8>",
                     "tensor<64x64xi8>", "tensor<2x64xi8>")
    if tuple(str(value.type) for value in arguments) != expected_args:
        raise ValueError("connected MX chain requires six operand resources")
    if [_operation_name(op) for op in function.body.block.ops] != [
            "mx_gemmini.contract", "mx_gemmini.readout_bf16",
            "mx_gemmini.vpu_execute", "mx_gemmini.spad_requant",
            "mx_gemmini.resident_contract", "func.return"]:
        raise ValueError("connected MX chain function contains an unexpected operation")
    ops = [op for op in function.walk() if _operation_name(op).startswith("mx_gemmini.")]
    if [_operation_name(op) for op in ops] != [
            "mx_gemmini.contract", "mx_gemmini.readout_bf16",
            "mx_gemmini.vpu_execute", "mx_gemmini.spad_requant",
            "mx_gemmini.resident_contract"]:
        raise ValueError("connected MX chain operation order differs")
    first, readout, vpu, requant, second = ops
    expected_results = (("tensor<64x64xbf16>",),
                        ("tensor<64x64xbf16>",),
                        ("tensor<64x64xbf16>",),
                        ("tensor<64x64xi8>", "tensor<64x2xi8>"),
                        ("tensor<64x64xi8>", "tensor<64x2xi8>"))
    if tuple(tuple(str(value.type) for value in op.results) for op in ops) != expected_results:
        raise ValueError("connected MX chain typed results differ from qualified mode")
    ret = function.get_return_op()
    if (not isinstance(ret, ReturnOp) or
            list(first.operands) != arguments[:4] or
            list(readout.operands) != list(first.results) or
            list(vpu.operands) != list(readout.results) or
            list(requant.operands) != list(vpu.results) or
            list(second.operands) != [*requant.results, *arguments[4:]] or
            list(ret.operands) != list(second.results)):
        raise ValueError("connected MX chain SSA handoff differs")
    if ([_text_attr(op, "site_id") for op in ops] != [
            "functional:matmul", "functional:matmul",
            "functional:matmul", "functional:matmul",
            "functional:matmul_1"] or
            _text_attr(first, "activation_format") != "fp8_e4m3" or
            _text_attr(first, "weight_format") != "fp8_e4m3" or
            _text_attr(first, "activation_projection") != "direct" or
            _text_attr(first, "weight_projection") != "direct" or
            _int_attr(first, "pe_mode") != 8):
        raise ValueError("connected MX chain sites or first precision differ")
    if (_int_attr(vpu, "src1_row") != facts["sp_bf16"] or
            _int_attr(vpu, "dst_row") != facts["sp_bf16"] or
            _int_attr(vpu, "immediate_bf16") != facts["bf16_scalar"] or
            _int_attr(requant, "source_row") != facts["sp_bf16"] or
            _int_attr(requant, "destination_row") != facts["sp_c1"] or
            _int_attr(second, "activation_row") != facts["sp_c1"] or
            not _bool_attr(requant, "tiled") or not _bool_attr(requant, "resident")):
        raise ValueError("connected MX chain source placement differs")
    vector = vpu_command(
        profile, kind=_text_attr(vpu, "kind"), src1_row=_int_attr(vpu, "src1_row"),
        src2_row=_int_attr(vpu, "src2_row"), dst_row=_int_attr(vpu, "dst_row"),
        rows=_int_attr(vpu, "rows"), reduction_length=_int_attr(vpu, "reduction_length"),
        broadcast=_bool_attr(vpu, "broadcast"),
        immediate_bf16=_int_attr(vpu, "immediate_bf16"),
        second_dst_row=_int_attr(vpu, "second_dst_row"))
    quant = spad_requant_command(
        profile, source_row=_int_attr(requant, "source_row"),
        destination_row=_int_attr(requant, "destination_row"),
        m=_int_attr(requant, "m"), n=_int_attr(requant, "n"),
        output_format=_text_attr(requant, "output_format"),
        tiled=_bool_attr(requant, "tiled"), resident=_bool_attr(requant, "resident"),
        scale_dram_address=_int_attr(requant, "scale_dram_address"),
        scale_buffer=_text_attr(requant, "scale_buffer"))
    attrs = {key: _int_attr(second, key) for key in
             ("activation_row", "weight_row", "output_row", "m", "n", "k")}
    attrs.update({key: _text_attr(second, key) for key in
                  ("activation_format", "weight_format", "output_format",
                   "weight_buffer", "weight_scales_buffer", "output_scales_buffer")})
    connected_body = (vector, Fence(), quant, Fence(),
                      *lower_resident_contract(profile, attrs))
    if connected_body != source_body:
        raise ValueError("connected MX chain target commands differ from source seam")
    first_commands = emit_verified_first_matrix_commands(profile, resources)
    return (*first_commands, Fence(), *connected_body)
