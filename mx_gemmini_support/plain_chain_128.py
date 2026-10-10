"""Source-bound connected lowering for Nicolas's plain 128³ FP8 MX chain."""

from __future__ import annotations

import hashlib
import json

from .command_ir import Command, Fence, Operand
from .physical_program import _cmd, _config_ld, _config_st, _transfer
from .resident_lowering import lower_resident_contract
from .target_profile import profile_sha256
from .verify_profile_ir import _int_attr, _operation_name, _text_attr, verify_ir


INPUTS = ("a1_activation", "a1_scales", "b1_weight", "b1_scales",
          "b2_weight", "b2_scales")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resource_digest(resources: dict[str, bytes]) -> str:
    hashes = {name: _sha(resources[name]) for name in INPUTS}
    return _sha(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode())


def _validate_frontend(frontend_mlir: str, manifest: dict, profile: dict) -> None:
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser

    report = verify_ir(frontend_mlir, profile)
    if (report["contracts"], report["resident_contracts"],
            report["vpu_commands"], report["spad_requants"]) != (2, 0, 0, 0):
        raise ValueError("plain 128³ chain needs two frontend MX contractions")
    expected = [(site, "quantized", "mxfp8", [128, 128, 128]) for site in
                ("functional:matmul", "functional:matmul_1")]
    if [(row.get("site_id"), row.get("status"), row.get("format"), row.get("shape"))
            for row in manifest.get("sites", [])] != expected:
        raise ValueError("plain 128³ frontend sites differ from selected chain")
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, frontend_mlir).parse_module()
    manifest_digest = _sha(json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    if _text_attr(module, "prov.quantization_manifest_sha256") != manifest_digest:
        raise ValueError("plain 128³ frontend manifest differs from captured IR")
    contracts = [op for op in module.walk() if _operation_name(op) == "mx_gemmini.contract"]
    if ([_text_attr(op, "site_id") for op in contracts] != [row[0] for row in expected] or
            any((_text_attr(op, "activation_format"), _text_attr(op, "weight_format"),
                 _text_attr(op, "activation_projection"),
                 _text_attr(op, "weight_projection"), _int_attr(op, "pe_mode")) !=
                ("fp8_e4m3", "fp8_e4m3", "direct", "direct", 8)
                for op in contracts)):
        raise ValueError("plain 128³ frontend precision or site differs")


def render_plain_chain_128(frontend_mlir: str, manifest: dict, profile: dict,
                           resources: dict[str, bytes], *, source_sha256: str,
                           header_sha256: str) -> str:
    """Bind the two captured sites to Nicolas's packed operands and resident edge."""
    _validate_frontend(frontend_mlir, manifest, profile)
    _validate_resources(resources)
    digest = profile_sha256(profile)
    policy = _sha(b"nicolas_plain_fp8_128_resident_chain_v1")
    binding = (f'contract_sha256 = "{source_sha256}", policy_sha256 = "{policy}", '
               f'manifest_sha256 = "{header_sha256}", profile_sha256 = "{digest}"')
    text = f'''module attributes {{mx.profile_sha256 = "{digest}",
  mx.contract_sha256 = "{source_sha256}", mx.policy_sha256 = "{policy}",
  prov.quantization_manifest_sha256 = "{header_sha256}",
  mx.frontend_mlir_sha256 = "{_sha(frontend_mlir.encode())}",
  mx.runtime_resources_sha256 = "{_resource_digest(resources)}"}} {{
  func.func @nicolas_plain_chain_128(
      %a1: tensor<128x128xi8>, %a1s: tensor<4x128xi8>,
      %b1: tensor<128x128xi8>, %b1s: tensor<4x128xi8>,
      %b2: tensor<128x128xi8>, %b2s: tensor<4x128xi8>)
      -> (tensor<128x128xi8>, tensor<128x4xi8>) {{
    %acc = "mx_gemmini.contract"(%a1, %a1s, %b1, %b1s) {{
      site_id = "functional:matmul", activation_format = "fp8_e4m3",
      weight_format = "fp8_e4m3", activation_projection = "direct",
      weight_projection = "direct", pe_mode = 8 : i32, {binding}}}
      : (tensor<128x128xi8>, tensor<4x128xi8>, tensor<128x128xi8>, tensor<4x128xi8>)
      -> tensor<128x128xbf16>
    %c1, %c1s = "mx_gemmini.readout_quantized"(%acc) {{
      site_id = "functional:matmul", output_format = "fp8_e4m3", {binding}}}
      : (tensor<128x128xbf16>) -> (tensor<128x128xi8>, tensor<128x4xi8>)
    %c2, %c2s = "mx_gemmini.resident_contract"(%c1, %c1s, %b2, %b2s) {{
      site_id = "functional:matmul_1", activation_row = 2048 : i32,
      weight_row = 15360 : i32, output_row = 4096 : i32,
      m = 128 : i32, n = 128 : i32, k = 128 : i32,
      activation_format = "fp8_e4m3", weight_format = "fp8_e4m3",
      output_format = "fp8_e4m3", weight_buffer = "b2_weight",
      weight_scales_buffer = "b2_scales", output_scales_buffer = "c2_scales",
      {binding}}}
      : (tensor<128x128xi8>, tensor<128x4xi8>, tensor<128x128xi8>, tensor<4x128xi8>)
      -> (tensor<128x128xi8>, tensor<128x4xi8>)
    func.return %c2, %c2s : tensor<128x128xi8>, tensor<128x4xi8>
  }}
}}
'''
    lower_plain_chain_128(text, frontend_mlir, manifest, profile, resources,
                          source_sha256=source_sha256, header_sha256=header_sha256)
    return text


def _validate_resources(resources: dict[str, bytes]) -> None:
    expected = {name: 16384 if name.endswith(("activation", "weight")) else 512
                for name in INPUTS}
    if {name: len(resources.get(name, b"")) for name in INPUTS} != expected:
        raise ValueError("plain 128³ source operands or scales differ")


def lower_plain_chain_128(mlir_text: str, frontend_mlir: str, manifest: dict,
                          profile: dict, resources: dict[str, bytes], *,
                          source_sha256: str, header_sha256: str
                          ) -> tuple[Command | Fence, ...]:
    """Check typed edges and lower MM1's resident output followed by MM2."""
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func, FuncOp, ReturnOp
    from xdsl.parser import Parser

    _validate_frontend(frontend_mlir, manifest, profile)
    _validate_resources(resources)
    checked = verify_ir(mlir_text, profile)
    if (checked["contracts"], checked["resident_contracts"],
            checked["vpu_commands"], checked["spad_requants"]) != (1, 1, 0, 0):
        raise ValueError("plain 128³ chain needs one MM1 and one resident MM2")
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    if (_text_attr(module, "mx.frontend_mlir_sha256") != _sha(frontend_mlir.encode()) or
            _text_attr(module, "mx.runtime_resources_sha256") != _resource_digest(resources) or
            _text_attr(module, "mx.contract_sha256") != source_sha256 or
            _text_attr(module, "prov.quantization_manifest_sha256") != header_sha256):
        raise ValueError("plain 128³ chain frontend or source binding differs")
    functions = [op for op in module.walk() if isinstance(op, FuncOp)]
    if len(functions) != 1 or functions[0].sym_name.data != "nicolas_plain_chain_128":
        raise ValueError("plain 128³ chain needs one connected function")
    function = functions[0]
    ops = list(function.body.block.ops)
    if [_operation_name(op) for op in ops] != ["mx_gemmini.contract",
            "mx_gemmini.readout_quantized", "mx_gemmini.resident_contract",
            "func.return"]:
        raise ValueError("plain 128³ chain operation order differs")
    mm1, readout, mm2, ret = ops
    args = list(function.body.block.args)
    if (tuple(str(arg.type) for arg in args) !=
            ("tensor<128x128xi8>", "tensor<4x128xi8>") * 3 or
            list(mm1.operands) != args[:4] or
            list(readout.operands) != list(mm1.results) or
            list(mm2.operands) != [*readout.results, *args[4:]] or
            not isinstance(ret, ReturnOp) or
            list(ret.operands) != list(mm2.results) or
            tuple(str(value.type) for value in mm1.results) !=
            ("tensor<128x128xbf16>",) or
            tuple(str(value.type) for value in readout.results) !=
            ("tensor<128x128xi8>", "tensor<128x4xi8>") or
            tuple(str(value.type) for value in mm2.results) !=
            ("tensor<128x128xi8>", "tensor<128x4xi8>")):
        raise ValueError("plain 128³ chain SSA tensor edges differ")
    if ([_text_attr(op, "site_id") for op in ops[:-1]] !=
            ["functional:matmul", "functional:matmul", "functional:matmul_1"] or
            any(_text_attr(mm1, name) != value for name, value in {
                "activation_format": "fp8_e4m3", "weight_format": "fp8_e4m3",
                "activation_projection": "direct", "weight_projection": "direct"}.items()) or
            _int_attr(mm1, "pe_mode") != 8 or
            _text_attr(readout, "output_format") != "fp8_e4m3"):
        raise ValueError("plain 128³ chain selected sites or mode differ")
    attrs = {name: _int_attr(mm2, name) for name in
             ("activation_row", "weight_row", "output_row", "m", "n", "k")}
    attrs.update({name: _text_attr(mm2, name) for name in
                  ("activation_format", "weight_format", "output_format",
                   "weight_buffer", "weight_scales_buffer", "output_scales_buffer")})
    if ((attrs["activation_row"], attrs["weight_row"], attrs["output_row"],
         attrs["m"], attrs["n"], attrs["k"]) !=
            (2048, 15360, 4096, 128, 128, 128) or
            (attrs["weight_buffer"], attrs["weight_scales_buffer"],
             attrs["output_scales_buffer"]) !=
            ("b2_weight", "b2_scales", "c2_scales")):
        raise ValueError("plain 128³ chain resident placement differs")
    rows = profile["resources"]["scratchpad_bytes"] // 16
    if rows != 16384 or profile["name"] != "MxGemminiRocketConfig":
        raise ValueError("plain 128³ chain needs Nicolas's 256 KiB plain MX profile")
    commands: list[Command | Fence] = [
        _cmd(7, 0, 0),
        _cmd(0, (1 << 16) | (1 << 2), 1 << 48),
        _cmd(27, Operand(buffer="a1_scales"), 512),
        _cmd(27, Operand(buffer="b1_scales"), (1 << 32) | 512),
        Fence(), _config_ld(128),
    ]
    for i in range(8):
        for k in range(8):
            commands.append(_transfer(2, "a1_activation", i * 16 * 128 + k * 16,
                                      (i * 8 + k) * 16))
    for j in range(8):
        for k in range(8):
            commands.append(_transfer(2, "b1_weight", j * 16 * 128 + k * 16,
                                      15360 + (j * 8 + k) * 16))
    commands.extend([
        Fence(), _config_st(2),
        _cmd(26, Operand(buffer="c1_scales", address_mask=(1 << 33) - 1,
                         or_bits=(1 << 63) | (8 << 51) | (8 << 42) | (8 << 33)), 1),
        _cmd(9, 0, (8 << 32) | (8 << 16) | 8),
        _cmd(24, 0, rows),
        _cmd(8, 0, (2048 << 32) | 0x200 | 0x38 | (1 << 10)),
        Fence(),
        # C1 readback is diagnostic. MM2 consumes the live scratchpad tile.
        _config_st(16),
    ])
    for row in range(0, 1024, 16):
        commands.append(_transfer(3, "c1_tiled_observed", row * 16, 2048 + row))
    commands.append(Fence())
    commands.extend(lower_resident_contract(profile, attrs))
    commands.append(_config_st(16))
    for row in range(0, 1024, 16):
        commands.append(_transfer(3, "c2_tiled", row * 16, 4096 + row))
    commands.append(Fence())
    return tuple(commands)
