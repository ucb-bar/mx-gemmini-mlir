"""Bind captured two-site PyTorch graphs to Nicolas's FP8 MX/VPU chain."""

from __future__ import annotations

import hashlib
import json

from .resident_pair_graph import INPUTS, input_digest
from .resident_vpu_graph import OUTPUTS, lower_connected_fp8_vpu_pair
from .target_profile import profile_sha256
from .verify_profile_ir import _int_attr, _operation_name, _text_attr, verify_ir


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render_connected_fp8_vpu_chain(frontend: str, manifest: dict, profile: dict,
                                   resources: dict[str, bytes], facts: dict) -> str:
    """Check both captured sites and the six source inputs before making SSA edges."""
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser

    report = verify_ir(frontend, profile)
    if (report["contracts"], report["resident_contracts"],
            report["vpu_commands"], report["spad_requants"]) != (2, 0, 0, 0):
        raise ValueError("narrow MX/VPU frontend needs exactly two contractions")
    first_width = facts.get("first_width", 64)
    width = facts.get("second_width")
    if first_width not in (64, 128):
        raise ValueError("MX/VPU source needs a supported square first tile")
    if type(width) is not int or width < 32 or width % 32:
        raise ValueError("MX/VPU source needs a complete E8M0 second width")
    expected = [("functional:matmul", "quantized", "mxfp8",
                 [first_width, first_width, first_width]),
                ("functional:matmul_1", "quantized", "mxfp8",
                 [first_width, width, first_width])]
    if [(site.get("site_id"), site.get("status"), site.get("format"),
         site.get("shape")) for site in manifest.get("sites", [])] != expected:
        raise ValueError("narrow MX/VPU captured sites or shapes differ")
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, frontend).parse_module()
    manifest_hash = _sha(json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    if _text_attr(module, "prov.quantization_manifest_sha256") != manifest_hash:
        raise ValueError("narrow MX/VPU manifest differs from captured IR")
    contracts = [op for op in module.walk() if _operation_name(op) == "mx_gemmini.contract"]
    if ([_text_attr(op, "site_id") for op in contracts] != [row[0] for row in expected] or
            any((_text_attr(op, "activation_format"), _text_attr(op, "weight_format"),
                 _text_attr(op, "activation_projection"),
                 _text_attr(op, "weight_projection"), _int_attr(op, "pe_mode")) !=
                ("fp8_e4m3", "fp8_e4m3", "direct", "direct", 8)
                for op in contracts)):
        raise ValueError("narrow MX/VPU frontend precision differs")
    sizes = {"a1_activation": first_width ** 2,
             "a1_scales": first_width ** 2 // 32,
             "b1_weight": first_width ** 2,
             "b1_scales": first_width ** 2 // 32,
             "b2_weight": first_width * width,
             "b2_scales": first_width * width // 32}
    if ({slot: len(resources.get(slot, b"")) for slot in INPUTS} != sizes or
            facts.get("profile_sha256") != profile_sha256(profile) or
            any(facts.get("resource_sha256", {}).get(slot) != _sha(resources[slot])
                for slot in INPUTS)):
        raise ValueError("narrow MX/VPU source inputs or profile differ")
    digest = profile_sha256(profile)
    source_digest = _sha(json.dumps(
        facts, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    binding = (f'contract_sha256 = "{source_digest}", '
               f'policy_sha256 = "{facts["policy_sha256"]}", '
               f'manifest_sha256 = "{facts["header_sha256"]}", '
               f'profile_sha256 = "{digest}"')
    payload_digest = input_digest(resources, {name: name for name in INPUTS})
    b_row = profile["resources"]["scratchpad_bytes"] // 16 - first_width * width // 16
    c1_row, c2_row = ((128, 512) if first_width == 64 else (2048, 8192))
    function_name = ("nicolas_narrow_vpu_pair" if first_width == 64 and width == 32 else
                     "nicolas_connected_vpu_pair")
    text = f'''module attributes {{mx.profile_sha256 = "{digest}",
  mx.contract_sha256 = "{source_digest}",
  mx.policy_sha256 = "{facts["policy_sha256"]}",
  prov.quantization_manifest_sha256 = "{facts["header_sha256"]}",
  mx.frontend_mlir_sha256 = "{_sha(frontend.encode())}",
  mx.runtime_resources_sha256 = "{payload_digest}"}} {{
  func.func @{function_name}(
      %a1: tensor<{first_width}x{first_width}xi8>, %a1s: tensor<{first_width // 32}x{first_width}xi8>,
      %b1: tensor<{first_width}x{first_width}xi8>, %b1s: tensor<{first_width // 32}x{first_width}xi8>,
      %b2: tensor<{first_width}x{width}xi8>, %b2s: tensor<{first_width // 32}x{width}xi8>)
      -> (tensor<{first_width}x{width}xi8>, tensor<{first_width}x{width // 32}xi8>) {{
    %acc = "mx_gemmini.contract"(%a1, %a1s, %b1, %b1s) {{
      site_id = "functional:matmul", activation_format = "fp8_e4m3",
      weight_format = "fp8_e4m3", activation_projection = "direct",
      weight_projection = "direct", pe_mode = 8 : i32, {binding}}}
      : (tensor<{first_width}x{first_width}xi8>, tensor<{first_width // 32}x{first_width}xi8>, tensor<{first_width}x{first_width}xi8>, tensor<{first_width // 32}x{first_width}xi8>)
      -> tensor<{first_width}x{first_width}xbf16>
    %bf16 = "mx_gemmini.readout_bf16"(%acc) {{
      site_id = "functional:matmul", {binding}}}
      : (tensor<{first_width}x{first_width}xbf16>) -> tensor<{first_width}x{first_width}xbf16>
    %vpu = "mx_gemmini.vpu_execute"(%bf16) {{
      site_id = "functional:matmul", kind = "muls",
      src1_row = 4096 : i32, src2_row = 0 : i32,
      dst_row = 4096 : i32, rows = {first_width * first_width * 2 // 16} : i32,
      reduction_length = 1 : i32, broadcast = false,
      immediate_bf16 = 16384 : i32, {binding}}}
      : (tensor<{first_width}x{first_width}xbf16>) -> tensor<{first_width}x{first_width}xbf16>
    %c1, %c1s = "mx_gemmini.spad_requant"(%vpu) {{
      site_id = "functional:matmul", source_row = 4096 : i32,
      destination_row = {c1_row} : i32, m = {first_width} : i32, n = {first_width} : i32,
      output_format = "fp8_e4m3", tiled = true, resident = true,
      scale_dram_address = 0 : i64, scale_buffer = "c1_scales", {binding}}}
      : (tensor<{first_width}x{first_width}xbf16>) -> (tensor<{first_width}x{first_width}xi8>, tensor<{first_width}x{first_width // 32}xi8>)
    %c2, %c2s = "mx_gemmini.resident_contract"(%c1, %c1s, %b2, %b2s) {{
      site_id = "functional:matmul_1", activation_row = {c1_row} : i32,
      weight_row = {b_row} : i32, output_row = {c2_row} : i32,
      m = {first_width} : i32, n = {width} : i32, k = {first_width} : i32,
      activation_format = "fp8_e4m3", weight_format = "fp8_e4m3",
      output_format = "fp8_e4m3", weight_buffer = "b2_weight",
      weight_scales_buffer = "b2_scales", output_scales_buffer = "c2_scales",
      {binding}}}
      : (tensor<{first_width}x{first_width}xi8>, tensor<{first_width}x{first_width // 32}xi8>, tensor<{first_width}x{width}xi8>, tensor<{first_width // 32}x{width}xi8>)
      -> (tensor<{first_width}x{width}xi8>, tensor<{first_width}x{width // 32}xi8>)
    func.return %c2, %c2s : tensor<{first_width}x{width}xi8>, tensor<{first_width}x{width // 32}xi8>
  }}
}}
'''
    pair = lower_connected_fp8_vpu_pair(
        text, profile, resources, buffers={name: name for name in INPUTS},
        outputs={name: name for name in OUTPUTS})
    if pair.second_width != width:
        raise ValueError("connected MX/VPU graph lost its MM2 width")
    return text


def render_narrow_vpu_chain(frontend: str, manifest: dict, profile: dict,
                            resources: dict[str, bytes], facts: dict) -> str:
    """Keep the original checked 32-column source binding API."""
    if facts.get("second_width") != 32:
        raise ValueError("narrow MX/VPU source needs 32 MM2 columns")
    return render_connected_fp8_vpu_chain(frontend, manifest, profile, resources, facts)
