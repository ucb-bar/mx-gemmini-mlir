"""Bind a captured rectangular PyTorch pair to Nicolas's checked FP8 bytes."""

from __future__ import annotations

import hashlib
import json

from .rectangular_source import FIRST_SHAPE, SECOND_SHAPE
from .resident_pair_graph import INPUTS, input_digest, lower_connected_fp8_pair
from .resident_pair_plan import plan_fp8_rectangular_pair
from .target_profile import profile_sha256
from .verify_profile_ir import _int_attr, _operation_name, _text_attr, verify_ir


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render_rectangular_chain(frontend: str, manifest: dict, profile: dict,
                             resources: dict[str, bytes], *,
                             source_sha256: str, header_sha256: str,
                             b2_header_sha256: str) -> str:
    """Require the two captured sites before making the resident SSA edge."""
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser

    report = verify_ir(frontend, profile)
    if (report["contracts"], report["resident_contracts"],
            report["vpu_commands"], report["spad_requants"]) != (2, 0, 0, 0):
        raise ValueError("rectangular frontend needs exactly two MX contractions")
    expected = [(site, "quantized", "mxfp8", list(shape)) for site, shape in zip(
        ("functional:matmul", "functional:matmul_1"),
        (FIRST_SHAPE, SECOND_SHAPE))]
    if [(site.get("site_id"), site.get("status"), site.get("format"),
         site.get("shape")) for site in manifest.get("sites", [])] != expected:
        raise ValueError("rectangular frontend sites or dimensions differ")
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, frontend).parse_module()
    manifest_hash = _sha(json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    if _text_attr(module, "prov.quantization_manifest_sha256") != manifest_hash:
        raise ValueError("rectangular frontend manifest differs from captured IR")
    contracts = [op for op in module.walk() if _operation_name(op) == "mx_gemmini.contract"]
    if ([_text_attr(op, "site_id") for op in contracts] != [row[0] for row in expected] or
            any((_text_attr(op, "activation_format"), _text_attr(op, "weight_format"),
                 _text_attr(op, "activation_projection"),
                 _text_attr(op, "weight_projection"), _int_attr(op, "pe_mode")) !=
                ("fp8_e4m3", "fp8_e4m3", "direct", "direct", 8)
                for op in contracts)):
        raise ValueError("rectangular frontend precision or site differs")
    sizes = {"a1_activation": 64 * 64, "a1_scales": 2 * 64,
             "b1_weight": 64 * 96, "b1_scales": 2 * 96,
             "b2_weight": 96 * 64, "b2_scales": 3 * 64}
    if {slot: len(resources.get(slot, b"")) for slot in INPUTS} != sizes:
        raise ValueError("rectangular pair source wire dimensions differ")
    plan = plan_fp8_rectangular_pair(
        profile, first_shape=FIRST_SHAPE, second_shape=SECOND_SHAPE,
        a_row=0, c1_row=2048, c2_row=4096)
    digest = profile_sha256(profile)
    policy = _sha(b"nicolas_rectangular_fp8_pair_v1" +
                  bytes.fromhex(b2_header_sha256))
    binding = (f'contract_sha256 = "{source_sha256}", policy_sha256 = "{policy}", '
               f'manifest_sha256 = "{header_sha256}", profile_sha256 = "{digest}"')
    payload_digest = input_digest(resources, {name: name for name in INPUTS})
    text = f'''module attributes {{mx.profile_sha256 = "{digest}",
  mx.contract_sha256 = "{source_sha256}", mx.policy_sha256 = "{policy}",
  prov.quantization_manifest_sha256 = "{header_sha256}",
  mx.b2_source_sha256 = "{b2_header_sha256}",
  mx.frontend_mlir_sha256 = "{_sha(frontend.encode())}",
  mx.runtime_resources_sha256 = "{payload_digest}"}} {{
  func.func @nicolas_rectangular_pair(
      %a1: tensor<64x64xi8>, %a1s: tensor<2x64xi8>,
      %b1: tensor<64x96xi8>, %b1s: tensor<2x96xi8>,
      %b2: tensor<96x64xi8>, %b2s: tensor<3x64xi8>)
      -> (tensor<64x64xi8>, tensor<64x2xi8>) {{
    %acc = "mx_gemmini.contract"(%a1, %a1s, %b1, %b1s) {{
      site_id = "functional:matmul", activation_format = "fp8_e4m3",
      weight_format = "fp8_e4m3", activation_projection = "direct",
      weight_projection = "direct", pe_mode = 8 : i32, {binding}}}
      : (tensor<64x64xi8>, tensor<2x64xi8>, tensor<64x96xi8>, tensor<2x96xi8>)
      -> tensor<64x96xbf16>
    %c1, %c1s = "mx_gemmini.readout_quantized"(%acc) {{
      site_id = "functional:matmul", output_format = "fp8_e4m3", {binding}}}
      : (tensor<64x96xbf16>) -> (tensor<64x96xi8>, tensor<64x3xi8>)
    %c2, %c2s = "mx_gemmini.resident_contract"(%c1, %c1s, %b2, %b2s) {{
      site_id = "functional:matmul_1", activation_row = {plan.c1_row} : i32,
      weight_row = {plan.b_row} : i32, output_row = {plan.c2_row} : i32,
      m = 64 : i32, n = 64 : i32, k = 96 : i32,
      activation_format = "fp8_e4m3", weight_format = "fp8_e4m3",
      output_format = "fp8_e4m3", weight_buffer = "b2_weight",
      weight_scales_buffer = "b2_scales", output_scales_buffer = "c2_scales",
      {binding}}}
      : (tensor<64x96xi8>, tensor<64x3xi8>, tensor<96x64xi8>, tensor<3x64xi8>)
      -> (tensor<64x64xi8>, tensor<64x2xi8>)
    func.return %c2, %c2s : tensor<64x64xi8>, tensor<64x2xi8>
  }}
}}
'''
    pair = lower_connected_fp8_pair(
        text, profile, resources, buffers={name: name for name in INPUTS},
        c1_scales="c1_scales", c1_tiled_observed="c1_tiled_observed",
        c2_tiled="c2_tiled")
    if pair.plan != plan:
        raise ValueError("rectangular pair graph differs from source plan")
    return text
