"""Bind Nicolas's packed FP4 64-cubed source to a connected resident pair."""

from __future__ import annotations

import hashlib
import json
import re

from .resident_pair_graph import INPUTS, input_digest, lower_connected_pair
from .resident_pair_plan import plan_fp4_resident_pair
from .source_fp6 import _array
from .target_profile import profile_sha256
from .verify_profile_ir import _operation_name, _text_attr, verify_ir


def source_resources(source: str, header: str) -> dict[str, bytes]:
    anchors = ('"include/matmul_fp4_64x64_chain.h"',
               "#define CHAIN_FLAGS (0x38 | LOOP_WS_REQUANT_TILED)",
               "SPAD_DEST1 = 2048", "SPAD_DEST2 = 4096",
               "gemmini_mxquant_config_mvout_resident")
    if any(anchor not in source for anchor in anchors) or source.count(
            "gemmini_loop_ws_spad(") != 2:
        raise ValueError("Nicolas FP4 resident source schedule differs")
    for macro, value in (("MATMUL_M", 64), ("MATMUL_N", 64),
                         ("MATMUL_K", 64), ("MATMUL_GK", 2),
                         ("MATMUL_GN", 2)):
        if not re.search(rf"^#define\s+{macro}\s+{value}\s*$", header, re.M):
            raise ValueError("Nicolas FP4 source header shape differs")

    def data(name: str, dimensions: str, count: int) -> bytes:
        return bytes(_array(header, name=name, ctype="uint8_t",
                            dimensions=dimensions, count=count, maximum=255))

    return {
        "a1_activation": data("A_in_hw", "[32][64]", 2048),
        "a1_scales": data("A_scales_row", "[2][64]", 128),
        "b1_weight": data("B_in", "[64][32]", 2048),
        "b1_scales": data("B_scales_col", "[2][64]", 128),
        "b2_weight": data("B2_in", "[64][32]", 2048),
        "b2_scales": data("B2_scales_col", "[2][64]", 128),
        "c1_codes_ref": data("C1_out", "[32][64]", 2048),
        "c1_scales_ref": data("C1_scales_out", "[64][2]", 128),
        "c2_codes_ref": data("C2_out", "[32][64]", 2048),
        "c2_scales_ref": data("C2_scales_out", "[64][2]", 128),
    }


def render_fp4_plain_chain(profile: dict, resources: dict[str, bytes], *,
                           source_sha256: str, header_sha256: str,
                           frontend_mlir: str | None = None,
                           frontend_manifest: dict | None = None) -> str:
    frontend_attrs = ""
    if frontend_mlir is not None or frontend_manifest is not None:
        _validate_frontend(frontend_mlir, frontend_manifest, profile)
        manifest_sha = hashlib.sha256(json.dumps(
            frontend_manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        frontend_attrs = (f'mx.frontend_mlir_sha256 = "{hashlib.sha256(frontend_mlir.encode()).hexdigest()}",\n'
                          f'  mx.frontend_manifest_sha256 = "{manifest_sha}",\n  ')
    plan = plan_fp4_resident_pair(profile, shape=(64, 64, 64),
                                  a_row=0, c1_row=2048, c2_row=4096)
    digest = profile_sha256(profile)
    policy = hashlib.sha256(b"nicolas_plain_fp4_64_resident_chain_v1").hexdigest()
    inputs_sha = input_digest(resources, {name: name for name in INPUTS})
    binding = (f'contract_sha256 = "{source_sha256}", policy_sha256 = "{policy}", '
               f'manifest_sha256 = "{header_sha256}", profile_sha256 = "{digest}"')
    text = f'''module attributes {{mx.profile_sha256 = "{digest}",
  mx.contract_sha256 = "{source_sha256}", mx.policy_sha256 = "{policy}",
  prov.quantization_manifest_sha256 = "{header_sha256}",
  {frontend_attrs}mx.runtime_resources_sha256 = "{inputs_sha}"}} {{
  func.func @nicolas_plain_fp4_chain(
      %a1: tensor<32x64xi8>, %a1s: tensor<2x64xi8>,
      %b1: tensor<64x32xi8>, %b1s: tensor<2x64xi8>,
      %b2: tensor<64x32xi8>, %b2s: tensor<2x64xi8>)
      -> (tensor<32x64xi8>, tensor<64x2xi8>) {{
    %acc = "mx_gemmini.contract"(%a1, %a1s, %b1, %b1s) {{
      site_id = "functional:matmul", activation_format = "fp4_e2m1",
      weight_format = "fp4_e2m1", activation_projection = "direct",
      weight_projection = "direct", pe_mode = 0 : i32, {binding}}}
      : (tensor<32x64xi8>, tensor<2x64xi8>, tensor<64x32xi8>, tensor<2x64xi8>)
      -> tensor<64x64xbf16>
    %c1, %c1s = "mx_gemmini.readout_quantized"(%acc) {{
      site_id = "functional:matmul", output_format = "fp4_e2m1", {binding}}}
      : (tensor<64x64xbf16>) -> (tensor<32x64xi8>, tensor<64x2xi8>)
    %c2, %c2s = "mx_gemmini.resident_contract"(%c1, %c1s, %b2, %b2s) {{
      site_id = "functional:matmul_1", activation_row = {plan.c1_row} : i32,
      weight_row = {plan.b_row} : i32, output_row = {plan.c2_row} : i32,
      m = 64 : i32, n = 64 : i32, k = 64 : i32,
      activation_format = "fp4_e2m1", weight_format = "fp4_e2m1",
      output_format = "fp4_e2m1", weight_buffer = "b2_weight",
      weight_scales_buffer = "b2_scales", output_scales_buffer = "c2_scales",
      {binding}}}
      : (tensor<32x64xi8>, tensor<64x2xi8>, tensor<64x32xi8>, tensor<2x64xi8>)
      -> (tensor<32x64xi8>, tensor<64x2xi8>)
    func.return %c2, %c2s : tensor<32x64xi8>, tensor<64x2xi8>
  }}
}}
'''
    lower_fp4_plain_chain(text, profile, resources, frontend_mlir=frontend_mlir,
                          frontend_manifest=frontend_manifest)
    return text


def lower_fp4_plain_chain(mlir_text: str, profile: dict,
                          resources: dict[str, bytes], *,
                          frontend_mlir: str | None = None,
                          frontend_manifest: dict | None = None):
    if frontend_mlir is not None or frontend_manifest is not None:
        _validate_frontend(frontend_mlir, frontend_manifest, profile)
        manifest_sha = hashlib.sha256(json.dumps(
            frontend_manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if (f'mx.frontend_mlir_sha256 = "{hashlib.sha256(frontend_mlir.encode()).hexdigest()}"'
                not in mlir_text or
                f'mx.frontend_manifest_sha256 = "{manifest_sha}"' not in mlir_text):
            raise ValueError("FP4 resident pair frontend binding differs")
    return lower_connected_pair(
        mlir_text, profile, resources,
        buffers={name: name for name in INPUTS},
        c1_scales="c1_scales", c1_tiled_observed="c1_tiled_observed",
        c2_tiled="c2_tiled", precision="fp4_e2m1").commands


def _validate_frontend(frontend_mlir: str | None, manifest: dict | None,
                       profile: dict) -> None:
    if not isinstance(frontend_mlir, str) or not isinstance(manifest, dict):
        raise ValueError("FP4 resident pair needs both frontend IR and manifest")
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser

    if verify_ir(frontend_mlir, profile)["contracts"] != 2:
        raise ValueError("FP4 resident frontend needs two MX contraction sites")
    sites = manifest.get("sites", [])
    expected = [("functional:matmul", "quantized", "mxfp4", [64, 64, 64]),
                ("functional:matmul_1", "quantized", "mxfp4", [64, 64, 64])]
    if [(site.get("site_id"), site.get("status"), site.get("format"),
         site.get("shape")) for site in sites] != expected:
        raise ValueError("FP4 resident frontend site manifest differs")
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, frontend_mlir).parse_module()
    contracts = [op for op in module.walk()
                 if _operation_name(op) == "mx_gemmini.contract"]
    if [_text_attr(op, "site_id") for op in contracts] != [item[0] for item in expected]:
        raise ValueError("FP4 resident frontend contraction sites differ")
