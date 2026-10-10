"""Bind Nicolas's plain FP8 chain and its checked 96-wide derived fixture."""

from __future__ import annotations

import hashlib
import json

from .command_ir import Command, Fence
from .resident_pair_graph import INPUTS, input_digest, lower_connected_fp8_pair
from .resident_pair_plan import plan_fp8_resident_pair
from .target_profile import profile_sha256
from .verify_profile_ir import _int_attr, _operation_name, _text_attr, verify_ir


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resource_digest(resources: dict[str, bytes]) -> str:
    return input_digest(resources, {name: name for name in INPUTS})


def _validate_frontend(frontend_mlir: str, manifest: dict, profile: dict,
                       width: int = 128) -> int:
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser

    report = verify_ir(frontend_mlir, profile)
    if (report["contracts"], report["resident_contracts"],
            report["vpu_commands"], report["spad_requants"]) != (2, 0, 0, 0):
        raise ValueError("plain chain needs two frontend MX contractions")
    sites = manifest.get("sites", [])
    m = sites[0].get("shape", [None])[0] if sites else None
    if type(m) is not int or m not in range(16, 129, 16):
        raise ValueError("plain chain needs a source-qualified row count")
    if width not in (96, 128):
        raise ValueError("plain chain width lacks a checked source fixture")
    expected = [(site, "quantized", "mxfp8", [m, width, width]) for site in
                ("functional:matmul", "functional:matmul_1")]
    if [(row.get("site_id"), row.get("status"), row.get("format"), row.get("shape"))
            for row in sites] != expected:
        raise ValueError("plain frontend sites differ from selected chain")
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, frontend_mlir).parse_module()
    manifest_digest = _sha(json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    if _text_attr(module, "prov.quantization_manifest_sha256") != manifest_digest:
        raise ValueError("plain frontend manifest differs from captured IR")
    contracts = [op for op in module.walk() if _operation_name(op) == "mx_gemmini.contract"]
    if ([_text_attr(op, "site_id") for op in contracts] != [row[0] for row in expected] or
            any((_text_attr(op, "activation_format"), _text_attr(op, "weight_format"),
                 _text_attr(op, "activation_projection"),
                 _text_attr(op, "weight_projection"), _int_attr(op, "pe_mode")) !=
                ("fp8_e4m3", "fp8_e4m3", "direct", "direct", 8)
                for op in contracts)):
        raise ValueError("plain frontend precision or site differs")
    return m


def render_plain_chain(frontend_mlir: str, manifest: dict, profile: dict,
                       resources: dict[str, bytes], *, source_sha256: str,
                       header_sha256: str, width: int = 128) -> str:
    """Bind two captured sites to checked wire data and a resident edge."""
    m = _validate_frontend(frontend_mlir, manifest, profile, width)
    _validate_resources(resources, m, width)
    plan = plan_fp8_resident_pair(
        profile, shape=(m, width, width), a_row=0, c1_row=2048, c2_row=4096)
    digest = profile_sha256(profile)
    policy = _sha(f"nicolas_plain_fp8_{width}_resident_chain_v1".encode())
    binding = (f'contract_sha256 = "{source_sha256}", policy_sha256 = "{policy}", '
               f'manifest_sha256 = "{header_sha256}", profile_sha256 = "{digest}"')
    text = f'''module attributes {{mx.profile_sha256 = "{digest}",
  mx.contract_sha256 = "{source_sha256}", mx.policy_sha256 = "{policy}",
  prov.quantization_manifest_sha256 = "{header_sha256}",
  mx.frontend_mlir_sha256 = "{_sha(frontend_mlir.encode())}",
  mx.runtime_resources_sha256 = "{_resource_digest(resources)}"}} {{
  func.func @nicolas_plain_chain_{width}(
      %a1: tensor<{m}x{width}xi8>, %a1s: tensor<{width // 32}x{m}xi8>,
      %b1: tensor<{width}x{width}xi8>, %b1s: tensor<{width // 32}x{width}xi8>,
      %b2: tensor<{width}x{width}xi8>, %b2s: tensor<{width // 32}x{width}xi8>)
      -> (tensor<{m}x{width}xi8>, tensor<{m}x{width // 32}xi8>) {{
    %acc = "mx_gemmini.contract"(%a1, %a1s, %b1, %b1s) {{
      site_id = "functional:matmul", activation_format = "fp8_e4m3",
      weight_format = "fp8_e4m3", activation_projection = "direct",
      weight_projection = "direct", pe_mode = 8 : i32, {binding}}}
      : (tensor<{m}x{width}xi8>, tensor<{width // 32}x{m}xi8>, tensor<{width}x{width}xi8>, tensor<{width // 32}x{width}xi8>)
      -> tensor<{m}x{width}xbf16>
    %c1, %c1s = "mx_gemmini.readout_quantized"(%acc) {{
      site_id = "functional:matmul", output_format = "fp8_e4m3", {binding}}}
      : (tensor<{m}x{width}xbf16>) -> (tensor<{m}x{width}xi8>, tensor<{m}x{width // 32}xi8>)
    %c2, %c2s = "mx_gemmini.resident_contract"(%c1, %c1s, %b2, %b2s) {{
      site_id = "functional:matmul_1", activation_row = {plan.c1_row} : i32,
      weight_row = {plan.b_row} : i32, output_row = {plan.c2_row} : i32,
      m = {m} : i32, n = {width} : i32, k = {width} : i32,
      activation_format = "fp8_e4m3", weight_format = "fp8_e4m3",
      output_format = "fp8_e4m3", weight_buffer = "b2_weight",
      weight_scales_buffer = "b2_scales", output_scales_buffer = "c2_scales",
      {binding}}}
      : (tensor<{m}x{width}xi8>, tensor<{m}x{width // 32}xi8>, tensor<{width}x{width}xi8>, tensor<{width // 32}x{width}xi8>)
      -> (tensor<{m}x{width}xi8>, tensor<{m}x{width // 32}xi8>)
    func.return %c2, %c2s : tensor<{m}x{width}xi8>, tensor<{m}x{width // 32}xi8>
  }}
}}
'''
    lower_plain_chain(text, frontend_mlir, manifest, profile, resources,
                      source_sha256=source_sha256, header_sha256=header_sha256,
                      width=width)
    return text


def _validate_resources(resources: dict[str, bytes], m: int, width: int = 128) -> None:
    expected = {name: (m * width if name == "a1_activation" else
                       m * width // 32 if name == "a1_scales" else
                       width * width if name.endswith("weight") else width * width // 32)
                for name in INPUTS}
    if {name: len(resources.get(name, b"")) for name in INPUTS} != expected:
        raise ValueError("plain chain wire operands or scales differ")


def lower_plain_chain(mlir_text: str, frontend_mlir: str, manifest: dict,
                      profile: dict, resources: dict[str, bytes], *,
                      source_sha256: str, header_sha256: str, width: int = 128
                      ) -> tuple[Command | Fence, ...]:
    """Bind source or source-derived bytes to the typed resident-pair lowerer."""
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func, FuncOp
    from xdsl.parser import Parser

    m = _validate_frontend(frontend_mlir, manifest, profile, width)
    _validate_resources(resources, m, width)
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    if (_text_attr(module, "mx.frontend_mlir_sha256") != _sha(frontend_mlir.encode()) or
            _text_attr(module, "mx.runtime_resources_sha256") != _resource_digest(resources) or
            _text_attr(module, "mx.contract_sha256") != source_sha256 or
            _text_attr(module, "prov.quantization_manifest_sha256") != header_sha256):
        raise ValueError("plain chain frontend or source binding differs")
    functions = [op for op in module.walk() if isinstance(op, FuncOp)]
    if len(functions) != 1 or functions[0].sym_name.data != f"nicolas_plain_chain_{width}":
        raise ValueError("plain chain needs one connected function")
    pair = lower_connected_fp8_pair(
        mlir_text, profile, resources, buffers={name: name for name in INPUTS},
        c1_scales="c1_scales", c1_tiled_observed="c1_tiled_observed",
        c2_tiled="c2_tiled")
    plan = pair.plan
    mm2 = list(functions[0].body.block.ops)[2]
    if (pair.first_site, pair.second_site) != (
            "functional:matmul", "functional:matmul_1"):
        raise ValueError("plain chain selected sites differ")
    if (plan.m, plan.n, plan.k) != (m, width, width):
        raise ValueError("plain chain source specialization needs its checked shape")
    if ((plan.c1_row, plan.c2_row) != (2048, 4096) or
            _text_attr(mm2, "output_scales_buffer") != "c2_scales"):
        raise ValueError("plain chain resident placement differs")
    if plan.rows != 16384 or profile["name"] != "MxGemminiRocketConfig":
        raise ValueError("plain chain needs Nicolas's 256 KiB plain MX profile")
    return pair.commands


# Preserve the published 128-wide Python API while the source fixture grows.
render_plain_chain_128 = render_plain_chain
lower_plain_chain_128 = lower_plain_chain
