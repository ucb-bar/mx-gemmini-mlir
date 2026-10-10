"""Check named MX MLIR operations against a source-bound target profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

from .command_ir import spad_requant_command, vpu_command
from .target_profile import load_profile, profile_sha256, require_compute


def _text_attr(op, name: str) -> str | None:
    value = op.attributes.get(name)
    return getattr(value, "data", None)


def _int_attr(op, name: str) -> int | None:
    value = op.attributes.get(name)
    if value is None:
        return None
    number = getattr(value, "value", None)
    number = getattr(number, "data", number)
    if type(number) is not int:
        raise ValueError(f"{op.name}: {name} must be an integer")
    return number


def _operation_name(op) -> str:
    # xDSL retains the parsed name of an unregistered dialect op here.
    return _text_attr(op, "op_name__") or op.name


def _bool_attr(op, name: str) -> bool:
    value = op.attributes.get(name)
    data = getattr(value, "data", None)
    if type(data) is bool:
        return data
    # xDSL parses MLIR's `true`/`false` as signed i1: true is -1.
    width = getattr(getattr(getattr(value, "type", None), "width", None), "data", None)
    number = getattr(getattr(value, "value", None), "data", None)
    if width == 1 and number in (-1, 0):
        return number == -1
    raise ValueError(f"{_operation_name(op)}: {name} must be Boolean")


def verify_ir(mlir_text: str, profile: dict) -> dict:
    from .resource_ir import INPUTS, LUTS, SCHEMA as RESOURCE_SCHEMA
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin, IntegerType, TensorType
    from xdsl.dialects.func import Func
    from xdsl.ir import Operation
    from xdsl.parser import Parser

    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    digest = profile_sha256(profile)
    if _text_attr(module, "mx.profile_sha256") != digest:
        raise ValueError("MX module profile digest differs from selected target profile")
    payload_digest = _text_attr(module, "mx.payload_manifest_sha256")
    if payload_digest is not None and (len(payload_digest) != 64 or
                                       any(c not in "0123456789abcdef" for c in payload_digest)):
        raise ValueError("MX module payload manifest digest is malformed")
    payload_json = _text_attr(module, "mx.payload_manifest_json")
    binding_schema = _text_attr(module, "mx.payload_binding_schema")
    if binding_schema is not None and (
            binding_schema != RESOURCE_SCHEMA or payload_json is None):
        raise ValueError("MX source resource binding schema is unsupported")
    payload_manifest = None
    if payload_json is not None:
        try:
            payload_manifest = json.loads(payload_json)
        except json.JSONDecodeError as error:
            raise ValueError("MX payload resource manifest is malformed") from error
        if (not isinstance(payload_manifest, dict) or
                json.dumps(payload_manifest, sort_keys=True, separators=(",", ":")) != payload_json or
                hashlib.sha256(payload_json.encode()).hexdigest() != payload_digest or
                payload_manifest.get("schema") not in {
                    "mx_gemmini.source_payload.v1",
                    "mx_gemmini.asymmetric_resource_manifest.v2"} or
                payload_manifest.get("profile_sha256") != digest or
                not isinstance(payload_manifest.get("site_id"), str) or
                not isinstance(payload_manifest.get("resources"), dict)):
            raise ValueError("MX payload resource manifest differs from module binding")
        from .source_payload import (ATTENTION_QK_CANDIDATE_ORIGIN,
                                     ATTENTION_PV_PROXY_ORIGIN,
                                     DERIVED_GEMM_FIXTURE_ORIGIN,
                                     validate_attention_qk_candidate,
                                     validate_attention_pv_proxy,
                                     validate_derived_gemm_fixture)
        if payload_manifest.get("origin") == ATTENTION_QK_CANDIDATE_ORIGIN:
            validate_attention_qk_candidate(payload_manifest)
        elif payload_manifest.get("origin") == ATTENTION_PV_PROXY_ORIGIN:
            validate_attention_pv_proxy(payload_manifest)
        elif payload_manifest.get("origin") == DERIVED_GEMM_FIXTURE_ORIGIN:
            validate_derived_gemm_fixture(payload_manifest)
        elif "source_derivation" in payload_manifest:
            raise ValueError("derived MX payload must declare its candidate origin")
        for resource_name, descriptor in payload_manifest["resources"].items():
            if (not isinstance(resource_name, str) or
                    re.fullmatch(r"[a-z][a-z0-9_]*", resource_name) is None or
                    not isinstance(descriptor, dict) or
                    descriptor.get("file") != f"{resource_name}.bin" or
                    not isinstance(descriptor.get("sha256"), str) or
                    len(descriptor["sha256"]) != 64 or
                    any(c not in "0123456789abcdef" for c in descriptor["sha256"]) or
                    not isinstance(descriptor.get("shape"), list) or
                    not descriptor["shape"] or
                    any(type(d) is not int or d <= 0 for d in descriptor["shape"]) or
                    descriptor.get("element_bits") not in {8, 16, 32} or
                    descriptor.get("bytes") !=
                    descriptor["element_bits"] // 8 * math.prod(descriptor["shape"]) or
                    not isinstance(descriptor.get("layout"), str) or
                    not descriptor["layout"]):
                raise ValueError(f"MX payload resource {resource_name} descriptor is malformed")
    contracts = encodes = requants = vpu_commands = spad_requants = resident_contracts = 0
    source_resources: dict[str, object] = {}
    lut_uploads: set[str] = set()
    for op in module.walk():
        name = _operation_name(op)
        if not name.startswith("mx_gemmini."):
            continue
        for local, module_name in (("contract_sha256", "mx.contract_sha256"),
                                   ("policy_sha256", "mx.policy_sha256"),
                                   ("manifest_sha256", "prov.quantization_manifest_sha256")):
            if _text_attr(op, local) != _text_attr(module, module_name):
                raise ValueError(f"{name}: {local} differs from module binding")
        if not _text_attr(op, "site_id"):
            raise ValueError(f"{name}: site ID is absent")
        if _text_attr(op, "profile_sha256") != digest:
            raise ValueError(f"{name}: profile digest differs from selected target profile")
        if name == "mx_gemmini.resource":
            resource_name = _text_attr(op, "resource_name")
            if (binding_schema != RESOURCE_SCHEMA or payload_manifest is None or
                    resource_name not in payload_manifest["resources"] or
                    resource_name in source_resources or
                    _text_attr(op, "payload_manifest_sha256") != payload_digest):
                raise ValueError("MX source resource differs from selected payload")
            descriptor = payload_manifest["resources"][resource_name]
            result_type = op.results[0].type if len(op.results) == 1 else None
            if (not isinstance(result_type, TensorType) or
                    list(result_type.get_shape()) != descriptor["shape"] or
                    not isinstance(result_type.element_type, IntegerType) or
                    result_type.element_type.width.data != descriptor["element_bits"] or
                    _text_attr(op, "resource_sha256") != descriptor["sha256"] or
                    _text_attr(op, "resource_layout") != descriptor["layout"] or
                    _text_attr(op, "site_id") != payload_manifest["site_id"]):
                raise ValueError(f"MX source resource {resource_name} type or digest differs")
            source_resources[resource_name] = op
        elif name == "mx_gemmini.upload_lut":
            target = _text_attr(op, "lut_target")
            resource_name = f"{target}_lut"
            resource_op = op.operands[0].owner if len(op.operands) == 1 else None
            if (binding_schema != RESOURCE_SCHEMA or target not in {
                    "activation", "weight", "output"} or
                    resource_name not in source_resources or
                    resource_op is not source_resources[resource_name] or
                    resource_name in lut_uploads or
                    _text_attr(op, "payload_manifest_sha256") != payload_digest):
                raise ValueError("MX LUT upload differs from checked source resource")
            lut_uploads.add(resource_name)
        elif name == "mx_gemmini.contract":
            local_payload = _text_attr(op, "payload_manifest_sha256")
            if payload_digest is not None:
                if local_payload != payload_digest or _text_attr(op, "payload_origin") not in {
                        "radiance_source_header_specialization",
                        "radiance_source_derived_attention_qk_candidate",
                        "radiance_source_derived_attention_pv_proxy",
                        "radiance_source_derived_gemm_fixture",
                        "nicolas_source_header_specialization",
                        "nicolas_generated_header_specialization"}:
                    raise ValueError("MX contract payload differs from selected source bundle")
            elif local_payload is not None or _text_attr(op, "payload_origin") is not None:
                raise ValueError("MX contract has a payload without module binding")
            if payload_manifest is not None:
                if (payload_manifest["site_id"] != _text_attr(op, "site_id") or
                        payload_manifest.get("origin") != _text_attr(op, "payload_origin")):
                    raise ValueError("MX contract resource manifest differs from site or origin")
                resources = payload_manifest["resources"]
                if not {"activation", "weight", "activation_scales", "weight_scales",
                        "golden_bf16"} <= set(resources):
                    raise ValueError("MX contraction is missing source operand resources")
                for side in ("activation", "weight"):
                    if (_text_attr(op, f"{side}_projection") == "lut" and
                            f"{side}_lut" not in resources):
                        raise ValueError(f"MX {side} LUT is absent from resource manifest")
                if (payload_manifest["schema"] == "mx_gemmini.source_payload.v1" and
                        payload_manifest.get("precision") == "FP6"):
                    expected_luts = {
                        "activation_lut": "row_pair_lut_6bit",
                        "weight_lut": "column_pair_lut_6bit",
                        "output_lut": "output_pair_lut_6bit",
                    }
                    if any(name not in resources or
                           resources[name]["shape"] != [64, 3] or
                           resources[name]["element_bits"] != 32 or
                           resources[name]["layout"] != layout
                           for name, layout in expected_luts.items()):
                        raise ValueError("MX FP6 per-row LUT banks are incomplete")
            if binding_schema == RESOURCE_SCHEMA:
                if len(op.operands) != len(INPUTS) or any(
                        operand.owner is not source_resources.get(resource_name)
                        for operand, resource_name in zip(op.operands, INPUTS)):
                    raise ValueError("MX contraction operands differ from source resources")
                expected_luts = set(LUTS) & set(payload_manifest["resources"])
                if lut_uploads != expected_luts:
                    raise ValueError("MX LUT banks must be uploaded before contraction")
            attributes = {name: _text_attr(op, name) for name in (
                "activation_format", "weight_format", "activation_projection", "weight_projection")}
            if any(value is None for value in attributes.values()):
                raise ValueError("profile-bound MX contract requires named operand formats and projections")
            mode = _int_attr(op, "pe_mode")
            if mode is None:
                matches = [cell for cell in profile["legal_compute"] if
                           all(cell[name] == value for name, value in attributes.items())]
                if len(matches) != 1:
                    raise ValueError("MX contract has no unique mode in selected profile")
                mode = matches[0]["pe_mode"]
            require_compute(profile, attributes["activation_format"], attributes["weight_format"],
                            pe_mode=mode,
                            activation_projection=attributes["activation_projection"],
                            weight_projection=attributes["weight_projection"])
            contracts += 1
        elif name == "mx_gemmini.encode":
            fmt, projection = _text_attr(op, "element_format"), _text_attr(op, "projection")
            if not any(fmt == cell[side + "_format"] and
                       projection == cell[side + "_projection"]
                       for cell in profile["legal_compute"] for side in ("activation", "weight")):
                raise ValueError("MX encode format/projection is absent from selected profile")
            encodes += 1
        elif name == "mx_gemmini.requantize":
            output = _text_attr(op, "output_format")
            if output is None or output not in profile["candidate_output_modes"] or output == "bf16":
                raise ValueError("MX requantize output is absent from selected profile")
            requants += 1
        elif name == "mx_gemmini.readout_bf16":
            layout = _text_attr(op, "memory_layout")
            if "memory_layout" in op.attributes and layout not in {
                    "row_major_bf16", "output_tile_major_bf16"}:
                raise ValueError("MX BF16 readout has an unsupported memory layout")
        elif name == "mx_gemmini.readout_quantized":
            output = _text_attr(op, "output_format")
            if output is None or output not in profile["candidate_output_modes"] or output == "bf16":
                raise ValueError("MX quantized readout output is absent from selected profile")
        elif name == "mx_gemmini.host_requantize":
            fp8 = (_text_attr(op, "output_format") == "fp8_e4m3" and
                   _text_attr(op, "quant_policy") == "radiance_header_fp8_v1" and
                   _text_attr(module, "mx.output_specialization") ==
                   "radiance_header_fp8_host_requant")
            fp6 = (_text_attr(op, "output_format") == "fp6_e3m2" and
                   _text_attr(op, "quant_policy") == "radiance_header_fp6_lut_v1" and
                   _text_attr(module, "mx.output_specialization") ==
                   "radiance_header_fp6_host_requant")
            if not fp8 and not fp6:
                raise ValueError("MX host requantize requires a Radiance FP8 header policy or FP6 LUT header policy")
            shape = payload_manifest.get("shape_mnk") if payload_manifest else None
            if (not isinstance(shape, list) or len(shape) != 3 or
                    any(type(d) is not int or d <= 0 for d in shape) or
                    shape[1] % 32 or len(op.operands) != (2 if fp6 else 1) or
                    len(op.results) != 2 or
                    not isinstance(op.operands[0].owner, Operation) or
                    _operation_name(op.operands[0].owner) != "mx_gemmini.readout_bf16"):
                raise ValueError("MX host requantize lacks a source-bound BF16 readout")
            if fp6 and (source_resources.get("output_lut") is None or
                        op.operands[1].owner is not source_resources["output_lut"]):
                raise ValueError("MX FP6 host requantize lacks its checked output LUT")
            for result, expected_shape in zip(op.results,
                                              ([shape[0] // (2 if fp6 else 1), shape[1]],
                                               [shape[0], shape[1] // 32])):
                result_type = result.type
                if (not isinstance(result_type, TensorType) or
                        list(result_type.get_shape()) != expected_shape or
                        not isinstance(result_type.element_type, IntegerType) or
                        result_type.element_type.width.data != 8):
                    raise ValueError("MX host requantize result shape differs from source payload")
        elif name == "mx_gemmini.vpu_execute":
            vpu_command(profile, kind=_text_attr(op, "kind"),
                        src1_row=_int_attr(op, "src1_row"), src2_row=_int_attr(op, "src2_row"),
                        dst_row=_int_attr(op, "dst_row"), rows=_int_attr(op, "rows"),
                        reduction_length=_int_attr(op, "reduction_length"),
                        broadcast=_bool_attr(op, "broadcast"),
                        immediate_bf16=_int_attr(op, "immediate_bf16"),
                        second_dst_row=_int_attr(op, "second_dst_row"))
            vpu_commands += 1
        elif name == "mx_gemmini.spad_requant":
            spad_requant_command(profile, source_row=_int_attr(op, "source_row"),
                                 destination_row=_int_attr(op, "destination_row"),
                                 m=_int_attr(op, "m"), n=_int_attr(op, "n"),
                                 output_format=_text_attr(op, "output_format"),
                                 tiled=_bool_attr(op, "tiled"),
                                 resident=_bool_attr(op, "resident"),
                                 scale_dram_address=_int_attr(op, "scale_dram_address"),
                                 scale_buffer=_text_attr(op, "scale_buffer"))
            spad_requants += 1
        elif name == "mx_gemmini.resident_contract":
            from .resident_lowering import validate_resident_contract
            validate_resident_contract(profile, {
                key: _int_attr(op, key) for key in
                ("activation_row", "weight_row", "output_row", "m", "n", "k")
            } | {
                key: _text_attr(op, key) for key in
                ("activation_format", "weight_format", "output_format",
                 "weight_buffer", "weight_scales_buffer", "output_scales_buffer")
            })
            resident_contracts += 1
        elif name not in {"mx_gemmini.readout_bf16", "mx_gemmini.readout_to_smem", "mx_gemmini.wait"}:
            raise ValueError(f"unknown MX operation {name}")
    if binding_schema == RESOURCE_SCHEMA:
        expected_resources = set(INPUTS) | (set(LUTS) & set(payload_manifest["resources"]))
        if (set(source_resources) != expected_resources or
                lut_uploads != (set(LUTS) & set(payload_manifest["resources"]))):
            raise ValueError("MX source resource or LUT upload set is incomplete")
    if not (contracts or vpu_commands or spad_requants or resident_contracts):
        raise ValueError("MX profile-bound IR has no executable or contraction operation")
    return {"schema": "mx_gemmini.profile_ir_check.v1",
            "status": profile["qualification"], "profile_sha256": digest,
            "contracts": contracts, "encodes": encodes, "requantizes": requants,
            "vpu_commands": vpu_commands, "spad_requants": spad_requants,
            "resident_contracts": resident_contracts,
            "source_resources": len(source_resources), "lut_uploads": len(lut_uploads)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlir", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--rtl-root", type=Path)
    parser.add_argument("--mx-opt", type=Path)
    args = parser.parse_args()
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    if args.mx_opt:
        subprocess.run([str(args.mx_opt), str(args.mlir), "-o", "/dev/null"], check=True)
    print(json.dumps(verify_ir(args.mlir.read_text(), profile), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
