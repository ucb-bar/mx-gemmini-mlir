"""Check named MX MLIR operations against a source-bound target profile."""

from __future__ import annotations

import argparse
import json
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
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
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
    contracts = encodes = requants = vpu_commands = spad_requants = resident_contracts = 0
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
        if name == "mx_gemmini.contract":
            local_payload = _text_attr(op, "payload_manifest_sha256")
            if payload_digest is not None:
                if local_payload != payload_digest or _text_attr(op, "payload_origin") not in {
                        "radiance_source_header_specialization",
                        "nicolas_source_header_specialization"}:
                    raise ValueError("MX contract payload differs from selected source bundle")
            elif local_payload is not None or _text_attr(op, "payload_origin") is not None:
                raise ValueError("MX contract has a payload without module binding")
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
        elif name == "mx_gemmini.readout_quantized":
            output = _text_attr(op, "output_format")
            if output is None or output not in profile["candidate_output_modes"] or output == "bf16":
                raise ValueError("MX quantized readout output is absent from selected profile")
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
    if not (contracts or vpu_commands or spad_requants or resident_contracts):
        raise ValueError("MX profile-bound IR has no executable or contraction operation")
    return {"schema": "mx_gemmini.profile_ir_check.v1",
            "status": profile["qualification"], "profile_sha256": digest,
            "contracts": contracts, "encodes": encodes, "requantizes": requants,
            "vpu_commands": vpu_commands, "spad_requants": spad_requants,
            "resident_contracts": resident_contracts}


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
