"""Bind one checked source payload bundle to a profile-bound MX MLIR site."""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

from .resource_ir import attach_source_resources
from .source_payload import load_bundle, manifest_sha256
from .target_profile import load_profile, profile_sha256
from .verify_profile_ir import _operation_name, _text_attr, verify_ir


_FORMAT = {"FP8": ("fp8_e4m3", "direct"),
           "FP4": ("fp4_e2m1", "direct"),
           "FP6": ("fp6_e3m2", "lut")}


def bind_payload(mlir_text: str, profile: dict, manifest: dict, *,
                 source_header_quantized: bool = False) -> str:
    """Mark a captured contraction with the exact materialized source bytes.

    This is an explicit source specialization of a model2MLIR capture. It
    preserves the frontend's structural contract and records that numerical
    operands came from the selected source header bundle.
    """
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin, StringAttr, TensorType, UnregisteredOp, i8
    from xdsl.dialects.func import Func, FuncOp, ReturnOp
    from xdsl.parser import Parser
    from xdsl.printer import Printer

    verify_ir(mlir_text, profile)
    if manifest.get("profile_sha256") != profile_sha256(profile):
        raise ValueError("payload profile digest differs from selected target")
    if manifest.get("origin") != "radiance_source_header_specialization":
        raise ValueError("payload origin is not an explicit source specialization")
    precision = manifest.get("precision")
    if precision not in _FORMAT:
        raise ValueError("payload precision is unknown")
    fmt, projection = _FORMAT[precision]
    site = manifest.get("site_id")
    if not isinstance(site, str) or not site:
        raise ValueError("payload site ID is absent")
    digest = manifest_sha256(manifest)
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    if "mx.payload_manifest_sha256" in module.attributes:
        raise ValueError("MX MLIR is already payload-bound")
    contracts = [op for op in module.walk()
                 if _operation_name(op) == "mx_gemmini.contract"]
    if len(contracts) != 1 or _text_attr(contracts[0], "site_id") != site:
        raise ValueError("payload v1 requires exactly one matching contraction site")
    contract = contracts[0]
    for side in ("activation", "weight"):
        if (_text_attr(contract, f"{side}_format") != fmt or
                _text_attr(contract, f"{side}_projection") != projection):
            raise ValueError("source payload precision/projection differs from MLIR contract")
    readouts = [op for op in module.walk()
                if _operation_name(op) == "mx_gemmini.readout_bf16" and
                _text_attr(op, "site_id") == site]
    if len(readouts) != 1:
        raise ValueError("source BF16 payload needs a matching MLIR readout")
    module.attributes["mx.payload_manifest_sha256"] = StringAttr(digest)
    contract.attributes["payload_manifest_sha256"] = StringAttr(digest)
    contract.attributes["payload_origin"] = StringAttr(manifest["origin"])
    source_resources = attach_source_resources(module, contract, manifest)
    if source_header_quantized and (
            precision not in {"FP8", "FP4", "FP6"} or
            manifest.get("source_quant_golden_convention") != "source_header" or
            manifest.get("output_specialization") is not None):
        raise ValueError("Radiance header requantization requires an FP8/FP4/FP6 source quantized driver")
    if manifest.get("output_format") is not None:
        if manifest["output_format"] not in {"fp8_e4m3", "fp4_e2m1", "fp6_e3m2"}:
            raise ValueError("source quantized output has an unsupported format")
        readout = readouts[0]
        function = readout.parent_op()
        if not isinstance(function, FuncOp) or not isinstance(function.get_return_op(), ReturnOp):
            raise ValueError("source quantized output requires a returning function")
        old_return = function.get_return_op()
        if list(old_return.operands) != list(readout.results):
            raise ValueError("source BF16 readout must be the returned value")
        block = readout.parent
        assert block is not None
        m, n, _ = manifest["shape_mnk"]
        codes_type = TensorType(i8, [m if source_header_quantized and precision != "FP6" else
                                     m // 2 if precision in {"FP4", "FP6"} else m, n])
        scales_type = TensorType(i8, [m, n // 32])
        if source_header_quantized:
            if precision == "FP6" and "output_lut" not in source_resources:
                raise ValueError("Radiance FP6 source requantization needs its output LUT")
            quant = UnregisteredOp.with_name("mx_gemmini.host_requantize").create(
                operands=[*readout.results, *([source_resources["output_lut"]]
                                              if precision == "FP6" else [])],
                result_types=[codes_type, scales_type],
                attributes={**readout.attributes,
                            "output_format": StringAttr(
                                "fp6_e3m2" if precision == "FP6" else "fp8_e4m3"),
                            "quant_policy": StringAttr(
                                "radiance_header_fp6_lut_v1" if precision == "FP6"
                                else "radiance_header_fp8_v1")})
            block.insert_op_before(quant, old_return)
        else:
            quant = UnregisteredOp.with_name("mx_gemmini.readout_quantized").create(
                operands=readout.operands, result_types=[codes_type, scales_type],
                attributes={**readout.attributes,
                            "output_format": StringAttr(manifest["output_format"])})
            block.insert_op_before(quant, readout)
        block.insert_op_before(ReturnOp(*quant.results), old_return)
        block.erase_op(old_return)
        if not source_header_quantized:
            block.erase_op(readout)
        function.update_function_type()
        module.attributes["mx.output_specialization"] = StringAttr(
            ("radiance_header_fp6_host_requant" if precision == "FP6"
             else "radiance_header_fp8_host_requant") if source_header_quantized else
            "matrix_vpu_x2_spad_requant_fp8" if manifest.get("output_specialization") ==
            "matrix_vpu_x2_spad_requant_fp8" else
            "source_bf16_fp6_lut_quantized" if
            manifest.get("output_specialization") == "bf16_fullout_to_fp6_lut_quantized" else
            "source_header_quantized")
    output = StringIO()
    Printer(stream=output).print_op(module)
    rendered = output.getvalue() + "\n"
    verify_ir(rendered, profile)
    return rendered


def append_vpu_spad_requant_x2(mlir_text: str, profile: dict, manifest: dict) -> str:
    """Compose typed VPU and resident requant after one source-bound matmul."""
    from xdsl.context import Context
    from xdsl.dialects.builtin import (Builtin, BoolAttr, IntegerAttr, StringAttr,
                                      TensorType, UnregisteredOp, bf16, i8, i32, i64)
    from xdsl.dialects.func import Func, FuncOp, ReturnOp
    from xdsl.parser import Parser
    from xdsl.printer import Printer

    if (manifest.get("output_specialization") != "matrix_vpu_x2_spad_requant_fp8" or
            manifest.get("shape_mnk") != [64, 64, 128]):
        raise ValueError("VPU/SPAD x2 composition needs the selected FP8 source specialization")
    verify_ir(mlir_text, profile)
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    if _text_attr(module, "mx.output_specialization") != "matrix_vpu_x2_spad_requant_fp8":
        raise ValueError("VPU/SPAD x2 composition needs a matching typed output readout")
    ops = [op for op in module.walk()
           if _operation_name(op).startswith("mx_gemmini.") and
           _operation_name(op) not in {"mx_gemmini.resource", "mx_gemmini.upload_lut"}]
    if [_operation_name(op) for op in ops] != ["mx_gemmini.contract", "mx_gemmini.readout_quantized"]:
        raise ValueError("VPU/SPAD x2 composition needs one bare contraction and readout")
    from .source_gemm import plan_mx_gemm
    plan = plan_mx_gemm(shape=(64, 64, 128), tile=tuple(manifest["tile_mnk"]),
                        datatype="FP8", quant_output=False, acc_to_gmem=False,
                        scratchpad_bytes=profile["resources"]["scratchpad_bytes"],
                        profile=profile)
    source_row, rows = plan["c_spad_dest"], plan["c_rows"]
    destination_row = source_row + rows + 256
    if destination_row + 256 > plan["scratchpad_rows"]:
        raise ValueError("VPU/SPAD x2 output has no nonoverlapping scratchpad lifetime")
    readout = ops[-1]
    block = readout.parent
    assert block is not None
    function = readout.parent_op()
    if not isinstance(function, FuncOp) or not isinstance(function.get_return_op(), ReturnOp):
        raise ValueError("VPU/SPAD x2 requires a returning source contraction")
    old_return = function.get_return_op()
    if list(old_return.operands) != list(readout.results):
        raise ValueError("VPU/SPAD x2 source readout must be returned")
    binding = {name: readout.attributes[name]
               for name in ("site_id", "profile_sha256", "contract_sha256",
                            "policy_sha256", "manifest_sha256")}
    i32attr = lambda value: IntegerAttr(value, i32)
    bf16_type = TensorType(bf16, [64, 64])
    codes_type = TensorType(i8, [64, 64])
    scales_type = TensorType(i8, [64, 2])
    bf16_readout = UnregisteredOp.with_name("mx_gemmini.readout_bf16").create(
        operands=list(ops[0].results), result_types=[bf16_type], attributes=binding)
    vpu = UnregisteredOp.with_name("mx_gemmini.vpu_execute").create(
        operands=list(bf16_readout.results), result_types=[bf16_type], attributes={
        **binding, "kind": StringAttr("muls"),
        "src1_row": i32attr(source_row), "src2_row": i32attr(0),
        "dst_row": i32attr(source_row), "rows": i32attr(rows),
        "reduction_length": i32attr(1), "broadcast": BoolAttr.from_bool(False),
        "immediate_bf16": i32attr(0x4000)})
    requant = UnregisteredOp.with_name("mx_gemmini.spad_requant").create(
        operands=list(vpu.results), result_types=[codes_type, scales_type], attributes={
        **binding, "source_row": i32attr(source_row),
        "destination_row": i32attr(destination_row),
        "m": i32attr(64), "n": i32attr(64),
        "output_format": StringAttr("fp8_e4m3"),
        "tiled": BoolAttr.from_bool(True), "resident": BoolAttr.from_bool(True),
        "scale_dram_address": IntegerAttr(0, i64),
        "scale_buffer": StringAttr("scratch_output_scales")})
    block.insert_op_before(bf16_readout, readout)
    block.insert_op_before(vpu, readout)
    block.insert_op_before(requant, readout)
    block.insert_op_before(ReturnOp(*requant.results), old_return)
    block.erase_op(old_return)
    block.erase_op(readout)
    function.update_function_type()
    output = StringIO()
    Printer(stream=output).print_op(module)
    rendered = output.getvalue() + "\n"
    verify_ir(rendered, profile)
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlir", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--rtl-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    manifest, _ = load_bundle(args.bundle)
    rendered = bind_payload(args.mlir.read_text(), profile, manifest)
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered)
    print(f"bound source payload {manifest_sha256(manifest)} -> {args.out}")


if __name__ == "__main__":
    main()
