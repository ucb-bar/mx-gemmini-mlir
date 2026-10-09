"""Bind one checked source payload bundle to a profile-bound MX MLIR site."""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

from .source_payload import load_bundle, manifest_sha256
from .target_profile import load_profile, profile_sha256
from .verify_profile_ir import _operation_name, _text_attr, verify_ir


_FORMAT = {"FP8": ("fp8_e4m3", "direct"),
           "FP4": ("fp4_e2m1", "direct"),
           "FP6": ("fp6_e3m2", "lut")}


def bind_payload(mlir_text: str, profile: dict, manifest: dict) -> str:
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
    if manifest.get("output_format") is not None:
        if manifest["output_format"] != "fp8_e4m3":
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
        codes_type = TensorType(i8, [m, n])
        scales_type = TensorType(i8, [m, n // 32])
        quant = UnregisteredOp.with_name("mx_gemmini.readout_quantized").create(
            operands=readout.operands, result_types=[codes_type, scales_type],
            attributes={**readout.attributes,
                        "output_format": StringAttr(manifest["output_format"])})
        block.insert_op_before(quant, readout)
        block.insert_op_before(ReturnOp(*quant.results), old_return)
        block.erase_op(old_return)
        block.erase_op(readout)
        function.update_function_type()
        module.attributes["mx.output_specialization"] = StringAttr(
            "source_header_quantized")
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
