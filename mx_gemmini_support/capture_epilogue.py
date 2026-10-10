"""Bind a captured pointwise MX epilogue through its source graph provenance."""

from __future__ import annotations

import hashlib
import json
import math
import struct

from .bind_payload import append_tilewise_vpu_muls, append_tilewise_vpu_x2
from .verify_profile_ir import _text_attr


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def append_captured_tilewise_vpu_muls(mlir_text: str, profile: dict,
                                      payload_manifest: dict, capture) -> str:
    """Lower exactly `matmul * finite BF16 scalar` from a model2MLIR capture.

    The frontend graph and final MLIR must match the digest-gated handoff.
    Unmatched or wider scalar values fail closed instead of silently changing
    the graph's arithmetic when encoded in the VPU immediate.
    """
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser

    trace = getattr(capture, "capture_trace", None)
    quant = getattr(capture, "quantization_manifest", None)
    frontend = getattr(capture, "mlir_text", None)
    if (not getattr(capture, "ok", False) or not isinstance(trace, dict) or
            not isinstance(quant, dict) or not isinstance(frontend, str) or
            trace.get("status") != "complete" or trace.get("blockers")):
        raise ValueError("scalar VPU binding needs a complete model2MLIR capture")
    original = trace.get("graphs", {}).get("original", {})
    original_digest = original.get("sha256")
    if (original.get("schema") != "m2m.frontend_graph.v1" or
            original.get("status") != "complete" or
            original_digest != quant.get("source_graph_sha256") or
            original_digest != _digest({k: v for k, v in original.items()
                                        if k != "sha256"}) or
            trace.get("mlir", {}).get("sha256") != hashlib.sha256(
                frontend.encode()).hexdigest()):
        raise ValueError("scalar VPU capture graph or MLIR digest differs")
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    if (_text_attr(module, "mx.source_mlir_sha256") != trace["mlir"]["sha256"] or
            _text_attr(module, "prov.quantization_manifest_sha256") != _digest(quant)):
        raise ValueError("scalar VPU capture differs from bound MX handoff")

    nodes = original.get("nodes", [])
    if (len(nodes) != 5 or [node.get("op") for node in nodes] != [
            "placeholder", "placeholder", "call_function", "call_function", "output"] or
            [node.get("target") for node in nodes[2:4]] != [
                "aten.matmul.default", "aten.mul.Tensor"] or
            nodes[2].get("args") != [
                {"node_id": node["id"], "value_id": node["id"] + ":v0"}
                for node in nodes[:2]] or
            not isinstance(nodes[3].get("args"), list) or
            len(nodes[3]["args"]) != 2 or
            nodes[3]["args"][0] != {
                "node_id": nodes[2]["id"], "value_id": nodes[2]["id"] + ":v0"} or
            nodes[4].get("args") != [[{
                "node_id": nodes[3]["id"], "value_id": nodes[3]["id"] + ":v0"}]]):
        raise ValueError("scalar VPU binding supports one returned matmul-scalar chain")
    scalar = nodes[3]["args"][1]
    try:
        finite = type(scalar) in {float, int} and math.isfinite(scalar)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("scalar VPU capture needs a finite literal")
    try:
        scalar_f32 = struct.unpack("<I", struct.pack("<f", scalar))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("scalar VPU capture literal exceeds BF16 range") from error
    if scalar_f32 & 0xffff or (scalar_f32 & 0x7f800000) == 0x7f800000:
        raise ValueError("scalar VPU capture literal is not exactly finite BF16")
    scalar_bf16 = scalar_f32 >> 16
    sites = quant.get("sites", [])
    selected_format = {"FP8": "mxfp8", "FP4": "mxfp4"}.get(
        payload_manifest.get("precision"))
    if (not isinstance(sites, list) or len(sites) != 1 or
            not isinstance(sites[0], dict) or
            sites[0].get("site_id") != payload_manifest.get("site_id") or
            sites[0].get("status") != "quantized" or
            sites[0].get("format") != selected_format or
            sites[0].get("shape") != payload_manifest.get("shape_mnk") or
            "linalg.matmul" not in frontend or
            'prov.aten = "aten.mul.Tensor"' not in frontend or
            f"{scalar:.6e}" not in frontend):
        raise ValueError("scalar VPU source site or frontend operation differs")
    if scalar_bf16 == 0x4000:
        return append_tilewise_vpu_x2(mlir_text, profile, payload_manifest)
    return append_tilewise_vpu_muls(mlir_text, profile, payload_manifest,
                                    scalar_bf16)
