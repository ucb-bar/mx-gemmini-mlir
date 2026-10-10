"""Lower a typed MM1→BF16 readout→scalar VPU chain→requant→MM2 graph.

This is the qualified DIM16 E4M3 64³ mode from Nicolas's MX+VPU build.
The graph and runtime input bytes, rather than a source C fixture, determine
the physical command stream. Source adapters can compare it to their audit.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from .command_ir import Command, Fence, spad_requant_command, vpu_command
from .first_matrix_lowering import emit_verified_first_matrix_commands
from .resident_lowering import lower_resident_contract
from .resident_pair_graph import INPUTS, input_digest
from .target_profile import profile_sha256
from .verify_profile_ir import (_bool_attr, _int_attr, _operation_name,
                                _text_attr, verify_ir)


OUTPUTS = ("c1_scales", "c1_bf16_observed", "c1_tiled", "c2_scales", "c2_tiled")
_BUFFER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")


@dataclass(frozen=True)
class ConnectedVpuPair:
    first_site: str
    second_site: str
    second_width: int
    commands: tuple[Command | Fence, ...]


def lower_connected_fp8_vpu_pair(mlir_text: str, profile: dict,
                                 resources: dict[str, bytes], *,
                                 buffers: dict[str, str],
                                 outputs: dict[str, str]) -> ConnectedVpuPair:
    """Compile an in-place BF16 scalar VPU chain and resident FP8 matrix pair.

    The typed module binds the six input byte streams by digest. The caller
    supplies distinct runtime C symbols for inputs, diagnostic readouts, and
    scale outputs. C1 stays resident between SPAD_REQUANT and MM2.
    """
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func, FuncOp, ReturnOp
    from xdsl.parser import Parser

    if set(buffers) != set(INPUTS) or set(outputs) != set(OUTPUTS):
        raise ValueError("connected MX VPU pair ABI slots differ")
    symbols = (*buffers.values(), *outputs.values())
    if (any(not isinstance(name, str) or not _BUFFER.fullmatch(name)
            for name in symbols) or len(set(symbols)) != len(INPUTS) + len(OUTPUTS)):
        raise ValueError("connected MX VPU pair needs distinct C buffer symbols")
    try:
        digest = input_digest(resources, buffers)
        lengths = {slot: len(resources[buffers[slot]]) for slot in INPUTS}
    except (KeyError, TypeError) as error:
        raise ValueError("connected MX VPU pair runtime input is absent") from error
    if any(lengths.get(name) != size for name, size in {
            "a1_activation": 4096, "a1_scales": 128,
            "b1_weight": 4096, "b1_scales": 128}.items()):
        raise ValueError("connected MX VPU pair MM1 input sizes differ from 64³")
    report = verify_ir(mlir_text, profile)
    vpu_count = report["vpu_commands"]
    if (report["contracts"] != 1 or not 1 <= vpu_count <= 16 or
            report["spad_requants"] != 1 or report["resident_contracts"] != 1):
        raise ValueError("connected MX VPU pair needs MM1, 1..16 VPU ops, requant, and MM2")
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    if (_text_attr(module, "mx.profile_sha256") != profile_sha256(profile) or
            _text_attr(module, "mx.runtime_resources_sha256") != digest):
        raise ValueError("connected MX VPU pair profile or input digest differs")
    functions = [op for op in module.walk() if isinstance(op, FuncOp)]
    if len(functions) != 1:
        raise ValueError("connected MX VPU pair needs one function")
    function = functions[0]
    ops = list(function.body.block.ops)
    if [_operation_name(op) for op in ops] != [
            "mx_gemmini.contract", "mx_gemmini.readout_bf16",
            *["mx_gemmini.vpu_execute"] * vpu_count,
            "mx_gemmini.spad_requant", "mx_gemmini.resident_contract",
            "func.return"]:
        raise ValueError("connected MX VPU pair operation order differs")
    mm1, readout = ops[:2]
    vpus = ops[2:2 + vpu_count]
    requant, mm2, ret = ops[2 + vpu_count:]
    n = _int_attr(mm2, "n")
    if type(n) is not int or n < 32 or n % 32:
        raise ValueError("connected MX VPU pair needs a complete E8M0 MM2 width")
    if (lengths["b2_weight"], lengths["b2_scales"]) != (64 * n, 2 * n):
        raise ValueError("connected MX VPU pair MM2 input sizes differ")
    args = list(function.body.block.args)
    first_args = ("tensor<64x64xi8>", "tensor<2x64xi8>",
                  "tensor<64x64xi8>", "tensor<2x64xi8>")
    quantized = ("tensor<64x64xi8>", "tensor<64x2xi8>")
    final = (f"tensor<64x{n}xi8>", f"tensor<64x{n // 32}xi8>")
    if (tuple(str(arg.type) for arg in args) != first_args +
            (f"tensor<64x{n}xi8>", f"tensor<2x{n}xi8>") or
            tuple(str(result.type) for result in mm1.results) != ("tensor<64x64xbf16>",) or
            tuple(str(result.type) for result in readout.results) != ("tensor<64x64xbf16>",) or
            any(tuple(str(result.type) for result in vpu.results) !=
                ("tensor<64x64xbf16>",) for vpu in vpus) or
            tuple(str(result.type) for result in requant.results) != quantized or
            tuple(str(result.type) for result in mm2.results) != final or
            list(mm1.operands) != args[:4] or
            list(readout.operands) != list(mm1.results) or
            list(vpus[0].operands) != list(readout.results) or
            any(list(right.operands) != list(left.results)
                for left, right in zip(vpus, vpus[1:])) or
            list(requant.operands) != list(vpus[-1].results) or
            list(mm2.operands) != [*requant.results, *args[4:]] or
            not isinstance(ret, ReturnOp) or
            list(ret.operands) != list(mm2.results)):
        raise ValueError("connected MX VPU pair SSA tensor edges differ")
    first_site = _text_attr(mm1, "site_id")
    second_site = _text_attr(mm2, "site_id")
    if ([_text_attr(op, "site_id") for op in (readout, *vpus, requant)] !=
            [first_site] * (vpu_count + 2) or first_site == second_site or
            any(_text_attr(mm1, name) != "fp8_e4m3" for name in
                ("activation_format", "weight_format")) or
            any(_text_attr(mm1, name) != "direct" for name in
                ("activation_projection", "weight_projection")) or
            _int_attr(mm1, "pe_mode") != 8):
        raise ValueError("connected MX VPU pair site or MM1 precision differs")
    attrs = {key: _int_attr(mm2, key) for key in
             ("activation_row", "weight_row", "output_row", "m", "n", "k")}
    attrs.update({key: _text_attr(mm2, key) for key in
                  ("activation_format", "weight_format", "output_format",
                   "weight_buffer", "weight_scales_buffer", "output_scales_buffer")})
    rows = profile["resources"]["scratchpad_bytes"] // 16
    if (tuple(attrs[key] for key in ("m", "n", "k")) != (64, n, 64) or
            tuple(attrs[key] for key in ("activation_row", "weight_row", "output_row")) !=
            (128, rows - 64 * n // 16, 512) or
            attrs["weight_buffer"] != buffers["b2_weight"] or
            attrs["weight_scales_buffer"] != buffers["b2_scales"] or
            attrs["output_scales_buffer"] != outputs["c2_scales"] or
            _text_attr(requant, "scale_buffer") != outputs["c1_scales"] or
            _int_attr(requant, "source_row") != 0x1000 or
            _int_attr(requant, "destination_row") != 128 or
            _int_attr(requant, "m") != 64 or _int_attr(requant, "n") != 64 or
            _text_attr(requant, "output_format") != "fp8_e4m3" or
            not _bool_attr(requant, "tiled") or not _bool_attr(requant, "resident") or
            _int_attr(requant, "scale_dram_address") != 0):
        raise ValueError("connected MX VPU pair placement or ABI differs")
    if any(_text_attr(vpu, "kind") not in {"muls", "adds"} or
           _int_attr(vpu, "src1_row") != 0x1000 or
           _int_attr(vpu, "src2_row") != 0 or
           _int_attr(vpu, "dst_row") != 0x1000 or
           _int_attr(vpu, "rows") != 512 or
           _int_attr(vpu, "reduction_length") != 1 or
           _bool_attr(vpu, "broadcast") or
           _int_attr(vpu, "second_dst_row") is not None for vpu in vpus):
        raise ValueError("connected MX VPU pair needs in-place scalar operations")
    vectors = tuple(vpu_command(
        profile, kind=_text_attr(vpu, "kind"), src1_row=0x1000, src2_row=0,
        dst_row=0x1000, rows=512, reduction_length=1,
        broadcast=False, immediate_bf16=_int_attr(vpu, "immediate_bf16"))
        for vpu in vpus)
    quant = spad_requant_command(
        profile, source_row=0x1000, destination_row=128,
        m=64, n=64, output_format="fp8_e4m3", tiled=True,
        resident=True, scale_dram_address=0,
        scale_buffer=outputs["c1_scales"])
    first_buffers = {**{slot: buffers[slot] for slot in INPUTS[:4]},
                     "c1_scales": outputs["c1_scales"],
                     "c1_bf16_observed": outputs["c1_bf16_observed"]}
    first = emit_verified_first_matrix_commands(
        profile, resources, buffers=first_buffers)
    vector_steps = tuple(step for vector in vectors for step in (vector, Fence()))
    return ConnectedVpuPair(
        first_site, second_site, n,
        (*first, Fence(), *vector_steps, quant, Fence(),
         *lower_resident_contract(profile, attrs)))
