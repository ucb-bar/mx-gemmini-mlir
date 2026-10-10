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
    first_width: int = 64
    c1_row: int = 128
    c2_row: int = 512


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
    m = _int_attr(mm2, "m")
    if type(m) is not int or m not in (64, 128) or _int_attr(mm2, "k") != m:
        raise ValueError("connected MX VPU pair needs a complete square MM1 tile")
    if type(n) is not int or n < 32 or n % 32:
        raise ValueError("connected MX VPU pair needs a complete E8M0 MM2 width")
    if {name: lengths[name] for name in INPUTS} != {
            "a1_activation": m * m, "a1_scales": m * m // 32,
            "b1_weight": m * m, "b1_scales": m * m // 32,
            "b2_weight": m * n, "b2_scales": m * n // 32}:
        raise ValueError("connected MX VPU pair physical input sizes differ")
    args = list(function.body.block.args)
    first_args = (f"tensor<{m}x{m}xi8>", f"tensor<{m // 32}x{m}xi8>",
                  f"tensor<{m}x{m}xi8>", f"tensor<{m // 32}x{m}xi8>")
    quantized = (f"tensor<{m}x{m}xi8>", f"tensor<{m}x{m // 32}xi8>")
    final = (f"tensor<{m}x{n}xi8>", f"tensor<{m}x{n // 32}xi8>")
    if (tuple(str(arg.type) for arg in args) != first_args +
            (f"tensor<{m}x{n}xi8>", f"tensor<{m // 32}x{n}xi8>") or
            tuple(str(result.type) for result in mm1.results) != (f"tensor<{m}x{m}xbf16>",) or
            tuple(str(result.type) for result in readout.results) != (f"tensor<{m}x{m}xbf16>",) or
            any(tuple(str(result.type) for result in vpu.results) !=
                (f"tensor<{m}x{m}xbf16>",) for vpu in vpus) or
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
    c1_row, c2_row = attrs["activation_row"], attrs["output_row"]
    bf16_start, bf16_end = 0x1000, 0x1000 + m * m * 2 // 16
    c1_end = c1_row + m * m // 16
    c2_end = c2_row + m * n // 16
    if (tuple(attrs[key] for key in ("m", "n", "k")) != (m, n, m) or
            attrs["weight_row"] != rows - m * n // 16 or
            any(not (end <= bf16_start or start >= bf16_end)
                for start, end in ((c1_row, c1_end), (c2_row, c2_end))) or
            attrs["weight_buffer"] != buffers["b2_weight"] or
            attrs["weight_scales_buffer"] != buffers["b2_scales"] or
            attrs["output_scales_buffer"] != outputs["c2_scales"] or
            _text_attr(requant, "scale_buffer") != outputs["c1_scales"] or
            _int_attr(requant, "source_row") != 0x1000 or
            _int_attr(requant, "destination_row") != c1_row or
            _int_attr(requant, "m") != m or _int_attr(requant, "n") != m or
            _text_attr(requant, "output_format") != "fp8_e4m3" or
            not _bool_attr(requant, "tiled") or not _bool_attr(requant, "resident") or
            _int_attr(requant, "scale_dram_address") != 0):
        raise ValueError("connected MX VPU pair placement or ABI differs")
    if any(_text_attr(vpu, "kind") not in {"muls", "adds"} or
           _int_attr(vpu, "src1_row") != 0x1000 or
           _int_attr(vpu, "src2_row") != 0 or
           _int_attr(vpu, "dst_row") != 0x1000 or
           _int_attr(vpu, "rows") != m * m * 2 // 16 or
           _int_attr(vpu, "reduction_length") != 1 or
           _bool_attr(vpu, "broadcast") or
           _int_attr(vpu, "second_dst_row") is not None for vpu in vpus):
        raise ValueError("connected MX VPU pair needs in-place scalar operations")
    vectors = tuple(vpu_command(
        profile, kind=_text_attr(vpu, "kind"), src1_row=0x1000, src2_row=0,
        dst_row=0x1000, rows=m * m * 2 // 16, reduction_length=1,
        broadcast=False, immediate_bf16=_int_attr(vpu, "immediate_bf16"))
        for vpu in vpus)
    quant = spad_requant_command(
        profile, source_row=0x1000, destination_row=c1_row,
        m=m, n=m, output_format="fp8_e4m3", tiled=True,
        resident=True, scale_dram_address=0,
        scale_buffer=outputs["c1_scales"])
    first_buffers = {**{slot: buffers[slot] for slot in INPUTS[:4]},
                     "c1_scales": outputs["c1_scales"],
                     "c1_bf16_observed": outputs["c1_bf16_observed"]}
    first = emit_verified_first_matrix_commands(
        profile, resources, buffers=first_buffers, shape=(m, m, m))
    vector_steps = tuple(step for vector in vectors for step in (vector, Fence()))
    return ConnectedVpuPair(
        first_site, second_site, n,
        (*first, Fence(), *vector_steps, quant, Fence(),
         *lower_resident_contract(profile, attrs)), m, c1_row, c2_row)
