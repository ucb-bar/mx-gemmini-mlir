"""Lower a typed FP8 contraction/readout/resident-contraction graph.

Source adapters supply runtime bytes and an ABI buffer map. This module
checks the SSA edges, profile, payload digest, scratchpad lifetimes, and
physical buffer sizes before issuing either matrix command stream.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

from .command_ir import Command, Fence
from .physical_program import _config_st, _transfer
from .resident_lowering import lower_resident_contract
from .resident_pair_plan import (ResidentPairPlan, lower_first_fp8_resident,
                                 plan_fp8_resident_pair)
from .verify_profile_ir import _int_attr, _operation_name, _text_attr, verify_ir


INPUTS = ("a1_activation", "a1_scales", "b1_weight", "b1_scales",
          "b2_weight", "b2_scales")
_BUFFER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")


@dataclass(frozen=True)
class ConnectedPair:
    plan: ResidentPairPlan
    first_site: str
    second_site: str
    commands: tuple[Command | Fence, ...]


def input_digest(resources: dict[str, bytes], buffers: dict[str, str]) -> str:
    """Hash each logical argument's bytes, independent of runtime symbol names."""
    hashes = {slot: hashlib.sha256(resources[buffers[slot]]).hexdigest()
              for slot in INPUTS}
    return hashlib.sha256(json.dumps(
        hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def lower_connected_fp8_pair(mlir_text: str, profile: dict,
                             resources: dict[str, bytes], *,
                             buffers: dict[str, str],
                             c1_scales: str, c1_tiled_observed: str,
                             c2_tiled: str, a_row: int = 0) -> ConnectedPair:
    """Compile one direct E4M3 MM1→resident MM2 pair to ordered RoCC commands.

    Diagnostic C1 readout is optional to consume on the host; it never reloads
    C1 before MM2. Numerical claims belong to source-qualified callers.
    """
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func, FuncOp, ReturnOp
    from xdsl.parser import Parser

    if (set(buffers) != set(INPUTS) or
            any(not isinstance(value, str) or not _BUFFER.fullmatch(value)
                for value in (*buffers.values(), c1_scales,
                              c1_tiled_observed, c2_tiled)) or
            len(set((*buffers.values(), c1_scales, c1_tiled_observed,
                     c2_tiled))) != len(INPUTS) + 3):
        raise ValueError("connected FP8 pair needs distinct named ABI buffers")
    checked = verify_ir(mlir_text, profile)
    if (checked["contracts"], checked["resident_contracts"],
            checked["vpu_commands"], checked["spad_requants"]) != (1, 1, 0, 0):
        raise ValueError("connected FP8 pair needs MM1 and resident MM2")
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    try:
        digest = input_digest(resources, buffers)
    except (KeyError, TypeError) as error:
        raise ValueError("connected FP8 pair runtime input is absent") from error
    if _text_attr(module, "mx.runtime_resources_sha256") != digest:
        raise ValueError("connected FP8 pair runtime payload digest differs")
    functions = [op for op in module.walk() if isinstance(op, FuncOp)]
    if len(functions) != 1:
        raise ValueError("connected FP8 pair needs one function")
    function = functions[0]
    ops = list(function.body.block.ops)
    if [_operation_name(op) for op in ops] != [
            "mx_gemmini.contract", "mx_gemmini.readout_quantized",
            "mx_gemmini.resident_contract", "func.return"]:
        raise ValueError("connected FP8 pair operation order differs")
    mm1, readout, mm2, ret = ops
    attrs = {name: _int_attr(mm2, name) for name in
             ("activation_row", "weight_row", "output_row", "m", "n", "k")}
    attrs.update({name: _text_attr(mm2, name) for name in
                  ("activation_format", "weight_format", "output_format",
                   "weight_buffer", "weight_scales_buffer", "output_scales_buffer")})
    m, n, k = attrs["m"], attrs["n"], attrs["k"]
    if any(type(value) is not int or value <= 0 for value in (m, n, k)) or k % 32 or n % 32:
        raise ValueError("connected FP8 pair shape needs complete scale groups")
    args = list(function.body.block.args)
    arg_types = (f"tensor<{m}x{k}xi8>", f"tensor<{k // 32}x{m}xi8>",
                 f"tensor<{k}x{n}xi8>", f"tensor<{k // 32}x{n}xi8>")
    output_types = (f"tensor<{m}x{n}xi8>", f"tensor<{m}x{n // 32}xi8>")
    if (tuple(str(arg.type) for arg in args) != arg_types + arg_types[2:] or
            tuple(str(result.type) for result in mm1.results) !=
            (f"tensor<{m}x{n}xbf16>",) or
            tuple(str(result.type) for result in readout.results) != output_types or
            tuple(str(result.type) for result in mm2.results) != output_types or
            list(mm1.operands) != args[:4] or
            list(readout.operands) != list(mm1.results) or
            list(mm2.operands) != [*readout.results, *args[4:]] or
            not isinstance(ret, ReturnOp) or
            list(ret.operands) != list(mm2.results)):
        raise ValueError("connected FP8 pair SSA tensor edges differ")
    first_site = _text_attr(mm1, "site_id")
    second_site = _text_attr(mm2, "site_id")
    if (first_site != _text_attr(readout, "site_id") or
            first_site == second_site or
            any(_text_attr(mm1, name) != value for name, value in {
                "activation_format": "fp8_e4m3", "weight_format": "fp8_e4m3",
                "activation_projection": "direct", "weight_projection": "direct"}.items()) or
            _int_attr(mm1, "pe_mode") != 8 or
            _text_attr(readout, "output_format") != "fp8_e4m3"):
        raise ValueError("connected FP8 pair site or precision differs")
    expected_lengths = (m * k, m * k // 32, k * n, k * n // 32,
                        k * n, k * n // 32)
    if any(len(resources[buffers[slot]]) != length
           for slot, length in zip(INPUTS, expected_lengths)):
        raise ValueError("connected FP8 pair physical input size differs")
    if (attrs["weight_buffer"] != buffers["b2_weight"] or
            attrs["weight_scales_buffer"] != buffers["b2_scales"] or
            not isinstance(attrs["output_scales_buffer"], str) or
            not _BUFFER.fullmatch(attrs["output_scales_buffer"]) or
            len(set((*buffers.values(), c1_scales, c1_tiled_observed,
                     c2_tiled, attrs["output_scales_buffer"]))) != len(INPUTS) + 4 or
            any(name in resources for name in (
                c1_scales, c1_tiled_observed, c2_tiled,
                attrs["output_scales_buffer"]))):
        raise ValueError("connected FP8 pair MM2 buffers differ from ABI")
    plan = plan_fp8_resident_pair(
        profile, shape=(m, n, k), a_row=a_row,
        c1_row=attrs["activation_row"], c2_row=attrs["output_row"])
    if attrs["weight_row"] != plan.b_row:
        raise ValueError("connected FP8 pair B placement differs from profile")
    commands: list[Command | Fence] = list(lower_first_fp8_resident(
        plan, activation_buffer=buffers["a1_activation"],
        activation_scales_buffer=buffers["a1_scales"],
        weight_buffer=buffers["b1_weight"],
        weight_scales_buffer=buffers["b1_scales"],
        output_scales_buffer=c1_scales))
    commands.append(_config_st(plan.dim))
    for row in range(0, plan.c_rows, plan.dim):
        commands.append(_transfer(3, c1_tiled_observed, row * plan.dim,
                                  plan.c1_row + row))
    commands.append(Fence())
    commands.extend(lower_resident_contract(profile, attrs))
    commands.append(_config_st(plan.dim))
    for row in range(0, plan.c_rows, plan.dim):
        commands.append(_transfer(3, c2_tiled, row * plan.dim,
                                  plan.c2_row + row))
    commands.append(Fence())
    return ConnectedPair(plan, first_site, second_site, tuple(commands))
