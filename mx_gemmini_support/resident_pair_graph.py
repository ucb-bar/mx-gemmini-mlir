"""Lower a typed MX contraction/readout/resident-contraction graph.

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
from .resident_pair_plan import (RectangularPairPlan, ResidentPairPlan,
                                 lower_first_fp4_resident,
                                 lower_first_fp6_resident,
                                 lower_first_fp8_rectangular,
                                 lower_first_fp8_resident,
                                 plan_fp4_resident_pair,
                                 plan_fp6_resident_pair,
                                 plan_fp8_rectangular_pair,
                                 plan_fp8_resident_pair)
from .verify_profile_ir import _int_attr, _operation_name, _text_attr, verify_ir


INPUTS = ("a1_activation", "a1_scales", "b1_weight", "b1_scales",
          "b2_weight", "b2_scales")
FP6_LUTS = ("a1_lut", "b1_lut", "b2_lut", "c1_lut", "c2_lut")
FP6_INPUTS = (*INPUTS, *FP6_LUTS)
_BUFFER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")


@dataclass(frozen=True)
class ConnectedPair:
    plan: ResidentPairPlan | RectangularPairPlan
    first_site: str
    second_site: str
    commands: tuple[Command | Fence, ...]


def input_digest(resources: dict[str, bytes], buffers: dict[str, str]) -> str:
    """Hash each logical argument's bytes, independent of runtime symbol names."""
    hashes = {slot: hashlib.sha256(resources[buffers[slot]]).hexdigest()
              for slot in buffers}
    return hashlib.sha256(json.dumps(
        hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def lower_connected_pair(mlir_text: str, profile: dict,
                             resources: dict[str, bytes], *,
                             buffers: dict[str, str],
                             c1_scales: str, c1_tiled_observed: str,
                             c2_tiled: str, a_row: int = 0,
                             precision: str = "fp8_e4m3") -> ConnectedPair:
    """Compile one source-backed E4M3, E2M1, or E3M2 resident pair.

    Diagnostic C1 readout is optional to consume on the host; it never reloads
    C1 before MM2. Numerical claims belong to source-qualified callers.
    """
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func, FuncOp, ReturnOp
    from xdsl.parser import Parser

    if precision not in ("fp8_e4m3", "fp4_e2m1", "fp6_e3m2"):
        raise ValueError("connected resident pair precision is unsupported")
    fp4 = precision == "fp4_e2m1"
    fp6 = precision == "fp6_e3m2"
    packed = fp4 or fp6
    input_slots = FP6_INPUTS if fp6 else INPUTS
    if (set(buffers) != set(input_slots) or
            any(not isinstance(value, str) or not _BUFFER.fullmatch(value)
                for value in (*buffers.values(), c1_scales,
                              c1_tiled_observed, c2_tiled)) or
            len(set((*buffers.values(), c1_scales, c1_tiled_observed,
                     c2_tiled))) != len(input_slots) + 3):
        raise ValueError("connected MX pair needs distinct named ABI buffers")
    checked = verify_ir(mlir_text, profile)
    if (checked["contracts"], checked["resident_contracts"],
            checked["vpu_commands"], checked["spad_requants"],
            checked["runtime_luts"]) != (1, 1, 0, 0, 6 if fp6 else 0):
        raise ValueError("connected MX pair needs MM1 and resident MM2")
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    try:
        digest = input_digest(resources, buffers)
    except (KeyError, TypeError) as error:
        raise ValueError("connected MX pair runtime input is absent") from error
    if _text_attr(module, "mx.runtime_resources_sha256") != digest:
        raise ValueError("connected MX pair runtime payload digest differs")
    functions = [op for op in module.walk() if isinstance(op, FuncOp)]
    if len(functions) != 1:
        raise ValueError("connected MX pair needs one function")
    function = functions[0]
    ops = list(function.body.block.ops)
    expected_ops = (["mx_gemmini.runtime_lut"] * 3 if fp6 else []) + [
        "mx_gemmini.contract", "mx_gemmini.readout_quantized"] + (
        ["mx_gemmini.runtime_lut"] * 3 if fp6 else []) + [
        "mx_gemmini.resident_contract", "func.return"]
    if [_operation_name(op) for op in ops] != expected_ops:
        raise ValueError("connected MX pair operation order differs")
    mm1, readout, mm2, ret = (ops[3], ops[4], ops[8], ops[9]) if fp6 else ops
    attrs = {name: _int_attr(mm2, name) for name in
             ("activation_row", "weight_row", "output_row", "m", "n", "k")}
    attrs.update({name: _text_attr(mm2, name) for name in
                  ("activation_format", "weight_format", "output_format",
                   "weight_buffer", "weight_scales_buffer", "output_scales_buffer")})
    if fp6:
        attrs["lut_groups"] = _int_attr(mm2, "lut_groups")
        attrs.update({name: _text_attr(mm2, name) for name in
                      ("weight_lut_buffer", "activation_lut_buffer",
                       "output_lut_buffer")})
    m, n, k = attrs["m"], attrs["n"], attrs["k"]
    if any(type(value) is not int or value <= 0 for value in (m, n, k)) or k % 32 or n % 32:
        raise ValueError("connected MX pair shape needs complete scale groups")
    args = list(function.body.block.args)
    first_k_match = re.fullmatch(
        rf"tensor<{m // (2 if packed else 1)}x(\d+)xi8>", str(args[0].type))
    if first_k_match is None:
        raise ValueError("connected MX pair MM1 activation shape differs")
    first_k = int(first_k_match.group(1))
    if first_k <= 0 or first_k % 32:
        raise ValueError("connected MX pair MM1 needs complete scale groups")
    arg_types = (f"tensor<{m // (2 if packed else 1)}x{first_k}xi8>",
                 f"tensor<{first_k // 32}x{m}xi8>",
                 f"tensor<{first_k}x{k // (2 if packed else 1)}xi8>",
                 f"tensor<{first_k // 32}x{k}xi8>",
                 f"tensor<{k}x{n // (2 if packed else 1)}xi8>",
                 f"tensor<{k // 32}x{n}xi8>")
    if fp6:
        arg_types += (f"tensor<{m // 2}x3xi32>",) * 5
    first_output_types = (f"tensor<{m // (2 if packed else 1)}x{k}xi8>",
                          f"tensor<{m}x{k // 32}xi8>")
    output_types = (f"tensor<{m // (2 if packed else 1)}x{n}xi8>",
                    f"tensor<{m}x{n // 32}xi8>")
    if (tuple(str(arg.type) for arg in args) != arg_types or
            tuple(str(result.type) for result in mm1.results) !=
            (f"tensor<{m}x{k}xbf16>",) or
            tuple(str(result.type) for result in readout.results) != first_output_types or
            tuple(str(result.type) for result in mm2.results) != output_types or
            list(mm1.operands) != args[:4] or
            list(readout.operands) != list(mm1.results) or
            list(mm2.operands) != [*readout.results, *args[4:6]] or
            not isinstance(ret, ReturnOp) or
            list(ret.operands) != list(mm2.results)):
        raise ValueError("connected MX pair SSA tensor edges differ")
    first_site = _text_attr(mm1, "site_id")
    second_site = _text_attr(mm2, "site_id")
    if (first_site != _text_attr(readout, "site_id") or
            first_site == second_site or
            any(_text_attr(mm1, name) != value for name, value in {
                "activation_format": precision, "weight_format": precision,
                "activation_projection": "lut" if fp6 else "direct",
                "weight_projection": "lut" if fp6 else "direct"}.items()) or
            _int_attr(mm1, "pe_mode") != (4 if fp6 else 0 if fp4 else 8) or
            _text_attr(readout, "output_format") != precision):
        raise ValueError("connected MX pair site or precision differs")
    if fp6:
        bindings = (
            (ops[0], "weight", "b1_lut", 7, first_site),
            (ops[1], "activation", "a1_lut", 6, first_site),
            (ops[2], "output", "c1_lut", 9, first_site),
            (ops[5], "weight", "b2_lut", 8, second_site),
            (ops[6], "activation", "c1_lut", 9, second_site),
            (ops[7], "output", "c2_lut", 10, second_site))
        if any(_text_attr(op, "site_id") != site or
               _text_attr(op, "lut_target") != target or
               _text_attr(op, "runtime_buffer") != buffers[slot] or
               _int_attr(op, "groups") != m // 2 or
               _int_attr(op, "entry_bits") != 6 or
               list(op.operands) != [args[index]]
               for op, target, slot, index, site in bindings):
            raise ValueError("connected FP6 LUT SSA sites or runtime buffers differ")
        if (attrs["weight_lut_buffer"] != buffers["b2_lut"] or
                attrs["activation_lut_buffer"] != buffers["c1_lut"] or
                attrs["output_lut_buffer"] != buffers["c2_lut"] or
                attrs["lut_groups"] != m // 2):
            raise ValueError("connected FP6 MM2 LUT reuse differs")
    expected_lengths = (m * first_k // (2 if packed else 1), m * first_k // 32,
                        first_k * k // (2 if packed else 1), first_k * k // 32,
                        k * n // (2 if packed else 1), k * n // 32)
    if fp6:
        expected_lengths += (m // 2 * 12,) * 5
    if any(len(resources[buffers[slot]]) != length
           for slot, length in zip(input_slots, expected_lengths)):
        raise ValueError("connected MX pair physical input size differs")
    if (attrs["weight_buffer"] != buffers["b2_weight"] or
            attrs["weight_scales_buffer"] != buffers["b2_scales"] or
            not isinstance(attrs["output_scales_buffer"], str) or
            not _BUFFER.fullmatch(attrs["output_scales_buffer"]) or
            len(set((*buffers.values(), c1_scales, c1_tiled_observed,
                     c2_tiled, attrs["output_scales_buffer"]))) != len(input_slots) + 4 or
            any(name in resources for name in (
                c1_scales, c1_tiled_observed, c2_tiled,
                attrs["output_scales_buffer"]))):
        raise ValueError("connected MX pair MM2 buffers differ from ABI")
    rectangular = first_k != k or n != k
    if packed:
        plan_packed = plan_fp6_resident_pair if fp6 else plan_fp4_resident_pair
        plan = plan_packed(
            profile, shape=(m, n, k), a_row=a_row,
            c1_row=attrs["activation_row"], c2_row=attrs["output_row"])
        if rectangular:
            raise ValueError("connected packed resident pair has no rectangular source fixture")
    elif rectangular:
        plan = plan_fp8_rectangular_pair(
            profile, first_shape=(m, k, first_k),
            second_shape=(m, n, k), a_row=a_row,
            c1_row=attrs["activation_row"], c2_row=attrs["output_row"])
    else:
        plan = plan_fp8_resident_pair(
            profile, shape=(m, n, k), a_row=a_row,
            c1_row=attrs["activation_row"], c2_row=attrs["output_row"],
            allow_a_c1_reuse=(
                (m, n, k, a_row, attrs["activation_row"], attrs["output_row"]) ==
                (64, 64, 64, 0, 128, 512) and
                profile["name"] == "MxGemminiRocketConfig"))
    if attrs["weight_row"] != plan.b_row:
        raise ValueError("connected MX pair B placement differs from profile")
    lower_first = (lower_first_fp6_resident if fp6 else
                   lower_first_fp4_resident if fp4 else
                   lower_first_fp8_rectangular if rectangular else lower_first_fp8_resident)
    first_kwargs = dict(
        activation_buffer=buffers["a1_activation"],
        activation_scales_buffer=buffers["a1_scales"],
        weight_buffer=buffers["b1_weight"],
        weight_scales_buffer=buffers["b1_scales"],
        output_scales_buffer=c1_scales)
    if fp6:
        first_kwargs.update(
            activation_lut_buffer=buffers["a1_lut"],
            weight_lut_buffer=buffers["b1_lut"],
            output_lut_buffer=buffers["c1_lut"])
    commands: list[Command | Fence] = list(lower_first(plan, **first_kwargs))
    commands.append(_config_st(plan.dim))
    first_output_rows = plan.c1_rows if rectangular else plan.c_rows
    for row in range(0, first_output_rows, plan.dim):
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


def lower_connected_fp8_pair(mlir_text: str, profile: dict,
                             resources: dict[str, bytes], *, buffers: dict[str, str],
                             c1_scales: str, c1_tiled_observed: str,
                             c2_tiled: str, a_row: int = 0) -> ConnectedPair:
    return lower_connected_pair(
        mlir_text, profile, resources, buffers=buffers, c1_scales=c1_scales,
        c1_tiled_observed=c1_tiled_observed, c2_tiled=c2_tiled, a_row=a_row)
