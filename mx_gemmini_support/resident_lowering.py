"""Profile-checked physical lowering for a DIM16 FP8 resident contraction.

The typed operation names the live scratchpad tile and source buffers. The
64-cubed VPU/requant and 128-cubed direct requant chains use the same physical
MM2 schedule with different source-checked scratchpad lifetimes.
"""

from __future__ import annotations

import re

from .command_ir import Command, Fence, Operand, spad_requant_command, vpu_command


_BUFFER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")


def validate_resident_contract(profile: dict, attrs: dict) -> None:
    if profile.get("transport") != "rocket_rocc" or not profile["resources"].get("requantizer"):
        raise ValueError("resident MX contraction needs a selected RoCC requantizer profile")
    if profile["geometry"].get("mesh_columns") != 16:
        raise ValueError("resident MX contraction requires the qualified DIM16 layout")
    shape = (attrs["m"], attrs["n"], attrs["k"])
    plain_shape = (type(shape[0]) is int and shape[0] in range(16, 129, 16) and
                   shape[1] in range(32, 129, 32) and shape[2] in (96, 128))
    if shape != (64, 64, 64) and not plain_shape:
        raise ValueError("resident MX contraction needs a supported complete tile")
    if shape == (64, 64, 64) and not profile["resources"].get("spad_requant"):
        raise ValueError("64³ resident MX contraction needs the qualified SPAD_REQUANT profile")
    if plain_shape and (
            profile["name"] != "MxGemminiRocketConfig" or
            profile["resources"].get("spad_requant") or
            profile["resources"].get("vpu")):
        raise ValueError("plain resident MX contraction needs Nicolas's plain MX profile")
    if any(attrs[key] != "fp8_e4m3" for key in
           ("activation_format", "weight_format", "output_format")):
        raise ValueError("resident MX contraction requires E4M3 inputs and output")
    for key in ("weight_buffer", "weight_scales_buffer", "output_scales_buffer"):
        if not isinstance(attrs[key], str) or not _BUFFER.fullmatch(attrs[key]):
            raise ValueError(f"resident MX {key} must name a runtime buffer")
    m, n, k = shape
    activation_rows, weight_rows, output_rows = (m * k // 16, k * n // 16,
                                                  m * n // 16)
    rows = profile["resources"]["scratchpad_bytes"] // 16
    a, b, c = (attrs[key] for key in ("activation_row", "weight_row", "output_row"))
    if any(type(value) is not int for value in (a, b, c)) or not (
            0 <= a and all(value % 16 == 0 for value in (a, b, c)) and
            a + activation_rows <= c and
            c + output_rows <= b and b + weight_rows == rows and
            rows <= 1 << 14):
        raise ValueError("resident MX scratchpad tile placement or lifetime differs")
    if "fp8_e4m3" not in profile["candidate_output_modes"]:
        raise ValueError("resident MX output mode is absent from selected profile")


def lower_resident_contract(profile: dict, attrs: dict) -> tuple[Command | Fence, ...]:
    """Issue source-audited B2 DMA, resident scales, and LOOP_WS_SPAD."""
    validate_resident_contract(profile, attrs)

    def cmd(funct: int, rs1: int | Operand, rs2: int | Operand) -> Command:
        return Command(funct, rs1 if isinstance(rs1, Operand) else Operand(immediate=rs1),
                       rs2 if isinstance(rs2, Operand) else Operand(immediate=rs2))

    m, n, k = attrs["m"], attrs["n"], attrs["k"]
    a, b, c = (attrs[key] for key in ("activation_row", "weight_row", "output_row"))
    i, j, kk = m // 16, n // 16, k // 16
    commands: list[Command | Fence] = [
        # gemmini_extended3_config_ex(WS, ..., E4M3 inputs/output).
        cmd(0, (1 << 16) | (1 << 2), 1 << 48),
        # gemmini_mx_load_scales(B2, sizeof(B2), weight selector 1).
        cmd(27, Operand(buffer=attrs["weight_scales_buffer"]), (1 << 32) | (k // 32 * n)),
        Fence(),
        # B2 is row-major in DRAM and operand-B tile-major in bank 3.
        cmd(0, (16 << 16) | (1 << 8) | 1, n),
    ]
    # Nicolas's square chain used j-major source traversal. For a rectangular
    # B matrix the model and RTL consume tile (k,j) at k*j+j, so preserve the
    # source's row-major K×N layout when K and N differ.
    tiles = ((tk, tj) for tk in range(kk) for tj in range(j)) if n != k else (
        (tk, tj) for tj in range(j) for tk in range(kk))
    for tk, tj in tiles:
        offset = (tk * 16 * n + tj * 16) if n != k else (tj * 16 * n + tk * 16)
        row = b + ((tk * j + tj) if n != k else (tj * kk + tk)) * 16
        commands.append(cmd(2, Operand(buffer=attrs["weight_buffer"], byte_offset=offset),
                            (16 << 48) | (16 << 32) | row))
    commands += [
        Fence(),
        cmd(0, 2, 2),  # gemmini_config_st(sizeof(uint16_t))
        # gemmini_mxquant_config_mvout_resident; A scales are already resident
        # from the preceding requant operation or source C1 preload.
        cmd(26, Operand(buffer=attrs["output_scales_buffer"],
                        address_mask=(1 << 33) - 1,
                        or_bits=(1 << 63) | (kk << 51) | (j << 42) | (i << 33)), 1),
        cmd(9, 0, (kk << 32) | (j << 16) | i),
        cmd(24, a, b + k * n // 16),
        cmd(8, 0, (c << 32) | 0x200 | 0x38 | (1 << 10)),
        Fence(),
    ]
    return tuple(commands)


def lower_single_resident_contract(mlir_text: str, profile: dict
                                   ) -> tuple[Command | Fence, ...]:
    """Lower one SSA-bound resident MM2 without a source-specific VPU seam."""
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func, FuncOp, ReturnOp
    from xdsl.parser import Parser

    from .verify_profile_ir import _int_attr, _operation_name, _text_attr, verify_ir

    report = verify_ir(mlir_text, profile)
    if (report["contracts"], report["vpu_commands"], report["spad_requants"],
            report["resident_contracts"]) != (0, 0, 0, 1):
        raise ValueError("standalone resident MX lowering needs exactly one MM2 op")
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    functions = [op for op in module.walk() if isinstance(op, FuncOp)]
    if len(functions) != 1:
        raise ValueError("standalone resident MX lowering needs one function")
    function = functions[0]
    ops = list(function.body.block.ops)
    if [_operation_name(op) for op in ops] != [
            "mx_gemmini.resident_contract", "func.return"]:
        raise ValueError("standalone resident MX function contains an unexpected op")
    contract, ret = ops
    attrs = {key: _int_attr(contract, key) for key in
             ("activation_row", "weight_row", "output_row", "m", "n", "k")}
    attrs.update({key: _text_attr(contract, key) for key in
                  ("activation_format", "weight_format", "output_format",
                   "weight_buffer", "weight_scales_buffer", "output_scales_buffer")})
    m, n, k = attrs["m"], attrs["n"], attrs["k"]
    expected = (f"tensor<{m}x{k}xi8>", f"tensor<{m}x{k // 32}xi8>",
                f"tensor<{k}x{n}xi8>", f"tensor<{k // 32}x{n}xi8>")
    results = (f"tensor<{m}x{n}xi8>", f"tensor<{m}x{n // 32}xi8>")
    args = list(function.body.block.args)
    if (tuple(str(arg.type) for arg in args) != expected or
            tuple(str(result.type) for result in contract.results) != results or
            list(contract.operands) != args or
            not isinstance(ret, ReturnOp) or
            list(ret.operands) != list(contract.results)):
        raise ValueError("standalone resident MX SSA tensor edges differ from MM2")
    return lower_resident_contract(profile, attrs)


def lower_resident_chain_commands(mlir_text: str, profile: dict, *,
                                  expected_sites: tuple[str, str] | None = None
                                  ) -> tuple[Command | Fence, ...]:
    """Require an ordered typed VPU, resident requant, and second contract."""
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser

    from .verify_profile_ir import _bool_attr, _int_attr, _operation_name, _text_attr, verify_ir

    report = verify_ir(mlir_text, profile)
    if (report["contracts"], report["vpu_commands"], report["spad_requants"],
            report["resident_contracts"]) != (0, 1, 1, 1):
        raise ValueError("resident chain requires one typed VPU, requant, and second contract")
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    ops = [op for op in module.walk() if _operation_name(op).startswith("mx_gemmini.")]
    if [_operation_name(op) for op in ops] != [
            "mx_gemmini.vpu_execute", "mx_gemmini.spad_requant",
            "mx_gemmini.resident_contract"]:
        raise ValueError("resident chain operation order differs")
    vpu, requant, contract = ops
    if expected_sites is not None and [
            _text_attr(op, "site_id") for op in ops] != [
                expected_sites[0], expected_sites[0], expected_sites[1]]:
        raise ValueError("resident chain sites differ from frontend contractions")
    if (_int_attr(vpu, "dst_row") != _int_attr(requant, "source_row") or
            _int_attr(requant, "destination_row") != _int_attr(contract, "activation_row") or
            not _bool_attr(requant, "resident") or not _bool_attr(requant, "tiled") or
            _int_attr(requant, "n") != _int_attr(contract, "k") or
            _int_attr(requant, "m") != _int_attr(contract, "m") or
            _text_attr(requant, "output_format") != _text_attr(contract, "activation_format") or
            _text_attr(vpu, "site_id") == _text_attr(contract, "site_id")):
        raise ValueError("resident chain scratchpad or scale handoff differs")
    vector = vpu_command(
        profile, kind=_text_attr(vpu, "kind"), src1_row=_int_attr(vpu, "src1_row"),
        src2_row=_int_attr(vpu, "src2_row"), dst_row=_int_attr(vpu, "dst_row"),
        rows=_int_attr(vpu, "rows"), reduction_length=_int_attr(vpu, "reduction_length"),
        broadcast=_bool_attr(vpu, "broadcast"),
        immediate_bf16=_int_attr(vpu, "immediate_bf16"),
        second_dst_row=_int_attr(vpu, "second_dst_row"))
    quant = spad_requant_command(
        profile, source_row=_int_attr(requant, "source_row"),
        destination_row=_int_attr(requant, "destination_row"),
        m=_int_attr(requant, "m"), n=_int_attr(requant, "n"),
        output_format=_text_attr(requant, "output_format"),
        tiled=_bool_attr(requant, "tiled"), resident=_bool_attr(requant, "resident"),
        scale_dram_address=_int_attr(requant, "scale_dram_address"),
        scale_buffer=_text_attr(requant, "scale_buffer"))
    attrs = {key: _int_attr(contract, key) for key in
             ("activation_row", "weight_row", "output_row", "m", "n", "k")}
    attrs.update({key: _text_attr(contract, key) for key in
                  ("activation_format", "weight_format", "output_format",
                   "weight_buffer", "weight_scales_buffer", "output_scales_buffer")})
    return (vector, Fence(), quant, Fence(), *lower_resident_contract(profile, attrs))
