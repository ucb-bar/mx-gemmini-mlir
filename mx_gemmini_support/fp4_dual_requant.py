"""Lower a typed BF16 tile into flat and operand-A-tiled FP4 images."""

from __future__ import annotations

import hashlib
import json

from .command_ir import Command, Fence, spad_requant_command
from .physical_program import _cmd, _config_ld, _config_st, _transfer
from .target_profile import profile_sha256
from .verify_profile_ir import _bool_attr, _int_attr, _operation_name, _text_attr, verify_ir


M, N = 64, 128
SOURCE_ROW, FLAT_ROW, TILED_ROW = 0, 0x1000, 0x2000
BUFFERS = ("X", "scales_hw", "scales_hw2", "codes_flat_hw", "codes_tiled_hw")


def render_fp4_dual_requant(profile: dict, *, source_sha256: str,
                           header_sha256: str) -> str:
    """Bind the two layouts of Nicolas's FP4 requant source to one SSA input."""
    digest = profile_sha256(profile)
    policy = {"m": M, "n": N, "source_row": SOURCE_ROW,
              "output_rows": [FLAT_ROW, TILED_ROW],
              "output_format": "fp4_e2m1", "source_sha256": source_sha256}
    policy_sha = hashlib.sha256(json.dumps(
        policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    attrs = (f'site_id = "nicolas:spad_requant_fp4", '
             f'profile_sha256 = "{digest}", '
             f'contract_sha256 = "{source_sha256}", '
             f'policy_sha256 = "{policy_sha}", '
             f'manifest_sha256 = "{header_sha256}"')
    def operation(name: str, destination: int, tiled: bool, scale: str) -> str:
        return f'''    %{name}, %{name}s = "mx_gemmini.spad_requant"(%x) {{
      source_row = {SOURCE_ROW} : i32, destination_row = {destination} : i32,
      m = {M} : i32, n = {N} : i32, output_format = "fp4_e2m1",
      tiled = {str(tiled).lower()}, resident = false,
      scale_dram_address = 0 : i64, scale_buffer = "{scale}",
      {attrs}}} : (tensor<{M}x{N}xbf16>)
      -> (tensor<{M // 2}x{N}xi8>, tensor<{M}x{N // 32}xi8>)
'''
    return f'''module attributes {{mx.profile_sha256 = "{digest}",
  mx.contract_sha256 = "{source_sha256}",
  mx.policy_sha256 = "{policy_sha}",
  prov.quantization_manifest_sha256 = "{header_sha256}"}} {{
  func.func @nicolas_spad_requant_fp4(%x: tensor<{M}x{N}xbf16>)
      -> (tensor<{M // 2}x{N}xi8>, tensor<{M}x{N // 32}xi8>,
          tensor<{M // 2}x{N}xi8>, tensor<{M}x{N // 32}xi8>) {{
{operation("flat", FLAT_ROW, False, "scales_hw")}{operation("tiled", TILED_ROW, True, "scales_hw2")}    func.return %flat, %flats, %tiled, %tileds
      : tensor<{M // 2}x{N}xi8>, tensor<{M}x{N // 32}xi8>,
        tensor<{M // 2}x{N}xi8>, tensor<{M}x{N // 32}xi8>
  }}
}}
'''


def lower_fp4_dual_requant(mlir_text: str, profile: dict) -> tuple[Command | Fence, ...]:
    """Lower both checked SSA uses of the same BF16 tile to Rocket commands."""
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func, FuncOp, ReturnOp
    from xdsl.parser import Parser

    report = verify_ir(mlir_text, profile)
    if (report["contracts"], report["spad_requants"]) != (0, 2):
        raise ValueError("dual FP4 requant needs exactly two typed operations")
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    funcs = [op for op in module.walk() if isinstance(op, FuncOp)]
    if len(funcs) != 1:
        raise ValueError("dual FP4 requant needs one function")
    args, ops = list(funcs[0].body.block.args), list(funcs[0].body.block.ops)
    if (len(args) != 1 or str(args[0].type) != f"tensor<{M}x{N}xbf16>" or
            [_operation_name(op) for op in ops] !=
            ["mx_gemmini.spad_requant", "mx_gemmini.spad_requant", "func.return"] or
            any(list(op.operands) != args or len(op.results) != 2 for op in ops[:2]) or
            not isinstance(ops[2], ReturnOp) or
            list(ops[2].operands) != [*ops[0].results, *ops[1].results]):
        raise ValueError("dual FP4 requant SSA edges differ")
    commands: list[Command | Fence] = [_cmd(7, 0, 0), _config_ld(16), _config_st(16)]
    commands.extend(_transfer(2, "X", row * 16, SOURCE_ROW + row)
                    for row in range(0, M * N // 8, 16))
    for op, expected in zip(ops[:2],
                            ((FLAT_ROW, False, "scales_hw", "codes_flat_hw"),
                             (TILED_ROW, True, "scales_hw2", "codes_tiled_hw"))):
        destination, tiled, scale, output = expected
        if (_text_attr(op, "site_id") != "nicolas:spad_requant_fp4" or
                _int_attr(op, "source_row") != SOURCE_ROW or
                _int_attr(op, "destination_row") != destination or
                (_int_attr(op, "m"), _int_attr(op, "n")) != (M, N) or
                _text_attr(op, "output_format") != "fp4_e2m1" or
                _bool_attr(op, "tiled") != tiled or _bool_attr(op, "resident") or
                _text_attr(op, "scale_buffer") != scale):
            raise ValueError("dual FP4 requant placement or output binding differs")
        commands.append(spad_requant_command(
            profile, source_row=SOURCE_ROW, destination_row=destination,
            m=M, n=N, output_format="fp4_e2m1", tiled=tiled,
            resident=False, scale_dram_address=0, scale_buffer=scale))
        commands.extend(_transfer(3, output, row * 16, destination + row)
                        for row in range(0, M * N // 32, 16))
    commands.append(Fence())
    return tuple(commands)
