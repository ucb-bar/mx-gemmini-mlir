"""Lower typed BF16 tiles into flat and operand-A-tiled FP4 or FP8 images."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from .command_ir import Command, Fence, spad_requant_command
from .physical_program import _cmd, _config_ld, _config_st, _transfer
from .target_profile import profile_sha256
from .verify_profile_ir import _bool_attr, _int_attr, _operation_name, _text_attr, verify_ir


M, N = 64, 128
SOURCE_ROW, FLAT_ROW, TILED_ROW = 0, 0x1000, 0x2000
BUFFERS = ("X", "scales_hw", "scales_hw2", "codes_flat_hw", "codes_tiled_hw")


@dataclass(frozen=True)
class DualRequantSpec:
    source_name: str
    m: int
    n: int
    output_format: str
    code_bytes_per_element: int

    def __post_init__(self) -> None:
        if (self.m <= 0 or self.n <= 0 or self.m % 16 or self.n % 32 or
                (self.output_format, self.code_bytes_per_element) not in
                {("fp4_e2m1", 2), ("fp8_e4m3", 1)}):
            raise ValueError("unsupported dual SPAD_REQUANT shape or format")

    @property
    def output_rows(self) -> int:
        return self.m // self.code_bytes_per_element

    @property
    def output_bytes(self) -> int:
        return self.m * self.n // self.code_bytes_per_element


FP4_SPEC = DualRequantSpec("spad_requant_fp4", M, N, "fp4_e2m1", 2)
FP8_SPEC = DualRequantSpec("spad_requant", 32, 64, "fp8_e4m3", 1)


def render_dual_requant(profile: dict, *, source_sha256: str,
                       header_sha256: str, spec: DualRequantSpec) -> str:
    """Bind the two layouts of Nicolas's requant source to one SSA input."""
    m, n = spec.m, spec.n
    digest = profile_sha256(profile)
    policy = {"m": m, "n": n, "source_row": SOURCE_ROW,
              "output_rows": [FLAT_ROW, TILED_ROW],
              "output_format": spec.output_format, "source_sha256": source_sha256}
    policy_sha = hashlib.sha256(json.dumps(
        policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    attrs = (f'site_id = "nicolas:{spec.source_name}", '
             f'profile_sha256 = "{digest}", '
             f'contract_sha256 = "{source_sha256}", '
             f'policy_sha256 = "{policy_sha}", '
             f'manifest_sha256 = "{header_sha256}"')
    def operation(name: str, destination: int, tiled: bool, scale: str) -> str:
        return f'''    %{name}, %{name}s = "mx_gemmini.spad_requant"(%x) {{
      source_row = {SOURCE_ROW} : i32, destination_row = {destination} : i32,
      m = {m} : i32, n = {n} : i32, output_format = "{spec.output_format}",
      tiled = {str(tiled).lower()}, resident = false,
      scale_dram_address = 0 : i64, scale_buffer = "{scale}",
      {attrs}}} : (tensor<{m}x{n}xbf16>)
      -> (tensor<{spec.output_rows}x{n}xi8>, tensor<{m}x{n // 32}xi8>)
'''
    return f'''module attributes {{mx.profile_sha256 = "{digest}",
  mx.contract_sha256 = "{source_sha256}",
  mx.policy_sha256 = "{policy_sha}",
  prov.quantization_manifest_sha256 = "{header_sha256}"}} {{
  func.func @nicolas_{spec.source_name}(%x: tensor<{m}x{n}xbf16>)
      -> (tensor<{spec.output_rows}x{n}xi8>, tensor<{m}x{n // 32}xi8>,
          tensor<{spec.output_rows}x{n}xi8>, tensor<{m}x{n // 32}xi8>) {{
{operation("flat", FLAT_ROW, False, "scales_hw")}{operation("tiled", TILED_ROW, True, "scales_hw2")}    func.return %flat, %flats, %tiled, %tileds
      : tensor<{spec.output_rows}x{n}xi8>, tensor<{m}x{n // 32}xi8>,
        tensor<{spec.output_rows}x{n}xi8>, tensor<{m}x{n // 32}xi8>
  }}
}}
'''


def lower_dual_requant(mlir_text: str, profile: dict,
                       spec: DualRequantSpec) -> tuple[Command | Fence, ...]:
    """Lower both checked SSA uses of the same BF16 tile to Rocket commands."""
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func, FuncOp, ReturnOp
    from xdsl.parser import Parser

    report = verify_ir(mlir_text, profile)
    if (report["contracts"], report["spad_requants"]) != (0, 2):
        raise ValueError("dual SPAD_REQUANT needs exactly two typed operations")
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    funcs = [op for op in module.walk() if isinstance(op, FuncOp)]
    if len(funcs) != 1:
        raise ValueError("dual SPAD_REQUANT needs one function")
    args, ops = list(funcs[0].body.block.args), list(funcs[0].body.block.ops)
    if (len(args) != 1 or str(args[0].type) != f"tensor<{spec.m}x{spec.n}xbf16>" or
            [_operation_name(op) for op in ops] !=
            ["mx_gemmini.spad_requant", "mx_gemmini.spad_requant", "func.return"] or
            any(list(op.operands) != args or len(op.results) != 2 for op in ops[:2]) or
            not isinstance(ops[2], ReturnOp) or
            list(ops[2].operands) != [*ops[0].results, *ops[1].results]):
        raise ValueError("dual SPAD_REQUANT SSA edges differ")
    commands: list[Command | Fence] = [_cmd(7, 0, 0), _config_ld(16), _config_st(16)]
    commands.extend(_transfer(2, "X", row * 16, SOURCE_ROW + row)
                    for row in range(0, spec.m * spec.n // 8, 16))
    for op, expected in zip(ops[:2],
                            ((FLAT_ROW, False, "scales_hw", "codes_flat_hw"),
                             (TILED_ROW, True, "scales_hw2", "codes_tiled_hw"))):
        destination, tiled, scale, output = expected
        if (_text_attr(op, "site_id") != f"nicolas:{spec.source_name}" or
                _int_attr(op, "source_row") != SOURCE_ROW or
                _int_attr(op, "destination_row") != destination or
                (_int_attr(op, "m"), _int_attr(op, "n")) != (spec.m, spec.n) or
                _text_attr(op, "output_format") != spec.output_format or
                _bool_attr(op, "tiled") != tiled or _bool_attr(op, "resident") or
                _text_attr(op, "scale_buffer") != scale or
                str(op.results[0].type) != f"tensor<{spec.output_rows}x{spec.n}xi8>" or
                str(op.results[1].type) != f"tensor<{spec.m}x{spec.n // 32}xi8>"):
            raise ValueError("dual SPAD_REQUANT placement or output binding differs")
        commands.append(spad_requant_command(
            profile, source_row=SOURCE_ROW, destination_row=destination,
            m=spec.m, n=spec.n, output_format=spec.output_format, tiled=tiled,
            resident=False, scale_dram_address=0, scale_buffer=scale))
        commands.extend(_transfer(3, output, row * 16, destination + row)
                        for row in range(0, spec.output_bytes // 16, 16))
    commands.append(Fence())
    return tuple(commands)
