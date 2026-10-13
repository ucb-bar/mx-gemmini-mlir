"""Render one typed source-bound MX attention command graph.

The frontend's QK→softmax→PV edges are audited by the vector binding. This
graph adds explicit scale, operand, matrix, and readout operations around the
already-qualified typed VPU/requant middle.
"""

from __future__ import annotations

from .command_ir import Command, Fence
from .memory_phase import (lower_linear_spad_mvout, lower_matrix_mvin,
                           lower_scale_load, lower_strided_scale_dma)
from .nicolas_attention_io import ATTENTION_BUFFERS, lower_dense_attention_program
from .nicolas_attention_matrix import lower_spad_matmul
from .nicolas_attention_shape import DenseAttentionShape
from .nicolas_attention_vector import lower_dense_attention_vector
from .target_profile import profile_sha256
from .verify_profile_ir import _int_attr, _operation_name, _text_attr, verify_ir


BUFFER_BYTES = DenseAttentionShape(32, 64, 64).buffer_bytes()
SCHEMA = "mx_gemmini.nicolas_dense_attention_program.v1"


def render_dense_attention_program(vector_mlir: str, profile: dict,
                                   binding: dict, *, phase: str = "whole") -> str:
    """Wrap the checked vector middle in ordered typed MX memory/matmul ops."""
    if phase not in {"whole", "setup", "check", "timed"}:
        raise ValueError("dense attention needs a selected issue phase")
    vector = lower_dense_attention_vector(vector_mlir, profile, binding)
    if len(vector) != 8:
        raise ValueError("dense attention needs the checked vector middle")
    shape = DenseAttentionShape.from_plan(binding["plan"])
    shape.validate(profile)
    buffer_bytes = shape.buffer_bytes()
    digest = profile_sha256(profile)
    source = binding["source_sha256"]
    common = f'profile_sha256 = "{digest}", source_sha256 = "{source}"'
    vector_ops = [line for line in vector_mlir.splitlines()
                  if line.strip().startswith(('"mx_gemmini.vpu_execute"',
                                              '"mx_gemmini.spad_requant"'))]
    if len(vector_ops) != 8:
        raise ValueError("dense attention typed vector operations differ")

    def memory(name: str, buffer: str, attrs: str, site: str) -> str:
        return (f'    "mx_gemmini.{name}"(%{buffer}) '
                f'{{site_id = "{site}", {attrs}, {common}}} : '
                f'(memref<{buffer_bytes[buffer]}xi8>) -> ()')

    def matmul(site: str, *, m: int, n: int, k: int,
               a: int, b_end: int, c: int, half: int) -> str:
        return (f'    "mx_gemmini.matmul_spad"() {{site_id = "{site}", '
                f'm = {m} : i32, n = {n} : i32, k = {k} : i32, '
                f'activation_row = {a} : i32, weight_end_row = {b_end} : i32, '
                f'output_row = {c} : i32, weight_half = {half} : i32, '
                f'scale_sink_buffer = "scale_sink", {common}}} : () -> ()')

    args = ", ".join(f"%{name}: memref<{buffer_bytes[name]}xi8>"
                     for name in ATTENTION_BUFFERS)
    lines = [
        "builtin.module attributes {",
        f'  mx.attention_program_schema = "{SCHEMA}",',
        *([f'  mx.attention_phase = "{phase}",'] if phase != "whole" else []),
        f'  mx.source_sha256 = "{source}",',
        f'  mx.source_mlir_sha256 = "{binding["frontend_mlir_sha256"]}",',
        f'  mx.profile_sha256 = "{digest}",',
        f'  mx.contract_sha256 = "{binding["contract_sha256"]}",',
        f'  mx.policy_sha256 = "{binding["policy_sha256"]}",',
        f'  prov.quantization_manifest_sha256 = "{binding["manifest_sha256"]}"',
        "} {",
        f"  func.func @mx_issue({args}) {{",
    ]
    setup = [
        memory("load_scales", "Q_SCALES",
               f'payload_bytes = {buffer_bytes["Q_SCALES"]} : i32, '
               'scale_target = "activation"',
               "scales:q"),
        memory("load_scales", "KT_SCALES",
               f'payload_bytes = {buffer_bytes["KT_SCALES"]} : i32, '
               'scale_target = "weight"',
               "scales:kt"),
        memory("scale_dma", "V_SCALES",
               f"row_bytes = {shape.d} : i32, rows = {shape.sk // 32} : i32, "
               f"pitch_bytes = {shape.d} : i32, destination_byte = 4096 : i32, "
               "selector = 1 : i32", "scales:v"),
    ]
    if phase in {"whole", "setup"}:
        lines.extend(setup)
    if phase == "whole":
        lines.append(
            '    "mx_gemmini.wait"() {site_id = "scales:fence", '
            f'profile_sha256 = "{digest}", '
            f'contract_sha256 = "{binding["contract_sha256"]}", '
            f'policy_sha256 = "{binding["policy_sha256"]}", '
            f'manifest_sha256 = "{binding["manifest_sha256"]}"'
            '} : () -> ()')
    if phase in {"whole", "check", "timed"}:
        lines.extend([
            memory("dma_matrix", "Q_IN",
                   f"matrix_rows = {shape.sq} : i32, matrix_cols = {shape.d} : i32, "
                   "burst_cols = 16 : i32, spad_row = 0 : i32", "operand:q"),
            memory("dma_matrix", "KT_IN",
                   f"matrix_rows = {shape.d} : i32, matrix_cols = {shape.sk} : i32, "
                   f"burst_cols = 16 : i32, spad_row = {shape.kt_row} : i32", "operand:kt"),
            memory("dma_matrix", "V_IN",
                   f"matrix_rows = {shape.sk} : i32, matrix_cols = {shape.d} : i32, "
                   f"burst_cols = 16 : i32, spad_row = {shape.v_row} : i32", "operand:v"),
            matmul("matmul_0", m=shape.sq, n=shape.sk, k=shape.d,
                   a=0, b_end=shape.score_row, c=shape.score_row, half=0),
        ])
    if phase in {"whole", "check"}:
        lines.append(
            memory("spad_mvout_linear", "S_hw",
                   f"total_bytes = {buffer_bytes['S_hw']} : i32, tile_cols = 16 : i32, "
                   f"spad_row = {shape.score_row} : i32", "score:readout"))
    if phase in {"whole", "check", "timed"}:
        lines.extend([
            *vector_ops,
            matmul("matmul_1", m=shape.sq, n=shape.d, k=shape.sk,
                   a=shape.probability_row, b_end=shape.output_row,
                   c=shape.output_row, half=1),
            memory("spad_mvout_linear", "O_hw",
                   f"total_bytes = {buffer_bytes['O_hw']} : i32, tile_cols = 16 : i32, "
                   f"spad_row = {shape.output_row} : i32", "output:readout"),
        ])
    if phase in {"whole", "check"}:
        lines.extend([
            memory("spad_mvout_linear", "P_hw",
                   f"total_bytes = {buffer_bytes['P_hw']} : i32, tile_cols = 16 : i32, "
                   f"spad_row = {shape.score_row} : i32", "probability:readout"),
            memory("spad_mvout_linear", "p_codes_hw",
                   f"total_bytes = {buffer_bytes['p_codes_hw']} : i32, "
                   f"tile_cols = 16 : i32, spad_row = {shape.probability_row} : i32",
                   "probability_codes:readout"),
        ])
    lines.extend(("    func.return", "  }", "}", ""))
    result = "\n".join(lines)
    report = verify_ir(result, profile)
    expected = {"whole": (10, 2, 7, 1), "setup": (3, 0, 0, 0),
                "check": (7, 2, 7, 1), "timed": (4, 2, 7, 1)}
    if (report["memory_phases"], report["spad_matmuls"],
            report["vpu_commands"], report["spad_requants"]) != expected[phase]:
        raise ValueError("dense attention typed stage inventory differs")
    return result


def lower_dense_attention_typed_program(mlir_text: str, vector_mlir: str,
                                        profile: dict, binding: dict
                                        ) -> tuple[Command | Fence, ...]:
    """Lower the exact checked graph through reusable MX physical lowerers."""
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func, FuncOp, ReturnOp
    from xdsl.parser import Parser

    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    actual_module = Parser(context, mlir_text).parse_module()
    program_phase = _text_attr(actual_module, "mx.attention_phase") or "whole"
    if program_phase not in {"whole", "setup", "check", "timed"}:
        raise ValueError("typed attention has an unsupported issue phase")
    expected = render_dense_attention_program(vector_mlir, profile, binding,
                                              phase=program_phase)
    report = verify_ir(mlir_text, profile)
    inventory = {"whole": (10, 2, 7, 1), "setup": (3, 0, 0, 0),
                 "check": (7, 2, 7, 1), "timed": (4, 2, 7, 1)}
    if (report["memory_phases"], report["spad_matmuls"],
            report["vpu_commands"], report["spad_requants"]) != inventory[program_phase]:
        raise ValueError("typed attention stage inventory differs")
    expected_module = Parser(context, expected).parse_module()
    def attributes(op) -> dict[str, str]:
        return {name: str(value) for name, value in op.attributes.items()
                if name != "op_name__"}
    if attributes(actual_module) != attributes(expected_module):
        raise ValueError("typed attention module binding differs")
    functions = [op for op in actual_module.walk() if isinstance(op, FuncOp)]
    if (len(functions) != 1 or len(actual_module.body.block.ops) != 1 or
            len(functions[0].body.blocks) != 1):
        raise ValueError("typed attention needs one function")
    actual_function, expected_function = functions[0], next(
        op for op in expected_module.walk() if isinstance(op, FuncOp))
    actual_args = list(actual_function.body.block.args)
    expected_args = list(expected_function.body.block.args)
    actual_ops = list(actual_function.body.block.ops)
    expected_ops = list(expected_function.body.block.ops)
    if (attributes(actual_function) != attributes(expected_function) or
            [str(arg.type) for arg in actual_args] !=
            [str(arg.type) for arg in expected_args] or
            len(actual_ops) != len(expected_ops)):
        raise ValueError("typed attention ABI or operation count differs")
    for actual, selected in zip(actual_ops, expected_ops):
        if (_operation_name(actual) != _operation_name(selected) or
                attributes(actual) != attributes(selected) or
                [actual_args.index(operand) for operand in actual.operands] !=
                [expected_args.index(operand) for operand in selected.operands] or
                actual.results or actual.regions):
            raise ValueError("typed attention operation or buffer binding differs")
    if not isinstance(actual_ops[-1], ReturnOp):
        raise ValueError("typed attention must return after readout")

    vector = iter(lower_dense_attention_vector(vector_mlir, profile, binding))
    commands: list[Command | Fence] = []
    for op in actual_ops[:-1]:
        name = _operation_name(op)
        buffer = (ATTENTION_BUFFERS[actual_args.index(op.operands[0])]
                  if op.operands else None)
        if name == "mx_gemmini.load_scales":
            memory_phase = lower_scale_load(
                profile, buffer=buffer, payload_bytes=_int_attr(op, "payload_bytes"),
                operand=_text_attr(op, "scale_target"))
            commands.extend(memory_phase.commands)
        elif name == "mx_gemmini.scale_dma":
            memory_phase = lower_strided_scale_dma(
                profile, buffer=buffer, row_bytes=_int_attr(op, "row_bytes"),
                rows=_int_attr(op, "rows"), pitch_bytes=_int_attr(op, "pitch_bytes"),
                destination_byte=_int_attr(op, "destination_byte"),
                selector=_int_attr(op, "selector"))
            commands.extend(memory_phase.commands)
        elif name == "mx_gemmini.dma_matrix":
            memory_phase = lower_matrix_mvin(
                profile, buffer=buffer, matrix_rows=_int_attr(op, "matrix_rows"),
                matrix_cols=_int_attr(op, "matrix_cols"),
                burst_cols=_int_attr(op, "burst_cols"),
                spad_row=_int_attr(op, "spad_row"))
            commands.extend(memory_phase.commands)
        elif name == "mx_gemmini.wait":
            commands.append(Fence())
        elif name == "mx_gemmini.matmul_spad":
            commands.extend(lower_spad_matmul(
                profile, m=_int_attr(op, "m"), n=_int_attr(op, "n"),
                k=_int_attr(op, "k"), a_row=_int_attr(op, "activation_row"),
                b_end_row=_int_attr(op, "weight_end_row"),
                output_row=_int_attr(op, "output_row"),
                weight_half=_int_attr(op, "weight_half"),
                scale_sink=_text_attr(op, "scale_sink_buffer")))
        elif name in ("mx_gemmini.vpu_execute", "mx_gemmini.spad_requant"):
            commands.append(next(vector))
        elif name == "mx_gemmini.spad_mvout_linear":
            memory_phase = lower_linear_spad_mvout(
                profile, buffer=buffer, total_bytes=_int_attr(op, "total_bytes"),
                tile_cols=_int_attr(op, "tile_cols"),
                spad_row=_int_attr(op, "spad_row"))
            commands.extend(memory_phase.commands)
        else:
            raise ValueError(f"typed attention has no lowering for {name}")
    result = tuple(commands)
    if result != lower_dense_attention_program(vector_mlir, profile, binding,
                                               phase=program_phase):
        raise ValueError("typed attention physical stream differs from source-bound program")
    return result
