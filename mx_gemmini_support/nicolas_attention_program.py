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
from .nicolas_attention_vector import lower_dense_attention_vector
from .target_profile import profile_sha256
from .verify_profile_ir import _int_attr, _operation_name, _text_attr, verify_ir


BUFFER_BYTES = {
    "Q_SCALES": 64, "KT_SCALES": 128, "V_SCALES": 128,
    "Q_IN": 2048, "KT_IN": 4096, "V_IN": 4096,
    "scale_sink": 2048, "S_hw": 4096, "p_scales": 64,
    "O_hw": 4096, "P_hw": 4096, "p_codes_hw": 2048,
}
SCHEMA = "mx_gemmini.nicolas_dense_attention_program.v1"


def render_dense_attention_program(vector_mlir: str, profile: dict,
                                   binding: dict) -> str:
    """Wrap the checked vector middle in ordered typed MX memory/matmul ops."""
    vector = lower_dense_attention_vector(vector_mlir, profile, binding)
    if len(vector) != 8:
        raise ValueError("dense attention needs the checked vector middle")
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
                f'(memref<{BUFFER_BYTES[buffer]}xi8>) -> ()')

    def matmul(site: str, *, a: int, b_end: int, c: int, half: int) -> str:
        return (f'    "mx_gemmini.matmul_spad"() {{site_id = "{site}", '
                f'm = 32 : i32, n = 64 : i32, k = 64 : i32, '
                f'activation_row = {a} : i32, weight_end_row = {b_end} : i32, '
                f'output_row = {c} : i32, weight_half = {half} : i32, '
                f'scale_sink_buffer = "scale_sink", {common}}} : () -> ()')

    args = ", ".join(f"%{name}: memref<{BUFFER_BYTES[name]}xi8>"
                     for name in ATTENTION_BUFFERS)
    lines = [
        "builtin.module attributes {",
        f'  mx.attention_program_schema = "{SCHEMA}",',
        f'  mx.source_sha256 = "{source}",',
        f'  mx.source_mlir_sha256 = "{binding["frontend_mlir_sha256"]}",',
        f'  mx.profile_sha256 = "{digest}",',
        f'  mx.contract_sha256 = "{binding["contract_sha256"]}",',
        f'  mx.policy_sha256 = "{binding["policy_sha256"]}",',
        f'  prov.quantization_manifest_sha256 = "{binding["manifest_sha256"]}"',
        "} {",
        f"  func.func @mx_issue({args}) {{",
        memory("load_scales", "Q_SCALES",
               'payload_bytes = 64 : i32, scale_target = "activation"',
               "scales:q"),
        memory("load_scales", "KT_SCALES",
               'payload_bytes = 128 : i32, scale_target = "weight"',
               "scales:kt"),
        memory("scale_dma", "V_SCALES",
               "row_bytes = 64 : i32, rows = 2 : i32, pitch_bytes = 64 : i32, "
               "destination_byte = 4096 : i32, selector = 1 : i32", "scales:v"),
        ('    "mx_gemmini.wait"() {site_id = "scales:fence", '
         f'profile_sha256 = "{digest}", '
         f'contract_sha256 = "{binding["contract_sha256"]}", '
         f'policy_sha256 = "{binding["policy_sha256"]}", '
         f'manifest_sha256 = "{binding["manifest_sha256"]}"'
         '} : () -> ()'),
        memory("dma_matrix", "Q_IN",
               "matrix_rows = 32 : i32, matrix_cols = 64 : i32, "
               "burst_cols = 16 : i32, spad_row = 0 : i32", "operand:q"),
        memory("dma_matrix", "KT_IN",
               "matrix_rows = 64 : i32, matrix_cols = 64 : i32, "
               "burst_cols = 16 : i32, spad_row = 128 : i32", "operand:kt"),
        memory("dma_matrix", "V_IN",
               "matrix_rows = 64 : i32, matrix_cols = 64 : i32, "
               "burst_cols = 16 : i32, spad_row = 768 : i32", "operand:v"),
        matmul("matmul_0", a=0, b_end=384, c=384, half=0),
        memory("spad_mvout_linear", "S_hw",
               "total_bytes = 4096 : i32, tile_cols = 16 : i32, "
               "spad_row = 384 : i32", "score:readout"),
        *vector_ops,
        matmul("matmul_1", a=640, b_end=1024, c=1024, half=1),
        memory("spad_mvout_linear", "O_hw",
               "total_bytes = 4096 : i32, tile_cols = 16 : i32, "
               "spad_row = 1024 : i32", "output:readout"),
        memory("spad_mvout_linear", "P_hw",
               "total_bytes = 4096 : i32, tile_cols = 16 : i32, "
               "spad_row = 384 : i32", "probability:readout"),
        memory("spad_mvout_linear", "p_codes_hw",
               "total_bytes = 2048 : i32, tile_cols = 16 : i32, "
               "spad_row = 640 : i32", "probability_codes:readout"),
        "    func.return", "  }", "}", "",
    ]
    result = "\n".join(lines)
    report = verify_ir(result, profile)
    if (report["memory_phases"], report["spad_matmuls"],
            report["vpu_commands"], report["spad_requants"]) != (10, 2, 7, 1):
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

    expected = render_dense_attention_program(vector_mlir, profile, binding)
    report = verify_ir(mlir_text, profile)
    if (report["memory_phases"], report["spad_matmuls"],
            report["vpu_commands"], report["spad_requants"]) != (10, 2, 7, 1):
        raise ValueError("typed attention stage inventory differs")
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    actual_module = Parser(context, mlir_text).parse_module()
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
            phase = lower_scale_load(
                profile, buffer=buffer, payload_bytes=_int_attr(op, "payload_bytes"),
                operand=_text_attr(op, "scale_target"))
            commands.extend(phase.commands)
        elif name == "mx_gemmini.scale_dma":
            phase = lower_strided_scale_dma(
                profile, buffer=buffer, row_bytes=_int_attr(op, "row_bytes"),
                rows=_int_attr(op, "rows"), pitch_bytes=_int_attr(op, "pitch_bytes"),
                destination_byte=_int_attr(op, "destination_byte"),
                selector=_int_attr(op, "selector"))
            commands.extend(phase.commands)
        elif name == "mx_gemmini.dma_matrix":
            phase = lower_matrix_mvin(
                profile, buffer=buffer, matrix_rows=_int_attr(op, "matrix_rows"),
                matrix_cols=_int_attr(op, "matrix_cols"),
                burst_cols=_int_attr(op, "burst_cols"),
                spad_row=_int_attr(op, "spad_row"))
            commands.extend(phase.commands)
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
            phase = lower_linear_spad_mvout(
                profile, buffer=buffer, total_bytes=_int_attr(op, "total_bytes"),
                tile_cols=_int_attr(op, "tile_cols"),
                spad_row=_int_attr(op, "spad_row"))
            commands.extend(phase.commands)
        else:
            raise ValueError(f"typed attention has no lowering for {name}")
    result = tuple(commands)
    if result != lower_dense_attention_program(vector_mlir, profile, binding):
        raise ValueError("typed attention physical stream differs from source-bound program")
    return result
