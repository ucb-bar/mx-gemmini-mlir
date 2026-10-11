"""Lower ordered, source-bound MX flash operations to one physical command IR."""

from __future__ import annotations

from .command_ir import (Command, Fence, spad_requant_command, vpu_command)
from .memory_phase import (lower_linear_spad_mvout, lower_matrix_mvin,
                           lower_strided_scale_dma)
from .nicolas_attention_matrix import (lower_flash_setup,
                                       lower_managed_spad_matmul)
from .verify_profile_ir import (_bool_attr, _int_attr, _operation_name,
                                _text_attr, verify_ir)


SCHEMA = "mx_gemmini.nicolas_flash_attention_program.v1"


def lower_flash_commands(mlir_text: str, profile: dict,
                         buffers: tuple[str, ...], buffer_bytes: tuple[int, ...]
                         ) -> tuple[Command | Fence, ...]:
    """Lower a checked flat flash graph while enforcing gated-half ownership."""
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func, FuncOp, ReturnOp
    from xdsl.parser import Parser

    if len(buffers) != len(buffer_bytes) or len(set(buffers)) != len(buffers):
        raise ValueError("flash graph needs a unique runtime buffer ABI")
    verify_ir(mlir_text, profile)
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    if _text_attr(module, "mx.attention_program_schema") != SCHEMA:
        raise ValueError("flash graph schema differs")
    functions = [op for op in module.walk() if isinstance(op, FuncOp)]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("flash graph needs one flat function")
    args = list(functions[0].body.block.args)
    if [str(arg.type) for arg in args] != [
            f"memref<{length}xi8>" for length in buffer_bytes]:
        raise ValueError("flash graph buffer ABI differs")
    ops = list(functions[0].body.block.ops)
    if not ops or not isinstance(ops[-1], ReturnOp) or ops[-1].operands:
        raise ValueError("flash graph needs a terminal return")
    commands: list[Command | Fence] = []
    pending_weight_halves: set[int] = set()
    half_bytes = profile["resources"]["scale_mem_config"]["size_bytes"] // 4
    for op in ops[:-1]:
        name = _operation_name(op)
        if op.results or op.regions:
            raise ValueError("flash physical operations may not hide SSA results or regions")
        buffer = (buffers[args.index(op.operands[0])] if len(op.operands) == 1
                  and op.operands[0] in args else None)
        if name in ("mx_gemmini.scale_dma", "mx_gemmini.gated_scale_dma"):
            if buffer is None:
                raise ValueError("flash scale DMA needs a runtime argument")
            gated = name == "mx_gemmini.gated_scale_dma"
            destination = _int_attr(op, "destination_byte")
            phase = lower_strided_scale_dma(
                profile, buffer=buffer, row_bytes=_int_attr(op, "row_bytes"),
                rows=_int_attr(op, "rows"), pitch_bytes=_int_attr(op, "pitch_bytes"),
                destination_byte=destination, selector=_int_attr(op, "selector"),
                source_offset_bytes=(_int_attr(op, "source_offset_bytes")
                                     if "source_offset_bytes" in op.attributes else 0),
                gated=gated)
            if gated:
                half = destination // half_bytes
                if half in pending_weight_halves:
                    raise ValueError("flash weight scale half is loaded twice before use")
                pending_weight_halves.add(half)
            commands.extend(phase.commands)
        elif name == "mx_gemmini.wait":
            if op.operands:
                raise ValueError("flash wait takes no buffer")
            commands.append(Fence())
        elif name == "mx_gemmini.dma_matrix":
            if buffer is None:
                raise ValueError("flash matrix DMA needs a runtime argument")
            commands.extend(lower_matrix_mvin(
                profile, buffer=buffer, matrix_rows=_int_attr(op, "matrix_rows"),
                matrix_cols=_int_attr(op, "matrix_cols"),
                burst_cols=_int_attr(op, "burst_cols"),
                spad_row=_int_attr(op, "spad_row"),
                source_stride_bytes=(_int_attr(op, "source_stride_bytes")
                                     if "source_stride_bytes" in op.attributes else None),
                source_offset_bytes=(_int_attr(op, "source_offset_bytes")
                                     if "source_offset_bytes" in op.attributes else 0)).commands)
        elif name == "mx_gemmini.flash_setup":
            if op.operands:
                raise ValueError("flash setup takes no buffer")
            commands.extend(lower_flash_setup(
                profile, store_stride_bytes=_int_attr(op, "store_stride_bytes")))
        elif name == "mx_gemmini.managed_matmul_spad":
            if op.operands:
                raise ValueError("flash managed matmul takes no buffer")
            half = _int_attr(op, "weight_half")
            if half not in pending_weight_halves:
                raise ValueError("flash managed matmul lacks its gated weight-scale load")
            pending_weight_halves.remove(half)
            sink = _text_attr(op, "scale_sink_buffer")
            if sink not in buffers:
                raise ValueError("flash managed matmul scale sink is unbound")
            commands.extend(lower_managed_spad_matmul(
                profile, m=_int_attr(op, "m"), n=_int_attr(op, "n"),
                k=_int_attr(op, "k"), a_row=_int_attr(op, "activation_row"),
                b_end_row=_int_attr(op, "weight_end_row"),
                output_row=_int_attr(op, "output_row"),
                weight_half=half, activation_half=_int_attr(op, "activation_half"),
                scale_sink=sink))
        elif name == "mx_gemmini.vpu_execute":
            if op.operands:
                raise ValueError("flash VPU operation takes physical rows")
            commands.append(vpu_command(
                profile, kind=_text_attr(op, "kind"),
                src1_row=_int_attr(op, "src1_row"),
                src2_row=_int_attr(op, "src2_row"),
                dst_row=_int_attr(op, "dst_row"), rows=_int_attr(op, "rows"),
                reduction_length=_int_attr(op, "reduction_length"),
                broadcast=_bool_attr(op, "broadcast"),
                immediate_bf16=_int_attr(op, "immediate_bf16"),
                second_dst_row=_int_attr(op, "second_dst_row")))
        elif name == "mx_gemmini.spad_requant":
            if op.operands:
                raise ValueError("flash requant takes physical rows")
            scale_buffer = _text_attr(op, "scale_buffer")
            if scale_buffer not in buffers:
                raise ValueError("flash requant scale buffer is unbound")
            commands.append(spad_requant_command(
                profile, source_row=_int_attr(op, "source_row"),
                destination_row=_int_attr(op, "destination_row"),
                m=_int_attr(op, "m"), n=_int_attr(op, "n"),
                output_format=_text_attr(op, "output_format"),
                tiled=_bool_attr(op, "tiled"), resident=_bool_attr(op, "resident"),
                scale_dram_address=_int_attr(op, "scale_dram_address"),
                scale_buffer=scale_buffer))
        elif name == "mx_gemmini.spad_mvout_linear":
            if buffer is None:
                raise ValueError("flash readout needs a runtime argument")
            commands.extend(lower_linear_spad_mvout(
                profile, buffer=buffer, total_bytes=_int_attr(op, "total_bytes"),
                tile_cols=_int_attr(op, "tile_cols"),
                spad_row=_int_attr(op, "spad_row")).commands)
        else:
            raise ValueError(f"flash graph has no lowering for {name}")
    if pending_weight_halves:
        raise ValueError("flash graph has an unused gated weight-scale load")
    return tuple(commands)
