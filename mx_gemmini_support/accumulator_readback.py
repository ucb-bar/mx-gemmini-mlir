"""Source-checked accumulator DRAM readout for Nicolas's MX experiments.

The pinned Spike extension cannot execute this destination. This lowering
produces hardware-directed RoCC commands, but deliberately makes no numerical
or timing claim about the resulting object.
"""

from __future__ import annotations

from io import StringIO

from .chunked_i import _parse_module
from .command_ir import Command, Fence, Operand
from .physical_program import PhysicalProgram, PhysicalStep
from .source_payload import manifest_sha256
from .target_profile import profile_sha256
from .verify_profile_ir import _operation_name, _text_attr, verify_ir


_ATTR = "mx.accumulator_dram_readout"
_ACC_BASE = 1 << 31
_DIM = 16

# Exact source hashes and tile choices of the three conditional DRAMMvout
# programs. The FP8 128x128x256 contraction uses two compiler K waves.
SOURCES = {
    "0c866be0565b1ea0c8f675bdf0468123da53acfd3e130dcb6d9aecc8c95d86a4":
        ("FP8", (64, 64, 64), (64, 64, 64), "fp8_64"),
    "c1729048ccab7836598b55bdb747d6c75489d792683abd300c10368627301155":
        ("FP4", (64, 64, 64), (64, 64, 64), "fp4_64"),
    "2d87a0a987c90f7855e69ae46e3429e490a6d4661a63f3d6d4d3043b777dd689":
        ("FP8", (128, 128, 256), (128, 128, 128), "fp8_128x256"),
}


def _check_source(profile: dict, manifest: dict) -> tuple[str, str, tuple[int, int, int]]:
    source_hash = manifest.get("source_driver_sha256")
    selected = SOURCES.get(source_hash)
    if selected is None:
        raise ValueError("accumulator readout needs a pinned DRAMMvout source")
    precision, shape, tile, label = selected
    if (manifest.get("origin") != "nicolas_source_header_specialization" or
            manifest.get("precision") != precision or
            manifest.get("output_format") is not None or
            manifest.get("shape_mnk") != list(shape) or
            manifest.get("tile_mnk") != list(tile) or
            manifest.get("profile_sha256") != profile_sha256(profile) or
            profile.get("name") != "MxGemminiRocketConfig" or
            profile.get("transport") != "rocket_rocc" or
            profile["geometry"]["mesh_columns"] != _DIM or
            profile["resources"].get("accumulator_bytes", 0) < shape[0] * shape[1] * 2):
        raise ValueError("accumulator readout differs from source/profile capacity")
    return label, source_hash, shape


def bind_accumulator_dram_readout(mlir_text: str, profile: dict, manifest: dict) -> str:
    """Select the accumulator route on one already payload-bound BF16 graph."""
    from xdsl.dialects.builtin import StringAttr
    from xdsl.printer import Printer

    label, _, _ = _check_source(profile, manifest)
    verify_ir(mlir_text, profile)
    module = _parse_module(mlir_text)
    readout = [op for op in module.walk()
               if _operation_name(op) == "mx_gemmini.readout_bf16"]
    if (_text_attr(module, "mx.payload_manifest_sha256") != manifest_sha256(manifest) or
            any(_text_attr(module, name) is not None for name in
                (_ATTR, "mx.smem_zero_readout", "mx.i_chunks", "mx.native_dram_loop")) or
            len(readout) != 1 or
            _text_attr(readout[0], "source_memory") is not None or
            [_operation_name(op) for op in module.walk()
             if _operation_name(op).startswith("mx_gemmini.") and
             _operation_name(op) != "mx_gemmini.resource"] != [
                "mx_gemmini.contract", "mx_gemmini.readout_bf16"]):
        raise ValueError("accumulator readout needs one unselected BF16 contraction")
    module.attributes[_ATTR] = StringAttr(label)
    readout[0].attributes["source_memory"] = StringAttr("accumulator")
    stream = StringIO()
    Printer(stream=stream).print_op(module)
    rendered = stream.getvalue() + "\n"
    verify_ir(rendered, profile)
    return rendered


def selected_accumulator_dram_readout(mlir_text: str, profile: dict,
                                      manifest: dict) -> bool:
    module = _parse_module(mlir_text)
    label = _text_attr(module, _ATTR)
    if label is None:
        return False
    readout = [op for op in module.walk()
               if _operation_name(op) == "mx_gemmini.readout_bf16"]
    expected, _, _ = _check_source(profile, manifest)
    if (label != expected or
            _text_attr(module, "mx.payload_manifest_sha256") !=
            manifest_sha256(manifest) or
            len(readout) != 1 or
            _text_attr(readout[0], "source_memory") != "accumulator" or
            any(_text_attr(module, name) is not None for name in
                ("mx.smem_zero_readout", "mx.i_chunks", "mx.native_dram_loop"))):
        raise ValueError("accumulator readout is not bound to selected payload")
    return True


def lower_accumulator_dram_readout(base: PhysicalProgram, profile: dict,
                                   manifest: dict) -> PhysicalProgram:
    """Route C to accumulator and encode source-shaped accumulator MVOUTs."""
    label, source_hash, shape = _check_source(profile, manifest)
    m, n, k = shape
    precision = manifest["precision"]
    plan = base.plan
    if (base.shape != shape or base.output_format != "bf16" or
            base.mode != "spike_serial" or
            plan.get("output_tiles") or
            plan.get("bf16_output_layout") not in {None, "row_major_bf16"} or
            plan.get("acc_to_gmem") or
            plan.get("c_rows") != m * n * 2 // _DIM or
            len(plan.get("waves", [])) != k // manifest["tile_mnk"][2] or
            any(step.phase in {"vpu", "spad_requant"} for step in base.steps)):
        raise ValueError("accumulator readout needs one checked BF16 source tile")

    # Nicolas's hardware branches use a 64-bit out_t with four BF16 lanes.
    # Their MVOUT loops group four J tiles for FP8 and use one J tile for FP4.
    tile_i = m // (_DIM * (2 if precision == "FP4" else 1))
    tile_j = n // (_DIM * (2 if precision == "FP4" else 1))
    group_j = tile_j // (4 if precision == "FP8" else 1)
    if group_j < 1 or (precision == "FP8" and tile_j % 4):
        raise ValueError("accumulator readout has incomplete source J groups")
    source_stride = n * (4 if precision == "FP4" else 2)
    old_dest = plan["c_spad_dest"]
    readout_commands = []
    for i in range(tile_i):
        for j in range(group_j):
            offset = i * _DIM * (2 if precision == "FP4" else 1) * n * 2 + j * n
            if offset >= m * n * 2:
                raise ValueError("accumulator MVOUT pointer exceeds BF16 output")
            acc_row = _ACC_BASE + (i * group_j + j) * _DIM
            readout_commands.append(Command(
                3, Operand(buffer="output_bf16", byte_offset=offset),
                Operand(immediate=(_DIM << 48) | (_DIM << 32) | acc_row)))
    if len(readout_commands) * _DIM * _DIM * 2 * 4 != m * n * 2:
        raise ValueError("accumulator MVOUT groups do not cover source output size")

    steps = []
    config_st = compute = readouts = 0
    for step in base.steps:
        command = step.command
        if isinstance(command, Command):
            if (step.phase == "configure" and command.funct == 0 and
                    command.rs1.immediate == 2):
                if command.rs2.immediate != n * 2:
                    raise ValueError("accumulator source store configuration changed")
                command = Command(0, command.rs1, Operand(immediate=source_stride))
                config_st += 1
            elif step.phase == "compute" and command.funct == 8:
                if (command.rs2.immediate is None or
                        command.rs2.immediate >> 32 != old_dest or
                        command.rs2.immediate & 0xffffffff not in {0x238, 0x2b8}):
                    raise ValueError("accumulator source compute destination changed")
                command = Command(8, command.rs1,
                                  Operand(immediate=(_ACC_BASE << 32) | 0x2b8))
                compute += 1
            elif step.phase == "readout":
                if command.funct == 0 and command.rs1.immediate == 2:
                    continue
                if command.funct != 3:
                    raise ValueError("accumulator source readout command changed")
                readouts += 1
                continue
        if step.phase == "readout" and isinstance(command, Fence):
            steps.extend(PhysicalStep("readout", None, cmd)
                         for cmd in readout_commands)
        steps.append(PhysicalStep(step.phase, step.wave, command))
    if (config_st != 1 or compute != len(plan["waves"]) or
            readouts != plan["c_rows"] // _DIM):
        raise ValueError("accumulator source needs one config, all waves, and full old readout")
    return PhysicalProgram(
        base.profile_sha256, base.payload_manifest_sha256, "rtl_accumulator",
        base.shape,
        {**plan, "acc_to_gmem": True, "move_out": "acc_dma",
         "waves": [{**wave, "move_acc_to_spad": False}
                   for wave in plan["waves"]],
         "bf16_output_layout": "row_major_bf16",
         "readout_transport": "rocket_rocc_accumulator_mvout",
         "readout_source_sha256": source_hash,
         "readout_source_mode": label,
         "scale_loading": "source_header_rocc_2d_not_nicolas_mmio_constant",
         "hardware_numerical_qualification": "unqualified"},
        tuple(steps), False, None, base.output_format,
        base.tiled_quant_readout, base.derived_vpu_scalar_bf16,
        base.derived_vpu_scalar_chain, "unqualified_hardware_path")
