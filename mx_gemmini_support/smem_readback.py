"""Source-bound zero-base scratchpad readout for Nicolas's FP8 64³ test."""

from __future__ import annotations

from io import StringIO

from .chunked_i import _parse_module
from .command_ir import Command, Operand
from .physical_program import PhysicalProgram, PhysicalStep
from .source_payload import manifest_sha256
from .target_profile import profile_sha256
from .verify_profile_ir import _operation_name, _text_attr, verify_ir


SOURCE_SHA256 = "7c3b43b252143a3495ecada4e6d23cfaccae6fa07e5922ebc02cc5737345eb41"
_ATTR = "mx.smem_zero_readout"


def _check_source(profile: dict, manifest: dict) -> None:
    if (manifest.get("origin") != "nicolas_source_header_specialization" or
            manifest.get("source_driver_sha256") != SOURCE_SHA256 or
            manifest.get("precision") != "FP8" or
            manifest.get("output_format") is not None or
            manifest.get("shape_mnk") != [64, 64, 64] or
            manifest.get("tile_mnk") != [64, 64, 64] or
            manifest.get("profile_sha256") != profile_sha256(profile) or
            profile.get("name") != "MxGemminiRocketConfig" or
            profile.get("transport") != "rocket_rocc" or
            profile["geometry"]["mesh_columns"] != 16):
        raise ValueError("zero-base scratchpad readout differs from pinned source/profile")


def bind_smem_zero_readout(mlir_text: str, profile: dict, manifest: dict) -> str:
    """Bind the checked scratchpad destination after payload specialization."""
    from xdsl.dialects.builtin import StringAttr
    from xdsl.printer import Printer

    _check_source(profile, manifest)
    verify_ir(mlir_text, profile)
    module = _parse_module(mlir_text)
    if (_text_attr(module, "mx.payload_manifest_sha256") != manifest_sha256(manifest) or
            any(_text_attr(module, name) is not None for name in
                (_ATTR, "mx.i_chunks", "mx.native_dram_loop")) or
            [_operation_name(op) for op in module.walk()
             if _operation_name(op).startswith("mx_gemmini.") and
             _operation_name(op) != "mx_gemmini.resource"] != [
                "mx_gemmini.contract", "mx_gemmini.readout_bf16"]):
        raise ValueError("zero-base scratchpad readout needs one source-bound contraction")
    module.attributes[_ATTR] = StringAttr("source64")
    stream = StringIO()
    Printer(stream=stream).print_op(module)
    rendered = stream.getvalue() + "\n"
    verify_ir(rendered, profile)
    return rendered


def selected_smem_zero_readout(mlir_text: str, profile: dict,
                                manifest: dict) -> bool:
    module = _parse_module(mlir_text)
    selected = _text_attr(module, _ATTR)
    if selected is None:
        return False
    if (selected != "source64" or
            _text_attr(module, "mx.payload_manifest_sha256") !=
            manifest_sha256(manifest) or
            any(_text_attr(module, name) is not None for name in
                ("mx.i_chunks", "mx.native_dram_loop"))):
        raise ValueError("zero-base scratchpad readout is not bound to selected payload")
    _check_source(profile, manifest)
    return True


def lower_smem_zero_readout(base: PhysicalProgram, profile: dict,
                             manifest: dict) -> PhysicalProgram:
    """Store C at scratchpad row zero and read it out through RoCC MVOUT."""
    _check_source(profile, manifest)
    if (base.shape != (64, 64, 64) or base.output_format != "bf16" or
            base.mode != "spike_serial" or not base.source_golden_preserving or
            base.derived_expected_bf16 is not None or
            base.plan.get("c_spad_dest") != 256 or
            base.plan.get("c_rows") != 512 or
            len(base.plan.get("waves", [])) != 1):
        raise ValueError("zero-base readout needs the checked single-wave FP8 plan")
    steps = []
    config_st = compute = readouts = 0
    for step in base.steps:
        command = step.command
        if not isinstance(command, Command):
            steps.append(step)
            continue
        if (step.phase == "configure" and command.funct == 0 and
                command.rs1.immediate == 2):
            if command.rs2.immediate != 128:
                raise ValueError("zero-base source store configuration changed")
            command = Command(0, command.rs1, Operand(immediate=16))
            config_st += 1
        elif step.phase == "compute" and command.funct == 8:
            if (command.rs2.immediate is None or
                    command.rs2.immediate >> 32 != 256 or
                    command.rs2.immediate & 0xffffffff != 0x238):
                raise ValueError("zero-base source compute destination changed")
            command = Command(8, command.rs1, Operand(immediate=0x238))
            compute += 1
        elif step.phase == "readout" and command.funct == 3:
            if (command.rs2.immediate is None or
                    command.rs2.immediate & 0xffffffff < 256 or
                    command.rs2.immediate & 0xffffffff >= 768):
                raise ValueError("zero-base source readout row changed")
            command = Command(3, command.rs1, Operand(
                immediate=command.rs2.immediate - 256))
            readouts += 1
        steps.append(PhysicalStep(step.phase, step.wave, command))
    if (config_st, compute, readouts) != (1, 1, 32):
        raise ValueError("zero-base source requires one store, compute, and 32 readouts")
    return PhysicalProgram(
        base.profile_sha256, base.payload_manifest_sha256, base.mode, base.shape,
        {**base.plan, "c_spad_dest": 0,
         "readout_transport": "spad_mvout_zero_base",
         "readout_source_sha256": SOURCE_SHA256},
        tuple(steps), base.source_golden_preserving, base.derived_expected_bf16,
        base.output_format, base.tiled_quant_readout,
        base.derived_vpu_scalar_bf16, base.derived_vpu_scalar_chain,
        base.golden_origin, base.golden_derivation)
