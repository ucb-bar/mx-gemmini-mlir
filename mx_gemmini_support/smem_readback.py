"""Source-bound zero-base scratchpad readout for Nicolas's Spike test paths."""

from __future__ import annotations

from io import StringIO

from .chunked_i import _parse_module
from .command_ir import Command, Operand
from .physical_program import PhysicalProgram, PhysicalStep
from .source_payload import manifest_sha256
from .target_profile import profile_sha256
from .verify_profile_ir import _operation_name, _text_attr, verify_ir


SOURCE_SHA256 = "7c3b43b252143a3495ecada4e6d23cfaccae6fa07e5922ebc02cc5737345eb41"
SOURCES = {
    SOURCE_SHA256: ("FP8", (64, 64, 64), (64, 64, 64), "source64"),
    "0c866be0565b1ea0c8f675bdf0468123da53acfd3e130dcb6d9aecc8c95d86a4":
        ("FP8", (64, 64, 64), (64, 64, 64), "dram64_fp8_spike"),
    "c1729048ccab7836598b55bdb747d6c75489d792683abd300c10368627301155":
        ("FP4", (64, 64, 64), (64, 64, 64), "dram64_fp4_spike"),
    "2d87a0a987c90f7855e69ae46e3429e490a6d4661a63f3d6d4d3043b777dd689":
        ("FP8", (128, 128, 256), (128, 128, 128), "dram128_fp8_spike"),
}
_ATTR = "mx.smem_zero_readout"


def _check_source(profile: dict, manifest: dict) -> tuple[str, tuple[int, int, int], str]:
    selected = SOURCES.get(manifest.get("source_driver_sha256"))
    if selected is None:
        raise ValueError("zero-base scratchpad readout source is not supported")
    precision, shape, tile, label = selected
    if (manifest.get("origin") != "nicolas_source_header_specialization" or
            manifest.get("precision") != precision or
            manifest.get("output_format") is not None or
            manifest.get("shape_mnk") != list(shape) or
            manifest.get("tile_mnk") != list(tile) or
            manifest.get("profile_sha256") != profile_sha256(profile) or
            profile.get("name") != "MxGemminiRocketConfig" or
            profile.get("transport") != "rocket_rocc" or
            profile["geometry"]["mesh_columns"] != 16):
        raise ValueError("zero-base scratchpad readout differs from pinned source/profile")
    return label, shape, manifest["source_driver_sha256"]


def bind_smem_zero_readout(mlir_text: str, profile: dict, manifest: dict) -> str:
    """Bind the checked scratchpad destination after payload specialization."""
    from xdsl.dialects.builtin import StringAttr
    from xdsl.printer import Printer

    label, _, _ = _check_source(profile, manifest)
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
    module.attributes[_ATTR] = StringAttr(label)
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
    if (selected != SOURCES.get(manifest.get("source_driver_sha256"), (None,) * 4)[3] or
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
    label, shape, source_sha256 = _check_source(profile, manifest)
    if (base.shape != shape or base.output_format != "bf16" or
            base.mode != "spike_serial" or not base.source_golden_preserving or
            base.derived_expected_bf16 is not None or
            base.plan.get("c_spad_dest") is None or
            base.plan.get("c_rows") != shape[0] * shape[1] * 2 // 16 or
            len(base.plan.get("waves", [])) != shape[2] // manifest["tile_mnk"][2]):
        raise ValueError("zero-base readout needs the checked source plan")
    old_dest = base.plan["c_spad_dest"]
    steps = []
    config_st = compute = readouts = 0
    for step in base.steps:
        command = step.command
        if not isinstance(command, Command):
            steps.append(step)
            continue
        if (step.phase == "configure" and command.funct == 0 and
                command.rs1.immediate == 2):
            if command.rs2.immediate != shape[1] * 2:
                raise ValueError("zero-base source store configuration changed")
            command = Command(0, command.rs1, Operand(immediate=16))
            config_st += 1
        elif step.phase == "compute" and command.funct == 8:
            if (command.rs2.immediate is None or
                    command.rs2.immediate >> 32 != old_dest or
                    command.rs2.immediate & 0xffffffff not in {0x238, 0x2b8}):
                raise ValueError("zero-base source compute destination changed")
            low = command.rs2.immediate & 0xffffffff
            command = Command(8, command.rs1, Operand(immediate=low))
            if low == 0x238:
                compute += 1
        elif step.phase == "readout" and command.funct == 3:
            if (command.rs2.immediate is None or
                    command.rs2.immediate & 0xffffffff < old_dest or
                    command.rs2.immediate & 0xffffffff >= old_dest + base.plan["c_rows"]):
                raise ValueError("zero-base source readout row changed")
            command = Command(3, command.rs1, Operand(
                immediate=command.rs2.immediate - old_dest))
            readouts += 1
        steps.append(PhysicalStep(step.phase, step.wave, command))
    if (config_st, compute, readouts) != (
            1, 1, base.plan["c_rows"] // 16):
        raise ValueError("zero-base source requires one final store and full readout")
    return PhysicalProgram(
        base.profile_sha256, base.payload_manifest_sha256, base.mode, base.shape,
        {**base.plan, "c_spad_dest": 0,
         "readout_transport": "spad_mvout_zero_base",
         "readout_source_sha256": source_sha256,
         "readout_source_mode": label},
        tuple(steps), base.source_golden_preserving, base.derived_expected_bf16,
        base.output_format, base.tiled_quant_readout,
        base.derived_vpu_scalar_bf16, base.derived_vpu_scalar_chain,
        base.golden_origin, base.golden_derivation)
