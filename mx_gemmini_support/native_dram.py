"""Source-bound native DRAM-loop schedule for Nicolas's FP8 MX program."""

from __future__ import annotations

from io import StringIO

from .chunked_i import _parse_module
from .command_ir import Command, Fence, Operand
from .physical_program import PhysicalProgram, PhysicalStep, _cmd
from .source_payload import manifest_sha256
from .target_profile import profile_sha256
from .verify_profile_ir import _operation_name, _text_attr, verify_ir


SOURCE_SHA256 = "53b5f60844725aa9e289a1c3833002fe777c2a3b41ea64a31b05c2237eb32277"
_SHAPE = [128, 128, 128]
_ATTR = "mx.native_dram_loop"


def _check_source(profile: dict, manifest: dict) -> None:
    if (manifest.get("origin") != "nicolas_source_header_specialization" or
            manifest.get("source_driver_sha256") != SOURCE_SHA256 or
            manifest.get("precision") != "FP8" or
            manifest.get("output_format") is not None or
            manifest.get("shape_mnk") != _SHAPE or
            manifest.get("tile_mnk") != _SHAPE or
            manifest.get("profile_sha256") != profile_sha256(profile) or
            profile.get("name") != "MxGemminiRocketConfig" or
            profile.get("transport") != "rocket_rocc" or
            profile["geometry"]["mesh_columns"] != 16):
        raise ValueError("native DRAM-loop schedule differs from pinned source and profile")


def bind_native_dram(mlir_text: str, profile: dict, manifest: dict) -> str:
    """Select Nicolas's native loop after the typed contraction is source-bound."""
    from xdsl.dialects.builtin import StringAttr
    from xdsl.printer import Printer

    _check_source(profile, manifest)
    verify_ir(mlir_text, profile)
    module = _parse_module(mlir_text)
    if (_text_attr(module, "mx.payload_manifest_sha256") != manifest_sha256(manifest) or
            _text_attr(module, _ATTR) is not None or
            _text_attr(module, "mx.i_chunks") is not None or
            [_operation_name(op) for op in module.walk()
             if _operation_name(op).startswith("mx_gemmini.") and
             _operation_name(op) != "mx_gemmini.resource"] != [
                "mx_gemmini.contract", "mx_gemmini.readout_bf16"]):
        raise ValueError("native DRAM loop needs one source-bound BF16 contraction")
    module.attributes[_ATTR] = StringAttr("single")
    stream = StringIO()
    Printer(stream=stream).print_op(module)
    rendered = stream.getvalue() + "\n"
    verify_ir(rendered, profile)
    return rendered


def selected_native_dram(mlir_text: str, profile: dict,
                         manifest: dict) -> bool:
    module = _parse_module(mlir_text)
    selected = _text_attr(module, _ATTR)
    if selected is None:
        return False
    if selected != "single" or _text_attr(module, "mx.payload_manifest_sha256") != \
            manifest_sha256(manifest) or _text_attr(module, "mx.i_chunks") is not None:
        raise ValueError("native DRAM loop is not bound to selected payload")
    _check_source(profile, manifest)
    return True


def lower_native_dram(base: PhysicalProgram, profile: dict,
                      manifest: dict) -> PhysicalProgram:
    """Issue LoopMatmul's DRAM A/B loads and C store, without explicit DMA."""
    _check_source(profile, manifest)
    plan = base.plan
    if (base.shape != (128, 128, 128) or base.output_format != "bf16" or
            base.mode != "spike_serial" or not base.source_golden_preserving or
            len(plan["waves"]) != 1 or
            len(plan.get("output_tiles", [{"index": 0}])) != 1 or
            plan["c_spad_dest"] != 1024):
        raise ValueError("native DRAM loop needs the checked single-wave 128³ plan")
    by_phase: dict[str, list[PhysicalStep]] = {}
    for step in base.steps:
        by_phase.setdefault(step.phase, []).append(step)
    if (set(by_phase) != {"configure", "upload_scales", "move_activation",
                          "move_weight", "select_scales", "compute", "readout"} or
            [step.command.funct for step in by_phase["configure"]
             if isinstance(step.command, Command)] != [7, 0, 0, 0, 0]):
        raise ValueError("native DRAM loop base command stream has changed")

    steps = [*by_phase["configure"][:2], *by_phase["upload_scales"],
             *by_phase["configure"][2:5], *by_phase["select_scales"]]

    def issue(command: Command | Fence) -> None:
        steps.append(PhysicalStep("native_dram_loop", None, command))

    issue(_cmd(9, 0, (8 << 32) | (8 << 16) | 8))
    issue(_cmd(10, Operand(buffer="activation"), Operand(buffer="weight")))
    issue(_cmd(11, 0, Operand(buffer="output_bf16")))
    issue(_cmd(12, 128, 128))
    issue(_cmd(13, 0, 128))
    issue(_cmd(8, (1 << 18) | (1 << 16), 0))
    issue(Fence())
    return PhysicalProgram(base.profile_sha256, base.payload_manifest_sha256,
                           base.mode, base.shape,
                           {**plan, "execution_transport": "native_dram_loop",
                            "native_dram_source_sha256": SOURCE_SHA256},
                           tuple(steps), base.source_golden_preserving,
                           base.derived_expected_bf16, base.output_format,
                           base.tiled_quant_readout,
                           base.derived_vpu_scalar_bf16,
                           base.derived_vpu_scalar_chain,
                           base.golden_origin)
