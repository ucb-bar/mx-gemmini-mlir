"""Source-bound I-axis MX loop chunking for Nicolas's FP8 experiments."""

from __future__ import annotations

from io import StringIO

from .command_ir import Command, Fence, Operand
from .physical_program import PhysicalProgram, PhysicalStep, _cmd
from .source_payload import manifest_sha256
from .target_profile import profile_sha256
from .verify_profile_ir import _operation_name, _text_attr, verify_ir


CHUNK_SOURCE_HASHES = {
    2: "c2cd8affbbbd079a3553bcfc8bf46246114c8a2ff62e38f07994447cb512661e",
    4: "12bc6da5f5a54f87a4df95664a8259f6b0610675a92d0259121cbf13fe891275",
}
_SHAPE = [128, 128, 128]


def _check_source(profile: dict, manifest: dict, chunks: int) -> None:
    if (chunks not in CHUNK_SOURCE_HASHES or
            manifest.get("origin") != "nicolas_source_header_specialization" or
            manifest.get("source_driver_sha256") != CHUNK_SOURCE_HASHES[chunks] or
            manifest.get("precision") != "FP8" or
            manifest.get("output_format") is not None or
            manifest.get("shape_mnk") != _SHAPE or
            manifest.get("tile_mnk") != _SHAPE or
            manifest.get("profile_sha256") != profile_sha256(profile) or
            profile.get("transport") != "rocket_rocc" or
            profile["geometry"]["mesh_columns"] != 16):
        raise ValueError("I-chunked MX schedule differs from pinned source and profile")


def _parse_module(mlir_text: str):
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser

    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    return Parser(context, mlir_text).parse_module()


def bind_i_chunks(mlir_text: str, profile: dict, manifest: dict,
                  chunks: int) -> str:
    """Record Nicolas's selected chunk schedule on a captured contraction."""
    from xdsl.dialects.builtin import StringAttr
    from xdsl.printer import Printer

    _check_source(profile, manifest, chunks)
    verify_ir(mlir_text, profile)
    module = _parse_module(mlir_text)
    if (_text_attr(module, "mx.payload_manifest_sha256") != manifest_sha256(manifest) or
            _text_attr(module, "mx.i_chunks") is not None or
            [_operation_name(op) for op in module.walk()
             if _operation_name(op).startswith("mx_gemmini.") and
             _operation_name(op) not in {"mx_gemmini.resource",
                                         "mx_gemmini.upload_lut"}] != [
                "mx_gemmini.contract", "mx_gemmini.readout_bf16"]):
        raise ValueError("I-chunked MX schedule needs one source-bound BF16 contraction")
    module.attributes["mx.i_chunks"] = StringAttr(str(chunks))
    stream = StringIO()
    Printer(stream=stream).print_op(module)
    rendered = stream.getvalue() + "\n"
    verify_ir(rendered, profile)
    return rendered


def selected_i_chunks(mlir_text: str, profile: dict,
                       manifest: dict) -> int | None:
    module = _parse_module(mlir_text)
    value = _text_attr(module, "mx.i_chunks")
    if value is None:
        return None
    if value not in {"2", "4"} or _text_attr(module, "mx.payload_manifest_sha256") != \
            manifest_sha256(manifest):
        raise ValueError("I-chunked MX schedule is not bound to the selected payload")
    chunks = int(value)
    _check_source(profile, manifest, chunks)
    return chunks


def lower_i_chunks(base: PhysicalProgram, profile: dict, manifest: dict,
                   chunks: int) -> PhysicalProgram:
    """Replace one MX loop with source-equivalent chunk launches and scale DMA."""
    _check_source(profile, manifest, chunks)
    plan = base.plan
    if (base.shape != (128, 128, 128) or base.output_format != "bf16" or
            base.mode != "spike_serial" or not base.source_golden_preserving or
            len(plan["waves"]) != 1 or
            len(plan.get("output_tiles", [{"index": 0}])) != 1 or
            plan["c_spad_dest"] != 1024 or
            plan["waves"][0]["a_spad_start"] != 0 or
            plan["waves"][0]["b_spad_end"] != 16384):
        raise ValueError("I-chunked MX requires the checked single-wave 128³ plan")
    by_phase: dict[str, list[PhysicalStep]] = {}
    for step in base.steps:
        by_phase.setdefault(step.phase, []).append(step)
    if (set(by_phase) != {"configure", "upload_scales", "move_activation",
                          "move_weight", "select_scales", "compute", "readout"} or
            len([step for step in by_phase["compute"]
                 if isinstance(step.command, Command)]) != 3):
        raise ValueError("I-chunked MX base command stream has changed")

    steps = list(by_phase["configure"])
    def issue(phase: str, command: Command | Fence, chunk: int | None = None) -> None:
        steps.append(PhysicalStep(phase, chunk, command))

    groups, n, dim = 4, 128, 16
    chunk_m = 128 // chunks
    chunk_i = chunk_m // dim
    tiles_j = tiles_k = 8
    scale_half = profile["resources"]["scale_mem_config"]["size_bytes"] // 4
    if scale_half != 4096:
        raise ValueError("I-chunked MX needs the pinned 4-KiB scale half")
    for chunk in range(chunks):
        a_offset = chunk * chunk_m
        a_dest = chunk * groups * chunk_m
        b_dest = chunk * groups * n
        for buffer, offset, pitch, count, dest, selector in (
                ("activation_scales", a_offset, 128, chunk_m, a_dest, 0),
                ("weight_scales", 0, 128, n, b_dest, 1)):
            if dest + groups * count > scale_half:
                raise ValueError("I-chunked MX scale upload exceeds selected half")
            rs1 = Operand(buffer=buffer, byte_offset=offset,
                          address_mask=(1 << 40) - 1, or_bits=pitch << 40)
            rs2 = (groups << 46) | (dest << 33) | (selector << 32) | count
            issue("upload_chunk_scales", _cmd(27, rs1, rs2), chunk)
    issue("upload_chunk_scales", Fence())
    steps.extend(by_phase["move_activation"])
    steps.extend(by_phase["move_weight"])
    selector_bits = (tiles_k * chunks << 51) | (tiles_j << 42) | (chunk_i << 33)
    issue("select_chunk_scales", _cmd(26, Operand(buffer="scratch_output_scales",
                                                 address_mask=(1 << 33) - 1,
                                                 or_bits=selector_bits), 1))
    for chunk in range(chunks):
        a_row = chunk * chunk_i * tiles_k * dim
        c_row = 1024 + chunk * chunk_m * n * 2 // dim
        issue("compute_chunk", _cmd(9, 0,
                                    (tiles_k << 32) | (tiles_j << 16) | chunk_i), chunk)
        issue("compute_chunk", _cmd(24, a_row, 16384), chunk)
        issue("compute_chunk", _cmd(8, 0, (c_row << 32) | 0x200 | 0x138), chunk)
    issue("compute_chunk", Fence())
    steps.extend(by_phase["readout"])
    return PhysicalProgram(base.profile_sha256, base.payload_manifest_sha256,
                           base.mode, base.shape,
                           {**plan, "i_chunks": chunks,
                            "chunk_issue_schedule": "nicolas_i_axis_loop_overlap_v1"},
                           tuple(steps), base.source_golden_preserving,
                           base.derived_expected_bf16, base.output_format,
                           base.tiled_quant_readout,
                           base.derived_vpu_scalar_bf16,
                           base.derived_vpu_scalar_chain,
                           base.golden_origin)
