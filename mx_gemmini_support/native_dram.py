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
CHUNK_SOURCE_SHA256 = {
    2: "eb80572c6e1a9ad1a700d4a98d28e106dda214bca377903cea30e37ebd3f7879",
    4: "88cab5f1ff321e1bb34687654d226dc134f0bd74c13d68ae1e88b9e79f20e99a",
}
LOOP_SCALE_SOURCE_SHA256 = {
    "ls2": "b7d8b95e0e8a3cc16730125f3c2bf5477a042754b065cc2e4725973a57fe4e34",
    "ls4": "946515729cd3d962253b238af23522b849bf6baef721238ec4f2dacfc784a706",
}
K_TILED_SOURCE_SHA256 = "586910dcd6445a3ba0fc0e9db485d85de3ea265b8efe7464931ba3ff005dfa08"
PRELOAD_VARIANT_SOURCE_SHA256 = {
    "nc2_2d": "4d5fffa5153779041d4f271cad33c6971af028b03031eb21d7d547d06aba4854",
    "nc2_wait": "c69f99fcc6bb791f705ce94c7ceeef8a10efc80a4fcdb47e1ba7e578ce5ea974",
}
_SHAPE = [128, 128, 128]
_ATTR = "mx.native_dram_loop"


def _check_source(profile: dict, manifest: dict,
                  selection: int | str) -> None:
    expected_source = {1: SOURCE_SHA256, **CHUNK_SOURCE_SHA256,
                       **LOOP_SCALE_SOURCE_SHA256,
                       **PRELOAD_VARIANT_SOURCE_SHA256,
                       "kt2": K_TILED_SOURCE_SHA256}.get(selection)
    if (manifest.get("origin") != "nicolas_source_header_specialization" or
            expected_source is None or
            manifest.get("source_driver_sha256") != expected_source or
            manifest.get("precision") != "FP8" or
            manifest.get("output_format") is not None or
            manifest.get("shape_mnk") != _SHAPE or
            manifest.get("tile_mnk") != _SHAPE or
            manifest.get("profile_sha256") != profile_sha256(profile) or
            profile.get("name") != "MxGemminiRocketConfig" or
            profile.get("transport") != "rocket_rocc" or
            profile["geometry"]["mesh_columns"] != 16):
        raise ValueError("native DRAM-loop schedule differs from pinned source and profile")


def bind_native_dram(mlir_text: str, profile: dict, manifest: dict,
                     chunks: int = 1, *, scale_mode: str = "preload",
                     k_tiles: int = 1) -> str:
    """Select Nicolas's native loop after the typed contraction is source-bound."""
    from xdsl.dialects.builtin import StringAttr
    from xdsl.printer import Printer

    selection: int | str = ("nc2_2d" if chunks == 2 and k_tiles == 1 and
                            scale_mode == "direct_2d" else
                            "nc2_wait" if chunks == 2 and k_tiles == 1 and
                            scale_mode == "wait" else
                            "kt2" if chunks == 2 and k_tiles == 2 and
                            scale_mode == "loop" else
                            chunks if scale_mode == "preload" and k_tiles == 1 else
                            f"ls{chunks}" if scale_mode == "loop" else "unknown")
    _check_source(profile, manifest, selection)
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
    module.attributes[_ATTR] = StringAttr("single" if selection == 1 else
                                          str(selection))
    stream = StringIO()
    Printer(stream=stream).print_op(module)
    rendered = stream.getvalue() + "\n"
    verify_ir(rendered, profile)
    return rendered


def selected_native_dram(mlir_text: str, profile: dict,
                         manifest: dict) -> int | str | None:
    module = _parse_module(mlir_text)
    selected = _text_attr(module, _ATTR)
    if selected is None:
        return None
    if selected not in {"single", "2", "4", "ls2", "ls4", "kt2", "nc2_2d", "nc2_wait"} or _text_attr(module, "mx.payload_manifest_sha256") != \
            manifest_sha256(manifest) or _text_attr(module, "mx.i_chunks") is not None:
        raise ValueError("native DRAM loop is not bound to selected payload")
    selection: int | str = (1 if selected == "single" else
                            int(selected) if selected in {"2", "4"} else selected)
    _check_source(profile, manifest, selection)
    return selection


def lower_native_dram(base: PhysicalProgram, profile: dict,
                      manifest: dict,
                      selection: int | str = 1) -> PhysicalProgram:
    """Issue LoopMatmul's DRAM A/B loads and C store, without explicit DMA."""
    _check_source(profile, manifest, selection)
    chunks = (2 if selection in {"kt2", "nc2_2d", "nc2_wait"} else
              int(selection[2:]) if isinstance(selection, str) else selection)
    loop_scales = isinstance(selection, str)
    if selection in {"nc2_2d", "nc2_wait"}:
        loop_scales = False
    scale_wait = selection == "nc2_wait"
    k_tiles = 2 if selection == "kt2" else 1
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

    steps = list(by_phase["configure"][:2])
    if loop_scales:
        pass
    elif chunks == 1:
        steps.extend(by_phase["upload_scales"])
    else:
        nc = 128 // chunks
        for c in range(chunks):
            for buffer, offset, count, dest, selector in (
                    ("activation_scales", 0, 128, c * 4 * 128, 0),
                    ("weight_scales", c * nc, nc, c * 4 * nc, 1)):
                rs1 = Operand(buffer=buffer, byte_offset=offset,
                              address_mask=(1 << 40) - 1,
                              or_bits=128 << 40)
                rs2 = (4 << 46) | (dest << 33) | (selector << 32) | count
                steps.append(PhysicalStep("upload_chunk_scales", c,
                                          _cmd(27, rs1, rs2)))
        if not scale_wait:
            steps.append(PhysicalStep("upload_chunk_scales", None, Fence()))
    steps.extend(by_phase["configure"][2:5])
    if loop_scales:
        pass
    elif chunks == 1:
        steps.extend(by_phase["select_scales"])
    else:
        j = 8 // chunks
        selector_bits = (8 * chunks << 51) | (j << 42) | (8 << 33)
        steps.append(PhysicalStep("select_scales", None,
                                  _cmd(26, Operand(
                                      buffer="scratch_output_scales",
                                      address_mask=(1 << 33) - 1,
                                      or_bits=selector_bits),
                                       1 | ((1 << 16) if scale_wait else 0))))

    def issue(command: Command | Fence) -> None:
        steps.append(PhysicalStep("native_dram_loop", None, command))

    for c in range(chunks):
        j = 8 // chunks
        nc = 128 // chunks
        for t in range(k_tiles):
            kt = 128 // k_tiles
            if loop_scales:
                steps.append(PhysicalStep("loop_scales", c * k_tiles + t, _cmd(
                    31, Operand(buffer="activation_scales",
                                byte_offset=t * kt // 32 * 128),
                    Operand(buffer="weight_scales",
                            byte_offset=t * kt // 32 * 128 + c * nc))))
                steps.append(PhysicalStep("loop_scales", c * k_tiles + t,
                                          _cmd(32, 128, 128)))
            issue(_cmd(9, 0, ((kt // 16) << 32) | (j << 16) | 8))
            issue(_cmd(10, Operand(buffer="activation", byte_offset=t * kt)
                       if k_tiles > 1 or c == 0 else 0,
                       Operand(buffer="weight",
                               byte_offset=t * kt * 128 + c * nc)))
            issue(_cmd(11, 0,
                       Operand(buffer="output_bf16", byte_offset=c * nc * 2)
                       if t == k_tiles - 1 else 0))
            issue(_cmd(12, 128, 128))
            issue(_cmd(13, 0, 128))
            b_spad_id = (1 if chunks == 1 else
                         1 + ((c * k_tiles + t) & 1))
            issue(_cmd(8, (1 << 18) | (b_spad_id << 16) | int(t > 0), 0))
    issue(Fence())
    return PhysicalProgram(base.profile_sha256, base.payload_manifest_sha256,
                           base.mode, base.shape,
                           {**plan, "execution_transport": "native_dram_loop",
                            "native_dram_source_sha256":
                            {1: SOURCE_SHA256, **CHUNK_SOURCE_SHA256,
                             **LOOP_SCALE_SOURCE_SHA256,
                             **PRELOAD_VARIANT_SOURCE_SHA256,
                             "kt2": K_TILED_SOURCE_SHA256}[selection],
                            **({} if chunks == 1 else {"native_dram_n_chunks": chunks}),
                            **({"native_dram_scale_mode": "loop_managed"}
                               if loop_scales else {}),
                            **({"native_dram_k_tiles": k_tiles}
                               if k_tiles > 1 else {}),
                            **({"native_dram_scale_wait": True}
                               if scale_wait else {})},
                           tuple(steps), base.source_golden_preserving,
                           base.derived_expected_bf16, base.output_format,
                           base.tiled_quant_readout,
                           base.derived_vpu_scalar_bf16,
                           base.derived_vpu_scalar_chain,
                           base.golden_origin)
