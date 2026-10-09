"""Trace the source GEMM's loop-FSM command fields from a bound MX contract.

The trace covers the source's prefetch/compute loop packets and scale-selector
updates. It is deliberately symbolic where the handwritten kernel translates
GPU addresses to host addresses or writes E8M0 scales through Muon shared memory.
It is not an executable program or a numerical result.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess

from .source_gemm import SourceGemm, plan_source_gemm, source_scratchpad_bytes
from .target_profile import profile_sha256, verify_profile_source
from .verify_profile_ir import _operation_name, _text_attr, verify_ir


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _macro(source: str, name: str) -> str:
    lines = source.splitlines()
    prefix = f"#define {name}("
    for index, line in enumerate(lines):
        if not line.startswith(prefix):
            continue
        body = [line]
        while body[-1].rstrip().endswith("\\"):
            index += 1
            if index >= len(lines):
                raise ValueError(f"unterminated source macro {name}")
            body.append(lines[index])
        return "\n".join(body)
    raise ValueError(f"source command macro {name} is absent")


def _source_command_abi(header: Path, mmio: Path) -> dict[str, str]:
    """Refuse to use these bit positions when the checked-in macros drift."""
    h = header.read_text()
    m = mmio.read_text()
    for name, value in (("k_LOOP_WS", 8), ("k_LOOP_WS_CONFIG_BOUNDS", 9),
                        ("k_LOOP_WS_CONFIG_SPAD_AB", 24), ("CONFIG_SCALE_MEM", 26),
                        ("k_MX_LOAD_SCALES", 27)):
        if not re.search(rf'^#define {name} {value}\s*$', h, re.MULTILINE):
            raise ValueError(f"Gemmini command ABI changed: {name}")
    loop = _macro(h, "gemmini_loop_ws_spad")
    scale = _macro(h, "gemmini_mxquant_config_mvout")
    scale_dma = _macro(h, "gemmini_mx_load_scales_2d")
    skips = _macro(m, "loop_matmul_skips")
    for fragment in (
        '((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(I)',
        'ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, B, k_LOOP_WS_CONFIG_SPAD_AB)',
        '((uint64_t)(C) << 32) | 0x200U | (skips)',
    ):
        if fragment not in loop:
            raise ValueError("Gemmini loop macro differs from traced command ABI")
    for fragment in (
        '((uint64_t)(scale_w_sel) << 61) | ((uint64_t)(scale_act_sel) << 60)',
        '((uint64_t)(k_bound) << 51) | ((uint64_t)(j_bound) << 42) | ((uint64_t)(i_bound) << 33)',
        '(uint64_t)(lut_update_granularity) & 0xFFFF, CONFIG_SCALE_MEM',
    ):
        if fragment not in scale:
            raise ValueError("Gemmini scale macro differs from traced command ABI")
    for fragment in (
        '((uint64_t)(dram_addr) & 0xFFFFFFFFFFULL) | ((uint64_t)(pitch) << 40)',
        '((uint64_t)(rows) << 46) | ((uint64_t)(dest) << 33) | ((uint64_t)(sel) << 32)',
        '((uint64_t)(row_bytes) & 0xFFFFFFFFu)',
    ):
        if fragment not in scale_dma:
            raise ValueError("Gemmini 2-D scale DMA macro differs from traced command ABI")
    if ('(((skip_lda) | ((skip_ldb) << 1) | ((skip_ldd) << 2) | '
            '((skip_ex) << 3) | ((skip_stc) << 4)) << 3)') not in skips:
        raise ValueError("Radiance MX loop skip encoding changed")
    return {"gemmini_header_sha256": _sha(header), "radiance_mmio_header_sha256": _sha(mmio)}


def _software_geometry(params: Path, profile: dict) -> dict[str, int | str]:
    source = params.read_text()
    values = {}
    for name in ("DIM", "BANK_NUM", "BANK_ROWS", "ADDR_LEN"):
        found = re.search(rf'^#define {name} (\d+)\s*$', source, re.MULTILINE)
        if found is None:
            raise ValueError(f"Gemmini software geometry changed: {name}")
        values[name] = int(found.group(1))
    if (values["DIM"] != profile["geometry"]["mesh_columns"] or
            values["DIM"] != profile["geometry"]["mesh_rows"] or
            values["BANK_NUM"] != profile["resources"]["scratchpad_banks"] or
            values["DIM"] * values["BANK_NUM"] * values["BANK_ROWS"] !=
            profile["resources"]["scratchpad_bytes"] or values["ADDR_LEN"] != 32):
        raise ValueError("Gemmini software geometry differs from selected MX profile")
    return {**values, "gemmini_params_sha256": _sha(params)}


def _scale_half_bytes(mmio: Path, profile: dict, rtl_root: Path) -> tuple[int, int, dict]:
    source = mmio.read_text()
    size = re.search(r'^#define GEMMINI_SF_MEM_SIZE\s+0x([0-9a-fA-F]+)$', source, re.M)
    if size is None or re.search(
            r'^#define GEMMINI_SF_MEM_BUFFER_OFFSET\s+\(GEMMINI_SF_MEM_SIZE / 4\)$',
            source, re.M) is None:
        raise ValueError("source MX scale buffer allocation changed")
    source_half = int(size.group(1), 16) // 4
    config = profile["resources"].get("scale_mem_config")
    if not isinstance(config, dict) or config.get("banks") != 8 or config.get("size_bytes") != 16384:
        raise ValueError("target MX scale buffer geometry needs separate derivation")
    target_half = config["size_bytes"] // 4
    controller = rtl_root / "src/main/scala/gemmini/Controller.scala"
    rtl = controller.read_text()
    if ('val sq_x = Cat(sq.sel, sq.dest(12))' not in rtl or
            'scale_loader_start.get.bits.dest  := unrolled_cmd.bits.cmd.rs2(45, 33)' not in rtl or
            'w_out.valid       := head_rdy && sel_r' not in rtl or
            'act_out.valid     := head_rdy && !sel_r' not in rtl):
        raise ValueError("target MX scale DMA half-selection decode changed")
    return source_half, target_half, {"controller_sha256": _sha(controller)}


def _scale_dma_packets(plan: dict, *, half_bytes: int) -> list[dict]:
    """Plan target funct-27 2-D DMA loads from source E8M0 matrix slices."""
    m, n, _ = plan["shape"]
    _, _, tk = plan["tile"]
    rows = tk // 32
    if rows <= 0 or rows > 255 or half_bytes != 4096:
        raise ValueError("MX scale DMA row count or selected half size is unsupported")
    packets = []
    for wave in plan["waves"]:
        group = wave["k_start"] // 32
        dest = (wave["index"] & 1) * half_bytes
        for side, columns, sel in (("A", m, 0), ("B", n, 1)):
            if (columns % 8 or columns * rows > half_bytes or columns >= 1 << 24 or
                    dest >= 1 << 13):
                raise ValueError("MX E8M0 scale slice exceeds one target buffer half")
            packets.append({
                "phase": "prologue_scale_dma" if wave["index"] == 0 else "next_scale_dma",
                "wave": wave["index"], "side": side, "funct": 27,
                "rs1_dram_address": f"&{side}_scales_{'row' if side == 'A' else 'col'}[{group}][0]",
                "rs1_static_high_bits": columns << 40,
                "rs2": (rows << 46) | (dest << 33) | (sel << 32) | columns,
                "row_bytes": columns, "rows": rows, "pitch_bytes": columns,
                "destination_byte": dest,
                "source_byte_offset": group * columns,
                "byte_count": rows * columns,
            })
    return packets


def _bound_contract(mlir_text: str, profile: dict, kernel: SourceGemm) -> str:
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser

    checked = verify_ir(mlir_text, profile)
    if (checked["contracts"], checked["encodes"], checked["requantizes"],
            checked["vpu_commands"], checked["spad_requants"]) != (1, 0, 0, 0, 0):
        raise ValueError("source loop trace needs one bound MX contraction and no other MX compute")
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    ops = [op for op in module.walk() if _operation_name(op).startswith("mx_gemmini.")]
    contracts = [op for op in ops if _operation_name(op) == "mx_gemmini.contract"]
    readouts = [op for op in ops if _operation_name(op) == "mx_gemmini.readout_bf16"]
    if len(contracts) != 1 or len(readouts) != 1 or len(ops) != 2 or kernel.quant_output:
        raise ValueError("source loop trace supports one BF16 output contraction")
    contract = contracts[0]
    expected = {"FP8": ("fp8_e4m3", "direct"), "FP6": ("fp6_e3m2", "lut"),
                "FP4": ("fp4_e2m1", "direct")}[kernel.datatype]
    for side in ("activation", "weight"):
        if (_text_attr(contract, f"{side}_format"),
                _text_attr(contract, f"{side}_projection")) != expected:
            raise ValueError(f"bound MX {side} mode differs from source driver")
    site = _text_attr(contract, "site_id")
    if _text_attr(readouts[0], "site_id") != site:
        raise ValueError("MX readout site differs from source contraction")
    return site


def _loop_packet(plan: dict, wave: dict, *, phase: str) -> list[dict]:
    """Expand gemmini_loop_ws_spad's three source macro instructions."""
    tm, tn, tk = plan["tile"]
    pe = 16 if plan["format"] == "fp8_e4m3" else 32
    i, j, k = tm // pe, tn // pe, tk // 16
    if any(not 0 < value <= 0xffff for value in (i, j, k)):
        raise ValueError("source GEMM loop bound exceeds 16-bit command field")
    if phase == "prefetch":
        # loop_matmul_skips(0, 0, 1, 1, 1): DMA only.
        skips, acc, c_row = 0xe0, 0, 0
    elif phase == "compute":
        # loop_matmul_skips(1, 1, 1, 0, !move_acc_to_spad).
        skips = 0x38 | (0 if wave["move_acc_to_spad"] else 0x80)
        acc, c_row = int(wave["accumulate"]), plan["c_spad_dest"]
    else:
        raise ValueError(f"unknown source GEMM loop phase {phase}")
    a_row, b_end = wave["a_spad_start"], wave["b_spad_end"]
    if (not 0 <= a_row < plan["scratchpad_rows"] or
            not 0 < b_end <= plan["scratchpad_rows"] or
            not 0 <= c_row < plan["scratchpad_rows"]):
        raise ValueError("source GEMM scratchpad address exceeds selected target")
    fields = ((9, 0, (k << 32) | (j << 16) | i),
              (24, a_row, b_end),
              (8, acc, (c_row << 32) | 0x200 | skips))
    return [{"phase": phase, "wave": wave["index"], "funct": funct,
             "rs1": rs1, "rs2": rs2} for funct, rs1, rs2 in fields]


def trace_bound_source_gemm_loops(mlir_text: str, kernel: SourceGemm, *,
                                  profile: dict, source_root: Path,
                                  rtl_root: Path) -> dict:
    """Derive source-ordered MX loop packets for one profile-bound MLIR site."""
    if not kernel.driver.resolve().is_relative_to(source_root.resolve()):
        raise ValueError("source GEMM driver must belong to the selected kernels checkout")
    verify_profile_source(profile, rtl_root)
    submodule = rtl_root / "software/gemmini-rocc-tests"
    pinned = subprocess.check_output(
        ["git", "-C", str(rtl_root), "ls-tree", "HEAD", "software/gemmini-rocc-tests"],
        text=True).split()[2]
    checked_out = subprocess.check_output(
        ["git", "-C", str(submodule), "rev-parse", "HEAD"], text=True).strip()
    if checked_out != pinned:
        raise ValueError("Gemmini software header differs from the selected RTL gitlink")
    site = _bound_contract(mlir_text, profile, kernel)
    mmio = source_root / "lib/include/mxgemmini_mmio.h"
    abi = _source_command_abi(
        rtl_root / "software/gemmini-rocc-tests/include/gemmini.h", mmio)
    geometry = _software_geometry(
        rtl_root / "software/gemmini-rocc-tests/include/gemmini_params.h", profile)
    source_scale_half, target_scale_half, scale_abi = _scale_half_bytes(
        mmio, profile, rtl_root)
    source_bytes = source_scratchpad_bytes(source_root / "lib/mxgemm/mxgemm_lib.hpp")
    source_plan = plan_source_gemm(kernel, scratchpad_bytes=source_bytes)
    target_plan = plan_source_gemm(
        kernel, scratchpad_bytes=profile["resources"]["scratchpad_bytes"], profile=profile)
    if "output_tiles" in target_plan:
        raise ValueError("source loop trace models one output tile only")
    waves = target_plan["waves"]
    packets = _loop_packet(target_plan, waves[0], phase="prefetch")
    for wave in waves:
        # CONFIG_SCALE_MEM fields are exact; host(C_scale_factors) occupies rs1[32:0].
        selector = (wave["index"] & 1) << 60 | (wave["index"] & 1) << 61
        tm, tn, tk = target_plan["tile"]
        pe = 16 if target_plan["format"] == "fp8_e4m3" else 32
        selector |= (tk // 16) << 51 | (tn // pe) << 42 | (tm // pe) << 33
        packets.append({"phase": "select_scales", "wave": wave["index"],
                        "funct": 26, "rs1_host_address": "C_scale_factors",
                        "rs1_static_high_bits": selector, "rs2": 1})
        if wave["prefetch_next"]:
            packets.extend(_loop_packet(target_plan, waves[wave["index"] + 1], phase="prefetch"))
        packets.extend(_loop_packet(target_plan, wave, phase="compute"))
    return {
        "schema": "mx_gemmini.source_loop_trace.v1",
        "status": "symbolic_matrix_loop_trace_only",
        "site_id": site,
        "source_driver": kernel.driver.name,
        "source_driver_sha256": _sha(kernel.driver),
        "source_library_sha256": _sha(source_root / "lib/mxgemm/mxgemm_lib.hpp"),
        "source_layout_c_row": source_plan["c_spad_dest"],
        "target_profile_sha256": profile_sha256(profile),
        "target_layout_c_row": target_plan["c_spad_dest"],
        "wave_count": len(waves),
        "abi": abi,
        "software_geometry": geometry,
        "scale_memory": {
            "source_half_bytes": source_scale_half,
            "target_half_bytes": target_scale_half,
            "abi": scale_abi,
            "source_trailing_upload_suppressed": True,
        },
        "scale_dma_packets": _scale_dma_packets(target_plan, half_bytes=target_scale_half),
        "gemmini_software_revision": checked_out,
        "packets": packets,
        "omitted": ["configuration commands", "Muon E8M0/LUT shared-memory writes (target uses symbolic funct-27 E8M0 DMA instead)",
                    "Muon memory fences and busy waits", "SIMT or accumulator move-out",
                    "GPU-to-host buffer address translation"],
    }
