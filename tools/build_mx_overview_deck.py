"""Build the editable MX Gemmini overview from the BAR slide template.

Requires python-pptx. All diagrams, labels, callouts, and table cells remain
native PowerPoint objects; the university seal is the template's master image.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


NAVY = RGBColor(16, 53, 82)
INK = RGBColor(49, 52, 85)
GREY = RGBColor(91, 89, 87)
CREAM = RGBColor(253, 248, 241)
PALE_BLUE = RGBColor(225, 232, 236)
SAND = RGBColor(237, 229, 220)
SAGE = RGBColor(229, 231, 218)
CODE_PURPLE = RGBColor(108, 61, 128)
CODE_GREEN = RGBColor(40, 112, 79)
CODE_ORANGE = RGBColor(164, 79, 37)
CODE_TEAL = RGBColor(0, 104, 116)

CODE_TOKEN = re.compile(
    r'"(?:[^"\\]|\\.)*"|%[A-Za-z][\w]*|\b(?:true|false|before|after|'
    r'__asm__|volatile|objdump|contract|return)\b|\b(?:0x[0-9a-fA-F]+|'
    r'\d+(?:\.\d+)?)\b|\b(?:mx_gemmini\.[A-Za-z_]+|'
    r'activation_format|weight_format|pe_mode|profile_sha256|'
    r'geometry|legal_compute|resources|mesh_rows|mesh_columns|'
    r'scratchpad_bytes|vpu|spad_requant)\b'
)


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def code_color(token: str, following: str):
    if token.startswith('"'):
        if token.startswith('"mx_gemmini.'):
            return CODE_PURPLE
        return CODE_PURPLE if following.lstrip().startswith(":") else CODE_GREEN
    if token.startswith("%"):
        return CODE_TEAL
    if token[0].isdigit():
        return CODE_ORANGE
    return CODE_PURPLE


def set_text(shape, text: str, *, size: float, bold: bool = False,
             color=INK, align=PP_ALIGN.CENTER) -> None:
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    for index, line in enumerate(text.split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = align
        paragraph.space_before = Pt(0)
        paragraph.space_after = Pt(0)
        run = paragraph.add_run()
        run.text = line
        run.font.name = "Inter"
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color


def add_text(slide, text: str, x: float, y: float, w: float, h: float,
             *, size: float, bold: bool = False, color=INK,
             align=PP_ALIGN.LEFT):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    shape.text_frame.margin_left = Inches(0.02)
    shape.text_frame.margin_right = Inches(0.02)
    shape.text_frame.margin_top = Inches(0.01)
    shape.text_frame.margin_bottom = Inches(0.01)
    set_text(shape, text, size=size, bold=bold, color=color, align=align)
    return shape


def add_panel(slide, x: float, y: float, w: float, h: float, fill):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                   Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = RGBColor(176, 165, 151)
    return shape


def add_code(slide, source: str, x: float, y: float, w: float, h: float,
             *, size: float = 11.2):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                 Inches(x), Inches(y), Inches(w), Inches(h))
    box.fill.solid()
    box.fill.fore_color.rgb = CREAM
    box.line.color.rgb = RGBColor(211, 202, 191)
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = False
    frame.vertical_anchor = MSO_ANCHOR.TOP
    frame.margin_left = Inches(0.12)
    frame.margin_top = Inches(0.10)
    frame.margin_right = Inches(0.08)
    frame.margin_bottom = Inches(0.04)
    for index, line in enumerate(source.split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = PP_ALIGN.LEFT
        paragraph.space_before = Pt(0)
        paragraph.space_after = Pt(0)
        offset = 0
        for match in CODE_TOKEN.finditer(line):
            for value, color in ((line[offset:match.start()], NAVY),
                                 (match.group(), code_color(match.group(),
                                                            line[match.end():]))):
                if value:
                    run = paragraph.add_run()
                    run.text = value
                    run.font.name = "Liberation Mono"
                    run.font.size = Pt(size)
                    run.font.color.rgb = color
            offset = match.end()
        if offset < len(line):
            run = paragraph.add_run()
            run.text = line[offset:]
            run.font.name = "Liberation Mono"
            run.font.size = Pt(size)
            run.font.color.rgb = NAVY
    return box


def clear_content(slide) -> None:
    # Keep the template title. All new content remains native PowerPoint art.
    for shape in list(slide.shapes)[1:]:
        shape._element.getparent().remove(shape._element)


def keep_slides(prs: Presentation) -> None:
    # Template slides: process 20, architecture 23, results table 33.
    slide_ids = list(prs.slides._sldIdLst)
    retained = [slide_ids[index] for index in (22, 19, 32)]
    for slide_id in slide_ids:
        if slide_id not in retained:
            prs.part.drop_rel(slide_id.rId)
        prs.slides._sldIdLst.remove(slide_id)
    for slide_id in retained:
        prs.slides._sldIdLst.append(slide_id)


def style_master(prs: Presentation) -> None:
    master = prs.slides[0].slide_layout.slide_master
    by_name = {shape.name: shape for shape in master.shapes}
    set_text(by_name["FRAME | Lab name"], "BAR LAB", size=14, bold=True,
             color=NAVY, align=PP_ALIGN.LEFT)
    set_text(by_name["FRAME | Purpose label"], "PROGRAM", size=10,
             color=INK, align=PP_ALIGN.RIGHT)
    set_text(by_name["FRAME | Purpose value"], "MX GEMMINI + VPU", size=10,
             color=INK, align=PP_ALIGN.RIGHT)
    by_name["PAGE"].text = ""


def page_number(slide, number: int, total: int = 3) -> None:
    add_text(slide, f"{number:02d} / {total:02d}", 11.45, 6.88, 1.35, 0.28,
             size=10, color=GREY, align=PP_ALIGN.RIGHT)


def architecture(slide, total: int = 3) -> None:
    set_text(slide.shapes[0], "MX dialect: typed IR to RV64", size=27,
             bold=True)
    clear_content(slide)
    add_panel(slide, 0.94, 1.64, 5.65, 2.98, SAND)
    add_panel(slide, 6.76, 1.64, 5.65, 2.98, PALE_BLUE)
    add_text(slide, "FP4 GEMM → typed MX", 1.15, 1.82, 5.2, 0.34,
             size=17.5, bold=True, color=NAVY)
    add_code(slide,
             '%acc = "mx_gemmini.contract"(%A,%As,%B,%Bs) {\n'
             '  activation_format = "fp4_e2m1",\n'
             '  weight_format = "fp4_e2m1", pe_mode = 0 : i32 }\n'
             '%out = "mx_gemmini.readout_bf16"(%acc)',
             1.15, 2.24, 5.20, 1.42, size=11.1)
    add_text(slide,
             "resource supplies packed A/B codes + E8M0 scales. contract "
             "computes C[m,n] = Σk A[m,k]×B[k,n] in the selected PE mode; "
             "readout_bf16 returns BF16. VPU can do softmax; Radiance "
             "uses Muon SIMT.",
             1.15, 3.76, 5.17, 0.75, size=10.6, color=INK)
    add_text(slide, "Compiled attention object → binary", 6.98, 1.82,
             5.2, 0.34, size=17.5, bold=True, color=NAVY)
    add_code(slide,
             '__asm__ volatile(\n'
             '  ".insn r 0x7b, 3, 2, x0, %0, %1"\n'
             '  : : "r"(Q_IN),\n'
             '      "r"(0x0010001000000000));\n'
             'objdump: 4a: 04b6b07b  .insn 4, 0x04b6b07b',
             6.98, 2.24, 5.20, 1.42, size=10.3)
    add_text(slide,
             "physical_program orders mvin → compute → VPU → mvout. "
             "RoCC C + data asm → RV64 GCC → ELF → Nicolas Spike.",
             6.98, 3.82, 5.17, 0.60, size=11.0, color=INK)

    add_text(slide, "Implemented transforms and checks", 0.96, 4.76,
             11.4, 0.30, size=16, bold=True, color=NAVY)
    transforms = (
        ("bind_profile", "Legal PE tuple +\nprofile digest"),
        ("MXBF16Catalog\nPreparation", "Select BF16\nMX sites"),
        ("schedule.plan", "Choose native\nloop policy"),
        ("MXVPUPostOffload", "Map softmax, add\nand reductions"),
        ("physical_program", "Order DMA, compute\nand VPU commands"),
    )
    for index, (name, action) in enumerate(transforms):
        x = 0.94 + index * 2.34
        add_panel(slide, x, 5.15, 2.15, 1.08,
                  (CREAM, PALE_BLUE, CREAM, SAGE, CREAM)[index])
        add_text(slide, name, x + 0.10, 5.25, 1.95, 0.37,
                 size=11.7, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
        add_text(slide, action, x + 0.10, 5.72, 1.95, 0.40,
                 size=10.8, color=INK, align=PP_ALIGN.CENTER)
    add_text(slide,
             "mx-gemmini-opt verifies typed IR. The standalone command path bypasses LLVM; "
             "the generic full-model host currently uses Merlin → LLVM → RV64.",
             0.96, 6.38, 11.45, 0.36, size=10.2, color=GREY)
    page_number(slide, 1, total)


def configurations(slide, total: int = 3) -> None:
    set_text(slide.shapes[0], "One JSON profile changes legal MLIR", size=27,
             bold=True)
    clear_content(slide)
    add_panel(slide, 0.94, 1.62, 6.04, 4.82, SAND)
    add_panel(slide, 7.16, 1.62, 5.25, 4.82, PALE_BLUE)
    add_text(slide, "Selected RTL profile fields", 1.15, 1.79,
             5.58, 0.34, size=17.2, bold=True, color=NAVY)
    add_code(slide,
             '{\n'
             ' "geometry": {"mesh_rows":16,"mesh_columns":16},\n'
             ' "legal_compute": [\n'
             '  {"activation_format":"fp4_e2m1",\n'
             '   "weight_format":"fp4_e2m1","pe_mode":0},\n'
             '  {"activation_format":"fp8_e4m3",\n'
             '   "weight_format":"fp8_e4m3","pe_mode":8}\n'
             ' ],\n'
             ' "resources": {"scratchpad_bytes":262144,\n'
             '               "vpu":true,"lut":false}\n'
             '}',
             1.15, 2.22, 5.57, 2.81, size=10.45)
    add_text(slide,
             "MxE4M3Fp4VpuGemminiRocketConfig · direct projections in both "
             "legal rows · two 8-lane BF16 VPUs.",
             1.15, 5.18, 5.57, 0.52, size=11.1, color=INK)
    add_text(slide,
             "RTL 266c593 + MxGen dba3e7 → 41 fragments / 40 SoC wrappers; "
             "each JSON stores source hashes.",
             1.15, 5.81, 5.57, 0.39, size=10.4, color=GREY)

    add_text(slide, "Before → after bind_profile", 7.38, 1.79,
             4.78, 0.34, size=17.2, bold=True, color=NAVY)
    add_code(slide,
             'before: contract {format = "mxfp4"}\n'
             'after:  contract {\n'
             '  activation_format = "fp4_e2m1",\n'
             '  weight_format = "fp4_e2m1",\n'
             '  pe_mode = 0 : i32,\n'
             '  profile_sha256 = "e9e33f..." }',
             7.38, 2.22, 4.79, 1.64, size=10.85)
    add_text(slide, "Compiler consequences", 7.38, 4.06,
             4.78, 0.30, size=14.1, bold=True, color=NAVY)
    add_text(slide,
             "• TinyLlama: 155 FP4 / 45 FP8 MX sites per stage.\n"
             "• PE 0 = direct FP4 × FP4; PE 8 = direct FP8 × FP8.\n"
             "• 256 KiB SPAD caps tiles; VPU/requant legal.\n"
             "• Selected profile: no operand LUT or FP6.\n"
             "• FP6×FP4 profile: E3M2 LUT × FP4 direct → PE 3; VPU off.\n"
             "• load_profile(profile, rtl_root=rtl) rejects drift.",
             7.38, 4.42, 4.78, 1.62, size=11.0, color=INK)
    add_text(slide,
             "npu-exploration supplies source kernels for per-mode Spike checks; "
             "the RTL profile alone is structural, not a performance claim.",
             0.96, 6.50, 11.45, 0.25, size=10.2, color=GREY)
    page_number(slide, 2, total)


def set_cell(cell, text: str, *, heading: bool = False, bold: bool = False,
             align=PP_ALIGN.CENTER, size: float = 14.0) -> None:
    cell.text = ""
    cell.margin_left = Inches(0.14)
    cell.margin_right = Inches(0.10)
    cell.margin_top = Inches(0.05)
    cell.margin_bottom = Inches(0.03)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = cell.text_frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.space_before = Pt(0)
    paragraph.space_after = Pt(0)
    run = paragraph.add_run()
    run.text = text
    run.font.name = "Inter"
    run.font.size = Pt(size)
    run.font.bold = heading or bold
    run.font.color.rgb = RGBColor(255, 255, 255) if heading else INK


def results(slide, smol_comparison: dict | None,
            tiny_run: dict | None, tiny_comparison: dict | None,
            layer_probe: dict | None, full_source: dict | None,
            safe_vpu: dict | None, full_vpu: dict | None,
            radiance: dict | None, prefill128: dict | None,
            equal_work: dict | None, tiny_logits: dict | None,
            smol_actions: dict | None, prefill_precision: dict | None,
            smol_reference_step: dict | None,
            smol_prefix_factor: dict | None,
            total: int = 3) -> None:
    shapes = slide.shapes
    title = ("MX results: full-model Spike" if tiny_logits or smol_actions
             else "MX results: Spike and numerical drift" if prefill_precision
             else "MX results: Spike and Radiance")
    set_text(shapes[0], title, size=27,
             bold=True)
    frame = shapes[1]
    table = frame.table
    table.rows[0].height = Inches(0.60)
    for row in list(table.rows)[1:]:
        row.height = Inches(0.63)
    smol_compiler = (smol_comparison["reference_cycles"]
                     if smol_comparison else None)
    smol_native = (smol_comparison["candidate_cycles"]
                   if smol_comparison else None)
    tiny_compiler = (tiny_run["total_performance"]["cycles"]
                     if tiny_run else None)
    tiny_native = (tiny_comparison["candidate_cycles"]
                   if tiny_comparison else None)

    def delta(compiler: int | None, native: int | None) -> str:
        return f"{(compiler / native - 1) * 100:+.3f}%" if compiler and native else "—"

    layer_row = (
        ("TinyLlama 128-token prefill · independent",
         f"{prefill128['stage_cycles']['compiler']:,}†",
         f"{prefill128['stage_cycles']['source']:,}†", "NO RATIO")
        if prefill128 else
        ("TinyLlama 64-token prefill · independent",
         f"{full_vpu['compiler_stage_cycles']:,}†",
         f"{full_vpu['source_stage_cycles']:,}†", "NO RATIO")
        if full_vpu else
        ("TinyLlama 64 tokens · functional Spike",
         f"{safe_vpu['stage_cycles']:,}†",
         f"{full_source['cycles']['selected_sum']:,}†", "NO RATIO")
        if full_source and safe_vpu else
        ("TinyLlama 64 tokens · source vs compiler",
         f"{full_source['compiler_stage_cycles']:,}†",
         f"{full_source['cycles']['selected_sum']:,}†", "NO RATIO")
        if full_vpu or full_source else
        ("TinyLlama cached layer · unpaired",
         f"{layer_probe['compiler_runs']['fp8_prepacked']['cycles']:,}†",
         f"{layer_probe['source_layer']['cycles']:,}†", "NO RATIO")
        if layer_probe else
        ("SmolVLA flow 1 step · issuer swap", "2,112,653,481",
         "2,112,919,925", "−0.0126%"))
    values = (
        ("Workload / compared code", "Compiler cycles", "Control/source cycles", "Compiler Δ"),
        ("SmolVLA full 12 steps · issuer swap",
         f"{smol_compiler:,}*" if smol_compiler else "pending",
         f"{smol_native:,}" if smol_native else "running",
         delta(smol_compiler, smol_native)),
        ("TinyLlama full 33 steps · issuer swap",
         f"{tiny_compiler:,}*" if tiny_compiler else "running",
         f"{tiny_native:,}" if tiny_native else "pending",
         delta(tiny_compiler, tiny_native)),
        layer_row,
        ("Nicolas dense attention · 64×256×128",
         f"{equal_work['compiler_modeled_cycles']:,}" if equal_work else "43,523",
         f"{equal_work['source_modeled_cycles']:,}" if equal_work else "43,641",
         f"{(equal_work['compiler_modeled_cycles'] / equal_work['source_modeled_cycles'] - 1) * 100:+.2f}%"
         if equal_work else "−0.27%"),
    )
    for row_index, row in enumerate(values):
        for col_index, text in enumerate(row):
            set_cell(table.cell(row_index, col_index), text,
                     heading=row_index == 0,
                     bold=row_index in (1, 2, 3) and col_index == 0,
                     align=PP_ALIGN.LEFT if col_index == 0 else PP_ALIGN.CENTER,
                     size=11.2 if row_index == 0 else 12.4)
    add_shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.94),
                                       Inches(5.10), Inches(11.46), Inches(0.61))
    add_shape.fill.solid()
    add_shape.fill.fore_color.rgb = PALE_BLUE
    add_shape.line.fill.background()
    step_comparison = None
    if smol_reference_step:
        full_cosine = smol_reference_step["metrics"]["full_vs_fp32"]["cosine"]
        reset_cosine = smol_reference_step["metrics"]["isolated_vs_fp32"]["cosine"]
        if smol_prefix_factor:
            prefix_cosine = smol_prefix_factor["metrics"]["factor_vs_fp32"]["cosine"]
            step_comparison = (
                "SmolVLA step 9 vs FP32: full "
                f"{full_cosine:.3f}; FP32 prefix only {prefix_cosine:.3f}; "
                f"FP32 prefix + FP32 state {reset_cosine:.3f}."
            )
        else:
            step_comparison = (
                "SmolVLA step 9 vs FP32: full "
                f"{full_cosine:.3f}; replay on FP32 cache/state "
                f"{reset_cosine:.3f} (prefix + history reset)."
            )
    if tiny_logits and smol_actions:
        prefill = tiny_logits["prefill_metrics"]
        decode = tiny_logits["decode_summary"]
        action = smol_actions["final_action_metrics"]
        first_flow = smol_actions["flow_step_metrics"][0]
        callout = (
            "FP32 drift on fixed synthetic inputs: TinyLlama prefill logits "
            f"cosine {prefill['cosine']:.3f}; decode mean {decode['cosine_mean']:.3f}, "
            f"top token {decode['top_token_matches']}/32.\n" +
            (step_comparison if step_comparison else
             "SmolVLA flow 0 → final action cosine "
             f"{first_flow['cosine']:.3f} → {action['cosine']:.3f}; "
             f"final normalized RMSE {action['normalized_rmse']:.3f}.")
        )
    elif tiny_logits:
        prefill = tiny_logits["prefill_metrics"]
        decode = tiny_logits["decode_summary"]
        control = ("All-FP8 prefill control cosine "
                   f"{prefill_precision['metrics']['all_fp8_spike']['cosine']:.3f}; "
                   if prefill_precision else "")
        callout = (
            "TinyLlama logits vs captured FP32 on fixed synthetic inputs: "
            f"prefill cosine {prefill['cosine']:.3f}; decode mean "
            f"{decode['cosine_mean']:.3f}, top token "
            f"{decode['top_token_matches']}/32.\n"
            f"{control}output hashes match the timed run; "
            "no generation-quality claim."
        )
    elif smol_actions and prefill_precision:
        mixed = prefill_precision["metrics"]["mixed_fp4_fp8_spike"]
        fp8 = prefill_precision["metrics"]["all_fp8_spike"]
        action = smol_actions["final_action_metrics"]
        first_flow = smol_actions["flow_step_metrics"][0]
        callout = (
            "TinyLlama prefill vs FP32: mixed FP4/FP8 cosine "
            f"{mixed['cosine']:.3f}, all-FP8 {fp8['cosine']:.3f}. "
            "33-step logits pending.\n" +
            (step_comparison if step_comparison else
             "SmolVLA flow 0 → final action vs FP32: cosine "
             f"{first_flow['cosine']:.3f} → {action['cosine']:.3f}; "
             f"final normalized RMSE {action['normalized_rmse']:.3f}.")
        )
    elif smol_actions:
        action = smol_actions["final_action_metrics"]
        first_flow = smol_actions["flow_step_metrics"][0]
        callout = (
            "SmolVLA flow 0 → final action vs captured FP32: cosine "
            f"{first_flow['cosine']:.3f} → {action['cosine']:.3f}; "
            f"final normalized RMSE {action['normalized_rmse']:.3f}.\n" +
            (step_comparison if step_comparison else
             "Full-session output hashes match the timed compiler run; "
             "these values do not establish task success.")
        )
    elif prefill_precision:
        mixed = prefill_precision["metrics"]["mixed_fp4_fp8_spike"]
        fp8 = prefill_precision["metrics"]["all_fp8_spike"]
        callout = (
            "Separate TinyLlama prefill vs FP32: mixed FP4/FP8 Spike cosine "
            f"{mixed['cosine']:.3f}; all-FP8 {fp8['cosine']:.3f}. "
            "Neither selects the FP32 top token.\n"
            "Remaining 32 decode logits and final SmolVLA actions pending; "
            "the independent cycle row has no valid speedup ratio."
        )
    else:
        callout = (
            "Radiance GEMMs: compiler matches 16,384 BF16 each; FP8/FP4 "
            "read-once B = 128/176 KiB versus 512 KiB FP8 re-stream.\n"
            "Radiance GQA: compiled MX PV matches 65,536 source-produced BF16 "
            "values. Muon source cycles remain unpaired."
            if radiance else
            "Full sessions: same host/VPU program; only the MX issuer changes. "
            "Near parity is expected.\n"
            "† Independent prefill: different host/VPU work, no speedup ratio. "
            "Fused attention: one kernel."
            if full_source else
            "Issuer-swap rows change only the MX command issuer; host and VPU "
            "code is the same.\n"
            "† The layer uses the same tokens and weights but different timed work "
            "and tensor formats. Its cycle ratio is invalid."
            if layer_probe else
            "Issuer-swap rows change only the MX command issuer; host and VPU "
            "code is the same.\n"
            "Nicolas's fused MX+VPU kernels have not yet been stitched and timed "
            "as complete models.")
    set_text(add_shape, callout,
             size=10.6, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    footer = shapes[2]
    footer.top = Inches(5.91)
    footer.height = Inches(0.66)
    if tiny_comparison and smol_comparison and prefill128:
        agreement = prefill128["comparisons"]["source_vs_compiler"]
        footer_text = (
            "* Issuer swap: host/VPU fixed; modeled cycles. Output hashes "
            "equal for SmolVLA 33/33 and TinyLlama 99/99.\n"
            "† All-FP8 prefill: 200 weights and 128 IDs match; cosine "
            f"{agreement['cosine']:.3f}, top tokens "
            f"{agreement['a_argmax']}/{agreement['b_argmax']}. "
            "Host/VPU differ; source 128 keys, compiler 160 slots; no ratio."
        )
    elif tiny_comparison and smol_comparison and full_vpu:
        agreement = full_vpu["comparisons"]["source_vpu_vs_compiler"]
        footer_text = (
            "* Issuer swap: host/VPU fixed; modeled cycles. Output hashes "
            "equal for SmolVLA 33/33 and TinyLlama 99/99.\n"
            "† Full prefill: 200 weights and 64 IDs match; cosine "
            f"{agreement['cosine']:.3f}, token {agreement['a_argmax']}. "
            "VPU adds: 44 source calls, 88 compiler regions; host math differs."
        )
    elif tiny_comparison and smol_comparison and full_source and safe_vpu:
        footer_text = (
            "* Issuer-swap rows use modeled e4m3_vpu cycles; hashes match "
            "SmolVLA 33/33 and TinyLlama 99/99.\n"
            f"† 64-token row uses functional rdcycle (timing model off); "
            f"source/compiler logits cosine "
            f"{safe_vpu['comparisons']['compiler_vs_source_quant']['cosine']:.3f}. "
            "Dense attention uses modeled cycles."
        )
    elif tiny_comparison and smol_comparison and full_source:
        footer_text = (
            "* Issuer-swap hashes match: SmolVLA 33/33; TinyLlama 99/99. "
            "This is not an independent accuracy check.\n"
            f"† Independent source vs compiler logits cosine "
            f"{full_source['compiler_source_logit_cosine']:.3f}. Dense "
            "attention is one fused kernel. Spike cycles are not FPGA latency."
        )
    elif tiny_comparison and smol_comparison and layer_probe:
        compiler_error = layer_probe["compiler_runs"]["fp8_prepacked"]["rel_fro_vs_pytorch"]
        source_error = layer_probe["source_layer"]["rel_fro_vs_pytorch"]
        footer_text = (
            "* Issuer-swap output hashes match: SmolVLA 33/33; TinyLlama 99/99. "
            "This is not an independent model accuracy check.\n"
            f"† Layer error vs PyTorch: compiler {compiler_error:.2%}; "
            f"source {source_error:.2%}. Dense attention is one fused kernel. "
            "Spike cycles are not FPGA latency."
        )
    elif tiny_comparison and smol_comparison:
        footer_text = (
            "* Output hashes match: SmolVLA 33/33; TinyLlama 99/99. This checks "
            "the captured outputs, not model accuracy.\n"
            "e4m3_vpu Spike cycles are not FPGA latency. Dense attention "
            "compares one fused source kernel, not a full model."
        )
    else:
        tiny_status = ("TinyLlama full receipt not supplied; "
                       "no model cycle total.")
        smol_status = ("Full SmolVLA issuer-swap outputs match (33/33)."
                       if smol_comparison else
                       "Full SmolVLA issuer-swap parity awaits its source run.")
        footer_text = (
            "* Completed timing receipt; independent quantized accuracy remains open. "
            f"{tiny_status}\n"
            "Nicolas e4m3_vpu Spike, host tracking off. Paired flow/attention "
            f"output hashes match. {smol_status}"
        )
    set_text(
        footer,
        footer_text,
        size=9.6, color=GREY, align=PP_ALIGN.LEFT,
    )
    page_number(slide, 3, total)


def radiance_results(slide, radiance: dict) -> None:
    add_text(slide, "Radiance MX: output and schedule",
             2.75, 0.48, 7.84, 0.55, size=25.4, bold=True,
             color=INK, align=PP_ALIGN.CENTER)
    rows = radiance["rows"]
    gemv = radiance["batched_gemv_rows"]
    def shape(row: dict) -> str:
        return "×".join(map(str, row["shape_mnk"]))

    def traffic(row: dict) -> str:
        return (f"{row['compiler_weight_bytes'] // 1024} KiB · "
                f"{row['compiler_weight_tile_reads']} reads")

    entries = [
        ("FP8 GEMM · read once", shape(rows[0]),
         f"{rows[0]['compiler_spike_bf16_values_matching_source_golden']:,} BF16; "
         "one 256×64 tile", traffic(rows[0])),
        ("FP4 down-projection · read once", shape(rows[1]),
         f"{rows[1]['compiler_spike_bf16_values_matching_source_golden']:,} BF16; "
         "one 256×64 tile", traffic(rows[1])),
        ("FP8 GEMM · four-pass re-stream", shape(rows[2]),
         f"{rows[2]['compiler_spike_bf16_values_matching_source_golden']:,} BF16; "
         "four 64×64 tiles", traffic(rows[2])),
        ("Batched GEMV · FP8/FP4", "4 source shapes",
         f"{sum(r['compiler_spike_bf16_values_matching_source_golden'] for r in gemv):,} "
         "BF16 across four controls", "one-read controls"),
        ("GQA attention · MX PV only", "16 cutpoints",
         f"{radiance['gqa_compiler_spike_pv_bf16_values_matched']:,} BF16 "
         "PV outputs with source P tiles", "whole kernel open"),
    ]
    table_shape = slide.shapes.add_table(
        len(entries) + 1, 4, Inches(0.94), Inches(1.67),
        Inches(11.46), Inches(3.24))
    table = table_shape.table
    for column, width in enumerate((3.1, 1.65, 4.1, 2.61)):
        table.columns[column].width = Inches(width)
    for index in range(len(entries) + 1):
        table.rows[index].height = Inches(0.54)
    for row_index, entry in enumerate(
            [("Radiance source kernel", "Shape M×N×K",
              "Compiler check on Nicolas Spike", "B traffic / limit"), *entries]):
        for col_index, value in enumerate(entry):
            set_cell(table.cell(row_index, col_index), value,
                     heading=row_index == 0,
                     bold=row_index > 0 and col_index == 0,
                     align=PP_ALIGN.LEFT if col_index in (0, 2) else PP_ALIGN.CENTER,
                     size=10.5 if row_index == 0 else 10.9)
    add_code(slide,
             'Radiance: TILE_M=256, TILE_N=64, TILE_K=64, FP8 read once\n'
             f"Compiler: tile=[{','.join(map(str, rows[0]['compiler_physical_tile_mnk']))}], "
             f"weight_reads={rows[0]['compiler_weight_tile_reads']}",
             0.94, 5.03, 5.6, 0.75, size=9.5)
    add_code(slide,
             'Radiance: four 64-row passes, B re-streamed each pass\n'
             f"Compiler: {rows[2]['compiler_output_tiles']} output tiles, "
             f"weight_reads={rows[2]['compiler_weight_tile_reads']}",
             6.73, 5.03, 5.67, 0.75, size=9.5)
    add_text(slide,
             "We supplied Radiance's packed data and tile sizes. The compiler "
             "matched output and weight traffic; it did not discover the "
             "schedule on its own. Radiance runs on Muon+MX, this check on "
             "Rocket+MX+VPU Spike, so there is no cycle ratio. GQA PV is one stage.",
             0.94, 6.02, 11.46, 0.62, size=10.2, color=GREY)
    page_number(slide, 4, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smol-full-comparison", type=Path)
    parser.add_argument("--tiny-full-run", type=Path)
    parser.add_argument("--tiny-full-comparison", type=Path)
    parser.add_argument("--tiny-layer-probe", type=Path)
    parser.add_argument("--tiny-full-source", type=Path)
    parser.add_argument("--tiny-full-safe-vpu", type=Path)
    parser.add_argument("--tiny-full-vpu", type=Path)
    parser.add_argument("--radiance-comparison", type=Path)
    parser.add_argument("--tiny-prefill128", type=Path)
    parser.add_argument("--equal-work-attention", type=Path)
    parser.add_argument("--tiny-full-logits", type=Path)
    parser.add_argument("--smol-action-values", type=Path)
    parser.add_argument("--smol-reference-step", type=Path)
    parser.add_argument("--smol-prefix-factor", type=Path)
    parser.add_argument("--tiny-prefill-precision", type=Path)
    args = parser.parse_args()
    if (args.tiny_full_run is None) != (args.tiny_full_comparison is None):
        parser.error("TinyLlama needs both the full run and equal-output comparison")
    smol_comparison = (json.loads(args.smol_full_comparison.read_text())
                       if args.smol_full_comparison else None)
    tiny_run = (json.loads(args.tiny_full_run.read_text())
                if args.tiny_full_run else None)
    tiny_comparison = (json.loads(args.tiny_full_comparison.read_text())
                       if args.tiny_full_comparison else None)
    layer_probe = (json.loads(args.tiny_layer_probe.read_text())
                   if args.tiny_layer_probe else None)
    full_source = (json.loads(args.tiny_full_source.read_text())
                   if args.tiny_full_source else None)
    safe_vpu = (json.loads(args.tiny_full_safe_vpu.read_text())
                if args.tiny_full_safe_vpu else None)
    full_vpu = (json.loads(args.tiny_full_vpu.read_text())
                if args.tiny_full_vpu else None)
    radiance = (json.loads(args.radiance_comparison.read_text())
                if args.radiance_comparison else None)
    prefill128 = (json.loads(args.tiny_prefill128.read_text())
                  if args.tiny_prefill128 else None)
    equal_work = (json.loads(args.equal_work_attention.read_text())
                  if args.equal_work_attention else None)
    tiny_logits = (json.loads(args.tiny_full_logits.read_text())
                   if args.tiny_full_logits else None)
    smol_actions = (json.loads(args.smol_action_values.read_text())
                    if args.smol_action_values else None)
    smol_reference_step = (json.loads(args.smol_reference_step.read_text())
                           if args.smol_reference_step else None)
    smol_prefix_factor = (json.loads(args.smol_prefix_factor.read_text())
                          if args.smol_prefix_factor else None)
    prefill_precision = (json.loads(args.tiny_prefill_precision.read_text())
                         if args.tiny_prefill_precision else None)
    if prefill_precision is not None:
        values = prefill_precision.get("metrics") or {}
        mixed = values.get("mixed_fp4_fp8_spike") or {}
        fp8 = values.get("all_fp8_spike") or {}
        artifacts = prefill_precision.get("artifact_sha256") or {}
        if (tiny_run is None
                or prefill_precision.get("schema") !=
                "mx_gemmini.tinyllama_prefill_precision.v1"
                or prefill_precision.get("numerical_qualification") !=
                "not_established"
                or prefill_precision.get("mixed_prefill_matches_completed_fast_run")
                is not True
                or prefill_precision.get("mixed_fast_run_sha256") !=
                file_sha256(args.tiny_full_run)
                or not artifacts
                or any(file_sha256(args.tiny_prefill_precision.parent / name)
                       != digest for name, digest in artifacts.items())
                or any(not isinstance(row.get("cosine"), (int, float))
                       or not math.isfinite(row["cosine"])
                       or not -1 <= row["cosine"] <= 1
                       for row in (mixed, fp8))):
            parser.error("TinyLlama prefill precision needs the verified archive")
    for name, receipt, source, schema in (
            ("TinyLlama logits", tiny_logits, args.tiny_full_logits,
             "mx_gemmini.tinyllama_full_logits.v1"),
            ("SmolVLA actions", smol_actions, args.smol_action_values,
             "mx_gemmini.smolvla_full_action_values.v1")):
        if receipt is None:
            continue
        artifacts = receipt.get("artifact_sha256") or {}
        if (receipt.get("schema") != schema
                or receipt.get("numerical_qualification") != "not_established"
                or receipt.get("output_digests_match_fast_compiler") is not True
                or not artifacts
                or any(file_sha256(source.parent / path)
                       != digest for path, digest in artifacts.items())):
            parser.error(f"{name} needs a verified output-value archive")
    if tiny_logits is not None:
        prefill = tiny_logits.get("prefill_metrics") or {}
        decode = tiny_logits.get("decode_summary") or {}
        if (tiny_run is None or tiny_comparison is None
                or tiny_logits.get("fast_run_sha256") !=
                file_sha256(args.tiny_full_run)
                or tiny_logits.get("fast_compiler_cycles") !=
                tiny_run["total_performance"]["cycles"]
                or decode.get("steps") != 32
                or type(decode.get("top_token_matches")) is not int
                or not 0 <= decode["top_token_matches"] <= 32
                or any(not isinstance(value, (int, float))
                       or not math.isfinite(value) or not -1 <= value <= 1
                       for value in (prefill.get("cosine"),
                                     decode.get("cosine_mean")))):
            parser.error("TinyLlama logits do not match the selected full run")
    if smol_actions is not None:
        action = smol_actions.get("final_action_metrics") or {}
        flow = smol_actions.get("flow_step_metrics") or []
        if (smol_comparison is None
                or smol_actions.get("fast_run_sha256") !=
                smol_comparison.get("reference_run_sha256")
                or smol_actions.get("fast_compiler_cycles") !=
                smol_comparison.get("reference_cycles")
                or len(flow) != 10
                or [row.get("step") for row in flow] != list(range(10))
                or flow[-1] != action
                or any(not isinstance(row.get("cosine"), (int, float))
                       or not math.isfinite(row["cosine"])
                       or not -1 <= row["cosine"] <= 1
                       or not isinstance(row.get("normalized_rmse"), (int, float))
                       or not math.isfinite(row["normalized_rmse"])
                       or row["normalized_rmse"] < 0
                       for row in flow)
                or not isinstance(action.get("cosine"), (int, float))
                or not math.isfinite(action["cosine"])
                or not -1 <= action["cosine"] <= 1
                or not isinstance(action.get("normalized_rmse"), (int, float))
                or not math.isfinite(action["normalized_rmse"])
                or action["normalized_rmse"] < 0):
            parser.error("SmolVLA actions do not match the selected full run")
    if smol_reference_step is not None:
        artifacts = smol_reference_step.get("artifact_sha256") or {}
        metrics = smol_reference_step.get("metrics") or {}
        isolated = metrics.get("isolated_vs_fp32") or {}
        full = metrics.get("full_vs_fp32") or {}
        if (smol_actions is None
                or smol_reference_step.get("schema") !=
                "mx_gemmini.smolvla_reference_step_replay_result.v1"
                or smol_reference_step.get("reference_step") != 9
                or smol_reference_step.get("cycle_ratio_valid") is not False
                or not artifacts
                or any(file_sha256(args.smol_reference_step.parent / path) != digest
                       for path, digest in artifacts.items())
                or smol_reference_step.get("timestep_diagnostic", {}).get("matched_steps") != 10
                or any(not isinstance(row.get("cosine"), (int, float))
                       or not math.isfinite(row["cosine"])
                       or not -1 <= row["cosine"] <= 1
                       for row in (isolated, full))
                or full.get("cosine") !=
                smol_actions["final_action_metrics"]["cosine"]):
            parser.error("SmolVLA isolated replay needs the verified full action archive")
    if smol_prefix_factor is not None:
        if smol_actions is None or smol_reference_step is None:
            parser.error("SmolVLA prefix control needs the full action and isolated replay")
        folder = args.smol_prefix_factor.parent
        artifacts = smol_prefix_factor.get("artifact_sha256") or {}
        values = smol_prefix_factor.get("metrics") or {}
        derived = json.loads((folder / "derived_replay_receipt.json").read_text())
        control_link = json.loads((folder / "link.json").read_text())
        full_link = json.loads((args.smol_action_values.parent / "tap_link.json").read_text())
        if (smol_prefix_factor.get("schema") !=
                "mx_gemmini.smolvla_prefix_factor_result.v1"
                or smol_prefix_factor.get("cycle_ratio_valid") is not False
                or smol_prefix_factor.get("numerical_qualification") !=
                "not_established"
                or not artifacts
                or any(file_sha256(folder / path) != digest
                       for path, digest in artifacts.items())
                or derived.get("compiled_run_sha256") !=
                file_sha256(args.smol_action_values.parent / "tap_run.json")
                or control_link["stage_object_sha256"]["flow_denoise"] !=
                full_link["stage_object_sha256"]["flow_denoise"]
                or control_link["profile_sha256"] != full_link["profile_sha256"]
                or control_link["shared_weight_sha256"] !=
                full_link["shared_weight_sha256"]
                or set(values) !=
                {"factor_vs_fp32", "full_vs_fp32", "factor_vs_full"}
                or any(not isinstance(row.get("cosine"), (int, float))
                       or not math.isfinite(row["cosine"])
                       or not -1 <= row["cosine"] <= 1
                       for row in values.values())
                or abs(values["full_vs_fp32"]["cosine"] -
                       smol_actions["final_action_metrics"]["cosine"]) > 1e-9):
            parser.error("SmolVLA prefix control needs its verified full action archive")
    if equal_work is not None:
        evidence_dir = args.equal_work_attention.parent
        if (equal_work.get("schema") !=
                "mx_gemmini.nicolas_dense_attention_shape_qualification.v2"
                or equal_work.get("status") !=
                "equal_work_typed_attention_matches_source_on_spike"
                or equal_work.get("shape_sq_sk_d") != [64, 256, 128]
                or equal_work.get("spike_reproduced_runs") != 2
                or any(type(equal_work.get(key)) is not int or equal_work[key] <= 0
                       for key in ("source_modeled_cycles",
                                   "compiler_modeled_cycles"))
                or hashlib.sha256((evidence_dir / "source_spike.log").read_bytes()).hexdigest()
                != equal_work.get("source_spike_log_sha256")
                or hashlib.sha256((evidence_dir / "spike_1.log").read_bytes()).hexdigest()
                != equal_work.get("spike_log_sha256")):
            parser.error("dense attention needs completed equal-work Spike evidence")
    if prefill128 is not None:
        pair = prefill128.get("comparisons", {}).get("source_vs_compiler", {})
        cycles = prefill128.get("stage_cycles", {})
        evidence_dir = args.tiny_prefill128.parent
        artifact_hashes = prefill128.get("artifact_sha256") or {}
        if (prefill128.get("schema") != "mx_gemmini.tinyllama_prefill128_source.v1"
                or prefill128.get("cycle_ratio_valid") is not False
                or prefill128.get("source_output_matches_generated_quantized_logits") is not True
                or prefill128.get("checkpoint_fp32_tensors_matched") != 200
                or prefill128.get("source_attention_valid_keys") != 128
                or prefill128.get("compiler_attention_key_slots") != 160
                or prefill128.get("perf_config") != "e4m3_vpu"
                or any(type(cycles.get(key)) is not int or cycles[key] <= 0
                       for key in ("source", "compiler"))
                or not 0 <= pair.get("cosine", -1) <= 1
                or not {"source_spike.log.gz", "compiler_spike.log.gz",
                        "source_generated_last_bf16.bin"} <= set(artifact_hashes)
                or any(hashlib.sha256((evidence_dir / name).read_bytes()).hexdigest()
                       != expected for name, expected in artifact_hashes.items())):
            parser.error("128-token prefill needs completed independent evidence")
        source_log = gzip.decompress(
            (evidence_dir / "source_spike.log.gz").read_bytes()).decode()
        compiler_log = gzip.decompress(
            (evidence_dir / "compiler_spike.log.gz").read_bytes()).decode()
        source_cycles = re.search(r"^stage_cycles64 0x([0-9a-f]+)$",
                                  source_log, re.M)
        compiler_cycles = re.search(r"^METRIC cycles (\d+)$",
                                    compiler_log, re.M)
        if (source_cycles is None or compiler_cycles is None
                or int(source_cycles.group(1), 16) != cycles["source"]
                or int(compiler_cycles.group(1)) != cycles["compiler"]
                or "llama MODEL benchmark finished (22 layer(s) + lm_head)."
                not in source_log
                or not re.search(r"^DONE$", compiler_log, re.M)):
            parser.error("128-token prefill logs do not support the slide counts")
    if radiance is not None:
        rows = radiance.get("rows") or []
        if (radiance.get("schema") != "mx_gemmini.radiance_kernel_comparison.v1"
                or radiance.get("paired_source_compiler_cycles") is not False
                or radiance.get("cycle_ratio_valid") is not False
                or len(rows) != 3
                or [r.get("precision") for r in rows] != ["fp8", "fp4", "fp8"]
                or any(r.get("compiler_spike_bf16_values_matching_source_golden")
                       != 16384 for r in rows)
                or [r.get("compiler_weight_bytes") for r in rows]
                != [131072, 180224, 524288]
                or [r.get("compiler_weight_tile_reads") for r in rows]
                != [32, 88, 128]
                or [r.get("shape_mnk") for r in rows]
                != [[256, 64, 2048], [256, 64, 5632], [256, 64, 2048]]
                or [r.get("radiance_source_schedule") for r in rows]
                != ["read_once_weight_stationary",
                    "read_once_weight_stationary",
                    "four_pass_weight_restream"]
                or [r.get("compiler_physical_tile_mnk") for r in rows]
                != [[256, 64, 64], [256, 64, 64], [64, 64, 64]]
                or [r.get("compiler_output_tiles") for r in rows]
                != [1, 1, 4]
                or [r.get("compiler_move_weight_steps") for r in rows]
                != [576, 880, 2304]
                or [r.get("compiler_spike_bf16_values_matching_source_golden")
                    for r in radiance.get("batched_gemv_rows", [])]
                != [4096, 8192, 16384, 16384]
                or radiance.get("gqa_compiler_spike_pv_bf16_values_matched")
                != 65536
                or any(hashlib.sha256(
                    (args.radiance_comparison.parent.parent / path).read_bytes()
                ).hexdigest() != radiance["evidence"][key]
                       for path, key in (
                           ("radiance_ws_read_once_1a7bc9c/index.json",
                            "read_once_index_sha256"),
                           ("radiance_restream_1df2c5f/index.json",
                            "restream_index_sha256"),
                           ("radiance_mx_vpu_legal_roster_266c593/index.json",
                            "legal_roster_index_sha256"),
                           ("radiance_batched_gemv_552a16b/index.json",
                            "batched_gemv_index_sha256"),
                           ("radiance_gqa_pv_roster_80f84ca/index.json",
                            "gqa_pv_index_sha256")))):
            parser.error("Radiance comparison needs source-bound command evidence")
    if full_vpu is not None:
        pair = (full_vpu.get("comparisons") or {}).get("source_vpu_vs_compiler") or {}
        if (full_vpu.get("schema") !=
                "mx_gemmini.nicolas_tinyllama_full_vpu_residual_spike.v1"
                or full_vpu.get("cycle_ratio_valid") is not False
                or full_vpu.get("source_timing_mode") != "both"
                or full_vpu.get("compiler_timing_mode") != "both"
                or full_vpu.get("compiler_counter_csr") != "cycle"
                or full_vpu.get("perf_config") != "e4m3_vpu"
                or full_vpu.get("exact_fp32_checkpoint_tensors_matched") != 200
                or full_vpu.get("source_vpu_matches_scalar_bits") is not True
                or full_vpu.get("source_stage_cycles", 0) <= 0
                or full_vpu.get("compiler_stage_cycles", 0) <= 0
                or not 0 <= pair.get("cosine", -1) <= 1
                or pair.get("a_argmax") != pair.get("b_argmax")):
            parser.error("independent full prefill needs matched modeled evidence")
    if safe_vpu is not None and full_source is None:
        parser.error("safe-VPU comparison also needs the independent source receipt")
    if full_source is not None:
        source_output = json.loads(
            args.tiny_full_source.with_name("output_comparison.json").read_text())
        cycles = full_source.get("cycles") or {}
        if (full_source.get("schema") !=
                "mx_gemmini.independent_tinyllama_m64_source_spike.v1"
                or full_source.get("cycle_ratio_valid") is not False
                or full_source.get("source_internal_bit_exact_checks_passed") is not True
                or type(cycles.get("selected_sum")) is not int
                or cycles["selected_sum"] <= 0
                or cycles.get("mx_mesh", 0) + cycles.get("scalar_host", 0) !=
                cycles["selected_sum"]
                or type(full_source.get("compiler_stage_cycles")) is not int
                or full_source["compiler_stage_cycles"] <= 0
                or not 0 <= full_source.get("compiler_source_logit_cosine", -1) <= 1
                or full_source.get("source_output_matches_prior_cycle_print_run") is not True
                or source_output.get("source_output_matches_generator_exact") is not True
                or source_output.get("compiler_cycles") !=
                full_source["compiler_stage_cycles"]
                or (source_output.get("comparisons") or {}).get(
                    "compiler_vs_source_quant", {}).get("cosine") !=
                full_source["compiler_source_logit_cosine"]):
            parser.error("full source needs completed unpaired evidence")
    if safe_vpu is not None:
        if (safe_vpu.get("schema") !=
                "mx_gemmini.tinyllama_full_safe_vpu_spike.v1"
                or safe_vpu.get("cycle_ratio_valid") is not False
                or safe_vpu.get("stage_cycles", 0) <= 0
                or safe_vpu.get("source_evidence_receipt_sha256") !=
                hashlib.sha256(args.tiny_full_source.read_bytes()).hexdigest()
                or safe_vpu.get("spike_extlib_sha256") !=
                full_source.get("spike_extlib_sha256")
                or any(safe_vpu.get(f"{key}_sha256") !=
                       full_source.get(f"compiler_{key}_sha256")
                       for key in ("capture_model", "checkpoint_weights",
                                   "precision_overrides", "profile"))
                or not 0 <= (safe_vpu.get("comparisons", {}).get(
                    "compiler_vs_source_quant", {}).get("cosine", -1)) <= 1):
            parser.error("safe-VPU result needs matched input and target evidence")
    if layer_probe is not None:
        source = layer_probe.get("source_layer") or {}
        compiled = (layer_probe.get("compiler_runs") or {}).get("fp8_prepacked") or {}
        if (layer_probe.get("schema") != "mx_gemmini.tinyllama_cached_layer_probe.v1"
                or layer_probe.get("timing_comparison_valid") is not False
                or type(source.get("cycles")) is not int or source["cycles"] <= 0
                or type(compiled.get("cycles")) is not int or compiled["cycles"] <= 0
                or compiled.get("completed") is not True):
            parser.error("cached layer needs completed unpaired evidence")
        source_path = args.tiny_layer_probe.parent / source["source_evidence"]
        source_receipt = json.loads(source_path.read_text())
        output = json.loads((args.tiny_layer_probe.parent / "output_metrics.json").read_text())
        if (source_receipt.get("source_layer_cycles") != source["cycles"]
                or source_receipt.get("source_rel_fro_hout_vs_pytorch_ppm") / 1e6
                != source.get("rel_fro_vs_pytorch")
                or output.get("output_sha256") != compiled.get("output_sha256")
                or output.get("compiler_to_m2m_golden_rel_fro")
                != compiled.get("rel_fro_vs_pytorch")):
            parser.error("cached layer numbers differ from source or output evidence")
    if tiny_run is not None and (
            tiny_run.get("complete") is not True or tiny_run.get("exit_code") != 0
            or len(tiny_run.get("step_performance", [])) != 33
            or len(tiny_run.get("output_digests", [])) != 99
            or not isinstance((tiny_run.get("total_performance") or {}).get("cycles"), int)):
        parser.error("TinyLlama full run is incomplete")
    for comparison, expected in ((smol_comparison, None),
                                 (tiny_comparison,
                                  tiny_run["total_performance"]["cycles"]
                                  if tiny_run else None)):
        if comparison is None:
            continue
        if (comparison.get("schema") != "mx_gemmini.equal_input_session_cycles.v1"
                or comparison.get("output_digest_parity") is not True
                or not isinstance(comparison.get("reference_cycles"), int)
                or comparison["reference_cycles"] <= 0
                or (expected is not None and comparison["reference_cycles"] != expected)
                or not isinstance(comparison.get("candidate_cycles"), int)
                or comparison["candidate_cycles"] <= 0):
            parser.error("full-session comparison lacks equal-output timing evidence")
        changed = comparison.get("changed_stage_objects")
        if (not isinstance(changed, list) or not changed
                or any(not isinstance(name, str)
                       or not name.endswith("/device_catalog/mx_catalog.o")
                       for name in changed)):
            parser.error("issuer-swap slide requires only MX catalog object changes")
    if tiny_comparison is not None and (
            tiny_comparison.get("reference_run_sha256") !=
            hashlib.sha256(args.tiny_full_run.read_bytes()).hexdigest()):
        parser.error("TinyLlama comparison is not bound to the selected run")
    prs = Presentation(args.template)
    keep_slides(prs)
    style_master(prs)
    total = 4 if radiance is not None else 3
    architecture(prs.slides[0], total)
    configurations(prs.slides[1], total)
    results(prs.slides[2], smol_comparison, tiny_run, tiny_comparison,
            layer_probe, full_source, safe_vpu, full_vpu, radiance, prefill128,
            equal_work, tiny_logits, smol_actions, prefill_precision,
            smol_reference_step, smol_prefix_factor, total)
    if radiance is not None:
        radiance_results(prs.slides.add_slide(prs.slides[2].slide_layout),
                         radiance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(args.out)


if __name__ == "__main__":
    main()
