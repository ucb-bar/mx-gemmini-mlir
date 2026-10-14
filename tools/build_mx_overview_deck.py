"""Build the editable MX Gemmini overview from the BAR slide template.

Requires python-pptx. All diagrams, labels, callouts, and table cells remain
native PowerPoint objects; the university seal is the template's master image.
"""

from __future__ import annotations

import argparse
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


def page_number(slide, number: int) -> None:
    add_text(slide, f"{number:02d} / 03", 11.45, 6.88, 1.35, 0.28,
             size=10, color=GREY, align=PP_ALIGN.RIGHT)


def architecture(slide) -> None:
    shapes = slide.shapes
    set_text(shapes[0], "MX Gemmini dialect to executable", size=27, bold=True)
    set_text(shapes[2],
             "TYPED TARGET IR  →  PROFILED SCHEDULE  →  RISC-V COMMAND PROGRAM",
             size=12, bold=True, color=GREY, align=PP_ALIGN.LEFT)
    for index, label, color in ((4, "Typed MX IR", GREY),
                                (11, "Legality + schedule", NAVY),
                                (18, "Binary + Spike", RGBColor(117, 132, 106))):
        set_text(shapes[index], label, size=18, bold=True, color=color)

    boxes = {
        6: "resource · contract\nupload_lut · encode",
        8: "vpu_execute · spad_requant\nreadout · wait",
        13: "RTL profile + payload\nPE tuple · capacity checks",
        15: "precision + tiling policy\nschedule.plan · native loops",
        20: "physical_program → RoCC\nDMA · compute · VPU",
        22: "RoCC C + data asm\nRV64 GCC → ELF → Spike",
    }
    for index, text in boxes.items():
        shape = shapes[index]
        shape.top = Inches(3.71 if index in (6, 13, 20) else 4.51)
        shape.height = Inches(0.68)
        set_text(shape, text, size=11.1, bold=True, color=INK)
    footer = shapes[23]
    footer.top = Inches(5.63)
    footer.height = Inches(0.51)
    set_text(
        footer,
        "Merlin transforms: MXBF16CatalogPreparation + MXVPUPostOffload; mx-gemmini-opt verifies typed IR.\n"
        "Standalone commands bypass LLVM; the full-model host currently uses Merlin → LLVM → RV64.",
        size=10.2, color=GREY, align=PP_ALIGN.LEFT,
    )
    page_number(slide, 1)


def configurations(slide) -> None:
    shapes = slide.shapes
    set_text(shapes[0], "One dialect, many MX configurations", size=27,
             bold=True)
    stages = (
        (3, 4, "RTL + MxGen", "41 MX fragments\n40 SoC wrappers\npinned source hashes"),
        (8, 9, "Profile JSON", "mesh / DMA / queues\nLUT + scales + VPU\nlegal PE tuples"),
        (13, 14, "Typed legality", "format + projection\nDIM / SRAM bounds\nreject invalid modes"),
        (18, 19, "Schedule + emit", "schedule.plan + tiles\nSPAD / DMA / loops\nRoCC command IR"),
        (23, 24, "Spike qualify", "Nicolas kernels\nnumeric + cycle receipts\nper config / mode"),
    )
    for heading, body, title, detail in stages:
        set_text(shapes[heading], title, size=17.2, bold=True)
        set_text(shapes[body], detail, size=13.0, color=INK)
    add_text(
        slide,
        "RTL/MxGen defines capabilities; npu-exploration supplies reference kernels. "
        "Selected DIM16 MX+VPU: E4M3 + FP4,\n"
        "two 8-lane BF16 VPUs and SPAD_REQUANT. FP6 is separate; "
        "each profile needs numerical qualification.",
        0.95, 6.05, 10.55, 0.61, size=10.2, color=GREY,
    )
    page_number(slide, 2)


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


def results(slide) -> None:
    shapes = slide.shapes
    set_text(shapes[0], "What paired Spike runs show", size=27, bold=True)
    frame = shapes[1]
    table = frame.table
    table._tbl.remove(table.rows[len(table.rows) - 1]._tr)
    table.rows[0].height = Inches(0.66)
    for row in list(table.rows)[1:]:
        row.height = Inches(0.72)
    values = (
        ("Paired workload", "Compiler cycles", "Nicolas cycles", "Compiler Δ"),
        ("Dense attention 64×256×128", "43,523", "43,641", "−0.27%"),
        ("SmolVLA flow · one step", "2,112,653,481", "2,112,919,925", "−0.0126%"),
        ("SmolVLA · 12-step matrix swap", "72,066,544,354*", "running", "—"),
    )
    for row_index, row in enumerate(values):
        for col_index, text in enumerate(row):
            set_cell(table.cell(row_index, col_index), text,
                     heading=row_index == 0,
                     bold=row_index == 3 and col_index == 0,
                     align=PP_ALIGN.LEFT if col_index == 0 else PP_ALIGN.CENTER,
                     size=12.2 if row_index == 0 else 13.0 if col_index == 0 else 13.1)
    # Keep the template's table fills while highlighting the active full run.
    add_shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.94),
                                       Inches(5.08), Inches(11.46), Inches(0.64))
    add_shape.fill.solid()
    add_shape.fill.fore_color.rgb = PALE_BLUE
    add_shape.line.fill.background()
    set_text(add_shape,
             "Paired kernel / flow results match outputs. The source swap changes MX issuers only; "
             "no matched full fused handwritten MX+VPU model baseline exists yet.",
             size=11.5, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    footer = shapes[2]
    footer.top = Inches(5.91)
    footer.height = Inches(0.66)
    set_text(
        footer,
        "* Completed compiler timing receipt; full-session numeric qualification pending. "
        "Spike e4m3_vpu model, host tracking off.\n"
        "Evidence: docs/evidence/attn_vpu_equal_work_20261013 and "
        "docs/evidence/nicolas_matrix_substitution_smolvla_20261013.",
        size=10.0, color=GREY, align=PP_ALIGN.LEFT,
    )
    page_number(slide, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prs = Presentation(args.template)
    keep_slides(prs)
    style_master(prs)
    architecture(prs.slides[0])
    configurations(prs.slides[1])
    results(prs.slides[2])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(args.out)


if __name__ == "__main__":
    main()
