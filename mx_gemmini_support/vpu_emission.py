"""Emit the existing single-VPU runtime-buffer command program and its ABI.

This is source emission, not native object construction or execution evidence.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .command_ir import emit_c
from .target_profile import profile_sha256
from .verify_profile_ir import verify_ir
from .vpu_elementwise_program import VpuElementwisePlan, lower_vpu_elementwise_program


@dataclass(frozen=True)
class VPUEmission:
    plan: VpuElementwisePlan
    c_source: str
    header: str
    physical: dict
    buffer_abi: list[dict]


def emit_vpu_elementwise(mlir_text: str, profile: dict, buffers: dict[str, str]) -> VPUEmission:
    """Lower exactly the graph family admitted by the handwritten object tool."""
    if profile.get("transport") != "rocket_rocc":
        raise ValueError("VPU elementwise object requires Rocket RoCC transport")
    report = verify_ir(mlir_text, profile)
    if ((report["contracts"], report["resident_contracts"],
         report["vpu_commands"], report["spad_requants"]) != (0, 0, 1, 0)):
        raise ValueError("VPU elementwise object needs one typed VPU operation")
    plan = lower_vpu_elementwise_program(mlir_text, profile, buffers)
    order = ("src1", "src2", "output") + (
        ("output2",) if plan.second_dst_row is not None else ())
    names = tuple(buffers[slot] for slot in order)
    c_source = emit_c(list(plan.commands), transport="rocket_rocc",
                             buffers=names)
    header = (
        "#ifndef MX_ISSUE_H\n#define MX_ISSUE_H\n\n"
        "/* Buffer order and sizes are in the companion buffer ABI metadata. */\n"
        "void mx_issue(" + ", ".join(f"const void *{name}" for name in names) +
        ");\n\n#endif\n")
    physical = {
        "schema": "mx_gemmini.vpu_elementwise_physical.v1",
        "profile_sha256": profile_sha256(profile),
        "bound_mlir_sha256": hashlib.sha256(mlir_text.encode("utf-8")).hexdigest(),
        "plan": {key: value for key, value in asdict(plan).items()
                 if key != "commands"},
        "commands": [asdict(command) for command in plan.commands],
    }
    minimum = {"src1": plan.rows * 16, "src2": plan.src2_rows * 16,
               "output": plan.output_rows * 16}
    if plan.second_dst_row is not None:
        minimum["output2"] = plan.rows // plan.reduction_length * 16
    abi = [{"name": buffers[slot], "slot": slot, "position": position,
            "role": "unused" if slot == "src2" and not plan.src2_rows else
                    "write" if slot.startswith("output") else "read",
            "minimum_bytes": minimum[slot], "alignment_bytes": 64,
            "layout": "row_major_bf16_lanes8"}
           for position, slot in enumerate(order)]
    return VPUEmission(plan, c_source, header, physical, abi)


def write_vpu_sources(directory: Path, emission: VPUEmission) -> tuple[Path, Path, Path]:
    """Write the established issuer/header/physical-plan artifacts."""
    issuer, header, physical = (directory / name for name in
                                ("mx_issue.c", "mx_issue.h", "physical_program.json"))
    issuer.write_text(emission.c_source, encoding="utf-8")
    header.write_text(emission.header, encoding="utf-8")
    physical.write_text(json.dumps(emission.physical, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return issuer, header, physical
