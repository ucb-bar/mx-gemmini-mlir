"""Export a payload-bound MX schedule as a callable, data-free RV64 issuer.

The standalone diagnostic supplies constant operand arrays and a golden.  A
whole-model caller instead supplies live packed operands, scales, and output
storage.  This module keeps the checked physical schedule but exposes those
addresses as a small C ABI.  Quantization and model tensor routing are separate
obligations; a caller must not mistake this encoded-operand ABI for dense f32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .command_ir import Command, emit_c
from .physical_program import PhysicalProgram
from .source_payload import manifest_sha256


_INPUTS = ("activation", "activation_scales", "weight", "weight_scales")
_OUTPUTS = ("output_bf16", "scratch_output_scales")


@dataclass(frozen=True)
class RuntimeIssuer:
    source: str
    shape_mnk: tuple[int, int, int]
    precision: str
    profile_sha256: str
    payload_manifest_sha256: str
    buffer_bytes: Mapping[str, int]


def emit_runtime_issuer(program: PhysicalProgram, manifest: dict, *,
                        symbol: str = "mx_issue_encoded") -> RuntimeIssuer:
    """Make a callable issuer for one checked FP8/FP4 BF16 matrix schedule.

    The issuer consumes encoded bytes by pointer.  It does not embed captured
    activation or weight bytes, and it has no golden or reference computation.
    The caller must provide the recorded buffer extents and maintain their
    lifetime until the final fence in the emitted schedule.
    """
    if manifest.get("schema") != "mx_gemmini.source_payload.v1":
        raise ValueError("runtime issuer needs a source payload manifest")
    if (manifest_sha256(manifest) != program.payload_manifest_sha256 or
            manifest.get("profile_sha256") != program.profile_sha256 or
            tuple(manifest.get("shape_mnk", ())) != program.shape):
        raise ValueError("runtime issuer program and payload identity differ")
    if program.mode != "spike_serial" or manifest.get("precision") not in {"FP8", "FP4"}:
        raise ValueError("runtime issuer needs a qualified serial FP8/FP4 schedule")
    if (program.output_format != "bf16" or program.derived_expected_bf16 is not None or
            not program.source_golden_preserving):
        raise ValueError("runtime issuer needs an unmodified BF16 matrix readout")
    if (program.plan.get("bf16_output_layout") not in (None, "row_major_bf16") or
            program.plan.get("bf16_output_layout") is None and
            len(program.plan.get("output_tiles", ())) > 1):
        raise ValueError("runtime issuer needs dense row-major BF16 output")
    resources = manifest.get("resources")
    if not isinstance(resources, dict):
        raise ValueError("runtime issuer needs resource extents")
    lengths: dict[str, int] = {}
    for name in _INPUTS:
        row = resources.get(name)
        if not isinstance(row, dict) or type(row.get("bytes")) is not int or row["bytes"] <= 0:
            raise ValueError(f"runtime issuer has no byte extent for {name}")
        lengths[name] = row["bytes"]
    m, n, _ = program.shape
    lengths["output_bf16"] = m * n * 2
    lengths["scratch_output_scales"] = 2048
    commands = [step.command for step in program.steps]
    referenced = {operand.buffer for command in commands if isinstance(command, Command)
                  for operand in (command.rs1, command.rs2) if operand.buffer is not None}
    if not set(_INPUTS).issubset(referenced) or "output_bf16" not in referenced:
        raise ValueError("runtime issuer schedule does not read every operand and write BF16")
    if referenced - set(_INPUTS) - set(_OUTPUTS):
        raise ValueError("runtime issuer schedule refers to a fixed or unsupported payload")
    buffers = _INPUTS + _OUTPUTS
    source = emit_c(commands, transport="rocket_rocc", buffers=buffers, symbol=symbol)
    return RuntimeIssuer(source, program.shape, manifest["precision"],
                         program.profile_sha256, program.payload_manifest_sha256,
                         lengths)
