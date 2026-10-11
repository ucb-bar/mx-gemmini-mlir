"""Compact C loops emit exactly the source-qualified MX command sequence."""

from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess

import pytest

from mx_gemmini_support.command_ir import Command, Fence
from mx_gemmini_support.compact_runtime_issuer import emit_compact_runtime_issuer
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/model2mlir_small_llama_mlp_fp4_padding_e9ded36_266c593"
PROFILE = load_profile(
    ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json")
FP8_ARCHIVE = ROOT / "docs/evidence/model2mlir_tinyllama_projection_a042643_266c593"
FP8_PROFILE = load_profile(
    ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json")
BUFFERS = ("activation", "activation_scales", "weight", "weight_scales",
           "output_bf16", "scratch_output_scales")
ADDRESSES = {name: 0x10000000 + index * 0x01000000
             for index, name in enumerate(BUFFERS)}


def _operand_value(operand) -> int:
    if operand.immediate is not None:
        return operand.immediate
    value = ADDRESSES[operand.buffer] + operand.byte_offset
    if operand.address_mask is not None:
        value = ((value & operand.address_mask) << operand.address_shift) | operand.or_bits
    return value


@pytest.mark.parametrize("archive,profile,precision", [
    (ARCHIVE / "g", PROFILE, "FP4"),
    (ARCHIVE / "dn", PROFILE, "FP4"),
    (FP8_ARCHIVE, FP8_PROFILE, "FP8"),
])
def test_compact_trace_matches_every_qualified_command(
        tmp_path, archive: Path, profile: dict, precision: str) -> None:
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler unavailable")
    manifest, resources = load_bundle(archive / "bundle")
    program = lower_bound_source((archive / "projection.payload_bound.mlir").read_text(),
                                 profile, manifest, resources)
    compact = emit_compact_runtime_issuer(
        profile, tuple(manifest["shape_mnk"]), tuple(manifest["tile_mnk"]),
        precision=precision)
    assert compact.plan == program.plan
    assert len(compact.source) < 7000
    (tmp_path / "compact.c").write_text(compact.source)
    (tmp_path / "trace.c").write_text('''#include <stdint.h>
uint64_t trace[10000][3];
uint64_t trace_count = 0;
void mx_trace_command(uint32_t funct, uint64_t rs1, uint64_t rs2) {
  if (trace_count >= 10000) __builtin_trap();
  trace[trace_count][0] = funct;
  trace[trace_count][1] = rs1;
  trace[trace_count++][2] = rs2;
}
void mx_trace_fence(void) {
  if (trace_count >= 10000) __builtin_trap();
  trace[trace_count][0] = UINT64_MAX;
  trace[trace_count][1] = 0;
  trace[trace_count++][2] = 0;
}
''')
    library = tmp_path / "trace.so"
    subprocess.run([cc, "-shared", "-fPIC", "-O2", "-std=c11", "-Wall", "-Wextra",
                    "-Werror", "-DMX_TRACE_COMMANDS", str(tmp_path / "compact.c"),
                    str(tmp_path / "trace.c"), "-o", str(library)], check=True)
    loaded = ctypes.CDLL(str(library))
    loaded.mx_issue_encoded.argtypes = [ctypes.c_void_p] * len(BUFFERS)
    loaded.mx_issue_encoded(*[ctypes.c_void_p(ADDRESSES[name]) for name in BUFFERS])
    count = ctypes.c_uint64.in_dll(loaded, "trace_count").value
    assert count == len(program.steps)
    recorded = ((ctypes.c_uint64 * 3) * 10000).in_dll(loaded, "trace")
    expected = [
        (step.command.funct, _operand_value(step.command.rs1),
         _operand_value(step.command.rs2)) if isinstance(step.command, Command)
        else (2**64 - 1, 0, 0)
        for step in program.steps
    ]
    assert [tuple(recorded[index]) for index in range(count)] == expected


def test_large_shape_has_compact_source_without_unrolling() -> None:
    compact = emit_compact_runtime_issuer(
        PROFILE, (128, 151936, 1536), (32, 32, 128), precision="FP4")
    assert len(compact.source) < 7000
    assert len(compact.plan["output_tiles"]) == 18992
    assert compact.buffer_bytes["output_bf16"] == 128 * 151936 * 2
