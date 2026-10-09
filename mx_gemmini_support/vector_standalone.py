"""Emit a standalone Rocket program for a captured VPU→requant seam."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .command_ir import Command, Fence, Operand, emit_c
from .physical_program import _cmd, _config_ld, _config_st, _transfer
from .vector_lowering import lower_vector_commands
from .resident_lowering import lower_resident_chain_commands
from .first_matrix_lowering import lower_first_matrix_commands


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_vector_requant_sources(directory: Path, mlir_text: str, profile: dict,
                                 resources: dict[str, bytes], facts: dict) -> dict:
    """Lower two typed ops and add only generic transfer and check plumbing."""
    if (facts.get("shape_mn") != [64, 64] or
            facts.get("sp_bf16") != 0x1000 or facts.get("sp_c1") != 128 or
            len(resources.get("c1_bf16", b"")) != 8192 or
            len(resources.get("c1_codes_ref", b"")) != 4096 or
            len(resources.get("c1_scales_ref", b"")) != 128):
        raise ValueError("VPU/requant standalone requires the captured 64x64 source resources")
    vector = lower_vector_commands(mlir_text, profile)
    if (len(vector) != 2 or vector[0].funct != 33 or vector[1].funct != 34 or
            vector[1].rs1.buffer != "c1_scales"):
        raise ValueError("VPU/requant standalone requires typed VPU then buffered requant")
    input_rows, output_rows = 512, 256
    commands: list[Command | Fence] = [
        _cmd(7, 0, 0), _config_ld(16), Fence()]
    for row in range(0, input_rows, 16):
        commands.append(_transfer(2, "c1_bf16", row * 16, facts["sp_bf16"] + row))
    commands.extend((Fence(), vector[0], Fence(), vector[1], Fence(),
                     _config_st(16)))
    for row in range(0, output_rows, 16):
        commands.append(_transfer(3, "output_tiled", row * 16,
                                  facts["sp_c1"] + row))
    commands.append(Fence())
    buffers = ("c1_bf16", "c1_scales", "output_tiled")
    issuer = emit_c(commands, transport="rocket_rocc", buffers=buffers)
    driver = '''#include <stdint.h>
#include <stdio.h>
extern const uint8_t c1_bf16[];
extern const uint8_t c1_codes_ref[];
extern const uint8_t c1_scales_ref[];
static uint8_t c1_scales[128] __attribute__((aligned(64)));
static uint8_t output_tiled[4096] __attribute__((aligned(64)));
void mx_issue(const void *c1_bf16, const void *c1_scales, const void *output_tiled);

int main(void) {
  mx_issue(c1_bf16, c1_scales, output_tiled);
  int code_errors = 0, scale_errors = 0;
  for (uint32_t row = 0; row < 64; ++row)
    for (uint32_t col = 0; col < 64; ++col) {
      uint32_t tiled = (((row / 16) * 4 + col / 16) * 16 + row % 16) * 16 + col % 16;
      uint32_t linear = row * 64 + col;
      if (output_tiled[tiled] != c1_codes_ref[linear]) {
        if (code_errors < 8)
          printf("CODE MISMATCH (%u,%u): got=0x%02x expected=0x%02x\\n",
                 row, col, output_tiled[tiled], c1_codes_ref[linear]);
        ++code_errors;
      }
    }
  for (uint32_t i = 0; i < 128; ++i)
    if (c1_scales[i] != c1_scales_ref[i]) {
      if (scale_errors < 8)
        printf("SCALE MISMATCH %u: got=0x%02x expected=0x%02x\\n",
               i, c1_scales[i], c1_scales_ref[i]);
      ++scale_errors;
    }
  printf("lowered VPU requant 64x64: %d FP8 code mismatches, %d E8M0 scale mismatches\\n",
         code_errors, scale_errors);
  return code_errors != 0 || scale_errors != 0;
}
'''
    assembly = [".section .rodata", ".balign 64"]
    for name in sorted(resources):
        assembly.extend((f".globl {name}", f"{name}:",
                         f'.incbin "{name}.bin"', ".balign 64"))
    assembly.append('.section .note.GNU-stack,"",@progbits')
    directory.mkdir(parents=True, exist_ok=False)
    for name, data in resources.items():
        (directory / f"{name}.bin").write_bytes(data)
    (directory / "mx_issue.c").write_text(issuer)
    (directory / "mx_driver.c").write_text(driver)
    (directory / "mx_data.S").write_text("\n".join(assembly) + "\n")
    physical = {"schema": "mx_gemmini.vector_requant_program.v1",
                "source_facts": facts, "typed_mlir_sha256": _sha(mlir_text.encode()),
                "command_count": sum(isinstance(command, Command) for command in commands),
                "fence_count": sum(isinstance(command, Fence) for command in commands),
                "ordered_functs": [command.funct for command in commands
                                   if isinstance(command, Command)]}
    (directory / "physical_program.json").write_text(
        json.dumps(physical, indent=2, sort_keys=True) + "\n")
    return {"schema": "mx_gemmini.vector_requant_sources.v1",
            "source_facts": facts, "typed_mlir_sha256": _sha(mlir_text.encode()),
            "command_count": physical["command_count"],
            "files_sha256": {path.name: _sha(path.read_bytes())
                             for path in sorted(directory.iterdir()) if path.is_file()}}


def write_resident_chain_sources(directory: Path, mlir_text: str, profile: dict,
                                 resources: dict[str, bytes], facts: dict, *,
                                 frontend_mlir: str | None = None) -> dict:
    """Emit the typed resident chain with C1 and C2 source-golden checks."""
    expected_lengths = {"c1_bf16": 8192, "c1_codes_ref": 4096,
                        "c1_scales_ref": 128, "b2_weight": 4096,
                        "b2_scales": 128, "c2_codes_ref": 4096,
                        "c2_scales_ref": 128}
    if any(len(resources.get(name, b"")) != length
           for name, length in expected_lengths.items()):
        raise ValueError("resident chain source resources differ from selected 64x64 layout")
    if facts.get("sp_bf16") != 0x1000 or facts.get("sp_c1") != 128:
        raise ValueError("resident chain source layout differs")
    body = lower_resident_chain_commands(
        mlir_text, profile,
        expected_sites=("functional:matmul", "functional:matmul_1")
        if frontend_mlir is not None else None)
    if frontend_mlir is None:
        commands: list[Command | Fence] = [_cmd(7, 0, 0), _config_ld(16), Fence()]
        for row in range(0, 512, 16):
            commands.append(_transfer(2, "c1_bf16", row * 16, 0x1000 + row))
    else:
        commands = list(lower_first_matrix_commands(frontend_mlir, profile, resources))
    commands.extend((Fence(), *body, _config_st(16)))
    for name, start in (("c1_tiled", 128), ("c2_tiled", 512)):
        for row in range(0, 256, 16):
            commands.append(_transfer(3, name, row * 16, start + row))
    commands.append(Fence())
    referenced = {operand.buffer for command in commands if isinstance(command, Command)
                  for operand in (command.rs1, command.rs2) if operand.buffer is not None}
    runtime = {"c1_scales", "c2_scales", "c1_tiled", "c2_tiled",
               "c1_bf16_observed"}
    if referenced - set(resources) - runtime:
        raise ValueError("resident chain command references an unbound source buffer")
    names = tuple(sorted(referenced))
    issuer = emit_c(commands, transport="rocket_rocc", buffers=names)
    driver = f'''#include <stdint.h>
#include <stdio.h>
extern const uint8_t c1_bf16[];
extern const uint8_t c1_codes_ref[];
extern const uint8_t c1_scales_ref[];
extern const uint8_t b2_weight[];
extern const uint8_t b2_scales[];
extern const uint8_t c2_codes_ref[];
extern const uint8_t c2_scales_ref[];
static uint8_t c1_scales[128] __attribute__((aligned(64)));
static uint8_t c2_scales[128] __attribute__((aligned(64)));
static uint8_t c1_tiled[4096] __attribute__((aligned(64)));
static uint8_t c2_tiled[4096] __attribute__((aligned(64)));
void mx_issue({", ".join(f"const void *{name}" for name in names)});

int main(void) {{
  mx_issue({", ".join(names)});
  int c1_codes = 0, c1_scale_errors = 0, c2_codes = 0, c2_scale_errors = 0;
  for (uint32_t row = 0; row < 64; ++row)
    for (uint32_t col = 0; col < 64; ++col) {{
      uint32_t tiled = (((row / 16) * 4 + col / 16) * 16 + row % 16) * 16 + col % 16;
      uint32_t linear = row * 64 + col;
      c1_codes += c1_tiled[tiled] != c1_codes_ref[linear];
      c2_codes += c2_tiled[tiled] != c2_codes_ref[linear];
    }}
  for (uint32_t i = 0; i < 128; ++i) {{
    c1_scale_errors += c1_scales[i] != c1_scales_ref[i];
    c2_scale_errors += c2_scales[i] != c2_scales_ref[i];
  }}
  printf("lowered resident chain: C1 %d codes %d scales, C2 %d codes %d scales mismatches\\n",
         c1_codes, c1_scale_errors, c2_codes, c2_scale_errors);
  return c1_codes || c1_scale_errors || c2_codes || c2_scale_errors;
}}
'''
    if frontend_mlir is not None:
        driver = f'''#include <stdint.h>
#include <stdio.h>
extern const uint8_t a1_activation[];
extern const uint8_t a1_scales[];
extern const uint8_t b1_weight[];
extern const uint8_t b1_scales[];
extern const uint8_t b2_weight[];
extern const uint8_t b2_scales[];
extern const uint8_t c1_bf16[];
extern const uint8_t c1_codes_ref[];
extern const uint8_t c1_scales_ref[];
extern const uint8_t c2_codes_ref[];
extern const uint8_t c2_scales_ref[];
static uint8_t c1_scales[128] __attribute__((aligned(64)));
static uint8_t c2_scales[128] __attribute__((aligned(64)));
static uint8_t c1_bf16_observed[8192] __attribute__((aligned(64)));
static uint8_t c1_tiled[4096] __attribute__((aligned(64)));
static uint8_t c2_tiled[4096] __attribute__((aligned(64)));
void mx_issue({", ".join(f"const void *{name}" for name in names)});

int main(void) {{
  mx_issue({", ".join(names)});
  int bf16_errors = 0;
  for (uint32_t i = 0; i < 8192; ++i)
    bf16_errors += c1_bf16_observed[i] != c1_bf16[i];
  int c1_codes = 0, c1_scale_errors = 0, c2_codes = 0, c2_scale_errors = 0;
  for (uint32_t row = 0; row < 64; ++row)
    for (uint32_t col = 0; col < 64; ++col) {{
      uint32_t tiled = (((row / 16) * 4 + col / 16) * 16 + row % 16) * 16 + col % 16;
      uint32_t linear = row * 64 + col;
      c1_codes += c1_tiled[tiled] != c1_codes_ref[linear];
      c2_codes += c2_tiled[tiled] != c2_codes_ref[linear];
    }}
  for (uint32_t i = 0; i < 128; ++i) {{
    c1_scale_errors += c1_scales[i] != c1_scales_ref[i];
    c2_scale_errors += c2_scales[i] != c2_scales_ref[i];
  }}
  printf("lowered full chain: C1 BF16 %d, C1 %d codes %d scales, C2 %d codes %d scales mismatches\\n",
         bf16_errors, c1_codes, c1_scale_errors, c2_codes, c2_scale_errors);
  return bf16_errors || c1_codes || c1_scale_errors || c2_codes || c2_scale_errors;
}}
'''
    assembly = [".section .rodata", ".balign 64"]
    for name in sorted(resources):
        assembly.extend((f".globl {name}", f"{name}:",
                         f'.incbin "{name}.bin"', ".balign 64"))
    assembly.append('.section .note.GNU-stack,"",@progbits')
    directory.mkdir(parents=True, exist_ok=False)
    for name, data in resources.items():
        (directory / f"{name}.bin").write_bytes(data)
    (directory / "mx_issue.c").write_text(issuer)
    (directory / "mx_driver.c").write_text(driver)
    (directory / "mx_data.S").write_text("\n".join(assembly) + "\n")
    physical = {"schema": ("mx_gemmini.full_chain_program.v1" if frontend_mlir is not None
                           else "mx_gemmini.resident_chain_program.v1"),
                "source_facts": facts, "typed_mlir_sha256": _sha(mlir_text.encode()),
                "command_count": sum(isinstance(command, Command) for command in commands),
                "fence_count": sum(isinstance(command, Fence) for command in commands),
                "ordered_functs": [command.funct for command in commands
                                   if isinstance(command, Command)]}
    if frontend_mlir is not None:
        physical["frontend_mlir_sha256"] = _sha(frontend_mlir.encode())
    (directory / "physical_program.json").write_text(
        json.dumps(physical, indent=2, sort_keys=True) + "\n")
    return {"schema": ("mx_gemmini.full_chain_sources.v1" if frontend_mlir is not None
                       else "mx_gemmini.resident_chain_sources.v1"),
            "source_facts": facts, "typed_mlir_sha256": _sha(mlir_text.encode()),
            "command_count": physical["command_count"],
            "files_sha256": {path.name: _sha(path.read_bytes())
                             for path in sorted(directory.iterdir()) if path.is_file()}}
