"""Emit standalone Rocket sources from a lowered physical MX program."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .command_ir import Command, emit_c
from .physical_program import PhysicalProgram


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_standalone_sources(directory: Path, program: PhysicalProgram,
                             resources: dict[str, bytes]) -> dict:
    """Write a source-independent command issuer, data object, and receipt."""
    if program.mode != "spike_serial":
        raise ValueError("standalone execution is qualified only for the serial Spike mode")
    if any(name not in resources for name in ("activation", "weight", "activation_scales",
                                              "weight_scales", "golden_bf16")):
        raise ValueError("standalone MX program lacks operand or golden resources")
    commands = [step.command for step in program.steps]
    referenced = {operand.buffer for command in commands if isinstance(command, Command)
                  for operand in (command.rs1, command.rs2) if operand.buffer is not None}
    runtime = {"output_bf16", "scratch_output_scales"}
    if referenced - set(resources) - runtime:
        raise ValueError("standalone MX command has an unbound payload resource")
    names = tuple(sorted(referenced))
    issuer = emit_c(commands, transport="rocket_rocc", buffers=names)
    m, n, k = program.shape
    runtime_declarations = (f"static uint8_t output_bf16[{m * n * 2}] __attribute__((aligned(64)));\n"
                            "static uint8_t scratch_output_scales[2048] __attribute__((aligned(64)));\n")
    externs = "".join(f"extern const uint8_t {name}[];\n" for name in sorted(resources))
    arguments = ", ".join(names)
    driver = f'''#include <stdint.h>
#include <stdio.h>
{externs}
{runtime_declarations}
void mx_issue({", ".join(f"const void *{name}" for name in names)});

int main(void) {{
  mx_issue({arguments});
  const uint16_t *got = (const uint16_t *)output_bf16;
  const uint16_t *expected = (const uint16_t *)golden_bf16;
  int errors = 0;
  for (uint32_t i = 0; i < {m * n}; ++i) {{
    if (got[i] != expected[i]) {{
      if (errors < 8)
        printf("MISMATCH %u: got=0x%04x expected=0x%04x\\n",
               i, got[i], expected[i]);
      ++errors;
    }}
  }}
  printf("lowered MX {m}x{n}x{k}: %d BF16 mismatches\\n", errors);
  return errors != 0;
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
    physical_json = json.dumps(program.receipt(), indent=2, sort_keys=True) + "\n"
    (directory / "physical_program.json").write_text(physical_json)
    receipt = {"schema": "mx_gemmini.standalone_sources.v1",
               "mode": program.mode,
               "profile_sha256": program.profile_sha256,
               "payload_manifest_sha256": program.payload_manifest_sha256,
               "shape_mnk": list(program.shape),
               "command_count": sum(isinstance(item, Command) for item in commands),
               "fence_count": len(commands) - sum(isinstance(item, Command) for item in commands),
               "files_sha256": {path.name: _sha(path.read_bytes())
                                for path in sorted(directory.iterdir()) if path.is_file()}}
    (directory / "artifact_manifest.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
