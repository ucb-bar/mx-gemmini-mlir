"""Emit standalone Rocket sources from a lowered physical MX program."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .command_ir import Command, emit_c
from .physical_program import PhysicalProgram
from . import radiance_fp6_host, radiance_fp8_host


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_standalone_sources(directory: Path, program: PhysicalProgram,
                             resources: dict[str, bytes]) -> dict:
    """Write a source-independent command issuer, data object, and receipt."""
    if program.mode not in {"spike_serial", "rtl_alternating"}:
        raise ValueError("unknown standalone MX physical mode")
    if program.output_format not in {"bf16", "fp8_e4m3", "fp8_e5m2", "fp4_e2m1",
                                     "fp6_e2m3", "fp6_e3m2",
                                     "radiance_header_fp8", "radiance_header_fp6"}:
        raise ValueError("standalone MX output format is not qualified")
    if (not program.source_golden_preserving and
            program.derived_expected_bf16 is None and
            program.output_format not in {"fp8_e4m3", "fp8_e5m2", "fp4_e2m1",
                                          "fp6_e2m3", "fp6_e3m2"} and
            program.golden_origin != "target_mesh_reference"):
        raise ValueError("source BF16 golden does not cover these MX VPU/requant operations")
    if any(name not in resources for name in ("activation", "weight", "activation_scales",
                                              "weight_scales", "golden_bf16")):
        raise ValueError("standalone MX program lacks operand or golden resources")
    commands = [step.command for step in program.steps]
    referenced = {operand.buffer for command in commands if isinstance(command, Command)
                  for operand in (command.rs1, command.rs2) if operand.buffer is not None}
    runtime = {"output_bf16", "output_quantized", "scratch_output_scales"}
    if referenced - set(resources) - runtime:
        raise ValueError("standalone MX command has an unbound payload resource")
    names = tuple(sorted(referenced))
    issuer = emit_c(commands, transport="rocket_rocc", buffers=names)
    m, n, k = program.shape
    quantized = program.output_format in {"fp8_e4m3", "fp8_e5m2", "fp4_e2m1",
                                          "fp6_e2m3", "fp6_e3m2"}
    packed_lut = program.plan.get("quant_output_layout") == "packed_even_odd_m_lut_indices"
    host_fp8 = program.output_format == "radiance_header_fp8"
    host_fp6 = program.output_format == "radiance_header_fp6"
    host_header = host_fp8 or host_fp6
    packed_fp4 = program.output_format == "fp4_e2m1"
    packed_fp6 = program.output_format == "fp6_e3m2"
    quant_name = ("golden_lut_indices" if packed_lut else "nicolas_fp6" if packed_fp6
                  else "nicolas_fp4" if packed_fp4 else "nicolas_fp8")
    source_quant_name = ("source_fp6_packed" if packed_fp6 else
                         "source_fp4_packed" if packed_fp4 and
                         "source_fp4_packed" in resources else "golden_fp8")
    quant_bytes = m * n // (2 if packed_lut or packed_fp4 or packed_fp6 else 1)
    if quantized:
        if packed_lut:
            if (program.output_format not in {"fp8_e4m3", "fp8_e5m2",
                                                  "fp6_e2m3", "fp6_e3m2"} or
                    len(resources.get("golden_lut_indices", b"")) != quant_bytes or
                    len(resources.get("golden_output_scales", b"")) != m * n // 32 or
                    "output_lut" not in resources):
                raise ValueError("packed LUT output lacks source codes, scales, or LUT")
        elif (quant_name not in resources or "nicolas_output_scales" not in resources or
              source_quant_name not in resources or
              "golden_output_scales" not in resources or
              m * n // 32 > 2048):
            raise ValueError("quantized output needs code and scale goldens within runtime capacity")
        if m * n // 32 > 2048:
            raise ValueError("quantized scale output exceeds runtime capacity")
        runtime_declarations = (
            f"static uint8_t output_quantized[{quant_bytes}] __attribute__((aligned(64)));\n"
            "static uint8_t scratch_output_scales[2048] __attribute__((aligned(64)));\n")
    elif host_header:
        code_name = "source_fp6_packed" if host_fp6 else "golden_fp8"
        code_size = m * n // 2 if host_fp6 else m * n
        if ((not program.source_golden_preserving and
              program.golden_origin != "target_mesh_reference") or
                len(resources.get(code_name, b"")) != code_size or
                len(resources.get("golden_output_scales", b"")) != m * n // 32 or
                (host_fp6 and len(resources.get("output_lut", b"")) != 64 * 12)):
            raise ValueError("Radiance header epilogue needs checked BF16, codes, scales, and FP6 LUT")
        runtime_declarations = (
            f"static uint8_t output_bf16[{m * n * 2}] __attribute__((aligned(64)));\n"
            f"static uint8_t output_quantized[{code_size}] __attribute__((aligned(64)));\n"
            f"static uint8_t scratch_output_scales[{m * n // 32}] __attribute__((aligned(64)));\n")
    else:
        runtime_declarations = (f"static uint8_t output_bf16[{m * n * 2}] __attribute__((aligned(64)));\n"
                                "static uint8_t scratch_output_scales[2048] __attribute__((aligned(64)));\n")
    resource_files = dict(resources)
    golden_name = "golden_bf16"
    if program.derived_expected_bf16 is not None:
        resource_files["derived_expected_bf16"] = program.derived_expected_bf16
        golden_name = "derived_expected_bf16"
    externs = "".join(f"extern const uint8_t {name}[];\n" for name in sorted(resource_files))
    arguments = ", ".join(names)
    if ("output_tiles" in program.plan and
            program.plan.get("bf16_output_layout") != "row_major_bf16"):
        tm, tn, _ = program.plan["tile"]
        comparison = f'''  for (uint32_t row = 0; row < {m}; ++row)
    for (uint32_t col = 0; col < {n}; ++col) {{
      uint32_t tile = (row / {tm}) * {n // tn} + col / {tn};
      uint32_t local = (row % {tm}) * {tn} + col % {tn};
      uint32_t got_index = tile * {tm * tn} + local;
      uint32_t expected_index = row * {n} + col;
      if (got[got_index] != expected[expected_index]) {{
        if (errors < 8)
          printf("MISMATCH (%u,%u): got=0x%04x expected=0x%04x\\n",
                 row, col, got[got_index], expected[expected_index]);
        ++errors;
      }}
    }}
'''
    else:
        comparison = f'''  for (uint32_t i = 0; i < {m * n}; ++i) {{
    if (got[i] != expected[i]) {{
      if (errors < 8)
        printf("MISMATCH %u: got=0x%04x expected=0x%04x\\n",
               i, got[i], expected[i]);
      ++errors;
    }}
  }}
'''
    driver = f'''#include <stdint.h>
#include <stdio.h>
{externs}
{runtime_declarations}
void mx_issue({", ".join(f"const void *{name}" for name in names)});

int main(void) {{
  mx_issue({arguments});
  const uint16_t *got = (const uint16_t *)output_bf16;
  const uint16_t *expected = (const uint16_t *){golden_name};
  int errors = 0;
{comparison}  printf("lowered MX {m}x{n}x{k}: %d BF16 mismatches\\n", errors);
  return errors != 0;
}}
'''
    if quantized:
        quant_label = (f"{program.output_format.split('_', 1)[1].upper()} packed-LUT-index"
                       if packed_lut else
                       "FP6 packed-index" if packed_fp6 else
                       "FP4 packed-code" if packed_fp4 else "FP8 code")
        code_loop = f'''  for (uint32_t i = 0; i < {quant_bytes}; ++i) {{
    if (output_quantized[i] != {quant_name}[i]) {{
      if (code_errors < 8)
        printf("CODE MISMATCH %u: got=0x%02x expected=0x%02x\\n",
               i, output_quantized[i], {quant_name}[i]);
      ++code_errors;
    }}
  }}
'''
        if program.tiled_quant_readout:
            if packed_fp4 or packed_fp6 or m % 16 or n % 16:
                raise ValueError("tiled quant readout requires complete FP8 DIM16 tiles")
            code_loop = f'''  for (uint32_t row = 0; row < {m}; ++row)
    for (uint32_t col = 0; col < {n}; ++col) {{
      uint32_t tiled = (((row / 16) * ({n} / 16) + col / 16) * 16 + row % 16) * 16 + col % 16;
      uint32_t linear = row * {n} + col;
      if (output_quantized[tiled] != {quant_name}[linear]) {{
        if (code_errors < 8)
          printf("CODE MISMATCH (%u,%u): got=0x%02x expected=0x%02x\\n",
                 row, col, output_quantized[tiled], {quant_name}[linear]);
        ++code_errors;
      }}
    }}
'''
        scale_loop = (f'''  for (uint32_t row = 0; row < {m}; ++row)
    for (uint32_t group = 0; group < {n // 32}; ++group) {{
      uint32_t got_index = row * {n // 32} + group;
      uint32_t expected_index = group * {m} + row;
      if (scratch_output_scales[got_index] != golden_output_scales[expected_index]) {{
        if (scale_errors < 8)
          printf("SCALE MISMATCH (%u,%u): got=0x%02x expected=0x%02x\\n",
                 row, group, scratch_output_scales[got_index],
                 golden_output_scales[expected_index]);
        ++scale_errors;
      }}
    }}
''' if packed_lut else f'''  for (uint32_t i = 0; i < {m * n // 32}; ++i) {{
    if (scratch_output_scales[i] != nicolas_output_scales[i]) {{
      if (scale_errors < 8)
        printf("SCALE MISMATCH %u: got=0x%02x expected=0x%02x\\n",
               i, scratch_output_scales[i], nicolas_output_scales[i]);
      ++scale_errors;
    }}
  }}
''')
        driver = f'''#include <stdint.h>
#include <stdio.h>
{externs}
{runtime_declarations}
void mx_issue({", ".join(f"const void *{name}" for name in names)});

int main(void) {{
  mx_issue({arguments});
  int code_errors = 0;
  int scale_errors = 0;
{code_loop}{scale_loop}
  printf("lowered MX {m}x{n}x{k}: %d {quant_label} mismatches, %d E8M0 scale mismatches\\n",
         code_errors, scale_errors);
  return code_errors != 0 || scale_errors != 0;
}}
'''
    elif host_fp8:
        driver = radiance_fp8_host.emit_driver(
            externs, runtime_declarations, arguments, names, m, n, k)
    elif host_fp6:
        driver = radiance_fp6_host.emit_driver(
            externs, runtime_declarations, arguments, names, m, n, k)
    assembly = [".section .rodata", ".balign 64"]
    for name in sorted(resource_files):
        assembly.extend((f".globl {name}", f"{name}:",
                         f'.incbin "{name}.bin"', ".balign 64"))
    assembly.append('.section .note.GNU-stack,"",@progbits')
    directory.mkdir(parents=True, exist_ok=False)
    for name, data in resource_files.items():
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
    if program.derived_expected_bf16 is not None:
        receipt["golden_basis"] = (
            "derived_bf16_store_relu" if program.golden_derivation ==
            "bf16_store_relu_sign_clear" else
            "derived_bf16_scalar_chain" if program.derived_vpu_scalar_chain is not None
            else "derived_bf16_x2" if program.derived_vpu_scalar_bf16 is None
            else "derived_bf16_adds" if program.plan.get("vector_tile_policy") ==
            "bf16_adds_scalar_each_output_tile_v1" else "derived_bf16_muls")
    if quantized:
        receipt["golden_basis"] = (f"nicolas_source_packed_{program.output_format}_lut_indices_and_e8m0"
                                   if packed_lut else
            "nicolas_vpu_x2_spad_requant_from_source_bf16" if program.tiled_quant_readout else
            "nicolas_fp6_e3m2_lut_from_source_bf16" if packed_fp6 else
            "nicolas_fp4_e3m1_e2m1_from_source_bf16" if packed_fp4 else
            "nicolas_mxquant_po2_rne_from_source_bf16")
        if packed_lut:
            receipt["source_quant_code_format"] = f"packed_{program.output_format}_lut_index"
            receipt["target_quant_code_format"] = f"packed_{program.output_format}_lut_index"
        elif packed_fp6:
            receipt["source_quant_code_format"] = "packed_fp6_lut_index"
            receipt["target_quant_code_format"] = "packed_fp6_lut_index"
            receipt["source_quant_code_differences"] = sum(
                a != b for a, b in zip(resources["source_fp6_packed"], resources["nicolas_fp6"]))
        elif packed_fp4:
            receipt["source_quant_code_format"] = (
                "packed_fp4_e2m1" if source_quant_name == "source_fp4_packed" else "fp8_e4m3")
            receipt["target_quant_code_format"] = "packed_fp4_e2m1"
            if source_quant_name == "source_fp4_packed":
                receipt["source_quant_code_differences"] = sum(
                    a != b for a, b in zip(resources[source_quant_name],
                                           resources["nicolas_fp4"]))
        else:
            receipt["source_quant_code_differences"] = sum(
                a != b for a, b in zip(resources["golden_fp8"], resources["nicolas_fp8"]))
        if not packed_lut:
            receipt["source_quant_scale_differences"] = sum(
                a != b for a, b in zip(resources["golden_output_scales"],
                                       resources["nicolas_output_scales"]))
    if host_header:
        receipt["golden_basis"] = ("radiance_header_fp6_from_mx_bf16" if host_fp6 else
                                   "radiance_header_fp8_from_mx_bf16")
        receipt["output_policy"] = ("radiance_header_fp6_v1" if host_fp6 else
                                    "radiance_header_fp8_v1")
    (directory / "artifact_manifest.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
