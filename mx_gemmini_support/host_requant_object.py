"""Compose the qualified Radiance host epilogue with a runtime MX issuer.

The arithmetic comes from the same C generators as standalone Spike programs.
Only the storage binding changes: module globals become runtime pointers.
"""

from __future__ import annotations

from . import radiance_fp6_host, radiance_fp8_host


def emit_runtime_kernel(output_format: str, m: int, n: int) -> str:
    """Return the source-verified host quantizer with explicit buffer inputs."""
    if output_format == "radiance_header_fp8":
        return radiance_fp8_host.emit_kernel(m, n, runtime_pointers=True)
    if output_format == "radiance_header_fp6":
        return radiance_fp6_host.emit_kernel(m, n, runtime_pointers=True)
    raise ValueError("MX source object has no runtime host epilogue for output format")


def emit_composed_c(commands_c: str, output_format: str, names: tuple[str, ...],
                    m: int, n: int) -> str:
    """One public function runs the MX commands, then the host output op."""
    if output_format not in {"radiance_header_fp8", "radiance_header_fp6"}:
        raise ValueError("MX composed object needs a host output specialization")
    required = {"output_bf16", "output_quantized", "scratch_output_scales"}
    if output_format == "radiance_header_fp6":
        required.add("output_lut")
    if not required.issubset(names):
        raise ValueError("MX host object lacks required runtime buffers")
    signature = ", ".join(
        f"{'void' if name in required - {'output_lut'} else 'const void'} *{name}"
        for name in names)
    arguments = ", ".join(names)
    if output_format == "radiance_header_fp6":
        quant_args = ("output_bf16, (const uint8_t *)output_lut, "
                      "(uint8_t *)output_quantized, (uint8_t *)scratch_output_scales")
        clear = f'''  volatile uint8_t *quant = (volatile uint8_t *)output_quantized;
  for (uint32_t i = 0; i < {m * n // 2}; ++i) quant[i] = 0;
'''
    else:
        quant_args = ("output_bf16, (uint8_t *)output_quantized, "
                      "(uint8_t *)scratch_output_scales")
        clear = ""
    return (commands_c + "\n" + emit_runtime_kernel(output_format, m, n) +
            f"\nvoid mx_issue({signature}) {{\n"
            f"  mx_issue_commands({arguments});\n" + clear +
            f"  radiance_header_requantize({quant_args});\n}}\n")
