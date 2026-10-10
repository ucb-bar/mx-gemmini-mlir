"""Compose the qualified Radiance host epilogue with a runtime MX issuer.

The arithmetic comes from the same C generators as standalone Spike programs.
Only the storage binding changes: module globals become runtime pointers.
Each rewrite is anchored to one exact generator fragment and fails on drift.
"""

from __future__ import annotations

from . import radiance_fp6_host, radiance_fp8_host


def _replace_one(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"Radiance host generator drifted at {old!r}")
    return source.replace(old, new, 1)


def _kernel(driver: str, first: str) -> str:
    if driver.count(first) != 1 or driver.count("\nint main(void) {") != 1:
        raise ValueError("Radiance host driver has no unique arithmetic body")
    return first + driver.split(first, 1)[1].split("\nint main(void) {", 1)[0]


def emit_runtime_kernel(output_format: str, m: int, n: int) -> str:
    """Return the source-verified host quantizer with explicit buffer inputs."""
    if output_format == "radiance_header_fp8":
        driver = radiance_fp8_host.emit_driver("", "", "", (), m, n, 0)
        source = _kernel(driver, "static float bf16_value(uint16_t bits) {")
        return _replace_one(
            source, "static void radiance_header_requantize(void) {",
            "static void radiance_header_requantize("
            "const void *output_bf16, uint8_t *output_quantized, "
            "uint8_t *scratch_output_scales) {")
    if output_format == "radiance_header_fp6":
        driver = radiance_fp6_host.emit_driver("", "", "", (), m, n, 0)
        source = _kernel(driver, "static uint32_t float_bits(float value) {")
        for old, new in (
            ("static uint8_t lut_code(uint32_t pair, uint32_t index) {",
             "static uint8_t lut_code(uint32_t pair, uint32_t index, "
             "const uint8_t *output_lut) {"),
            ("static uint8_t nearest_lut_index(uint32_t pair, uint8_t code) {",
             "static uint8_t nearest_lut_index(uint32_t pair, uint8_t code, "
             "const uint8_t *output_lut) {"),
            ("lut_code(pair, index)", "lut_code(pair, index, output_lut)"),
            ("nearest_lut_index(row >> 1, code)",
             "nearest_lut_index(row >> 1, code, output_lut)"),
            ("static void radiance_header_requantize(void) {",
             "static void radiance_header_requantize("
             "const void *output_bf16, const uint8_t *output_lut, "
             "uint8_t *output_quantized, uint8_t *scratch_output_scales) {"),
        ):
            source = _replace_one(source, old, new)
        return source
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
