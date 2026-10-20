"""Compose independent packed-input projections under one completion boundary."""

from dataclasses import dataclass

from .scheduled_native_issuer import emit_scheduled_native_issuer


@dataclass(frozen=True)
class ProjectionGroupKernel:
    source: str
    workspace_bytes: int
    output_shapes: tuple[tuple[int, int], ...]
    schedule_digests: tuple[str, ...]


def emit_projection_group(profile, projections, *, symbol, workspace_symbol):
    """Retain original tiles/weights and complete every readout before returning.

    Intermediate BF16 destinations occupy disjoint regions of the existing
    serial workspace. Even native-loop counts preserve slot parity across
    projections. The caller proves a common immutable packed input, independent
    immutable weights, fresh disjoint outputs and a closed source call group.
    """
    if (
        not 2 <= len(projections) <= 8
        or not symbol.isidentifier()
        or not workspace_symbol.isidentifier()
    ):
        raise ValueError(
            "projection grouping needs 2..8 checked projections and C symbols"
        )
    shapes = [tuple(p["shape"]) for p in projections]
    tiles = [tuple(p["tile_mnk"]) for p in projections]
    if any(
        len(s) != 3 or any(type(x) is not int or x <= 0 for x in s)
        for s in (*shapes, *tiles)
    ):
        raise ValueError("projection grouping needs positive static matrix extents")
    m, _, k = shapes[0]
    if (
        any((s[0], s[2]) != (m, k) for s in shapes)
        or len(set(tiles)) != 1
        or any(any(type(x) is not int or x <= 0 for x in shape) for shape in shapes)
    ):
        raise ValueError(
            "grouped projections require one input shape and tile partition"
        )
    sources, calls, declarations, readouts, digests = [], [], [], [], []
    offset = 0
    for index, (projection, shape, tile) in enumerate(
        zip(projections, shapes, tiles, strict=True)
    ):
        rows, n, reduction = shape
        tm, tn, tk = tile
        loops = (rows // tm) * (n // tn) * (reduction // tk)
        if loops % 2:
            raise ValueError(
                "projection grouping requires even native-loop slot parity"
            )
        fixed = projection["weight_symbols"]
        if len(fixed) != 2 or any(not name.isidentifier() for name in fixed):
            raise ValueError("grouped projection has invalid immutable weight symbols")
        issuer_name = f"{symbol}_issue_{index}"
        schedule = emit_scheduled_native_issuer(
            profile,
            shape,
            precision="FP8",
            symbol=issuer_name,
            tile_mnk=tile,
            defer_completion=True,
        )
        sources.append(schedule.issuer.source)
        digests.append(schedule.digest)
        declarations += [f"extern const uint8_t {name}[];" for name in fixed]
        calls.append(
            f"  {issuer_name}(packed,packed+{m * k}u,{fixed[0]},{fixed[1]},"
            f"{workspace_symbol}+{offset}u,{workspace_symbol}+SCRATCH_OFFSET);"
        )
        readouts.append(f"""  for (uint32_t i=0; i<{m * n}u; ++i) {{
    union {{ uint32_t u; float f; }} word;
    word.u=(uint32_t)((const uint16_t *)({workspace_symbol}+{offset}u))[i]<<16;
    out{index}[i]=word.f;
  }}""")
        offset += m * n * 2
    scratch = (offset + 63) // 64 * 64
    if scratch + 2048 > 0x7FFFFFFF:
        raise ValueError("grouped readouts exceed the runtime byte extent")
    params = ", ".join(
        ["const uint8_t *packed", *(f"float *out{i}" for i in range(len(shapes)))]
    )
    guard = " || ".join(["!packed", *(f"!out{i}" for i in range(len(shapes)))])
    source = (
        "\n".join(sources)
        + "\n"
        + "\n".join(declarations)
        + f"""
extern unsigned char {workspace_symbol}[];
#define SCRATCH_OFFSET {scratch}u
void {symbol}({params}) {{
  if ({guard}) __builtin_trap();
{chr(10).join(calls)}
  MX_FENCE();
{chr(10).join(readouts)}
}}
"""
    )
    return ProjectionGroupKernel(
        source, scratch + 2048, tuple((s[0], s[1]) for s in shapes), tuple(digests)
    )
