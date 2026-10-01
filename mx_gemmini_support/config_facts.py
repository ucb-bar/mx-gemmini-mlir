"""Read the selected standalone MX configuration from pinned Scala source."""

from __future__ import annotations

import re


def _arguments(source: str, marker: str) -> str:
    if source.count(marker) != 1:
        raise ValueError(f"expected one configuration declaration {marker!r}")
    start = source.index(marker) + len(marker)
    depth = 1
    quoted = False
    escaped = False
    comment = False
    for index in range(start, len(source)):
        char = source[index]
        if comment:
            if char == "\n":
                comment = False
            continue
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if source[index:index + 2] == "//":
            comment = True
            continue
        if char == '"':
            quoted = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return source[start:index]
    raise ValueError(f"unclosed configuration declaration {marker!r}")


def _assignments(body: str) -> dict[str, str]:
    # Split only top-level arguments. Parenthesized capacities and LUT settings
    # may contain commas, and a comment must not change the selected value.
    parts = []
    token = []
    depth = 0
    for line in body.splitlines():
        code = line.split("//", 1)[0]
        for char in code + "\n":
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
                if depth < 0:
                    raise ValueError("malformed configuration argument nesting")
            if char == "," and depth == 0:
                parts.append("".join(token).strip())
                token = []
            else:
                token.append(char)
    if depth != 0:
        raise ValueError("unclosed configuration argument")
    parts.append("".join(token).strip())
    values = {}
    for part in parts:
        if not part:
            continue
        name, separator, value = part.partition("=")
        name = name.strip()
        if not separator or not name.isidentifier() or name in values:
            raise ValueError(f"invalid or duplicate configuration argument {name!r}")
        values[name] = value.strip()
    return values


def selected_config_facts(source: str) -> dict:
    """Project inherited and overridden scalar fields with their Scala origin."""
    default = _assignments(_arguments(
        source, "val defaultMxFPConfig = GemminiArrayConfig[MxFloat, Float, Float](",
    ))
    override = _assignments(_arguments(
        source, "val standaloneMxFPConfig = defaultMxFPConfig.copy(",
    ))
    selected = {**default, **override}
    fields = {}
    integer_keys = (
        "meshRows", "meshColumns", "sp_banks", "acc_banks", "scaleSize",
        "spad_read_delay", "tile_latency", "acc_latency", "mesh_output_delay",
    )
    boolean_keys = (
        "use_mx_scaling", "enable_lut", "has_nonlinear_activations",
        "has_max_pool", "has_normalizations", "ex_read_from_acc", "ex_write_to_spad",
    )
    for name in integer_keys:
        value = selected.get(name)
        if value is None or not re.fullmatch(r"\d+", value):
            raise ValueError(f"selected {name} is not a literal integer")
        fields[name] = {"value": int(value), "origin": "standalone" if name in override else "default"}
    for name in boolean_keys:
        value = selected.get(name)
        if value not in {"true", "false"}:
            raise ValueError(f"selected {name} is not a literal Boolean")
        fields[name] = {"value": value == "true", "origin": "standalone" if name in override else "default"}
    for name in ("sp_capacity", "acc_capacity"):
        value = selected.get(name, "")
        match = re.fullmatch(r"CapacityInKilobytes\((\d+)\)", value)
        if match is None:
            raise ValueError(f"selected {name} is not a literal KiB capacity")
        fields[name + "_kib"] = {
            "value": int(match.group(1)), "origin": "standalone" if name in override else "default",
        }
    lut = selected.get("lut", "None")
    if lut != "None" and not lut.startswith("Some(GemminiLUTConfig("):
        raise ValueError("selected LUT declaration is not understood")
    fields["lut_present"] = {"value": lut != "None", "origin": "standalone" if "lut" in override else "default"}
    return fields


def check_operation_placement(spec: dict, fields: dict) -> None:
    """Refuse a software accelerator operation disabled by the selected RTL config."""
    operations = spec.get("operations")
    if not isinstance(operations, dict):
        raise ValueError("software spec has no operations")
    disabled = {
        "elementwise_map": "has_nonlinear_activations",
        "normalization": "has_normalizations",
        "pooling": "has_max_pool",
    }
    for name, row in operations.items():
        if not isinstance(row, dict) or row.get("placement") != "accelerator":
            continue
        families = row.get("families") or []
        if not isinstance(families, list) or not all(isinstance(family, str) for family in families):
            raise ValueError(f"{name}: invalid operation families")
        families = set(families) | {name}
        for family, switch in disabled.items():
            if family in families and fields[switch]["value"] is False:
                raise ValueError(f"{name}: selected RTL disables accelerator {family}")


def check_selected_config(spec: dict, contract: dict, fields: dict) -> None:
    """Cross-check contract claims against the selected source configuration."""
    if not fields["use_mx_scaling"]["value"]:
        raise ValueError("selected RTL configuration disables MX scaling")
    if fields["scaleSize"]["value"] != contract["block_size"]:
        raise ValueError("contract scale block size differs from selected RTL configuration")
    if fields["meshRows"]["value"] != 16 or fields["meshColumns"]["value"] != 16:
        raise ValueError("selected mesh geometry differs from DIM16 contract assumptions")
    if not fields["enable_lut"]["value"] or not fields["lut_present"]["value"]:
        raise ValueError("selected RTL configuration lacks the FP6 LUT")
    check_operation_placement(spec, fields)
