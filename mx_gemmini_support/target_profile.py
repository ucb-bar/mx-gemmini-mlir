"""Source-bound structural MX target profiles.

This projects *named* configurations from the selected Gemmini/MxGen Scala
sources. It intentionally fails on source expressions it cannot resolve. A
profile describes elaborated capabilities; it is not evidence that a command
sequence or numerical mode has passed simulation.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .config_facts import _arguments, _assignments


SCHEMA = "mx_gemmini.target_profile.v2"
FORMATS = {
    "fp4_e2m1": ("FP4", 2),
    "fp6_e2m3": ("FP6_E2M3", 4),
    "fp6_e3m2": ("FP6_E3M2", 3),
    "fp8_e4m3": ("FP8_E4M3", 4),
    "fp8_e5m2": ("FP8_E5M2", 3),
}
_BY_SCALA = {value[0]: name for name, value in FORMATS.items()}
_BASE_MODE = {
    (2, 2): 0, (2, 3): 1, (2, 4): 2,
    (3, 2): 3, (3, 3): 4, (3, 4): 5,
    (4, 2): 6, (4, 3): 7, (4, 4): 8,
}
_SCALA_FILES = (
    "src/main/scala/gemmini/ConfigsFP.scala",
    "src/main/scala/gemmini/Arithmetic.scala",
    "src/main/scala/gemmini/GemminiConfigs.scala",
    "src/main/scala/gemmini/MxConfigFragments.scala",
    "chipyard/GemminiConfigs.scala",
    "mxgen/src/main/scala/mxgen/MxParameters.scala",
    "mxgen/src/main/scala/mxgen/Classifier.scala",
    "src/main/scala/gemmini/GemminiISA.scala",
    "src/main/scala/gemmini/vpu/Vpu.scala",
    "src/main/scala/gemmini/SpadRequant.scala",
)


def canonical_bytes(profile: dict[str, Any]) -> bytes:
    return (json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n").encode()


def profile_sha256(profile: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(profile)).hexdigest()


def _read_sources(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    sources, hashes = {}, {}
    for relative in _SCALA_FILES:
        raw = (root / relative).read_bytes()
        sources[relative] = raw.decode()
        hashes[relative] = hashlib.sha256(raw).hexdigest()
    return sources, hashes


def _revision(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def verify_profile_source(profile: dict[str, Any], root: Path) -> None:
    """Reject source drift and capability edits against the selected RTL checkout."""
    if profile.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA}")
    source = profile.get("source")
    if not isinstance(source, dict):
        raise ValueError("MX profile lacks source identity")
    _, hashes = _read_sources(root.resolve())
    expected = {"gemmini_revision": _revision(root),
                "mxgen_revision": _revision(root / "mxgen"),
                "sha256": hashes}
    if source != expected:
        raise ValueError("MX profile differs from selected Gemmini/MxGen source closure")
    generated = export_profiles(root)
    name = profile.get("name")
    if name not in generated or profile != generated[name]:
        raise ValueError("MX profile capabilities differ from selected Gemmini/MxGen sources")


def load_profile(path: Path, *, rtl_root: Path | None = None) -> dict[str, Any]:
    profile = json.loads(path.read_text())
    if not isinstance(profile, dict) or profile.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA}")
    if not isinstance(profile.get("legal_compute"), list) or not profile["legal_compute"]:
        raise ValueError("MX profile has no compute capabilities")
    if profile.get("qualification") != "structural_unqualified":
        raise ValueError("unknown MX profile qualification state")
    if rtl_root is not None:
        verify_profile_source(profile, rtl_root)
    return profile


def _named_bodies(source: str, prefix: str) -> dict[str, str]:
    markers = list(re.finditer(rf"^  {prefix} (\w+) = ", source, re.M))
    result = {}
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(source)
        result[marker.group(1)] = source[marker.end():end].strip()
    return result


def _first_call_body(source: str, marker: str) -> str:
    """Return the outer call body; nested `.copy` calls may use the same name."""
    start = source.index(marker) + len(marker)
    depth = 1
    quoted = comment = escaped = False
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
    raise ValueError(f"unclosed Scala call {marker}")


def _resolve_array_configs(source: str) -> dict[str, dict[str, str]]:
    section = source[source.index("object GemminiMxFPConfigs {"):]
    section = section[:section.index("\nclass GemminiMxFPDefaultConfig")]
    bodies = _named_bodies(section, "lazy val")
    if "defaultMxFPConfig" not in bodies:
        raise ValueError("MX default array config is absent")
    resolved: dict[str, dict[str, str]] = {}

    def resolve(name: str, active: frozenset[str] = frozenset()) -> dict[str, str]:
        if name in resolved:
            return resolved[name]
        if name in active or name not in bodies:
            raise ValueError(f"unknown or cyclic MX array config {name}")
        body = bodies[name]
        if name == "defaultMxFPConfig":
            values = _assignments(_arguments(body, "GemminiArrayConfig[MxFloat, Float, Float]("))
        else:
            match = re.match(r"(\w+)(?:\.copy\()", body)
            if match:
                values = {**resolve(match.group(1), active | {name}),
                          **_assignments(_first_call_body(body, ".copy("))}
            else:
                alias = re.fullmatch(r"(\w+)", re.sub(r"//[^\n]*", "", body).strip())
                if not alias:
                    raise ValueError(f"unrecognized MX array config expression: {name}")
                values = resolve(alias.group(1), active | {name}).copy()
        resolved[name] = values
        return values

    for name in bodies:
        resolve(name)
    return resolved


def _mx_config_methods(source: str) -> dict[str, str]:
    section = source[source.index("object MxConfig {"):]
    section = section[:section.index("\n// MX FLOAT BUNDLE")]
    markers = list(re.finditer(r"^  def (\w+)(?:\s*=|\()", section, re.M))
    return {
        marker.group(1): section[marker.start():markers[index + 1].start()
                                       if index + 1 < len(markers) else len(section)]
        for index, marker in enumerate(markers)
    }


def _format_set(text: str) -> set[str]:
    if "MxFormat.all" in text:
        return set(FORMATS)
    names = re.findall(r"MxFormat\.(FP4|FP6_E2M3|FP6_E3M2|FP8_E4M3|FP8_E5M2)", text)
    if not names:
        raise ValueError("MX format set is absent")
    return {_BY_SCALA[name] for name in names}


def _pe_capability(methods: dict[str, str], config_name: str) -> tuple[set[str], set[str], set[int]]:
    if config_name not in methods:
        raise ValueError(f"unknown MxConfig.{config_name}")
    body = methods[config_name]
    if config_name in {"mxGemminiE5M2", "mxGemminiAll"}:
        lhs, rhs, inherited = _pe_capability(methods, "mxGemmini")
        additions = _format_set(body)
        lhs, rhs = lhs | additions, rhs | additions
        modes = inherited
    else:
        sets = re.findall(r"Set\(([^)]*)\)", body)
        if len(sets) >= 2:
            lhs, rhs = _format_set(sets[0]), _format_set(sets[1])
        elif "MxFormat.all, MxFormat.all" in body:
            lhs = rhs = set(FORMATS)
        else:
            raise ValueError(f"cannot resolve format sets for MxConfig.{config_name}")
        modes = {_BASE_MODE[(FORMATS[a][1], FORMATS[b][1])] for a in lhs for b in rhs}
    if "MxPEParams.allModes" in body:
        modes = set(range(12))
    else:
        match = re.search(r"modesOverride\s*=\s*Some\(List\(([^)]*)\)\)", body)
        if match:
            modes = {int(number) for number in re.findall(r"MxPEParams\.mode(\d+)", match.group(1))}
    return lhs, rhs, modes


def _selected_pe_config(values: dict[str, str]) -> str:
    lhs = values.get("inputType", "")
    rhs = values.get("weightType", "")
    explicit = re.search(r"MxConfig\.(\w+)\)", lhs)
    if explicit:
        other = re.search(r"MxConfig\.(\w+)\)", rhs)
        if not other or other.group(1) != explicit.group(1):
            raise ValueError("activation and weight use different PE configurations")
        return explicit.group(1)
    widths = [re.search(r"MxFloat\((\d+),\s*(\d+),", operand) for operand in (lhs, rhs)]
    if any(item is None for item in widths):
        raise ValueError("cannot infer MxFloat operand widths")
    if any(int(item.group(1)) >= 5 for item in widths):
        return "mxGemminiE5M2"
    if any(int(item.group(2)) >= 4 for item in widths):
        return "mxGemminiAll"
    return "mxGemmini"


def _int_field(values: dict[str, str], key: str, default: int | None = None) -> int:
    raw = values.get(key)
    if raw is None and default is not None:
        return default
    if raw is None or not re.fullmatch(r"\d+", raw):
        raise ValueError(f"MX {key} is not a literal integer")
    return int(raw)


def _capacity(values: dict[str, str], key: str) -> int:
    match = re.fullmatch(r"CapacityInKilobytes\((\d+)\)", values.get(key, ""))
    if not match:
        raise ValueError(f"MX {key} is not a literal KiB capacity")
    return int(match.group(1)) * 1024


def _bool_field(values: dict[str, str], key: str, default: bool = False) -> bool:
    raw = values.get(key)
    if raw is None:
        return default
    if raw not in {"true", "false"}:
        raise ValueError(f"MX {key} is not a literal Boolean")
    return raw == "true"


def _scala_integer(raw: str) -> int:
    """Evaluate the small literal integer grammar used by MX config fields."""
    raw = raw.strip()
    if re.fullmatch(r"(?:0x[0-9a-fA-F]+|\d+)L?", raw):
        return int(raw.removesuffix("L"), 0)
    for separator, operation in (("+", lambda a, b: a + b),
                                 ("<<", lambda a, b: a << b),
                                 ("*", lambda a, b: a * b)):
        if separator in raw:
            parts = raw.split(separator)
            if len(parts) >= 2:
                result = _scala_integer(parts[0])
                for part in parts[1:]:
                    result = operation(result, _scala_integer(part))
                return result
    raise ValueError(f"unrecognized MX Scala integer expression {raw!r}")


def _case_defaults(source: str, case_name: str) -> dict[str, str]:
    body = _arguments(source, f"case class {case_name}(")
    body = re.sub(r"(?m)^(\s*\w+)\s*:\s*[A-Za-z][A-Za-z0-9_]*(?:\[[^]]+\])?\s*=", r"\1 =", body)
    return _assignments(body)


def _option_parameters(configs: dict[str, dict[str, str]], name: str, field: str,
                       case_name: str, defaults: dict[str, str]) -> dict[str, str] | None:
    raw = configs[name].get(field, "None").strip()
    if raw == "None":
        return None
    constructor = f"Some({case_name}("
    if raw.startswith(constructor):
        body = _first_call_body(raw, constructor)
        # LUT constructors may use positional numBits/numEntries; parse those
        # separately and keep the named parameters relevant to compiler paths.
        if case_name == "GemminiLUTConfig":
            body = re.sub(r"^\s*Seq\([^)]*\)\s*,\s*Seq\([^)]*\)\s*,", "", body)
        return {**defaults, **_assignments(body)}
    reference = re.match(rf"(?:Some\()?([A-Za-z_]\w*)\.{field}\.(?:get\.copy\(|map\(_\.copy\()", raw)
    if reference:
        base = _option_parameters(configs, reference.group(1), field, case_name, defaults)
        if base is None:
            raise ValueError(f"MX {name}.{field} copies a missing option")
        return {**base, **_assignments(_first_call_body(raw, ".copy("))}
    raise ValueError(f"unrecognized MX {name}.{field} option expression {raw!r}")


def _hardware_options(configs: dict[str, dict[str, str]], name: str,
                      source: str) -> dict[str, Any]:
    scale = _option_parameters(configs, name, "scale_mem", "GemminiScalingFactorMemConfig",
                               _case_defaults(source, "GemminiScalingFactorMemConfig"))
    requant = _option_parameters(configs, name, "requantizer", "GemminiRequantizerConfig",
                                 _case_defaults(source, "GemminiRequantizerConfig"))
    lut = _option_parameters(configs, name, "lut", "GemminiLUTConfig",
                             {"rdataWidth": "6", "raddrWidth": "4",
                              "projFormat": "LutFP6E3M2", "actCodeWidth": "0", "weiCodeWidth": "0"})
    result: dict[str, Any] = {"scale_mem": None, "requantizer": None, "lut": None}
    if scale is not None:
        result["scale_mem"] = {
            field: _scala_integer(scale[key]) for field, key in (
                ("base_address", "baseAddr"), ("size_bytes", "sizeInBytes"),
                ("subbank_line_bytes", "subbankLineSizeInBytes"),
                ("subbanks_per_bank", "subbanksPerBank"), ("banks", "numBanks"))}
    if requant is not None:
        result["requantizer"] = {
            field: _scala_integer(requant[key]) for field, key in (
                ("base_address", "baseAddr"), ("input_lanes", "numInputLanes"),
                ("output_lanes", "numOutputLanes"),
                ("min_output_bits", "minOutputBits"),
                ("max_output_bits", "maxOutputBits"))}
        # This field is absent from most overrides and then inherits the case-class default.
        result["requantizer"]["gpu_input_lanes"] = _scala_integer(requant["numGPUInputLanes"])
    if lut is not None:
        projection = lut["projFormat"].strip()
        if projection not in {"LutFP6E3M2", "LutFP6E2M3", "LutFP8E4M3", "LutFP8E5M2"}:
            raise ValueError(f"unknown MX LUT projection {projection}")
        result["lut"] = {
            "projection_format": projection,
            "read_data_bits": _scala_integer(lut["rdataWidth"]),
            "address_bits": _scala_integer(lut["raddrWidth"]),
            "activation_code_bits": _scala_integer(lut["actCodeWidth"]) or _scala_integer(lut["rdataWidth"]),
            "weight_code_bits": _scala_integer(lut["weiCodeWidth"]) or _scala_integer(lut["rdataWidth"]),
        }
    return result


def _projection_choices(fmt: str, has_lut: bool, has_quad_mode: bool) -> tuple[str, ...]:
    if fmt == "fp4_e2m1":
        return ("direct",)
    if fmt == "fp8_e4m3":
        return ("direct", "lut") if has_lut and has_quad_mode else ("direct",)
    return ("lut",) if has_lut else ()


def _legal_tuples(lhs: set[str], rhs: set[str], modes: set[int], has_lut: bool) -> list[dict[str, Any]]:
    result = []
    for a in sorted(lhs):
        for b in sorted(rhs):
            sig_a, sig_b = FORMATS[a][1], FORMATS[b][1]
            for act_projection in _projection_choices(a, has_lut, bool(modes & {9, 10, 11})):
                for weight_projection in _projection_choices(b, has_lut, bool(modes & {9, 10, 11})):
                    act_lut = act_projection == "lut"
                    weight_lut = weight_projection == "lut"
                    mode = _BASE_MODE[(sig_a, sig_b)]
                    if sig_a == sig_b == 4 and act_lut and weight_lut:
                        mode = 9
                    elif sig_a == 4 and sig_b in {2, 3} and act_lut:
                        mode = 10
                    elif sig_b == 4 and sig_a in {2, 3} and weight_lut:
                        mode = 11
                    if mode in modes:
                        result.append({"activation_format": a, "weight_format": b,
                                       "activation_projection": act_projection,
                                       "weight_projection": weight_projection,
                                       "pe_mode": mode})
    return result


def _fragment_classes(source: str) -> dict[str, str]:
    markers = list(re.finditer(r"^class (Gemmini\w*MxFP\w*Config) extends (?:Config\(|(\w+))", source, re.M))
    result = {}
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(source)
        body = source[marker.end():end]
        selected = re.search(r"GemminiMxFPConfigs\.(\w+)", body)
        if selected:
            result[marker.group(1)] = selected.group(1)
        elif marker.group(2):
            result[marker.group(1)] = marker.group(2)
        else:
            raise ValueError(f"cannot resolve Gemmini config class {marker.group(1)}")
    return result


def _chipyard_classes(source: str) -> dict[str, str]:
    markers = list(re.finditer(r"^class ((?:Mx|TestMx|TestRequantizer)\w*Config) extends (?:Config\(|(\w+))", source, re.M))
    result = {}
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(source)
        body = source[marker.end():end]
        selected = re.search(r"new gemmini\.(Gemmini\w+Config)", body)
        if selected:
            result[marker.group(1)] = selected.group(1)
        elif marker.group(2):
            result[marker.group(1)] = marker.group(2)
        else:
            raise ValueError(f"cannot resolve Chipyard config class {marker.group(1)}")
    return result


def _chipyard_overrides(body: str) -> dict[str, int | None]:
    """Record explicit SoC wrapper fields without inventing inherited defaults."""
    def literal(pattern: str) -> int | None:
        match = re.search(pattern, body)
        return int(match.group(1)) if match else None

    return {
        "system_bus_width_bits": literal(r"WithSystemBusWidth\((\d+)\)"),
        "rocket_cores": literal(r"WithNHugeCores\((\d+)\)"),
        "l2_banks": literal(r"WithNBanks\((\d+)\)"),
        "l2_outer_latency_cycles": literal(r"WithInclusiveCache\(outerLatencyCycles\s*=\s*(\d+)\)"),
        "serial_tl_sink_bits": literal(r"TLBUNDLE_PARAMS\.copy\(sinkBits\s*=\s*(\d+)\)"),
    }


def _chipyard_bodies(source: str) -> dict[str, str]:
    markers = list(re.finditer(r"^class ((?:Mx|TestMx|TestRequantizer)\w*Config) extends ", source, re.M))
    return {marker.group(1): source[marker.end():markers[index + 1].start()
                                         if index + 1 < len(markers) else len(source)]
            for index, marker in enumerate(markers)}


def export_profiles(root: Path) -> dict[str, dict[str, Any]]:
    """Return structural profiles for every MX fragment and Chipyard wrapper."""
    root = root.resolve()
    sources, hashes = _read_sources(root)
    configs = _resolve_array_configs(sources[_SCALA_FILES[0]])
    methods = _mx_config_methods(sources["mxgen/src/main/scala/mxgen/MxParameters.scala"])
    hardware_source = sources["src/main/scala/gemmini/MxConfigFragments.scala"]
    arithmetic = sources["src/main/scala/gemmini/Arithmetic.scala"]
    classifier = sources["mxgen/src/main/scala/mxgen/Classifier.scala"]
    isa = sources["src/main/scala/gemmini/GemminiISA.scala"]
    vpu_abi = sources["src/main/scala/gemmini/vpu/Vpu.scala"]
    spad_abi = sources["src/main/scala/gemmini/SpadRequant.scala"]
    for source, marker in (
        (arithmetic, "if (act.expWidth >= 5 || wei.expWidth >= 5) MxConfig.mxGemminiE5M2"),
        (arithmetic, "else if (act.sigWidth >= 4 || wei.sigWidth >= 4) MxConfig.mxGemminiAll"),
        (classifier, "object requiredPEMode"),
        (classifier, "actLutEn && weiLut && a.sig === 4.U && w.sig === 4.U, 9.U"),
        (classifier, "actLutEn && a.sig === 4.U && (w.sig === 2.U || w.sig === 3.U), 10.U"),
        (classifier, "weiLut && w.sig === 4.U && (a.sig === 2.U || a.sig === 3.U), 11.U"),
        (isa, "val VPU_EXEC = 33.U"),
        (isa, "val SPAD_REQUANT = 34.U"),
        (vpu_abi, "rs1 = src1[13:0] | src2[27:14] | dst[41:28] | rows[57:42]"),
        (vpu_abi, "rs2 = op[3:0] | bcast[4] | rlen[14:5] | imm[31:16]"),
        (spad_abi, "rs1 = src[13:0] | dst[27:14] | tiled[28] | resident[29] | scale DRAM addr[62:30]"),
        (spad_abi, "rs2 = M[15:0] | N[31:16] | fp4[32]"),
    ):
        if marker not in source:
            raise ValueError(f"MX format/mode selection changed: {marker}")
    fragments = _fragment_classes(sources[_SCALA_FILES[0]])
    wrappers = _chipyard_classes(sources[_SCALA_FILES[4]])
    wrapper_bodies = _chipyard_bodies(sources[_SCALA_FILES[4]])
    if len(fragments) < 40 or len(wrappers) < 40:
        raise ValueError("named MX configuration census shrank unexpectedly")
    source_identity = {"gemmini_revision": _revision(root),
                       "mxgen_revision": _revision(root / "mxgen"),
                       "sha256": hashes}
    result = {}
    selections = {**{name: (name, None) for name in fragments},
                  **{name: (fragment, name) for name, fragment in wrappers.items()}}
    for name, (fragment, chipyard_config) in sorted(selections.items()):
        fragment_class = fragment
        seen = set()
        while fragment not in configs:
            if fragment in seen:
                raise ValueError(f"unknown Gemmini config class {fragment}")
            seen.add(fragment)
            if fragment in wrappers:
                fragment = wrappers[fragment]
            elif fragment in fragments:
                fragment = fragments[fragment]
            else:
                raise ValueError(f"unknown Gemmini config class {fragment}")
        values = configs[fragment]
        pe_config = _selected_pe_config(values)
        lhs, rhs, modes = _pe_capability(methods, pe_config)
        lut_expression = values.get("lut", "None")
        if lut_expression != "None" and not lut_expression.startswith("Some(GemminiLUTConfig("):
            raise ValueError(f"unrecognized LUT declaration in {fragment}")
        has_lut = (_bool_field(values, "enable_lut", True) and lut_expression != "None")
        hardware = _hardware_options(configs, fragment, hardware_source)
        if has_lut != (hardware["lut"] is not None):
            raise ValueError(f"MX LUT enable and construction disagree in {fragment}")
        mmio_raw = values.get("mx_mmio_base", "None")
        mmio = None
        if mmio_raw != "None":
            if not mmio_raw.startswith("Some("):
                raise ValueError(f"unrecognized MX MMIO base in {fragment}")
            mmio = _scala_integer(_first_call_body(mmio_raw, "Some("))
        vpu = None
        if _bool_field(values, "has_vpu"):
            raw = values.get("vpu_params", "")
            if not raw.startswith("gemmini.vpu.VpuParams("):
                raise ValueError(f"MX VPU parameters are absent in {fragment}")
            fields = _assignments(_first_call_body(raw, "gemmini.vpu.VpuParams("))
            vpu = {"units": _int_field(fields, "units"),
                   "exp_sub": _bool_field(fields, "expSub"),
                   "exp_sum": _bool_field(fields, "expSum")}
        profile = {
            "schema": SCHEMA,
            "name": name,
            "source": source_identity,
            "chipyard_config": chipyard_config,
            "chipyard_overrides": (_chipyard_overrides(wrapper_bodies[chipyard_config])
                                   if chipyard_config else None),
            "gemmini_fragment": fragment_class,
            "gemmini_config": fragment,
            "pe_config": pe_config,
            "transport": "rocket_rocc",
            "geometry": {"mesh_rows": _int_field(values, "meshRows"),
                         "mesh_columns": _int_field(values, "meshColumns"),
                         "tile_rows": _int_field(values, "tileRows"),
                         "tile_columns": _int_field(values, "tileColumns")},
            "resources": {"scratchpad_bytes": _capacity(values, "sp_capacity"),
                          "scratchpad_banks": _int_field(values, "sp_banks"),
                          "accumulator_bytes": _capacity(values, "acc_capacity"),
                          "accumulator_banks": _int_field(values, "acc_banks"),
                          "scale_block_elements": _int_field(values, "scaleSize"),
                          "lut": has_lut,
                          "vpu": _bool_field(values, "has_vpu"),
                          "spad_requant": _bool_field(values, "has_spad_requant"),
                          "requantizer": values.get("requantizer", "None") != "None",
                          "dma_max_bytes": _int_field(values, "dma_maxbytes"),
                          "dma_bus_width_bits": _int_field(values, "dma_buswidth"),
                          "max_spad_writer_bytes": _int_field(values, "max_spad_writer_bytes"),
                          "max_in_flight_mem_reqs": _int_field(values, "max_in_flight_mem_reqs"),
                          "load_queue_entries": _int_field(values, "ld_queue_length"),
                          "store_queue_entries": _int_field(values, "st_queue_length"),
                          "execute_queue_entries": _int_field(values, "ex_queue_length"),
                          "has_mx_mmio_requant": _bool_field(values, "has_mx_mmio_requant", True),
                          "has_loop_retire_counter": _bool_field(values, "has_loop_retire_counter"),
                          "performance_counters": _int_field(values, "num_counter"),
                          "mx_mmio_base": mmio,
                          "scale_mem_config": hardware["scale_mem"],
                          "lut_config": hardware["lut"],
                          "requantizer_config": hardware["requantizer"],
                          "vpu_config": vpu},
            "legal_compute": _legal_tuples(lhs, rhs, modes, has_lut),
            "candidate_output_modes": ["bf16"] + (["fp4_e2m1", "fp6_e3m2", "fp8_e4m3"]
                                                     if values.get("requantizer", "None") != "None" else []),
            "qualification": "structural_unqualified",
        }
        if not profile["legal_compute"]:
            raise ValueError(f"{name} has no legal MX compute tuple")
        result[name] = profile
    return result


def require_compute(profile: dict[str, Any], activation: str, weight: str,
                    *, pe_mode: int, activation_projection: str,
                    weight_projection: str) -> None:
    if profile.get("schema") != SCHEMA:
        raise ValueError("expected source-bound MX target profile v2")
    selected = {"activation_format": activation, "weight_format": weight,
                "activation_projection": activation_projection,
                "weight_projection": weight_projection, "pe_mode": pe_mode}
    if selected not in profile.get("legal_compute", []):
        raise ValueError(f"MX compute tuple is absent from {profile.get('name', 'selected')} profile: {selected}")
