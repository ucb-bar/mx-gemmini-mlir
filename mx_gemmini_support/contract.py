"""Compile the authored software spec into one executable MX contract view."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import yaml

SCHEMA = "mx_gemmini.quant_contract.v1"


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def compile_contract(raw: bytes) -> dict[str, Any]:
    """Project only machine-consumed fields; reject conflicting declarations."""
    source = _require_mapping(yaml.safe_load(raw), "software spec")
    if source.get("schema") != "merlin.software_spec.v1" or source.get("target") != "mx_gemmini":
        raise ValueError("selected software spec is not the MX Gemmini contract")
    evidence = _require_mapping(source.get("evidence"), "evidence")
    rtl_commit = evidence.get("rtl_commit")
    mxgen_commit = evidence.get("mxgen_commit")
    if any(not isinstance(value, str) or len(value) != 40 or
           any(digit not in "0123456789abcdef" for digit in value)
           for value in (rtl_commit, mxgen_commit)):
        raise ValueError("contract needs exact Gemmini and MxGen commit pins")
    numerics = _require_mapping(source.get("numerical_semantics"), "numerical_semantics")
    quantization = _require_mapping(source.get("quantization"), "quantization")
    elements = _require_mapping(quantization.get("element_formats"), "element_formats")
    operations = _require_mapping(source.get("operations"), "operations")
    formats: dict[str, dict[str, Any]] = {}
    for row in quantization.get("formats") or ():
        row = _require_mapping(row, "quantization format")
        name = row.get("operand_dtype")
        if name in formats or name not in ("mxfp8", "mxfp6", "mxfp4"):
            raise ValueError(f"duplicate or unsupported MX format {name!r}")
        if row.get("accumulator_dtype") != "bf16" or row.get("scale_encoding") != "e8m0":
            raise ValueError(f"{name}: unsupported accumulator or scale encoding")
        operation_names = row.get("eligible_operations")
        if not isinstance(operation_names, list) or len(operation_names) != 1:
            raise ValueError(f"{name}: expected one selected contraction declaration")
        operation = _require_mapping(operations.get(operation_names[0]), f"{name} operation")
        if operation.get("placement") != "accelerator" or operation.get("operand_dtypes") != [name]:
            raise ValueError(f"{name}: selected operation does not match its format")
        if row.get("block_size") != numerics.get("block_size") or row.get("block_size") != 32:
            raise ValueError(f"{name}: unsupported scale block size")
        element = _require_mapping(elements.get(name), f"{name} element")
        for key in ("exponent_bits", "fraction_bits", "exponent_bias", "max_positive_code"):
            if type(element.get(key)) is not int or element[key] < 1:
                raise ValueError(f"{name}: invalid {key}")
        if element["max_positive_code"] >= 1 << (element["exponent_bits"] + element["fraction_bits"]):
            raise ValueError(f"{name}: positive code exceeds element width")
        semantics = _require_mapping(_require_mapping(operation.get("numerical_contract"),
                                                    f"{name} numerical contract").get("semantics"),
                                     f"{name} semantics")
        if semantics.get("operand_format") != name or semantics.get("block_size") != row["block_size"]:
            raise ValueError(f"{name}: numerical contract differs from quantization declaration")
        shape = _require_mapping(operation.get("shape_bounds"), f"{name} shape bounds")
        for axis in ("M", "N", "K"):
            bound = _require_mapping(shape.get(axis), f"{name} {axis} bound")
            if type(bound.get("min")) is not int or type(bound.get("multiple_of")) is not int:
                raise ValueError(f"{name}: noninteger {axis} bound")
        formats[name] = {
            "element": dict(element),
            "shape_bounds": {axis: dict(shape[axis]) for axis in ("M", "N", "K")},
            "site_modes": dict(row.get("site_modes") or {}),
            "pe_mode": semantics.get("pe_mode"),
            "config_ex_format_code": semantics.get("config_ex_format_code"),
            "packing": semantics.get("packing"),
        }
    if set(formats) != {"mxfp8", "mxfp6", "mxfp4"}:
        raise ValueError("selected RTL contract needs FP8, FP6, and FP4 declarations")
    intermediate = _require_mapping(elements.get("e3m1_intermediate"), "FP4 intermediate")
    output = _require_mapping(quantization.get("output_requantization"), "output_requantization")
    if set(output.get("formats") or ()) != set(formats) or output.get("source") != "accumulator":
        raise ValueError("output requantization differs from selected format set")
    projection = {
        "schema": SCHEMA,
        "target": source["target"],
        "status": source.get("status", "unreviewed"),
        "source_sha256": _digest(raw),
        "rtl_commit": rtl_commit,
        "rtl_config": evidence.get("rtl_config"),
        "rtl_config_class": evidence.get("rtl_config_class"),
        "mxgen_commit": mxgen_commit,
        "block_size": numerics["block_size"],
        "scale_encoding": numerics["scale_encoding"],
        "scale_rule": numerics["scale_rule"],
        "zero_block_scale_e8m0": numerics["zero_block_scale_e8m0"],
        "operand_rounding": numerics["operand_rounding"],
        "formats": formats,
        "e3m1_intermediate": dict(intermediate),
        "output_requantization": dict(output),
    }
    json.dumps(projection, sort_keys=True, allow_nan=False)
    return projection


def canonical_bytes(contract: Mapping[str, Any]) -> bytes:
    return json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def contract_digest(contract: Mapping[str, Any]) -> str:
    return _digest(canonical_bytes(contract))
