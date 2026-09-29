"""Explicit per-model MX format and reviewed FP6 LUT selection."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import yaml

SCHEMA = "mx_gemmini.quantization_policy.v1"
FORMATS = frozenset({"mxfp8", "mxfp6", "mxfp4", "host"})


@dataclass(frozen=True)
class MxPolicy:
    source_sha256: str
    default_format: str
    module_overrides: dict[str, str]
    functional_overrides: dict[str, str]
    graph_sha256: str | None
    fp6_codebooks: dict[str, Any]
    output_fp6_codebooks: dict[str, Any]
    output_chains: dict[str, dict[str, str]]

    def module_format(self, fqn: str) -> str:
        return self.module_overrides.get(fqn, self.default_format)

    def functional_format(self, site_id: str) -> str:
        return self.functional_overrides.get(site_id, self.default_format)

    def codebooks(self, site_id: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        selected = self.fp6_codebooks.get(site_id, self.fp6_codebooks.get("default"))
        if not isinstance(selected, dict) or selected.get("status") != "reviewed":
            raise ValueError(f"FP6 site {site_id!r} requires reviewed activation and weight codebooks")
        values = []
        for name in ("activation", "weight"):
            row = selected.get(name)
            if not isinstance(row, list) or len(row) != 16 or any(type(v) is not int or not 0 <= v < 64 for v in row):
                raise ValueError(f"FP6 {name} codebook must have exactly sixteen E3M2 codes")
            if len(set(row)) != 16:
                raise ValueError(f"FP6 {name} codebook must not repeat element codes")
            values.append(tuple(row))
        return values[0], values[1]

    def output_codebook(self, site_id: str) -> tuple[int, ...]:
        selected = self.output_fp6_codebooks.get(site_id, self.output_fp6_codebooks.get("default"))
        if not isinstance(selected, dict) or selected.get("status") != "reviewed":
            raise ValueError(f"FP6 output site {site_id!r} requires a reviewed output codebook")
        row = selected.get("codes")
        if (not isinstance(row, list) or len(row) != 16
                or any(type(v) is not int or not 0 <= v < 64 for v in row)
                or len(set(row)) != 16):
            raise ValueError("FP6 output codebook must have sixteen distinct E3M2 codes")
        return tuple(row)


def load_policy(raw: bytes) -> MxPolicy:
    document = yaml.safe_load(raw)
    if not isinstance(document, Mapping) or document.get("schema") != SCHEMA:
        raise ValueError(f"selected MX policy must declare {SCHEMA}")
    unknown = set(document) - {"schema", "default_format", "module_overrides",
                               "functional_overrides", "source_graph_sha256",
                               "fp6_codebooks", "output_fp6_codebooks", "output_chains"}
    if unknown:
        raise ValueError(f"MX policy has unknown fields: {sorted(unknown)}")
    default = document.get("default_format")
    if default not in FORMATS:
        raise ValueError("MX policy default_format is unsupported")
    overrides = []
    for key in ("module_overrides", "functional_overrides"):
        rows = document.get(key) or {}
        if not isinstance(rows, Mapping) or any(
            not isinstance(site, str) or not site or fmt not in FORMATS
            for site, fmt in rows.items()
        ):
            raise ValueError(f"MX policy {key} must map exact site IDs to supported formats")
        overrides.append(dict(rows))
    graph_sha = document.get("source_graph_sha256")
    if (overrides[1] or document.get("output_chains")) and (
        not isinstance(graph_sha, str) or len(graph_sha) != 64
        or any(char not in "0123456789abcdef" for char in graph_sha)
    ):
        raise ValueError("functional overrides and output chains require an exact source graph sha256")
    codebooks = document.get("fp6_codebooks") or {}
    if not isinstance(codebooks, Mapping):
        raise ValueError("fp6_codebooks must be a mapping")
    output_books = document.get("output_fp6_codebooks") or {}
    if not isinstance(output_books, Mapping):
        raise ValueError("output_fp6_codebooks must be a mapping")
    chains = document.get("output_chains") or {}
    if not isinstance(chains, Mapping):
        raise ValueError("output_chains must map exact producer site IDs")
    for source, row in chains.items():
        if (not isinstance(source, str) or not isinstance(row, Mapping)
                or set(row) != {"consumer", "format"}
                or not isinstance(row["consumer"], str)
                or row["format"] not in FORMATS - {"host"}):
            raise ValueError("output_chains needs consumer site ID and MX output format")
    policy = MxPolicy(
        hashlib.sha256(raw).hexdigest(), default, overrides[0], overrides[1],
        graph_sha, dict(codebooks), dict(output_books),
        {name: dict(row) for name, row in chains.items()},
    )
    for site in codebooks:
        policy.codebooks(site)
    for site in output_books:
        policy.output_codebook(site)
    return policy
