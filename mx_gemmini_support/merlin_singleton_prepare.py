"""Bind explicitly host-placed matrix sites to Merlin's unit reduction proof."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class MXSingletonReductionPreparation:
    """Preserve source arithmetic before MX catalog selection, with exact coverage."""

    def __init__(self, regions: frozenset[str], *, downstream=None):
        if not regions:
            raise ValueError("singleton preparation needs explicit host matrix sites")
        self.regions = regions
        self.downstream = downstream

    def __call__(self, source: Path, workdir: Path) -> Path:
        from merlin.frontends.linalg_mlir import parse_mlir_file
        from merlin.xdsl_dialects._common import text
        from merlin.xdsl_dialects.lowering.singleton_reductions import (
            parallelize_singleton_reductions,
        )

        source, workdir = Path(source), Path(workdir)
        before = source.read_bytes()
        module = parse_mlir_file(source)

        def selected(op):
            return getattr(op.attributes.get("prov.region_id"), "data", None) in self.regions

        result, transformed = parallelize_singleton_reductions(module, select=selected)
        records = [{"region": op.attributes["prov.region_id"].data,
                    "removed_reduction_axes": list(axes),
                    "operand_types": [str(value.type) for value in op.operands],
                    "indexing_maps": [str(attr.data) for attr in op.indexing_maps],
                    "scalar_operations": [step.name for step in op.body.block.ops],
                    "numeric_contract": "unchanged_scalar_body_initializer_and_types"}
                   for op, axes in transformed]
        if len(records) != len(self.regions) or {row["region"] for row in records} != self.regions:
            raise ValueError("host matrix policy has no unique proven prepared singleton reduction")
        if source.read_bytes() != before:
            raise ValueError("singleton preparation changed its immutable input")
        workdir.mkdir(parents=True, exist_ok=True)
        output = workdir / "host_singleton.mlir"
        output.write_text(text(result), encoding="utf-8")
        (workdir / "singleton_reductions.json").write_text(json.dumps({
            "schema": "mx_gemmini.host_singleton_reductions.v1",
            "source_sha256": hashlib.sha256(before).hexdigest(),
            "selected_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "regions": records}, indent=2) + "\n", encoding="utf-8")
        return self.downstream(output, workdir / "bf16") if self.downstream else output
