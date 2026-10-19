"""Bind MX packed-activation consumers to generic SSA preparation reuse."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


class MXSharedActivationPostOffload:
    """Use only checked catalog entries; leave selection experimental and explicit."""

    def __init__(self, catalog_path: Path, *, max_live_bytes: int):
        self.catalog_path = Path(catalog_path)
        self.max_live_bytes = max_live_bytes

    def __call__(self, prepared: Path, sidecar: Path, workdir: Path) -> Path:
        from merlin.frontends.linalg_mlir import parse_mlir_file
        from merlin.xdsl_dialects._common import text
        from merlin.xdsl_dialects.lowering.shared_preparation import (
            TensorPreparation,
            share_tensor_preparations,
        )
        from xdsl.dialects.builtin import TensorType, i8

        prepared, sidecar, workdir = map(Path, (prepared, sidecar, workdir))
        catalog = json.loads(self.catalog_path.read_text())
        routing = json.loads(sidecar.read_text())
        if (Path(routing["catalog_manifest"]).resolve() != self.catalog_path.resolve()
                or catalog["source_sha256"] != routing["model_sha256"]
                or catalog.get("shared_activation_packing") != "experimental_explicit_tensor"):
            raise ValueError("shared activation catalog differs from selected source routing")
        bindings = {row["region"]: row for row in catalog["bindings"]}
        if len(bindings) != len(catalog["bindings"]) or len(routing["routed"]) != len(bindings):
            raise ValueError("shared activation source binding count changed")
        kernels = {row["symbol"]: row for row in catalog["kernels"]
                   if row.get("activation_preparation")}
        selected = {}
        for row in routing["routed"]:
            binding = bindings.get(row["source_region"])
            if (binding is None or binding["source_operation_ordinal"] != row["source_operation_ordinal"]
                    or binding["tensor_types"] != row["tensor_types"]):
                raise ValueError("shared activation source binding changed")
            kernel = kernels.get(binding["symbol"])
            if kernel is not None:
                previous = selected.setdefault(row["symbol"], kernel)
                if previous != kernel:
                    raise ValueError("shared activation entry has inconsistent bindings")
        module = parse_mlir_file(prepared)
        expected = Counter(row["symbol"] for row in catalog["bindings"] if row["symbol"] in kernels)
        calls = [o for o in module.walk() if o.name == "func.call"
                 and o.callee.root_reference.data in selected]
        if Counter(selected[o.callee.root_reference.data]["symbol"] for o in calls) != expected:
            raise ValueError("shared activation calls differ from bound source")

        def select(call):
            kernel = selected.get(call.callee.root_reference.data)
            if kernel is None:
                return None
            d, packing = kernel["dimensions"], kernel["activation_preparation"]
            if (not kernel["prepacked_checkpoint_weight"] or kernel["batched"]
                    or len(call.arguments) != 3
                    or tuple(str(v.type) for v in call.arguments) != (
                        f"tensor<{d['m']}x{d['k']}xf32>", f"tensor<{d['k']}x{d['n']}xf32>",
                        f"tensor<{d['m']}x{d['n']}xf32>")):
                raise ValueError("shared activation consumer type or immutable weight changed")
            return TensorPreparation(0, packing["prepare_symbol"],
                                     TensorType(i8, (packing["packed_bytes"],)),
                                     packing["consumer_symbol"], tuple(packing["consumer_arguments"]))

        result, groups = share_tensor_preparations(module, select, max_live_bytes=self.max_live_bytes)
        workdir.mkdir(parents=True, exist_ok=True)
        output = workdir / "mx_shared_activation.mlir"
        output.write_text(text(result), encoding="utf-8")
        digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        (workdir / "mx_shared_activation_rewrite.json").write_text(json.dumps({
            "schema": "mx_gemmini.shared_activation_rewrite.v1",
            "source_sha256": digest(prepared), "output_sha256": digest(output),
            "catalog_sha256": digest(self.catalog_path), "sidecar_sha256": digest(sidecar),
            "max_live_prepared_bytes": self.max_live_bytes,
            "selection": "experimental_legality_and_byte_budget_only",
            "groups": groups,
            "removed_preparations": sum(g["removed_preparations"] for g in groups),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output
