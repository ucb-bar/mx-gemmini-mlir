"""Verify saved model2MLIR, manifest, and MX dialect files as one selection."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from .handoff import validate_handoff


def verify_files(source_mlir: Path, handoff_mlir: Path, manifest_file: Path,
                 contract_file: Path, policy_file: Path, mx_opt: Path) -> dict:
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser

    source_text = source_mlir.read_text()
    handoff_text = handoff_mlir.read_text()
    manifest = json.loads(manifest_file.read_text())
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, handoff_text).parse_module()
    capture = SimpleNamespace(ok=True, module=module,
                              quantization_manifest=manifest, mlir_text=source_text)
    bound = validate_handoff(capture, contract_file.read_bytes(), policy_file.read_bytes())
    observed_source = getattr(module.attributes.get("mx.source_mlir_sha256"), "data", None)
    if observed_source != hashlib.sha256(source_text.encode()).hexdigest():
        raise ValueError("MX handoff is not bound to the selected source MLIR bytes")
    contracted = set()
    requantized = set()
    for op in module.walk():
        name = getattr(getattr(op, "op_name", None), "data", op.name)
        if name not in {"mx_gemmini.contract", "mx_gemmini.requantize"}:
            continue
        site = getattr(op.attributes.get("site_id"), "data", None)
        if name == "mx_gemmini.contract":
            contracted.add(site)
        else:
            requantized.add(site)
    selected = {site["site_id"] for site in manifest["sites"] if site["status"] == "quantized"}
    chains = {site["site_id"] for site in manifest["sites"] if site.get("output_chain")}
    if contracted != selected or requantized != chains:
        raise ValueError("MX dialect operation sites differ from selected manifest census")
    subprocess.run([str(mx_opt), str(handoff_mlir), "-o", "/dev/null"], check=True)
    return {"schema": "mx_gemmini.handoff_check.v1", "status": "contract_handoff",
            "manifest_sha256": bound["manifest_sha256"],
            "source_mlir_sha256": bound["source_mlir_sha256"],
            "quantized_sites": len(selected), "output_chains": len(chains)}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Verify a saved MX dialect handoff")
    parser.add_argument("--source-mlir", type=Path, required=True)
    parser.add_argument("--handoff-mlir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--mx-opt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_files(args.source_mlir, args.handoff_mlir, args.manifest,
                                  args.contract, args.policy, args.mx_opt),
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
