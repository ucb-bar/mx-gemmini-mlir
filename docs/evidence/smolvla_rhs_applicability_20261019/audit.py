"""Replay the current batch repetition proof on the archived complete SmolVLA IR."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import repeated_batch_input, tensor_broadcast
from merlin.llvmlower.repeated_batch_input import (
    prove_repeated_batch_input,
    validate_repeated_batch_input,
)
from xdsl.dialects.func import CallOp

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent / "smolvla_latest_session_20261019"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def inspect():
    manifest_raw = (ARCHIVE / "artifacts.json").read_bytes()
    manifest = json.loads(manifest_raw)
    archive = ARCHIVE / "executions.tar.xz"
    assert archive.stat().st_size == manifest["archive_bytes"]
    with archive.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == manifest["archive_sha256"]
    files = {}
    for program in ("prefix_encode", "flow_denoise"):
        for name in ("device_host_abi/mx_vpu_matrix_transport.mlir",
                     "device_catalog/mx_catalog.json", "device/catalog_binding.json"):
            logical = f"candidate/stages/{program}/{name}"
            files[logical] = manifest["aliases"].get(logical, logical)
    bodies = {}
    with tarfile.open(archive, "r:xz") as source:
        for name in set(files.values()):
            member = source.getmember(name)
            assert member.isfile()
            bodies[name] = source.extractfile(member).read()
    data = {logical: bodies[stored] for logical, stored in files.items()}
    for name, raw in data.items():
        assert len(raw) == manifest["members"][name]["bytes"]
        assert digest(raw) == manifest["members"][name]["sha256"]

    stages = []
    for program in ("prefix_encode", "flow_denoise"):
        prefix = f"candidate/stages/{program}/"
        source_name = prefix + "device_host_abi/mx_vpu_matrix_transport.mlir"
        catalog_raw = data[prefix + "device_catalog/mx_catalog.json"]
        catalog = json.loads(catalog_raw)
        binding = json.loads(data[prefix + "device/catalog_binding.json"])
        assert binding["source_sha256"] == catalog["source_sha256"]
        assert binding["manifest_sha256"] == digest(catalog_raw)
        assert binding["catalog_object_sha256"] == catalog["compilation"]["object_sha256"]
        kernels = {row["symbol"]: row for row in catalog["kernels"]}
        assert len(kernels) == len(catalog["kernels"])
        selected = {name: native for name, native in binding["entry_to_kernel"].items()
                    if kernels[native]["batched"]}
        module = parse_mlir_text(data[source_name].decode())
        module.verify()
        observations = []
        for ordinal, operation in enumerate(module.walk()):
            if not isinstance(operation, CallOp) or operation.callee.root_reference.data not in selected:
                continue
            rhs = operation.operands[1]
            row = {"ordinal": ordinal, "callee": operation.callee.root_reference.data,
                   "rhs_type": str(rhs.type), "producer": rhs.owner.name}
            try:
                proof = prove_repeated_batch_input(rhs)
                validate_repeated_batch_input(proof)
            except ValueError as exc:
                row.update(status="refused", reason=str(exc))
            else:
                row.update(status="proved", proof=proof.to_dict())
            chain, current, seen = [], rhs, set()
            while hasattr(current.owner, "name") and current.owner not in seen:
                owner = current.owner
                seen.add(owner)
                item = {"operation": owner.name, "result_type": str(current.type)}
                if owner.name == "linalg.transpose":
                    item["permutation"] = list(owner.permutation.get_values())
                if owner.name == "linalg.generic":
                    item["scalar_body"] = [op.name for op in owner.body.block.ops]
                    item["input_types"] = [str(value.type) for value in owner.inputs]
                    item["indexing_maps"] = [str(value) for value in owner.indexing_maps]
                chain.append(item)
                if owner.name in ("tensor.expand_shape", "tensor.collapse_shape", "linalg.transpose"):
                    current = owner.operands[0]
                elif owner.name == "linalg.generic" and len(owner.inputs) == 1:
                    current = owner.inputs[0]
                else:
                    break
            row["source_chain"] = chain
            observations.append(row)
        assert len(observations) == sum(row["symbol"] in set(selected.values())
                                        for row in catalog["bindings"])
        assert {row["callee"] for row in observations} == set(selected)
        stages.append({"stage": program, "source_member": source_name,
                       "batch_calls": len(observations),
                       "proved_calls": sum(row["status"] == "proved" for row in observations),
                       "refusal_counts": dict(Counter(row["reason"] for row in observations
                                                      if row["status"] == "refused")),
                       "observations": observations})
    return {"schema": "archived_batch_rhs_applicability.v1",
            "scope": "source_eligibility_only_not_new_runtime_or_general_impossibility_proof",
            "archive_sha256": manifest["archive_sha256"],
            "manifest_sha256": digest(manifest_raw),
            "inputs": {name: digest(raw) for name, raw in data.items()},
            "proof_sources": {Path(module.__file__).name: digest(Path(module.__file__).read_bytes())
                              for module in (repeated_batch_input, tensor_broadcast)},
            "stages": stages}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()
    result = inspect()
    if args.write_report:
        with args.write_report.open("x") as output:
            output.write(json.dumps(result, indent=2) + "\n")
    else:
        assert result == json.loads((HERE / "report.json").read_text())
    for stage in result["stages"]:
        print(f'{stage["stage"]}: {stage["proved_calls"]}/{stage["batch_calls"]} proved; '
              f'{stage["refusal_counts"]}')
