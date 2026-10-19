"""Check the promoted relation analysis against its frozen source and real inputs."""

from __future__ import annotations

import argparse
import ast
import copy
import gzip
import hashlib
import json
import types
from pathlib import Path

from merlin.frontends import attention_dataflow
from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir

from mx_gemmini_support import merlin_attention_pairs

HERE = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def executable_ast(raw):
    tree = ast.parse(raw)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def inspect():
    raw = (HERE / "analysis_before.py.txt").read_bytes()
    assert sha(raw) == "7ca67424d0cbbb70756a35433fbf5ce9c11f9e21da21343cd2388d9b24637efa"
    current = Path(attention_dataflow.__file__).read_bytes()
    assert executable_ast(raw) == executable_ast(current)
    legacy = types.ModuleType("frozen_attention_relations_380fc56")
    exec(compile(raw, "analysis_before.py.txt", "exec"), legacy.__dict__)
    input_report = HERE.parent / "semantic_policy_placement_20261019/report.json"
    input_raw = input_report.read_bytes()
    inputs = json.loads(input_raw)["inputs"]
    assert len(inputs) == 7
    rows = []
    for entry in inputs:
        path = Path(entry["path"])
        data = path.read_bytes()
        assert sha(data) == entry["file_sha256"]
        source = gzip.decompress(data) if path.suffix == ".gz" else data
        assert sha(source) == entry["source_sha256"]
        inventory = parse_linalg_mlir(source.decode())
        original = copy.deepcopy(inventory)
        before = legacy.find_attention_pairs(inventory)
        after = attention_dataflow.find_attention_pairs(inventory)
        through_provider = merlin_attention_pairs.find_attention_pairs(inventory)
        assert before == after == through_provider
        assert inventory == original
        rows.append({"path": str(path), "source_sha256": sha(source),
                     "pair_count": len(after), "pairs": after})
        print(path.name, len(after), "identical source relationships", flush=True)
    return {"scope": "source_relationship_and_ownership_only_no_new_lowering_or_runtime_claim",
            "before_revision": "380fc56", "before_sha256": sha(raw),
            "core_source_sha256": sha(current),
            "provider_source_sha256": sha(Path(merlin_attention_pairs.__file__).read_bytes()),
            "non_doc_ast_sha256": sha(executable_ast(current).encode()),
            "input_report_sha256": sha(input_raw), "inputs": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    result = inspect()
    if args.record:
        with (HERE / "report.json").open("x") as output:
            output.write(json.dumps(result, indent=2) + "\n")
    else:
        assert result == json.loads((HERE / "report.json").read_text())
