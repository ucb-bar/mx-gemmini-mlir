"""Equal sample values without a source proof retain the complete original route."""

import hashlib
import io
import json
import struct
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / "fallback.json").read_text())
    raw = (here / manifest["archive"]).read_bytes()
    assert len(raw) == manifest["archive_bytes"] and sha(raw) == manifest["archive_sha256"]
    data = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts and member.name not in data
            data[member.name] = archive.extractfile(member).read()
    assert set(data) == set(manifest["members"])
    for name, raw in data.items():
        assert manifest["members"][name] == {"sha256": sha(raw), "bytes": len(raw)}

    def load(name):
        return json.loads(data[name])

    comparison = load("comparison.json")
    capture = load("capture/capture/capture_receipt.json")
    assert sha(data["capture/capture/capture_receipt.json"]) == comparison["capture_receipt_sha256"]
    for name, identity in capture["artifacts"].items():
        raw = data["capture/capture/" + name]
        assert identity == {"sha256": sha(raw), "bytes": len(raw)}
    with np.load(io.BytesIO(data["capture/capture/inputs.npz"]), allow_pickle=False) as inputs:
        rhs = inputs["in1"]
        assert rhs.shape == (6, 23, 37)
        for index in range(6):
            assert rhs[index].tobytes() == rhs[index // 3 * 3].tobytes()
    outputs = []
    for role in ["baseline", "retained"]:
        prefix = role + "/"
        row = comparison["results"][role]
        stage = load(prefix + "mx_stage_probe.json")
        assert stage["spike"]["exit_code"] == 0 and stage["spike"]["done"] and not stage["op_profile"]
        assert stage["spike"]["environment"] == comparison["identity"]["environment"]
        assert stage["execution_policy"] == comparison["identity"]["execution_policy"]
        assert sha(data[prefix + "mx_stage_probe.json"]) == row["stage_receipt_sha256"]
        image = data[prefix + "model.elf"]
        assert sha(image) == row["elf_sha256"] and len(image) == row["elf_bytes"] == 66352
        recipe = load(prefix + "compilation_recipe.json")
        assert recipe["status"] == "completed" and recipe["executable"]["sha256"] == sha(image)
        admissions = [name for name in data if name.startswith(prefix) and name.endswith("/policy/admission.json")]
        assert len(admissions) == 1 and sha(data[admissions[0]]) == row["admission_sha256"]
        assert load(admissions[0])["elf_sha256"] == sha(image) and load(admissions[0])["status"] == "accepted"
        console = data[prefix + "spike.log"]
        assert sha(console) == row["console_sha256"]
        lines = console.decode().splitlines()
        observed, = [line.split() for line in lines if line.startswith("OUT ")]
        assert int(observed[1]) == len(observed[2:]) == 2622
        words = struct.pack("<" + "I" * 2622, *map(int, observed[2:]))
        assert sha(words) == row["output_sha256"]
        assert [line.split()[-1] for line in lines if line.startswith(("OUT_SHA256 ", "OUT_SHA256_INDEX "))] == [
            sha(words), sha(words)]
        metrics = dict(line.split()[1:] for line in lines if line.startswith("METRIC "))
        assert int(metrics["cycles"]) == row["cycles"] == 1091675
        assert metrics["build_hash"] == stage["build_hash"] == row["build_hash"]
        assert metrics["memref_rank_mismatch"] == "0" and lines.count("DONE") == 1
        outputs.append(words)
    rewrite = load("retained/device_host_abi/mx_batch_rhs_reuse_rewrite.json")
    assert rewrite["selected_calls"] == 0 and rewrite["consumers"] == []
    assert rewrite["source_sha256"] == rewrite["output_sha256"] == sha(
        data["retained/device_host_abi/mx_batch_rhs_reuse.mlir"])
    assert rewrite["catalog_sha256"] == sha(data["retained/device_catalog/mx_catalog.json"])
    for name in ["mx_dense_kernel_0.c", "mx_dense_kernel_0.o", "mx_issue_0.c", "mx_issue_0.o", "runtime_quant.o"]:
        assert data["baseline/device_catalog/" + name] == data["retained/device_catalog/" + name]
    assert outputs[0] == outputs[1] and comparison["all_words_exact"]
    assert comparison["results"]["baseline"]["perf"] == comparison["results"]["retained"]["perf"]
    assert comparison["results"]["baseline"]["arena_estimate"] == comparison["results"]["retained"]["arena_estimate"]
    print("PASS:", len(data), "files, two actual ELFs, equal sample batches without source proof;")
    print("zero selected calls, original IR/native objects, all 2622 words and 1091675 cycles unchanged")


if __name__ == "__main__":
    main()
