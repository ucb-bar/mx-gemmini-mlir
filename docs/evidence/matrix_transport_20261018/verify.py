"""Verify complete matched matrix/residual executions without extracting files."""

from array import array
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def words(console, count):
    values, metrics, digest, indexed = [], {}, None, None
    lines = console.decode().splitlines()
    assert lines.count("DONE") == 1
    for line in lines:
        p = line.split()
        if not p:
            continue
        if p[0] == "OUT":
            assert not values and int(p[1]) == count and len(p[2:]) == count
            values.extend(map(int, p[2:]))
        elif p[0] == "OUT_BEGIN":
            assert p == ["OUT_BEGIN", "f32", str(count)] and not values
        elif p[0] == "OUT_CHUNK":
            assert p[1] == "f32" and int(p[2]) == len(values) and int(p[3]) == len(p[4:])
            values.extend(map(int, p[4:]))
        elif p[0] == "OUT_SHA256":
            assert digest is None and p[1:4] == ["f32le", str(count), str(count * 4)]
            digest = p[4]
        elif p[0] == "OUT_SHA256_INDEX":
            assert indexed is None and p[1:5] == ["0", "rawbytes", str(count), str(count * 4)]
            indexed = p[5]
        elif p[0] == "METRIC":
            assert p[1] not in metrics
            metrics[p[1]] = p[2]
    assert len(values) == count and all(0 <= value <= 0xffffffff for value in values)
    raw = array("I", values)
    assert raw.itemsize == 4
    if sys.byteorder != "little":
        raw.byteswap()
    raw = raw.tobytes()
    assert sha(raw) == digest == indexed
    return raw, metrics


def main():
    here = Path(__file__).resolve().parent
    for name, expected in json.loads((here / "artifacts.json").read_text()).items():
        assert sha((here / name).read_bytes()) == expected
    with tarfile.open(fileobj=io.BytesIO((here / "executions.tar.xz").read_bytes()), mode="r:xz") as archive:
        data = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    inventory = json.loads(data.pop("members.json"))
    assert data.keys() == inventory.keys()
    for name, expected in inventory.items():
        assert sha(data[name]) == expected, name
    for probe in ("tail19", "bias1", "mlp160", "mlp1", "shared"):
        report = json.loads(data[probe + "/comparison.json"])
        assert report["all_words_exact"] and "not_whole_model" in report["scope"]
        cap = json.loads(data[probe + "/capture/capture_receipt.json"])
        assert sha(data[probe + "/capture/capture_receipt.json"]) == report["capture_receipt_sha256"]
        for name, expected in cap["artifacts"].items():
            key = probe + "/capture/" + name
            if key in data:
                assert sha(data[key]) == expected["sha256"]
        baseline = None
        for label, row in report["results"].items():
            p = probe + "/" + label + "/"
            receipt = json.loads(data[p + "mx_stage_probe.json"])
            run = receipt["spike"]
            assert run["done"] and run["exit_code"] == 0
            assert sha(data[p + "mx_stage_probe.json"]) == row["stage_receipt_sha256"]
            assert run["environment"] == report["identity"]["environment"]
            assert run["extension_sha256"] == report["identity"]["extension_sha256"]
            assert receipt["execution_policy"] == report["identity"]["execution_policy"]
            assert receipt["profile_sha256"] == report["identity"]["profile_sha256"]
            assert receipt["capture_model_sha256"] == cap["artifacts"]["model.mlir"]["sha256"]
            assert receipt["checkpoint_weights_sha256"] == cap["artifacts"]["weights.safetensors"]["sha256"]
            image = data[p + "model.elf"]
            recipe = json.loads(data[p + "compilation_recipe.json"])
            admission = json.loads(data[p + "admission.json"])
            assert image[:4] == b"\x7fELF" and len(image) == row["elf_bytes"]
            assert sha(image) == row["elf_sha256"] == recipe["executable"]["sha256"]
            assert recipe["status"] == "completed" and admission["status"] == "accepted"
            assert admission["elf_sha256"] == sha(image)
            assert sha(data[p + "admission.json"]) == row["admission_sha256"]
            console = data[p + "spike.log"]
            assert sha(console) == row["console_sha256"]
            raw, metrics = words(console, row["output_elements"])
            assert sha(raw) == row["output_sha256"]
            assert int(metrics["cycles"]) == row["cycles"]
            assert metrics["build_hash"] == row["build_hash"] and int(metrics["memref_rank_mismatch"]) == 0
            if baseline is None:
                baseline = raw
            assert raw == baseline
            if label == "transport":
                assert len(row["private_transport_edges"]) == (2 if probe == "shared" else 1)
            if probe == "shared":
                assert len(row["groups"]) == 1
        first = report["results"][report["baseline"]]["cycles"]
        for row in report["results"].values():
            assert abs(row["cycles_saved_percent"] - 100 * (1 - row["cycles"] / first)) < 1e-10
        print(probe, {k: r["cycles"] for k, r in report["results"].items()}, "all words exact")
    print("PASS: complete words, image/admission custody and matched timing")


if __name__ == "__main__":
    main()
