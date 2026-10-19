"""Reopen complete execution words, admitted images and matched stage costs."""

from array import array
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def output(console, expected):
    words, metrics, first_hash, indexed_hash = [], {}, None, None
    lines = console.decode().splitlines()
    assert lines.count("DONE") == 1
    for line in lines:
        p = line.split()
        if not p:
            continue
        if p[0] == "OUT_BEGIN":
            assert p == ["OUT_BEGIN", "f32", str(expected)] and not words
        elif p[0] == "OUT_CHUNK":
            assert p[1] == "f32" and int(p[2]) == len(words) and int(p[3]) == len(p[4:])
            words.extend(map(int, p[4:]))
        elif p[0] == "OUT":
            assert not words and int(p[1]) == expected and len(p[2:]) == expected
            words.extend(map(int, p[2:]))
        elif p[0] == "OUT_SHA256":
            assert first_hash is None and p[1:4] == ["f32le", str(expected), str(expected * 4)]
            first_hash = p[4]
        elif p[0] == "OUT_SHA256_INDEX":
            assert indexed_hash is None and p[1:5] == ["0", "rawbytes", str(expected), str(expected * 4)]
            indexed_hash = p[5]
        elif p[0] == "METRIC":
            assert p[1] not in metrics
            metrics[p[1]] = p[2]
    assert len(words) == expected and all(0 <= w <= 0xffffffff for w in words)
    raw = array("I", words)
    assert raw.itemsize == 4
    if sys.byteorder != "little":
        raw.byteswap()
    raw = raw.tobytes()
    assert digest(raw) == first_hash == indexed_hash
    return raw, metrics


def main():
    here = Path(__file__).resolve().parent
    for name, expected in json.loads((here / "artifacts.json").read_text()).items():
        assert digest((here / name).read_bytes()) == expected, name
    with tarfile.open(fileobj=io.BytesIO((here / "executions.tar.xz").read_bytes()), mode="r:xz") as archive:
        data = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    inventory = json.loads(data.pop("members.json"))
    assert data.keys() == inventory.keys()
    for name, expected in inventory.items():
        assert digest(data[name]) == expected, name
    for probe in ("tails19", "qkv160", "qkv1"):
        report = json.loads(data[probe + "/comparison.json"])
        assert report["all_words_exact"]
        assert "not_whole_model" in report["scope"]
        capture = json.loads(data[probe + "/capture/capture_receipt.json"])
        assert digest(data[probe + "/capture/capture_receipt.json"]) == report["capture_receipt_sha256"]
        for name, artifact in capture["artifacts"].items():
            key = probe + "/capture/" + name
            if key in data:
                assert digest(data[key]) == artifact["sha256"]
        baseline = None
        for variant, row in report["results"].items():
            prefix = probe + "/" + variant + "/"
            receipt = json.loads(data[prefix + "mx_stage_probe.json"])
            run = receipt["spike"]
            assert run["done"] and run["exit_code"] == 0
            assert digest(data[prefix + "mx_stage_probe.json"]) == row["stage_receipt_sha256"]
            assert run["environment"] == report["identity"]["environment"]
            assert run["extension_sha256"] == report["identity"]["extension_sha256"]
            assert receipt["execution_policy"] == report["identity"]["execution_policy"]
            assert receipt["profile_sha256"] == report["identity"]["profile_sha256"]
            assert receipt["capture_model_sha256"] == capture["artifacts"]["model.mlir"]["sha256"]
            assert receipt["checkpoint_weights_sha256"] == capture["artifacts"]["weights.safetensors"]["sha256"]
            image = data[prefix + "model.elf"]
            recipe = json.loads(data[prefix + "compilation_recipe.json"])
            admission = json.loads(data[prefix + "admission.json"])
            assert image[:4] == b"\x7fELF" and len(image) == row["elf_bytes"]
            assert digest(image) == row["elf_sha256"] == recipe["executable"]["sha256"]
            assert recipe["status"] == "completed" and admission["status"] == "accepted"
            assert admission["elf_sha256"] == digest(image)
            assert digest(data[prefix + "admission.json"]) == row["admission_sha256"]
            console = data[prefix + "spike.log"]
            assert digest(console) == row["console_sha256"]
            raw, metrics = output(console, row["output_elements"])
            assert digest(raw) == row["output_sha256"]
            assert int(metrics["cycles"]) == row["cycles"]
            assert metrics["build_hash"] == row["build_hash"]
            assert int(metrics["memref_rank_mismatch"]) == 0
            for line in console.decode().splitlines():
                parts = line.split()
                perf = row["perf"]
                if line.startswith("gemmini perf ["):
                    assert parts[2].strip("[]:") == perf["mode"]
                    assert tuple(map(int, (parts[3], parts[5], parts[7]))) == (
                        perf["commands"], perf["loops"], perf["fences"])
                elif line.startswith("  mesh:"):
                    assert (int(parts[1]), int(parts[3]), int(parts[7])) == (
                        perf["mesh_tiles"], perf["mesh_busy_cycles"], perf["vpu_commands"])
                elif line.startswith("  load:"):
                    assert int(parts[1]) == perf["load_bytes"]
                    assert int(parts[parts.index("scales:") + 1].strip(";")) == perf["scale_bytes"]
            if baseline is None:
                baseline = raw
            assert raw == baseline, (probe, variant)
        base = report["results"][report["baseline"]]["cycles"]
        for row in report["results"].values():
            assert abs(row["cycles_saved_percent"] - 100 * (1 - row["cycles"] / base)) < 1e-10
        print(probe, {k: v["cycles"] for k, v in report["results"].items()}, "all words exact")
    tails = json.loads(data["tails19/comparison.json"])
    assert not tails["results"]["budget_refused"]["groups"]
    assert tails["results"]["budget_refused"]["cycles"] == tails["results"]["baseline"]["cycles"]
    print("PASS: complete stage equivalence, image custody, admission records and timing arithmetic")


if __name__ == "__main__":
    main()
