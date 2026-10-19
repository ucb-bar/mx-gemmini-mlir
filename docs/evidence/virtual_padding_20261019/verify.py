"""Reopen complete native projection pairs and optional actual decoder images."""

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def output_words(console, expected):
    words, begun, legacy = [], False, False
    lines = console.splitlines()
    assert lines.count("DONE") == 1
    for line in lines:
        fields = line.split()
        if fields[:1] == ["OUT"]:
            assert not begun and not legacy and int(fields[1]) == expected
            words = list(map(int, fields[2:]))
            legacy = True
        elif fields[:1] == ["OUT_BEGIN"]:
            assert not begun and not legacy and fields[1:] == ["f32", str(expected)]
            begun = True
        elif fields[:1] == ["OUT_CHUNK"]:
            assert begun and not legacy and fields[1] == "f32" and int(fields[2]) == len(words)
            assert int(fields[3]) == len(fields[4:])
            words.extend(map(int, fields[4:]))
    assert len(words) == expected and (begun or legacy)
    raw = b"".join(word.to_bytes(4, "little") for word in words)
    assert f"OUT_SHA256 f32le {expected} {len(raw)} {digest(raw)}" in lines
    assert f"OUT_SHA256_INDEX 0 rawbytes {expected} {len(raw)} {digest(raw)}" in lines
    return words, digest(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, help="optional mx_model_goal_20261018 with actual decoder ELFs")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "artifacts.json").read_text())
    archive = root / manifest["archive"]["name"]
    assert archive.stat().st_size == manifest["archive"]["bytes"]
    assert digest(archive.read_bytes()) == manifest["archive"]["sha256"]
    with tarfile.open(archive) as tar:
        assert all(member.isfile() for member in tar.getmembers())
        assert {member.name for member in tar.getmembers()} == set(manifest["members"])
        data = {member.name: tar.extractfile(member).read() for member in tar.getmembers()}
    for name, raw in data.items():
        assert len(raw) == manifest["members"][name]["bytes"]
        assert digest(raw) == manifest["members"][name]["sha256"]

    def read(name):
        return json.loads(data[name])

    for label, case in manifest["cases"].items():
        base = f"cases/{label}/"
        capture = read(base + "capture/capture_receipt.json")
        for name in ("model.mlir", "inputs.npz", "golden.npy", "weights.safetensors"):
            raw = data[base + "capture/" + name]
            assert digest(raw) == capture["artifacts"][name]["sha256"]
            assert len(raw) == capture["artifacts"][name]["bytes"]
        comparison = read(base + "comparison.json")
        assert comparison["all_words_exact"]
        arrays, receipts, images = [], [], []
        for name in ("current", "virtual"):
            build = base + name + "/"
            receipt = read(build + "mx_stage_probe.json")
            run = receipt["spike"]
            assert run["done"] and run["exit_code"] == 0 and not receipt["op_profile"]
            assert receipt["share_activation_packing"] and receipt["virtual_zero_padding"] == (name == "virtual")
            assert receipt["capture_model_sha256"] == digest(data[base + "capture/model.mlir"])
            assert receipt["checkpoint_weights_sha256"] == digest(data[base + "capture/weights.safetensors"])
            image = data[build + "model.elf"]
            recipe = read(build + "compilation_recipe.json")
            assert recipe["status"] == "completed" and recipe["executable"]["sha256"] == digest(image)
            admissions = [key for key in data if key.startswith(build + "invocations/")
                          and key.endswith("/policy/admission.json")]
            assert len(admissions) == 1
            admission = read(admissions[0])
            assert admission["status"] == "accepted" and admission["elf_sha256"] == digest(image)
            console = data[build + "spike.log"].decode()
            words, output_hash = output_words(console, case["rows"] * 81)
            expected = comparison["results"][name]
            assert output_hash == expected["output_sha256"]
            cycles = [int(line.split()[2]) for line in console.splitlines() if line.startswith("METRIC cycles ")]
            assert cycles == [expected["cycles"]]
            arrays.append(words)
            receipts.append(receipt)
            images.append(image)
        assert arrays[0] == arrays[1]
        for field in ("profile_sha256", "execution_policy", "spike_isa"):
            assert receipts[0][field] == receipts[1][field]
        for field in ("environment", "extension_sha256"):
            assert receipts[0]["spike"][field] == receipts[1]["spike"][field]
        before, after = (comparison["results"][name]["cycles"] for name in ("current", "virtual"))
        if case["rows"] == 160:
            assert images[0] == images[1] and before == after
        print(f"{label}: {before} -> {after}, {100*(1-after/before):.4f}% fewer cycles; every word exact")
    selected = manifest["decoder_static_resources"]
    raw = (root / selected["name"]).read_bytes()
    assert len(raw) == selected["bytes"] and digest(raw) == selected["sha256"]
    resources = json.loads(raw)
    if args.build_root is not None:
        tool = Path(resources["readelf"]["path"])
        assert digest(tool.read_bytes()) == resources["readelf"]["sha256"]
        for record in resources["builds"].values():
            elf = args.build_root / record["directory"] / "model.elf"
            assert elf.stat().st_size == record["elf_bytes"] and digest(elf.read_bytes()) == record["elf_sha256"]
            native = subprocess.check_output([str(tool), "-SW", str(elf)], text=True)
            assert native == record["readelf_stdout"]


if __name__ == "__main__":
    main()
