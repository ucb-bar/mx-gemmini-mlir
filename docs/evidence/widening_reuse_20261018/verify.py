"""Reopen retained executables, receipts and every complete output word."""

import hashlib
import json
import tarfile
from pathlib import Path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
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
        row = manifest["members"][name]
        assert len(raw) == row["bytes"] and digest(raw) == row["sha256"], name

    def read(name):
        return json.loads(data[name])

    for rows in (1, 19):
        base = f"runs/shared_widening_projection{rows}/"
        capture = read(base + "capture/capture_receipt.json")
        for name in ("model.mlir", "inputs.npz", "golden.npy", "weights.safetensors"):
            raw = data[base + "capture/" + name]
            assert digest(raw) == capture["artifacts"][name]["sha256"]
            assert len(raw) == capture["artifacts"][name]["bytes"]
        for suffix, precision in (("", "FP8"), ("_fp4", "FP4")):
            comparison = read(base + f"comparison{suffix}.json")
            assert comparison["all_words_exact"] is True
            arrays, receipts = [], []
            for label in ("baseline", "shared"):
                build = base + label + suffix + "/"
                receipt = read(build + "mx_stage_probe.json")
                run = receipt["spike"]
                assert run["done"] and run["exit_code"] == 0 and not receipt["op_profile"]
                assert receipt["capture_model_sha256"] == digest(data[base + "capture/model.mlir"])
                assert receipt["checkpoint_weights_sha256"] == digest(data[base + "capture/weights.safetensors"])
                elf_hash = digest(data[build + "model.elf"])
                recipe = read(build + "compilation_recipe.json")
                assert recipe["status"] == "completed" and recipe["executable"]["sha256"] == elf_hash
                admissions = [name for name in data if name.startswith(build + "invocations/")
                              and name.endswith("/policy/admission.json")]
                assert len(admissions) == 1
                admission = read(admissions[0])
                assert admission["status"] == "accepted" and admission["elf_sha256"] == elf_hash
                console = data[build + "spike.log"].decode()
                lines = console.splitlines()
                assert lines.count("DONE") == 1
                outputs = [line.split() for line in lines if line.startswith("OUT ")]
                assert len(outputs) == 1 and int(outputs[0][1]) == rows * 81
                words = tuple(map(int, outputs[0][2:]))
                assert len(words) == rows * 81
                raw = b"".join(word.to_bytes(4, "little") for word in words)
                assert f"OUT_SHA256 f32le {len(words)} {len(raw)} {digest(raw)}" in lines
                assert f"OUT_SHA256_INDEX 0 rawbytes {len(words)} {len(raw)} {digest(raw)}" in lines
                cycles = [int(line.split()[2]) for line in lines if line.startswith("METRIC cycles ")]
                assert cycles == [comparison["results"][label]["cycles"]]
                assert digest(raw) == comparison["results"][label]["output_sha256"]
                arrays.append(words)
                receipts.append(receipt)
            assert arrays[0] == arrays[1]
            for field in ("profile_sha256", "execution_policy", "spike_isa"):
                assert receipts[0][field] == receipts[1][field]
            for field in ("environment", "extension_sha256"):
                assert receipts[0]["spike"][field] == receipts[1]["spike"][field]
            rewrite = read(base + "shared" + suffix + "/device_host_abi/mx_shared_activation_rewrite.json")
            assert len(rewrite["groups"]) == 1 and rewrite["removed_preparations"] == 2
            before = comparison["results"]["baseline"]["cycles"]
            after = comparison["results"]["shared"]["cycles"]
            print(f"{precision} rows={rows}: {before} -> {after}, {100 * (1-after/before):.4f}% fewer cycles; all words exact")


if __name__ == "__main__":
    main()
