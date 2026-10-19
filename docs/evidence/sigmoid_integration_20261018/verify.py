"""Reopen complete projection-stage outputs, images and fusion obligations."""

import hashlib
import json
import struct
import tarfile
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    for name, digest in json.loads((here / "artifacts.json").read_text()).items():
        assert sha((here / name).read_bytes()) == digest, name
    with tarfile.open(here / "executions.tar.xz", "r:xz") as archive:
        data = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    inventory = json.loads(data.pop("members.json"))
    assert inventory.keys() == data.keys()
    for name, digest in inventory.items():
        assert sha(data[name]) == digest, name
    report = json.loads(data["comparison.json"])
    assert report["all_words_exact"]
    previous = None
    for label, row in report["results"].items():
        prefix = label + "/"
        image = data[prefix + "model.elf"]
        assert image[:4] == b"\x7fELF" and sha(image) == row["elf_sha256"]
        recipe = json.loads(data[prefix + "compilation_recipe.json"])
        assert recipe["status"] == "completed" and recipe["executable"]["sha256"] == sha(image)
        admissions = [v for k, v in data.items() if k.startswith(prefix + "invocations/")
                      and k.endswith("/policy/admission.json")]
        assert len(admissions) == 1 and sha(admissions[0]) == row["admission_sha256"]
        admission = json.loads(admissions[0])
        assert admission["status"] == "accepted" and admission["elf_sha256"] == sha(image)
        console = data[prefix + "spike.log"]
        assert sha(console) == row["console_sha256"]
        lines = console.decode().splitlines()
        assert lines.count("DONE") == 1
        outputs = [line.split() for line in lines if line.startswith("OUT ")]
        assert len(outputs) == 1
        words = list(map(int, outputs[0][2:]))
        assert int(outputs[0][1]) == len(words) == row["output_elements"] == 703
        raw = struct.pack("<" + "I" * len(words), *words)
        assert sha(raw) == row["output_sha256"]
        assert previous is None or raw == previous
        previous = raw
        assert [int(line.split()[2]) for line in lines if line.startswith("METRIC cycles ")] == [row["cycles"]]
        receipt = json.loads(data[prefix + "mx_stage_probe.json"])
        assert sha(data[prefix + "mx_stage_probe.json"]) == row["stage_receipt_sha256"]
        assert receipt["raw_selected_source_coverage_complete"] and receipt["spike"]["done"]
        assert receipt["vpu_elementwise_lowered_regions"] == 2 and receipt["vpu_unary_lowered_regions"] == 1
        assert not any(receipt["vpu_residual_eligible"].values())
        if label == "connected":
            chains = receipt["connected_vpu_sigmoid_multiply_chains"]
            assert len(chains) == 1 and len(chains[0]["regions"]) == 2 and chains[0]["unary_region"]
            rewrite = json.loads(data[prefix + "device_host_abi/mx_connected_sigmoid_rewrite.json"])
            assert rewrite["chains"] == chains
            assert sha(data[prefix + "device_host_abi/mx_connected_sigmoid.mlir"]) == rewrite["output_sha256"]
        print(label, row["cycles"], "cycles, all703 words exact")
    print("PASS: complete compiler stages; this does not establish full-model numerical quality")


if __name__ == "__main__":
    main()
