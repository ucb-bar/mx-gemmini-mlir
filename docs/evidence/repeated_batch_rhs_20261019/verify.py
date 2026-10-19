"""Reopen complete native batches and a compiled broadcast/transpose graph."""

import hashlib
import importlib.util
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
    manifest = json.loads((here / "artifacts.json").read_text())
    raw = (here / manifest["archive"]).read_bytes()
    assert len(raw) == manifest["archive_bytes"] and sha(raw) == manifest["archive_sha256"]
    data = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts
            assert member.name not in data
            data[member.name] = archive.extractfile(member).read()
    assert set(data) == set(manifest["members"])
    for name, raw in data.items():
        assert manifest["members"][name] == {"sha256": sha(raw), "bytes": len(raw)}
    reader_path = here.parent / "broadcast_multiply_20261019/verify.py"
    assert sha(reader_path.read_bytes()) == manifest["elf_reader_sha256"]
    spec = importlib.util.spec_from_file_location("retained_elf_reader", reader_path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)

    def load(name):
        return json.loads(data[name])

    def array(name):
        return np.load(io.BytesIO(data[name]), allow_pickle=False)

    def device_counters(console):
        result = reader.counters(console)
        summary, = [line for line in console.splitlines() if line.startswith("gemmini perf [e4m3_vpu]:")]
        fields = summary.split(": ", 1)[1].split(";", 1)[0].split(", ")
        result["loops"] = int(fields[1].split()[0])
        mesh, = [line.strip() for line in console.splitlines() if line.strip().startswith("mesh:")]
        mesh_fields = mesh.split()
        result.update(mesh_tiles=int(mesh_fields[1]), mesh_busy_cycles=int(mesh_fields[3]),
                      vpu_commands=int(mesh_fields[7]))
        return result

    def admission(prefix, record, image):
        paths = [name for name in data if name.startswith(prefix) and name.endswith("/policy/admission.json")]
        assert len(paths) == 1
        actual = load(paths[0])
        assert actual["status"] == "accepted" and actual["elf_sha256"] == sha(image)
        assert record["elf_sha256"] == sha(image) and record["status"] == "accepted"
        assert record["report_sha256"] == sha(data[paths[0]])

    profile = load("profile.json")
    profile_sha = sha((json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n").encode())
    comparison = load("graph/comparison.json")
    extension_sha = comparison["identity"]["extension_sha256"]
    images, word_count = 0, 0
    for label in manifest["sections"]:
        prefix = "sections/" + label + "/"
        report = load(prefix + "receipt.json")
        assert report["profile_sha256"] == profile_sha and report["extension_sha256"] == extension_sha
        for name, digest in report["files_sha256"].items():
            assert sha(data[prefix + name]) == digest
        batch, m, n, k = report["shape_bmnk"]
        repeat = report["repeat"]
        a, b = array(prefix + "a.npy"), array(prefix + "b.npy")
        assert a.shape == (batch, m, k) and b.shape == (batch, k, n)
        payload = a.astype("<f4").tobytes() + b.astype("<f4").tobytes()
        assert payload == data[prefix + "payload.bin"]
        for index in range(batch):
            assert b[index].tobytes() == b[index // repeat * repeat].tobytes()
        outputs, counters = [], []
        for role in ["baseline", "retained"]:
            base = prefix + role + "/"
            result = report["results"][role]
            image = data[base + "section.elf"]
            reader.embedded_payload(image, payload)
            admission(base, result["admission"], image)
            environment = result["environment"]
            assert [environment[key] for key in ["GEMMINI_MODE", "GEMMINI_PERF_CONFIG", "GEMMINI_PERF_SET"]] == [
                "both", "e4m3_vpu", "mem.host_tracking=0"]
            assert Path(environment["GEMMINI_PERF_OUT"]).name == "perf.txt"
            lines = data[base + "spike.log"].decode().splitlines()
            words, observations = {}, {"PERF": [], "STATUS": [], "TRAP": []}
            for line in lines:
                fields = line.split()
                if not fields:
                    continue
                if fields[0] == "WORDS":
                    run, start = map(int, fields[1:3])
                    values = words.setdefault(run, [])
                    assert start == len(values)
                    values.extend(int(value, 16) for value in fields[3:])
                elif fields[0] in observations:
                    observations[fields[0]].append(dict((key, int(value)) for key, value in
                                                       (field.split("=", 1) for field in fields[1:])))
            refusal = report["refusal"] != "none"
            runs = 1 if refusal else 3
            assert list(words) == list(range(runs)) and all(len(v) == batch * m * n for v in words.values())
            assert observations["STATUS"] == result["status"] == [{"run": i, "guard": 0} for i in range(runs)]
            assert observations["PERF"] == result["performance"] and observations["TRAP"] == result["traps"]
            if refusal:
                assert result["exit_code"] == 77 and not observations["PERF"] and "DONE" not in lines
                assert len(observations["TRAP"]) == 1 and observations["TRAP"][0]["cause"] == 3
                reader.check_breakpoint(image, observations["TRAP"][0]["pc"])
                # Failure happens during preparation, before publishing this batch.
                completed = repeat + 1 if report["refusal"] == "lhs" else repeat
                expected = np.full((batch, m, n), 0xA5A50000, dtype="<u4")
                expected[:completed] = (a[:completed].astype(np.int32) @ b[:completed].astype(np.int32)).astype(
                    "<f4").view("<u4")
                assert np.array_equal(np.asarray(words[0], dtype="<u4").reshape(batch, m, n), expected)
            else:
                assert result["exit_code"] == 0 and not observations["TRAP"] and lines.count("DONE") == 1
                assert [row["run"] for row in observations["PERF"]] == [0, 1, 2]
                for run, values in words.items():
                    changed = b.copy()
                    if run:
                        group = run - 1
                        changed[group * repeat:(group + 1) * repeat, 0, 0] *= -1
                    golden = (a.astype(np.int32) @ changed.astype(np.int32)).astype("<f4").view("<u4")
                    assert golden.tobytes() == array(base + f"golden_{run}.npy").tobytes()
                    assert golden.tobytes() == np.asarray(values, dtype="<u4").tobytes()
            observed = np.asarray(list(words.values()), dtype="<u4").tobytes()
            assert observed == data[base + "observed_words.bin"] and sha(observed) == result["words_sha256"]
            outputs.append(observed)
            counters.append(device_counters(data[base + "perf.txt"].decode()))
            images += 1
            word_count += sum(map(len, words.values()))
        assert outputs[0] == outputs[1] and counters[0] == counters[1]
        assert report["results"]["baseline"]["workspace_bytes"] == report["results"]["retained"]["workspace_bytes"]
        if not refusal:
            before = report["results"]["baseline"]["performance"][0]["cycles"]
            after = report["results"]["retained"]["performance"][0]["cycles"]
            print(label, before, "->", after, "cold native batch cycles")
        else:
            print(label, "preserved complete partial output and actual breakpoint")

    capture_prefix = "graph/capture/capture/"
    capture = load(capture_prefix + "capture_receipt.json")
    assert sha(data[capture_prefix + "capture_receipt.json"]) == comparison["capture_receipt_sha256"]
    for name, identity in capture["artifacts"].items():
        assert identity == {"bytes": len(data[capture_prefix + name]), "sha256": sha(data[capture_prefix + name])}
    assert comparison["all_words_exact"] and comparison["identity"]["profile_sha256"] == profile_sha
    outputs, counters = [], []
    for role in ["baseline", "retained"]:
        prefix = "graph/" + role + "/"
        row = comparison["results"][role]
        stage = load(prefix + "mx_stage_probe.json")
        image = data[prefix + "model.elf"]
        assert sha(image) == row["elf_sha256"] and len(image) == row["elf_bytes"]
        assert sha(data[prefix + "mx_stage_probe.json"]) == row["stage_receipt_sha256"]
        assert stage["spike"]["exit_code"] == 0 and stage["spike"]["done"] and not stage["op_profile"]
        assert stage["spike"]["environment"] == comparison["identity"]["environment"]
        assert stage["execution_policy"] == comparison["identity"]["execution_policy"]
        assert stage["capture_model_sha256"] == sha(data[capture_prefix + "model.mlir"])
        assert stage["checkpoint_weights_sha256"] == sha(data[capture_prefix + "weights.safetensors"])
        assert stage["arena_estimate"] == row["arena_estimate"]
        recipe = load(prefix + "compilation_recipe.json")
        assert recipe["status"] == "completed" and recipe["executable"]["sha256"] == sha(image)
        admissions = [name for name in data if name.startswith(prefix) and name.endswith("/policy/admission.json")]
        assert len(admissions) == 1 and sha(data[admissions[0]]) == row["admission_sha256"]
        assert load(admissions[0])["elf_sha256"] == sha(image) and load(admissions[0])["status"] == "accepted"
        console = data[prefix + "spike.log"]
        assert sha(console) == row["console_sha256"]
        lines = console.decode().splitlines()
        chunks = [line.split() for line in lines if line.startswith("OUT ")]
        assert len(chunks) == 1 and int(chunks[0][1]) == row["output_elements"] == 2622
        bits = list(map(int, chunks[0][2:]))
        assert len(bits) == 2622
        values = struct.pack("<" + "I" * len(bits), *bits)
        assert sha(values) == row["output_sha256"] and np.isfinite(np.frombuffer(values, dtype="<f4")).all()
        assert [line.split()[-1] for line in lines if line.startswith(("OUT_SHA256 ", "OUT_SHA256_INDEX "))] == [
            sha(values), sha(values)]
        metrics = dict(line.split()[1:] for line in lines if line.startswith("METRIC "))
        assert int(metrics["cycles"]) == row["cycles"] and metrics["memref_rank_mismatch"] == "0"
        assert metrics["build_hash"] == row["build_hash"] == stage["build_hash"] and lines.count("DONE") == 1
        catalog = load(prefix + "device_catalog/mx_catalog.json")
        assert sha(data[prefix + "device_catalog/mx_catalog.o"]) == catalog["compilation"]["object_sha256"]
        for name, digest in catalog["compilation"]["source_sha256"].items():
            filename = {"runtime_group_prepare": "runtime_group_prepare.h", "runtime_quant_header": "runtime_quant.h",
                        "runtime_quant_lut": "runtime_quant_lut.h"}.get(name, name + ".c")
            assert sha(data[prefix + "device_catalog/" + filename]) == digest
        if role == "retained":
            rewrite = load(prefix + "device_host_abi/mx_batch_rhs_reuse_rewrite.json")
            assert sha(data[prefix + "device_host_abi/mx_batch_rhs_reuse.mlir"]) == rewrite["output_sha256"]
            assert sha(data[prefix + "device_catalog/mx_catalog.json"]) == rewrite["catalog_sha256"]
            assert sha(data[prefix + "device_signatures.json"]) == rewrite["sidecar_sha256"]
            assert sum(item["calls"] for item in rewrite["consumers"]) == 1
            assert rewrite["consumers"][0]["input_proofs"][0]["contiguous_batch_repeat"] == 3
        outputs.append(values)
        counters.append(device_counters(console.decode()))
        images += 1
        word_count += len(bits)
    assert outputs[0] == outputs[1] and counters[0] == counters[1]
    before, after = [comparison["results"][role]["cycles"] for role in ["baseline", "retained"]]
    assert comparison["results"]["retained"]["cycles_saved_percent"] == 100 * (1 - after / before)
    print("complete compiled graph", before, "->", after, "cycles")
    print("PASS:", len(data), "files,", images, "actual ELFs,", word_count, "observed output words")


if __name__ == "__main__":
    main()
