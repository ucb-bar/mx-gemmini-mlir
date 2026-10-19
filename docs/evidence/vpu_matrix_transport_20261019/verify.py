"""Reopen compiled reverse-transport stages and exceptional producer boundaries."""

import hashlib
import importlib.util
import io
import json
import struct
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    manifest = json.loads((HERE / "artifacts.json").read_text())
    archive = (HERE / "executions.tar.xz").read_bytes()
    assert sha(archive) == manifest["archive_sha256"] and len(archive) == manifest["archive_bytes"]
    data = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:xz") as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts
            assert member.name not in data
            data[member.name] = tar.extractfile(member).read()
    assert set(data) == set(manifest["members"])
    for name, raw in data.items():
        assert {"sha256": sha(raw), "bytes": len(raw)} == manifest["members"][name]
    reader_path = HERE.parent / "broadcast_multiply_20261019/verify.py"
    assert sha(reader_path.read_bytes()) == manifest["elf_reader_sha256"]
    spec = importlib.util.spec_from_file_location("retained_elf_reader", reader_path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)

    def load(name):
        return json.loads(data[name])

    profile = load("profile.json")
    profile_sha = sha((json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n").encode())
    images, words_checked = 0, 0
    for label, pair in manifest["pairs"].items():
        comparison = load(f"stages/{label}/comparison.json")
        capture = f'captures/{pair["rows"]}/capture/'
        capture_receipt = load(capture + "capture_receipt.json")
        for name in ["model.mlir", "weights.safetensors", "inputs.npz", "golden.npy"]:
            assert sha(data[capture + name]) == capture_receipt["artifacts"][name]["sha256"]
        assert comparison["capture_receipt_sha256"] == sha(data[capture + "capture_receipt.json"])
        assert comparison["all_words_exact"] is True
        assert comparison["identity"]["profile_sha256"] == profile_sha
        assert comparison["identity"]["environment"] == {
            "GEMMINI_MODE": "both", "GEMMINI_PERF_CONFIG": "e4m3_vpu",
            "GEMMINI_PERF_SET": "mem.host_tracking=0", "GEMMINI_PERF_OUT": None}
        arrays = []
        counters = []
        for role in ["baseline", "compact"]:
            prefix = f"stages/{label}/{role}/"
            receipt = load(prefix + "mx_stage_probe.json")
            row = comparison["results"][role]
            image = data[prefix + "model.elf"]
            assert sha(image) == row["elf_sha256"] and len(image) == row["elf_bytes"]
            assert sha(data[prefix + "mx_stage_probe.json"]) == row["stage_receipt_sha256"]
            assert receipt["spike"]["exit_code"] == 0 and receipt["spike"]["done"] is True
            assert not receipt["op_profile"] and receipt["spike"]["environment"] == comparison["identity"]["environment"]
            assert receipt["capture_model_sha256"] == sha(data[capture + "model.mlir"])
            assert receipt["checkpoint_weights_sha256"] == sha(data[capture + "weights.safetensors"])
            assert receipt["execution_policy"] == comparison["identity"]["execution_policy"]
            assert receipt["arena_estimate"] == row["arena_estimate"]
            recipe = load(prefix + "compilation_recipe.json")
            assert recipe["status"] == "completed" and recipe["executable"]["sha256"] == sha(image)
            admissions = [n for n in data if n.startswith(prefix + "invocations/") and n.endswith("/policy/admission.json")]
            assert len(admissions) == 1 and sha(data[admissions[0]]) == row["admission_sha256"]
            admission = load(admissions[0])
            assert admission["status"] == "accepted" and admission["elf_sha256"] == sha(image)
            raw = data[prefix + "spike.log"]
            assert sha(raw) == row["console_sha256"]
            lines = raw.decode().splitlines()
            assert lines.count("DONE") == 1
            outputs = [s.split() for s in lines if s.startswith("OUT ")]
            assert len(outputs) == 1 and int(outputs[0][1]) == pair["rows"] * 81
            bits = list(map(int, outputs[0][2:]))
            assert len(bits) == pair["rows"] * 81
            values = struct.pack("<" + "I" * len(bits), *bits)
            assert sha(values) == row["output_sha256"] and len(bits) == row["output_elements"]
            assert np.isfinite(np.frombuffer(values, dtype="<f4")).all()
            hashes = [s.split()[-1] for s in lines if s.startswith(("OUT_SHA256 ", "OUT_SHA256_INDEX "))]
            assert hashes == [sha(values), sha(values)]
            metrics = dict(s.split()[1:] for s in lines if s.startswith("METRIC "))
            assert int(metrics["cycles"]) == row["cycles"] and metrics["memref_rank_mismatch"] == "0"
            assert metrics["build_hash"] == receipt["build_hash"] == row["build_hash"]
            arrays.append(values)
            counters.append(reader.counters(raw.decode()))
            if role == "compact":
                rewrite_path = prefix + "device_host_abi/mx_vpu_matrix_transport_rewrite.json"
                rewrite = load(rewrite_path)
                assert rewrite == receipt["private_vpu_matrix_transport_rewrite"]
                assert sha(data[prefix + "device_host_abi/mx_vpu_matrix_transport.mlir"]) == rewrite["output_sha256"]
                assert sha(data[prefix + "device_catalog/mx_catalog.json"]) == rewrite["catalog_sha256"]
                assert len(rewrite["edges"]) == (0 if label == "1_virtual" else 1)
                catalog = load(prefix + "device_catalog/mx_catalog.json")
                if label == "1_virtual":
                    assert not any(k.get("activation_preparation", {}).get("private_bf16_input") for k in catalog["kernels"])
                else:
                    for kernel in catalog["kernels"]:
                        packing = kernel["activation_preparation"]
                        assert packing["private_bf16_input"]["padding_workspace_bytes"] <= packing["padding_workspace_bytes"]
            images += 1
            words_checked += len(bits)
        assert arrays[0] == arrays[1] and counters[0] == counters[1]
        before, after = [comparison["results"][role]["cycles"] for role in ["baseline", "compact"]]
        assert comparison["results"]["compact"]["cycles_saved_percent"] == 100 * (1 - after / before)
        print(label, before, "->", after, "complete compiled stage cycles")

    for label in manifest["boundaries"]:
        prefix = f"boundaries/{label}/"
        report = load(prefix + "receipt.json")
        assert report["profile_sha256"] == profile_sha
        assert report["extension_sha256"] == comparison["identity"]["extension_sha256"]
        for name, digest in report["files_sha256"].items():
            assert sha(data[prefix + name]) == digest
        payload = data[prefix + "payload.bin"]
        with np.load(io.BytesIO(data[prefix + "capture_inputs.npz"]), allow_pickle=False) as inputs:
            assert payload == b"".join(inputs[f"in{i}"].astype("<f4").tobytes() for i in range(3))
        count = int(np.prod(report["shape"]))
        arrays = []
        for dtype in ["f32", "bf16"]:
            base = prefix + dtype + "/"
            image = data[base + "section.elf"]
            reader.embedded_payload(image, payload)
            result = report["results"][dtype]
            assert result["elf_admission"]["elf_sha256"] == sha(image)
            assert result["elf_admission"]["status"] == "accepted"
            lines = data[base + "spike.log"].decode().splitlines()
            words = []
            for line in lines:
                p = line.split()
                if p and p[0] == "WORDS_CHUNK":
                    assert int(p[1]) == len(words)
                    words.extend(int(v, 16) for v in p[2:])
            assert len(words) == count
            raw = np.asarray(words, dtype="<u2").tobytes()
            assert raw == data[base + "observed_words.bin"] and sha(raw) == result["word_sha256"]
            statuses = [dict((k, int(v)) for k, v in (part.split("=", 1) for part in s.split()[1:]))
                        for s in lines if s.startswith("WORDS_STATUS ")]
            assert statuses == [{"guard": 0, "low_bits": 0}] and result["status"] == statuses[0]
            if report["expect_refusal"]:
                assert result["exit_code"] == 77 and len(result["traps"]) == 1 and "DONE" not in lines
                assert result["traps"][0]["cause"] == 3
                reader.check_breakpoint(image, result["traps"][0]["pc"])
            else:
                assert result["exit_code"] == 0 and not result["traps"] and lines.count("DONE") == 1
            arrays.append(raw)
            words_checked += count
            images += 1
        assert arrays[0] == arrays[1]
        print(label, count, "complete normalized words identical; refusal", report["expect_refusal"])
    print(f"PASS: {images} actual images; {words_checked:,} complete stage/boundary words checked.")
    print("Small compiled stages and boundary qualification; no whole-model transport result yet.")


if __name__ == "__main__":
    main()
