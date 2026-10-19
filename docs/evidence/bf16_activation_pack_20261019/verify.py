"""Reconstruct packed bytes and reopen compact-input section ELF payloads."""

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def bf16(values):
    bits = values.astype("<f4").view("<u4")
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view("<f4")


def main():
    manifest = json.loads((HERE / "artifacts.json").read_text())
    readers = {}
    for name, digest in manifest["readers"].items():
        path = HERE.parent / name
        assert sha(path.read_bytes()) == digest
        spec = importlib.util.spec_from_file_location(name.split("/")[0], path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        readers[name.split("/")[0]] = module
    packing = readers["group_word_packing_20261019"]
    elf_reader = readers["broadcast_multiply_20261019"]
    raw = (HERE / "executions.tar.xz").read_bytes()
    assert sha(raw) == manifest["archive_sha256"] and len(raw) == manifest["archive_bytes"]
    data = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            assert member.isfile() and not name.is_absolute() and ".." not in name.parts
            assert member.name not in data
            data[member.name] = archive.extractfile(member).read()
    assert set(data) == set(manifest["members"])
    for name, content in data.items():
        assert {"sha256": sha(content), "bytes": len(content)} == manifest["members"][name]
    profile = json.loads(data["profile.json"])
    profile_sha = sha((json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n").encode())
    checked_bytes, checked_producer_words = 0, 0
    for label in manifest["cases"]:
        prefix = "cases/" + label + "/"
        report = json.loads(data[prefix + "receipt.json"])
        assert report["exit_code"] == 0
        assert report["profile_sha256"] == profile_sha
        assert report["extension_sha256"] == "5058ebfd1b1457d5897a87ad2208f67cb98e72c27ba111a771c981e5f6b5cdc5"
        assert report["defines"] == ["-DMX_QUANT_GROUP_WORDS", "-DMX_QUANT_INLINE_GROUP", "-DMX_QUANT_BF16_INPUT"]
        for name, digest in report["files_sha256"].items():
            assert sha(data[prefix + name]) == digest
        for name, digest in report["source_sha256"].items():
            assert sha(data[prefix + "sources/" + name]) == digest
        words = np.load(io.BytesIO(data[prefix + "input_words.npy"]), allow_pickle=False)
        assert words.dtype == np.dtype("<u2") and words.shape == (report["rows"], report["cols"])
        values = (words.astype("<u4") << 16).view("<f4")
        expected_codes, expected_scales, status = packing.reconstruct(
            report | {"kind": "activation", "live_rows": report["rows"]}, values)
        assert status == report["reference_status"]
        assert data[prefix + "expected_codes.bin"] == expected_codes
        assert data[prefix + "expected_scales.bin"] == expected_scales
        producer_input = b""
        producer = report.get("producer")
        if producer:
            assert report["scope"] == "connected_producer_plus_host_packing_section_no_model_timing"
            capture = producer["capture_receipt"]
            for name in ["source.mlir", "inputs.npz", "bf16_policy_golden.npy"]:
                assert sha(data[prefix + "sources/" + name]) == capture["files_sha256"][name]
            with np.load(io.BytesIO(data[prefix + "sources/inputs.npz"]), allow_pickle=False) as stored:
                arrays = [stored[f"in{i}"] for i in range(3)]
            assert [list(a.shape) for a in arrays] == producer["input_shapes"]
            assert all(a.dtype == np.float32 for a in arrays)
            producer_input = b"".join(a.astype("<f4").tobytes() for a in arrays)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                first = bf16(bf16(arrays[0]) * bf16(arrays[1]))
                computed = bf16(first * bf16(arrays[2]))
            golden = np.load(io.BytesIO(data[prefix + "sources/bf16_policy_golden.npy"]), allow_pickle=False)
            assert np.isfinite(golden).all() and computed.tobytes() == golden.tobytes() == values.tobytes()
            checked_producer_words += words.size * 6
        else:
            assert report["scope"] == "host_packing_section_only_no_producer_or_model_timing"
        payload = producer_input + values.tobytes() + words.tobytes() + expected_codes + expected_scales
        assert payload == data[prefix + "payload.bin"]
        image = data[prefix + "section.elf"]
        elf_reader.embedded_payload(image, payload)
        admission = report["elf_admission"]
        assert admission["status"] == "accepted" and admission["elf_sha256"] == sha(image)
        retained = [n for n in data if n.startswith(prefix + "invocations/") and n.endswith("/policy/admission.json")]
        assert len(retained) == 1 and sha(data[retained[0]]) == admission["report_sha256"]
        assert json.loads(data[retained[0]])["status"] == "accepted"
        lines = data[prefix + "spike.log"].decode().splitlines()
        rows = [dict((key, int(value)) for key, value in (part.split("=", 1) for part in line.split()[1:]))
                for line in lines if line.startswith("PACK ")]
        assert rows == report["checks"] and lines.count("DONE") == 1
        assert [(r["run"], r["variant"]) for r in rows] == [
            (rep, order ^ (rep & 1) ^ int(report["candidate_first"])) for rep in range(3) for order in range(2)]
        fnv = 2166136261
        for byte in expected_codes + expected_scales:
            fnv = ((fnv ^ byte) * 16777619) & 0xFFFFFFFF
        assert all(r["status"] == status and r["guard"] == r["mismatch"] == 0
                   and r["hash"] == fnv and r["cycles"] > 0 for r in rows)
        assert report["environment"]["GEMMINI_MODE"] == "both"
        assert report["environment"]["GEMMINI_PERF_CONFIG"] == "e4m3_vpu"
        assert report["environment"]["GEMMINI_PERF_SET"] == "mem.host_tracking=0"
        checked_bytes += (len(expected_codes) + len(expected_scales)) * 6
        old, new = [[r["cycles"] for r in rows if r["variant"] == v] for v in [0, 1]]
        print(label, old, "->", new, "cycles; complete packed bytes reconstructed")
    observation = json.loads(data["object_observation/comparison.json"])
    for variant in ["default", "words", "bf16"]:
        image = data["object_observation/" + variant + ".o"]
        assert sha(image) == observation[variant]["sha256"]
        assert len(image) == observation[variant]["bytes"]
        assert data["object_observation/" + variant + ".su"].decode() == observation[variant]["stack_usage"]
    assert observation["default"]["sha256"] == "f52f5698f7c9d90b0789329cfe66001c383558c967774aae985adf3235083ae0"
    print(f"PASS: {len(manifest['cases'])} actual ELF payloads, {checked_bytes:,} packed-byte checks, "
          f"{checked_producer_words:,} producer-word checks.")
    print("Sections only. Graph integration and whole-model qualification remain pending.")


if __name__ == "__main__":
    main()
