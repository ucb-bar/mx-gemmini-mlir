"""Reopen native broadcast pairs, embedded references, traps and resource sizes."""

import hashlib
import io
import json
import re
import struct
import tarfile
from pathlib import Path

import numpy as np


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def sections(image):
    assert image[:6] == b"\x7fELF\x02\x01" and struct.unpack_from("<H", image, 18)[0] == 243
    offset = struct.unpack_from("<Q", image, 40)[0]
    size, count = struct.unpack_from("<HH", image, 58)
    assert size == 64 and count > 0 and offset + size * count <= len(image)
    return [struct.unpack_from("<IIQQQQIIQQ", image, offset + i * size) for i in range(count)]


def embedded_payload(image, expected):
    rows, found = sections(image), []
    for row in rows:
        if row[1] != 2:  # SHT_SYMTAB
            continue
        strings = rows[row[6]]
        names = image[strings[4]:strings[4] + strings[5]]
        assert row[9] == 24
        for offset in range(row[4], row[4] + row[5], row[9]):
            name, _, _, section, address, _ = struct.unpack_from("<IBBHQQ", image, offset)
            if names[name:names.index(b"\0", name)] != b"payload":
                continue
            target = rows[section]
            position = target[4] + address - target[3]
            assert target[1] != 8 and 0 <= address - target[3] <= target[5] - len(expected)
            found.append(image[position:position + len(expected)])
    assert found == [expected]


def check_breakpoint(image, pc):
    locations = [row[4] + pc - row[3] for row in sections(image)
                 if row[2] & 4 and row[3] <= pc < row[3] + row[5]]
    assert len(locations) == 1
    offset = locations[0]
    assert image[offset:offset + 2] == b"\x02\x90" or image[offset:offset + 4] == b"\x73\x00\x10\x00"


def bf16(value):
    words = np.asarray(value, dtype="<f4").view("<u4")
    return ((words + np.uint32(0x7fff) + ((words >> 16) & 1)) & np.uint32(0xffff0000)).view("<f4")


def counters(native):
    patterns = {"commands": r"\[e4m3_vpu\]: (\d+) commands", "fences": r"\b(\d+) fences;",
                "load_bytes": r"load: (\d+) bytes", "store_puts": r"\b(\d+) Puts;",
                "scale_bytes": r"scales: (\d+) bytes"}
    result = {}
    for name, pattern in patterns.items():
        matches = re.findall(pattern, native)
        assert len(matches) == 1
        result[name] = int(matches[0])
    return result


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "artifacts.json").read_text())
    archive = root / manifest["archive"]["name"]
    assert archive.stat().st_size == manifest["archive"]["bytes"]
    assert sha(archive.read_bytes()) == manifest["archive"]["sha256"]
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert all(member.isfile() for member in members) and len({m.name for m in members}) == len(members)
        assert {m.name for m in members} == set(manifest["members"])
        data = {m.name: tar.extractfile(m).read() for m in members}
    for name, raw in data.items():
        assert sha(raw) == manifest["members"][name]["sha256"]
        assert len(raw) == manifest["members"][name]["bytes"]
    resource_file = root / manifest["resources"]["name"]
    resource_raw = resource_file.read_bytes()
    assert sha(resource_raw) == manifest["resources"]["sha256"]
    assert len(resource_raw) == manifest["resources"]["bytes"]
    resources = json.loads(resource_raw)
    def read(name):
        return json.loads(data[name])

    profile_sha = sha((json.dumps(read("profile.json"), sort_keys=True, separators=(",", ":")) + "\n").encode())
    semantic = read("semantic_check_current.json")
    for label, case in sorted(manifest["cases"].items()):
        capture = "captures/" + case["fixture"] + "/"
        prefix = "cases/" + label + "/"
        report = read(prefix + "receipt.json")
        original = read(capture + "receipt.json")
        assert report["capture_receipt"] == original
        assert report["profile_sha256"] == profile_sha
        for name, digest in original["files_sha256"].items():
            assert sha(data[capture + name]) == digest
        with np.load(io.BytesIO(data[capture + "inputs.npz"]), allow_pickle=False) as stored:
            inputs = [stored[f"in{i}"] for i in range(3)]
        golden = np.load(io.BytesIO(data[capture + "bf16_policy_golden.npy"]), allow_pickle=False)
        assert [list(value.shape) for value in inputs] == report["input_shapes"]
        payload = b"".join(value.astype("<f4").tobytes() for value in (*inputs, golden))
        assert payload == data[prefix + "payload.bin"]
        for name, digest in report["source_files_sha256"].items():
            assert sha(data[prefix + name]) == digest
        for path, digest in report["implementation_sha256"].items():
            assert sha(data[prefix + "implementation/" + Path(path).name]) == digest
        assert read(prefix + "invocation.json")["exit_code"] == 0
        assert semantic[label]["source_sha256"] == sha(data[capture + "source.mlir"])
        refusal = report["schema"] == "mx_gemmini.stage_multiply_refusal.v1"
        if refusal:
            assert any(np.isnan(value).any() for value in inputs)
        else:
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                reference = bf16(bf16(bf16(inputs[0]) * bf16(inputs[1])) * bf16(inputs[2]))
            assert np.array_equal(reference.view("<u4"), golden.view("<u4"))
            digest = 2166136261
            for word in golden.view("<u4").reshape(-1):
                digest = ((digest ^ int(word)) * 16777619) & 0xffffffff
        for variant, result in report["results"].items():
            base = prefix + variant + "/"
            for name, digest_file in result["files_sha256"].items():
                assert sha(data[base + name]) == digest_file
            image = data[base + "section.elf"]
            allocated = sections(image)
            assert resources[label][variant] == {
                "elf_bytes": len(image),
                "allocated_nobits_bytes": sum(row[5] for row in allocated if row[2] & 2 and row[1] == 8),
                "allocated_file_bytes": sum(row[5] for row in allocated if row[2] & 2 and row[1] != 8)}
            embedded_payload(image, payload)
            native = data[base + "spike.log"].decode()
            assert counters(data[base + "perf.txt"].decode()) == result["counters"]
            assert sha(data[base + "issue.c"]) == semantic[label][variant + "_c_sha256"]
            if refusal:
                assert result["exit_code"] == 77 and result["refusal_trap"]["cause"] == "3"
                assert native.splitlines().count("REFUSAL_BEGIN") == 1 and "REFUSAL_RETURNED" not in native
                assert "REFUSAL_TRAP cause=3 pc=" + result["refusal_trap"]["pc"] in native
                check_breakpoint(image, int(result["refusal_trap"]["pc"]))
            else:
                checks = [dict(word.split("=", 1) for word in line.split()[1:]) for line in native.splitlines()
                          if line.startswith("SECTION ")]
                assert checks == result["checks"] and len(checks) == 3
                assert [int(row["run"]) for row in checks] == [0, 1, 2]
                assert all(int(row["mismatches"]) == 0 and int(row["hash"]) == digest for row in checks)
        if refusal:
            print(label + ": original NaN refusal; both actual trap instructions reopened")
            continue
        before, after = (report["results"][name] for name in ("materialized", "connected"))
        saving = min(int(row["cycles"]) for row in before["checks"][1:]) - max(int(row["cycles"]) for row in after["checks"][1:])
        assert saving == report["warm_saved_cycles"]
        expected = (-report["fallback_prefix_load_bytes_per_call"] * 3 if report["expected_fallback"] else
                    report["removed_transfer_bytes_per_call"] // 2 * 3)
        assert before["counters"]["load_bytes"] - after["counters"]["load_bytes"] == expected
        print(f"{label}: {saving} warm cycles saved; all {golden.size} words checked on all three invocations")
    print("PASS: compiled sections only; whole-model performance and numerical qualification remain separate")


if __name__ == "__main__":
    main()
