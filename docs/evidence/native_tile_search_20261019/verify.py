"""Reopen paired native ELFs, independent integer references and exact traffic."""

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import numpy as np


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def work(shape, tile):
    m, n, k = shape
    tm, nc, kt = tile
    assert m % tm == n % nc == k % kt == 0
    loops = m // tm * (n // nc) * (k // kt)
    data = loops * (tm + nc) * kt
    return (data + data // 32, 5 + 8 * loops, (tm + nc) * (kt + kt // 32), tm * nc * 2), data


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "artifacts.json").read_text())
    archive = root / manifest["archive"]["name"]
    assert archive.stat().st_size == manifest["archive"]["bytes"]
    assert sha(archive.read_bytes()) == manifest["archive"]["sha256"]
    reader_path = root.parent / "broadcast_multiply_20261019/verify.py"
    assert sha(reader_path.read_bytes()) == manifest["elf_reader_sha256"]
    spec = importlib.util.spec_from_file_location("retained_elf_reader", reader_path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert all(member.isfile() for member in members)
        assert len({member.name for member in members}) == len(members)
        assert {member.name for member in members} == set(manifest["members"])
        data = {member.name: tar.extractfile(member).read() for member in members}
    for name, raw in data.items():
        assert {"bytes": len(raw), "sha256": sha(raw)} == manifest["members"][name]
    observation = manifest["full_build_observation"]
    observed_raw = (root / observation["name"]).read_bytes()
    assert len(observed_raw) == observation["bytes"] and sha(observed_raw) == observation["sha256"]

    def read(name):
        return json.loads(data[name])

    profile_sha = sha((json.dumps(read("profile.json"), sort_keys=True, separators=(",", ":")) + "\n").encode())
    word_pairs = independent_words = 0
    hashes = {}
    for label, selection in sorted(manifest["cases"].items()):
        prefix = "cases/" + label + "/"
        report, invocation = read(prefix + "receipt.json"), read(prefix + "invocation.json")
        assert report["exit_code"] == 0 and report["profile_sha256"] == profile_sha
        assert report["extension_sha256"] == "5058ebfd1b1457d5897a87ad2208f67cb98e72c27ba111a771c981e5f6b5cdc5"
        for name, digest in report["files_sha256"].items():
            assert sha(data[prefix + name]) == digest
        arrays = {name: np.load(io.BytesIO(data[prefix + name + ".npy"]), allow_pickle=False)
                  for name in ("a", "asc", "b", "bsc", "golden")}
        m, n, k = report["shape_mnk"]
        assert arrays["a"].shape == (m, k) and arrays["b"].shape == (k, n)
        assert arrays["asc"].shape == (k // 32, m) and arrays["bsc"].shape == (k // 32, n)
        assert arrays["golden"].shape == (m, n) and arrays["golden"].dtype == np.dtype("<u2")
        payload = bytearray()
        for name, array in arrays.items():
            payload.extend(bytes((-len(payload)) % 64))
            assert len(payload) == report["payload_offsets"][name]
            payload.extend(array.tobytes())
        assert bytes(payload) == data[prefix + "payload.bin"]
        image = data[prefix + "section.elf"]
        reader.embedded_payload(image, bytes(payload))
        admission = invocation["admission"]
        assert admission["status"] == "accepted" and admission["elf_sha256"] == sha(image)
        witnesses = [name for name in data if name.startswith(prefix + "invocations/") and name.endswith("/policy/admission.json")]
        assert len(witnesses) == 1 and sha(data[witnesses[0]]) == admission["report_sha256"]
        assert read(witnesses[0])["status"] == "accepted"
        environment = invocation["environment"]
        assert environment["GEMMINI_MODE"] == "both" and environment["GEMMINI_PERF_CONFIG"] == "e4m3_vpu"
        assert environment["GEMMINI_PERF_SET"] == "mem.host_tracking=0"
        baseline, candidate = (tuple(report["schedules"][name]["tile_mnk"]) for name in ("baseline", "candidate"))
        old, old_data = work((m, n, k), baseline)
        new, new_data = work((m, n, k), candidate)
        policy = selection["policy"]
        assert tuple(policy["baseline"]) == baseline and tuple(policy["baseline_work"]) == old
        assert baseline[2] == candidate[2]  # Original accumulation partition.
        accepted = selection["accepted_by_current_policy"]
        assert accepted == all(a <= b for a, b in zip(new, old))
        assert tuple(policy["selected"]) == (candidate if accepted else baseline)
        assert tuple(policy["selected_work"]) == (new if accepted else old)
        counts = reader.counters(data[prefix + "perf.txt"].decode())
        assert counts["commands"] == 3 * (old[1] + new[1])
        assert counts["load_bytes"] == 3 * (old_data + new_data)
        assert counts["scale_bytes"] == 3 * (old_data + new_data) // 32
        performance = data[prefix + "perf.txt"].decode()
        reads = int(performance.split("store: ", 1)[1].split()[0])
        assert reads == 6 * m * n // 32
        # The refused 80-column partitions create extra partial writes despite
        # identical logical output bytes. Do not invent a Puts prediction there.
        assert counts["store_puts"] >= reads
        if accepted:
            assert counts["store_puts"] == reads
        assert counts["fences"] == 20  # Observed combined harness scope, unchanged across pairs.
        lines = data[prefix + "spike.log"].decode().splitlines()
        checks = [dict(part.split("=", 1) for part in line.split()[1:]) for line in lines if line.startswith("TILE ")]
        assert checks == report["checks"] and lines.count("DONE") == 1
        assert [int(row["run"]) for row in checks] == [0, 1, 2]
        assert all(int(row[key]) == 0 for row in checks for key in ("mismatch", "oracle", "guard"))
        assert len({row["hash"] for row in checks}) == 1
        key = (m, n, k, report["pattern"])
        hashes.setdefault(key, checks[0]["hash"])
        assert checks[0]["hash"] == hashes[key]
        if report["pattern"] == "integer":
            assert set(np.unique(arrays["a"])) <= {0x38, 0xb8}
            assert set(np.unique(arrays["b"])) <= {0x38, 0xb8}
            assert np.all(arrays["asc"] == 127) and np.all(arrays["bsc"] == 127)
            a = np.where(arrays["a"] == 0x38, 1, -1).astype(np.int32)
            b = np.where(arrays["b"] == 0x38, 1, -1).astype(np.int32)
            bits = (a @ b).astype("<f4").view("<u4")
            golden = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype("<u2")
            assert np.array_equal(golden, arrays["golden"])
            digest = 2166136261
            for word in golden.reshape(-1):
                digest = ((digest ^ int(word)) * 16777619) & 0xffffffff
            assert int(checks[0]["hash"]) == digest
            independent_words += m * n * 6
        else:
            assert report["pattern"] == "fp8" and np.all(arrays["golden"] == 0)
        word_pairs += m * n * 3
        print(f"{label}: {'selected' if accepted else 'rejected'}; {m * n} complete output pairs on each of three runs")
    identity = read("default_source_identity.json")
    assert len(identity["unique_issuers"]) == 7
    for name, row in identity["unique_issuers"].items():
        assert row["default_c_identical"]
        assert sha(data["default_sources/" + name + ".c"]) == row["c_sha256"]
        assert sha(data["default_sources/" + name + ".mk"]) == row["mk_sha256"]
    assert len(manifest["cases"]) == 12
    print(f"PASS: 12 actual ELF payloads, {word_pairs} complete word pairs, {independent_words} independent integer-word checks")
    print("Scope: encoded issuer sections; whole-model numerical/performance qualification remains separate")


if __name__ == "__main__":
    main()
