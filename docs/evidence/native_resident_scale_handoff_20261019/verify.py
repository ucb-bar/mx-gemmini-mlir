"""Independently reopen complete probes, rejected timing and checkpoint audit."""

import argparse
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np

HERE = Path(__file__).resolve().parent
MARKER = b"gemmini perf: MODEL DEADLOCK"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def bf16(values):
    words = np.asarray(values, dtype="<f4").view("<u4")
    return ((words + np.uint32(0x7fff) + ((words >> 16) & 1)) >> 16).astype("<u2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, help="also reopen every audited actual log/receipt")
    args = parser.parse_args()
    manifest = json.loads((HERE / "artifacts.json").read_bytes())
    raw = (HERE / "probes.tar.xz").read_bytes()
    assert len(raw) == manifest["archive_bytes"] and sha(raw) == manifest["archive_sha256"]
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            assert member.isfile() and not name.is_absolute() and ".." not in name.parts
            assert member.name not in files
            files[member.name] = archive.extractfile(member).read()
    assert set(files) == set(manifest["members"])
    for name, value in files.items():
        assert {"bytes": len(value), "sha256": sha(value)} == manifest["members"][name]

    def load(name):
        return json.loads(files[name])

    metadata = load("metadata.json")
    assert metadata["target_revision"].startswith("92c2b93")
    assert metadata["extension_sha256"] == "5058ebfd1b1457d5897a87ad2208f67cb98e72c27ba111a771c981e5f6b5cdc5"
    for key in ["source_manifest.json", "timing_source_manifest.json"]:
        study = load("source_study/" + key)
        assert study["source_to_existing_extension_binary_equivalence"] == "not_established_by_this_retrieval"
        assert study["libgemmini_gitlink_revision"] == "876631072ecc7cc8197df5dc69c4d26bd4cd5f54"
        for name, record in study["sources"].items():
            content = files["source_study/" + name]
            assert sha(content) == record["sha256"] and len(content) == record["bytes"]
            assert record["firecrawl_text_byte_identical"] is True
    assert b"state_[x] = FREE" in files["source_study/perf_scale_loader.cc"]
    assert b"gemmini perf: MODEL DEADLOCK" in files["source_study/perf_model.cc"]
    helper = {}
    exec(compile(files["frozen_target/spike_timing.py"], "spike_timing.py", "exec"), helper)
    summaries = {}
    for label, original_directory in metadata["cases"].items():
        prefix = "cases/" + label + "/"
        result = load(prefix + "result.json")
        assert set(result["reports"]) == {"baseline", "resident"}
        assert result["extension_sha256"] == metadata["extension_sha256"]
        assert all(command["exit_code"] == 0 for command in result["commands"])
        m, n, k = result["shape_mnk"]
        assert result["native_tile_mnk"] == [m, n, 32]
        assert min(m, n, k) > 0 and m % 16 == n % 16 == k % 32 == 0
        profile = load(prefix + "profile.json")
        canonical = (json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n").encode()
        assert sha(canonical) == result["profile_sha256"] == study["profile_sha256"]
        assert result["source_manifest_sha256"] == sha(files["source_study/source_manifest.json"])
        scalar = np.array([result["scalar_bf16"] << 16], dtype="<u4").view("<f4")[0]
        goldens, payload, offsets = [], bytearray(), []
        for index in range(3):
            arrays = {name: np.load(io.BytesIO(files[prefix + f"{name}_{index}.npy"]), allow_pickle=False)
                      for name in ["a", "asc", "b", "bsc", "golden"]}
            assert arrays["a"].shape == (m, k) and arrays["b"].shape == (k, n)
            assert set(np.unique(arrays["a"])) <= {0x38, 0xb8}
            assert set(np.unique(arrays["b"])) <= {0x38, 0xb8}
            assert arrays["asc"].shape == (k // 32, m) and np.all(arrays["asc"] == 127)
            assert arrays["bsc"].shape == (k // 32, n) and np.all(arrays["bsc"] == 127)
            a, b = np.where(arrays["a"] == 0x38, 1, -1), np.where(arrays["b"] == 0x38, 1, -1)
            matrix = bf16((a.astype(np.int32) @ b.astype(np.int32)).astype("<f4"))
            expected = bf16((matrix.astype("<u4") << 16).view("<f4") * scalar)
            assert np.array_equal(expected, arrays["golden"])
            goldens.append(expected.astype(">u2").tobytes())
            positions = []
            for name in ["a", "asc", "b", "bsc", "golden"]:
                payload.extend(bytes((-len(payload)) % 64))
                positions.append(len(payload))
                payload.extend(arrays[name].tobytes())
            offsets.append((index, *positions))
        assert bytes(payload) == files[prefix + "payload.bin"]
        assert len(set(goldens)) == 3
        counts, cycles = {}, {}
        for role in ["baseline", "resident"]:
            pre = prefix + role + "/"
            row = result["reports"][role]
            invocation = load(pre + "invocation.json")
            assert invocation["command"][1] == "--isa=rv64gc_zicntr"
            assert invocation["command"][3] == "--extension=gemmini"
            assert sum(cmd["argv"] == invocation["command"] for cmd in result["commands"]) == 1
            environment = invocation["environment"]
            assert {key: environment[key] for key in ["GEMMINI_MODE", "GEMMINI_PERF_CONFIG", "GEMMINI_PERF_SET"]} == {
                "GEMMINI_MODE": "both", "GEMMINI_PERF_CONFIG": "e4m3_vpu", "GEMMINI_PERF_SET": "mem.host_tracking=0"}
            elf = files[pre + "section.elf"]
            assert elf.startswith(b"\x7fELF") and len(elf) == row["elf_bytes"]
            assert sha(elf) == row["elf_sha256"] == invocation["admission"]["elf_sha256"]
            assert invocation["admission"]["status"] == "accepted"
            admission_path = invocation["admission"]["report_path"].split("/" + original_directory + "/", 1)[1]
            assert sha(files[prefix + admission_path]) == invocation["admission"]["report_sha256"]
            driver = files[pre + "driver.c"].decode()
            pattern = r"errors \+= invoke\((\d+), payload\+(\d+), payload\+(\d+),\s*payload\+(\d+), payload\+(\d+), \(const uint16_t \*\)\(payload\+(\d+)\)\);"
            assert [tuple(map(int, match)) for match in re.findall(pattern, driver)] == offsets
            log = files[pre + "spike.log"]
            lines = log.decode().splitlines()
            values = [bytes.fromhex(line.split()[-1]) for line in lines if line.startswith("WORDS ")]
            assert values == goldens
            assert [sha(v) for v in values] == row["full_words_sha256"]
            probes = [dict(part.split("=") for part in line.split()[1:]) for line in lines if line.startswith("PROBE ")]
            assert probes == row["rows"] and [int(v["run"]) for v in probes] == [0, 1, 2]
            assert all(v["mismatch"] == v["guard"] == "0" for v in probes)
            assert lines.count("DONE errors=0") == 1
            perf = files[pre + "perf.txt"].decode().splitlines()
            first, mesh, traffic = [line.split() for line in perf[:3]]
            counters = dict(commands=int(first[3]), loops=int(first[5]), fences=int(first[7]),
                            mesh_tiles=int(mesh[1]), mesh_busy=int(mesh[3]), vpu_commands=int(mesh[7]),
                            load_bytes=int(traffic[1]), scale_bytes=int(traffic[traffic.index("scales:") + 1]))
            assert counters == row["counters"]
            rejected = role == "resident" and label in {"nonzero_skip", "zero_compute"}
            assert (MARKER in log) == rejected
            try:
                helper["reject_deadlocked_timing"](log.decode())
            except ValueError:
                assert rejected
            else:
                assert not rejected
            expected_tiles = 3 * (m // 16) * (n // 16) * (k // 16)
            if not rejected:
                assert row["timing_work_complete"] is True
                assert counters["loops"] == 3 * (k // 32 + int(role == "resident"))
                assert counters["mesh_tiles"] == expected_tiles
                assert counters["mesh_busy"] == expected_tiles * 16
                assert counters["vpu_commands"] == 3
                assert counters["scale_bytes"] == 3 * (m + n) * (k // 32)
            else:
                assert row["timing_work_complete"] is False
                assert counters["mesh_tiles"] < expected_tiles and counters["vpu_commands"] == 1
            counts[role] = counters
            cycles[role] = sum(int(v["cycles"]) for v in probes)
        control = files[prefix + "baseline/native.c"].decode()
        candidate = files[prefix + "resident/native.c"].decode()
        c_line = [line for line in control.splitlines() if "MX_CMD(11, 0," in line]
        assert len(c_line) == 1
        assert control.replace(c_line[0], "      MX_CMD(11, 0, 0); /* Preserve native compute; no DRAM C store. */") == candidate
        if label not in {"nonzero_skip", "zero_compute"}:
            assert counts["baseline"]["load_bytes"] - counts["resident"]["load_bytes"] == 3 * m * n * 2
            assert cycles["resident"] < cycles["baseline"]
            summaries[label] = dict(shape=[m, n, k], cycles=cycles,
                                    reduction_percent=100 * (1 - cycles["resident"] / cycles["baseline"]))
    # Harness failures do not establish a target numerical failure or a speedup.
    assert b"KeyError: 'manual_resident'" in files["harness_failures/report_key_error/runner.log"]
    for label in ["fixed_count_wrong_shape", "payload_offset_substitution", "payload_offset_diagnostic"]:
        assert b"baseline/spike failed: 1" in files[f"harness_failures/{label}/runner.log"]
    assert b"58 passed" in files["checks/focused_tests.log"]
    audit = load("audit/completed_log_audit.json")
    assert audit["stage_count"] == 91 and audit["session_count"] == 22
    assert len(audit["rows"]) == 113
    for row in audit["rows"]:
        receipt = files["audit/receipts/" + row["receipt"]]
        assert sha(receipt) == row["receipt_sha256"]
        data = json.loads(receipt)
        if row["kind"] == "complete_session":
            assert data["complete"] is True and data["exit_code"] == 0 and data["timing_model_active"] is True
            assert data["log_sha256"] == row["sha256"]
        else:
            assert data["spike"]["done"] is True and data["spike"]["exit_code"] == 0
        if args.build_root:
            path = PurePosixPath(row["console"])
            assert not path.is_absolute() and ".." not in path.parts
            log = (args.build_root / str(path)).read_bytes()
            assert len(log) == row["bytes"] and sha(log) == row["sha256"] and MARKER not in log
            assert (args.build_root / row["receipt"]).read_bytes() == receipt
    print("PASS: all retained words, payload pointers, guards, admitted images, source pins, timing rejections and complete-work counters")
    print("Completed-log audit:", "actual logs/receipts rechecked" if args.build_root else "retained audit observations/receipts checked; actual logs need --build-root")
    print(json.dumps(summaries, sort_keys=True))


if __name__ == "__main__":
    main()
