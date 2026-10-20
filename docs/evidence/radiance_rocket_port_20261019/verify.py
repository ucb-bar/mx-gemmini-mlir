"""Audit the complete retained Rocket port smoke and paired-kernel observations."""

import hashlib
import io
import json
import re
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np

HERE = Path(__file__).resolve().parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def replay_archive():
    manifest = json.loads((HERE / "replay_artifacts.json").read_text())
    raw = (HERE / "replays.tar.xz").read_bytes()
    assert (
        digest(raw) == manifest["archive_sha256"]
        and len(raw) == manifest["archive_bytes"]
    )
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts
            assert member.name not in files
            files[member.name] = archive.extractfile(member).read()
    assert set(files) == set(manifest["members"])
    for name, body in files.items():
        assert digest(body) == manifest["members"][name]["sha256"]
        assert len(body) == manifest["members"][name]["bytes"]
    return manifest, files


def check_work(files, prefix):
    program = np.frombuffer(files[prefix + "phases.bin"], dtype="<u4").reshape(-1, 32)
    tiles = loops = matrix_bytes = exp_elements = exp_commands = 0
    for words in program:
        d = [int(value) for value in words]
        if d[0] == 1:
            rows, columns, reduction = d[15] * d[18], d[16] * 64, d[17] * d[19]
            tiles += d[14] * rows * columns * reduction // 4096
            loops += d[14] * d[15] * d[16] * d[17]
            matrix_bytes += d[14] * d[15] * d[16] * reduction * (d[18] + 64)
        elif d[0] == 4:
            exp_elements += d[6] * d[5] * d[4]
            exp_commands += d[6] * d[5] * ((d[4] + 8191) // 8192)
        elif d[0] == 7:
            exp_elements += d[6] * d[5]
            exp_commands += d[6] * ((d[5] + 8191) // 8192)
    perf = files[prefix + "perf.txt"].decode()
    assert re.findall(
        r"mesh: ([0-9]+) tiles, ([0-9]+) busy cycles; VPU: ([0-9]+) commands", perf
    ) == [(str(tiles), str(tiles * 16), str(exp_commands))]
    assert re.findall(r"commands, ([0-9]+) loops", perf) == [str(loops)]
    assert re.findall(r"load: ([0-9]+) bytes", perf) == [
        str(matrix_bytes + 2 * exp_elements)
    ]
    assert re.findall(r"scales: ([0-9]+) bytes", perf) == [str(matrix_bytes // 32)]


def main():
    manifest = json.loads((HERE / "artifacts.json").read_text())
    raw = (HERE / "qualifications.tar.xz").read_bytes()
    assert digest(raw) == manifest["archive_sha256"]
    assert len(raw) == manifest["archive_bytes"]
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts
            assert member.name not in files
            files[member.name] = archive.extractfile(member).read()
    assert set(files) == set(manifest["members"])
    for name, body in files.items():
        assert len(body) == manifest["members"][name]["bytes"]
        assert digest(body) == manifest["members"][name]["sha256"]
    summary = {}
    for label in ("smoke64", "smoke128"):
        prefix = label + "/"
        check_work(files, prefix)
        receipt = json.loads(files[prefix + "receipt.json"])
        assert receipt["status"] == "completed_requires_matched_comparison"
        assert receipt["mode"] == "random_smoke"
        assert receipt["source_revision"] == manifest["radiance_revision"]
        assert receipt["full_model_speedup_qualified"] is False
        assert receipt["profile_sha256"] == manifest["profile_sha256"]
        assert receipt["extension_sha256"] == manifest["extension_sha256"]
        assert receipt["admission"]["status"] == "accepted"
        assert digest(files[prefix + "session.elf"]) == receipt["elf_sha256"]
        assert digest(files[prefix + "driver.c"]) == receipt["port_source_sha256"]
        for name, sha in receipt["files_sha256"].items():
            assert digest(files[prefix + name]) == sha
        for region, row in receipt["regions"].items():
            assert (
                digest(files[prefix + f"data/llama.region{region}.bin"])
                == row["blob_sha256"]
            )
        for name, sha in receipt["source_files_sha256"].items():
            assert digest(files["radiance-source/" + name]) == sha
        console = files[prefix + "spike.log"].decode()
        assert (
            "MODEL DEADLOCK" not in console
            and console.splitlines().count("PORT_DONE") == 1
        )
        assert re.findall(r"^PORT_PHASES_COMPLETED ([0-9]+)$", console, re.M) == ["26"]
        assert re.findall(r"^PORT_TOTAL ([0-9]+) guards=0$", console, re.M) == [
            str(receipt["cycles"])
        ]
        checks = json.loads(files[prefix + "checks.json"])
        header_checks = re.findall(
            r'\{"([^"]+)", (0x[0-9a-f]+)u, (0x[0-9a-f]+)u, ([0-9]+)u, [0-9]+, ([01]), [0-9]+u\}',
            files[prefix + "llama_data.h"].decode(),
        )
        assert len(checks) == len(header_checks)
        for check, (name, device, reference, elements, byte_kind) in zip(
            checks, header_checks, strict=True
        ):
            device, reference = int(device, 16), int(reference, 16)
            size = int(elements) * (1 if byte_kind == "1" else 2)
            assert (
                check["name"] == name
                and check["address"] == device
                and check["bytes"] == size
            )
            region = next(
                r
                for r, base in ((1, 0x20000000), (2, 0x40000000), (3, 0x60000000))
                if base <= reference < base + receipt["regions"][str(r)]["bytes"]
            )
            base = receipt["regions"][str(region)]["virtual_base"]
            blob = files[prefix + f"data/llama.region{region}.bin"]
            assert (
                bytes.fromhex(check["golden"])
                == blob[reference - base : reference - base + size]
            )
            assert device + size <= reference or reference + size <= device
        observations = {i: bytearray() for i in range(len(checks))}
        for line in console.splitlines():
            if not line.startswith("PORT_BYTES "):
                continue
            _, i, offset, count, *words = line.split()
            i, offset, count = int(i), int(offset), int(count)
            assert i in observations and offset == len(observations[i])
            assert 1 <= count <= 32 and count == len(words)
            observations[i].extend(int(word, 16) for word in words)
        exact_bytes = 0
        for i, check in enumerate(checks):
            actual = bytes(observations[i])
            assert len(actual) == check["bytes"]
            assert actual == files[prefix + f"observed{i}.bin"]
            assert digest(actual) == receipt["results"][i]["observed_sha256"]
            golden = bytes.fromhex(check["golden"])
            mismatches = sum(a != b for a, b in zip(actual, golden, strict=True))
            assert mismatches == receipt["results"][i]["byte_mismatches"]
            if i < 11:
                assert actual == golden
                exact_bytes += len(actual)
            if check["kind"] == "bf16":
                values = (np.frombuffer(actual, "<u2").astype(np.uint32) << 16).view(
                    "<f4"
                )
                assert np.isfinite(values).all()
        phases = re.findall(r"^PORT_PHASE ([0-9]+) ([0-9]+) ([0-9]+)$", console, re.M)
        assert [int(row[0]) for row in phases] == list(range(26))
        assert all(int(row[2]) > 0 for row in phases)
        summary[label] = {
            "source_exact_pre_exp_bytes": exact_bytes,
            "full_observed_tensors": len(checks),
            "guards": "intact",
        }
    for label in manifest["paired_native_controls"]:
        prefix = "native-pairs/" + label + "/"
        receipt = json.loads(files[prefix + "receipt.json"])
        assert receipt["status"] == "complete_full_outputs_matched"
        assert receipt["not_a_radiance_muon_runtime_or_full_model_comparison"] is True
        for name, sha in receipt["files_sha256"].items():
            assert digest(files[prefix + name]) == sha
        payload = files[prefix + "payload.bin"]
        elements = receipt["shape_mnk"][0] * receipt["shape_mnk"][1]
        offset = receipt["offsets"]["golden_bf16"]
        expected = payload[offset : offset + elements * 2]
        assert len(expected) == elements * 2
        for run in (0, 1):
            assert receipt["runs"][run]["admission"]["status"] == "accepted"
            assert (
                digest(files[prefix + f"pair{run}.elf"])
                == receipt["runs"][run]["elf_sha256"]
            )
            console = files[prefix + f"spike{run}.log"].decode()
            assert (
                "MODEL DEADLOCK" not in console
                and console.splitlines().count("PAIR_DONE") == 1
            )
            words = {}
            for role, trial, offset, count, body in re.findall(
                r"^PAIR_WORDS (native|compiler) ([0-9]+) ([0-9]+) ([0-9]+) (.+)$",
                console,
                re.M,
            ):
                key = role, int(trial)
                target = words.setdefault(key, [])
                values = [int(word, 16) for word in body.split()]
                assert int(offset) == len(target) and len(values) == int(count)
                target.extend(values)
            assert set(words) == {
                (role, trial) for role in ("native", "compiler") for trial in range(3)
            }
            for values in words.values():
                assert np.asarray(values, "<u2").tobytes() == expected
            times = re.findall(
                r"^PAIR trial=([0-9]+) native=([0-9]+) compiler=([0-9]+) mismatch=0 oracle=0 guard=0$",
                console,
                re.M,
            )
            assert [int(t[0]) for t in times] == [0, 1, 2]
            assert all(int(t[1]) > 0 and int(t[2]) > 0 for t in times)
            m, n, k = receipt["shape_mnk"]
            pe = 32 if receipt["precision"] == "FP4" else 16
            tiles = 6 * (m // pe) * (n // pe) * (k // 16)
            perf = files[prefix + f"perf{run}.txt"].decode()
            assert re.findall(
                r"mesh: ([0-9]+) tiles, ([0-9]+) busy cycles; VPU: ([0-9]+) commands",
                perf,
            ) == [(str(tiles), str(tiles * 16), "0")]
            for actual, retained in zip(
                times, receipt["runs"][run]["checks"], strict=True
            ):
                assert [int(v) for v in actual] == [
                    retained[k] for k in ("trial", "native", "compiler")
                ]
    replay_manifest, replays = replay_archive()
    replay_summary = []
    for entry in replay_manifest["replays"]:
        rp, bp = entry["replay_prefix"], entry["build_prefix"]
        inputs = files if entry["build_archive"] == "qualifications" else replays
        replay = json.loads(replays[rp + "receipt.json"])
        assert replay["passed"] and replay["phase_count"] == 26
        assert replay["matrix_reference_calls"] == 17
        assert replay["source_receipt_sha256"] == digest(inputs[bp + "receipt.json"])
        assert replay["producer_sha256"] == digest(replays[rp + "producer.py"])
        assert (
            replay["vpu_reference_header_sha256"]
            == digest(replays[rp + "vpu_ref.h"])
            == "459726e52e7d19c51bada75b2e3ded16f64371799d5bc443750eb22e8616b17b"
        )
        checks = json.loads(inputs[bp + "checks.json"])
        assert len(checks) == len(replay["results"]) == 32
        checked = 0
        for i, row in enumerate(replay["results"]):
            expected, actual = (
                replays[rp + f"expected{i}.bin"],
                inputs[bp + f"observed{i}.bin"],
            )
            assert expected == actual and digest(expected) == row["expected_sha256"]
            assert (
                row["byte_mismatches"] == 0
                and len(expected) == row["bytes"] == checks[i]["bytes"]
            )
            checked += len(expected)
        check_work(inputs, bp)
        replay_summary.append(
            {
                "case": bp,
                "tensor_checks": 32,
                "checked_bytes": checked,
                "all_bits_equal": True,
            }
        )
    print(
        json.dumps(
            {
                "passed": True,
                "smoke": summary,
                "paired_native_controls": len(manifest["paired_native_controls"]),
                "reference_replays": replay_summary,
                "full_model_comparison": "not supplied by this qualification archive",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
