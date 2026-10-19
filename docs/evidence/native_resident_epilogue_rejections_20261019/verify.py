"""Reopen native epilogue probes and independently enforce both rejections."""

import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def bf16(values):
    words = np.asarray(values, dtype="<f4").view("<u4")
    return ((words + np.uint32(0x7fff) + ((words >> 16) & 1)) >> 16).astype("<u2")


def main():
    manifest = json.loads((HERE / "artifacts.json").read_bytes())
    raw = (HERE / "probe.tar.xz").read_bytes()
    assert len(raw) == manifest["archive_bytes"] and sha(raw) == manifest["archive_sha256"]
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts
            assert member.name not in files
            files[member.name] = archive.extractfile(member).read()
    assert set(files) == set(manifest["members"])
    for name, value in files.items():
        assert {"bytes": len(value), "sha256": sha(value)} == manifest["members"][name]

    def load(name):
        return json.loads(files[name])

    def array(name):
        return np.load(io.BytesIO(files[f"probe/{name}.npy"]), allow_pickle=False)

    result = load("probe/result.json")
    assert result["scope"] == "finite_integer_native_section_only; no full-model, exceptional-value or hardware qualification"
    assert result["shape_mnk"] == [32, 32, 64] and result["native_tile_mnk"] == [32, 32, 32]
    assert result["scalar_bf16"] == 0x3f8d
    assert all(command["exit_code"] == 0 for command in result["commands"])
    assert sum(command["log"] == "spike.log" for command in result["commands"]) == 3
    profile = load("probe/profile.json")
    canonical = (json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n").encode()
    assert sha(canonical) == result["profile_sha256"]
    study = load("source_study/source_manifest.json")
    assert sha(files["source_study/source_manifest.json"]) == result["source_manifest_sha256"]
    assert study["profile_sha256"] == result["profile_sha256"]
    assert study["profile_isa_pin_matches_retrieved_bytes"] is True
    assert study["source_to_existing_extension_binary_equivalence"] == "not_established_by_this_retrieval"
    for name, source in study["sources"].items():
        assert sha(files["source_study/" + name]) == source["sha256"]
        assert len(files["source_study/" + name]) == source["bytes"]
    assert "FileNotFoundError" in files["prior_attempt/runner.log"].decode()
    assert "llvm-readelf" in files["prior_attempt/runner.log"].decode()

    scalar = np.array([result["scalar_bf16"] << 16], dtype="<u4").view("<f4")[0]
    goldens, payload = [], bytearray()
    for index in range(3):
        ac, bc = array(f"a_{index}"), array(f"b_{index}")
        assert ac.shape == (32, 64) and bc.shape == (64, 32)
        assert set(np.unique(ac)) <= {0x38, 0xb8} and set(np.unique(bc)) <= {0x38, 0xb8}
        assert array(f"asc_{index}").shape == array(f"bsc_{index}").shape == (2, 32)
        assert np.all(array(f"asc_{index}") == 127) and np.all(array(f"bsc_{index}") == 127)
        a, b = np.where(ac == 0x38, 1, -1), np.where(bc == 0x38, 1, -1)
        matrix = bf16((a.astype(np.int32) @ b.astype(np.int32)).astype("<f4"))
        expected = bf16((matrix.astype("<u4") << 16).view("<f4") * scalar)
        assert np.array_equal(expected, array(f"golden_{index}"))
        goldens.append(expected)
        for name in ["a", "asc", "b", "bsc", "golden"]:
            payload.extend(bytes((-len(payload)) % 64))
            payload.extend(array(f"{name}_{index}").tobytes())
    assert bytes(payload) == files["probe/payload.bin"]
    assert len({sha(array(f"a_{index}").tobytes()) for index in range(3)}) == 3
    values, cycles, counters = {}, {}, {}
    for role in ["baseline", "resident", "manual_resident"]:
        prefix = f"probe/{role}/"
        row = result["reports"][role]
        invocation = load(prefix + "invocation.json")
        assert sum(c["argv"] == invocation["command"] for c in result["commands"]) == 1
        assert invocation["command"][-1].endswith(f"/{role}/section.elf")
        assert invocation["command"][1] == "--isa=rv64gc_zicntr"
        assert invocation["command"][2].endswith(
            "/docs/evidence/nicolas_e4m3_lut_requant_public_74bceeb_266c593/physical/libgemmini.so")
        assert invocation["command"][3] == "--extension=gemmini"
        admission = invocation["admission"]
        elf = files[prefix + "section.elf"]
        assert len(elf) == row["elf_bytes"] and sha(elf) == row["elf_sha256"] == admission["elf_sha256"]
        assert admission["status"] == "accepted"
        admission_name = prefix + "invocations/" + admission["report_path"].split("/invocations/", 1)[1]
        assert sha(files[admission_name]) == admission["report_sha256"]
        assert load(admission_name) == {k: v for k, v in admission.items() if k not in {"report_path", "report_sha256"}}
        assert invocation["environment"]["GEMMINI_MODE"] == "both"
        assert invocation["environment"]["GEMMINI_PERF_CONFIG"] == "e4m3_vpu"
        assert invocation["environment"]["GEMMINI_PERF_SET"] == "mem.host_tracking=0"
        console = files[prefix + "spike.log"].decode()
        assert console.endswith("\n") and console.splitlines().count("DONE errors=0") == 1
        rows, words = [], []
        for line in console.splitlines():
            if line.startswith("PROBE "):
                rows.append(dict(item.split("=", 1) for item in line.split()[1:]))
            elif line.startswith("WORDS "):
                label, index, encoded = line.split()
                assert index == f"run={len(words)}"
                words.append(bytes.fromhex(encoded))
        assert rows == row["rows"] and [r["run"] for r in rows] == ["0", "1", "2"]
        assert all(int(r["mismatch"]) == int(r["guard"]) == 0 for r in rows)
        assert len(words) == 3 and [sha(w) for w in words] == row["full_words_sha256"]
        for index, observed in enumerate(words):
            assert observed == goldens[index].astype(">u2").tobytes() and len(observed) == 2048
        values[role] = words
        cycles[role] = sum(int(r["cycles"]) for r in rows)
        first, mesh, traffic = [line.split() for line in files[prefix + "perf.txt"].decode().splitlines()[:3]]
        assert first[:3] == ["gemmini", "perf", "[e4m3_vpu]:"]
        actual = {
            "commands": int(first[3]), "loops": int(first[5]), "fences": int(first[7]),
            "mesh_tiles": int(mesh[1]), "mesh_busy": int(mesh[3]), "vpu_commands": int(mesh[7]),
            "load_bytes": int(traffic[1]), "scale_bytes": int(traffic[traffic.index("scales:") + 1]),
        }
        assert actual == row["counters"]
        complete = all(actual[k] == v for k, v in {
            "loops": 9 if role == "resident" else 6, "mesh_tiles": 48,
            "mesh_busy": 768, "vpu_commands": 3, "scale_bytes": 384}.items())
        assert row["timing_work_complete"] is complete
        expected_status = ("section_counters_complete_not_full_model_qualification" if complete else
                           "incomplete_work_counters_refuse_cycle_comparison")
        assert row["performance_qualification"] == expected_status
        counters[role] = actual
    assert values["baseline"] == values["resident"] == values["manual_resident"]
    assert result["reports"]["baseline"]["timing_work_complete"] is True
    assert result["reports"]["manual_resident"]["timing_work_complete"] is True
    assert result["reports"]["resident"]["timing_work_complete"] is False
    assert counters["resident"]["mesh_tiles"] == 16 and counters["resident"]["vpu_commands"] == 1
    assert cycles["manual_resident"] > cycles["baseline"]
    assert counters["manual_resident"]["load_bytes"] < counters["baseline"]["load_bytes"]
    assert counters["manual_resident"]["commands"] > counters["baseline"]["commands"]
    assert counters["manual_resident"]["fences"] > counters["baseline"]["fences"]
    # Bind the exact store-only mutation and finite scalar path being rejected.
    source = files["probe/resident/native.c"].decode()
    assert source.count("MX_CMD(11, 0, 0);") == 1
    assert "UINT64_C(0x0000000000020002)" in files["probe/resident/epilogue.c"].decode()
    assert "UINT64_C(0x0000000000000278)" in files["probe/resident/epilogue.c"].decode()
    print("PASS: three actual admitted images, three changed input pairs, all 3072 BF16 words per image and guards match.")
    print("Native store-only candidate REFUSED: incomplete timing work counters; no performance ratio accepted.")
    print(f"Manual resident candidate REJECTED: {cycles['baseline']} -> {cycles['manual_resident']} section cycles; "
          f"{100 * (cycles['manual_resident'] / cycles['baseline'] - 1):.4f}% slower despite fewer load bytes.")


if __name__ == "__main__":
    main()
