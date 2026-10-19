"""Verify complete small sessions and separately admitted large constant pooling."""

import argparse
import hashlib
import io
import json
import mmap
import struct
import tarfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def members(directory, archive_name, pinned=None):
    meta_raw = (directory / "artifacts.json").read_bytes()
    meta = json.loads(meta_raw)
    if pinned:
        assert sha(meta_raw) == pinned["manifest_sha256"]
        assert meta["archive_sha256"] == pinned["archive_sha256"] and meta["archive_bytes"] == pinned["archive_bytes"]
    raw = (directory / archive_name).read_bytes()
    assert sha(raw) == meta["archive_sha256"] and len(raw) == meta["archive_bytes"]
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts and member.name not in files
            files[member.name] = archive.extractfile(member).read()
    stored = frozenset(files)
    for name, original in meta.get("aliases", {}).items():
        assert name not in files and original in stored
        files[name] = files[original]
    assert set(files) == set(meta["members"])
    for name, raw in files.items():
        assert sha(raw) == meta["members"][name]["sha256"] and len(raw) == meta["members"][name]["bytes"]
    return meta, files


def allocated_sections(path):
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as raw:
        assert raw[:6] == b"\x7fELF\x02\x01"
        offset = struct.unpack_from("<Q", raw, 40)[0]
        size, count, names = struct.unpack_from("<HHH", raw, 58)
        assert size == 64
        rows = [struct.unpack_from("<IIQQQQIIQQ", raw, offset + index * size) for index in range(count)]
        strings = rows[names]
        table = raw[strings[4]:strings[4] + strings[5]]
        return {table[row[0]:table.index(b"\0", row[0])].decode():
                {"bytes": row[5], "type": row[1], "flags": row[2]} for row in rows if row[2] & 2 and row[5]}



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path)
    args = parser.parse_args()
    manifest, files = members(HERE, "executions.tar.xz")
    by_source = {}
    for name, raw in files.items():
        source = manifest["members"][name]["source_path"]
        assert source not in by_source or by_source[source] == raw
        by_source[source] = raw

    def document(name):
        return json.loads(files[name])

    expected_steps = [(0, 0), (1, 0), (1, 1), (1, 2)]

    def observe(role):
        log = files[f"{role}/run/spike.log"].decode()
        assert log.endswith("\n") and "\nDONE\n" in log
        records = [line.split() for line in log.splitlines() if line.startswith("SESSION_")]
        expected_tags = ["SESSION_PREP_PERF", "SESSION_PREP_PERF"]
        for index, step in enumerate(expected_steps):
            if index == 1:
                expected_tags.append("SESSION_PREP_PERF")
            expected_tags += ["SESSION_BEGIN", "SESSION_PERF", "SESSION_SHA", "SESSION_HEX", "SESSION_SHA"]
        expected_tags += ["SESSION_TOTAL", "SESSION_EXECUTION_TOTAL"]
        assert [row[0] for row in records] == expected_tags
        begin = [tuple(map(int, row[1:])) for row in records if row[0] == "SESSION_BEGIN"]
        assert begin == expected_steps
        perf = [tuple(map(int, row[1:])) for row in records if row[0] == "SESSION_PERF"]
        assert [row[:2] for row in perf] == expected_steps and all(min(row[2:]) > 0 for row in perf)
        preparation = [tuple(map(int, row[1:])) for row in records if row[0] == "SESSION_PREP_PERF"]
        assert [row[0] for row in preparation] == [-1, 0, 1]
        sums = tuple(sum(row[index] for row in perf) for index in (2, 3))
        total = next(tuple(map(int, row[1:])) for row in records if row[0] == "SESSION_TOTAL")
        execution = next(tuple(map(int, row[1:])) for row in records if row[0] == "SESSION_EXECUTION_TOTAL")
        assert total == sums
        assert execution == tuple(sums[index] + sum(row[index + 1] for row in preparation) for index in (0, 1))
        digests = {}
        for row in records:
            if row[0] == "SESSION_SHA":
                key = tuple(map(int, row[1:4]))
                assert key not in digests and int(row[4]) == 2048 and len(row[5]) == 64
                digests[key] = row[5]
        assert list(digests) == [(program, step, output) for program, step in expected_steps for output in (0, 1)]
        dumps = {}
        for row in records:
            if row[0] == "SESSION_HEX":
                key = tuple(map(int, row[1:4]))
                data = bytes.fromhex(row[5])
                assert key not in dumps and len(data) == int(row[4]) == 2048
                assert sha(data) == digests[key]
                dumps[key] = data
        assert list(dumps) == [(program, step, 0) for program, step in expected_steps]
        run = document(role + "/run/mx_session_run.json")
        assert run["status"] == "complete_numeric_qualification_pending" and run["exit_code"] == 0
        assert run["complete"] is True and sha(log.encode()) == run["log_sha256"]
        assert sha(files[f"{role}/run/gemmini_perf.txt"]) == run["timing_report_sha256"]
        assert run["execution_total_performance"]["cycles"] == execution[0]
        assert run["timing_mode"] == "both" and run["timing_config"] == "e4m3_vpu"
        assert run["timing_overrides"] == "mem.host_tracking=0"
        return digests, dumps, execution, run

    reference, candidate = observe("reference"), observe("candidate")
    assert reference[:2] == candidate[:2]
    assert reference[2][0] == 640228 and candidate[2][0] == 640134
    links = {}
    for role, observed in [("reference", reference), ("candidate", candidate)]:
        link = document(role + "/link/mx_session_link.json")
        links[role] = link
        assert sha(files[role + "/link/session.elf"]) == link["elf_sha256"]
        assert len(files[role + "/link/session.elf"]) == link["elf_bytes"]
        assert observed[3]["link_receipt_sha256"] == sha(files[role + "/link/mx_session_link.json"])
        admission = link["linked_elf_admission"]
        assert admission["status"] == "accepted" and admission["elf_sha256"] == link["elf_sha256"]
        assert sha(files[role + "/link/admission.json"]) == admission["report_sha256"]
        raw_recipe = by_source[link["host_math"]["link_recipe"]]
        recipe = json.loads(raw_recipe)
        assert sha(raw_recipe) == link["host_math"]["link_recipe_sha256"]
        assert recipe["status"] == "completed" and recipe["executable"]["sha256"] == link["elf_sha256"]
        for row in observed[3]["output_dumps"]:
            raw = files[role + "/run/" + row["path"]]
            assert raw == observed[1][row["program"], row["step"], row["output"]]
            assert len(raw) == row["bytes"] == 2048 and sha(raw) == row["sha256"]
    for key in ["stage_object_sha256", "profile_sha256", "shared_weight_sha256", "shared_weight_bytes", "programs", "bindings", "execution_timing_scope"]:
        assert links["reference"][key] == links["candidate"][key]
    assert links["reference"]["host_math"]["contract"] == links["candidate"]["host_math"]["contract"]
    comparison = document("comparison.json")
    assert comparison["reference_cycles"] == reference[2][0] and comparison["candidate_cycles"] == candidate[2][0]
    assert comparison["output_digest_parity"] is True and comparison["changed_stage_objects"] == []
    for role, observed in [("reference", reference), ("candidate", candidate)]:
        assert comparison[role + "_run_sha256"] == sha(files[role + "/run/mx_session_run.json"])
        assert comparison[role + "_link_sha256"] == sha(files[role + "/link/mx_session_link.json"])
        for index, program in enumerate(comparison["programs"]):
            run = observed[3]
            step = sum(r["cycles"] for r in run["step_performance"] if r["program"] == index)
            prep = sum(r["cycles"] for r in run["preparation_performance"] if r["program"] == index)
            assert program[role + "_cycles"] == step + prep
    small_pool = links["candidate"]["shared_catalog_constants"]
    assert small_pool == document("candidate/link/constant_pool/receipt.json")
    assert small_pool["pool"]["original_bytes"] == 2176 and small_pool["pool"]["pooled_bytes"] == 1088
    assert sha(files["candidate/link/constant_pool/shared_constants.o"]) == small_pool["pool_object_sha256"]
    dependency = manifest["large_reference_dependency"]
    assert dependency["directory"] == "cached_latest_session_20261019"
    _, previous = members(HERE.parent / dependency["directory"], "executions.tar.xz", dependency)
    old_link = json.loads(previous["candidate/link/mx_session_link.json"])
    large = document("large/link/mx_session_link.json")
    for key in ["stage_object_sha256", "profile_sha256", "shared_weight_sha256", "shared_weight_bytes", "programs", "bindings", "execution_timing_scope", "mem_bytes"]:
        assert large[key] == old_link[key]
    assert large["host_math"]["contract"] == old_link["host_math"]["contract"]
    pool = document("large/link/constant_pool/receipt.json")
    assert pool == large["shared_catalog_constants"]
    assert len(pool["pool"]["bindings"]) == 620 and len(pool["pool"]["blobs"]) == 310
    assert pool["pool"]["original_bytes"] == 2133491712 and pool["pool"]["pooled_bytes"] == 1066745856
    assert sum(row["bytes"] for row in pool["pool"]["blobs"]) == 1066745856
    for group, observed in pool["groups"].items():
        assert len(observed["reader_proofs"]) == 625
        assert observed["original_object_sha256"] == large["stage_object_sha256"][group]["device_catalog/mx_catalog.o"]
        prefix = "large/link/constant_pool/" + group + "/"
        assert sha(files[prefix + "code.o"]) == observed["code_sha256"]
        for name, proof in observed["reader_proofs"].items():
            assert sha(files[prefix + name + ".c"]) == proof["source_sha256"]
            assert sha(files[prefix + name + ".o"]) == proof["object_sha256"]
            assert proof["source_sha256"] == sha(previous[f"candidate/stages/{group}/device_catalog/{name}.c"])
    assert all(row["exit_code"] == 0 for row in pool["commands"])
    for path, digest in pool["producer_and_tool_pins"].items():
        name = "frozen_producers/" + PurePosixPath(path).name
        if name in files:
            assert sha(files[name]) == digest
    assert all(row["access"] == "readonly_bytes_address_unobserved" for row in pool["pool"]["bindings"])
    for role, link, raw_link in [("reference", old_link, previous["candidate/link/mx_session_link.json"]),
                                 ("candidate", large, files["large/link/mx_session_link.json"])]:
        resource = document("large/resources.json")["images"][role]
        assert resource["elf_bytes"] == link["elf_bytes"] and resource["elf_sha256"] == link["elf_sha256"]
        assert resource["arena_layout"] == link["arena_layout"]
        if args.build_root:
            image = args.build_root / resource["directory"] / "session.elf"
            assert image.stat().st_size == resource["elf_bytes"]
            with image.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == resource["elf_sha256"]
            assert allocated_sections(image) == resource["allocated_sections"]
    admission = document("large/link/admission.json")
    assert sha(files["large/link/admission.json"]) == large["linked_elf_admission"]["report_sha256"]
    assert admission["status"] == "accepted" and admission["elf_sha256"] == large["elf_sha256"]
    raw_recipe = by_source[large["host_math"]["link_recipe"]]
    assert sha(raw_recipe) == large["host_math"]["link_recipe_sha256"]
    recipe = json.loads(raw_recipe)
    assert recipe["status"] == "completed" and recipe["executable"]["sha256"] == large["elf_sha256"]
    producer = files["frozen_producers/link_merlin_session.py"]
    assert sha(producer) == recipe["producer"]["sha256"]
    assert "--share-catalog-constants" in document("large/job/link.command.json")
    assert document("large/job/status.json")["exit_code"] is None
    live = document("large/live_environment.json")
    assert live["command"][-1] == large["elf"]
    assert live["environment"]["GEMMINI_MODE"] == "both" and live["environment"]["GEMMINI_PERF_CONFIG"] == "e4m3_vpu"
    assert live["environment"]["GEMMINI_PERF_SET"] == "mem.host_tracking=0"
    assert document("large/resources.json")["runtime_result"] == "pending"
    assert document("failed_first_attempt/status.json")["exit_code"] == 1
    assert "session_malloc.o" in files["failed_first_attempt/link.log"].decode()
    assert large["elf_bytes"] == 1373933592 and old_link["elf_bytes"] == 2439193888
    print("PASS: complete small recurrent sessions, 8 hashes and 2048 full F32 words identical; 640228 -> 640134 inclusive cycles.")
    print("PASS: original full stage objects and 1250 fresh reader proofs; admitted large image 2439193888 -> 1373933592 bytes.")
    print("Large runtime/output/timing qualification remains pending; small ELF grows 16408 bytes despite 1088 fewer constant bytes.")


if __name__ == "__main__":
    main()
