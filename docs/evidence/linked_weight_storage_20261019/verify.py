"""Verify fresh-byte artifact sharing preserves a complete recurrent session."""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import tarfile
from pathlib import Path, PurePosixPath


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-live-storage", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / "artifacts.json").read_text())
    raw = (here / manifest["archive"]).read_bytes()
    assert len(raw) == manifest["archive_bytes"] and sha(raw) == manifest["archive_sha256"]
    files, by_source = {}, {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts and member.name not in files
            files[member.name] = archive.extractfile(member).read()
    assert set(files) == set(manifest["members"])
    for name, raw in files.items():
        record = manifest["members"][name]
        assert sha(raw) == record["sha256"] and len(raw) == record["bytes"]
        source = record["source_path"]
        assert source not in by_source or by_source[source] == raw
        by_source[source] = raw
    baseline = here.parent / "guarded_session_link_20261019"
    for filename, key in [("verify.py", "baseline_verifier_sha256"),
                          ("executions.tar.xz", "baseline_archive_sha256"),
                          ("artifacts.json", "baseline_manifest_sha256")]:
        assert sha((baseline / filename).read_bytes()) == manifest[key]
    spec = importlib.util.spec_from_file_location("retained_session_baseline", baseline / "verify.py")
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    with contextlib.redirect_stdout(io.StringIO()):
        reader.main()

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
    assert reference[:3] == candidate[:3] and candidate[2][0] == 640228
    assert reference[3]["output_digests"] == candidate[3]["output_digests"]
    assert reference[3]["execution_total_performance"] == candidate[3]["execution_total_performance"]
    assert files["reference/run/gemmini_perf.txt"] == files["candidate/run/gemmini_perf.txt"]
    comparison = document("comparison.json")
    assert comparison["reference_cycles"] == comparison["candidate_cycles"] == 640228
    assert comparison["candidate_over_reference"] == 1.0 and comparison["output_digest_parity"]
    assert comparison["changed_stage_objects"] == []
    links = []
    for role, observed in [("reference", reference), ("candidate", candidate)]:
        link = document(role + "/link/mx_session_link.json")
        links.append(link)
        assert sha(files[role + "/link/session.elf"]) == link["elf_sha256"]
        assert observed[3]["link_receipt_sha256"] == sha(files[role + "/link/mx_session_link.json"])
        admission = link["linked_elf_admission"]
        assert admission["status"] == "accepted" and admission["elf_sha256"] == link["elf_sha256"]
        assert sha(by_source[admission["report_path"]]) == admission["report_sha256"]
        recipe = json.loads(by_source[link["host_math"]["link_recipe"]])
        assert recipe["status"] == "completed" and recipe["executable"]["sha256"] == link["elf_sha256"]
        assert sha(by_source[link["host_math"]["link_recipe"]]) == link["host_math"]["link_recipe_sha256"]
        for dump in observed[3]["output_dumps"]:
            raw = files[role + "/run/" + dump["path"]]
            key = (dump["program"], dump["step"], dump["output"])
            assert len(raw) == dump["bytes"] == 2048 and sha(raw) == dump["sha256"]
            assert raw == observed[1][key]
        assert len(observed[3]["output_dumps"]) == 4
    assert files["reference/link/session.elf"] == files["candidate/link/session.elf"]
    assert links[0]["stage_object_sha256"] == links[1]["stage_object_sha256"]
    assert links[0]["host_math"]["contract"] == links[1]["host_math"]["contract"]
    assert links[0]["shared_weight_sha256"] == links[1]["shared_weight_sha256"]
    storage = document("candidate/link/linked_weight_object_storage_reuse.json")
    assert storage == links[1]["linked_weight_object_storage_reuse"]
    assert storage["method"] == "fresh_full_byte_check_then_shared_inode" and storage["shared_inode"]
    assert storage["fresh_producer_exit_code"] == 0 and not storage["already_shared"]
    assert storage["fresh_producer_command"][-1] == "weights.bin"
    assert by_source[storage["previous"]] == by_source[storage["fresh"]]
    assert sha(by_source[storage["fresh"]]) == storage["sha256"] and len(by_source[storage["fresh"]]) == storage["bytes"] == 744
    assert sha(by_source[storage["fresh_source"]]) == storage["fresh_source_sha256"] == links[1]["shared_weight_sha256"]
    assert len(by_source[storage["fresh_source"]]) == links[1]["shared_weight_bytes"] == 128
    large = document("full_session/link.json")
    large_storage = document("full_session/storage.json")
    large_recipe = document("full_session/recipe.json")
    large_admission = document("full_session/admission.json")
    observation = document("full_session/observation.json")
    assert large_storage == large["linked_weight_object_storage_reuse"]
    assert large_storage["bytes"] == observation["weight_object_bytes"] == 262513256
    assert large_storage["sha256"] == observation["weight_object_sha256"]
    assert large_storage["shared_inode"] and observation["shared_inode_observed"]
    assert large_storage["fresh_producer_exit_code"] == 0
    assert large_storage["fresh_source_sha256"] == large["shared_weight_sha256"]
    assert large["shared_weight_bytes"] == 262512640
    assert large_recipe["status"] == "completed" and large_admission["status"] == "accepted"
    assert large_recipe["executable"]["sha256"] == large_admission["elf_sha256"] == large["elf_sha256"] == observation["elf_sha256"]
    assert observation["runtime_result"] == "pending"
    assert observation["scope"] == "complete_fresh_link_and_storage_identity_not_completed_runtime_or_numerical_qualification"
    weight_inputs = [item for item in large_recipe["commands"][-1]["inputs"] if item["path"] == large_storage["fresh"]]
    assert len(weight_inputs) == 1 and weight_inputs[0]["sha256"] == large_storage["sha256"]
    if args.check_live_storage:
        assert os.path.samefile(storage["previous"], storage["fresh"])
        assert sha(Path(storage["previous"]).read_bytes()) == sha(Path(storage["fresh"]).read_bytes()) == storage["sha256"]
        assert os.path.samefile(large_storage["previous"], large_storage["fresh"])
        with Path(large_storage["fresh"]).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == large_storage["sha256"]
        image = Path(observation["elf"])
        assert image.stat().st_size == observation["elf_bytes"]
        with image.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == observation["elf_sha256"]
    print("PASS: pinned baseline and", len(files), "files; two identical actual ELFs; 8 output/state hashes;")
    print("2048 equal F32 dump words, 128 unchanged weight bytes, 640228 inclusive cycles unchanged")


if __name__ == "__main__":
    main()
