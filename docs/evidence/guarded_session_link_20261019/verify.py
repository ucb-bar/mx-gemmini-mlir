"""Reopen archived recurrent-session outputs, timing, and link custody; stdlib only."""

import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / "artifacts.json").read_text())
    archive = here / "executions.tar.xz"
    assert archive.stat().st_size == manifest["archive_bytes"]
    assert sha(archive.read_bytes()) == manifest["archive_sha256"]
    files = {}
    with tarfile.open(archive) as stream:
        for member in stream:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts
            assert member.name not in files
            files[member.name] = stream.extractfile(member).read()
    assert set(files) == set(manifest["artifacts"])
    by_source = {}
    for name, identity in manifest["artifacts"].items():
        assert len(files[name]) == identity["bytes"] and sha(files[name]) == identity["sha256"]
        by_source[identity["source_path"]] = files[name]

    def document(name):
        return json.loads(files["fixture/" + name])

    def identity(record):
        data = by_source[record["path"]]
        assert len(data) == record["bytes"] and sha(data) == record["sha256"]

    expected_steps = [(0, 0), (1, 0), (1, 1), (1, 2)]

    def observe(role):
        log = files[f"fixture/{role}/spike.log"].decode()
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
        run = document(role + "/mx_session_run.json")
        assert run["status"] == "complete_numeric_qualification_pending" and run["exit_code"] == 0
        assert run["complete"] is True and sha(log.encode()) == run["log_sha256"]
        assert sha(files[f"fixture/{role}/gemmini_perf.txt"]) == run["timing_report_sha256"]
        assert run["execution_total_performance"]["cycles"] == execution[0]
        assert run["timing_mode"] == "both" and run["timing_config"] == "e4m3_vpu"
        assert run["timing_overrides"] == "mem.host_tracking=0"
        return digests, dumps, execution, run

    native = observe("native_run")
    guarded = observe("guarded_run_checked")
    original = observe("guarded_run")
    assert native[:2] == guarded[:2] == original[:2]
    assert guarded[2] == original[2]
    assert (native[2][0], guarded[2][0]) == (679140, 640228)
    comparison = document("checked_comparison.json")
    assert comparison["reference_cycles"] == native[2][0] and comparison["candidate_cycles"] == guarded[2][0]
    assert comparison["output_digest_parity"] is True and comparison["changed_stage_objects"] == []
    assert comparison["session_host_math"]["reference"] is None
    assert native[3]["spike_sha256"] == guarded[3]["spike_sha256"]
    assert native[3]["extension_sha256"] == guarded[3]["extension_sha256"]
    links = [document(role + "_link/mx_session_link.json") for role in ("native", "guarded")]
    for role, link in zip(("native", "guarded"), links):
        assert sha(files[f"fixture/{role}_link/session.elf"]) == link["elf_sha256"]
        admission = link["linked_elf_admission"]
        assert admission["status"] == "accepted" and admission["elf_sha256"] == link["elf_sha256"]
        assert sha(by_source[admission["report_path"]]) == admission["report_sha256"]
        for program, objects in link["stage_object_sha256"].items():
            for path, digest in objects.items():
                assert sha(files[f"fixture/{role}_{program}/{path}"]) == digest
    assert links[0]["stage_object_sha256"] == links[1]["stage_object_sha256"]
    record = links[1]["host_math"]
    guard = record["contract"]
    assert comparison["session_host_math"]["candidate"] == guard
    assert sha(files["toolchain/libm.a"]) == guard["library_sha256"]
    recipe = json.loads(by_source[record["link_recipe"]])
    assert sha(by_source[record["link_recipe"]]) == record["link_recipe_sha256"]
    assert recipe["status"] == "completed" and recipe["executable"]["sha256"] == links[1]["elf_sha256"]
    identity(recipe["producer"])
    for preparation in recipe["preparation"].values():
        identity(preparation)
    command = recipe["commands"][-1]
    for item in command["inputs"]:
        identity(item)
    identity(command["output"])
    assert command["argv"].count("-Wl,--wrap=" + guard["source_symbol"]) == 1
    suppliers = recipe["link_suppliers"]
    assert set(suppliers["symbols"]) == {guard["source_symbol"], guard["body_symbol"]}
    for symbol, item in suppliers["symbols"].items():
        identity(item["selected_input"])
        assert item["selected_input"]["sha256"] == guard["library_sha256"]
        assert command["argv"].count("-Wl,--trace-symbol=" + symbol) == 1
        definitions = [line.rsplit(": ", 1)[0].split(": ", 1)[1] for line in suppliers["stderr"].splitlines()
                       if line.endswith(": definition of " + symbol)]
        assert definitions == [item["definition"]]
        assert item["definition"].startswith(item["selected_input"]["path"] + "(")
    for name, key in (("host_math.c", "source_sha256"), ("host_math.o", "object_sha256")):
        assert sha(files["fixture/guarded_link/host_math/" + name]) == record[key]
    for role in ("native_run", "guarded_run_checked"):
        counter = files[f"fixture/{role}/gemmini_perf.txt"].decode()
        assert "308 commands, 4 loops, 220 fences" in counter
        assert "16 tiles, 256 busy cycles; VPU: 16 commands" in counter
        assert "load: 38912 bytes" in counter and "scales: 192 bytes" in counter
        assert "host tracking off" in counter
    assert "explicit session selection" in files["fixture/missing_selection_refused.log"].decode()
    profile = json.loads(files["profile.json"])
    canonical = (json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n").encode()
    assert sha(canonical) == links[0]["profile_sha256"] == links[1]["profile_sha256"]
    print(f"PASS: {len(files)} artifacts; 8 equal state/output hashes; 2048 equal F32 dump words; "
          "679140 -> 640228 inclusive cycles; original library suppliers and final images reopened")


if __name__ == "__main__":
    main()
