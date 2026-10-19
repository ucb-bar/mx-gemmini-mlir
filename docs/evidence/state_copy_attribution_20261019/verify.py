"""Reopen actual runtime object custody and complete state-copy observations."""

import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / "artifacts.json").read_text())
    raw = (here / manifest["archive"]).read_bytes()
    assert len(raw) == manifest["archive_bytes"] and sha(raw) == manifest["archive_sha256"]
    data = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts and member.name not in data
            data[member.name] = archive.extractfile(member).read()
    assert set(data) == set(manifest["members"])
    for name, raw in data.items():
        assert manifest["members"][name] == {"sha256": sha(raw), "bytes": len(raw)}
    report = json.loads(data["measurement/receipt.json"])
    for name, digest in report["files_sha256"].items():
        assert sha(data["measurement/" + name]) == digest
    profile_sha = sha((json.dumps(json.loads(data["profile.json"]), sort_keys=True, separators=(",", ":")) + "\n").encode())
    assert report["profile_sha256"] == profile_sha and report["exit_code"] == 0
    admissions = [name for name in data if name.endswith("/policy/admission.json")]
    assert len(admissions) == 1
    admission = json.loads(data[admissions[0]])
    assert sha(data[admissions[0]]) == report["admission"]["report_sha256"]
    assert admission["status"] == "accepted" and admission["elf_sha256"] == sha(data["measurement/section.elf"])
    recipe = json.loads(data["measurement/original_compilation_recipe.json"])
    for name in ["merlin_model", "crt", "console", "libc_min"]:
        rows = [row for row in recipe["commands"] if Path(row.get("output", {}).get("path", "")).name == name + ".o"]
        assert len(rows) == 1 and rows[0]["output"]["sha256"] == sha(data["measurement/" + name + ".o"])
        for item in rows[0]["inputs"]:
            assert sha(data["runtime_inputs/" + Path(item["path"]).name]) == item["sha256"]
    assert report["per_call_bytes"] == sum(pair["bytes"] for pair in report["pairs"]) == 7209136
    assert report["pairs"] == [{"input": 49, "output": 1, "bytes": 7208960}, {"input": 50, "output": 2, "bytes": 176}]
    lines = data["measurement/spike.log"].decode().splitlines()
    observations = [dict(field.split("=", 1) for field in line.split()[1:]) for line in lines if line.startswith("STATE_COPY ")]
    assert observations == report["performance"] and lines.count("DONE") == 1
    assert [row["run"] for row in observations] == ["0", "1", "2", "3"]
    assert [row["calls"] for row in observations] == ["1", "1", "1", "32"]
    assert [row["cycles"] for row in observations] == ["4506607"] * 3 + ["144210804"]
    assert all(row["bad"] == "0" for row in observations)
    env = report["environment"]
    assert [env[key] for key in ["GEMMINI_MODE", "GEMMINI_PERF_CONFIG", "GEMMINI_PERF_SET"]] == [
        "both", "e4m3_vpu", "mem.host_tracking=0"]
    print("PASS:", len(data), "files, actual ELF and four runtime objects, complete bytes/canaries checked by retained driver")
    print("32 full state copies:", observations[-1]["cycles"], "cycles;",
          100 * int(observations[-1]["cycles"]) / manifest["reference_cached_cycles"], "% of reference cached session")


if __name__ == "__main__":
    main()
