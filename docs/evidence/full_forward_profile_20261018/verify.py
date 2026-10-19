"""Recompute the retained profile and hash every complete output word in memory."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path,
                        help="optional mx_model_goal_20261018 directory holding the actual ELF")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "artifacts.json").read_text())
    archive = root / manifest["archive"]["name"]
    assert archive.stat().st_size == manifest["archive"]["bytes"]
    assert digest(archive.read_bytes()) == manifest["archive"]["sha256"]
    with tarfile.open(archive) as tar:
        assert all(member.isfile() for member in tar.getmembers())
        assert {member.name for member in tar.getmembers()} == set(manifest["members"])
        data = {member.name: tar.extractfile(member).read() for member in tar.getmembers()}
    for name, raw in data.items():
        assert len(raw) == manifest["members"][name]["bytes"]
        assert digest(raw) == manifest["members"][name]["sha256"]

    class MemoryPath:
        def __init__(self, name=""):
            self.name = name

        def __truediv__(self, name):
            return MemoryPath(self.name + ("/" if self.name else "") + str(name))

        def exists(self):
            return self.name in data

        def read_text(self, **_kwargs):
            return data[self.name].decode()

        def read_bytes(self):
            return data[self.name]

    # Use the pinned reader retained with the run, without extracting scratch
    # files or importing a subsequently changed compiler implementation.
    namespace = {"__name__": "retained_profile_reader"}
    exec(compile(data["reader/report_stage_profile.py"], "retained_profile_reader", "exec"), namespace)  # noqa: S102
    report = namespace["report_stage_profile"](MemoryPath(), MemoryPath("spike.log"), MemoryPath("spike.log"))
    assert report == json.loads(data["stage_profile_report.json"])
    receipt = json.loads(data["mx_stage_probe.json"])
    assert receipt["spike"]["done"] and receipt["spike"]["exit_code"] == 0
    assert receipt["op_profile"] is True
    validation = json.loads(data["complete_output_validation.json"])
    assert validation["validated"] and validation["complete_output_elements"] == 5120000
    assert validation["console_sha256"] == digest(data["spike.log"])
    count, expected, word_digest = 0, None, hashlib.sha256()
    for line in data["spike.log"].decode().splitlines():
        parts = line.split()
        if parts[:1] == ["OUT_BEGIN"]:
            assert expected is None and parts[1] == "f32"
            expected = int(parts[2])
        elif parts[:1] == ["OUT_CHUNK"]:
            assert parts[1] == "f32" and int(parts[2]) == count and len(parts[4:]) == int(parts[3])
            word_digest.update(b"".join(int(word).to_bytes(4, "little") for word in parts[4:]))
            count += int(parts[3])
    observed = validation["output_digests"]["indexed"][0]
    assert count == expected == observed["elements"] == 5120000
    assert word_digest.hexdigest() == observed["sha256"]
    recipe = json.loads(data["compilation_recipe.json"])
    admissions = [json.loads(raw) for name, raw in data.items() if name.endswith("/policy/admission.json")]
    assert len(admissions) == 1 and admissions[0]["status"] == "accepted"
    assert recipe["status"] == "completed" and recipe["executable"]["sha256"] == admissions[0]["elf_sha256"]
    if args.build_root is not None:
        elf = args.build_root / "onepass160_shared_transport_profile/model.elf"
        assert digest(elf.read_bytes()) == recipe["executable"]["sha256"]
    print(json.dumps({"scope": "instrumented_call_intervals_not_clean_speedup",
                      "cycles": report["cycles"], "coverage": report["attribution_coverage"],
                      "by_owner": report["by_owner"], "output_sha256": word_digest.hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
