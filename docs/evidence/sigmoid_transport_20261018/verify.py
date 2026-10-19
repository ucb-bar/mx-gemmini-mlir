"""Check full section words, admitted ELF identities and matched warm timings."""

from array import array
import hashlib
import json
from pathlib import Path
import sys
import tarfile


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    for name, digest in json.loads((here / "artifacts.json").read_text()).items():
        assert sha((here / name).read_bytes()) == digest
    with tarfile.open(here / "executions.tar.xz", "r:xz") as archive:
        data = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    inventory = json.loads(data.pop("members.json"))
    assert inventory.keys() == data.keys()
    for name, digest in inventory.items():
        assert sha(data[name]) == digest, name
    for case in ("tail137", "exhaustive", "fallback8311", "real_mlp64"):
        report = json.loads(data[case + "/receipt.json"])
        assert report["all_words_equal"] and report["scope"].startswith("section_not_whole_model")
        baseline = None
        for variant, row in report["results"].items():
            p = case + "/" + variant + "/"
            log = data[p + "spike.log"]
            assert sha(log) == row["console_sha256"]
            values, checks = array("I"), []
            lines = log.decode().splitlines()
            assert lines.count("DONE") == 1
            for line in lines:
                parts = line.split()
                if parts and parts[0] == "WORDS":
                    assert int(parts[1]) == len(values) and int(parts[2]) == len(parts[3:])
                    values.extend(int(word, 16) for word in parts[3:])
                elif parts and parts[0] == "SECTION":
                    checks.append(list(map(int, parts[1:])))
            assert len(values) == row["output_elements"] and values.itemsize == 4
            if sys.byteorder != "little":
                values.byteswap()
            raw = values.tobytes()
            assert raw == data[p + "output.bin"] and sha(raw) == row["all_words_sha256"]
            assert checks == row["checks"] and len(checks) == 3 and not any(r[2] for r in checks)
            if baseline is None:
                baseline = raw
            assert raw == baseline
            image = data[p + "section.elf"]
            admission = json.loads(data[p + "admission.json"])
            assert image[:4] == b"\x7fELF" and sha(image) == row["elf_sha256"]
            assert admission["status"] == "accepted" and admission["elf_sha256"] == sha(image)
            assert sha(data[p + "admission.json"]) == row["elf_admission"]["report_sha256"]
        rows = report["results"]
        assert len({r["perf"]["vpu_commands"] for r in rows.values()}) == (2 if case == "fallback8311" else 1)
        print(case, {name: min(c[1] for c in row["checks"][1:]) for name, row in rows.items()}, "all words exact")
    print("PASS: all12 ELFs and complete section outputs; late fallback remains unprofitable")


if __name__ == "__main__":
    main()
