"""Reopen native call evidence, library members and actual ELF function bytes."""

import hashlib
import importlib.util
import json
import struct
import tarfile
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def archive_members(raw):
    assert raw[:8] == b"!<arch>\n"
    at, names, members = 8, b"", {}
    while at < len(raw):
        header = raw[at:at + 60]
        assert len(header) == 60 and header[-2:] == b"`\n"
        name, size = header[:16].decode().strip(), int(header[48:58])
        value = raw[at + 60:at + 60 + size]
        assert len(value) == size
        if name == "//":
            names = value
        elif name.startswith("/") and name[1:].isdigit():
            start = int(name[1:])
            name = names[start:names.index(b"/\n", start)].decode()
            assert name not in members
            members[name] = value
        elif name != "/":
            name = name.removesuffix("/")
            assert name not in members
            members[name] = value
        at += 60 + size + (size & 1)
    assert at == len(raw)
    return members


def functions(raw, sections):
    result = {}
    for row in sections(raw):
        if row[1] != 2:
            continue
        rows = sections(raw)
        strings = rows[row[6]]
        names = raw[strings[4]:strings[4] + strings[5]]
        for offset in range(row[4], row[4] + row[5], row[9]):
            name, info, _, section, address, size = struct.unpack_from("<IBBHQQ", raw, offset)
            if not section or info & 15 != 2 or not size:
                continue
            target = rows[section]
            position = target[4] + address - target[3]
            assert target[1] != 8 and 0 <= address - target[3] <= target[5] - size
            label = names[name:names.index(b"\0", name)].decode()
            assert label not in result
            result[label] = raw[position:position + size]
    return result


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "artifacts.json").read_text())
    archive = root / manifest["archive"]["name"]
    assert len(archive.read_bytes()) == manifest["archive"]["bytes"]
    assert sha(archive.read_bytes()) == manifest["archive"]["sha256"]
    reader_path = root.parent / "broadcast_multiply_20261019/verify.py"
    assert sha(reader_path.read_bytes()) == manifest["elf_reader_sha256"]
    spec = importlib.util.spec_from_file_location("retained_elf_reader", reader_path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    with tarfile.open(archive) as tar:
        rows = tar.getmembers()
        assert all(row.isfile() for row in rows) and len({row.name for row in rows}) == len(rows)
        assert {row.name for row in rows} == set(manifest["members"])
        data = {row.name: tar.extractfile(row).read() for row in rows}
    for name, raw in data.items():
        assert {"bytes": len(raw), "sha256": sha(raw)} == manifest["members"][name]
    read = lambda name: json.loads(data[name])
    selected = "cases/selected/"
    report, guard = read(selected + "receipt.json"), read(selected + "guard.json")
    assert report["guard"] == guard == json.loads((root / "guard.json").read_text())
    assert report["exit_code"] == 0
    for name, digest in report["files_sha256"].items():
        assert sha(data[selected + name]) == digest
    assert sha(data[selected + "libm.a"]) == guard["library_sha256"]
    profile_sha = sha((json.dumps(read("profile.json"), sort_keys=True, separators=(",", ":")) + "\n").encode())
    assert report["profile_sha256"] == profile_sha
    assert guard["source_symbol"] == "expf" and guard["body_symbol"] == "__ieee754_expf"
    assert guard["max_magnitude_bits"] == 0x42a00000 and guard["exact_points"] == [[0xff800000, 0]]
    library = archive_members(data[selected + "libm.a"])
    for name, digest in read(selected + "library_review_objects.json").items():
        assert library[name] == data[selected + name] and sha(library[name]) == digest
    # The reviewed wrapper's two finite thresholds strictly contain [-80, 80].
    wrapper_object = library["libm_a-wf_exp.o"]
    assert bytes.fromhex("0000807f1772b142b5f1cfc2") in wrapper_object
    upper, lower = struct.unpack("<ff", bytes.fromhex("1772b142b5f1cfc2"))
    assert lower < -80 < 80 < upper
    final_functions = functions(data[selected + "section.elf"], reader.sections)
    assert final_functions["finitef"] == functions(library["libm_a-sf_finite.o"], reader.sections)["finitef"]
    for name in ("expf", "__ieee754_expf", "__wrap_expf", "run_original", "run_selected"):
        assert final_functions[name]
    assert len(final_functions["__wrap_expf"]) == 40
    admission = report["admission"]
    assert admission["status"] == "accepted" and admission["elf_sha256"] == sha(data[selected + "section.elf"])
    candidates = [name for name in data if name.startswith(selected + "invocations/") and name.endswith("/policy/admission.json")]
    assert len(candidates) == 1 and sha(data[candidates[0]]) == admission["report_sha256"]
    assert read(candidates[0])["status"] == "accepted"
    assert report["environment"]["GEMMINI_MODE"] == "both"
    assert report["environment"]["GEMMINI_PERF_CONFIG"] == "e4m3_vpu"
    assert report["environment"]["GEMMINI_PERF_SET"] == "mem.host_tracking=0"
    link = report["commands"][-1]
    assert "-lm" not in link and "-Wl,--wrap=expf" in link
    assert any(argument.endswith("/libm.a") for argument in link)
    baseline_hashes = {}
    for label in manifest["cases"]:
        prefix = "cases/" + label + "/"
        image = data[prefix + "section.elf"]
        assert functions(image, reader.sections)["expf"]
        if label != "selected":
            status = read(prefix + "status.json")
            assert status["exit_code"] == 0 and status["elf_sha"] == sha(image)
            admission = read(prefix + "invocation.json")["admission"]
            assert admission["status"] == "accepted" and admission["elf_sha256"] == sha(image)
        lines = data[prefix + "spike.log"].decode().splitlines()
        assert [line for line in lines if line.startswith("QUALIFY ")] == ["QUALIFY cases=655360 mismatch=0"]
        assert lines.count("DONE") == 1 and not any(line.startswith("DIFF ") for line in lines)
        checks = [dict(part.split("=", 1) for part in line.split()[1:]) for line in lines if line.startswith("EXP ")]
        count = 4 if label == "selected" else 3
        assert [(int(row["kind"]), int(row["run"])) for row in checks] == [(kind, rep) for kind in range(count) for rep in range(3)]
        assert all(int(row["mismatch"]) == 0 for row in checks)
        if label == "selected":
            assert checks == report["checks"]
        for kind in range(count):
            group = [row for row in checks if int(row["kind"]) == kind]
            assert len({(row["original"], row["selected"], row["hash"]) for row in group}) == 1
            row = group[0]
            if kind < 3:
                baseline_hashes.setdefault(kind, row["hash"])
                assert baseline_hashes[kind] == row["hash"]
            if kind in (2, 3):
                zero_hash = 2166136261
                for _ in range(32768):
                    zero_hash = zero_hash * 16777619 & 0xffffffff
                assert int(row["hash"]) == zero_hash
            print(f"{label} distribution{kind}: {row['original']} -> {row['selected']} cycles")
    print("PASS: three native images and 1,966,080 return/errno/flag comparisons; host-call sections only")


if __name__ == "__main__":
    main()
