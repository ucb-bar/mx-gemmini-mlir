"""Reopen the retained scalar target probe; no full-model qualification."""

import hashlib
import json
import re
import tarfile
from decimal import Decimal, localcontext
from pathlib import Path

HERE = Path(__file__).resolve().parent
sha = lambda raw: hashlib.sha256(raw).hexdigest()
manifest = json.loads((HERE / "artifacts.json").read_text())
archive = HERE / "probe.tar.xz"
assert sha(archive.read_bytes()) == manifest["archive_sha256"]
with tarfile.open(archive, "r:xz") as tar:
    assert set(tar.getnames()) == set(manifest["members"])
    files = {name: tar.extractfile(name).read() for name in tar.getnames()}
for name, raw in files.items():
    assert len(raw) == manifest["members"][name]["bytes"]
    assert sha(raw) == manifest["members"][name]["sha256"]
record = json.loads(files["comparison.json"])
assert record["passed"] and record["all_compared_return_words_equal"]
assert record["rounding_mode"] == "rne" and record["floating_effects"] == "unobserved"
assert sha(files["probe.elf"]) == record["elf_sha256"]
assert sha(files["input.bin"]) == record["input_sha256"]
assert sha(files["perf_input.bin"]) == record["performance_input_sha256"]
assert len(files["input.bin"]) == 4 * record["input_count"]
assert len(files["perf_input.bin"]) == 4 * record["performance_input_count"]
assert files["input.bin"] in files["probe.elf"]
assert files["perf_input.bin"] in files["probe.elf"]
log = files["spike.log"].decode()
assert "ERRORS 0\n" in log and "*** FAILED ***" not in log
for variant in (0, 1):
    trials = [
        (int(i), int(c))
        for i, v, c in re.findall(r"^CYCLES (\d+) (\d+) (\d+)$", log, re.MULTILINE)
        if int(v) == variant
    ]
    assert [i for i, _ in trials] == list(range(4))
    assert [c for _, c in trials] == record["cycles"][str(variant)]
with localcontext() as ctx:
    ctx.prec = 100
    assert Decimal(-104).exp() < Decimal(2) ** -150
print(
    json.dumps(
        {"passed": True, "scope": manifest["scope"], "cycles": record["cycles"]},
        indent=2,
    )
)
