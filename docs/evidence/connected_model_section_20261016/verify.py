"""Check the archived tensors, BF16 numerical policy, logs and traffic receipts."""

import gzip
import hashlib
import json
from pathlib import Path
import re
import struct


def sha(data):
    return hashlib.sha256(data).hexdigest()


def narrow(value):
    word = struct.unpack("<I", struct.pack("<f", value))[0]
    return ((word + 0x7fff + ((word >> 16) & 1)) >> 16) & 0xffff


def widen(value):
    return struct.unpack("<f", struct.pack("<I", value << 16))[0]


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        assert sha((root / name).read_bytes()) == expected, name
    for name in ("tiny_mlp", "tail137", "fallback137"):
        directory = root / name
        receipt = json.loads((directory / "receipt.json").read_text())
        assert receipt["boundary_policy"] == "bf16_rne_each_vpu_call"
        payload = gzip.decompress((directory / "payload.bin.gz").read_bytes())
        assert sha(payload) == receipt["files_sha256"]["payload.bin"]
        count = receipt["elements"]
        assert len(payload) == count * 16
        expected_hash = 2166136261
        for i in range(count):
            a, b, c = [struct.unpack_from("<f", payload, (j * count + i) * 4)[0] for j in range(3)]
            middle = narrow(widen(narrow(a)) * widen(narrow(b)))
            result = narrow(widen(middle) * widen(narrow(c))) << 16
            golden = struct.unpack_from("<I", payload, (3 * count + i) * 4)[0]
            assert result == golden, (name, i)
            expected_hash = ((expected_hash ^ golden) * 16777619) & 0xffffffff
        assert json.loads((directory / "native_ir_verify.json").read_text())["exit_code"] == 0
        for filename in ("source.mlir", "outlined.mlir", "dispatch_program.json", "driver.c",
                         "plan.json", "calibrated_plan.json"):
            assert sha((directory / filename).read_bytes()) == receipt["files_sha256"][filename]
        for variant, record in receipt["results"].items():
            data = directory / variant
            assert sha(gzip.decompress((data / "issue.c.gz").read_bytes())) == record["files_sha256"]["issue.c"]
            for filename in ("spike.log", "perf.txt"):
                assert sha((data / filename).read_bytes()) == record["files_sha256"][filename]
            checks = [[int(v) for v in row] for row in re.findall(
                r"SECTION run=(\d+) cycles=(\d+) mismatches=(\d+) hash=(\d+)",
                (data / "spike.log").read_text())]
            assert checks == record["checks"] and [r[0] for r in checks] == [0, 1, 2]
            assert all(r[2] == 0 and r[3] == expected_hash for r in checks)
        before, after = receipt["results"]["materialized"], receipt["results"]["connected"]
        saved = min(r[1] for r in before["checks"][1:]) - max(r[1] for r in after["checks"][1:])
        assert saved == receipt["measured_warm_saved_cycles"]
        plan = json.loads((directory / "calibrated_plan.json").read_text())
        assert bool(plan["handoffs"]) == receipt["production_selected"] == (saved > 0)
        expected_load_saving = 0 if receipt["expected_fallback"] else ((count + 7) // 8) * 16 * 3
        assert before["counters"]["load_bytes"] - after["counters"]["load_bytes"] == expected_load_saving
        if name == "tiny_mlp":
            capture = receipt["capture_receipt"]
            assert capture["original_layer_golden_checked"]
            assert capture["model2mlir_revision"] == "905d2c918f839596c9d1198cbdeb42d4c12a325b"
        print(f"{name}: {count} policy-correct outputs; {saved} warm modeled cycles saved")


if __name__ == "__main__":
    verify(Path(__file__).resolve().parent)
