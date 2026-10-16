"""Check portable captured-graph, command, numerical, and Spike evidence."""

import hashlib
import json
from pathlib import Path
import re
import struct


def verify(root: Path) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["schema"] == "mx_gemmini.connected_dispatch_archive.v1"
    for rows, case in manifest["cases"].items():
        directory = root / f"rows{rows}"
        for name, expected in case["files_sha256"].items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, name
        receipt = json.loads((directory / "receipt.json").read_text())
        assert receipt["production_selected"] and receipt["rows"] == int(rows)
        assert receipt["model2mlir_revision"] == "905d2c918f839596c9d1198cbdeb42d4c12a325b"
        assert json.loads((directory / "native_ir_verify.json").read_text())["exit_code"] == 0
        for field, name in (("source_mlir_sha256", "source.mlir"), ("input_sha256", "input.bin"),
                            ("golden_sha256", "golden.bin"), ("capture_trace_sha256", "capture_trace.json"),
                            ("calibrated_plan_sha256", "calibrated_plan.json")):
            assert receipt[field] == case["files_sha256"][name]
        baseline, connected = receipt["results"]["baseline"], receipt["results"]["connected"]
        assert baseline["driver_sha256"] == connected["driver_sha256"]
        commands = {}
        for variant, measured in receipt["results"].items():
            for field, name in (("issuer_sha256", "issue.c"), ("driver_sha256", "driver.c"),
                                ("commands_sha256", "commands.json"), ("spike_log_sha256", "spike.log"),
                                ("perf_sha256", "perf.txt")):
                assert measured[field] == case["files_sha256"][f"{variant}/{name}"]
            commands[variant] = json.loads((directory / variant / "commands.json").read_text())
            assert len(commands[variant]) == measured["command_count"]
            log = (directory / variant / "spike.log").read_text()
            checks = [[int(v) for v in row] for row in re.findall(
                r"HANDOFF run=(\d+) cycles=(\d+) mismatches=(\d+) hash=(\d+)", log)]
            assert checks == measured["checks"] and [r[0] for r in checks] == [0, 1, 2]
            assert all(r[2] == 0 for r in checks)
            sensitive = re.findall(r"ROUNDING sensitive=(\d+)", log)
            assert len(sensitive) == 3 and all(int(n) > 0 for n in sensitive)
        assert len({r[3] for m in (baseline, connected) for r in m["checks"]}) == 1
        assert [c for c in commands["baseline"] if c.get("funct") == 33] == [
            c for c in commands["connected"] if c.get("funct") == 33]
        plan = json.loads((directory / "calibrated_plan.json").read_text())
        assert len(plan["handoffs"]) == 1
        intermediate = plan["handoffs"][0]["buffer"]
        assert not any(c.get("rs1", {}).get("buffer") == intermediate for c in commands["connected"])
        assert baseline["command_count"] - connected["command_count"] == 2 * ((int(rows) + 15) // 16)
        assert baseline["counters"]["load_bytes"] - connected["counters"]["load_bytes"] == int(rows) * 16 * 3
        assert connected["removed_transfer_bytes_per_call"] == int(rows) * 16 * 2
        savings = min(r[1] for r in baseline["checks"][1:]) - max(r[1] for r in connected["checks"][1:])
        assert savings == receipt["measured_warm_saved_cycles"] == plan["handoffs"][0]["saved_cycles"] > 0

        def widen(value):
            return struct.unpack("<f", struct.pack("<I", value << 16))[0]

        def narrow(value):
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
            return (bits + 0x7fff + ((bits >> 16) & 1)) >> 16

        inputs = struct.unpack(f"<{int(rows) * 8}H", (directory / "input.bin").read_bytes())
        golden = struct.unpack(f"<{int(rows) * 8}H", (directory / "golden.bin").read_bytes())
        assert tuple(narrow(widen(narrow(widen(x) * 1.1015625)) + 0.333984375) for x in inputs) == golden
        print(f"rows={rows}: bit-exact outputs; {savings} fewer warm modeled cycles")


if __name__ == "__main__":
    verify(Path(__file__).resolve().parent)
