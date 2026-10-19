"""Reopen complete cached-session timing, output streams and FP32 diagnostics."""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np

HERE = Path(__file__).resolve().parent
SCOPE = ("reset_plus_program_state_handoff_plus_existing_step_intervals; "
         "console_and_phase_bookkeeping_excluded")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def metrics(actual, expected):
    assert actual.shape == expected.shape == (1, 32000)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    lhs, rhs = actual.astype(np.float64), expected.astype(np.float64)
    rmse = float(np.sqrt(np.mean((lhs - rhs) ** 2)))
    actual_top, reference_top = int(actual.argmax()), int(expected.argmax())

    def logs(x):
        value = x - x.max(axis=-1, keepdims=True)
        return value - np.log(np.exp(value).sum(axis=-1, keepdims=True))

    q, p = logs(rhs), logs(lhs)
    return {
        "cosine": float(np.clip(np.sum(lhs * rhs) / max(float(np.linalg.norm(lhs) * np.linalg.norm(rhs)), 1e-300), -1, 1)),
        "rmse": rmse, "normalized_rmse": rmse / max(float(np.sqrt(np.mean(rhs * rhs))), 1e-300),
        "actual_top_token": actual_top, "reference_top_token": reference_top,
        "top_token_equal": actual_top == reference_top,
        "reference_top_rank_in_actual": 1 + int(np.count_nonzero(actual > actual[0, reference_top])),
        "kl_nats": float(max(0, np.sum(np.exp(q) * (q - p)))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, help="optional retained mx_model_goal_20261018 directory")
    args = parser.parse_args()
    manifest = json.loads((HERE / "artifacts.json").read_text())
    raw = (HERE / "executions.tar.xz").read_bytes()
    assert sha(raw) == manifest["archive_sha256"] and len(raw) == manifest["archive_bytes"]
    data = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts
            assert member.name not in data
            data[member.name] = archive.extractfile(member).read()
    assert set(data) == set(manifest["members"])
    for name, value in data.items():
        assert {"sha256": sha(value), "bytes": len(value)} == manifest["members"][name]

    def load(name):
        return json.loads(data[name])

    diagnostic = load("fp32_diagnostic.json")
    assert diagnostic["numerical_qualification"] == "not_established"
    assert diagnostic["all99_output_state_hashes_equal"] and diagnostic["all33_complete_logit_tensors_equal"]
    for name, digest in diagnostic["references"].items():
        assert sha(data["capture/" + name]) == digest
    expected = [np.load(io.BytesIO(data["capture/stages/prefill/golden.npy"]), allow_pickle=False)]
    with np.load(io.BytesIO(data["capture/stages/decode/session_quality_fp32.npz"]), allow_pickle=False) as archive:
        assert archive["logits"].shape == (32, 1, 32000)
        expected.extend(archive["logits"])
    runs, links, outputs, counters = {}, {}, {}, {}
    for role, label in [("baseline", "placement_baseline"), ("optimized", "optimized"), ("virtual", "virtual")]:
        lp, rp = role + "/link/", role + "/run/"
        link, run = load(lp + "mx_session_link.json"), load(rp + "mx_session_run.json")
        runs[role], links[role] = run, link
        assert run["link_receipt_sha256"] == sha(data[lp + "mx_session_link.json"])
        assert run["complete"] is True and run["exit_code"] == 0
        assert run["status"] == "complete_numeric_qualification_pending" and run["timing_model_active"] is True
        assert run["expected_events"] == run["observed_events"] == 132
        assert run["timing_mode"] == "both" and run["timing_config"] == "e4m3_vpu"
        assert run["timing_overrides"] == "mem.host_tracking=0"
        assert run["execution_timing_scope"] == link["execution_timing_scope"] == SCOPE
        assert run["performance_schema"] == link["performance_schema"]
        assert run["preparation_performance_schema"] == link["preparation_performance_schema"] == "spike_rdcycle_rdinstret_reset_prepare.v1"
        assert [(p["name"], p["steps"], p["outputs"]) for p in link["programs"]] == [("prefill", 1, 3), ("decode", 32, 3)]
        assert load(role + "/job/status.json")["exit_code"] == 0
        assert sha(data[rp + "spike.log"]) == run["log_sha256"]
        live = load(rp + "live_environment.json")
        assert live["command"] == run["command"]
        for name, value in [("GEMMINI_MODE", "both"), ("GEMMINI_PERF_CONFIG", "e4m3_vpu"), ("GEMMINI_PERF_SET", "mem.host_tracking=0")]:
            assert live["environment"][name] == value
        console = data[rp + "spike.log"].decode()
        assert console.splitlines().count("DONE") == 1 and "METRIC memref_rank_mismatch 0\n" in console
        assert not any(s.startswith(("FATAL", "FAIL ")) for s in console.splitlines())
        lines = [s.split() for s in console.splitlines() if s.startswith("SESSION_")]
        cursor = 0

        def take(name, width):
            nonlocal cursor
            parts = lines[cursor]
            cursor += 1
            assert parts[0] == name and len(parts) == width
            return parts[1:]

        preparations, steps, digests, values = [], [], [], {}
        for program in [-1, 0, 1]:
            p, cycles, inst = map(int, take("SESSION_PREP_PERF", 4))
            assert p == program and min(cycles, inst) > 0
            preparations.append(dict(program=p, cycles=cycles, instructions=inst))
            if program == -1:
                continue
            for step in range(1 if program == 0 else 32):
                assert list(map(int, take("SESSION_BEGIN", 3))) == [program, step]
                p, s, cycles, inst = map(int, take("SESSION_PERF", 5))
                assert (p, s) == (program, step) and min(cycles, inst) > 0
                steps.append(dict(program=p, step=s, cycles=cycles, instructions=inst))
                for output, size in enumerate([128000, 7208960, 176]):
                    p, s, o, n, digest = take("SESSION_SHA", 6)
                    assert list(map(int, [p, s, o, n])) == [program, step, output, size]
                    digests.append(dict(program=program, step=step, output=output, bytes=size, sha256=digest))
                    if output == 0:
                        p, s, o, n, encoded = take("SESSION_HEX", 6)
                        assert list(map(int, [p, s, o, n])) == [program, step, 0, size]
                        content = bytes.fromhex(encoded)
                        assert len(content) == size and sha(content) == digest
                        values[program, step, 0] = content
        total = dict(zip(["cycles", "instructions"], map(int, take("SESSION_TOTAL", 3))))
        execution = dict(zip(["cycles", "instructions"], map(int, take("SESSION_EXECUTION_TOTAL", 3))))
        assert cursor == len(lines)
        assert steps == run["step_performance"] and preparations == run["preparation_performance"]
        assert digests == run["output_digests"] and len(digests) == 99
        assert total == run["total_performance"] and execution == run["execution_total_performance"]
        for key in ["cycles", "instructions"]:
            assert total[key] == sum(r[key] for r in steps)
            assert execution[key] == total[key] + sum(r[key] for r in preparations)
            assert run["preparation_total_performance"][key] == sum(r[key] for r in preparations)
        assert len(values) == len(run["output_dumps"]) == 33
        for row in run["output_dumps"]:
            key = row["program"], row["step"], row["output"]
            path = PurePosixPath(row["path"])
            assert not path.is_absolute() and ".." not in path.parts
            assert data[rp + str(path)] == values[key] and len(values[key]) == row["bytes"] == 128000
            assert sha(values[key]) == row["sha256"]
        outputs[role] = values
        stored = diagnostic["runs"][role]
        assert stored["run_sha256"] == sha(data[rp + "mx_session_run.json"])
        assert stored["link_sha256"] == sha(data[lp + "mx_session_link.json"])
        measured = []
        for index, (key, content) in enumerate(values.items()):
            observed = metrics(np.frombuffer(content, dtype="<f4").reshape(1, 32000), expected[index])
            row = stored["rows"][index]
            assert (row["program"], row["step"]) == key[:2]
            for name, value in observed.items():
                assert abs(row[name] - value) < 1e-10
            measured.append(observed)
        summary = stored["summary"]
        assert abs(summary["mean_kl_nats"] - np.mean([r["kl_nats"] for r in measured])) < 1e-10
        assert abs(summary["worst_kl_nats"] - max(r["kl_nats"] for r in measured)) < 1e-10
        assert summary["top_token_matches"] == sum(r["top_token_equal"] for r in measured)
        assert sha(data[rp + "gemmini_perf.txt"]) == run["timing_report_sha256"]
        perf = data[rp + "gemmini_perf.txt"].decode().splitlines()
        first, mesh, traffic = [line.split() for line in perf[:3]]
        assert first[:3] == ["gemmini", "perf", "[e4m3_vpu]:"] and mesh[0] == "mesh:" and traffic[0] == "load:"
        counters[role] = {
            "commands": int(first[3]), "loops": int(first[5]), "fences": int(first[7]),
            "host_stalled": int(first[first.index("stalled") + 1]),
            "mesh_tiles": int(mesh[1]), "mesh_busy": int(mesh[3]), "vpu_commands": int(mesh[7]),
            "load_bytes": int(traffic[1]), "gets": int(traffic[4]),
            "store_reads": int(traffic[traffic.index("store:") + 1]),
            "puts": int(traffic[traffic.index("store:") + 3]),
            "scale_bytes": int(traffic[traffic.index("scales:") + 1]),
        }
        admission = load(lp + "admission.json")
        assert sha(data[lp + "admission.json"]) == link["linked_elf_admission"]["report_sha256"]
        assert admission["status"] == "accepted" and admission["elf_sha256"] == link["elf_sha256"]
        if args.build_root:
            image = args.build_root / f"cached_session_{label}_link/session.elf"
            assert image.stat().st_size == link["elf_bytes"]
            with image.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == link["elf_sha256"]
        print(role, execution["cycles"], "cycles; full protocol, taps and diagnostics checked")
    for role in ["optimized", "virtual"]:
        assert outputs[role] == outputs["baseline"]
        assert runs[role]["output_digests"] == runs["baseline"]["output_digests"]
        for key in ["programs", "bindings", "profile_sha256", "shared_weight_sha256", "shared_weight_bytes", "mem_bytes"]:
            assert links[role][key] == links["baseline"][key]
        for key in ["spike_sha256", "extension_sha256", "timing_mode", "timing_config", "timing_overrides"]:
            assert runs[role][key] == runs["baseline"][key]
        assert runs[role]["command"][:-1] == runs["baseline"]["command"][:-1]
    assert links["optimized"]["stage_object_sha256"]["prefill"] == links["virtual"]["stage_object_sha256"]["prefill"]
    assert {k: v for k, v in counters["optimized"].items() if k != "host_stalled"} == {
        k: v for k, v in counters["virtual"].items() if k != "host_stalled"}
    elapsed = {role: run["execution_total_performance"] for role, run in runs.items()}
    assert elapsed["baseline"]["cycles"] - elapsed["virtual"]["cycles"] == (
        elapsed["baseline"]["instructions"] - elapsed["virtual"]["instructions"]
        + counters["baseline"]["host_stalled"] - counters["virtual"]["host_stalled"])
    for name, first, second in [("optimized_comparison.json", "baseline", "optimized"), ("virtual_comparison.json", "optimized", "virtual")]:
        report = load(name)
        for field, role in [("reference", first), ("candidate", second)]:
            run = runs[role]
            assert report[field + "_cycles"] == run["execution_total_performance"]["cycles"]
            assert report[field + "_run_sha256"] == sha(data[role + "/run/mx_session_run.json"])
            assert report[field + "_link_sha256"] == sha(data[role + "/link/mx_session_link.json"])
            for index, program in enumerate(report["programs"]):
                step = sum(r["cycles"] for r in run["step_performance"] if r["program"] == index)
                prep = sum(r["cycles"] for r in run["preparation_performance"] if r["program"] == index)
                assert program[field + "_cycles"] == step + prep
                assert program[field + "_step_cycles"] == step and program[field + "_preparation_cycles"] == prep
        assert report["candidate_over_reference"] == report["candidate_cycles"] / report["reference_cycles"]
    print("PASS: three complete 128+32 sessions, all 99 hashes and 33 full logit tensors identical.")
    print("Captured FP32 diagnostics recomputed; no independent cached-source quality threshold established.")


if __name__ == "__main__":
    main()
