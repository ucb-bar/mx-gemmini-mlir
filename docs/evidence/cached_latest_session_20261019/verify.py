"""Reopen complete cached-session timing, output streams and FP32 diagnostics."""

import argparse
import hashlib
import io
import json
import mmap
import struct
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


def allocated_sections(path):
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as raw:
        assert raw[:6] == b"\x7fELF\x02\x01"
        offset = struct.unpack_from("<Q", raw, 40)[0]
        size, count, names = struct.unpack_from("<HHH", raw, 58)
        assert size == 64
        rows = [struct.unpack_from("<IIQQQQIIQQ", raw, offset + index * size) for index in range(count)]
        strings = rows[names]
        table = raw[strings[4]:strings[4] + strings[5]]
        return {table[row[0]:table.index(b"\0", row[0])].decode():
                {"bytes": row[5], "type": row[1], "flags": row[2]} for row in rows if row[2] & 2 and row[5]}



def archive_members(directory, pinned=None):
    manifest_raw = (directory / "artifacts.json").read_bytes()
    manifest = json.loads(manifest_raw)
    if pinned is not None:
        assert sha(manifest_raw) == pinned["manifest_sha256"]
        assert manifest["archive_sha256"] == pinned["archive_sha256"]
        assert manifest["archive_bytes"] == pinned["archive_bytes"]
    raw = (directory / "executions.tar.xz").read_bytes()
    assert sha(raw) == manifest["archive_sha256"] and len(raw) == manifest["archive_bytes"]
    data = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts
            assert member.name not in data
            data[member.name] = archive.extractfile(member).read()
    stored = frozenset(data)
    assert len(stored) == manifest.get("stored_members", len(stored))
    for name, original in manifest.get("aliases", {}).items():
        path = PurePosixPath(name)
        assert not path.is_absolute() and ".." not in path.parts
        assert name not in data and original in stored
        data[name] = data[original]
    assert set(data) == set(manifest["members"])
    for name, value in data.items():
        assert sha(value) == manifest["members"][name]["sha256"], name
        assert len(value) == manifest["members"][name]["bytes"], name
    return manifest, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, help="optional retained mx_model_goal_20261018 directory")
    args = parser.parse_args()
    manifest, data = archive_members(HERE)
    dependency = manifest["baseline_dependency"]
    assert dependency["directory"] == "cached_session_reuse_20261019"
    assert dependency["role"] == "virtual"
    _, previous = archive_members(HERE.parent / dependency["directory"], dependency)
    for name, raw in previous.items():
        if name.startswith("virtual/"):
            target = "reference/" + name.removeprefix("virtual/")
        elif name.startswith("capture/"):
            target = name
        else:
            continue
        assert target not in data
        data[target] = raw

    def load(name):
        return json.loads(data[name])

    resources = load("resources.json")
    assert resources["scope"] == "observed_final_images_and_literal_allocations_not_peak_memory"
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
    for role, label in [("reference", "virtual"), ("candidate", "latest")]:
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
        assert summary["positions"] == 33
        assert abs(summary["cosine_min"] - min(r["cosine"] for r in measured)) < 1e-10
        for stage in ["prefill", "decode"]:
            assert sha(data[f"{role}/stages/{stage}/model.o"]) == link["stage_object_sha256"][stage]["model.o"]
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
        resource = resources["images"][role]
        assert resource["bytes"] == link["elf_bytes"] and resource["sha256"] == link["elf_sha256"]
        assert resource["directory"] == f"cached_session_{label}_link"
        for stage in ["prefill", "decode"]:
            probe = load(f"{role}/stages/{stage}/mx_stage_probe.json")
            stage_resource = resources["stages"][role][stage]
            assert stage_resource["allocations"] == probe["arena_estimate"]
            assert stage_resource["arena_mb"] == probe["arena_mb"] == 2048
            assert stage_resource["arena_base"] == probe["spike_arena_base"]
            assert link["static_malloc_estimates"][stage] == probe["arena_estimate"]
            catalog = load(f"{role}/stages/{stage}/device_catalog/mx_catalog.json")
            assert stage_resource["native_workspace_bytes"] == catalog["compilation"]["shared_workspace_bytes"]
            assert probe["profile_sha256"] == catalog["profile_sha256"] == link["profile_sha256"]
            if role == "candidate":
                for name, digest in link["stage_object_sha256"][stage].items():
                    archived = f"{role}/stages/{stage}/" + name
                    if archived in data:
                        assert sha(data[archived]) == digest
                    else:
                        external_path = str(Path(link["elf"]).parent.parent / stage_resource["directory"] / name)
                        assert manifest["external_images_and_large_headers"][external_path]["sha256"] == digest
                    if args.build_root:
                        original = args.build_root / stage_resource["directory"] / name
                        with original.open("rb") as stream:
                            assert hashlib.file_digest(stream, "sha256").hexdigest() == digest
                quant = data[f"{role}/stages/{stage}/device_catalog/runtime_quant.c"]
                assert sha(quant) == catalog["compilation"]["source_sha256"]["runtime_quant"]
        if args.build_root:
            image = args.build_root / f"cached_session_{label}_link/session.elf"
            assert image.stat().st_size == link["elf_bytes"]
            with image.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == link["elf_sha256"]
            assert allocated_sections(image) == resource["allocated_sections"]
        print(role, execution["cycles"], "cycles; full protocol, taps and diagnostics checked")
    assert outputs["candidate"] == outputs["reference"]
    assert runs["candidate"]["output_digests"] == runs["reference"]["output_digests"]
    assert diagnostic["runs"]["candidate"]["rows"] == diagnostic["runs"]["reference"]["rows"]
    for key in ["programs", "bindings", "profile_sha256", "shared_weight_sha256", "shared_weight_bytes", "mem_bytes"]:
        assert links["candidate"][key] == links["reference"][key]
    for key in ["spike_sha256", "extension_sha256", "timing_mode", "timing_config", "timing_overrides"]:
        assert runs["candidate"][key] == runs["reference"][key]
    assert runs["candidate"]["command"][:-1] == runs["reference"]["command"][:-1]
    for stage in ["prefill", "decode"]:
        first = load(f"reference/stages/{stage}/mx_stage_probe.json")
        second = load(f"candidate/stages/{stage}/mx_stage_probe.json")
        for key in ["capture_model_sha256", "checkpoint_weights_sha256", "execution_policy", "precision_overrides"]:
            assert first[key] == second[key], (stage, key)
    report = load("comparison.json")
    assert report["output_digest_parity"] is True and report["execution_timing_scope"] == SCOPE
    for field in ["reference", "candidate"]:
        run, link = runs[field], links[field]
        assert report[field + "_cycles"] == run["execution_total_performance"]["cycles"]
        assert report[field + "_run_sha256"] == sha(data[field + "/run/mx_session_run.json"])
        assert report[field + "_link_sha256"] == sha(data[field + "/link/mx_session_link.json"])
        assert report[field + "_reset_cycles"] == run["preparation_performance"][0]["cycles"]
        assert report["session_host_math"][field] == (link.get("host_math") or {}).get("contract")
        for index, program in enumerate(report["programs"]):
            assert program["name"] == link["programs"][index]["name"]
            assert program["steps"] == link["programs"][index]["steps"]
            step = sum(r["cycles"] for r in run["step_performance"] if r["program"] == index)
            prep = sum(r["cycles"] for r in run["preparation_performance"] if r["program"] == index)
            assert program[field + "_cycles"] == step + prep
            assert program[field + "_step_cycles"] == step and program[field + "_preparation_cycles"] == prep
    for key in ["profile_sha256", "execution_timing_scope"]:
        assert report[key] == links["candidate"][key]
    for key in ["spike_sha256", "extension_sha256", "timing_config", "timing_overrides"]:
        assert report[key] == runs["candidate"][key]
    changed = sorted(stage + "/" + name for stage, objects in links["candidate"]["stage_object_sha256"].items()
                     for name, digest in objects.items() if links["reference"]["stage_object_sha256"][stage][name] != digest)
    assert report["changed_stage_objects"] == changed
    assert report["candidate_over_reference"] == report["candidate_cycles"] / report["reference_cycles"]
    elapsed = {role: run["execution_total_performance"] for role, run in runs.items()}
    assert elapsed["reference"]["cycles"] - elapsed["candidate"]["cycles"] == (
        elapsed["reference"]["instructions"] - elapsed["candidate"]["instructions"]
        + counters["reference"]["host_stalled"] - counters["candidate"]["host_stalled"])
    assert links["reference"].get("host_math") is None
    host_math = links["candidate"]["host_math"]
    assert sha(data["toolchain/libm.a"]) == host_math["contract"]["library_sha256"]
    prefix = "candidate/link/host_math/"
    assert sha(data[prefix + "host_math.c"]) == host_math["source_sha256"]
    assert sha(data[prefix + "host_math.o"]) == host_math["object_sha256"]
    compile_raw = data[prefix + "compilation_recipe.json"]
    assert sha(compile_raw) == host_math["compile_recipe_sha256"]
    compile_recipe = json.loads(compile_raw)
    assert compile_recipe["status"] == "completed"
    assert compile_recipe["object"]["sha256"] == host_math["object_sha256"]
    assert sha(data[prefix + "guard.json"]) == compile_recipe["preparation"]["guard"]["sha256"]
    assert load(prefix + "guard.json") == host_math["contract"]
    recipe_path = PurePosixPath(host_math["link_recipe"])
    assert recipe_path.parts[-3:] == ("host_math", "link_attempt_1", "compilation_recipe.json")
    recipe_raw = data[prefix + "link_attempt_1/compilation_recipe.json"]
    assert sha(recipe_raw) == host_math["link_recipe_sha256"]
    recipe = json.loads(recipe_raw)
    assert recipe["status"] == "completed" and recipe["executable"]["sha256"] == links["candidate"]["elf_sha256"]
    assert sha(data["frozen_tools/link_merlin_session.py"]) == recipe["producer"]["sha256"] == compile_recipe["producer"]["sha256"]
    assert "-Wl,--wrap=expf" in recipe["commands"][-1]["argv"]
    assert set(recipe["link_suppliers"]["symbols"]) == {"expf", "__ieee754_expf"}
    for symbol, selected in recipe["link_suppliers"]["symbols"].items():
        assert selected["selected_input"]["sha256"] == host_math["contract"]["library_sha256"]
        assert recipe["commands"][-1]["argv"].count("-Wl,--trace-symbol=" + symbol) == 1
        definitions = [line.rsplit(": ", 1)[0].split(": ", 1)[1] for line in recipe["link_suppliers"]["stderr"].splitlines()
                       if line.endswith(": definition of " + symbol)]
        assert definitions == [selected["definition"]]
    for stage in ["prefill", "decode"]:
        assert host_math["stage_guards"][stage] == host_math["contract"]
        assert load(f"candidate/stages/{stage}/mx_stage_probe.json")["host_math_guard"] == host_math["contract"]
    print("Counters, wider process scope:", json.dumps(counters, sort_keys=True))
    print("PASS: two complete 128+32 sessions, 99 hashes and 33 full logit tensors identical.")
    print("Captured FP32 diagnostics recomputed; no independent cached-source quality threshold established.")


if __name__ == "__main__":
    main()
