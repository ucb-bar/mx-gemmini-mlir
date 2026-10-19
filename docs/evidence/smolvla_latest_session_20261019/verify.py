"""Recheck completed SmolVLA sessions without importing either compiler."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import mmap
import re
import struct
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np

HERE = Path(__file__).resolve().parent
SCOPE = ("reset_plus_program_state_handoff_plus_existing_step_intervals; "
         "console_and_phase_bookkeeping_excluded")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def metrics(actual, expected):
    assert actual.shape == expected.shape == (1, 50, 32)
    assert actual.dtype == expected.dtype == np.float32
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    lhs, rhs = actual.astype(np.float64), expected.astype(np.float64)
    delta = lhs - rhs
    rmse = float(np.sqrt(np.mean(delta * delta)))
    return {
        "mae": float(np.mean(np.abs(delta))),
        "rmse": rmse,
        "normalized_rmse": rmse / max(float(np.sqrt(np.mean(rhs * rhs))), 1e-300),
        "max_abs": float(np.max(np.abs(delta))),
        "cosine": float(np.clip(np.sum(lhs * rhs) / max(
            float(np.linalg.norm(lhs) * np.linalg.norm(rhs)), 1e-300), -1.0, 1.0)),
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


def verify(build_root=None):
    manifest = json.loads((HERE / "artifacts.json").read_text())
    raw = (HERE / "executions.tar.xz").read_bytes()
    assert sha(raw) == manifest["archive_sha256"]
    assert len(raw) == manifest["archive_bytes"]
    data = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts
            assert member.name not in data
            data[member.name] = archive.extractfile(member).read()
    stored = frozenset(data)
    assert len(stored) == manifest["stored_members"]
    for name, original in manifest["aliases"].items():
        path = PurePosixPath(name)
        assert not path.is_absolute() and ".." not in path.parts
        assert name not in data and original in stored
        data[name] = data[original]
    assert set(data) == set(manifest["members"])
    for name, value in data.items():
        assert sha(value) == manifest["members"][name]["sha256"], name
        assert len(value) == manifest["members"][name]["bytes"], name

    def load(name):
        return json.loads(data[name])

    report = load("comparison.json")
    quality = load("fp32_comparison.json")
    resources = load("resources.json")
    assert resources["scope"] == "observed_final_images_and_literal_allocations_not_peak_memory"
    links, runs, values, counters = {}, {}, {}, {}
    for role, label in [("reference", "latest_control"), ("candidate", "latest_optimized")]:
        lp, rp = f"{role}/link/", f"{role}/run/"
        link, run = load(lp + "mx_session_link.json"), load(rp + "mx_session_run.json")
        links[role], runs[role] = link, run
        assert run["link_receipt_sha256"] == sha(data[lp + "mx_session_link.json"])
        assert run["complete"] is True and run["exit_code"] == 0
        assert run["status"] == "complete_numeric_qualification_pending"
        assert run["expected_events"] == run["observed_events"] == 45
        assert run["timing_model_active"] is True and run["timing_mode"] == "both"
        assert run["timing_config"] == "e4m3_vpu"
        assert run["timing_overrides"] == "mem.host_tracking=0"
        assert run["execution_timing_scope"] == link["execution_timing_scope"] == SCOPE
        assert [p["name"] for p in link["programs"]] == [
            "prefix_encode", "flow_denoise", "action_decode"]
        assert [p["steps"] for p in link["programs"]] == [1, 10, 1]
        assert [p["outputs"] for p in link["programs"]] == [2, 3, 1]
        assert load(f"{role}/job/status.json")["exit_code"] == 0
        live = load(rp + "live_environment.json")
        assert live["command"] == run["command"]
        assert live["environment"]["GEMMINI_MODE"] == "both"
        assert live["environment"]["GEMMINI_PERF_CONFIG"] == "e4m3_vpu"
        assert live["environment"]["GEMMINI_PERF_SET"] == "mem.host_tracking=0"
        assert sha(data[rp + "spike.log"]) == run["log_sha256"]
        console = data[rp + "spike.log"].decode()
        assert console.splitlines().count("DONE") == 1
        assert "METRIC memref_rank_mismatch 0\n" in console
        assert not any(line.startswith(("FATAL", "FAIL ")) for line in console.splitlines())
        lines = [line.split() for line in console.splitlines() if line.startswith("SESSION_")]
        cursor = 0

        def take(kind, width):
            nonlocal cursor
            row = lines[cursor]
            cursor += 1
            assert row[0] == kind and len(row) == width, (role, row, kind)
            return row[1:]

        prep, steps, digests, dumped = [], [], [], {}
        sizes = [[113, 2314240], [6400, 2314240, 4], [6400]]
        for program in [-1, 0, 1, 2]:
            p, cycles, inst = map(int, take("SESSION_PREP_PERF", 4))
            assert p == program and min(cycles, inst) > 0
            prep.append(dict(program=p, cycles=cycles, instructions=inst))
            if program == -1:
                continue
            for step in range(link["programs"][program]["steps"]):
                assert list(map(int, take("SESSION_BEGIN", 3))) == [program, step]
                p, s, cycles, inst = map(int, take("SESSION_PERF", 5))
                assert (p, s) == (program, step) and min(cycles, inst) > 0
                steps.append(dict(program=p, step=s, cycles=cycles, instructions=inst))
                for output, size in enumerate(sizes[program]):
                    p, s, o, n, digest = take("SESSION_SHA", 6)
                    assert list(map(int, [p, s, o, n])) == [program, step, output, size]
                    assert re.fullmatch("[0-9a-f]{64}", digest)
                    digests.append(dict(program=program, step=step, output=output,
                                        bytes=size, sha256=digest))
                    if program > 0 and output == 0:
                        p, s, o, n, encoded = take("SESSION_HEX", 6)
                        assert list(map(int, [p, s, o, n])) == [program, step, 0, size]
                        content = bytes.fromhex(encoded)
                        assert len(content) == size and sha(content) == digest
                        dumped[program, step, output] = content
        total = dict(zip(["cycles", "instructions"], map(int, take("SESSION_TOTAL", 3))))
        execution = dict(zip(["cycles", "instructions"], map(int, take("SESSION_EXECUTION_TOTAL", 3))))
        assert cursor == len(lines)
        assert steps == run["step_performance"] and prep == run["preparation_performance"]
        assert digests == run["output_digests"] and len(digests) == 33
        assert total == run["total_performance"]
        assert execution == run["execution_total_performance"]
        for key in ["cycles", "instructions"]:
            assert total[key] == sum(row[key] for row in steps)
            assert run["preparation_total_performance"][key] == sum(row[key] for row in prep)
            assert execution[key] == total[key] + sum(row[key] for row in prep)
        assert len(run["output_dumps"]) == len(dumped) == 11
        for row in run["output_dumps"]:
            key = row["program"], row["step"], row["output"]
            path = PurePosixPath(row["path"])
            assert not path.is_absolute() and ".." not in path.parts
            content = data[rp + str(path)]
            assert content == dumped[key] and len(content) == row["bytes"] == 6400
            assert sha(content) == row["sha256"]
        assert dumped[1, 9, 0] == dumped[2, 0, 0]
        values[role] = dumped
        for index, stored in enumerate(report["programs"]):
            selected_steps = [r for r in steps if r["program"] == index]
            selected_prep = [r for r in prep if r["program"] == index]
            step_cycles = sum(r["cycles"] for r in selected_steps)
            prep_cycles = sum(r["cycles"] for r in selected_prep)
            assert stored[f"{role}_step_cycles"] == step_cycles
            assert stored[f"{role}_preparation_cycles"] == prep_cycles
            assert stored[f"{role}_cycles"] == step_cycles + prep_cycles
        assert report[f"{role}_cycles"] == execution["cycles"]
        assert report[f"{role}_reset_cycles"] == prep[0]["cycles"]
        assert report[f"{role}_run_sha256"] == sha(data[rp + "mx_session_run.json"])
        assert report[f"{role}_link_sha256"] == sha(data[lp + "mx_session_link.json"])
        admission = load(lp + "admission.json")
        assert sha(data[lp + "admission.json"]) == link["linked_elf_admission"]["report_sha256"]
        assert admission["elf_sha256"] == link["elf_sha256"]
        assert admission["status"] == link["linked_elf_admission"]["status"] == "accepted"
        for path, digest in link["linked_elf_admission_selection"]["source_pins"]:
            assert sha(data["policy/" + Path(path).name]) == digest
        for program in link["programs"]:
            model = data[f'capture/stages/{program["name"]}/model.mlir']
            assert sha(model) == program["capture_model_sha256"]
        layout = link["arena_layout"]
        assert resources["images"][role]["elf_bytes"] == link["elf_bytes"]
        assert resources["images"][role]["elf_sha256"] == link["elf_sha256"]
        assert layout["static_end"] <= layout["weights_base"]
        assert layout["weights_base"] + link["shared_weight_bytes"] <= layout["arena_base"]
        assert layout["arena_base"] + layout["arena_bytes"] <= 0x80000000 + link["mem_bytes"]
        perf = data[rp + "gemmini_perf.txt"]
        assert sha(perf) == run["timing_report_sha256"]
        text = perf.decode()
        patterns = [r"(\d+) commands, (\d+) loops, (\d+) fences", r"mesh: (\d+) tiles, (\d+) busy cycles; VPU: (\d+) commands", r"load: (\d+) bytes in (\d+) Gets .*store: (\d+) reads, (\d+) Puts; scales: (\d+) bytes"]
        counters[role] = tuple(int(n) for pattern in patterns for n in re.search(pattern, text).groups())
        assert "host tracking off" in text
        if build_root is not None:
            image = build_root / f"smolvla_{label}_link/session.elf"
            assert image.stat().st_size == link["elf_bytes"]
            with image.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == link["elf_sha256"]
            assert allocated_sections(image) == resources["images"][role]["allocated_sections"]

    base, other = links["reference"], links["candidate"]
    for key in ["session_root", "programs", "bindings", "profile_sha256", "shared_weight_sha256",
                "shared_weight_bytes", "program_weight_pool", "mem_bytes", "performance_schema",
                "preparation_performance_schema", "execution_timing_scope"]:
        assert base[key] == other[key], key
    for key in ["spike_sha256", "extension_sha256", "timing_mode", "timing_config", "timing_overrides"]:
        assert runs["reference"][key] == runs["candidate"][key], key
    assert runs["reference"]["command"][:-1] == runs["candidate"]["command"][:-1]
    assert runs["reference"]["output_digests"] == runs["candidate"]["output_digests"]
    assert values["reference"] == values["candidate"]
    assert counters["reference"][1] == counters["candidate"][1]  # same matrix loops
    assert counters["reference"][3:6] == counters["candidate"][3:6]  # same mesh and VPU arithmetic
    assert counters["reference"] == (18034641, 446775, 201109, 48141352, 770261632, 142409,
                                     12580427280, 340109841, 81102340, 89721028, 315957264)
    assert counters["candidate"] == (17727441, 446775, 196309, 48141352, 770261632, 142409,
                                     12541105680, 337652241, 78644740, 87263428, 315957264)
    changed = [f"{stage}/{name}" for stage in sorted(base["stage_object_sha256"])
               for name in sorted(base["stage_object_sha256"][stage])
               if base["stage_object_sha256"][stage][name] != other["stage_object_sha256"][stage][name]]
    assert changed == report["changed_stage_objects"]
    assert report["candidate_over_reference"] == report["candidate_cycles"] / report["reference_cycles"]
    assert report["execution_timing_scope"] == SCOPE and report["output_digest_parity"] is True
    for role in ["reference", "candidate"]:
        for stage in ["prefix_encode", "flow_denoise"]:
            probe = load(f"{role}/stages/{stage}/mx_stage_probe.json")
            assert len(probe["shared_activation_rewrite"]["groups"]) == (32 if stage == "prefix_encode" else 24)
            catalog = load(f"{role}/stages/{stage}/device_catalog/mx_catalog.json")
            resource = resources["stages"][role][stage]
            assert resource["allocations"] == probe["arena_estimate"]
            assert sha(data[f"{role}/stages/{stage}/device_catalog/runtime_quant.c"]) == resource["source_quant_sha256"]
            assert resource["source_quant_sha256"] == catalog["compilation"]["source_sha256"]["runtime_quant"]
            assert sha(data[f"{role}/stages/{stage}/device_catalog/runtime_quant.o"]) == resource["quant_object_sha256"]
            assert probe.get("native_tile_policy", "legacy") == "legacy" and not probe.get("virtual_zero_padding")
            if role == "candidate":
                rewrite = load(f"{role}/stages/{stage}/device_host_abi/mx_vpu_elementwise_rewrite.json")
                assert len(rewrite["connected_multiplies"]) == (25 if stage == "prefix_encode" else 0)
                assert len(probe["private_vpu_matrix_transport_rewrite"]["edges"]) == 0
                assert len(load(f"{role}/stages/{stage}/device_host_abi/mx_matrix_transport_rewrite.json")["edges"]) == 0
                assert len(load(f"{role}/stages/{stage}/device_host_abi/mx_connected_sigmoid_rewrite.json")["chains"]) == 0
                assert catalog["integer_group_packing"] is True
    for stage in ["prefix_encode", "flow_denoise"]:
        assert load(f"reference/stages/{stage}/mx_stage_probe.json")["execution_policy"] == load(
            f"candidate/stages/{stage}/mx_stage_probe.json")["execution_policy"]
    assert links["reference"].get("host_math") is None
    host_math = links["candidate"]["host_math"]
    assert report["session_host_math"]["candidate"] == host_math["contract"]
    assert sha(data["toolchain/libm.a"]) == host_math["contract"]["library_sha256"]
    math_prefix = "candidate/link/host_math/"
    assert sha(data[math_prefix + "host_math.c"]) == host_math["source_sha256"]
    assert sha(data[math_prefix + "host_math.o"]) == host_math["object_sha256"]
    recipe_raw = data[math_prefix + "link_attempt_0/compilation_recipe.json"]
    assert sha(recipe_raw) == host_math["link_recipe_sha256"]
    recipe = json.loads(recipe_raw)
    assert recipe["status"] == "completed" and recipe["executable"]["sha256"] == links["candidate"]["elf_sha256"]
    assert sha(data["frozen_tools/link_merlin_session.py"]) == recipe["producer"]["sha256"]
    assert "-Wl,--wrap=expf" in recipe["commands"][-1]["argv"]
    assert set(recipe["link_suppliers"]["symbols"]) == {"expf", "__ieee754_expf"}
    for symbol, selected in recipe["link_suppliers"]["symbols"].items():
        assert selected["selected_input"]["sha256"] == host_math["contract"]["library_sha256"]
        assert recipe["commands"][-1]["argv"].count("-Wl,--trace-symbol=" + symbol) == 1
        definitions = [line.rsplit(": ", 1)[0].split(": ", 1)[1] for line in recipe["link_suppliers"]["stderr"].splitlines()
                       if line.endswith(": definition of " + symbol)]
        assert definitions == [selected["definition"]]
    assert quality["numerical_qualification"] == "not_established"
    assert quality["final_action_matches_flow_bytes"] is True
    assert quality["run_receipt_sha256"] == sha(data["candidate/run/mx_session_run.json"])
    assert quality["link_receipt_sha256"] == sha(data["candidate/link/mx_session_link.json"])
    golden = data["capture/stages/flow_denoise/session_goldens.npz"]
    assert quality["reference_sha256"] == sha(golden)
    with np.load(io.BytesIO(golden), allow_pickle=False) as archive:
        expected = archive["actions"]
    assert expected.shape == (10, 1, 50, 32)
    for step, stored in enumerate(quality["flow_step_metrics"]):
        actual = np.frombuffer(values["candidate"][1, step, 0], dtype="<f4").reshape(1, 50, 32)
        assert stored["step"] == step
        for key, value in metrics(actual, expected[step]).items():
            assert abs(value - stored[key]) < 1e-10, (step, key)
    assert quality["final_action_metrics"] == quality["flow_step_metrics"][-1]
    control_quality = load("control_fp32_comparison.json")
    assert control_quality["flow_step_metrics"] == quality["flow_step_metrics"]
    assert control_quality["final_action_metrics"] == quality["final_action_metrics"]
    assert control_quality["numerical_qualification"] == "not_established"
    print(f"PASS: both complete 1+10+1 sessions, 33 hashes and 11 full tensors identical; "
          f'{report["reference_cycles"]:,} -> {report["candidate_cycles"]:,} cycles')
    print("FP32 diagnostic recomputed; action quality threshold remains unestablished.")
    print("Large ELF hashes rechecked." if build_root else "ELFs remain external; admission receipts checked, not replayed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, help="optional original mx_model_goal_20261018 directory")
    verify(parser.parse_args().build_root)
