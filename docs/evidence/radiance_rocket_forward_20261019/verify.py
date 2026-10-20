"""Reopen complete model observations, reference logits and diagnostic profiles."""

import hashlib
import importlib.util
import io
import json
import re
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def unpack(name, record):
    raw = (HERE / name).read_bytes()
    assert len(raw) == record["bytes"] and sha(raw) == record["sha256"]
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as archive:
        for item in archive:
            path = PurePosixPath(item.name)
            assert item.isfile() and not path.is_absolute() and ".." not in path.parts
            assert item.name not in files
            files[item.name] = archive.extractfile(item).read()
    assert set(files) == set(record["members"])
    for name, raw in files.items():
        assert len(raw) == record["members"][name]["bytes"]
        assert sha(raw) == record["members"][name]["sha256"]
    return files


def port_outputs(files, phase_timing):
    receipt = json.loads(files["source/receipt.json"])
    checks = json.loads(files["source/checks.json"])
    console = files["source/spike.log"].decode()
    assert "MODEL DEADLOCK" not in console
    assert console.splitlines().count("PORT_DONE") == 1
    assert re.findall(r"^PORT_PHASES_COMPLETED ([0-9]+)$", console, re.M) == ["266"]
    assert re.findall(r"^PORT_TOTAL ([0-9]+) guards=0$", console, re.M) == [
        str(receipt["cycles"])
    ]
    observed = {i: bytearray() for i in range(len(checks))}
    for line in console.splitlines():
        if line.startswith("PORT_BYTES "):
            _, index, offset, count, *words = line.split()
            index, offset, count = int(index), int(offset), int(count)
            assert index in observed and offset == len(observed[index])
            assert 1 <= count <= 32 and count == len(words)
            assert all(re.fullmatch("[0-9a-f]{2}", word) for word in words)
            observed[index].extend(int(word, 16) for word in words)
    for index, check in enumerate(checks):
        assert len(observed[index]) == check["bytes"]
        assert sha(observed[index]) == receipt["results"][index]["observed_sha256"]
    phases = re.findall(r"^PORT_PHASE ([0-9]+) ([0-9]+) ([0-9]+)$", console, re.M)
    assert len(phases) == (266 if phase_timing else 0)
    if phases:
        assert [int(x[0]) for x in phases] == list(range(266))
        assert all(int(x[2]) > 0 for x in phases)
    return receipt, {i: bytes(raw) for i, raw in observed.items()}, phases


def quality(reference, candidate):
    def logsoftmax(x):
        x = x.astype(np.float64)
        assert np.isfinite(x).all()
        x -= x.max(-1, keepdims=True)
        return x - np.log(np.exp(x).sum(-1, keepdims=True))

    a, b = logsoftmax(reference), logsoftmax(candidate)
    kl = np.maximum(0, (np.exp(a) * (a - b)).sum(-1))
    return {
        "mean_kl_nats": float(kl.mean()),
        "worst_kl_nats": float(kl.max()),
        "top1_agreement": int((reference.argmax(-1) == candidate.argmax(-1)).sum()),
        "positions": len(reference),
    }


def main():
    manifest = json.loads((HERE / "artifacts.json").read_text())
    data = unpack("forward.tar.xz", manifest["archives"]["forward.tar.xz"])
    profiles = unpack("profiles.tar.xz", manifest["archives"]["profiles.tar.xz"])
    source_receipt, outputs, _ = port_outputs(data, False)
    profile_receipt, profile_outputs, phases = port_outputs(profiles, True)
    assert outputs == profile_outputs
    assert data["source/phases.bin"] == profiles["source/phases.bin"]
    for index, raw in outputs.items():
        assert data[f"source/observed{index}.bin"] == raw
    for prefix, receipt, files in (
        ("source/", source_receipt, data),
        ("source/", profile_receipt, profiles),
    ):
        assert receipt["status"] == "completed_requires_matched_comparison"
        assert receipt["source_revision"] == manifest["source_revision"]
        assert receipt["admission"]["status"] == "accepted"
        assert receipt["admission"]["elf_sha256"] == receipt["elf_sha256"]
        assert sha(files[prefix + "driver.c"]) == receipt["port_source_sha256"]
    # Derive source work independently from its actual phase program.
    program = np.frombuffer(data["source/phases.bin"], "<u4").reshape(-1, 32)
    assert len(program) == 266 and [int(x[1]) for x in phases] == program[:, 1].tolist()
    tiles = loops = matrix_bytes = exp_elements = exp_commands = 0
    for words in program:
        d = [int(x) for x in words]
        if d[0] == 1:
            count = d[14] * d[15] * d[16] * d[17]
            loops += count
            tiles += count * (d[18] // 16) * 4 * (d[19] // 16)
            matrix_bytes += count * (d[18] + 64) * d[19]
        elif d[0] == 4:
            exp_elements += d[6] * d[5] * d[4]
            exp_commands += d[6] * d[5] * ((d[4] + 8191) // 8192)
        elif d[0] == 7:
            exp_elements += d[6] * d[5]
            exp_commands += d[6] * ((d[5] + 8191) // 8192)
    for files in (data, profiles):
        perf = files["source/perf.txt"].decode()
        assert re.findall(
            r"mesh: ([0-9]+) tiles, ([0-9]+) busy cycles; VPU: ([0-9]+) commands", perf
        ) == [(str(tiles), str(tiles * 16), str(exp_commands))]
        assert re.findall(r"commands, ([0-9]+) loops", perf) == [str(loops)]
        assert re.findall(r"load: ([0-9]+) bytes", perf) == [
            str(matrix_bytes + 2 * exp_elements)
        ]
        assert re.findall(r"scales: ([0-9]+) bytes", perf) == [str(matrix_bytes // 32)]
    comparison = json.loads(data["comparison.json"])
    control = HERE.parent / "matched_compact_batch_rhs_forward_20261019"
    assert (
        sha((control / "artifacts.json").read_bytes())
        == manifest["compiler_control_manifest_sha256"]
    )
    control_manifest = json.loads((control / "artifacts.json").read_text())
    archive = module("frozen_compiler_control", control / "verify.py")
    retained = archive.composed_evidence(control, control_manifest, set())
    parser = module(
        "frozen_full_output_parser", HERE.parent / "matched_forward_20261018/verify.py"
    )
    compiler, metrics = parser.parse(retained["candidate/spike.log"])
    profile_compiler, profile_metrics = parser.parse(profiles["compiler/spike.log"])
    assert np.array_equal(compiler.view(np.uint32), profile_compiler.view(np.uint32))
    reference = np.load(
        io.BytesIO(data["capture/golden.npy"]), allow_pickle=False
    ).reshape(160, 32000)
    source = np.frombuffer(outputs[0], "<f4").reshape(160, 32000)
    assert sha(outputs[0]) == comparison["source_logits_sha256"]
    assert sha(compiler.astype("<f4").tobytes()) == comparison["compiler_logits_sha256"]
    for name in ("inputs.npz", "golden.npy"):
        assert sha(data["capture/" + name]) == comparison["capture_files_sha256"][name]
    for name, rows in (
        ("quality_all_160", slice(None)),
        ("quality_late_33", slice(-33, None)),
    ):
        for role, values in (("source", source), ("compiler", compiler)):
            actual = quality(reference[rows], values[rows])
            expected = comparison[name][role]
            for key in actual:
                assert abs(actual[key] - expected[key]) < 1e-12
        a, b = comparison[name]["source"], comparison[name]["compiler"]
        gate = (
            b["mean_kl_nats"] <= 1.05 * a["mean_kl_nats"]
            and b["worst_kl_nats"] <= 1.05 * a["worst_kl_nats"]
            and b["top1_agreement"] >= a["top1_agreement"]
        )
        assert gate == comparison[name]["passed"]
    assert comparison["quality_late_33"]["passed"]
    assert not comparison["quality_all_160"]["passed"]
    assert comparison["qualified_speedup"] is None
    assert source_receipt["cycles"] == comparison["source_cycles"] == 7082331012
    assert int(metrics["cycles"]) == comparison["compiler_cycles"] == 10326818367
    diagnostic = json.loads(profiles["kernel_profile_comparison.json"])
    assert diagnostic["source_profile_cycles"] == profile_receipt["cycles"]
    assert diagnostic["compiler_profile_cycles"] == int(profile_metrics["cycles"])
    assert diagnostic["clean_e2e_speedup"] is None
    assert [x["cycles"] for x in diagnostic["source_phases"]] == [
        int(x[2]) for x in phases
    ]
    names = [
        "init",
        "norm1",
        "qkv",
        "rope",
        "qk",
        "softmax",
        "pv",
        "oquant",
        "oproj",
        "norm2",
        "gateup",
        "silu",
        "down",
        "final",
        "lmhead",
    ]
    by_phase = Counter()
    for _, operation, ticks in phases:
        by_phase[names[int(operation) & 255]] += int(ticks)
    assert dict(by_phase) == diagnostic["source_cycles_by_phase_kind"]
    assert sha(data["source/receipt.json"]) == comparison["source_receipt_sha256"]
    # Recompute compiler attribution from actual intervals and source-bound aliases.
    catalog_raw = profiles["compiler/device_catalog/mx_catalog.json"]
    catalog = json.loads(catalog_raw)
    entries = json.loads(profiles["compiler/device/catalog_binding.json"])[
        "entry_to_kernel"
    ]
    for kernel in catalog["kernels"]:
        for field, key in (
            ("activation_preparation", "consumer_symbol"),
            ("private_bf16_transport", "symbol"),
            ("prepared_private_bf16_transport", "symbol"),
        ):
            if kernel.get(field):
                alias = kernel[field][key]
                assert entries.setdefault(alias, kernel["symbol"]) == kernel["symbol"]
    repeated = json.loads(
        profiles["compiler/device_host_abi/mx_batch_rhs_reuse_rewrite.json"]
    )
    assert repeated["catalog_sha256"] == sha(catalog_raw)
    for consumer in repeated["consumers"]:
        kernel = entries[consumer["symbol"]]
        assert entries.setdefault(consumer["alternative"], kernel) == kernel
    table = json.loads(profiles["compiler/op_profile_table.json"])
    samples = re.findall(
        r"^PROF ([0-9]+) ([0-9]+) ([0-9]+)$",
        profiles["compiler/spike.log"].decode(),
        re.M,
    )
    observed = {int(i): (int(t), int(h)) for i, t, h in samples}
    assert len(observed) == len(samples) and all(0 <= i < len(table) for i in observed)
    attributed = sum(t for t, _ in observed.values())
    assert attributed == int(profile_metrics["prof_total_ticks"])
    assert (
        sum(h for _, h in observed.values()) == int(profile_metrics["prof_marks"]) - 1
    )
    assert attributed <= int(profile_metrics["cycles"])
    owners, matrix_calls = Counter(), Counter()
    for row in table:
        callee = (row.get("callee") or "").removeprefix("@")
        if callee in entries:
            matrix_calls[entries[callee]] += 1
        ticks, hits = observed.get(row["id"], (0, 0))
        if hits:
            owner = (
                "mx"
                if callee in entries
                else "vpu"
                if callee.startswith("mx_vpu_")
                else "host"
            )
            owners[owner] += ticks
    assert matrix_calls == Counter(b["symbol"] for b in catalog["bindings"])
    attribution = json.loads(profiles["compiler_profile_attribution.json"])
    assert attribution["by_owner"] == dict(owners)
    assert attribution["attributed_cycles"] == attributed
    print(
        json.dumps(
            {
                "passed": True,
                "source_cycles": comparison["source_cycles"],
                "compiler_cycles": comparison["compiler_cycles"],
                "all_5120000_logits_checked": True,
                "profiled_outputs_unchanged": True,
                "full_quality_gate": "failed_mean_KL",
                "qualified_speedup": None,
                "large_ELFs_retained_in_portable_archive": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
