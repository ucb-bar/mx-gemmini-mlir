"""Check complete repeated-RHS forward execution and unchanged independent-source gates."""

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import mmap
import re
import struct
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_archive(folder, manifest):
    raw = (folder / "executions.tar.xz").read_bytes()
    assert sha(raw) == manifest["archive_sha256"] and len(raw) == manifest["archive_bytes"]
    data = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            assert member.isfile() and not name.is_absolute() and ".." not in name.parts
            assert member.name not in data
            data[member.name] = archive.extractfile(member).read()
    assert set(data) == set(manifest["members"])
    for name, raw in data.items():
        assert sha(raw) == manifest["members"][name]["sha256"]
        assert len(raw) == manifest["members"][name]["bytes"]
    return data


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path)
    args = parser.parse_args()
    manifest = json.loads((HERE / "artifacts.json").read_text())
    data = checked_archive(HERE, manifest)
    dependency = manifest["baseline_dependency"]
    previous = HERE.parent / dependency["directory"]
    previous_raw = (previous / "artifacts.json").read_bytes()
    assert sha(previous_raw) == dependency["manifest_sha256"]
    previous_manifest = json.loads(previous_raw)
    assert previous_manifest["archive_sha256"] == dependency["archive_sha256"]
    assert previous_manifest["archive_bytes"] == dependency["archive_bytes"]
    assert dependency["role"] == "candidate"
    retained = checked_archive(previous, previous_manifest)
    for name, raw in retained.items():
        if name.startswith("candidate/"):
            alias = "baseline/" + name.removeprefix("candidate/")
        elif name.startswith("frozen_candidate_quant/"):
            alias = "frozen_baseline_quant/" + name.removeprefix("frozen_candidate_quant/")
        elif name.startswith("capture/") or name == "host_math_guard.json":
            alias = name
        else:
            continue
        assert alias not in data
        data[alias] = raw
    reader_path = HERE.parent / "matched_forward_20261018/verify.py"
    assert sha(reader_path.read_bytes()) == manifest["reader_sha256"]
    spec = importlib.util.spec_from_file_location("retained_forward_reader", reader_path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)

    def load(name):
        return json.loads(data[name])

    report = load("comparison.json")
    resources = load("resources.json")
    assert resources["scope"] == "actual_images_sections_and_literal_allocations_not_peak_memory"
    capture = load("capture/capture_receipt.json")
    assert report["capture_receipt_sha256"] == sha(data["capture/capture_receipt.json"])
    assert capture["artifacts"]["model.mlir"]["sha256"] == sha(data["capture/model.mlir"])
    assert capture["artifacts"]["inputs.npz"]["sha256"] == sha(data["capture/inputs.npz"])
    source_root = HERE.parent / "nicolas_tinyllama_onepass160_same_input_20261014"
    source_manifest = json.loads((source_root / "matched_archive_manifest.json").read_text())
    for name, digest in source_manifest["files"].items():
        assert sha((source_root / name).read_bytes()) == digest
    boundary = HERE.parent / "nicolas_tinyllama_lookup_equal_boundary_20261015"
    console = gzip.decompress((boundary / "source_spike.log.gz").read_bytes())
    assert sha(console) == report["source_boundary"]["source_log_sha256"]
    assert sha((boundary / "fair_comparison.json").read_bytes()) == report["source_boundary"]["fair_comparison_sha256"]
    assert sha((boundary / "run_receipt.json").read_bytes()) == report["source_boundary"]["source_run_receipt_sha256"]
    assert [int(x, 16) for x in re.findall(rb"^stage_cycles64 0x([0-9a-f]{16})$", console, re.M)] == [report["source_cycles"]] == [12762360525]
    reference = np.load(source_root / "fp32_late_logits.npy", allow_pickle=False)
    top = np.load(source_root / "fp32_argmax_160.npy", allow_pickle=False).reshape(-1)
    source_words = np.fromfile(source_root / "source_reference_late_bf16.bin", dtype="<u2")
    source = (source_words.astype("<u4") << 16).view("<f4").reshape(33, 32000)
    arrays = {}
    for label, row in report["records"].items():
        prefix = label + "/"
        values, observed = reader.parse(data[prefix + "spike.log"])
        arrays[label] = values
        assert np.isfinite(values).all()
        assert sha(data[prefix + "spike.log"]) == row["console_sha256"]
        assert sha(values.astype("<f4").tobytes()) == row["output_words_sha256"]
        assert int(observed["cycles"]) == row["cycles"]
        assert observed["memref_rank_mismatch"] == "0" and not row["timing_instrumented"]
        assert int((values.argmax(-1) == top).sum()) == row["top1_agreement_all_160"] == 133
        receipt = load(prefix + "mx_stage_probe.json")
        assert receipt.get("host_math_guard") == load("host_math_guard.json")
        validation = load(prefix + "mx_stage_output_validation.json")
        recipe = load(prefix + "compilation_recipe.json")
        admission = load(prefix + "admission.json")
        assert sha(data[prefix + "mx_stage_probe.json"]) == row["stage_receipt_sha256"]
        assert sha(data[prefix + "mx_stage_output_validation.json"]) == row["complete_output_validation_sha256"]
        assert validation["validated"] and validation["original_stage_receipt_sha256"] == row["stage_receipt_sha256"]
        assert validation["console_sha256"] == row["console_sha256"]
        assert receipt["spike"]["exit_code"] == 0 and receipt["spike"]["done"]
        assert load(prefix + "job/status.json")["exit_code"] == 0
        assert observed["build_hash"] == receipt["build_hash"]
        assert recipe["status"] == "completed" and admission["status"] == "accepted"
        assert recipe["executable"]["sha256"] == admission["elf_sha256"] == row["elf_sha256"]
        assert sha(data[prefix + "compilation_recipe.json"]) == row["recipe_sha256"]
        assert sha(data[prefix + "admission.json"]) == row["admission_sha256"]
        assert sha(data[prefix + "live_environment.json"]) == row["live_environment_sha256"]
        live = load(prefix + "live_environment.json")
        assert live["environment"]["GEMMINI_MODE"] == "both"
        assert live["environment"]["GEMMINI_PERF_CONFIG"] == "e4m3_vpu"
        assert live["environment"]["GEMMINI_PERF_SET"] == "mem.host_tracking=0"
        assert sha(data[prefix + "device_catalog/mx_catalog.json"]) == row["catalog_sha256"]
        assert sha(data[prefix + "device_catalog/runtime_quant.o"]) == "684996bdf505131d7a4f61b8a0b504830afedcdc0306bad3d9dcbda242019fd4"
        frozen = "frozen_candidate_quant/" if label == "candidate" else "frozen_baseline_quant/"
        assert receipt["native_tile_policy"] == "movement"
        assert receipt["vpu_matrix_transport"] is True
        assert receipt.get("reuse_batch_rhs", False) == (label == "candidate")
        resource = resources["builds"][label]
        assert resource["elf_bytes"] == row["elf_bytes"]
        assert resource["elf_sha256"] == row["elf_sha256"]
        assert resource["literal_allocations"] == row["arena_estimate"] == receipt["arena_estimate"]
        assert resource["arena_mb"] == receipt["arena_mb"] == 2048
        assert resource["arena_base"] == receipt["spike_arena_base"] == 0xd0000000
        assert resource["shared_workspace_source_sha256"] == sha(data[prefix + "device_catalog/shared_workspace.c"])
        assert resource["quant_object_sha256"] == sha(data[prefix + "device_catalog/runtime_quant.o"])
        assert sha(data[frozen + "host_math.o"]) in {item["sha256"] for item in recipe["commands"][-1]["inputs"]}
        assert "-Wl,--wrap=expf" in recipe["commands"][-1]["argv"]
        assert set(recipe["link_suppliers"]["symbols"]) == {"expf", "__ieee754_expf"}
        catalog = load(prefix + "device_catalog/mx_catalog.json")
        assert sha(data[frozen + "runtime_quant.c"]) == catalog["compilation"]["source_sha256"]["runtime_quant"]
        patterns = [
            (r"gemmini perf \[e4m3_vpu\]: (\d+) commands, (\d+) loops, (\d+) fences",
             ["commands", "loops", "fences"]),
            (r"mesh: (\d+) tiles, (\d+) busy cycles; VPU: (\d+) commands",
             ["mesh_tiles", "mesh_busy_cycles", "vpu_commands"]),
            (r"load: (\d+) bytes .*scales: (\d+) bytes", ["load_bytes", "scale_bytes"]),
        ]
        for pattern, keys in patterns:
            matches = re.findall(pattern, data[prefix + "spike.log"].decode())
            assert len(matches) == 1
            for key, count in zip(keys, matches[0], strict=True):
                assert row["performance"][key] == int(count)
        rewrite = load(prefix + "device_host_abi/mx_vpu_elementwise_rewrite.json")
        assert len(rewrite["connected_multiplies"]) == 45
        assert receipt.get("prepare_vpu_broadcast_operands") is True
        assert sha(data[prefix + "device_host_abi/mx_vpu_elementwise.mlir"]) == rewrite["output_sha256"]
        for kind, logits in [("source", source), ("compiler", values[-33:])]:
            for key, value in reader.metrics(reference, logits).items():
                assert abs(row["quality_late_33"][kind][key] - value) < 1e-12
        original, actual = row["quality_late_33"]["source"], row["quality_late_33"]["compiler"]
        assert actual["mean_kl_nats"] <= 1.05 * max(original["mean_kl_nats"], 1e-6)
        assert actual["worst_kl_nats"] <= 1.05 * max(original["worst_kl_nats"], 1e-6)
        assert actual["top1_agreement"] >= original["top1_agreement"]
        if args.build_root is not None:
            name = resources["builds"][label]["directory"]
            image = args.build_root / name / "model.elf"
            assert image.stat().st_size == row["elf_bytes"]
            assert allocated_sections(image) == resources["builds"][label]["allocated_sections"]
            with image.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == row["elf_sha256"]
        print(label, row["cycles"], "cycles; full words and unchanged quality gate checked")
    catalogs = [load(label + "/device_catalog/mx_catalog.json") for label in ("baseline", "candidate")]
    assert len(catalogs[0]["kernels"]) == len(catalogs[1]["kernels"]) == 157
    for before, after in zip(catalogs[0]["kernels"], catalogs[1]["kernels"], strict=True):
        assert all(before[key] == after[key] for key in ("symbol", "precision", "dimensions", "padded_shape_mnk", "tile_mnk"))
    rewrite = load("candidate/device_host_abi/mx_vpu_matrix_transport_rewrite.json")
    assert len(rewrite["edges"]) == 44
    assert rewrite == load("candidate/mx_stage_probe.json")["private_vpu_matrix_transport_rewrite"]
    assert rewrite["catalog_sha256"] == report["records"]["candidate"]["catalog_sha256"]
    assert sha(data["candidate/device_host_abi/mx_vpu_matrix_transport.mlir"]) == rewrite["output_sha256"]
    base_catalog, rhs_catalog = catalogs
    assert all({k: v for k, v in after.items() if k != "batch_rhs_reuse"} == before
               for before, after in zip(base_catalog["bindings"], rhs_catalog["bindings"], strict=True))
    x, y = (value["compilation"]["source_sha256"] for value in catalogs)
    assert {key for key in x.keys() & y.keys() if x[key] != y[key]} == {"shared_activation_shims", "checkpoint_weights"}
    additions = {row["batch_rhs_reuse"]["kernel_symbol"] for row in rhs_catalog["bindings"] if "batch_rhs_reuse" in row}
    assert y.keys() - x.keys() == additions and len(additions) == 2
    for name in additions:
        assert sha(data["candidate/device_catalog/" + name + ".c"]) == y[name]
    for before, after in zip(base_catalog["kernels"], rhs_catalog["kernels"], strict=True):
        for key in ("codes_sha256", "scales_sha256", "workspace_bytes", "schedule_sha256", "instruction_set_sha256"):
            assert before.get(key) == after.get(key)
    assert rhs_catalog["batch_rhs_reuse"] == "experimental_exact_contiguous_repetition"
    source = rhs_catalog["batch_rhs_source"]
    assert source["sha256"] == rhs_catalog["source_sha256"]
    assert sha(data["candidate/device_catalog/" + source["file"]]) == source["sha256"]
    rhs = load("candidate/device_host_abi/mx_batch_rhs_reuse_rewrite.json")
    assert rhs["selected_calls"] == 44 and len(rhs["consumers"]) == 2
    assert rhs["catalog_sha256"] == report["records"]["candidate"]["catalog_sha256"]
    assert rhs["sidecar_sha256"] == sha(data["candidate/device_signatures.json"])
    assert rhs["source_sha256"] == sha(data["candidate/device_host_abi/mx_vpu_matrix_transport.mlir"])
    assert rhs["output_sha256"] == sha(data["candidate/device_host_abi/mx_batch_rhs_reuse.mlir"])
    for consumer in rhs["consumers"]:
        assert consumer["argument"] == 1 and consumer["calls"] == 22
        assert len(consumer["input_proofs"]) == 22
        for proof in consumer["input_proofs"]:
            assert proof["contiguous_batch_repeat"] == 8 and proof["shape"][0] == 32
    endpoints = [row["batch_rhs_reuse"] for row in rhs_catalog["bindings"] if "batch_rhs_reuse" in row]
    assert len(endpoints) == 44
    for endpoint in endpoints:
        assert endpoint["completion"] == "synchronous_native_issuer"
        assert endpoint["lifetime"] == "private_serial_batch_call"
        assert endpoint["repeat"] == endpoint["source_proof"]["contiguous_batch_repeat"] == 8
    assert resources["builds"]["baseline"]["literal_allocations"] == resources["builds"]["candidate"]["literal_allocations"]
    assert resources["builds"]["baseline"]["shared_workspace_source_sha256"] == resources["builds"]["candidate"]["shared_workspace_source_sha256"]
    assert report["records"]["baseline"]["performance"] == report["records"]["candidate"]["performance"]
    assert arrays["baseline"].tobytes() == arrays["candidate"].tobytes()
    assert report["compiler_words_equal"] and report["source_relative_quality_passed"]
    assert report["source_full_logits_available"] is False
    assert report["same_logical_timing_boundary"] and report["uninstrumented_cycle_comparison"]
    assert report["candidate_to_source_cycle_ratio_if_qualified"] == report["records"]["candidate"]["cycles"] / report["source_cycles"]
    print("PASS: all 5,120,000 compiler words identical; source gate covers 33 retained rows.")


if __name__ == "__main__":
    main()
