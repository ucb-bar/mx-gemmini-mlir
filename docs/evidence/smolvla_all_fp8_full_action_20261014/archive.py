"""Archive and independently verify a full all-FP8 SmolVLA Spike session."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import struct

import numpy as np


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "smolvla_full_action_values_20261014"
COUNTS = {"prefix_encode": (243, 186), "flow_denoise": (148, 116)}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    if (actual.shape != expected.shape or actual.dtype != np.float32
            or expected.dtype != np.float32
            or not np.isfinite(actual).all() or not np.isfinite(expected).all()):
        raise ValueError("action comparison needs finite, matching f32 arrays")
    lhs, rhs = actual.astype(np.float64), expected.astype(np.float64)
    difference = lhs - rhs
    denominator = max(float(np.linalg.norm(lhs) * np.linalg.norm(rhs)), 1e-300)
    reference_rms = max(float(np.sqrt(np.mean(rhs * rhs))), 1e-300)
    rmse = float(np.sqrt(np.mean(difference * difference)))
    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": rmse,
        "normalized_rmse": rmse / reference_rms,
        "max_abs": float(np.max(np.abs(difference))),
        "cosine": float(np.clip(np.sum(lhs * rhs) / denominator, -1.0, 1.0)),
    }


def verify() -> dict:
    receipt = json.loads((HERE / "receipt.json").read_text())
    if (receipt.get("schema") != "mx_gemmini.smolvla_all_fp8_full_action.v1"
            or receipt.get("numerical_qualification") != "not_established"
            or receipt.get("equal_output_speedup_valid") is not False):
        raise ValueError("unexpected all-FP8 archive schema")
    for name, expected in receipt["artifact_sha256"].items():
        path = HERE / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"archive artifact differs: {name}")

    link = json.loads((HERE / "link.json").read_text())
    baseline = json.loads((HERE / "baseline_link.json").read_text())
    run = json.loads((HERE / "run.json").read_text())
    report = json.loads((HERE / "fp32_comparison.json").read_text())
    for key in ("schema", "profile_sha256", "shared_weight_sha256",
                "shared_weight_bytes", "programs", "bindings",
                "dump_outputs", "performance_schema", "mem_bytes"):
        if link[key] != baseline[key]:
            raise ValueError(f"all-FP8 and mixed links differ in {key}")
    for key in ("arena_base", "arena_bytes", "weights_base", "mem_bytes"):
        if link["arena_layout"][key] != baseline["arena_layout"][key]:
            raise ValueError(f"all-FP8 and mixed links differ in arena {key}")
    if (link["stage_object_sha256"]["action_decode"]
            != baseline["stage_object_sha256"]["action_decode"]
            or link["stage_object_sha256"]["prefix_encode"]
            == baseline["stage_object_sha256"]["prefix_encode"]
            or link["stage_object_sha256"]["flow_denoise"]
            == baseline["stage_object_sha256"]["flow_denoise"]):
        raise ValueError("expected precision-stage object changes are absent")
    if [(p["name"], p["steps"], p["selected_operations"])
            for p in link["programs"]] != [
                ("prefix_encode", 1, 243), ("flow_denoise", 10, 148),
                ("action_decode", 1, 0)]:
        raise ValueError("SmolVLA stage roster differs")

    for program, (sites, promoted) in COUNTS.items():
        probe = json.loads((HERE / f"{program}_probe.json").read_text())
        catalog = json.loads((HERE / f"{program}_catalog.json").read_text())
        manifest = json.loads((HERE / f"{program}_overrides.json").read_text())
        linked = next(row for row in link["programs"] if row["name"] == program)
        if (manifest["schema"] != "mx_gemmini.precision_overrides.v1"
                or manifest["source_mlir_sha256"] != linked["capture_model_sha256"]
                or probe["capture_model_sha256"] != linked["capture_model_sha256"]
                or manifest["profile_sha256"] != link["profile_sha256"]
                or probe["profile_sha256"] != link["profile_sha256"]
                or catalog["profile_sha256"] != link["profile_sha256"]
                or manifest["regions"] != probe["precision_overrides"]
                or manifest["regions"] != catalog["precision_overrides"]
                or len(manifest["regions"]) != promoted
                or set(manifest["regions"].values()) != {"mxfp8"}
                or probe["selected_operations"] != sites
                or catalog["covered_contractions"] != sites
                or catalog["coverage_complete"] is not True):
            raise ValueError(f"{program} FP8 plan or capture differs")
        bindings = catalog["bindings"]
        names = [row["region"] for row in bindings]
        if (len(bindings) != sites or len(names) != len(set(names))
                or not set(manifest["regions"]).issubset(names)
                or any(row["target_dtypes"][:2] != ["f8E4M3FN"] * 2
                       for row in bindings)):
            raise ValueError(f"{program} does not bind every site to FP8")

    if (run.get("status") != "complete_numeric_qualification_pending"
            or run.get("complete") is not True
            or run["link_receipt_sha256"] != sha(HERE / "link.json")
            or run["timing_config"] != "e4m3_vpu"
            or run["timing_overrides"] != "mem.host_tracking=0"
            or run["observed_events"] != run["expected_events"]
            or report["run_receipt_sha256"] != sha(HERE / "run.json")
            or report["link_receipt_sha256"] != sha(HERE / "link.json")
            or report["reference_sha256"] != sha(HERE / "session_goldens.npz")):
        raise ValueError("all-FP8 run or FP32 comparison identity differs")
    log = gzip.decompress((HERE / "spike.log.gz").read_bytes())
    if (hashlib.sha256(log).hexdigest() != run["log_sha256"]
            or b"DONE\n" not in log
            or sha(HERE / "gemmini_perf.txt") != run["timing_report_sha256"]):
        raise ValueError("Spike log or timing report differs")

    selected = {(row["program"], row["step"], row["output"]): row
                for row in run["output_dumps"]}
    expected_keys = ({(1, step, output) for step in range(10)
                      for output in (0, 2)} | {(2, 0, 0)})
    if set(selected) != expected_keys:
        raise ValueError("flow, timestep, or action output is missing")
    with np.load(HERE / "session_goldens.npz", allow_pickle=False) as goldens:
        expected = goldens["actions"]
    if expected.shape != (10, 1, 50, 32) or expected.dtype != np.float32:
        raise ValueError("captured FP32 action trajectory differs")

    timestep = np.float32(1.0)
    for step in range(10):
        flow = selected[1, step, 0]
        path = HERE / flow["path"]
        if (Path(flow["path"]).parent != Path("output_dumps")
                or flow["bytes"] != 6400
                or path.stat().st_size != 6400
                or sha(path) != flow["sha256"]):
            raise ValueError(f"flow output {step} differs")
        actual = np.frombuffer(path.read_bytes(), dtype="<f4").reshape(1, 50, 32)
        observed = metrics(actual, expected[step])
        stored = report["flow_step_metrics"][step]
        if (stored["step"] != step
                or any(abs(observed[key] - stored[key]) > 1e-10
                       for key in observed)):
            raise ValueError(f"FP32 comparison for flow {step} differs")
        timestep = np.float32(timestep + np.float32(-0.1))
        value = selected[1, step, 2]
        path = HERE / value["path"]
        if (Path(value["path"]).parent != Path("output_dumps")
                or value["bytes"] != 4 or path.stat().st_size != 4
                or sha(path) != value["sha256"]
                or struct.unpack("<f", path.read_bytes())[0] != float(timestep)):
            raise ValueError(f"FP32 scalar timestep {step} differs")
    action = selected[2, 0, 0]
    action_path = HERE / action["path"]
    last_flow = HERE / selected[1, 9, 0]["path"]
    if (Path(action["path"]).parent != Path("output_dumps")
            or action["bytes"] != 6400 or action_path.stat().st_size != 6400
            or sha(action_path) != action["sha256"]
            or action_path.read_bytes() != last_flow.read_bytes()
            or report["final_action_matches_flow_bytes"] is not True
            or report["final_action_metrics"] != report["flow_step_metrics"][-1]):
        raise ValueError("final action does not match final flow state")
    return {
        "final_action_cosine": report["final_action_metrics"]["cosine"],
        "final_action_normalized_rmse":
            report["final_action_metrics"]["normalized_rmse"],
        "modeled_cycles": run["total_performance"]["cycles"],
    }


def archive(args: argparse.Namespace) -> None:
    if (HERE / "receipt.json").exists():
        raise ValueError("all-FP8 archive already exists")
    run = json.loads(args.run.read_text())
    report = json.loads(args.comparison.read_text())
    if (run.get("status") != "complete_numeric_qualification_pending"
            or run.get("complete") is not True
            or run.get("link_receipt_sha256") != sha(args.link)
            or report.get("run_receipt_sha256") != sha(args.run)
            or report.get("link_receipt_sha256") != sha(args.link)):
        raise ValueError("all-FP8 run or FP32 comparison is incomplete")

    HERE.mkdir(parents=True, exist_ok=True)
    files = {
        "run.json": args.run,
        "link.json": args.link,
        "baseline_link.json": args.baseline_link,
        "fp32_comparison.json": args.comparison,
        "session_goldens.npz": PRIOR / "session_goldens.npz",
        "gemmini_perf.txt": args.run.parent / "gemmini_perf.txt",
    }
    for program, build, manifest in (
            ("prefix_encode", args.prefix_build, args.prefix_manifest),
            ("flow_denoise", args.flow_build, args.flow_manifest)):
        files[f"{program}_probe.json"] = build / "mx_stage_probe.json"
        files[f"{program}_catalog.json"] = build / "device_catalog/mx_catalog.json"
        files[f"{program}_overrides.json"] = manifest
    for name, source in files.items():
        shutil.copyfile(source, HERE / name)
    (HERE / "spike.log.gz").write_bytes(gzip.compress(
        (args.run.parent / "spike.log").read_bytes(), mtime=0))
    names = [*files, "spike.log.gz"]
    (HERE / "output_dumps").mkdir(exist_ok=True)
    for row in run["output_dumps"]:
        relative = Path(row["path"])
        if relative.parent != Path("output_dumps"):
            raise ValueError("output dump path escapes the archive")
        source = args.run.parent / relative
        if sha(source) != row["sha256"]:
            raise ValueError(f"output dump differs: {relative}")
        shutil.copyfile(source, HERE / relative)
        names.append(str(relative))
    receipt = {
        "schema": "mx_gemmini.smolvla_all_fp8_full_action.v1",
        "scope": "one prefix, ten flow steps, one action; 391 FP8 matrix sites",
        "numerical_qualification": "not_established",
        "equal_output_speedup_valid": False,
        "artifact_sha256": {name: sha(HERE / name) for name in names},
    }
    (HERE / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verify(), indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    for option in ("run", "link", "baseline-link", "comparison",
                   "prefix-build", "flow-build", "prefix-manifest",
                   "flow-manifest"):
        parser.add_argument(f"--{option}", type=Path)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(), indent=2, sort_keys=True))
        return
    for name in ("run", "link", "baseline_link", "comparison",
                 "prefix_build", "flow_build", "prefix_manifest",
                 "flow_manifest"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} required")
    archive(args)


if __name__ == "__main__":
    main()
