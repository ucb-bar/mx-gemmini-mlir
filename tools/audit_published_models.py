"""Audit published model outputs, target placement, timing and source comparisons."""

from __future__ import annotations

import argparse
import collections
import datetime
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from mx_gemmini_support.spike_timing import reject_deadlocked_timing
from tools.model_release import ROOT, evidence, probe, reader, registry, sha


def radiance_artifacts(source):
    """Reopen retained code, command plans and output-comparison observations."""
    from tools.qualify_radiance_restream import prove_restream_weight_reads
    from tools.qualify_radiance_ws_roster import _prove_weight_read_once

    cases = []
    for directory, receipt_key in (("radiance_ws_read_once_1a7bc9c", "read_once_index_sha256"),
                                   ("radiance_restream_1df2c5f", "restream_index_sha256"),
                                   ("radiance_batched_gemv_552a16b", "batched_gemv_index_sha256")):
        root = ROOT / "docs/evidence" / directory
        if sha(root / "index.json") != source["evidence"][receipt_key]:
            raise ValueError("Radiance comparison index changed")
        index = json.loads((root / "index.json").read_text())
        for row in index.get("rows", [index]):
            case = root / row["directory"] if "directory" in row else root
            program_raw = gzip.decompress((case / "physical_program.json.gz").read_bytes())
            if hashlib.sha256(program_raw).hexdigest() != row["physical_program_sha256"]:
                raise ValueError("Radiance command plan changed")
            program = json.loads(program_raw)
            if "m_tiles" in row:
                traffic = prove_restream_weight_reads(program)
            else:
                m, n, k = row["shape_mnk"]
                traffic = _prove_weight_read_once(program, m=m, n=n, k=k, tile_k=row["tile_mnk"][2], precision=row["precision"])
            if any(row.get(key) != value for key, value in traffic.items()):
                raise ValueError("Radiance traffic observation differs from command plan")
            for filename in ("mx_driver.c", "mx_data.S", "mx_issue.c"):
                raw = (gzip.decompress((case / (filename + ".gz")).read_bytes()) if filename == "mx_issue.c"
                       else (case / filename).read_bytes())
                if hashlib.sha256(raw).hexdigest() != row["files_sha256"][filename]:
                    raise ValueError("Radiance retained issuer/driver changed")
            if sha(case / "spike.log") != row["spike_log_sha256"]:
                raise ValueError("Radiance retained console changed")
            for label in ("first", "repro"):
                run = json.loads((case / ("spike_" + label + ".json")).read_text())
                if run["spike_exit_code"] != 0 or run["status"] != "source_golden_matched_on_pinned_spike":
                    raise ValueError("Radiance source-output comparison did not complete")
                for key in ("compared_bf16_outputs", "files_sha256", "object_sha256", "elf_sha256", "extension_sha256", "spike_log_sha256"):
                    if run[key] != row[key]:
                        raise ValueError("Radiance result differs from its archived index")
            cases.append({"directory": row.get("directory", directory), "compared_bf16_outputs": row["compared_bf16_outputs"],
                          "physical_program_sha256": row["physical_program_sha256"], "traffic": traffic})
    return cases


def audit(out, *, build_root=None, radiance_root=None):
    if out.exists():
        raise ValueError("audit output directory must be fresh")
    out.mkdir(parents=True)
    release = registry()
    report = {"schema": "mx_gemmini.published_model_audit.v1", "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "release_sha256": sha(ROOT / "examples/models/release.json"), "models": {}, "checks": [],
              "scope": "retained completed executions and compiler artifacts; no new FPGA or dataset-accuracy result",
              "actual_external_images_checked": build_root is not None}
    profile = json.loads((ROOT / release["profile"]["path"]).read_text())
    canonical = (json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if hashlib.sha256(canonical).hexdigest() != release["profile"]["canonical_sha256"]:
        raise ValueError("selected profile changed")
    for name, item in release["models"].items():
        command = [sys.executable, str(ROOT / item["archive"] / "verify.py")]
        if build_root:
            command += ["--build-root", str(build_root)]
        result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        (out / (name + ".log")).write_bytes(result.stdout)
        if result.returncode:
            raise ValueError(f"{name} retained-evidence verification failed; see {name}.log")
        _, files = evidence(name)
        log_count = 0
        for key, value in files.items():
            if key.endswith(("spike.log", "spike.log.gz")):
                reject_deadlocked_timing((gzip.decompress(value) if key.endswith(".gz") else value).decode())
                log_count += 1
        stages = {}
        for stage, record in item["stages"].items():
            observed = probe(files, item, stage)
            if observed["capture_model_sha256"] != record["capture_model_sha256"]:
                raise ValueError("capture model identity changed")
            if observed["checkpoint_weights_sha256"] != record["weights_sha256"]:
                raise ValueError("checkpoint identity changed")
            if "policy" in record:
                policy_path = ROOT / record["policy"]["path"]
                policy = json.loads(policy_path.read_text())
                if sha(policy_path) != record["policy"]["sha256"] or policy != observed["execution_policy"]:
                    raise ValueError("published execution policy differs from actual build")
                placements = {family: dict(collections.Counter(entries.values()))
                              for family, entries in policy["vpu_placements"].items()}
                if placements != record["placements"]:
                    raise ValueError("published host/VPU placement census differs")
                prefix = "candidate/" if item["kind"] == "forward" else f"candidate/stages/{stage}/"
                catalog = json.loads(files[prefix + "device_catalog/mx_catalog.json"])
                kernels = {row["symbol"]: row for row in catalog["kernels"]}
                formats = dict(collections.Counter(kernels[row["symbol"]]["precision"] for row in catalog["bindings"]))
                if formats != record["matrix_bindings"] or set(formats) != {"mxfp8"}:
                    raise ValueError("published matrix precision census differs")
                selected_formats = dict(collections.Counter(value for value in policy["matrix_formats"].values() if value != "host"))
                if selected_formats != formats:
                    raise ValueError("matrix policy and actual device bindings differ")
                stages[stage] = {"matrix_bindings": formats, "placements": record["placements"],
                                 "capture_model_sha256": record["capture_model_sha256"], "weights_sha256": record["weights_sha256"]}
            else:
                if observed["selected_operations"] != 0:
                    raise ValueError("host-only action stage contains selected matrix work")
                stages[stage] = {"execution": "compiled_host", "capture_model_sha256": record["capture_model_sha256"]}
        if item["kind"] == "forward":
            parser = reader(ROOT / "docs/evidence/matched_forward_20261018/verify.py")
            _, metrics = parser.parse(files["candidate/spike.log"])
            if int(metrics["cycles"]) != item["cycles"]:
                raise ValueError("published forward cycles differ from actual console")
        else:
            run = json.loads(files["candidate/run/mx_session_run.json"])
            link = json.loads(files["candidate/link/mx_session_link.json"])
            if run["execution_total_performance"]["cycles"] != item["cycles"] or link["elf_sha256"] != item["elf"]["sha256"]:
                raise ValueError("published cycle/image identity differs from completed run")
        report["models"][name] = {"status": "passed", "cycles": item["cycles"], "elf": item["elf"],
                                  "stages": stages, "retained_logs_scanned": log_count,
                                  "verifier_sha256": sha(ROOT / item["archive"] / "verify.py"),
                                  "verification_log_sha256": sha(out / (name + ".log"))}
    heads = ROOT / "docs/evidence/smolvla_attention_cutpoint_20261014/verify_compiled_pair.py"
    result = subprocess.run([sys.executable, str(heads), "--archive", "compiled_pair_pipeline"], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (out / "attention-heads.log").write_bytes(result.stdout)
    if result.returncode:
        raise ValueError("paired attention-head verification failed")
    report["checks"].append({"name": "paired_masked_attention_heads", "status": "passed",
                             "source_repository": "npu-exploration", "source_revision": release["repositories"]["npu-exploration"]["revision"],
                             "log_sha256": sha(out / "attention-heads.log")})
    late = ROOT / "docs/evidence/smolvla_attention_session_20261014/verify.py"
    result = subprocess.run([sys.executable, str(late)], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (out / "late-attention-heads.log").write_bytes(result.stdout)
    if result.returncode:
        raise ValueError("final-step attention-head verification failed")
    report["checks"].append({"name": "paired_final_step_attention_heads", "status": "passed",
                             "log_sha256": sha(out / "late-attention-heads.log")})
    source = json.loads((ROOT / "docs/evidence/radiance_kernel_comparison_20261014/receipt.json").read_text())
    revision = release["repositories"]["radiance-kernels"]["revision"]
    if source["radiance_revision"] != revision or source["cycle_ratio_valid"] is not False or source["paired_source_compiler_cycles"] is not False:
        raise ValueError("Radiance comparison attribution or timing scope differs")
    report["checks"].append({"name": "radiance_retained_kernel_artifacts", "status": "passed",
                             "cases": radiance_artifacts(source),
                             "scope": "frozen command/source bytes, traffic proofs and retained complete-output comparison observations; no new kernel execution"})
    checked = []
    if radiance_root:
        for row in [*source["rows"], *source["batched_gemv_rows"]]:
            historical = subprocess.run(["git", "-C", str(radiance_root), "show", revision + ":" + row["source"]],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            raw = historical.stdout if historical.returncode == 0 else (radiance_root / row["source"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["source_sha256"]:
                raise ValueError("pinned Radiance source differs from comparison receipt")
            checked.append(row["source"])
    report["checks"].append({"name": "radiance_kernel_control_scope", "status": "passed", "source_revision": revision,
                             "paired_cycles_available": False, "pinned_source_files_reopened": checked,
                             "source_check_scope": "driver bytes matched by SHA-256; compatible working-tree files allowed when historical Git objects are absent"})
    report["limitations"] = ["Spike timing uses mem.host_tracking=0; FPGA latency is unmeasured",
                             "Independent full TinyLlama distributions cover 33 positions; all compiler outputs cover 160",
                             "Cached TinyLlama and full SmolVLA have no independent source-session performance baseline",
                             "SmolVLA action errors are diagnostics, without a robot-task acceptance threshold",
                             "Exact historical ELF rebuilding still needs external checkpoint and toolchain inputs"]
    report["status"] = "passed_with_documented_scope"
    (out / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "models": list(report["models"]),
                      "actual_external_images_checked": report["actual_external_images_checked"]}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--build-root", type=Path)
    parser.add_argument("--radiance-root", type=Path)
    args = parser.parse_args()
    audit(args.out, build_root=args.build_root, radiance_root=args.radiance_root)


if __name__ == "__main__":
    main()
