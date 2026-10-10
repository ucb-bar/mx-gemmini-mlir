"""Replay the public three-site MX/VPU object compiler from Nicolas's source.

Four typed graph/profile/schedule combinations use `compile_object`. Each
compiled issuer must reproduce the previously qualified source-bound object,
then execute the source C1/C2 goldens on the pinned Spike extension.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from mx_gemmini_support.full_chain_pipelined import (
    FULL_INPUTS, FULL_OUTPUTS, audit_full_chain_pipelined)
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision
from tools.compile_nicolas_chain_pipelined import _run_spike


ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "profiles/gemmini-mx-cleanup-266c593"
CASES = (
    ("fp4_vpu_serial", "nicolas_chain_pipelined_full_266c593", "",
     "MxE4M3Fp4VpuGemminiRocketConfig", "program_order_with_dependency_fences"),
    ("fp4_vpu_pipelined", "nicolas_chain_pipelined_full_266c593", "pipelined",
     "MxE4M3Fp4VpuGemminiRocketConfig", "pipelined"),
    ("e4m3_vpu_serial", "nicolas_chain_pipelined_e4m3_only_266c593", "compiled",
     "MxE4M3VpuGemminiRocketConfig", "program_order_with_dependency_fences"),
    ("e4m3_vpu_pipelined", "nicolas_chain_pipelined_e4m3_only_266c593",
     "compiled_pipelined", "MxE4M3VpuGemminiRocketConfig", "pipelined"),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rtl-root", "riscv-root", "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    for name in ("rtl_root", "riscv_root", "mx_opt", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    if _git_revision(args.rtl_root) != "266c593f2cb51d7e3fe83fc0317072b585ac3c52":
        parser.error("Nicolas MX+VPU RTL revision differs from pinned source")
    if not args.mx_opt.is_file():
        parser.error("native MX verifier is absent")
    software = args.rtl_root / "software/gemmini-rocc-tests"
    paths = (software / "bareMetalC/chain_pipelined.c",
             software / "include/matmul_fp8_64x64_chain.h",
             software / "bareMetalC/matmul_tiled_fp8_64x64_chain.c",
             software / "bareMetalC/chain_vpu_spad_requant.c")
    args.out_dir.mkdir(parents=True)
    inputs = args.out_dir / "inputs"
    inputs.mkdir()
    (inputs / "abi.json").write_text(json.dumps({
        "schema": "mx_gemmini.full_vpu_branch_buffer_map.v1",
        "inputs": {name: name for name in FULL_INPUTS},
        "outputs": {name: name for name in FULL_OUTPUTS},
        "source_reference": "c1_bf16",
    }, indent=2, sort_keys=True) + "\n")
    rows = []
    reference_inputs = None
    for name, evidence_name, subdir, profile_name, schedule in CASES:
        profile_path = PROFILE_DIR / f"{profile_name}.json"
        profile = load_profile(profile_path, rtl_root=args.rtl_root)
        resources, facts = audit_full_chain_pipelined(*paths, profile)
        resource_sha = {key: hashlib.sha256(resources[key]).hexdigest()
                        for key in (*FULL_INPUTS, "c1_bf16")}
        if reference_inputs is None:
            reference_inputs = resource_sha
            for key in resource_sha:
                (inputs / f"{key}.bin").write_bytes(resources[key])
        elif resource_sha != reference_inputs:
            raise ValueError("Nicolas source operands differ across VPU profiles")
        baseline = ROOT / "docs/evidence" / evidence_name / subdir
        mlir, preloaded = baseline / "connected.mlir", baseline / "preloaded.mlir"
        case = args.out_dir / name
        command = [
            sys.executable, "-m", "tools.compile_object",
            "--mlir", str(mlir), "--preloaded-mlir", str(preloaded),
            "--profile", str(profile_path), "--rtl-root", str(args.rtl_root),
            "--riscv-root", str(args.riscv_root),
            "--resources-dir", str(inputs), "--abi-json", str(inputs / "abi.json"),
            "--mx-opt", str(args.mx_opt), "--issue-schedule", schedule,
            "--out-dir", str(case),
        ]
        result = subprocess.run(command, cwd=ROOT, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                check=False)
        (args.out_dir / f"{name}.compile.log").write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(f"public MX object compile failed: {name}")
        obj = json.loads((case / "object_manifest.json").read_text())
        compiled = json.loads((case / "compile_manifest.json").read_text())
        if (compiled["lowering_family"] != "full_vpu_branch" or
                obj["object_sha256"] != _sha(baseline / "mx_issue.o") or
                obj["issuer_c_sha256"] != _sha(baseline / "mx_issue.c") or
                obj["input_sha256"] != {key: reference_inputs[key]
                                                for key in FULL_INPUTS} or
                obj["source_reference_sha256"] != reference_inputs["c1_bf16"] or
                obj["profile_sha256"] != profile_sha256(profile)):
            raise ValueError(f"public MX object differs from source-qualified stream: {name}")
        names = tuple(entry["name"] for entry in obj["buffer_abi"])
        qualified = _run_spike(
            case, args.rtl_root, args.riscv_root, case / "mx_issue.o",
            resources, names, include_mm1=True)
        (case / "spike_qualification.json").write_text(
            json.dumps(qualified, indent=2, sort_keys=True) + "\n")
        if (qualified["status"] != "full_three_site_chain_matched_on_pinned_spike" or
                qualified["spike_exit_code"] != 0 or
                qualified["compared_fp8_codes"] != 16384 or
                qualified["compared_e8m0_scales"] != 512 or
                qualified["compared_c1_bf16_values"] != 4096):
            raise ValueError(f"public MX object failed full source parity: {name}")
        rows.append({
            "name": name, "profile_sha256": profile_sha256(profile),
            "schedule": schedule, "bound_mlir_sha256": _sha(mlir),
            "preloaded_mlir_sha256": _sha(preloaded),
            "baseline_object_sha256": _sha(baseline / "mx_issue.o"),
            "object_sha256": obj["object_sha256"],
            "object_manifest_sha256": _sha(case / "object_manifest.json"),
            "compile_manifest_sha256": _sha(case / "compile_manifest.json"),
            "spike_qualification_sha256": _sha(case / "spike_qualification.json"),
            "elf_sha256": qualified["elf_sha256"],
            "spike_log_sha256": qualified["spike_log_sha256"],
            "source_sha256": facts["source_sha256"],
        })
        print(f"{name}: source outputs matched on Spike", flush=True)
    index = {
        "schema": "mx_gemmini.public_full_vpu_branch_replay.v1",
        "status": "four_public_branch_objects_matched_source_on_pinned_spike",
        "compiler_revision": _git_revision(ROOT),
        "rtl_revision": _git_revision(args.rtl_root),
        "input_sha256": reference_inputs,
        "compared_c1_bf16_values": 4 * 4096,
        "compared_fp8_codes": 4 * 16384,
        "compared_e8m0_scales": 4 * 512,
        "rows": rows,
    }
    (args.out_dir / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"four public MX/VPU branch objects matched source: {args.out_dir}")


if __name__ == "__main__":
    main()
