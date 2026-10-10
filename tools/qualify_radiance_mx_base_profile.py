"""Run selected Radiance FP8/FP4/FP6 source cases on a pinned MX profile.

The inputs are model2MLIR's archived handoffs for these exact source drivers.
By default this rebinds all three handoffs to MxGemminiRocketConfig. A selected
profile and subset of source precisions may be supplied to qualify other legal
DIM16 Rocket configurations. The command exports the source
payload, lowers physical MX commands, builds RV64 ELFs, and compares every
BF16 output on the pinned Rocket/RoCC Spike extension.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from mx_gemmini_support.bind_profile import bind_handoff
from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _sha, _source_closure


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "docs/evidence/radiance_mx_gemm_latest_e9ded36_ee22/frontend"
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json"
DRIVERS = {
    "fp8": "mxgemm.fp8.m128n128k512.tm128tn128tk128.fullout",
    "fp4": "mxgemm.fp4.m64n64k128.tm64tn64tk64.fullout",
    "fp6": "mxgemm.fp6.m128n128k2048.tm128tn128tk128.fullout",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "rtl-root", "riscv-root", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--case", action="append", choices=tuple(DRIVERS),
                        help="select a source precision; repeat for several (default: all three)")
    args = parser.parse_args()
    source, rtl, riscv, out = (args.source_root.resolve(), args.rtl_root.resolve(),
                               args.riscv_root.resolve(), args.out_dir.resolve())
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    selected_profile = args.profile.resolve()
    selected_cases = {name: stem for name, stem in DRIVERS.items()
                      if args.case is None or name in args.case}
    if args.case is not None and len(set(args.case)) != len(args.case):
        parser.error("each --case may be selected only once")
    profile = load_profile(selected_profile, rtl_root=rtl)
    if (profile["transport"] != "rocket_rocc" or
            profile["geometry"]["mesh_columns"] != 16 or
            not profile["legal_compute"]):
        parser.error("selected MX profile needs legal DIM16 Rocket compute")
    base_contract = (selected_profile == PROFILE.resolve() and
                     len(selected_cases) == len(DRIVERS))
    if base_contract and (profile["name"] != "MxGemminiRocketConfig" or
                          len(profile["legal_compute"]) != 3 or
                          not profile["resources"]["lut"] or
                          profile["resources"]["vpu"]):
        parser.error("selected plain MX profile differs from the three-precision RTL config")
    frontend_index = json.loads((FRONTEND / "index.json").read_text())
    if (frontend_index["model2mlir_revision"] !=
            "e9ded36eb85abf2d9097ac4dc11457c825853388" or
            frontend_index["rtl_revision"] != _git_revision(rtl)):
        parser.error("archived model2MLIR capture or Nicolas RTL revision differs")
    rows = {Path(row["driver"]).stem: row for row in frontend_index["rows"]}
    staged = []
    for precision, stem in selected_cases.items():
        row = rows[stem]
        captured = FRONTEND / stem
        receipt = json.loads((captured / "receipt.json").read_text())
        handoff = captured / "mx_gemm.handoff.mlir"
        frontend = captured / "mx_gemm.model2mlir.mlir"
        driver = source / row["driver"]
        kernel = read_source_gemm(driver)
        if (not kernel.data_header_present or
                row["precision"] != precision.upper() or
                row["shape_mnk"] != list(kernel.shape) or
                row["tile_mnk"] != list(kernel.tile) or
                row["quant_output"] or
                _sha(driver) != row["driver_sha256"] or
                _sha(kernel.data_header) != row["header_sha256"] or
                _sha(handoff) != receipt["handoff_mlir_sha256"] or
                _sha(frontend) != row["source_mlir_sha256"] or
                _sha(captured / "receipt.json") != row["receipt_sha256"]):
            raise ValueError(f"Radiance source or model2MLIR capture differs: {stem}")
        staged.append((precision, stem, row, handoff, driver, kernel,
                       bind_handoff(handoff.read_text(), profile)))
    out.mkdir(parents=True)
    inputs = out / "inputs"
    inputs.mkdir()
    results = []
    for precision, stem, row, handoff, driver, kernel, bound_handoff in staged:
        selected_mlir = inputs / f"{stem}.mlir"
        selected_mlir.write_text(bound_handoff)
        result_dir = out / precision
        command = [sys.executable, "-m", "tools.qualify_source_mx",
                   "--mlir", str(selected_mlir), "--driver", str(driver),
                   "--profile", str(selected_profile), "--rtl-root", str(rtl),
                   "--riscv-root", str(riscv), "--out-dir", str(result_dir)]
        run = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, check=False)
        (inputs / f"{precision}.compile.log").write_text(run.stdout)
        if run.returncode != 0:
            raise RuntimeError(f"MX {precision} qualification failed: {run.stdout[-3000:]}")
        artifact = json.loads((result_dir / "build/artifact_manifest.json").read_text())
        if (artifact["status"] != "source_golden_matched_on_pinned_spike" or
                artifact["spike_exit_code"] != 0 or
                artifact["compared_bf16_outputs"] != kernel.shape[0] * kernel.shape[1] or
                artifact["profile_sha256"] != profile_sha256(profile) or
                artifact["bound_mlir_sha256"] != _sha(result_dir / "payload_bound.mlir")):
            raise RuntimeError(f"MX {precision} did not match every source BF16 output")
        results.append({
            "precision": precision.upper(), "driver": row["driver"],
            "driver_sha256": _sha(driver),
            "header_sha256": _sha(kernel.data_header),
            "handoff_mlir_sha256": _sha(handoff),
            "profile_bound_mlir_sha256": _sha(selected_mlir),
            "payload_bound_mlir_sha256": _sha(result_dir / "payload_bound.mlir"),
            "bundle_manifest_sha256": _sha(result_dir / "bundle/manifest.json"),
            "physical_program_sha256": _sha(result_dir / "build/physical_program.json"),
            "generated_issue_sha256": _sha(result_dir / "build/mx_issue.c"),
            "artifact_manifest_sha256": _sha(result_dir / "build/artifact_manifest.json"),
            "elf_sha256": artifact["elf_sha256"],
            "spike_log_sha256": artifact["spike_log_sha256"],
            "compared_bf16_outputs": artifact["compared_bf16_outputs"],
            "status": artifact["status"],
        })
    summary = {
        "schema": ("mx_gemmini.radiance_plain_mx_profile_source_ladder.v1"
                   if base_contract else
                   "mx_gemmini.radiance_selected_mx_profile_source_cases.v1"),
        "status": ("three_source_precisions_matched_on_pinned_spike"
                   if base_contract else
                   "selected_source_precisions_matched_on_pinned_spike"),
        "scope": ("archived model2MLIR e9ded36 captures rebound to Nicolas MxGemminiRocketConfig; source operands and BF16 goldens from byte-checked Radiance headers"
                  if base_contract else
                  "archived model2MLIR e9ded36 captures rebound to the selected Nicolas MX profile; source operands and BF16 goldens from byte-checked Radiance headers"),
        "source_revision": _git_revision(source),
        "rtl_revision": _git_revision(rtl),
        "model2mlir_revision": frontend_index["model2mlir_revision"],
        "frontend_index_sha256": _sha(FRONTEND / "index.json"),
        "profile_sha256": profile_sha256(profile),
        "compiler_revision": _git_revision(ROOT),
        "compiler_source_closure_sha256": _source_closure(
            ROOT, sorted((ROOT / "mx_gemmini_support").glob("*.py")) +
            sorted((ROOT / "tools").glob("*.py"))),
        "compared_bf16_outputs": sum(row["compared_bf16_outputs"] for row in results),
        "rows": results,
    }
    if not base_contract:
        summary["profile_name"] = profile["name"]
    (out / "qualification.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"{profile['name']} {','.join(selected_cases)}: {summary['compared_bf16_outputs']} "
          "source BF16 outputs matched Nicolas Spike")


if __name__ == "__main__":
    main()
