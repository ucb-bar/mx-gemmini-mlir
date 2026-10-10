"""Run selected Radiance FP8/FP4/FP6 source cases on a pinned MX profile.

The inputs are model2MLIR's archived handoffs for these exact source drivers.
By default this rebinds all three handoffs to MxGemminiRocketConfig. A selected
profile and subset of source precisions may be supplied to qualify other legal
DIM8, DIM16, or DIM32 Rocket configurations. The command exports the source
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
    parser.add_argument("--all-fullout", action="store_true",
                        help="compile all 23 archived BF16-output Radiance MX GEMM drivers")
    parser.add_argument("--all-requant", action="store_true",
                        help="compile all eight archived quantized-output Radiance MX GEMM drivers")
    parser.add_argument("--precision", action="append", choices=("FP4", "FP6", "FP8"),
                        help="restrict an all-fullout or all-requant roster to this precision")
    parser.add_argument("--rtl-product-floor-reference", action="store_true",
                        help="use the versioned target mesh reference with Nicolas's product floor")
    args = parser.parse_args()
    if int(args.all_fullout) + int(args.all_requant) + int(bool(args.case)) > 1:
        parser.error("choose one of --all-fullout, --all-requant, or --case")
    if args.precision and not (args.all_fullout or args.all_requant):
        parser.error("--precision requires --all-fullout or --all-requant")
    if args.precision and len(args.precision) != len(set(args.precision)):
        parser.error("each --precision may be selected only once")
    source, rtl, riscv, out = (args.source_root.resolve(), args.rtl_root.resolve(),
                               args.riscv_root.resolve(), args.out_dir.resolve())
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    selected_profile = args.profile.resolve()
    if args.case is not None and len(set(args.case)) != len(args.case):
        parser.error("each --case may be selected only once")
    profile = load_profile(selected_profile, rtl_root=rtl)
    if (profile["transport"] != "rocket_rocc" or
            profile["geometry"]["mesh_columns"] not in {8, 16, 32} or
            profile["geometry"]["mesh_rows"] != profile["geometry"]["mesh_columns"] or
            not profile["legal_compute"]):
        parser.error("selected MX profile needs legal square-mesh Rocket compute")
    mesh_reference = profile["geometry"]["mesh_columns"] != 16
    if args.rtl_product_floor_reference and not mesh_reference:
        parser.error("RTL product-floor reference requires DIM8 or DIM32")
    frontend_index = json.loads((FRONTEND / "index.json").read_text())
    if (frontend_index["model2mlir_revision"] !=
            "e9ded36eb85abf2d9097ac4dc11457c825853388" or
            frontend_index["rtl_revision"] != _git_revision(rtl)):
        parser.error("archived model2MLIR capture or Nicolas RTL revision differs")
    rows = {Path(row["driver"]).stem: row for row in frontend_index["rows"]}
    selected_cases = ({stem: stem for stem, row in rows.items()
                       if not row["quant_output"]} if args.all_fullout else
                      {stem: stem for stem, row in rows.items()
                       if row["quant_output"]} if args.all_requant else
                      {name: stem for name, stem in DRIVERS.items()
                       if args.case is None or name in args.case})
    if args.precision:
        selected_cases = {label: stem for label, stem in selected_cases.items()
                          if rows[stem]["precision"] in args.precision}
        expected_counts = ({"FP4": 3, "FP6": 2, "FP8": 3} if args.all_requant else
                           {"FP4": 8, "FP6": 5, "FP8": 10})
        if len(selected_cases) != sum(expected_counts[name] for name in args.precision):
            parser.error("archived selected-precision roster differs from source capture")
    if not selected_cases:
        parser.error("selected precision roster has no captured drivers")
    if args.all_fullout and not args.precision and len(selected_cases) != 23:
        parser.error("archived Radiance BF16-output roster differs from 23 drivers")
    if args.all_requant and not args.precision and len(selected_cases) != 8:
        parser.error("archived Radiance requant roster differs from eight drivers")
    base_contract = (not args.all_fullout and not args.all_requant and
                     selected_profile == PROFILE.resolve() and
                     len(selected_cases) == len(DRIVERS))
    if base_contract and (profile["name"] != "MxGemminiRocketConfig" or
                          len(profile["legal_compute"]) != 3 or
                          not profile["resources"]["lut"] or
                          profile["resources"]["vpu"]):
        parser.error("selected plain MX profile differs from the three-precision RTL config")
    staged = []
    for label, stem in selected_cases.items():
        row = rows[stem]
        precision = row["precision"].lower()
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
                row["quant_output"] != args.all_requant or
                _sha(driver) != row["driver_sha256"] or
                _sha(kernel.data_header) != row["header_sha256"] or
                _sha(handoff) != receipt["handoff_mlir_sha256"] or
                _sha(frontend) != row["source_mlir_sha256"] or
                _sha(captured / "receipt.json") != row["receipt_sha256"]):
            raise ValueError(f"Radiance source or model2MLIR capture differs: {stem}")
        staged.append((label, precision, stem, row, handoff, driver, kernel,
                       bind_handoff(handoff.read_text(), profile)))
    out.mkdir(parents=True)
    inputs = out / "inputs"
    inputs.mkdir()
    results = []
    for label, precision, stem, row, handoff, driver, kernel, bound_handoff in staged:
        selected_mlir = inputs / f"{stem}.mlir"
        selected_mlir.write_text(bound_handoff)
        result_dir = out / label
        command = [sys.executable, "-m", "tools.qualify_source_mx",
                   "--mlir", str(selected_mlir), "--driver", str(driver),
                   "--profile", str(selected_profile), "--rtl-root", str(rtl),
                   "--riscv-root", str(riscv), "--out-dir", str(result_dir)]
        if args.all_requant:
            command.append("--source-header-quantized")
        if mesh_reference:
            command.append("--target-mesh-reference")
            if args.all_fullout or args.all_requant or args.rtl_product_floor_reference:
                command.append("--rtl-product-floor-reference")
        run = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, check=False)
        (inputs / f"{label}.compile.log").write_text(run.stdout)
        if run.returncode != 0:
            raise RuntimeError(f"MX {precision} qualification failed: {run.stdout[-3000:]}")
        artifact = json.loads((result_dir / "build/artifact_manifest.json").read_text())
        expected_status = ("target_mesh_reference_matched_on_pinned_spike" if
                           mesh_reference else "radiance_header_matched_on_pinned_spike"
                           if args.all_requant else "source_golden_matched_on_pinned_spike")
        if (artifact["status"] != expected_status or
                artifact["spike_exit_code"] != 0 or
                artifact["profile_sha256"] != profile_sha256(profile) or
                artifact["bound_mlir_sha256"] != _sha(result_dir / "payload_bound.mlir")):
            raise RuntimeError(f"MX {precision} did not match every selected reference output")
        m, n, _ = kernel.shape
        if args.all_requant:
            basis = "target" if mesh_reference else "source"
            code_field = (f"compared_{basis}_fp6_packed_bytes" if precision == "fp6" else
                          f"compared_{basis}_fp8_codes")
            quant_bytes = m * n // (2 if precision == "fp6" else 1)
            if (artifact.get(code_field) != quant_bytes or
                    artifact.get(f"compared_{basis}_e8m0_scales") != m * n // 32):
                raise RuntimeError(f"MX {precision} did not match every quantized output")
        elif artifact.get("compared_bf16_outputs") != m * n:
            raise RuntimeError(f"MX {precision} did not match every BF16 output")
        result = {
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
            "status": artifact["status"],
        }
        if args.all_requant:
            result["compared_quantized_output_bytes"] = quant_bytes
            result["compared_output_scales"] = m * n // 32
        else:
            result["compared_bf16_outputs"] = artifact["compared_bf16_outputs"]
        if mesh_reference:
            source_golden = result_dir / "source_golden_bf16.bin"
            target_manifest = json.loads((result_dir / "bundle/manifest.json").read_text())
            if (target_manifest["target_mesh_reference"]["source_golden_sha256"] !=
                    _sha(source_golden)):
                raise RuntimeError("target mesh reference lost source golden provenance")
            result["source_golden_sha256"] = _sha(source_golden)
            result["target_golden_sha256"] = _sha(result_dir / "bundle/golden_bf16.bin")
            result["target_mesh_reference"] = target_manifest["target_mesh_reference"]
            if args.all_requant:
                result["target_quant_reference"] = target_manifest["target_quant_reference"]
        results.append(result)
    summary = {
        "schema": ("mx_gemmini.radiance_target_mesh_requant_subset.v1"
                   if args.precision and args.all_requant and mesh_reference else
                   "mx_gemmini.radiance_source_requant_subset.v1"
                   if args.precision and args.all_requant else
                   "mx_gemmini.radiance_target_mesh_fullout_subset.v1"
                   if args.precision and mesh_reference else
                   "mx_gemmini.radiance_source_fullout_subset.v1"
                   if args.precision else
                   "mx_gemmini.radiance_target_mesh_requant_roster.v1"
                   if args.all_requant and mesh_reference else
                   "mx_gemmini.radiance_source_requant_roster.v1"
                   if args.all_requant else
                   "mx_gemmini.radiance_target_mesh_fullout_roster.v1"
                   if args.all_fullout and mesh_reference else
                   "mx_gemmini.radiance_plain_mx_profile_source_ladder.v1" if base_contract else
                   "mx_gemmini.radiance_target_mesh_reference_cases.v1" if mesh_reference else
                   "mx_gemmini.radiance_selected_mx_profile_source_cases.v1"),
        "status": ("selected_precision_requant_reference_matched_on_pinned_spike"
                   if args.precision and mesh_reference and args.all_requant else
                   "selected_precision_requant_source_matched_on_pinned_spike"
                   if args.precision and args.all_requant else
                   "selected_precision_fullout_reference_matched_on_pinned_spike"
                   if args.precision and mesh_reference else
                   "selected_precision_fullout_source_matched_on_pinned_spike"
                   if args.precision else
                   "requant_target_mesh_reference_roster_matched_on_pinned_spike"
                   if args.all_requant and mesh_reference else
                   "requant_source_header_roster_matched_on_pinned_spike"
                   if args.all_requant else
                   "fullout_target_mesh_reference_roster_matched_on_pinned_spike"
                   if args.all_fullout and mesh_reference else
                   "three_source_precisions_matched_on_pinned_spike"
                   if base_contract else
                   "target_mesh_reference_precisions_matched_on_pinned_spike" if mesh_reference else
                   "selected_source_precisions_matched_on_pinned_spike"),
        "scope": ("archived model2MLIR e9ded36 requant captures rebound to the selected Nicolas MX profile; byte-checked Radiance operands, derived target mesh BF16 references, and derived Radiance header quantized output"
                  if args.all_requant and mesh_reference else
                  "archived model2MLIR e9ded36 requant captures rebound to the selected Nicolas MX profile; source BF16 and quantized goldens from byte-checked Radiance headers"
                  if args.all_requant else
                  "archived model2MLIR e9ded36 captures rebound to Nicolas MxGemminiRocketConfig; source operands and BF16 goldens from byte-checked Radiance headers"
                  if base_contract else
                  "archived model2MLIR e9ded36 captures rebound to the selected Nicolas MX profile; byte-checked Radiance operands and derived target mesh BF16 reference" if mesh_reference else
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
        "rows": results,
    }
    if args.all_requant:
        summary["compared_quantized_output_bytes"] = sum(
            row["compared_quantized_output_bytes"] for row in results)
        summary["compared_output_scales"] = sum(
            row["compared_output_scales"] for row in results)
    else:
        summary["compared_bf16_outputs"] = sum(
            row["compared_bf16_outputs"] for row in results)
    if args.precision:
        summary["selected_precisions"] = sorted(args.precision)
    if not base_contract:
        summary["profile_name"] = profile["name"]
    (out / "qualification.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.all_requant:
        print(f"{profile['name']} {len(selected_cases)} requant cases: "
              f"{summary['compared_quantized_output_bytes']} output bytes and "
              f"{summary['compared_output_scales']} E8M0 scales matched Nicolas Spike")
    else:
        print(f"{profile['name']} {','.join(selected_cases)}: {summary['compared_bf16_outputs']} "
              f"{'target mesh reference' if mesh_reference else 'source'} BF16 outputs matched Nicolas Spike")


if __name__ == "__main__":
    main()
