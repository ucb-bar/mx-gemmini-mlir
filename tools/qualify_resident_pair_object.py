"""Link a data-free resident-pair object to Nicolas's comparison harness.

The checked source frontend and packed header build the harness. The linked
program uses the supplied object as its only MX command issuer, then compares
all C1/C2 codes and scales on the selected stock Spike extension. The 96-wide
case uses a checked source-wire slice and a pinned model-derived reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _run, _sha
from tools.emit_resident_pair_object import OUTPUTS
from mx_gemmini_support.resident_pair_graph import INPUTS


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("object-dir", "frontend-dir", "rtl-root", "riscv-root", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--mx-opt", type=Path)
    parser.add_argument("--source-rows", type=int, choices=tuple(range(16, 129, 16)),
                        required=True)
    parser.add_argument("--source-width", type=int, choices=(64, 96, 128), default=128)
    parser.add_argument("--baseline-manifest", type=Path)
    args = parser.parse_args()
    if args.source_width == 64 and args.source_rows != 64:
        parser.error("the direct 64³ resident source requires 64 rows")
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    object_dir = args.object_dir.resolve()
    object_manifest_path = object_dir / "object_manifest.json"
    obj = object_dir / "mx_issue.o"
    object_receipt = json.loads(object_manifest_path.read_text())
    if (object_receipt.get("schema") !=
            "mx_gemmini.resident_pair_linkable_object.v1" or
            object_receipt.get("status") != "rv64_rocc_resident_pair_object_built" or
            object_receipt.get("shape_mnk") != [args.source_rows, args.source_width,
                                                 args.source_width] or
            object_receipt.get("profile_sha256") != profile_sha256(profile) or
            object_receipt.get("object_sha256") != _sha(obj) or
            object_receipt.get("allocated_data_section_bytes") != 0 or
            object_receipt.get("embedded_operand_bytes") != 0 or
            object_receipt.get("embedded_golden_bytes") != 0):
        raise ValueError("resident pair object differs from selected source profile")
    abi = object_receipt["buffer_abi"]
    if ({entry.get("slot"): entry.get("name") for entry in abi} !=
            {name: name for name in (*INPUTS, *OUTPUTS)} or
            len(abi) != len(INPUTS) + len(OUTPUTS)):
        raise ValueError("Nicolas source harness needs the standard runtime symbol ABI")
    args.out_dir.mkdir(parents=True)
    reference = args.out_dir / "reference"
    command = [sys.executable, "-m", "tools.qualify_nicolas_resident_128",
               "--rtl-root", str(args.rtl_root.resolve()),
               "--riscv-root", str(args.riscv_root.resolve()),
               "--profile", str(args.profile.resolve()),
               "--connected-frontend-dir", str(args.frontend_dir.resolve()),
               "--source-rows", str(args.source_rows),
               "--source-width", str(args.source_width),
               "--out-dir", str(reference)]
    if args.mx_opt:
        command += ["--mx-opt", str(args.mx_opt.resolve())]
    _run(command, cwd=ROOT, log=args.out_dir / "reference_build.log")
    reference_receipt_path = reference / "build/artifact_manifest.json"
    reference_receipt = json.loads(reference_receipt_path.read_text())
    source_issuer = reference / "build/mx_issue.c"
    if (reference_receipt["spike_exit_code"] != 0 or
            reference_receipt["compared_fp8_codes"] != args.source_rows * args.source_width or
            reference_receipt["compared_e8m0_scales"] !=
            args.source_rows * (args.source_width // 32) or
            object_receipt["issuer_c_sha256"] != _sha(source_issuer) or
            object_receipt["bound_mlir_sha256"] !=
            reference_receipt["bound_mlir_sha256"] or
            object_receipt["rtl_revision"] != reference_receipt["rtl_revision"]):
        raise ValueError("resident pair object differs from source-qualified commands")
    build = reference / "build"
    riscv_cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    spike = args.riscv_root / "bin/spike"
    bench = args.rtl_root / "software/gemmini-rocc-tests/riscv-tests/benchmarks/common"
    harness_objects = sorted((path for path in build.glob("mx_*.o")
                              if path.name != "mx_0.o"),
                             key=lambda path: int(path.stem.removeprefix("mx_")))
    if not harness_objects or not riscv_cc.is_file() or not spike.is_file():
        raise ValueError("source comparison harness or RISC-V tools are incomplete")
    elf = args.out_dir / "linked.elf"
    _run([str(riscv_cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"), str(obj),
          *(str(path) for path in harness_objects), "-lm", "-lgcc", "-o", str(elf)],
         cwd=args.out_dir, log=args.out_dir / "link.log")
    extension = build / "libgemmini.so"
    result = subprocess.run(
        [str(spike), f"--extlib={extension}", "--extension=gemmini", str(elf)],
        cwd=args.out_dir, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False)
    log = args.out_dir / "spike.log"
    log.write_text(result.stdout)
    marker = (f"lowered connected {args.source_rows}x{args.source_width}: C1 0 codes 0 scales; "
              "C2 0 codes 0 scales")
    passed = result.returncode == 0 and marker in result.stdout
    receipt = {
        "schema": "mx_gemmini.resident_pair_object_spike.v1",
        "status": (("derived_source_connected_object_matched_on_pinned_spike" if passed else
                    "derived_source_connected_object_failed_on_pinned_spike")
                   if args.source_width == 96 else
                   ("source_connected_object_matched_on_pinned_spike" if passed else
                    "source_connected_object_failed_on_pinned_spike")),
        "shape_mnk": [args.source_rows, args.source_width, args.source_width],
        "compared_c1_fp8_codes": args.source_rows * args.source_width,
        "compared_c1_e8m0_scales": args.source_rows * (args.source_width // 32),
        "compared_c2_fp8_codes": args.source_rows * args.source_width,
        "compared_c2_e8m0_scales": args.source_rows * (args.source_width // 32),
        "object_manifest_sha256": _sha(object_manifest_path),
        "object_sha256": _sha(obj),
        "source_reference_manifest_sha256": _sha(reference_receipt_path),
        "source_issuer_sha256": _sha(source_issuer),
        "linked_elf_sha256": _sha(elf), "spike_log_sha256": _sha(log),
        "spike_exit_code": result.returncode,
        "profile_sha256": profile_sha256(profile),
        "compiler_revision": _git_revision(ROOT),
        "rtl_revision": _git_revision(args.rtl_root),
        "spike_sha256": _sha(spike),
    }
    if args.source_width == 96:
        receipt["reference_kind"] = "source_wire_slice_with_pinned_mesh_model_outputs"
        receipt["model_sha256"] = reference_receipt["model_sha256"]
    (args.out_dir / "qualification_manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if args.baseline_manifest is not None:
        baseline = json.loads(args.baseline_manifest.read_text())
        stable = ("schema", "status", "shape_mnk", "compared_c1_fp8_codes",
                  "compared_c1_e8m0_scales", "compared_c2_fp8_codes",
                  "compared_c2_e8m0_scales", "object_manifest_sha256",
                  "object_sha256", "source_reference_manifest_sha256",
                  "source_issuer_sha256", "linked_elf_sha256",
                  "spike_log_sha256", "spike_exit_code", "profile_sha256",
                  "rtl_revision", "spike_sha256")
        if any(receipt[key] != baseline.get(key) for key in stable):
            raise ValueError("resident pair object Spike result differs from baseline")
    print(f"{receipt['status']}: {elf}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
