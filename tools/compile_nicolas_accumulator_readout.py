"""Compile Nicolas DRAMMvout source geometries to unqualified accumulator objects.

This command reuses the archived pinned model2MLIR handoff and source payload.
It checks the source C hash and audits the generated accumulator command fields.
The original hardware C path uses an MMIO gateway and constant E8M0 scales;
this Rocket RoCC object uses the source header scales. No FPGA or Spike result
is claimed for this route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from mx_gemmini_support.accumulator_readback import bind_accumulator_dram_readout
from mx_gemmini_support.bind_payload import bind_payload, select_bf16_output_layout
from mx_gemmini_support.bind_profile import bind_handoff
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile
from tools.qualify_nicolas_plain_matrix_object import CASES, RTL_REVISION


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_dram_mvout_spike_fallback_public_6ed3fcb_266c593"
CASES_HARDWARE = {
    "fp8_64x64x64_dram_mvout_spike": (128, 4),
    "fp4_64x64x64_dram_mvout_spike": (256, 4),
    "fp8_128x128x256_dram_mvout_spike": (256, 16),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                   text=True).strip()


def _audit_commands(physical: dict, case_key: str, source_hash: str) -> dict:
    case = CASES[case_key]
    m, n, _ = case.shape
    precision = case.precision
    stride, expected_count = CASES_HARDWARE[case_key]
    tiles_i = m // (32 if precision == "FP4" else 16)
    tiles_j = n // (32 if precision == "FP4" else 16)
    j_groups = tiles_j if precision == "FP4" else tiles_j // 4
    expected = [(i * (32 if precision == "FP4" else 16) * n * 2 + j * n,
                 0x80000000 + (i * j_groups + j) * 16)
                for i in range(tiles_i) for j in range(j_groups)]
    steps = physical["steps"]
    configs = [s["command"] for s in steps if s["phase"] == "configure" and
               s["command"].get("funct") == 0 and
               s["command"]["rs1"]["immediate"] == 2]
    compute = [s["command"] for s in steps if s["phase"] == "compute" and
               s["command"].get("funct") == 8]
    moves = [s["command"] for s in steps if s["phase"] == "readout" and
             s["command"].get("funct") == 3]
    got = [(c["rs1"]["byte_offset"], c["rs2"]["immediate"] & 0xffffffff)
           for c in moves]
    if (physical["mode"] != "rtl_accumulator" or
            physical["plan"]["readout_source_sha256"] != source_hash or
            physical["plan"]["hardware_numerical_qualification"] != "unqualified" or
            physical.get("source_golden_preserving") is not False or
            len(configs) != 1 or configs[0]["rs2"]["immediate"] != stride or
            len(compute) != case.shape[2] // case.tile[2] or
            {c["rs2"]["immediate"] for c in compute} != {
                (0x80000000 << 32) | 0x2b8} or
            len(moves) != expected_count or got != expected or
            any(c["rs1"]["buffer"] != "output_bf16" or
                c["rs2"]["immediate"] >> 32 != (16 << 16) | 16
                for c in moves)):
        raise ValueError("generated accumulator route differs from pinned source geometry")
    return {"schema": "mx_gemmini.nicolas_accumulator_command_audit.v1",
            "case": case_key, "source_driver_sha256": source_hash,
            "status": "accumulator_compute_and_mvout_geometry_matched",
            "compute_waves": len(compute), "mvout_commands": len(moves),
            "mvout_byte_offsets_and_acc_rows": [list(pair) for pair in got],
            "store_stride_bytes": stride,
            "numerical_qualification": "unqualified",
            "source_hardware_transport": "mx_mmio_gateway",
            "compiler_transport": "rocket_rocc",
            "source_hardware_scale_loading": "constant_e8m0_0x7f_mmio",
            "compiler_scale_loading": "source_header_2d_rocc"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES_HARDWARE, required=True)
    for name in ("profile", "rtl-root", "riscv-root", "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    for name in ("profile", "rtl_root", "riscv_root", "mx_opt", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    if _revision(args.rtl_root) != RTL_REVISION:
        raise ValueError("selected Nicolas RTL revision differs from pinned source")
    case = CASES[args.case]
    source = args.rtl_root / "software/gemmini-rocc-tests/bareMetalC" / case.source_name
    if _sha(source) != case.source_sha256:
        raise ValueError("selected Nicolas DRAMMvout C hash differs")
    source_text = source.read_text()
    for marker in ("uint32_t c_dest = acc_addr;", "uint32_t c_flag = 0xb8;",
                   "gemmini_mvout((void *) dram_ptr, acc_tile_addr);",
                   "sf_mem[i] = 0x7f7f7f7f7f7f7f7f;"):
        if marker not in source_text:
            raise ValueError("Nicolas hardware branch no longer matches audited route")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    evidence = ARCHIVE / args.case
    manifest, resources = load_bundle(evidence / "bundle")
    if manifest["source_driver_sha256"] != case.source_sha256:
        raise ValueError("archived source payload differs from pinned C")
    if all(value == 0x7f for name in ("activation_scales", "weight_scales")
           for value in resources[name]):
        raise ValueError("source header scales no longer expose hardware scale divergence")
    text = bind_handoff((evidence / "handoff.mlir").read_text(), profile)
    text = bind_payload(text, profile, manifest)
    text = select_bf16_output_layout(text, profile, manifest)
    text = bind_accumulator_dram_readout(text, profile, manifest)
    args.out_dir.mkdir(parents=True)
    bound = args.out_dir / "accumulator_bound.mlir"
    bound.write_text(text)
    object_dir = args.out_dir / "object"
    command = [sys.executable, "-m", "tools.compile_object",
               "--mlir", str(bound), "--bundle", str(evidence / "bundle"),
               "--profile", str(args.profile), "--rtl-root", str(args.rtl_root),
               "--riscv-root", str(args.riscv_root), "--mx-opt", str(args.mx_opt),
               "--out-dir", str(object_dir)]
    compiled = subprocess.run(command, cwd=ROOT, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (args.out_dir / "compile.log").write_text(compiled.stdout)
    if compiled.returncode:
        raise RuntimeError("MX accumulator object compilation failed; see compile.log")
    physical_path = object_dir / "physical_program.json"
    physical = json.loads(physical_path.read_text())
    audit = _audit_commands(physical, args.case, case.source_sha256)
    object_manifest = json.loads((object_dir / "object_manifest.json").read_text())
    if (object_manifest["mode"] != "rtl_accumulator" or
            object_manifest["allocated_data_section_bytes"] != 0 or
            object_manifest["stock_spike_supported"] is not False):
        raise ValueError("accumulator object lost its execution-scope guard")
    audit.update({"rtl_revision": RTL_REVISION,
                  "source_handoff_sha256": _sha(evidence / "handoff.mlir"),
                  "source_bundle_manifest_sha256": _sha(evidence / "bundle/manifest.json"),
                  "bound_mlir_sha256": _sha(bound),
                  "physical_program_sha256": _sha(physical_path),
                  "object_sha256": _sha(object_dir / "mx_issue.o"),
                  "object_manifest_sha256": _sha(object_dir / "object_manifest.json")})
    (args.out_dir / "accumulator_command_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(object_dir / "mx_issue.o")


if __name__ == "__main__":
    main()
