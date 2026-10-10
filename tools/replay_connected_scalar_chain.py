"""Replay a source-bound two-VPU MX chain on Nicolas's pinned Spike model.

The first operation is Nicolas's MULS ×2. The added ADDS +0 is an explicitly
derived identity operation: it exercises the ordered SSA and physical command
chain while preserving the independent source C1/C2 references.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from mx_gemmini_support.resident_pair_graph import INPUTS
from mx_gemmini_support.resident_vpu_graph import OUTPUTS
from tools.qualify_narrow_vpu_object import _append_zero_adds


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "docs/evidence/"
          "nicolas_narrow_vpu_pair_64x64x64_64x32x64_9cd918c")
PROFILE = (ROOT / "profiles/gemmini-mx-cleanup-266c593/"
           "MxE4M3Fp4VpuGemminiRocketConfig.json")
RTL_REVISION = "266c593f2cb51d7e3fe83fc0317072b585ac3c52"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], log: Path) -> None:
    result = subprocess.run(command, cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            check=False)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {command[2]}; see {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rtl-root", "riscv-root", "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    for name in ("rtl_root", "riscv_root", "mx_opt", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    revision = subprocess.check_output(
        ["git", "-C", str(args.rtl_root), "rev-parse", "HEAD"], text=True).strip()
    baseline = json.loads((SOURCE / "index.json").read_text())
    if (revision != RTL_REVISION or baseline.get("rtl_revision") != revision or
            baseline.get("status") !=
            "source_mx_vpu_and_narrow_mm2_matched_on_pinned_spike" or
            not args.mx_opt.is_file()):
        parser.error("selected RTL, source evidence, or native verifier differs")
    args.out_dir.mkdir(parents=True)
    for part in ("capture", "bound"):
        target = args.out_dir / part
        target.mkdir()
        for source in (SOURCE / part).iterdir():
            data = source.read_bytes()
            (target / (source.name[:-3] if source.suffix == ".gz" else source.name)
             ).write_bytes(gzip.decompress(data) if source.suffix == ".gz" else data)
    mlir = args.out_dir / "connected_adds.mlir"
    mlir.write_text(_append_zero_adds((args.out_dir / "bound/connected.mlir").read_text()))
    abi = args.out_dir / "abi.json"
    abi.write_text(json.dumps({
        "schema": "mx_gemmini.resident_vpu_buffer_map.v1",
        "inputs": {name: name for name in INPUTS},
        "outputs": {name: name for name in OUTPUTS},
    }, indent=2, sort_keys=True) + "\n")
    obj = args.out_dir / "object"
    _run([sys.executable, "-m", "tools.compile_object",
          "--mlir", str(mlir), "--profile", str(PROFILE),
          "--rtl-root", str(args.rtl_root),
          "--riscv-root", str(args.riscv_root),
          "--resources-dir", str(args.out_dir / "bound"),
          "--abi-json", str(abi), "--mx-opt", str(args.mx_opt),
          "--out-dir", str(obj)], args.out_dir / "compile_object.log")
    qualified = args.out_dir / "qualification"
    _run([sys.executable, "-m", "tools.qualify_narrow_vpu_object",
          "--capture-dir", str(args.out_dir / "capture"),
          "--bound-dir", str(args.out_dir / "bound"),
          "--derived-zero-adds-mlir", str(mlir),
          "--object-dir", str(obj), "--rtl-root", str(args.rtl_root),
          "--riscv-root", str(args.riscv_root),
          "--mx-opt", str(args.mx_opt), "--out-dir", str(qualified)],
         args.out_dir / "qualify.log")
    object_manifest = json.loads((obj / "object_manifest.json").read_text())
    receipt = json.loads((qualified / "qualification_manifest.json").read_text())
    commands = json.loads((obj / "physical_program.json").read_text())["commands"]
    vpus = [command for command in commands
            if command["kind"] == "command" and command["funct"] == 33]
    if (len(vpus) != 2 or
            [(item["rs2"]["immediate"] & 0xf,
              item["rs2"]["immediate"] >> 16) for item in vpus] !=
            [(4, 0x4000), (3, 0)] or
            object_manifest.get("allocated_data_section_bytes") != 0 or
            receipt.get("status") !=
            "derived_zero_adds_vpu_chain_matched_on_pinned_spike" or
            receipt.get("spike_exit_code") != 0 or
            (receipt.get("compared_c1_bf16_values"),
             receipt.get("compared_c1_fp8_codes"),
             receipt.get("compared_c1_e8m0_scales"),
             receipt.get("compared_c2_fp8_codes"),
             receipt.get("compared_c2_e8m0_scales")) != (4096, 4096, 128, 2048, 64)):
        raise ValueError("derived connected scalar chain lacks complete Spike parity")
    index = {
        "schema": "mx_gemmini.connected_scalar_chain_spike.v1",
        "status": "two_ordered_vpu_ops_matched_source_identity_on_pinned_spike",
        "scope": "Nicolas source MM1/MM2 plus derived ADDS +0; identity case only",
        "rtl_revision": revision,
        "compiler_revision": object_manifest["compiler_revision"],
        "baseline_index_sha256": _sha(SOURCE / "index.json"),
        "bound_mlir_sha256": _sha(mlir),
        "object_manifest_sha256": _sha(obj / "object_manifest.json"),
        "qualification_manifest_sha256": _sha(qualified / "qualification_manifest.json"),
        "object_sha256": _sha(obj / "mx_issue.o"),
        "elf_sha256": _sha(qualified / "build/mx_program.elf"),
        "spike_log_sha256": _sha(qualified / "build/spike.log"),
        "compared_c1_bf16_values": 4096,
        "compared_c1_fp8_codes": 4096,
        "compared_c1_e8m0_scales": 128,
        "compared_c2_fp8_codes": 2048,
        "compared_c2_e8m0_scales": 64,
    }
    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"two ordered VPU ops matched every source reference on Spike: {args.out_dir}")


if __name__ == "__main__":
    main()
