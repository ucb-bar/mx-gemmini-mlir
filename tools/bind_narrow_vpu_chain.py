"""Bind a captured PyTorch pair to Nicolas's checked MX/VPU 32-column slice."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from mx_gemmini_support.narrow_vpu_chain import render_narrow_vpu_chain
from mx_gemmini_support.narrow_vpu_source import derive_narrow_vpu_resources
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _require_gitlink, _sha, _source_closure


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture-dir", "rtl-root", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--mx-opt", type=Path)
    args = parser.parse_args()
    for name in ("capture_dir", "rtl_root", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    software = args.rtl_root / "software/gemmini-rocc-tests"
    _require_gitlink(args.rtl_root, "software/gemmini-rocc-tests")
    source = software / "bareMetalC/chain_vpu_spad_requant.c"
    first_source = software / "bareMetalC/matmul_tiled_fp8_64x64_chain.c"
    header = software / "include/matmul_fp8_64x64_chain.h"
    receipt_path = args.capture_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    frontend = args.capture_dir / "nicolas_chain.profile_bound.mlir"
    manifest_path = args.capture_dir / "quantization_manifest.json"
    if (receipt.get("schema") !=
            "mx_gemmini.nicolas_narrow_vpu_chain_model2mlir_capture.v1" or
            receipt.get("first_shape_mnk") != [64, 64, 64] or
            receipt.get("second_shape_mnk") != [64, 32, 64] or
            receipt.get("profile_sha256") != profile_sha256(profile) or
            receipt.get("source_sha256") != _sha(source) or
            receipt.get("header_sha256") != _sha(header) or
            receipt.get("bound_mlir_sha256") != _sha(frontend) or
            receipt.get("manifest_sha256") != _sha(manifest_path) or
            receipt.get("source_mlir_sha256") != _sha(
                args.capture_dir / "nicolas_chain.model2mlir.mlir") or
            receipt.get("handoff_mlir_sha256") != _sha(
                args.capture_dir / "nicolas_chain.handoff.mlir")):
        raise ValueError("narrow MX/VPU capture or source provenance differs")
    resources, facts = derive_narrow_vpu_resources(
        source, header, first_source, profile)
    bound = render_narrow_vpu_chain(
        frontend.read_text(), json.loads(manifest_path.read_text()),
        profile, resources, facts)
    args.out_dir.mkdir(parents=True)
    mlir = args.out_dir / "connected.mlir"
    mlir.write_text(bound)
    if args.mx_opt is not None:
        subprocess.run([str(args.mx_opt.resolve()), str(mlir), "-o", "/dev/null"],
                       check=True)
    for name, data in resources.items():
        (args.out_dir / f"{name}.bin").write_bytes(data)
    binding = {
        "schema": "mx_gemmini.nicolas_narrow_vpu_chain_binding.v1",
        "status": "source_mx_vpu_requant_and_narrow_mm2_bound_to_typed_graph",
        "first_shape_mnk": [64, 64, 64], "second_shape_mnk": [64, 32, 64],
        "reference_kind": "unchanged_nicolas_vpu_chain_left_output_block",
        "capture_receipt_sha256": _sha(receipt_path),
        "source_sha256": _sha(source), "first_source_sha256": _sha(first_source),
        "header_sha256": _sha(header),
        "profile_sha256": profile_sha256(profile),
        "bound_mlir_sha256": _sha(mlir),
        "resources_sha256": {name: hashlib.sha256(data).hexdigest()
                             for name, data in sorted(resources.items())},
        "compiler_revision": _git_revision(ROOT),
        "compiler_source_closure_sha256": _source_closure(
            ROOT, sorted((ROOT / "mx_gemmini_support").glob("*.py")) +
            sorted((ROOT / "tools").glob("*.py"))),
        "rtl_revision": _git_revision(args.rtl_root),
    }
    (args.out_dir / "binding_manifest.json").write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n")
    print(f"bound narrow MX/VPU pair: {mlir}")


if __name__ == "__main__":
    main()
