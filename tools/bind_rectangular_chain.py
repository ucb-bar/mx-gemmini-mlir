"""Bind Nicolas's captured rectangular pair to checked wire inputs and goldens."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from mx_gemmini_support.rectangular_chain import render_rectangular_chain
from mx_gemmini_support.rectangular_source import derive_rectangular_resources
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _require_gitlink, _sha, _source_closure


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json"


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
    source = software / "bareMetalC/matmul_tiled_fp8_64x96x64.c"
    header = software / "include/matmul_fp8_64x96x64.h"
    b2_header = software / "include/matmul_fp8_128x128_chain.h"
    model = software / "fp8_matmul_model.py"
    capture = json.loads((args.capture_dir / "receipt.json").read_text())
    second_shape = capture.get("second_shape_mnk")
    if second_shape not in ([64, 32, 96], [64, 64, 96]):
        raise ValueError("rectangular capture MM2 shape differs")
    second_width = second_shape[1]
    frontend = args.capture_dir / "nicolas_chain.profile_bound.mlir"
    manifest_path = args.capture_dir / "quantization_manifest.json"
    if (capture.get("schema") !=
            "mx_gemmini.nicolas_rectangular_chain_model2mlir_capture.v1" or
            capture.get("first_shape_mnk") != [64, 96, 64] or
            capture.get("profile_sha256") != profile_sha256(profile) or
            capture.get("source_sha256") != _sha(source) or
            capture.get("header_sha256") != _sha(header) or
            capture.get("b2_header_sha256") != _sha(b2_header) or
            capture.get("bound_mlir_sha256") != _sha(frontend) or
            capture.get("manifest_sha256") != _sha(manifest_path) or
            capture.get("source_mlir_sha256") != _sha(
                args.capture_dir / "nicolas_chain.model2mlir.mlir") or
            capture.get("handoff_mlir_sha256") != _sha(
                args.capture_dir / "nicolas_chain.handoff.mlir")):
        raise ValueError("rectangular capture or source provenance differs")
    resources = derive_rectangular_resources(
        header, b2_header, model_path=model, second_width=second_width)
    bound = render_rectangular_chain(
        frontend.read_text(), json.loads(manifest_path.read_text()),
        profile, resources, source_sha256=_sha(source),
        header_sha256=_sha(header), b2_header_sha256=_sha(b2_header))
    args.out_dir.mkdir(parents=True)
    mlir = args.out_dir / "connected.mlir"
    mlir.write_text(bound)
    if args.mx_opt is not None:
        subprocess.run([str(args.mx_opt.resolve()), str(mlir), "-o", "/dev/null"],
                       check=True)
    for name, data in resources.items():
        (args.out_dir / f"{name}.bin").write_bytes(data)
    receipt = {
        "schema": "mx_gemmini.nicolas_rectangular_chain_binding.v1",
        "status": "source_c1_and_model_c2_bound_to_typed_pair",
        "first_shape_mnk": [64, 96, 64],
        "second_shape_mnk": second_shape,
        "reference_kind": "unchanged_nicolas_c1_and_pinned_model_derived_c2",
        "capture_receipt_sha256": _sha(args.capture_dir / "receipt.json"),
        "source_sha256": _sha(source), "header_sha256": _sha(header),
        "b2_header_sha256": _sha(b2_header), "model_sha256": _sha(model),
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
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"bound rectangular MX pair: {mlir}")


if __name__ == "__main__":
    main()
