"""Bind the checked-in FP6 GEMM bytes and LUTs to a selected MX RTL profile."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from mx_gemmini_support.source_fp6 import read_source_fp6_payload
from mx_gemmini_support.source_loop_trace import trace_bound_source_gemm_loops
from mx_gemmini_support.source_gemm import (plan_source_gemm, read_source_gemm,
                                            source_scratchpad_bytes)
from mx_gemmini_support.policy import load_policy
from mx_gemmini_support.target_profile import load_profile, profile_sha256


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--rtl-root", required=True, type=Path)
    parser.add_argument("--mlir", required=True, type=Path)
    parser.add_argument("--capture-receipt", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    driver = args.driver.resolve()
    if not driver.is_relative_to(source_root):
        parser.error("source FP6 driver must be inside the selected Radiance checkout")
    kernel = read_source_gemm(driver)
    payload = read_source_fp6_payload(kernel)
    policy = load_policy(args.policy.read_bytes())
    if policy.codebooks("functional:matmul") != (payload.activation_lut_line0,
                                                   payload.weight_lut_line0):
        parser.error("structural FP6 frontend policy must use source LUT line zero")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    if not profile["resources"]["lut"]:
        parser.error("selected MX target profile lacks FP6 LUT memory")
    source_plan = plan_source_gemm(
        kernel, scratchpad_bytes=source_scratchpad_bytes(
            source_root / "lib/mxgemm/mxgemm_lib.hpp"))
    target_plan = plan_source_gemm(
        kernel, scratchpad_bytes=profile["resources"]["scratchpad_bytes"],
        profile=profile)
    capture = json.loads(args.capture_receipt.read_text())
    source_revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    if (capture.get("source_revision") != source_revision or
            capture.get("source_driver_sha256") != _sha(driver) or
            capture.get("source_data_header_sha256") != payload.header_sha256 or
            capture.get("policy_sha256") != _sha(args.policy) or
            capture.get("target_binding", {}).get("profile_sha256") != profile_sha256(profile) or
            capture.get("target_binding", {}).get("bound_mlir_sha256") != _sha(args.mlir)):
        parser.error("FP6 capture differs from selected source, policy, MLIR, or target")
    trace = trace_bound_source_gemm_loops(
        args.mlir.read_text(), kernel, profile=profile,
        source_root=source_root, rtl_root=args.rtl_root)
    if trace["wave_count"] != len(target_plan["waves"]):
        parser.error("source FP6 trace and payload have different K waves")
    receipt = {
        "schema": "mx_gemmini.source_fp6_payload.v1",
        "status": "bound_source_packed_data_and_lut_roundtrip_only",
        "source_revision": source_revision,
        "source_driver": str(driver.relative_to(source_root)),
        "source_driver_sha256": _sha(driver),
        "source_header_sha256": payload.header_sha256,
        "target_profile_name": profile["name"],
        "target_profile_sha256": profile_sha256(profile),
        "model2mlir_revision": capture["model2mlir_revision"],
        "bound_mlir_sha256": _sha(args.mlir),
        "structural_policy_sha256": policy.source_sha256,
        "structural_policy_role": "PyTorch capture uses source LUT line zero as a sample; source payload retains all 64 lines per bank",
        "site_id": trace["site_id"],
        "source_shape": list(kernel.shape),
        "source_tile": list(kernel.tile),
        "k_waves": len(target_plan["waves"]),
        "lut_lines_per_bank": 64,
        "lut_granularity_shift": 1,
        "source_scratchpad_bytes": source_plan["scratchpad_bytes"],
        "target_scratchpad_bytes": target_plan["scratchpad_bytes"],
        "source_c_row": source_plan["c_spad_dest"],
        "target_c_row": target_plan["c_spad_dest"],
        "payload_lengths": {
            name: len(getattr(payload, name))
            for name in ("activation_bytes", "weight_bytes", "activation_lut_bytes",
                         "weight_lut_bytes", "output_lut_bytes", "activation_scale_bytes",
                         "weight_scale_bytes", "golden_bf16_bytes")
        },
        "payload_digests": payload.digests(),
        "scope": "source FP6 payload bound to typed MLIR and selected RTL; no MX execution or numerical parity",
    }
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"bound {receipt['k_waves']} source FP6 K waves and three 64-line LUT banks")


if __name__ == "__main__":
    main()
