"""Differential review of the pinned MX operand path against microscaling-quant.

This compares operand quantization, not the MX mesh, host operations, or a model.
The external checkout is read-only and must match its reviewed revision.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from .contract import compile_contract
from .torchao_quant import quantize_mx_gemmini, verify_kernel_contract

REFERENCE_COMMIT = "ea3f5bb4ee274e747ebdc70f463314b104bb45d0"
FORMATS = (("mxfp8", "MXFP8_E4M3", None),
           ("mxfp6", "MXFP6_E3M2", None),
           ("mxfp4", "MXFP4", (3, 1)))


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _inputs() -> dict[str, torch.Tensor]:
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16)
    finite = values[torch.isfinite(values)].reshape(-1, 32)
    generator = torch.Generator().manual_seed(29)
    edges = torch.tensor([
        0.0, -0.0, 0.0625, -0.0625, 0.125, 0.25, 0.375, 0.5,
        0.75, 1.0, 1.125, 1.1875, 1.25, 1.5, 1.75, 2.0,
        4.0, 6.0, 28.0, 448.0, 2.0**-23, 2.0**-14,
        2.0**-126, 2.0**-127, 2.0**-133, 2.0**-134,
        -(2.0**-126), -(2.0**-127), -1.125, -1.1875, -1.25, -448.0,
    ]).to(torch.bfloat16)
    return {"all_finite_bf16": finite,
            "random": torch.randn(64, 32, generator=generator).to(torch.bfloat16),
            "zero": torch.zeros(64, 32, dtype=torch.bfloat16),
            "edges": edges.repeat(64, 1)}


def review(reference_root: str | Path, *, contract_bytes: bytes | None = None) -> dict:
    root = Path(reference_root).resolve()
    if _git(root, "rev-parse", "HEAD") != REFERENCE_COMMIT:
        raise ValueError("microscaling-quant revision differs from the reviewed reference")
    if _git(root, "status", "--porcelain", "--", "mxq"):
        raise ValueError("microscaling-quant source tree has local changes")
    if contract_bytes is None:
        contract_bytes = (Path(__file__).parent / "contracts/software-spec.yaml").read_bytes()
    contract = compile_contract(contract_bytes)
    verify_kernel_contract(contract)
    sys.path.insert(0, str(root))
    try:
        from mxq import block, scale_factor
        if not Path(block.__file__).resolve().is_relative_to(root):
            raise ValueError("mxq import did not resolve to the selected reference checkout")
        cases = _inputs()
        results = {}
        for own_format, ref_format, via in FORMATS:
            rows = {}
            for name, value in cases.items():
                ours, _, scales = quantize_mx_gemmini(value, own_format)
                ref_codes, ref_scales = block.mxgemmini.quantize(
                    value, ref_format, axis=-1, rounding_mode="rne",
                    scale_floor=scale_factor.HARDWARE_FLOOR, via=via)
                theirs = (ref_codes * ref_scales).to(torch.bfloat16)
                expected_scales = (torch.log2(ref_scales).to(torch.int32) + 127).to(torch.uint8)
                bit_differences = ours.view(torch.int16) != theirs.view(torch.int16)
                zero_sign = bit_differences & (ours == 0) & (theirs == 0)
                numeric = ours != theirs
                scale = scales != expected_scales
                row = {"elements": value.numel(),
                       "numerical_mismatches": int(numeric.sum()),
                       "scale_mismatches": int(scale.sum()),
                       "zero_sign_differences": int(zero_sign.sum()),
                       "other_bit_mismatches": int((bit_differences & ~zero_sign).sum())}
                if row["numerical_mismatches"] or row["scale_mismatches"] or row["other_bit_mismatches"]:
                    raise ValueError(f"{own_format}/{name} differs from the selected reference: {row}")
                rows[name] = row
            results[own_format] = rows
        return {"schema": "mx_gemmini.operand_review.v1",
                "rtl_commit": contract["rtl_commit"],
                "mxgen_commit": contract["mxgen_commit"],
                "reference_commit": REFERENCE_COMMIT,
                "status": "operand_numerics_crosscheck", "formats": results}
    finally:
        sys.path.pop(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare selected MX operands with microscaling-quant")
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("--contract", type=Path,
                        default=Path(__file__).parent / "contracts/software-spec.yaml")
    args = parser.parse_args()
    print(json.dumps(review(args.reference_root, contract_bytes=args.contract.read_bytes()),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
