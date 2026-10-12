"""Project a selected MX RTL profile into Merlin's generic capability grammar.

This is structural capability metadata, not Phase 0 admission or an executable
backend.  The source profile is checked separately against its RTL checkout.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from .target_profile import SCHEMA, load_profile, profile_sha256


_DIRECT_FORMATS = {
    "fp4_e2m1": "mxfp4",
    "fp8_e4m3": "mxfp8",
}


def contract_for_profile(profile: dict) -> dict:
    """Declare only direct operand modes and VPU work the profile establishes."""
    if profile.get("schema") != SCHEMA or profile.get("qualification") != "structural_unqualified":
        raise ValueError("expected a structurally observed MX target profile")
    source, resources = profile.get("source"), profile.get("resources")
    if not isinstance(source, dict) or not isinstance(resources, dict):
        raise ValueError("MX profile has no source or resource facts")
    if profile.get("transport") != "rocket_rocc":
        raise ValueError("Merlin whole-model contract requires a Rocket RoCC profile")
    modes = profile.get("legal_compute")
    if not isinstance(modes, list) or not modes:
        raise ValueError("MX profile has no legal compute modes")
    declared: list[dict] = []
    rules: list[dict] = []
    seen_modes: set[tuple[str, str, int]] = set()
    for mode in modes:
        if not isinstance(mode, dict):
            raise ValueError("malformed MX compute mode")
        lhs, rhs = mode.get("activation_format"), mode.get("weight_format")
        if lhs not in _DIRECT_FORMATS or rhs not in _DIRECT_FORMATS:
            # Other profiles can still be selected through their own contracts.
            # This projector never aliases E2M3/E3M2 or a LUT to direct FP8/FP4.
            continue
        if mode.get("activation_projection") != "direct" or mode.get("weight_projection") != "direct":
            continue
        pe_mode = mode.get("pe_mode")
        if type(pe_mode) is not int:
            raise ValueError("MX profile direct mode has no integer PE selector")
        mode_key = (lhs, rhs, pe_mode)
        if mode_key in seen_modes:
            continue
        seen_modes.add(mode_key)
        rule = {"in": _DIRECT_FORMATS[lhs], "weight": _DIRECT_FORMATS[rhs], "acc": "bf16"}
        if rule not in rules:
            rules.append(rule)
        declared.append({"activation_format": lhs, "weight_format": rhs, "pe_mode": pe_mode})
    if not rules or resources.get("scale_block_elements") != 32:
        raise ValueError("MX profile has no supported direct E8M0 compute path")
    unit = {
        "name": "mx_pe", "kind": "systolic",
        "dtypes": sorted({dtype for rule in rules for dtype in (rule["in"], rule["weight"])}),
        "ops": ["matmul"], "scaling": "block_e8m0",
        "requant": {"ref": "mx_gemmini.spad_requant"},
        "exposure": "external_backend", "accumulate": rules,
        "semantic_capabilities": [{
            "family": "contraction", "dtypes": sorted({rule["in"] for rule in rules}),
            "operand_pairs": [{"in": rule["in"], "weight": rule["weight"]} for rule in rules],
            "ranks": [2, 3, 4], "arbitrary_mnk": False, "batch": True,
            "notes": "MX source tails require explicit zero padding and scale regeneration",
        }],
    }
    units = [unit]
    if resources.get("vpu") is True:
        config = resources.get("vpu_config") or {}
        if not isinstance(config, dict) or not config.get("exp_sub") or not config.get("exp_sum"):
            raise ValueError("MX VPU profile lacks the observed softmax primitives")
        units.append({
            "name": "vpu", "kind": "vector", "dtypes": ["bf16"],
            "ops": ["softmax", "add", "sub", "mul", "div", "rsqrt", "sigmoid", "square", "neg", "gelu", "pow10000", "rsum", "rmax"], "exposure": "external_backend",
            "semantic_capabilities": [{
                "family": "softmax", "dtypes": ["bf16"], "ranks": [2],
                "notes": "Source-checked row softmax; f32 boundaries explicitly round to BF16",
            }, {
                "family": "elementwise_map", "dtypes": ["bf16"], "ranks": [1, 2, 3, 4],
                "notes": "Source-checked add, sub, mul, rcp-plus-mul div, rsqrt, sigmoid, square, neg, tanh-GELU, and 10000-to-variable power with explicit BF16 rounding at f32 boundaries",
            }, {
                "family": "reduction", "dtypes": ["bf16"], "ranks": [2, 3, 4],
                "notes": "Source-checked last-axis sum and max with identity init and explicit f32/BF16 boundaries",
            }],
        })
    return {
        "name": "mx_gemmini", "version": "0.2", "family": "tensor_resident",
        "endpoint_kind": "external_backend",
        "source": {"rtl_repository": "https://github.com/ucb-bar/gemmini",
                   "rtl_commit": source.get("gemmini_revision"),
                   "mxgen_commit": source.get("mxgen_revision"),
                   "rtl_config": profile.get("name"),
                   "profile_sha256": profile_sha256(profile),
                   "profile_source_sha256": copy.deepcopy(source.get("sha256"))},
        "compute_units": units,
        "mx_direct_modes": declared,
        "mx_resources": copy.deepcopy(resources),
        "runtime": {"default_backend": "simulator"},
        "compiler_obligations": [
            "bind exact profile and numerical policy to every generated object",
            "pad and re-quantize unsupported tails before MX dispatch",
            "account for every captured operation and provide a compiled host owner",
            "preserve MX/VPU fences, scales, live state, and full result readback",
        ],
        "certification": {"status": "structural_unqualified", "phase0_admitted": False},
        "notes": "Source-derived capability selection only; Spike and RTL qualification are separate.",
    }


def render_contract(profile: dict) -> str:
    return yaml.safe_dump(contract_for_profile(profile), sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the selected MX profile as Merlin capabilities")
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--rtl-root", type=Path, help="verify the profile against the selected RTL sources")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--check", action="store_true", help="compare existing output without writing")
    args = parser.parse_args()
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    rendered = render_contract(profile)
    if args.check:
        if args.out.read_text(encoding="utf-8") != rendered:
            raise SystemExit("selected Merlin MX contract differs from profile projection")
    else:
        args.out.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
