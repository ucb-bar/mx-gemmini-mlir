"""Bind npu's versioned guest C activation callback to one MX source region.

This verifies source bytes before generating a live model-stage adapter. It
does not build or link the guest callback, nor assert numerical/Spike parity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class ModelActivationCallback:
    region_id: str
    settings_json: str
    source_sha256: str
    headers_sha256: tuple[tuple[str, str], ...]
    source_path: Path
    abi_header_path: Path
    symbol: str = "npu_mx_quantize_activation_v1"

    def __post_init__(self) -> None:
        if (not isinstance(self.region_id, str) or not self.region_id or
                self.symbol != "npu_mx_quantize_activation_v1" or
                not isinstance(self.settings_json, str) or
                not isinstance(self.source_sha256, str) or
                len(self.source_sha256) != 64 or
                not isinstance(self.headers_sha256, tuple) or
                "model_activation_abi.h" not in dict(self.headers_sha256)):
            raise ValueError("model activation callback contract is malformed")
        try:
            settings = json.loads(self.settings_json)
            canonical = json.dumps(settings, sort_keys=True,
                                   separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("model activation callback settings are not finite JSON") from error
        if not isinstance(settings, dict) or canonical != self.settings_json:
            raise ValueError("model activation callback settings are not canonical JSON")

    def verify_files(self) -> None:
        if _sha(self.source_path) != self.source_sha256:
            raise ValueError("activation callback source differs from bound SHA-256")
        for name, digest in self.headers_sha256:
            path = (self.abi_header_path if name == "model_activation_abi.h" else
                    self.source_path.parent / name)
            if _sha(path) != digest:
                raise ValueError(f"activation callback header {name} differs from bound SHA-256")


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"activation callback needs a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_model_activation_callback(runtime: dict, settings: dict, *,
                                   region_id: str, source_path: Path,
                                   abi_header_path: Path) -> ModelActivationCallback:
    """Verify the exact user C source and all headers named by host_c_v1."""
    if (not isinstance(runtime, dict) or set(runtime) !=
            {"kind", "symbol", "source_sha256", "headers_sha256"} or
            runtime["kind"] != "host_c_v1" or
            runtime["symbol"] != "npu_mx_quantize_activation_v1" or
            not isinstance(runtime["headers_sha256"], dict) or
            "model_activation_abi.h" not in runtime["headers_sha256"]):
        raise ValueError("model activation callback needs a host_c_v1 recipe")
    if not isinstance(region_id, str) or not region_id:
        raise ValueError("model activation callback needs an exact source region")
    if not isinstance(settings, dict):
        raise ValueError("model activation callback needs finite JSON settings")
    try:
        settings_json = json.dumps(settings, sort_keys=True,
                                   separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("model activation callback needs finite JSON settings") from error
    source_path, abi_header_path = Path(source_path), Path(abi_header_path)
    if _sha(source_path) != runtime["source_sha256"]:
        raise ValueError("activation callback source differs from recipe SHA-256")
    headers = runtime["headers_sha256"]
    for name, digest in headers.items():
        if (not isinstance(name, str) or not name or Path(name).name != name or
                not isinstance(digest, str) or len(digest) != 64):
            raise ValueError("activation callback header manifest is malformed")
        path = abi_header_path if name == "model_activation_abi.h" else source_path.parent / name
        if _sha(path) != digest:
            raise ValueError(f"activation callback header {name} differs from recipe SHA-256")
    return ModelActivationCallback(region_id, settings_json,
                                   runtime["source_sha256"],
                                   tuple(sorted(headers.items())),
                                   source_path, abi_header_path)


def bind_callbacks_from_model_plan(plan: dict, admission: dict, *,
                                   regions: frozenset[str], source_path: Path,
                                   abi_header_path: Path) -> dict[str, ModelActivationCallback]:
    """Select exact admitted npu matrix sites for the direct live-A catalog."""
    if (not isinstance(plan, dict) or
            plan.get("schema") != "npu_exploration.model_quantization_plan.v2" or
            plan.get("status") != "planning_only_no_executable" or
            not isinstance(admission, dict) or
            admission.get("schema") != "mx_gemmini.model_quantization_admission.v1" or
            admission.get("status") != "structural_admission_only_no_executable" or
            admission.get("run_recipe_digest_status") != "recomputed_from_selected_recipe"):
        raise ValueError("live activation binding needs an admitted npu model plan")
    canonical = (json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if (admission.get("canonical_plan_sha256") != hashlib.sha256(canonical).hexdigest() or
            admission.get("source_mlir_sha256") != plan.get("source_mlir_sha256") or
            admission.get("profile_sha256") != plan.get("profile_sha256")):
        raise ValueError("live activation binding plan differs from MX admission")
    quantizer = plan.get("quantizer")
    if (not isinstance(quantizer, dict) or quantizer.get("kind") != "external_wire" or
            not isinstance(quantizer.get("settings"), dict) or
            not isinstance(quantizer.get("activation_runtime"), dict)):
        raise ValueError("live activation binding needs an external host_c_v1 quantizer")
    sites = plan.get("sites")
    if not isinstance(sites, list) or not isinstance(regions, frozenset) or not regions:
        raise ValueError("live activation binding needs exact selected source sites")
    by_region = {site.get("region_id"): site for site in sites if isinstance(site, dict)}
    if len(by_region) != len(sites) or any(region not in by_region for region in regions):
        raise ValueError("live activation binding selected unknown or duplicate source sites")
    result = {}
    for region in sorted(regions):
        site = by_region[region]
        choice = site.get("requested")
        if (not isinstance(choice, dict) or
                site.get("status") != "structural_candidate_not_executable" or
                not isinstance(site.get("shape_mnk"), list) or
                len(site["shape_mnk"]) != 3 or
                choice.get("placement") != "mesh" or
                choice.get("activation_format") != choice.get("weight_format") or
                choice.get("activation_format") not in {"fp4_e2m1", "fp8_e4m3"} or
                choice.get("activation_projection") != "direct" or
                choice.get("weight_projection") != "direct" or
                choice.get("output_format") != "bf16"):
            raise ValueError(f"live activation binding has no direct rank-2 FP4/FP8 site {region}")
        result[region] = bind_model_activation_callback(
            quantizer["activation_runtime"], quantizer["settings"],
            region_id=region, source_path=source_path,
            abi_header_path=abi_header_path)
    return result
