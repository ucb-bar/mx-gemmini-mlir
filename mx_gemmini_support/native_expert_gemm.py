"""Generate one profile-bound gpt-oss native-B GEMM with user-selected live A.

This standalone tile bridge verifies checkpoint B bytes and the guest C
activation callback. It has no captured model source site, guest ELF, or Spike
parity; the output status says so explicitly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .dense_contraction import emit_dense_contraction
from .gptoss_native_tile import bind_native_expert_tile
from .model_activation_callback import bind_model_activation_callback
from .target_profile import profile_sha256, require_compute


SCHEMA = "mx_gemmini.gptoss_native_expert_gemm_stage.v1"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_regular(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"native expert stage needs a regular file: {path}")
    return path.read_bytes()


def _unique_json(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON key {name!r}")
        result[name] = value
    return result


def _load(path: Path) -> tuple[dict, bytes]:
    raw = _read_regular(path)
    document = json.loads(raw, object_pairs_hook=_unique_json)
    if not isinstance(document, dict):
        raise ValueError("native expert stage needs a JSON object")
    return document, raw


def _c_bytes(symbol: str, data: bytes) -> str:
    lines = [", ".join(f"0x{value:02x}" for value in data[i:i + 16])
             for i in range(0, len(data), 16)]
    return ("#include <stdint.h>\n"
            f"const uint8_t {symbol}[{len(data)}] __attribute__((aligned(64))) = {{\n"
            + "\n".join(f"  {line}," for line in lines) + "\n};\n")


def emit_native_expert_gemm_stage(*, tile_dir: Path, fixture_dir: Path,
                                  extractor_receipt_path: Path, run_recipe_path: Path,
                                  profile: dict, site_id: str,
                                  callback_source_path: Path, abi_header_path: Path,
                                  out_dir: Path) -> dict:
    """Recheck the original native B receipt and emit an unlinked BF16 stage."""
    tile_dir, fixture_dir, out_dir = Path(tile_dir), Path(fixture_dir), Path(out_dir)
    manifest, manifest_raw = _load(tile_dir / "manifest.json")
    extractor, extractor_raw = _load(Path(extractor_receipt_path))
    run, run_raw = _load(Path(run_recipe_path))
    if (manifest.get("schema") != "mx_gemmini.gptoss_native_expert_tile.v1" or
            manifest.get("profile_sha256") != profile_sha256(profile) or
            not isinstance(manifest.get("hardware_mode"), dict)):
        raise ValueError("native expert tile needs the selected source-bound FP4 profile")
    mode = manifest["hardware_mode"]
    if any(mode.get(key) != value for key, value in {
            "activation_format": "fp4_e2m1", "weight_format": "fp4_e2m1",
            "activation_projection": "direct", "weight_projection": "direct"}.items()):
        raise ValueError("native expert tile is not a direct FP4 A/B mode")
    require_compute(profile, "fp4_e2m1", "fp4_e2m1",
                    activation_projection="direct", weight_projection="direct",
                    pe_mode=mode.get("pe_mode"))
    if profile["legal_compute"].count(mode) != 1:
        raise ValueError("native expert tile mode is ambiguous in selected profile")
    if (not isinstance(site_id, str) or not site_id or
            run.get("schema") != "npu_exploration.model_run.v2" or
            not isinstance(run.get("name"), str) or not run["name"]):
        raise ValueError("native expert stage needs an explicit model run v2 site ID")
    default, overrides = run.get("matrix_default"), run.get("matrix_overrides", {})
    if (not isinstance(default, dict) or not isinstance(overrides, dict) or
            site_id in overrides and not isinstance(overrides[site_id], dict)):
        raise ValueError("native expert stage needs a valid exact-site matrix decision")
    choice = {**default, **overrides.get(site_id, {})}
    source_tensors = manifest.get("source_tensors")
    if (not isinstance(source_tensors, dict) or
            set(source_tensors) != {"blocks", "scales"} or
            any(not isinstance(row, dict) for row in source_tensors.values())):
        raise ValueError("native expert tile has no checkpoint tensor provenance")
    native = choice.get("native_weight")
    expected_native = {
        "checkpoint_revision": manifest.get("checkpoint_revision"),
        "codes_tensor": source_tensors["blocks"].get("name"),
        "scales_tensor": source_tensors["scales"].get("name"),
        "codes_sha256": source_tensors["blocks"].get("raw_sha256"),
        "scales_sha256": source_tensors["scales"].get("raw_sha256"),
    }
    if (choice.get("placement") != "mesh" or
            choice.get("activation_format") != "fp4_e2m1" or
            choice.get("weight_format") != "fp4_e2m1" or
            choice.get("activation_projection") != "direct" or
            choice.get("weight_projection") != "direct" or
            choice.get("activation_mode") != "dynamic_per_call" or
            choice.get("weight_source") != "preserve_native" or
            choice.get("output_format") != "bf16" or
            native != expected_native):
        raise ValueError("native expert stage recipe must select the exact checkpoint FP4 B and live FP4 A")
    quantizer = run.get("quantizer")
    if (not isinstance(quantizer, dict) or quantizer.get("kind") != "external_wire" or
            not isinstance(quantizer.get("provider"), str) or
            not quantizer["provider"] or
            not isinstance(quantizer.get("implementation_sha256"), str) or
            len(quantizer["implementation_sha256"]) != 64):
        raise ValueError("native expert stage needs a user-selected external activation rule")
    callback = bind_model_activation_callback(
        quantizer.get("activation_runtime"), quantizer.get("settings"),
        region_id=site_id, source_path=Path(callback_source_path),
        abi_header_path=Path(abi_header_path))
    activation_info = manifest.get("activation")
    shape = manifest.get("shape_mnk")
    if (not isinstance(shape, list) or len(shape) != 3 or
            any(type(value) is not int or value <= 0 for value in shape) or
            not isinstance(activation_info, dict) or
            activation_info.get("format") != "bf16" or
            activation_info.get("file") != "a_bf16.bin"):
        raise ValueError("native expert tile has invalid original A fixture geometry")
    m, n, k = shape
    original_a = _read_regular(tile_dir / "a_bf16.bin")
    if (_sha(original_a) != activation_info.get("sha256") or
            len(original_a) != activation_info.get("bytes")):
        raise ValueError("native expert tile original A fixture changed")
    rebuilt_b, rebuilt_scales, rebuilt_manifest = bind_native_expert_tile(
        extractor, extractor_raw, fixture_dir,
        projection=manifest["projection"], expert_index=manifest["expert_index"],
        output_start=manifest["output_start"], output_count=manifest["output_count"],
        block_start=manifest["block_start"], block_count=manifest["block_count"],
        activation_bf16=original_a, activation_m=m,
        expected_revision=manifest["checkpoint_revision"], profile=profile)
    b_codes = _read_regular(tile_dir / "b_codes.bin")
    b_scales = _read_regular(tile_dir / "b_scales.bin")
    if (rebuilt_manifest != manifest or b_codes != rebuilt_b or
            b_scales != rebuilt_scales or
            _sha(b_codes) != manifest.get("b_codes_sha256") or
            _sha(b_scales) != manifest.get("b_scales_sha256")):
        raise ValueError("native expert B tile differs from original checkpoint receipt")
    generated = emit_dense_contraction(
        profile, (m, n, k), precision="FP4", symbol="mx_gptoss_expert_tile",
        issuer_symbol="mx_gptoss_expert_issue", output_dtype="bf16",
        prepacked_weight_symbols=("gptoss_native_b_codes", "gptoss_native_b_scales"),
        activation_callback=callback)
    if (generated.shape_plan.padded_shape_mnk[1:] != (n, k) or
            len(b_codes) != generated.issuer.buffer_bytes["weight"] or
            len(b_scales) != generated.issuer.buffer_bytes["weight_scales"]):
        raise ValueError("native expert B tile does not fit generated physical GEMM")
    if out_dir.exists():
        raise ValueError("native expert stage output directory must be new")
    out_dir.mkdir(parents=True)
    sources = {
        "native_weight.c": (_c_bytes("gptoss_native_b_codes", b_codes) +
                            _c_bytes("gptoss_native_b_scales", b_scales)).encode(),
        "matrix_issuer.c": generated.issuer.source.encode(),
        "model_entry.c": generated.kernel.source.encode(),
        "user_activation_quantizer.c": _read_regular(callback.source_path),
        "runtime_quant.h": _read_regular(Path(__file__).with_name("runtime_quant.h")),
    }
    if _sha(sources["user_activation_quantizer.c"]) != callback.source_sha256:
        raise ValueError("activation callback source changed after binding")
    for name, digest in callback.headers_sha256:
        original = (callback.abi_header_path if name == "model_activation_abi.h" else
                    callback.source_path.parent / name)
        if name in sources or not name or Path(name).name != name:
            raise ValueError("activation callback header collides with generated stage file")
        raw = _read_regular(original)
        if _sha(raw) != digest:
            raise ValueError("activation callback header changed after binding")
        sources[name] = raw
    for name, raw in sources.items():
        (out_dir / name).write_bytes(raw)
        if _sha(_read_regular(out_dir / name)) != _sha(raw):
            raise ValueError(f"generated native expert stage file changed: {name}")
    receipt = {
        "schema": SCHEMA,
        "status": "generated_source_only_no_model_capture_elf_or_spike_parity",
        "site_id": site_id,
        "site_binding_status": "user_named_expert_fixture_no_model2mlir_region",
        "model": manifest["model"],
        "checkpoint_revision": manifest["checkpoint_revision"],
        "extractor_receipt_sha256": _sha(extractor_raw),
        "native_tile_manifest_sha256": _sha(manifest_raw),
        "profile_sha256": profile_sha256(profile),
        "run_recipe_sha256": _sha(run_raw),
        "callback_source_sha256": callback.source_sha256,
        "callback_headers_sha256": dict(callback.headers_sha256),
        "logical_shape_mnk": shape,
        "physical_shape_mnk": list(generated.shape_plan.padded_shape_mnk),
        "activation_input": "live_contiguous_float32_lhs",
        "original_bf16_a_fixture_sha256": activation_info["sha256"],
        "b_codes_sha256": _sha(b_codes), "b_scales_sha256": _sha(b_scales),
        "checkpoint_source_tensors": source_tensors,
        "output_dtype": "bf16",
        "issuer_mode": generated.issuer_mode,
        "source_files_sha256": {name: _sha(raw) for name, raw in sources.items()},
    }
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
