"""Source data binding is byte-exact and refuses stale or mismatched inputs."""

from __future__ import annotations

import os
import json
from pathlib import Path

import pytest

from mx_gemmini_support.bind_payload import bind_payload
from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.source_payload import load_bundle, manifest_sha256, write_bundle
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.verify_profile_ir import _operation_name, verify_ir


REPO = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("RADIANCE_KERNELS_ROOT", "/nonexistent"))
FP4_SOURCE = Path(os.environ.get("RADIANCE_FP4_GENERATED_ROOT", "/nonexistent"))
PROFILES = REPO / "profiles/gemmini-mx-cleanup-266c593"


@pytest.mark.parametrize("precision,root,driver,profile_name,capture", [
    ("FP8", SOURCE, "mxgemm.fp8.m128n128k512.tm128tn128tk128.fullout.cpp",
     "MxE4M3Fp4VpuGemminiRocketConfig", "gemm"),
    ("FP4", FP4_SOURCE, "mxgemm.fp4.m64n64k128.tm64tn64tk64.fullout.cpp",
     "MxE4M3Fp4VpuGemminiRocketConfig", "fp4"),
    ("FP6", SOURCE, "mxgemm.fp6.m128n128k2048.tm128tn128tk128.fullout.cpp",
     "MxE3M2OnlyGemminiRocketConfig", "fp6"),
])
def test_source_bundle_binds_real_data_to_captured_mlir(
        tmp_path, precision, root, driver, profile_name, capture):
    selected = root / "kernels/gemm_mxgemmini" / driver
    if not selected.is_file() or not selected.with_name(
            f"mxgemm.data.{precision.lower()}.{driver.split('.')[2]}.h").is_file():
        pytest.skip("requires Radiance source data header")
    profile = load_profile(PROFILES / f"{profile_name}.json")
    kernel = read_source_gemm(selected)
    manifest = write_bundle(tmp_path / "bundle", kernel, site_id="functional:matmul",
                            profile_sha256=profile_sha256(profile))
    loaded, resources = load_bundle(tmp_path / "bundle")
    assert loaded == manifest
    assert len(resources["golden_bf16"]) == kernel.shape[0] * kernel.shape[1] * 2
    if precision == "FP6":
        assert all(len(resources[f"{side}_lut"]) == 768
                   for side in ("activation", "weight", "output"))
    mlir = (REPO / f"docs/evidence/model2mlir_radiance_mx_{capture}_bound_20261009.mlir").read_text()
    bound = bind_payload(mlir, profile, manifest)
    assert f'mx.payload_manifest_sha256 = "{manifest_sha256(manifest)}"' in bound
    assert verify_ir(bound, profile)["contracts"] == 1
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin, StringAttr
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser
    from xdsl.printer import Printer
    from io import StringIO
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, bound).parse_module()
    embedded = json.loads(module.attributes["mx.payload_manifest_json"].data)
    assert embedded == manifest
    if precision == "FP6":
        for name in ("activation_lut", "weight_lut", "output_lut"):
            assert embedded["resources"][name]["shape"] == [64, 3]
            assert embedded["resources"][name]["sha256"] == manifest["resources"][name]["sha256"]
    embedded["resources"]["activation"]["sha256"] = "0" * 64
    module.attributes["mx.payload_manifest_json"] = StringAttr(
        json.dumps(embedded, sort_keys=True, separators=(",", ":")))
    stream = StringIO()
    Printer(stream=stream).print_op(module)
    with pytest.raises(ValueError, match="resource manifest differs"):
        verify_ir(stream.getvalue(), profile)
    if precision == "FP6":
        module = Parser(context, bound).parse_module()
        embedded = json.loads(module.attributes["mx.payload_manifest_json"].data)
        embedded["resources"].pop("activation_lut")
        truncated_digest = manifest_sha256(embedded)
        module.attributes["mx.payload_manifest_sha256"] = StringAttr(truncated_digest)
        module.attributes["mx.payload_manifest_json"] = StringAttr(
            json.dumps(embedded, sort_keys=True, separators=(",", ":")))
        contract = next(op for op in module.walk()
                        if _operation_name(op) == "mx_gemmini.contract")
        contract.attributes["payload_manifest_sha256"] = StringAttr(truncated_digest)
        stream = StringIO()
        Printer(stream=stream).print_op(module)
        with pytest.raises(ValueError, match="activation LUT is absent"):
            verify_ir(stream.getvalue(), profile)
    with pytest.raises(ValueError, match="already payload-bound"):
        bind_payload(bound, profile, manifest)
    target = tmp_path / "bundle" / "activation.bin"
    target.write_bytes(bytes([target.read_bytes()[0] ^ 1]) + target.read_bytes()[1:])
    with pytest.raises(ValueError, match="digest or size differs"):
        load_bundle(tmp_path / "bundle")


def test_payload_binding_rejects_wrong_mode():
    fp8 = load_profile(PROFILES / "MxE4M3Fp4VpuGemminiRocketConfig.json")
    fp6 = load_profile(PROFILES / "MxE3M2OnlyGemminiRocketConfig.json")
    mlir = (REPO / "docs/evidence/model2mlir_radiance_mx_fp6_bound_20261009.mlir").read_text()
    manifest = {"profile_sha256": profile_sha256(fp6),
                "origin": "radiance_source_header_specialization",
                "precision": "FP6", "site_id": "functional:matmul"}
    with pytest.raises(ValueError, match="profile digest differs"):
        bind_payload(mlir, fp8, manifest)
