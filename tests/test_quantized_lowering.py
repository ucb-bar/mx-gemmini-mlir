"""Source-specialized FP8 requantization remains a typed physical lowering."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mx_gemmini_support.bind_payload import bind_payload
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.source_payload import load_bundle, read_source_payload, write_bundle
from mx_gemmini_support.quant_reference import quantize_bf16_fp8_output
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile, profile_sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("RADIANCE_KERNELS_ROOT", "/nonexistent"))
RTL = Path(os.environ.get("MX_GEMMINI_RTL_ROOT", "/nonexistent"))
MXQ = Path(os.environ.get("MXQ_ROOT", "/nonexistent"))


@pytest.mark.parametrize("m,n,k", [(64, 64, 64), (128, 128, 256)])
def test_quantized_capture_provenance_and_output_scope(m, n, k):
    stem = f"model2mlir_radiance_mx_fp8_{m}x{n}x{k}_quant"
    evidence = ROOT / "docs/evidence"
    receipt = json.loads((evidence / f"{stem}_capture_receipt.json").read_text())
    assert receipt["source_quant_output"] is True
    assert receipt["mx_support_revision"] == "ea447fe84cb49d01ff53620e30d26c8fc146ac4e"
    assert receipt["model2mlir_revision"] == "7485a829c0195af0ec42820837d609e62e466564"
    assert receipt["source_shape"] == [m, n, k]
    assert receipt["opaque_calls"] == {}
    assert "specializes the terminal readout" in receipt["frontend_output_scope"]
    for field, suffix in (("source_mlir_sha256", "source.mlir"),
                          ("handoff_mlir_sha256", "handoff.mlir"),
                          ("quantization_manifest_sha256", "manifest.json")):
        data = (evidence / f"{stem}_{suffix}").read_bytes()
        assert receipt[field] == hashlib.sha256(data).hexdigest()
    assert receipt["target_binding"]["bound_mlir_sha256"] == hashlib.sha256(
        (evidence / f"{stem}_bound.mlir").read_bytes()).hexdigest()


@pytest.mark.parametrize("m,n,k,tile_k", [(64, 64, 64, 64), (128, 128, 256, 256)])
def test_fp8_quantized_output_lowers_with_checked_source_and_nicolas_goldens(
        tmp_path, m, n, k, tile_k):
    driver = SOURCE / ("kernels/gemm_mxgemmini/"
                       f"mxgemm.fp8.singletile.tm{m}tn{n}tk{tile_k}.requant.cpp")
    header = SOURCE / f"kernels/gemm_mxgemmini/mxgemm.data.fp8.m{m}n{n}k{k}.h"
    if not driver.is_file() or not header.is_file() or not RTL.is_dir():
        pytest.skip("requires generated Radiance requant header and Nicolas RTL")
    profile = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json",
        rtl_root=RTL)
    manifest = write_bundle(tmp_path / "bundle", read_source_gemm(driver),
                            site_id="functional:matmul",
                            profile_sha256=profile_sha256(profile))
    _, resources = load_bundle(tmp_path / "bundle")
    # The model2MLIR contraction is captured from PyTorch; the source payload
    # explicitly specializes its terminal readout to quantized codes and scales.
    stem = f"model2mlir_radiance_mx_fp8_{m}x{n}x{k}_quant"
    frontend = (ROOT / f"docs/evidence/{stem}_bound.mlir").read_text()
    bound = bind_payload(frontend, profile, manifest)
    assert bound == (ROOT / f"docs/evidence/{stem}_payload_bound.mlir").read_text()
    assert '"mx_gemmini.readout_quantized"' in bound
    assert 'mx.output_specialization = "source_header_quantized"' in bound
    program = lower_bound_source(bound, profile, manifest, resources)
    assert program.output_format == "fp8_e4m3"
    receipt = write_standalone_sources(tmp_path / "artifact", program, resources)
    saved = json.loads((ROOT / f"docs/evidence/compiled_mx_fp8_{m}x{n}x{k}_quant_20261009.json").read_text())
    assert saved["files_sha256"] == receipt["files_sha256"]
    assert saved["bound_mlir_sha256"] == hashlib.sha256(bound.encode()).hexdigest()
    assert saved["status"] == "nicolas_oracle_matched_on_pinned_spike"
    assert saved["compiler_revision"] == "ea447fe84cb49d01ff53620e30d26c8fc146ac4e"
    assert saved["compared_fp8_codes"] == m * n
    assert saved["compared_e8m0_scales"] == m * n // 32
    assert saved["source_quant_code_differences"] == receipt["source_quant_code_differences"]
    assert saved["source_quant_scale_differences"] == receipt["source_quant_scale_differences"]
    assert receipt["golden_basis"] == "nicolas_mxquant_po2_rne_from_source_bf16"
    assert receipt["source_quant_code_differences"] > 0
    assert receipt["source_quant_scale_differences"] > 0
    driver_c = (tmp_path / "artifact/mx_driver.c").read_text()
    assert "FP8 code mismatches" in driver_c
    assert "E8M0 scale mismatches" in driver_c


def test_nicolas_oracle_agrees_with_pinned_mxquant_for_full_source_output():
    driver = SOURCE / "kernels/gemm_mxgemmini/mxgemm.fp8.singletile.tm128tn128tk256.requant.cpp"
    header = SOURCE / "kernels/gemm_mxgemmini/mxgemm.data.fp8.m128n128k256.h"
    if not driver.is_file() or not header.is_file() or not MXQ.is_dir():
        pytest.skip("requires generated source header and pinned MXQuant")
    revision = subprocess.check_output(["git", "-C", str(MXQ), "rev-parse", "HEAD"],
                                       text=True).strip()
    assert revision == "b4af5430bac147f4a16126931cc0177367cc3982"
    sys.path.insert(0, str(MXQ))
    import torch
    from mxq.block.mxgemmini import quantize
    from mxq.scale_factor import HARDWARE_FLOOR

    kernel = read_source_gemm(driver)
    source = read_source_payload(kernel)
    bf16 = source["golden_bf16"].data
    ours_codes, ours_scales = quantize_bf16_fp8_output(bf16, 128, 128)
    values = torch.frombuffer(bytearray(bf16), dtype=torch.uint16).view(
        torch.bfloat16).reshape(128, 128).float()
    codes, scales = quantize(values, "MXFP8_E4M3", axis=1,
                             rounding_mode="rne", scale_floor=HARDWARE_FLOOR)
    mxq_codes = bytes(codes.to(torch.float8_e4m3fn).view(torch.uint8).flatten().tolist())
    mxq_scales = bytes((torch.log2(scales).to(torch.int16) + 127).flatten().tolist())
    assert ours_codes == mxq_codes
    assert ours_scales == mxq_scales


@pytest.mark.parametrize("m,n,k", [(64, 64, 64), (128, 128, 128)])
def test_fp4_quantized_capture_and_packed_spike_receipt(tmp_path, m, n, k):
    evidence = ROOT / "docs/evidence"
    stem = f"model2mlir_radiance_mx_fp4_{m}x{n}x{k}_quant"
    capture = json.loads((evidence / f"{stem}_capture_receipt.json").read_text())
    assert capture["source_quant_output"] is True
    assert capture["mx_support_revision"] == "19cf6cfdc1553c1f05927e5dbf661792829cee98"
    assert capture["model2mlir_revision"] == "7485a829c0195af0ec42820837d609e62e466564"
    assert capture["source_shape"] == [m, n, k]
    assert capture["opaque_calls"] == {}
    for field, suffix in (("source_mlir_sha256", "source.mlir"),
                          ("handoff_mlir_sha256", "handoff.mlir"),
                          ("quantization_manifest_sha256", "manifest.json")):
        assert capture[field] == hashlib.sha256(
            (evidence / f"{stem}_{suffix}").read_bytes()).hexdigest()
    driver = SOURCE / ("kernels/gemm_mxgemmini/"
                       f"mxgemm.fp4.singletile.tm{m}tn{n}tk{k}.requant.cpp")
    header = SOURCE / f"kernels/gemm_mxgemmini/mxgemm.data.fp4.m{m}n{n}k{k}.h"
    if not driver.is_file() or not header.is_file() or not RTL.is_dir():
        pytest.skip("requires generated Radiance FP4 header and Nicolas RTL")
    profile = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json",
        rtl_root=RTL)
    manifest = write_bundle(tmp_path / "bundle", read_source_gemm(driver),
                            site_id="functional:matmul",
                            profile_sha256=profile_sha256(profile))
    _, resources = load_bundle(tmp_path / "bundle")
    bound = bind_payload((evidence / f"{stem}_bound.mlir").read_text(), profile, manifest)
    assert bound == (evidence / f"{stem}_payload_bound.mlir").read_text()
    assert '"mx_gemmini.readout_quantized"' in bound
    program = lower_bound_source(bound, profile, manifest, resources)
    assert program.output_format == "fp4_e2m1"
    generated = write_standalone_sources(tmp_path / "artifact", program, resources)
    saved = json.loads((evidence / f"compiled_mx_fp4_{m}x{n}x{k}_quant_20261009.json").read_text())
    assert saved["files_sha256"] == generated["files_sha256"]
    assert saved["bound_mlir_sha256"] == hashlib.sha256(bound.encode()).hexdigest()
    assert saved["compiler_revision"] == "19cf6cfdc1553c1f05927e5dbf661792829cee98"
    assert saved["status"] == "nicolas_oracle_matched_on_pinned_spike"
    assert saved["compared_fp4_packed_bytes"] == m * n // 2
    assert saved["compared_e8m0_scales"] == m * n // 32
    assert saved["source_quant_code_format"] == "fp8_e4m3"
    assert saved["target_quant_code_format"] == "packed_fp4_e2m1"
    assert saved["source_quant_scale_differences"] == m * n // 32
