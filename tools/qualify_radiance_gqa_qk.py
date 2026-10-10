"""Compile a source-derived GQA QK quantization candidate on Nicolas's Spike.

The committed Radiance attention generator's Q/K codes overflow Nicolas's
reduced-precision MX product path. This command changes those bytes under an
explicit E8M0 scale-shift policy. Passing means the candidate first QK tile
matches its hardware-aware oracle; it does not prove attention source parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from mx_gemmini_support.source_attention_qk import (
    SOURCE_REVISION, SUBMODULE_REVISION, GENERATOR_SHA256,
    read_first_gqa_qk, derive_shifted_qk, source_product_overflow_count)
from mx_gemmini_support.source_gemm import SourceGemm
from mx_gemmini_support.source_payload import (
    ATTENTION_QK_CANDIDATE_ORIGIN, make_manifest)
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.qualify_radiance_ws_roster import (
    MODEL2MLIR_REVISION, MXQ_REVISION, RTL_REVISION,
    _revision, _semantic_manifest_sha)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _run(command: list[str], log: Path) -> None:
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "model2mlir-root", "mxq-root", "rtl-root",
                 "riscv-root", "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--scale-shift", required=True, type=int)
    parser.add_argument("--baseline-index", type=Path)
    args = parser.parse_args()
    for name in ("source_root", "model2mlir_root", "mxq_root", "rtl_root",
                 "riscv_root", "mx_opt", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    for path, revision in ((args.source_root, SOURCE_REVISION),
                           (args.source_root / "lib/mxgemmini", SUBMODULE_REVISION),
                           (args.model2mlir_root, MODEL2MLIR_REVISION),
                           (args.mxq_root, MXQ_REVISION),
                           (args.rtl_root, RTL_REVISION)):
        if _revision(path) != revision:
            raise ValueError(f"selected source differs from pinned QK candidate: {path}")
    directory = args.source_root / "kernels/flash_attention_mx_gqa"
    generator = directory / "fa_gen_data.py"
    if _sha(generator) != GENERATOR_SHA256:
        raise ValueError("Radiance GQA data generator changed")
    header = directory / "include/fa_data.h"
    if not header.exists():
        subprocess.run([sys.executable, str(generator), "--Sq", "64", "--Sk", "256",
                        "--d", "64", "--block_n", "64", "--n_q", "8", "--n_kv", "2",
                        "--q_pos0", "64", "--out", "include/fa_data.h"],
                       cwd=directory, check=True)
    tile = read_first_gqa_qk(args.source_root)
    sys.path[:0] = [str(ROOT), str(args.model2mlir_root), str(args.mxq_root),
                    str(args.source_root / "lib/mxgemmini")]
    import torch
    import m2m
    import mxq
    import fp8_matmul_model as low_level_model
    from m2m.capture.external_quantization import ExternalQuantizationConfig
    from m2m.coverage import opaque_report
    from mx_gemmini_support.bind_payload import bind_payload
    from mx_gemmini_support.bind_profile import bind_handoff
    from mx_gemmini_support.handoff import render_handoff, validate_handoff
    from mx_gemmini_support.verify_profile_ir import verify_ir

    if (Path(m2m.__file__).resolve().parents[1] != args.model2mlir_root or
            Path(mxq.__file__).resolve().parents[1] != args.mxq_root or
            Path(low_level_model.__file__).resolve() !=
            (args.source_root / "lib/mxgemmini/fp8_matmul_model.py").resolve()):
        raise ValueError("QK frontend or numerical model resolved to another checkout")
    profile = load_profile(PROFILE, rtl_root=args.rtl_root)
    resources, policy = derive_shifted_qk(tile, args.scale_shift, torch=torch,
                                          low_level_model=low_level_model)

    class Qk(torch.nn.Module):
        def forward(self, query: torch.Tensor, key_transposed: torch.Tensor):
            return torch.matmul(query, key_transposed)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        examples = (torch.randn(64, 64), torch.randn(64, 64))
    contract = ROOT / "mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml"
    quant_policy = ROOT / "examples/default-policy.yaml"
    captured = m2m.convert(
        Qk().eval(), examples,
        quantization=ExternalQuantizationConfig("mx_gemmini", contract, quant_policy),
        backend="fx_importer", capture_trace=True)
    if not captured.ok or opaque_report(captured.mlir_text):
        raise ValueError(f"model2MLIR QK capture failed: {captured.diagnostics}")
    sites = captured.quantization_manifest["sites"]
    if (len(sites) != 1 or
            (sites[0]["site_id"], sites[0]["status"], sites[0]["format"],
             sites[0]["shape"]) !=
            ("functional:matmul", "quantized", "mxfp8", [64, 64, 64])):
        raise ValueError("model2MLIR did not select the source QK contraction")
    validate_handoff(captured, contract.read_bytes(), quant_policy.read_bytes())
    handoff = render_handoff(captured, contract.read_bytes(), quant_policy.read_bytes())
    bound = bind_handoff(handoff, profile)
    source_gemm = SourceGemm(tile.driver, tile.header, (64, 64, 64),
                             (64, 64, 64), "FP8", False, False, True)
    manifest = make_manifest(source_gemm, resources, site_id="functional:matmul",
                             profile_sha256=profile_sha256(profile))
    manifest["origin"] = ATTENTION_QK_CANDIDATE_ORIGIN
    manifest["source_derivation"] = policy
    payload_bound = bind_payload(bound, profile, manifest)
    verify_ir(payload_bound, profile)
    args.out_dir.mkdir(parents=True)
    frontend = args.out_dir / "frontend"
    frontend.mkdir()
    for name, content in (("model2mlir.mlir", captured.mlir_text),
                          ("handoff.mlir", handoff), ("profile_bound.mlir", bound),
                          ("payload_bound.mlir", payload_bound)):
        (frontend / name).write_text(content)
    _run([str(args.mx_opt), str(frontend / "payload_bound.mlir"),
          "-o", "/dev/null"], args.out_dir / "mx_parse.log")
    bundle = args.out_dir / "bundle"
    bundle.mkdir()
    for name, resource in resources.items():
        (bundle / f"{name}.bin").write_bytes(resource.data)
    _write(bundle / "manifest.json", manifest)
    build = args.out_dir / "build"
    _run([sys.executable, "-m", "tools.compile_mx", "--mlir",
          str(frontend / "payload_bound.mlir"), "--bundle", str(bundle),
          "--profile", str(PROFILE), "--rtl-root", str(args.rtl_root),
          "--riscv-root", str(args.riscv_root), "--out-dir", str(build),
          "--run-spike"], args.out_dir / "compile.log")
    receipt = json.loads((build / "artifact_manifest.json").read_text())
    if (receipt["status"] != "source_golden_matched_on_pinned_spike" or
            receipt["compared_bf16_outputs"] != 4096 or
            receipt["source_header_sha256"] != _sha(tile.header)):
        raise ValueError("source-derived QK candidate did not match on Spike")
    # Count the original source pairs that exceed the E4M3 product's max finite
    # value. This is diagnostic; the Spike result below qualifies only the shift.
    over_product = source_product_overflow_count(tile)
    index = {
        "schema": "mx_gemmini.radiance_gqa_qk_candidate.v1",
        "status": "shifted_source_qk_candidate_matched_on_pinned_spike",
        "scope": "first QK tile only; changed source operand codes and scales; no full attention or original source golden parity",
        "source_revision": SOURCE_REVISION, "submodule_revision": SUBMODULE_REVISION,
        "model2mlir_revision": MODEL2MLIR_REVISION, "mxq_revision": MXQ_REVISION,
        "rtl_revision": RTL_REVISION, "compiler_revision": _revision(ROOT),
        "source_driver_sha256": _sha(tile.driver),
        "source_header_sha256": _sha(tile.header),
        "source_arrays_sha256": tile.hashes(),
        "original_source_products_over_448": over_product,
        "policy": policy,
        "frontend_mlir_sha256": _sha(frontend / "model2mlir.mlir"),
        "bound_mlir_sha256": _sha(frontend / "payload_bound.mlir"),
        "bundle_manifest_sha256": _sha(bundle / "manifest.json"),
        "physical_program_sha256": _sha(build / "physical_program.json"),
        "files_sha256": receipt["files_sha256"],
        "object_sha256": receipt["object_sha256"],
        "elf_sha256": receipt["elf_sha256"],
        "spike_log_sha256": receipt["spike_log_sha256"],
        "spike_manifest_semantic_sha256": _semantic_manifest_sha(receipt),
        "compared_bf16_outputs": receipt["compared_bf16_outputs"],
    }
    if args.baseline_index and json.loads(args.baseline_index.read_text()) != index:
        raise ValueError("GQA QK candidate or compiled artifacts differ from baseline")
    _write(args.out_dir / "index.json", index)
    print(f"qualified source-derived GQA QK tile: 4096 BF16 outputs; "
          f"original source has {over_product} oversized raw products")


if __name__ == "__main__":
    main()
