"""Compile a receipt-bound model2MLIR stage with Merlin and the MX catalog.

This is an experimental whole-stage build/run probe. Its result is not a
quantized PyTorch parity certificate or a complete multi-stage session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture_model(stage: Path) -> Path:
    source = stage / "model.mlir"
    receipt = stage / "capture_receipt.json"
    if any(path.is_symlink() or not path.is_file() for path in (source, receipt)):
        raise ValueError("MX whole-stage build needs plain model.mlir and capture_receipt.json")
    data = json.loads(receipt.read_text(encoding="utf-8"))
    record = (data.get("artifacts") or {}).get("model.mlir") or {}
    if (data.get("schema") != "m2m.capture-receipt.v1"
            or record.get("bytes") != source.stat().st_size
            or record.get("sha256") != hashlib.sha256(source.read_bytes()).hexdigest()):
        raise ValueError("MX whole-stage build refuses a changed model2MLIR capture source")
    return source


def _can_hash_first_output(stage: Path) -> bool:
    """Enable Merlin's canonical first-output digest for supported scalar types."""
    golden = stage / "golden.npy"
    if not golden.is_file() or golden.is_symlink():
        raise ValueError("MX whole-stage build needs a plain golden.npy")
    return np.load(golden, mmap_mode="r", allow_pickle=False).dtype in {
        np.dtype("float32"), np.dtype("int64"), np.dtype("bool")}


def _load_precision_overrides(path: Path, source_sha: str,
                              profile_sha: str) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("MX precision overrides need a regular JSON file")
    data = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or data.get("schema") != "mx_gemmini.precision_overrides.v1"
            or data.get("source_mlir_sha256") != source_sha
            or data.get("profile_sha256") != profile_sha
            or not isinstance(data.get("regions"), dict)):
        raise ValueError("MX precision overrides differ from the captured source or profile")
    return data["regions"]


def _raw_selected_regions(source: Path, profile: dict, *,
                          precision_overrides: dict[str, str] | None = None) -> set[str]:
    from mx_gemmini_support.merlin_inventory import build_mx_worklist_from_merlin
    from mx_gemmini_support.model_precision_plan import build_model_precision_plan

    worklist = build_mx_worklist_from_merlin(source.read_text(encoding="utf-8"), profile)
    precision = build_model_precision_plan(
        worklist, profile, precision_overrides=precision_overrides)
    regions = [site["region_id"] for site in precision["sites"]
               if site["selected_format"] in {"mxfp4", "mxfp8"}]
    if len(regions) != len(set(regions)):
        raise ValueError("MX raw capture has duplicate selected source regions")
    return set(regions)


def _require_source_coverage(expected_regions: set[str], catalog: dict) -> None:
    actual_regions = [binding["region"] for binding in catalog["bindings"]]
    if len(actual_regions) != len(set(actual_regions)):
        raise RuntimeError("MX linked catalog has duplicate source regions")
    missing_regions = sorted(expected_regions - set(actual_regions))
    extra_regions = sorted(set(actual_regions) - expected_regions)
    if missing_regions or extra_regions or catalog["unbound_precision_proposals"]:
        raise RuntimeError(
            "MX stage build did not cover every raw selected operation: "
            f"missing={missing_regions}, extra={extra_regions}, "
            f"unbound={len(catalog['unbound_precision_proposals'])}"
        )


def _observe_spike_evidence(console: str, source: Path, build_hash: str,
                            *, first_typed: bool) -> dict:
    """Check completed console records against the captured result ABI."""
    from merlin.common.mlir_query import forward_signature
    from merlin.runtime.backends.spike_model import parse_console
    from merlin.runtime.output_digest import observe_output_digests

    _, outputs = forward_signature(source)
    parsed = parse_console(console)
    if parsed["output_dtype"] != outputs[0][1]:
        raise ValueError("Spike first-output wire dtype differs from the capture")
    hash_lines = [line for line in console.splitlines()
                  if line.startswith("METRIC build_hash ")]
    if hash_lines != [f"METRIC build_hash {build_hash}"]:
        raise ValueError("Spike build hash differs from the linked stage")
    if parsed["metrics"].get("memref_rank_mismatch") != 0:
        raise ValueError("Spike reported a memref rank mismatch")
    return observe_output_digests(console, outputs, require_first_typed=first_typed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--rtl-root", type=Path)
    parser.add_argument("--merlin-src", type=Path, required=True)
    parser.add_argument("--llvm-install", type=Path, required=True)
    parser.add_argument("--riscv-root", type=Path, required=True)
    parser.add_argument("--arena-mb", type=int,
                        help="explicit Spike arena size; omitted uses Merlin's LLVM-site estimate")
    parser.add_argument("--output-dump-cap", type=int, default=1)
    parser.add_argument("--diagnostic-host-vpu", action="store_true",
                        help="keep current MX contraction routing but execute VPU-eligible "
                             "operations on the host for a numerical control")
    parser.add_argument("--cache-checkpoint-weights", action="store_true",
                        help="pack source-proven checkpoint RHS values once per compiled stage")
    parser.add_argument("--precision-overrides", type=Path,
                        help="receipt-bound FP8 overrides for sensitive FP4 source regions")
    parser.add_argument("--compile-timeout-s", type=int,
                        default=int(os.environ.get("MERLIN_COMPILE_TIMEOUT_S", "3600")),
                        help="wall-clock limit per compiler/link command; 0 disables it")
    parser.add_argument("--spike-extlib", type=Path,
                        help="run the built ELF on Spike with this pinned Gemmini extension")
    parser.add_argument("--spike-timeout", type=int, default=600)
    args = parser.parse_args()
    stage = args.stage.resolve(strict=True)
    work = args.workdir.absolute()
    if work.exists():
        parser.error("--workdir must not already exist")
    source = _capture_model(stage)
    output_sha256 = _can_hash_first_output(stage)
    capture_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if ((args.arena_mb is not None and args.arena_mb <= 0)
            or args.output_dump_cap <= 0 or args.spike_timeout <= 0
            or args.compile_timeout_s < 0):
        parser.error("arena, dump cap and Spike timeout must be positive; compile timeout cannot be negative")
    profile_path = args.profile.resolve(strict=True)
    riscv_root = args.riscv_root.resolve(strict=True)
    llvm_root = args.llvm_install.resolve(strict=True)
    merlin_src = args.merlin_src.resolve(strict=True)
    if not (merlin_src / "merlin/runtime/backends/spike_model.py").is_file():
        parser.error("--merlin-src must be Merlin's src directory")
    from mx_gemmini_support.target_profile import load_profile

    profile = load_profile(profile_path, rtl_root=args.rtl_root)
    profile_sha = profile_sha256(profile)
    if args.precision_overrides is not None and args.precision_overrides.is_symlink():
        parser.error("--precision-overrides must be a regular file")
    override_path = (args.precision_overrides.resolve(strict=True)
                     if args.precision_overrides is not None else None)
    overrides = (_load_precision_overrides(override_path, capture_sha, profile_sha)
                 if override_path is not None else None)
    override_sha = _sha(override_path) if override_path is not None else None
    os.environ["MERLIN_TARGET_PATH"] = str(ROOT)
    os.environ["MERLIN_COMPILER_PYTHON"] = sys.executable
    os.environ["MERLIN_MLIR_INSTALL"] = str(llvm_root)
    os.environ["MERLIN_CLANG"] = str(llvm_root / "bin/clang")
    os.environ["MERLIN_RISCV_GCC"] = str(riscv_root / "bin/riscv64-unknown-elf-gcc")
    os.environ["MERLIN_SPIKE"] = str(riscv_root / "bin/spike")
    os.environ["MERLIN_COMPILE_TIMEOUT_S"] = str(args.compile_timeout_s)
    sys.path.insert(0, str(merlin_src))
    from merlin.llvmlower.device_build import DeviceRouting
    from merlin.runtime.backends import spike_model
    from mx_gemmini_support.merlin_bf16_prepare import MXBF16CatalogPreparation
    from mx_gemmini_support.merlin_catalog_builder import make_merlin_catalog_builder
    from mx_gemmini_support.merlin_vpu_inventory import audit_softmax_boundaries
    from mx_gemmini_support.merlin_vpu_softmax import selected_softmax_sites
    from mx_gemmini_support.merlin_vpu_add import MXVPUPostOffload, selected_add_sites
    from mx_gemmini_support.merlin_vpu_elementwise import selected_elementwise_sites
    from mx_gemmini_support.merlin_vpu_unary import selected_unary_sites
    from mx_gemmini_support.merlin_vpu_reduce import selected_row_reduce_sites
    from mx_gemmini_support.target_profile import profile_sha256

    vpu_inventory = audit_softmax_boundaries(source.read_text(encoding="utf-8"), profile)
    vpu_selected = selected_softmax_sites(source.read_text(encoding="utf-8"), profile)
    if vpu_inventory["direct_profile_matches_unlowered"]:
        raise RuntimeError("captured direct BF16 rank-two VPU softmax has no integrated lowering")
    expected_regions = _raw_selected_regions(
        source, profile, precision_overrides=overrides)
    if args.cache_checkpoint_weights and not expected_regions:
        parser.error("checkpoint weight caching requires selected MX contractions")
    weights_path = stage / "weights.safetensors"
    if args.cache_checkpoint_weights and (weights_path.is_symlink()
                                          or not weights_path.is_file()):
        parser.error("checkpoint weight caching requires a regular captured weights file")
    weights_sha = _sha(weights_path) if args.cache_checkpoint_weights else None
    if args.diagnostic_host_vpu and not expected_regions:
        parser.error("--diagnostic-host-vpu requires selected MX contractions")
    device = (DeviceRouting(
        device="mx_gemmini", package_dir=ROOT,
        operand_dtype="catalog_mixed_direct_fp4_fp8", accum_dtype="bf16",
        prepared_transform=MXBF16CatalogPreparation(
            profile, precision_overrides=overrides),
        catalog_builder=make_merlin_catalog_builder(
            profile_path=profile_path, riscv_root=riscv_root,
            cached_weight_stage=stage if args.cache_checkpoint_weights else None,
            precision_overrides=overrides),
        post_offload_transform=(None if args.diagnostic_host_vpu
                                else MXVPUPostOffload(profile)),
    ) if expected_regions else None)
    result = spike_model.build(
        stage, work, arena_mb=args.arena_mb if args.arena_mb is not None else 2048,
        auto_arena=args.arena_mb is None, backend="scalar", device=device,
        cflags_override=["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2"],
        output_dump_cap=args.output_dump_cap, output_sha256=output_sha256,
        output_all_sha256=True,
    )
    if hashlib.sha256(source.read_bytes()).hexdigest() != capture_sha:
        raise RuntimeError("Merlin changed the receipt-bound model2MLIR capture source")
    if weights_sha is not None and _sha(weights_path) != weights_sha:
        raise RuntimeError("captured checkpoint changed during MX stage compilation")
    if override_sha is not None and _sha(override_path) != override_sha:
        raise RuntimeError("MX precision overrides changed during stage compilation")
    elf = Path(result["elf"])
    catalog_path = work / "device_catalog/mx_catalog.json"
    if device is None:
        if catalog_path.exists():
            raise RuntimeError("host-only stage unexpectedly emitted an MX catalog")
        catalog = {"covered_contractions": 0, "unbound_precision_proposals": [],
                   "unique_kernels": 0, "bindings": []}
    else:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        _require_source_coverage(expected_regions, catalog)
        if catalog.get("precision_overrides", {}) != (overrides or {}):
            raise RuntimeError("MX catalog lost the selected precision overrides")
        for binding in catalog["bindings"]:
            if (binding["region"] in (overrides or {}) and
                    binding["target_dtypes"][:2] != ["f8E4M3FN", "f8E4M3FN"]):
                raise RuntimeError("MX catalog override did not bind FP8 operands")
    rewrite_path = work / "device_host_abi/mx_vpu_softmax_rewrite.json"
    lowered_vpu = (json.loads(rewrite_path.read_text(encoding="utf-8"))["regions"]
                   if rewrite_path.is_file() else [])
    selected_vpu_regions = {row["region"] for row in vpu_selected}
    lowered_vpu_regions = {row["region"] for row in lowered_vpu}
    catalog_vpu_regions = set(catalog.get("vpu_softmax", {}).get("source_regions", ()))
    softmax_coverage_invalid = (
        len(lowered_vpu_regions) != len(lowered_vpu)
        or (not args.diagnostic_host_vpu and
            (selected_vpu_regions != lowered_vpu_regions or
             selected_vpu_regions != catalog_vpu_regions))
        or (args.diagnostic_host_vpu and bool(lowered_vpu_regions))
    )
    if softmax_coverage_invalid:
        raise RuntimeError("MX VPU softmax source selection, rewrite, and catalog disagree")
    # The inventory is taken before rewriting; reconcile its ownership with
    # the actual linked catalog so the final receipt cannot label a VPU call
    # as host-owned merely because the direct profile matcher was narrower.
    vpu_inventory["inventory_phase"] = "reconciled_after_link"
    for row in vpu_inventory["regions"]:
        if row["region"] in lowered_vpu_regions:
            row["owner"] = "mx_gemmini_vpu"
            row["reason"] = "source_bound_converted_vpu_row_softmax"
    vpu_inventory["host_owned_regions"] = sum(
        row["owner"] == "compiled_host" for row in vpu_inventory["regions"])
    add_rewrite_path = work / "device_host_abi/mx_vpu_add_rewrite.json"
    lowered_add = (json.loads(add_rewrite_path.read_text(encoding="utf-8"))["regions"]
                   if add_rewrite_path.is_file() else [])
    catalog_add_regions = set(catalog.get("vpu_add", {}).get("source_regions", ()))
    if (len({row["region"] for row in lowered_add}) != len(lowered_add) or
            not {row["region"] for row in lowered_add} <= catalog_add_regions):
        raise RuntimeError("MX VPU add rewrite exceeds its source-bound catalog")
    elementwise_rewrite_path = work / "device_host_abi/mx_vpu_elementwise_rewrite.json"
    lowered_elementwise = (json.loads(elementwise_rewrite_path.read_text(encoding="utf-8"))["regions"]
                           if elementwise_rewrite_path.is_file() else [])
    catalog_elementwise_regions = set(catalog.get("vpu_elementwise", {}).get("source_regions", ()))
    if (len({row["region"] for row in lowered_elementwise}) != len(lowered_elementwise) or
            not {row["region"] for row in lowered_elementwise} <= catalog_elementwise_regions):
        raise RuntimeError("MX VPU elementwise rewrite exceeds its source-bound catalog")
    unary_rewrite_path = work / "device_host_abi/mx_vpu_unary_rewrite.json"
    lowered_unary = (json.loads(unary_rewrite_path.read_text(encoding="utf-8"))["regions"]
                     if unary_rewrite_path.is_file() else [])
    catalog_unary_regions = set(catalog.get("vpu_unary", {}).get("source_regions", ()))
    if (len({row["region"] for row in lowered_unary}) != len(lowered_unary) or
            not {row["region"] for row in lowered_unary} <= catalog_unary_regions):
        raise RuntimeError("MX VPU unary rewrite exceeds its source-bound catalog")
    reduce_rewrite_path = work / "device_host_abi/mx_vpu_reduce_rewrite.json"
    lowered_reduce = (json.loads(reduce_rewrite_path.read_text(encoding="utf-8"))["regions"]
                      if reduce_rewrite_path.is_file() else [])
    catalog_reduce_regions = set(catalog.get("vpu_reduce", {}).get("source_regions", ()))
    if (len({row["region"] for row in lowered_reduce}) != len(lowered_reduce) or
            not {row["region"] for row in lowered_reduce} <= catalog_reduce_regions):
        raise RuntimeError("MX VPU reduction rewrite exceeds its source-bound catalog")
    residual_eligible = {}
    post_receipt = work / "device_host_abi/post_offload_transform.json"
    if post_receipt.is_file():
        selected_path = Path(json.loads(post_receipt.read_text())["selected_path"])
        selected_text = selected_path.read_text()
        residual_eligible = {
            "softmax": len(selected_softmax_sites(selected_text, profile)),
            "add": len(selected_add_sites(selected_text, profile)),
            "elementwise": len(selected_elementwise_sites(selected_text, profile)),
            "unary": len(selected_unary_sites(selected_text, profile)),
            "reduce": len(selected_row_reduce_sites(selected_text, profile)),
        }
        if any(residual_eligible.values()):
            raise RuntimeError(f"MX VPU eligible source sites remained on host: {residual_eligible}")
    bf16_receipt = work / "device_prepared/mx_bf16_catalog.json"
    widened_bf16 = (json.loads(bf16_receipt.read_text(encoding="utf-8"))["widened_operations"]
                    if bf16_receipt.is_file() else 0)
    status = ("linked_diagnostic_host_vpu_numeric_qualification_pending"
              if args.diagnostic_host_vpu else
              "linked_experimental_numeric_qualification_pending")
    route = ("mx_gemmini_and_host_vpu_control" if args.diagnostic_host_vpu else
             "mx_gemmini_and_host" if device is not None else "host_only")
    receipt = {
        "schema": "mx_gemmini.merlin_stage_probe.v1",
        "status": status,
        "capture_model_sha256": capture_sha,
        "profile_sha256": profile_sha256(profile),
        "precision_overrides_sha256": override_sha,
        "precision_overrides": overrides or {},
        "catalog_manifest": str(catalog_path) if device is not None else None,
        "execution_route": route,
        "diagnostic_host_vpu": args.diagnostic_host_vpu,
        "selected_operations": catalog["covered_contractions"],
        "expected_selected_operations": len(expected_regions),
        "raw_selected_source_coverage_complete": True,
        "unbound_precision_proposals": len(catalog["unbound_precision_proposals"]),
        "widened_bf16_contractions": widened_bf16,
        "vpu_softmax_inventory": vpu_inventory,
        "vpu_softmax_selected_regions": len(vpu_selected),
        "vpu_softmax_lowered_regions": len(lowered_vpu),
        "vpu_softmax_host_regions": vpu_inventory["host_owned_regions"],
        "vpu_add_lowered_regions": len(lowered_add),
        "vpu_elementwise_lowered_regions": len(lowered_elementwise),
        "vpu_unary_lowered_regions": len(lowered_unary),
        "vpu_reduce_lowered_regions": len(lowered_reduce),
        "vpu_residual_eligible": residual_eligible,
        "unique_kernels": catalog["unique_kernels"],
        "cache_checkpoint_weights": args.cache_checkpoint_weights,
        "cached_checkpoint_kernels": catalog.get("cached_checkpoint_kernels", 0),
        "checkpoint_weights_sha256": weights_sha,
        "elf": str(elf), "elf_bytes": elf.stat().st_size,
        "build_hash": result.get("build_hash"),
        "mem_bytes": result["mem_bytes"],
        "arena_mb": result["arena_bytes"] // (1024 * 1024),
        "arena_estimate": result["arena_estimate"],
        "arena_selection": "llvm_site_estimate" if args.arena_mb is None else "explicit",
        "compile_timeout_s": args.compile_timeout_s,
        "first_output_sha256_enabled": output_sha256,
        "all_output_sha256_enabled": True,
    }
    if args.spike_extlib is not None:
        extlib = args.spike_extlib.resolve(strict=True)
        if not extlib.is_file():
            raise ValueError("Spike extension must be a regular shared library")
        command = [str(riscv_root / "bin/spike"), "--isa=rv64gc",
                   f"--extlib={extlib}", "--extension=gemmini",
                   f"-m{hex(spike_model.DRAM_BASE)}:{hex(result['mem_bytes'])}",
                   str(elf)]
        log = work / "spike.log"
        with log.open("wb") as stream:
            try:
                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                           timeout=args.spike_timeout, check=False)
                exit_code = completed.returncode
            except subprocess.TimeoutExpired:
                exit_code = 124
        console = log.read_text(encoding="utf-8", errors="replace")
        receipt["spike"] = {
            "command": command, "extension_sha256": hashlib.sha256(extlib.read_bytes()).hexdigest(),
            "exit_code": exit_code, "done": exit_code == 0 and "DONE\n" in console,
            "log": str(log),
        }
        if receipt["spike"]["done"]:
            try:
                receipt["spike"]["output_digests"] = _observe_spike_evidence(
                    console, source, result["build_hash"], first_typed=output_sha256)
            except (ValueError, spike_model.SpikeModelError) as exc:
                receipt["spike"]["done"] = False
                receipt["spike"]["output_evidence_error"] = str(exc)
        if not receipt["spike"]["done"]:
            receipt["status"] = "spike_inference_incomplete"
    path = work / "mx_stage_probe.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(path), "status": receipt["status"],
                      "selected_operations": receipt["selected_operations"],
                      "elf_bytes": receipt["elf_bytes"]}, sort_keys=True))
    if args.spike_extlib is not None and not receipt["spike"]["done"]:
        raise SystemExit("MX Spike run did not complete; inspect the probe receipt and log")


if __name__ == "__main__":
    main()
