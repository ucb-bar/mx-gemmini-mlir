"""Compile a receipt-bound model2MLIR stage with Merlin and the MX catalog.

This is an experimental whole-stage build/run probe. Its result is not a
quantized PyTorch parity certificate or a complete multi-stage session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _generated_weight_projection():
    """The standalone harness owns every caller and readonly checkpoint span."""
    from merlin.llvmlower.entry_weight_projection import GeneratedDispatchABI

    return GeneratedDispatchABI(
        entry_symbol="forward", compiler_generated_dispatch_only=True,
        entry_address_unobserved=True, nontrapping_arithmetic=True,
        arithmetic_flags_unobserved=True, generated_weight_storage_readonly=True,
        weight_addresses_unobserved=True,
    )


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


def _require_softmax_coverage(selected: set[str], lowered: list[dict],
                              catalog: set[str], *,
                              enabled: frozenset[str] | None,
                              class_enabled: bool) -> None:
    actual = [row["region"] for row in lowered]
    expected_calls = (set(enabled) if enabled is not None else
                      selected if class_enabled else set())
    expected_catalog = set(enabled) if enabled is not None else selected
    if (len(actual) != len(set(actual)) or set(actual) != expected_calls or
            catalog != expected_catalog):
        raise RuntimeError("MX VPU softmax source selection, rewrite, and catalog disagree")


def _observe_spike_evidence(console: str, source: Path, build_hash: str,
                            *, first_typed: bool, require_complete: bool = False) -> dict:
    """Check completed console records against the captured result ABI."""
    from merlin.common.mlir_query import forward_signature
    from merlin.runtime.backends.spike_model import parse_console
    from merlin.runtime.output_digest import observe_output_digests

    from mx_gemmini_support.spike_timing import reject_deadlocked_timing

    reject_deadlocked_timing(console)
    _, outputs = forward_signature(source)
    parsed = parse_console(console, expected_output_elements=(
        int(np.prod(outputs[0][0])) if require_complete else None))
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
    parser.add_argument("--spike-arena-base", type=lambda value: int(value, 0),
                        help="absolute Spike arena base for models with large linked static data")
    parser.add_argument("--output-dump-cap", type=int, default=1)
    parser.add_argument("--diagnostic-host-vpu", action="store_true",
                        help="keep current MX contraction routing but execute VPU-eligible "
                             "operations on the host for a numerical control")
    parser.add_argument("--diagnostic-vpu-classes",
                        help="comma-separated VPU classes to enable in a numerical control: "
                             "softmax,add,elementwise,unary,reduce")
    parser.add_argument("--diagnostic-elementwise-ops",
                        help="comma-separated elementwise operations to enable within "
                             "a VPU class control: mul,add,sub,div")
    parser.add_argument("--diagnostic-elementwise-regions",
                        help="comma-separated exact source regions to enable within "
                             "the elementwise VPU class")
    parser.add_argument("--cache-checkpoint-weights", action="store_true",
                        help="pack source-proven checkpoint RHS values once per compiled stage")
    parser.add_argument("--prepack-checkpoint-weights", action="store_true",
                        help="embed source-proven packed checkpoint RHS values at build time")
    parser.add_argument("--prune-unused-checkpoint-weights", action="store_true",
                        help="remove unused immutable weights from the generated entry ABI "
                             "after prepacking; sessions must regenerate projection-aware callers")
    parser.add_argument("--reuse-prepacked-data", type=Path,
                        help="hardlink packed data from this catalog after checking freshly quantized bytes")
    parser.add_argument("--reuse-prepacked-object", type=Path,
                        help="share an existing immutable data object after checking every freshly "
                             "packed byte, symbol and fresh target metadata")
    parser.add_argument("--op-profile", action="store_true",
                        help="instrument Merlin's prepared top-level operations for a diagnostic Spike build")
    parser.add_argument("--precision-overrides", type=Path,
                        help="receipt-bound FP8 overrides for sensitive FP4 source regions")
    parser.add_argument("--execution-policy", type=Path,
                        help="capture-bound MX matrix precision and per-site VPU routing")
    parser.add_argument("--generic-input-view", action="store_true",
                        help="explicitly use a byte-bound native generic printer view for source inventories")
    parser.add_argument("--connect-vpu-multiplies", action="store_true",
                        help="experimentally retain private BF16 multiply intermediates; "
                             "requires --execution-policy; measure profitability separately")
    parser.add_argument("--connect-vpu-sigmoid-multiplies", action="store_true",
                        help="experimentally keep selected sigmoid and two multiplies resident; "
                             "requires connected multiplies, prepacking and a site policy")
    parser.add_argument("--connect-vpu-broadcast-multiplies", action="store_true",
                        help="experimentally connect projected multiply pairs and either consumer operand order; "
                             "requires connected multiplies; measure profitability separately")
    parser.add_argument("--prepare-vpu-broadcast-operands", action="store_true",
                        help="experimentally prepare repeated BF16 broadcast inputs once per call; "
                             "requires broadcast multiply connections")
    parser.add_argument("--vpu-broadcast-preparation-budget-bytes", type=int, default=1048576,
                        help="aggregate aligned private broadcast preparation budget (default: 1 MiB)")
    parser.add_argument("--share-activation-packing", action="store_true",
                        help="experimentally share explicit packed activation tensors; requires "
                             "prepacked weights and an execution policy")
    parser.add_argument("--matrix-vpu-transport", action="store_true",
                        help="experimentally preserve BF16 on private matrix-to-VPU add edges; "
                             "requires prepacked weights and an execution policy")
    parser.add_argument("--vpu-matrix-transport", action="store_true",
                        help="experimentally keep private connected VPU outputs as BF16 into shared packing; "
                             "requires connected multiplies, shared packing and integer groups")
    parser.add_argument("--virtual-zero-padding", action="store_true",
                        help="experimentally encode known zero activation rows without materializing them")
    parser.add_argument("--integer-quant-groups", action="store_true",
                        help="experimentally use BF16-word group preparation and inline its fixed-axis loops")
    parser.add_argument("--vpu-softmax-implementation", choices=("bf16", "f32_reduce", "f32_striped16"),
                        default="bf16", help="arithmetic for policy-selected VPU softmax: "
                        "all-BF16, or batched BF16 exponentials with FP32 normalization")
    parser.add_argument("--host-softmax-policy",
                        choices=("torch_2_10_cpu_softmax_f32_avx2", "libm_f32_ordered_finite"),
                        help="explicit finite-input FP32 host softmax: retain libm/ordered "
                             "normalization, or select Torch SLEEF/eight-lane arithmetic")
    parser.add_argument(
        "--fuse-gated-projections",
        action="store_true",
        help="consume private gate/up BF16 matrix outputs through hybrid sigmoid into FP8 packing",
    )
    parser.add_argument("--gated-activation-arithmetic", choices=("reciprocal", "divide"), default="reciprocal",
                        help="explicit gated FP32 arithmetic: reciprocal/multiply or gate/divide")
    parser.add_argument("--vpu-sigmoid-implementation", choices=("bf16", "f32_reciprocal"),
                        default="bf16", help="policy-selected sigmoid arithmetic: all BF16 or BF16 EXP with FP32 reciprocal")
    parser.add_argument("--fuse-softmax-packing", action="store_true",
                        help="stream private hybrid softmax into batch-local FP8 matrix inputs")
    parser.add_argument("--group-projections", action="store_true",
                        help="complete independent immutable projections of one packed input together")
    parser.add_argument("--fuse-rms-packing", action="store_true",
                        help="stream qualified private BF16 VPU RMS sequences into FP8 packing")
    parser.add_argument("--fuse-f32-rms-packing", action="store_true",
                        help="stream closed FP32 RMS normalization directly into canonical FP8 packing")
    parser.add_argument("--f32-rms-reduction", choices=("ordered", "striped16"), default="ordered",
                        help="FP32 RMS sum order: source sequential order or explicit 16-lane pair sums")
    parser.add_argument("--reuse-batch-rhs", action="store_true",
                        help="experimentally retain RHS packing across source-proved contiguous batch repetitions")
    parser.add_argument("--compact-batch-rhs", action="store_true",
                        help="experimentally bind one source matrix per proved batch group; requires RHS reuse")
    parser.add_argument("--host-math-guard", type=Path,
                        help="explicit reviewed guarded-f32 library contract; requires pinned endpoint suppliers")
    parser.add_argument("--native-tile-policy", choices=("legacy", "movement", "movement_k256"), default="legacy",
                        help="native partition: legacy, guarded movement with original K, or explicit FP8 K<=256 alternative")
    parser.add_argument("--activation-packing-budget-bytes", type=int, default=1048576,
                        help="maximum simultaneously live shared packed tensors (default: 1 MiB)")
    parser.add_argument("--retain-shared-activation-rhs", action="store_true",
                        help="diagnostic ablation: share packing but retain unused RHS descriptors")
    parser.add_argument("--structural-elf-admission", action="store_true",
                        help="explicitly select Merlin's digest/ISA linked-image admission "
                             "for this compiler experiment; this imposes no instruction policy")
    parser.add_argument("--compile-timeout-s", type=int,
                        default=int(os.environ.get("MERLIN_COMPILE_TIMEOUT_S", "3600")),
                        help="wall-clock limit per compiler/link command; 0 disables it")
    parser.add_argument("--spike-extlib", type=Path,
                        help="run the built ELF on Spike with this pinned Gemmini extension")
    parser.add_argument("--spike-isa", choices=("rv64gc", "rv64gc_zicntr"),
                        default="rv64gc_zicntr",
                        help="Spike ISA; zicntr supports generated cycle-counter reads")
    parser.add_argument("--spike-timeout", type=int, default=600)
    args = parser.parse_args()
    if args.host_softmax_policy is not None and args.execution_policy is None:
        parser.error("host softmax arithmetic requires --execution-policy with host softmax sites")
    if args.gated_activation_arithmetic != "reciprocal" and not args.fuse_gated_projections:
        parser.error("gated divide arithmetic requires --fuse-gated-projections")
    if args.fuse_f32_rms_packing and (
            not args.share_activation_packing or not args.integer_quant_groups or args.fuse_rms_packing):
        parser.error("FP32 RMS fusion requires shared integer-group packing without BF16 RMS fusion")
    if args.f32_rms_reduction != "ordered" and not args.fuse_f32_rms_packing:
        parser.error("--f32-rms-reduction requires --fuse-f32-rms-packing")
    if args.fuse_f32_rms_packing and args.op_profile:
        parser.error("FP32 RMS packing currently requires uninstrumented timing")
    if args.fuse_gated_projections and (
        not args.share_activation_packing
        or not args.integer_quant_groups
        or args.vpu_sigmoid_implementation != "f32_reciprocal"
    ):
        parser.error(
            "gated projection packing requires shared preparation, integer groups and hybrid sigmoid"
        )
    if args.fuse_gated_projections and args.op_profile:
        parser.error(
            "gated projection packing currently requires uninstrumented timing"
        )
    if args.vpu_sigmoid_implementation != "bf16" and (args.execution_policy is None
            or args.connect_vpu_sigmoid_multiplies):
        parser.error("hybrid sigmoid requires an explicit policy without connected BF16 sigmoid")
    if args.fuse_softmax_packing and (args.vpu_softmax_implementation not in {"f32_reduce", "f32_striped16"}
            or not args.compact_batch_rhs or not args.integer_quant_groups):
        parser.error("softmax packing requires hybrid arithmetic, compact RHS and integer groups")
    if args.fuse_softmax_packing and args.op_profile:
        parser.error("streamed softmax packing currently requires uninstrumented timing")
    if args.group_projections and (not args.share_activation_packing or args.retain_shared_activation_rhs):
        parser.error("--group-projections requires shared packing without retained RHS descriptors")
    if args.group_projections and args.op_profile:
        parser.error("grouped projections currently require uninstrumented timing")
    if args.fuse_rms_packing and not args.vpu_matrix_transport:
        parser.error("--fuse-rms-packing requires --vpu-matrix-transport")
    if args.compact_batch_rhs and not args.reuse_batch_rhs:
        parser.error("--compact-batch-rhs requires --reuse-batch-rhs")
    if args.reuse_batch_rhs and (not args.integer_quant_groups or args.execution_policy is None):
        parser.error("--reuse-batch-rhs requires integer groups and an explicit execution policy")
    if args.connect_vpu_multiplies and args.execution_policy is None:
        parser.error("--connect-vpu-multiplies requires an explicit --execution-policy")
    if args.connect_vpu_broadcast_multiplies and not args.connect_vpu_multiplies:
        parser.error("--connect-vpu-broadcast-multiplies requires --connect-vpu-multiplies")
    if args.prepare_vpu_broadcast_operands and not args.connect_vpu_broadcast_multiplies:
        parser.error("--prepare-vpu-broadcast-operands requires --connect-vpu-broadcast-multiplies")
    if args.vpu_broadcast_preparation_budget_bytes < 0:
        parser.error("broadcast preparation budget cannot be negative")
    if args.connect_vpu_sigmoid_multiplies and (not args.connect_vpu_multiplies
            or not args.prepack_checkpoint_weights or args.execution_policy is None):
        parser.error("--connect-vpu-sigmoid-multiplies requires connected multiplies, prepacking and a policy")
    if args.share_activation_packing and (not args.prepack_checkpoint_weights
                                         or args.execution_policy is None):
        parser.error("--share-activation-packing requires prepacked weights and an execution policy")
    if args.activation_packing_budget_bytes <= 0:
        parser.error("--activation-packing-budget-bytes must be positive")
    if args.matrix_vpu_transport and (not args.prepack_checkpoint_weights or args.execution_policy is None):
        parser.error("--matrix-vpu-transport requires prepacked weights and an execution policy")
    if args.vpu_matrix_transport and not (args.connect_vpu_multiplies and args.share_activation_packing
                                         and args.integer_quant_groups):
        parser.error("--vpu-matrix-transport requires connected multiplies, shared packing and integer groups")
    if args.retain_shared_activation_rhs and not args.share_activation_packing:
        parser.error("--retain-shared-activation-rhs requires --share-activation-packing")
    if args.prune_unused_checkpoint_weights and not args.prepack_checkpoint_weights:
        parser.error("--prune-unused-checkpoint-weights requires --prepack-checkpoint-weights")
    if args.reuse_prepacked_data is not None and not args.prepack_checkpoint_weights:
        parser.error("--reuse-prepacked-data requires --prepack-checkpoint-weights")
    if args.reuse_prepacked_object is not None and not args.prepack_checkpoint_weights:
        parser.error("--reuse-prepacked-object requires --prepack-checkpoint-weights")
    if args.execution_policy is not None and (
            args.precision_overrides is not None or args.diagnostic_host_vpu or
            args.diagnostic_vpu_classes is not None or
            args.diagnostic_elementwise_ops is not None or
            args.diagnostic_elementwise_regions is not None):
        parser.error("--execution-policy cannot be combined with diagnostic routing flags")
    all_vpu_classes = frozenset(("softmax", "add", "elementwise", "unary", "reduce"))
    if args.diagnostic_host_vpu and args.diagnostic_vpu_classes is not None:
        parser.error("choose one VPU diagnostic mode")
    if args.diagnostic_host_vpu:
        enabled_vpu_classes = frozenset()
    elif args.diagnostic_vpu_classes is not None:
        enabled_vpu_classes = frozenset(args.diagnostic_vpu_classes.split(","))
        if not enabled_vpu_classes or not enabled_vpu_classes <= all_vpu_classes:
            parser.error("diagnostic VPU classes must be a nonempty subset of "
                         "softmax,add,elementwise,unary,reduce")
    else:
        enabled_vpu_classes = all_vpu_classes
    enabled_elementwise_operations = None
    if args.diagnostic_elementwise_ops is not None:
        valid_elementwise = frozenset(("mul", "add", "sub", "div"))
        enabled_elementwise_operations = frozenset(
            args.diagnostic_elementwise_ops.split(","))
        if (args.diagnostic_vpu_classes is None or
                "elementwise" not in enabled_vpu_classes or
                not enabled_elementwise_operations or
                not enabled_elementwise_operations <= valid_elementwise):
            parser.error("diagnostic elementwise operations require the "
                         "elementwise VPU class and a nonempty subset of "
                         "mul,add,sub,div")
    enabled_elementwise_regions = None
    if args.diagnostic_elementwise_regions is not None:
        enabled_elementwise_regions = frozenset(
            args.diagnostic_elementwise_regions.split(","))
        if (args.diagnostic_vpu_classes is None or
                "elementwise" not in enabled_vpu_classes or
                not enabled_elementwise_regions or
                "" in enabled_elementwise_regions):
            parser.error("diagnostic elementwise regions require the "
                         "elementwise VPU class and nonempty source regions")
    diagnostic_vpu = (enabled_vpu_classes != all_vpu_classes or
                      enabled_elementwise_operations is not None or
                      enabled_elementwise_regions is not None)
    if args.cache_checkpoint_weights and args.prepack_checkpoint_weights:
        parser.error("choose one checkpoint weight preparation mode")
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
    from mx_gemmini_support.target_profile import load_profile, profile_sha256

    profile = load_profile(profile_path, rtl_root=args.rtl_root)
    profile_sha = profile_sha256(profile)
    input_read_view = None
    input_read_view_receipt_sha = None
    inventory_source = source
    if args.generic_input_view:
        from merlin.llvmlower.generic_form import (
            create_generic_read_view,
            recheck_generic_read_view,
        )

        input_read_view = create_generic_read_view(source, work / "input_read_view")
        inventory_source = recheck_generic_read_view(input_read_view)
        input_read_view_receipt_sha = _sha(work / "input_read_view/generic_read_view.json")
    source_text = inventory_source.read_text(encoding="utf-8")
    policy_path = None
    execution_policy = None
    enabled_site_regions = None
    host_singleton_regions = frozenset()
    if args.execution_policy is not None:
        from mx_gemmini_support.execution_policy import load_policy

        if args.execution_policy.is_symlink():
            parser.error("--execution-policy must be a regular file")
        policy_path = args.execution_policy.resolve(strict=True)
        execution_policy, resolved_policy = load_policy(
            policy_path, source_text, profile)
        enabled_site_regions = resolved_policy["enabled_vpu_regions"]
        host_singleton_regions = resolved_policy["host_singleton_regions"]
    if args.precision_overrides is not None and args.precision_overrides.is_symlink():
        parser.error("--precision-overrides must be a regular file")
    override_path = (args.precision_overrides.resolve(strict=True)
                     if args.precision_overrides is not None else None)
    overrides = (_load_precision_overrides(override_path, capture_sha, profile_sha)
                 if override_path is not None else None)
    if execution_policy is not None:
        overrides = resolved_policy["precision_overrides"]
    override_sha = _sha(override_path) if override_path is not None else None
    policy_sha = _sha(policy_path) if policy_path is not None else None
    os.environ["MERLIN_TARGET_PATH"] = str(ROOT)
    os.environ.setdefault("MERLIN_COMPILER_PYTHON", sys.executable)
    os.environ["MERLIN_MLIR_INSTALL"] = str(llvm_root)
    os.environ["MERLIN_CLANG"] = str(llvm_root / "bin/clang")
    os.environ["MERLIN_RISCV_GCC"] = str(riscv_root / "bin/riscv64-unknown-elf-gcc")
    os.environ["MERLIN_SPIKE"] = str(riscv_root / "bin/spike")
    os.environ["MERLIN_COMPILE_TIMEOUT_S"] = str(args.compile_timeout_s)
    sys.path.insert(0, str(merlin_src))
    from merlin.llvmlower.device_build import DeviceRouting
    from merlin.model_check.admission import service as structural_admission
    from merlin.runtime.backends import spike_model

    from mx_gemmini_support.merlin_bf16_prepare import MXBF16CatalogPreparation
    from mx_gemmini_support.merlin_catalog_builder import make_merlin_catalog_builder
    from mx_gemmini_support.merlin_vpu_add import MXVPUPostOffload, selected_add_sites
    from mx_gemmini_support.merlin_vpu_elementwise import selected_elementwise_sites
    from mx_gemmini_support.merlin_vpu_inventory import audit_softmax_boundaries
    from mx_gemmini_support.merlin_vpu_reduce import selected_row_reduce_sites
    from mx_gemmini_support.merlin_vpu_softmax import selected_softmax_sites
    from mx_gemmini_support.merlin_vpu_unary import selected_unary_sites

    vpu_inventory = audit_softmax_boundaries(source_text, profile)
    vpu_selected = selected_softmax_sites(source_text, profile)
    if args.host_softmax_policy is not None and not (
            {row["region"] for row in vpu_selected} - enabled_site_regions["softmax"]):
        parser.error("host softmax arithmetic policy has no host softmax sites")
    if vpu_inventory["direct_profile_matches_unlowered"]:
        raise RuntimeError("captured direct BF16 rank-two VPU softmax has no integrated lowering")
    expected_regions = _raw_selected_regions(
        inventory_source, profile, precision_overrides=overrides)
    if not host_singleton_regions <= expected_regions:
        raise RuntimeError("host singleton policy differs from raw matrix coverage")
    expected_regions -= host_singleton_regions
    fixed_weights = args.cache_checkpoint_weights or args.prepack_checkpoint_weights
    if fixed_weights and not expected_regions:
        parser.error("checkpoint weight preparation requires selected MX contractions")
    weights_path = stage / "weights.safetensors"
    if fixed_weights and (weights_path.is_symlink() or not weights_path.is_file()):
        parser.error("checkpoint weight preparation requires a regular captured weights file")
    weights_sha = _sha(weights_path) if fixed_weights else None
    if fixed_weights:
        capture_artifacts = json.loads((stage / "capture_receipt.json").read_text(
            encoding="utf-8"))["artifacts"]
        for name in ("weights.safetensors", "weights.safetensors.manifest.json"):
            artifact = stage / name
            record = capture_artifacts.get(name) or {}
            if (artifact.is_symlink() or not artifact.is_file()
                    or record.get("bytes") != artifact.stat().st_size
                    or record.get("sha256") != (_sha(artifact) if name.endswith(".json")
                                                else weights_sha)):
                parser.error(f"checkpoint artifact differs from capture receipt: {name}")
    if (diagnostic_vpu or execution_policy is not None) and not expected_regions and (
            not host_singleton_regions or any(enabled_site_regions.values())):
        parser.error("VPU diagnostics require selected MX contractions")
    preparation = MXBF16CatalogPreparation(profile, precision_overrides=overrides)
    host_preparation = None
    if host_singleton_regions:
        from mx_gemmini_support.merlin_singleton_prepare import (
            MXSingletonReductionPreparation,
        )

        if expected_regions:
            preparation = MXSingletonReductionPreparation(
                host_singleton_regions, downstream=preparation)
        else:
            host_preparation = MXSingletonReductionPreparation(host_singleton_regions)
    device = (DeviceRouting(
        device="mx_gemmini", package_dir=ROOT,
        operand_dtype="catalog_mixed_direct_fp4_fp8", accum_dtype="bf16",
        linked_elf_admission=(structural_admission("mx_gemmini")
                              if args.structural_elf_admission else None),
        prepared_transform=preparation,
        catalog_builder=make_merlin_catalog_builder(
            profile_path=profile_path, riscv_root=riscv_root,
            cached_weight_stage=stage if args.cache_checkpoint_weights else None,
            prepacked_weight_stage=stage if args.prepack_checkpoint_weights else None,
            precision_overrides=overrides,
            enabled_vpu_regions=enabled_site_regions,
            connect_multiplies=args.connect_vpu_multiplies,
            connect_broadcast_multiplies=args.connect_vpu_broadcast_multiplies,
            prepare_broadcast_operands=args.prepare_vpu_broadcast_operands,
            broadcast_preparation_budget_bytes=args.vpu_broadcast_preparation_budget_bytes,
            connect_sigmoid_multiplies=args.connect_vpu_sigmoid_multiplies,
            share_activation_packing=args.share_activation_packing,
            retain_shared_activation_rhs=args.retain_shared_activation_rhs,
            matrix_vpu_transport=args.matrix_vpu_transport,
            vpu_matrix_transport=args.vpu_matrix_transport,
            virtual_zero_padding=args.virtual_zero_padding,
            integer_group_packing=args.integer_quant_groups,
            reuse_batch_rhs=args.reuse_batch_rhs,
            compact_batch_rhs=args.compact_batch_rhs,
            vpu_softmax_implementation=args.vpu_softmax_implementation,
            vpu_sigmoid_implementation=args.vpu_sigmoid_implementation,
            f32_rms_reduction=args.f32_rms_reduction,
            fuse_rms_packing=args.fuse_rms_packing,
            fuse_f32_rms_packing=args.fuse_f32_rms_packing,
            group_projections=args.group_projections,
            fuse_softmax_packing=args.fuse_softmax_packing,
            fuse_gated_projections=args.fuse_gated_projections,
            gated_activation_arithmetic=args.gated_activation_arithmetic,
            native_tile_policy=args.native_tile_policy,
            reuse_prepacked_data=args.reuse_prepacked_data,
            reuse_prepacked_object=args.reuse_prepacked_object),
        post_offload_transform=MXVPUPostOffload(
            profile,
            enabled_classes=enabled_vpu_classes,
            enabled_elementwise_operations=enabled_elementwise_operations,
            enabled_elementwise_regions=enabled_elementwise_regions,
            enabled_site_regions=enabled_site_regions,
            connect_multiplies=args.connect_vpu_multiplies,
            connect_broadcast_multiplies=args.connect_vpu_broadcast_multiplies,
            connect_sigmoid_multiplies=args.connect_vpu_sigmoid_multiplies,
            share_activation_packing=args.share_activation_packing,
            activation_packing_budget_bytes=args.activation_packing_budget_bytes,
            matrix_vpu_transport=args.matrix_vpu_transport,
            vpu_matrix_transport=args.vpu_matrix_transport,
            fuse_rms_packing=args.fuse_rms_packing,
            fuse_f32_rms_packing=args.fuse_f32_rms_packing,
            group_projections=args.group_projections,
            fuse_softmax_packing=args.fuse_softmax_packing,
            fuse_gated_projections=args.fuse_gated_projections,
            host_softmax_policy=args.host_softmax_policy,
            batch_rhs_catalog=(work / "device_catalog/mx_catalog.json" if args.reuse_batch_rhs else None),
            fixed_weight_catalog=(work / "device_catalog/mx_catalog.json"
                                  if args.prepack_checkpoint_weights else None)),
    ) if expected_regions else None)
    guarded_math = None
    if args.host_math_guard is not None:
        from merlin.runtime.host_math import GuardedF32Call

        guarded_math = GuardedF32Call.from_dict(json.loads(args.host_math_guard.read_text()))
    result = spike_model.build(
        stage, work, arena_mb=args.arena_mb if args.arena_mb is not None else 2048,
        auto_arena=args.arena_mb is None, backend="scalar", device=device,
        spike_arena_base=(args.spike_arena_base if args.spike_arena_base is not None
                          else spike_model.ARENA_BASE),
        cflags_override=["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2"],
        output_dump_cap=args.output_dump_cap, output_sha256=output_sha256,
        output_all_sha256=True, op_profile=args.op_profile,
        prepared_model_transform=host_preparation,
        host_math_policy="guarded_f32" if guarded_math is not None else "native",
        guarded_math=guarded_math,
        entry_weight_projection=(
            _generated_weight_projection() if args.prune_unused_checkpoint_weights else None),
    )
    profile_table_path = work / "op_profile_table.json"
    profile_op_count = None
    if args.op_profile:
        if not profile_table_path.is_file() or profile_table_path.is_symlink():
            raise RuntimeError("Merlin did not emit the requested operation profile table")
        profile_op_count = len(json.loads(profile_table_path.read_text(encoding="utf-8")))
        if profile_op_count > 1_000_000:
            raise RuntimeError("operation profile exceeds the supported counter table")
    if hashlib.sha256(source.read_bytes()).hexdigest() != capture_sha:
        raise RuntimeError("Merlin changed the receipt-bound model2MLIR capture source")
    if weights_sha is not None and _sha(weights_path) != weights_sha:
        raise RuntimeError("captured checkpoint changed during MX stage compilation")
    if override_sha is not None and _sha(override_path) != override_sha:
        raise RuntimeError("MX precision overrides changed during stage compilation")
    if policy_sha is not None and _sha(policy_path) != policy_sha:
        raise RuntimeError("MX execution policy changed during stage compilation")
    elf = Path(result["elf"])
    singleton_receipt = None
    singleton_path = work / ("device_prepared" if device is not None else
                             "prepared_model_transform") / "singleton_reductions.json"
    if host_singleton_regions:
        singleton_receipt = json.loads(singleton_path.read_text(encoding="utf-8"))
        host_records = singleton_receipt["regions"]
        if (len(host_records) != len(host_singleton_regions) or
                {row["region"] for row in host_records} != host_singleton_regions):
            raise RuntimeError("prepared host singleton coverage differs from selected policy")
    catalog_path = work / "device_catalog/mx_catalog.json"
    if device is None:
        if catalog_path.exists():
            raise RuntimeError("host-only stage unexpectedly emitted an MX catalog")
        catalog = {"covered_contractions": 0, "unbound_precision_proposals": [],
                   "unique_kernels": 0, "bindings": []}
        if enabled_site_regions is not None:
            catalog["resolved_vpu_regions"] = {name: [] for name in enabled_site_regions}
            catalog.update({f"vpu_{name}": {"source_regions": []}
                            for name in enabled_site_regions})
    else:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        _require_source_coverage(expected_regions, catalog)
        if catalog.get("precision_overrides", {}) != (overrides or {}):
            raise RuntimeError("MX catalog lost the selected precision overrides")
        for binding in catalog["bindings"]:
            if (binding["region"] in (overrides or {}) and
                    binding["target_dtypes"][:2] != ["f8E4M3FN", "f8E4M3FN"]):
                raise RuntimeError("MX catalog override did not bind FP8 operands")
            if execution_policy is not None:
                expected_dtype = ("f8E4M3FN" if
                                  execution_policy["matrix_formats"][binding["region"]]
                                  == "mxfp8" else "f4E2M1FN")
                if binding["target_dtypes"][:2] != [expected_dtype, expected_dtype]:
                    raise RuntimeError("MX linked catalog differs from execution policy")
    gated_projection_rewrite = None
    if args.fuse_gated_projections:
        gated_path = work / "device_host_abi/mx_gated_projections_rewrite.json"
        gated_projection_rewrite = json.loads(gated_path.read_text())
        from mx_gemmini_support.merlin_gated_projection import (
            prepared_matrix_consumers,
            source_matrix_callees,
        )

        endpoints = {row["symbol"]: row for row in catalog["gated_projection_packing"]}
        source_callees = source_matrix_callees(
            catalog, json.loads((work / "device_signatures.json").read_text())
        )

        def valid_gated_terminal(row):
            endpoint = endpoints[row["symbol"]]
            if row["original_prepare"] is not None:
                return (
                    row["original_prepare"] in endpoint["prepare_symbols"]
                    and row["original_matrix_consumer"] is None
                    and row["matrix_consumer_kernel"] is None
                )
            matrix = prepared_matrix_consumers(catalog, endpoint, source_callees).get(
                row["original_matrix_consumer"]
            )
            return (
                matrix is not None and matrix["symbol"] == row["matrix_consumer_kernel"]
            )

        sigmoid = {
            row["symbol"]
            for row in catalog["vpu_unary"]["kernels"]
            if row["operation"] == "sigmoid"
            and row.get("implementation") == "f32_reciprocal"
        }
        if (
            gated_projection_rewrite["catalog_sha256"] != _sha(catalog_path)
            or gated_projection_rewrite["sidecar_sha256"]
            != _sha(work / "device_signatures.json")
            or gated_projection_rewrite["output_sha256"]
            != _sha(gated_path.with_name("mx_gated_projections.mlir"))
            or any(
                row["symbol"] not in endpoints
                or any(
                    row[key] != endpoints[row["symbol"]][key]
                    for key in ("source_kernels", "source_regions", "shape_mnk")
                )
                or row["original_sigmoid"] not in sigmoid
                or not valid_gated_terminal(row)
                for row in gated_projection_rewrite["fusions"]
            )
        ):
            raise RuntimeError(
                "gated projection packing differs from its compiled endpoints"
            )
    softmax_packing_rewrite = None
    if args.fuse_softmax_packing:
        packing_path = work / "device_host_abi/mx_softmax_packing_rewrite.json"
        softmax_packing_rewrite = json.loads(packing_path.read_text())
        endpoints = {row["symbol"]: row for row in catalog["softmax_packing"]}
        if (softmax_packing_rewrite["catalog_sha256"] != _sha(catalog_path)
                or softmax_packing_rewrite["sidecar_sha256"] != _sha(work / "device_signatures.json")
                or softmax_packing_rewrite["output_sha256"] != _sha(packing_path.with_name("mx_softmax_packing.mlir"))
                or any(row["symbol"] not in endpoints
                       or any(row[key] != endpoints[row["symbol"]][key]
                              for key in ("original_softmax", "source_kernel", "packed_bytes"))
                       or row["original_consumer"] not in endpoints[row["symbol"]]["original_consumers"]
                       for row in softmax_packing_rewrite["fusions"])):
            raise RuntimeError("softmax packing differs from its compiled endpoints")
    projection_rewrite = None
    if args.group_projections:
        group_path = work / "device_host_abi/mx_projection_groups_rewrite.json"
        projection_rewrite = json.loads(group_path.read_text())
        endpoints = {row["symbol"]: row for row in catalog["projection_groups"]}
        if (projection_rewrite["catalog_sha256"] != _sha(catalog_path)
                or projection_rewrite["sidecar_sha256"] != _sha(work / "device_signatures.json")
                or projection_rewrite["output_sha256"] != _sha(group_path.with_name("mx_projection_groups.mlir"))
                or any(row["symbol"] not in endpoints
                       or any(row[key] != endpoints[row["symbol"]][key]
                              for key in ("consumer_symbols", "kernel_symbols", "source_regions"))
                       for row in projection_rewrite["groups"])):
            raise RuntimeError("projection grouping differs from its compiled endpoints")
    f32_rms_rewrite = None
    if args.fuse_f32_rms_packing:
        from mx_gemmini_support.merlin_gated_projection import (
            prepared_consumers_for_packed_input,
            source_matrix_callees,
        )

        f32_path = work / "device_host_abi/mx_f32_rms_packing_rewrite.json"
        f32_rms_rewrite = json.loads(f32_path.read_text())
        endpoints = {row["symbol"]: row for row in catalog["f32_rms_packing"]}
        source_callees = source_matrix_callees(
            catalog, json.loads((work / "device_signatures.json").read_text()))

        def valid_f32_rms(row):
            endpoint = endpoints.get(row["symbol"])
            if endpoint is None or any(row[key] != endpoint[key] for key in ("shape", "reduction")):
                return False
            if row["original_prepare"] is not None:
                return (row["original_prepare"] in endpoint["prepare_symbols"]
                        and row["original_matrix_consumer"] is None
                        and row["matrix_consumer_kernel"] is None)
            consumers = prepared_consumers_for_packed_input(
                catalog, rows=endpoint["rows"], width=endpoint["width"],
                packed_bytes=endpoint["packed_bytes"], prepare_symbols=endpoint["prepare_symbols"],
                source_callees=source_callees)
            matrix = consumers.get(row["original_matrix_consumer"])
            return matrix is not None and matrix["symbol"] == row["matrix_consumer_kernel"]

        if (f32_rms_rewrite["catalog_sha256"] != _sha(catalog_path)
                or f32_rms_rewrite["sidecar_sha256"] != _sha(work / "device_signatures.json")
                or f32_rms_rewrite["output_sha256"] != _sha(f32_path.with_name("mx_f32_rms_packing.mlir"))
                or any(not valid_f32_rms(row) for row in f32_rms_rewrite["fusions"])):
            raise RuntimeError("FP32 RMS packing differs from its compiled endpoints")
    rms_rewrite = None
    if args.fuse_rms_packing:
        rms_path = work / "device_host_abi/mx_rms_packing_rewrite.json"
        rms_rewrite = json.loads(rms_path.read_text())
        endpoints = {row["symbol"]: row for row in catalog["rms_packing"]}
        if (rms_rewrite["catalog_sha256"] != _sha(catalog_path)
                or rms_rewrite["sidecar_sha256"] != _sha(work / "device_signatures.json")
                or rms_rewrite["output_sha256"] != _sha(rms_path.with_name("mx_rms_packing.mlir"))
                or any(row["symbol"] not in endpoints or row["shape"] != endpoints[row["symbol"]]["shape"]
                       for row in rms_rewrite["fusions"])):
            raise RuntimeError("RMS packing rewrite differs from its compiled endpoints")
    rewrite_path = work / "device_host_abi/mx_vpu_softmax_rewrite.json"
    lowered_vpu = (json.loads(rewrite_path.read_text(encoding="utf-8"))["regions"]
                   if rewrite_path.is_file() else [])
    selected_vpu_regions = {row["region"] for row in vpu_selected}
    lowered_vpu_regions = {row["region"] for row in lowered_vpu}
    catalog_vpu_regions = set(catalog.get("vpu_softmax", {}).get("source_regions", ()))
    resolved_vpu_regions = (catalog.get("resolved_vpu_regions")
                            if enabled_site_regions is not None else None)
    if enabled_site_regions is not None and (
            not isinstance(resolved_vpu_regions, dict) or
            set(resolved_vpu_regions) != set(enabled_site_regions)):
        raise RuntimeError("MX catalog lost the resolved VPU execution policy")
    _require_softmax_coverage(
        selected_vpu_regions, lowered_vpu, catalog_vpu_regions,
        enabled=(frozenset(resolved_vpu_regions["softmax"])
                 if resolved_vpu_regions is not None else None),
        class_enabled="softmax" in enabled_vpu_classes)
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
    sigmoid_path = work / "device_host_abi/mx_connected_sigmoid_rewrite.json"
    sigmoid_chains = []
    if args.connect_vpu_sigmoid_multiplies and device is not None:
        fused = json.loads(sigmoid_path.read_text())
        sigmoid_chains = fused["chains"]
        keys = ("regions", "unary_region", "shape", "symbol", "kernel_symbol")
        expected = [{key: row[key] for key in keys} for row in
                    catalog["vpu_elementwise"]["connected_sigmoid_multiplies"]]
        if (sigmoid_chains != expected or fused["catalog_sha256"] != _sha(catalog_path)
                or fused["sidecar_sha256"] != _sha(work / "device_signatures.json")
                or fused["output_sha256"] != _sha(sigmoid_path.with_name("mx_connected_sigmoid.mlir"))):
            raise RuntimeError("MX sigmoid chain source obligations differ from compiled routing")
    elif sigmoid_path.exists():
        raise RuntimeError("unselected sigmoid chain rewrite cannot alter source coverage")
    lowered_elementwise = (json.loads(elementwise_rewrite_path.read_text(encoding="utf-8"))["regions"]
                           if elementwise_rewrite_path.is_file() else [])
    lowered_elementwise.extend({"region": region, "symbol": chain["symbol"], "operation": "mul",
                                "fused_into": "sigmoid_multiply_chain"}
                               for chain in sigmoid_chains for region in chain["regions"])
    catalog_elementwise_regions = set(catalog.get("vpu_elementwise", {}).get("source_regions", ()))
    if (len({row["region"] for row in lowered_elementwise}) != len(lowered_elementwise) or
            not {row["region"] for row in lowered_elementwise} <= catalog_elementwise_regions):
        raise RuntimeError("MX VPU elementwise rewrite exceeds its source-bound catalog")
    connected_multiplies = (json.loads(elementwise_rewrite_path.read_text()).get("connected_multiplies", [])
                            if elementwise_rewrite_path.is_file() else [])
    planned_multiplies = catalog.get("vpu_elementwise", {}).get("connected_multiplies", [])
    if connected_multiplies != [
            {key: row[key] for key in ("regions", "shape", "symbol")} for row in planned_multiplies]:
        raise RuntimeError("MX connected multiply rewrite differs from its source-bound catalog")
    unary_rewrite_path = work / "device_host_abi/mx_vpu_unary_rewrite.json"
    lowered_unary = (json.loads(unary_rewrite_path.read_text(encoding="utf-8"))["regions"]
                     if unary_rewrite_path.is_file() else [])
    lowered_unary.extend({"region": chain["unary_region"], "symbol": chain["symbol"],
                          "operation": "sigmoid", "fused_into": "sigmoid_multiply_chain"}
                         for chain in sigmoid_chains)
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
    if enabled_site_regions is not None:
        lowered_by_class = {
            "softmax": lowered_vpu, "add": lowered_add,
            "elementwise": lowered_elementwise, "unary": lowered_unary,
            "reduce": lowered_reduce,
        }
        for name, rows in lowered_by_class.items():
            expected = set(resolved_vpu_regions[name])
            if {row["region"] for row in rows} != expected:
                raise RuntimeError(f"MX VPU {name} did not obey the execution policy")
            if set(catalog[f"vpu_{name}"]["source_regions"]) != expected:
                raise RuntimeError(f"MX VPU {name} catalog differs from execution policy")
    host_softmax_path = work / "device_host_abi/mx_host_softmax_rewrite.json"
    host_softmax_rewrite = (json.loads(host_softmax_path.read_text())
                            if host_softmax_path.is_file() else None)
    if args.host_softmax_policy is not None:
        expected_host_softmax = {row["region"]: list(row["shape"]) for row in vpu_selected
                                 if row["region"] not in resolved_vpu_regions["softmax"]}
        if (host_softmax_rewrite is None
                or host_softmax_rewrite["numeric_policy"] != args.host_softmax_policy
                or not host_softmax_rewrite["finite_input_contract"]
                or not host_softmax_rewrite["finite_maximum_specialization"]
                or {row["region"]: row["shape"] for row in host_softmax_rewrite["regions"]}
                != expected_host_softmax
                or len(host_softmax_rewrite["regions"]) != len(expected_host_softmax)):
            raise RuntimeError("host softmax arithmetic rewrite differs from the execution policy")
    elif host_softmax_rewrite is not None:
        raise RuntimeError("host softmax arithmetic rewrite was not requested")
    residual_eligible = {}
    post_receipt = work / "device_host_abi/post_offload_transform.json"
    if post_receipt.is_file():
        selected_path = Path(json.loads(post_receipt.read_text())["selected_path"])
        selected_text = selected_path.read_text()
        residual_eligible = {
            "softmax": len(selected_softmax_sites(selected_text, profile)),
            "add": len(selected_add_sites(selected_text, profile)),
            "elementwise": sum(
                (enabled_elementwise_operations is None or
                 site["operation"] in enabled_elementwise_operations) and
                (enabled_elementwise_regions is None or
                 site["region"] in enabled_elementwise_regions)
                for site in selected_elementwise_sites(selected_text, profile)),
            "unary": len(selected_unary_sites(selected_text, profile)),
            "reduce": len(selected_row_reduce_sites(selected_text, profile)),
        }
        if enabled_site_regions is None and any(
                count for name, count in residual_eligible.items()
                if name in enabled_vpu_classes):
            raise RuntimeError(f"MX VPU eligible source sites remained on host: {residual_eligible}")
    bf16_receipt = work / "device_prepared/mx_bf16_catalog.json"
    widened_bf16 = (json.loads(bf16_receipt.read_text(encoding="utf-8"))["widened_operations"]
                    if bf16_receipt.is_file() else 0)
    status = ("linked_policy_numeric_qualification_pending"
              if execution_policy is not None else
              "linked_diagnostic_host_vpu_numeric_qualification_pending"
              if args.diagnostic_host_vpu else
              "linked_diagnostic_partial_vpu_numeric_qualification_pending"
              if diagnostic_vpu else
              "linked_experimental_numeric_qualification_pending")
    route = ("mx_gemmini_per_site_policy" if execution_policy is not None else
             "mx_gemmini_and_host_vpu_control" if args.diagnostic_host_vpu else
             "mx_gemmini_and_partial_vpu_control" if diagnostic_vpu else
             "mx_gemmini_and_host" if device is not None else "host_only")
    if input_read_view is not None:
        recheck_generic_read_view(input_read_view)
        view_receipt = work / "input_read_view/generic_read_view.json"
        if (_sha(view_receipt) != input_read_view_receipt_sha
                or json.loads(view_receipt.read_text()) != input_read_view):
            raise RuntimeError("selected input printer receipt changed during compilation or execution")
    receipt = {
        "schema": "mx_gemmini.merlin_stage_probe.v1",
        "status": status,
        "capture_model_sha256": capture_sha,
        "generic_input_read_view": input_read_view,
        "generic_input_read_view_receipt_sha256": input_read_view_receipt_sha,
        "inventory_source_sha256": _sha(inventory_source),
        "profile_sha256": profile_sha256(profile),
        "precision_overrides_sha256": override_sha,
        "execution_policy_sha256": policy_sha,
        "execution_policy": execution_policy,
        "connect_vpu_multiplies": args.connect_vpu_multiplies,
        "connect_vpu_broadcast_multiplies": args.connect_vpu_broadcast_multiplies,
        "prepare_vpu_broadcast_operands": args.prepare_vpu_broadcast_operands,
        "vpu_broadcast_preparation_budget_bytes": args.vpu_broadcast_preparation_budget_bytes,
        "connect_vpu_sigmoid_multiplies": args.connect_vpu_sigmoid_multiplies,
        "elf_admission_selection": ("structural_digest_and_declared_isa"
                                    if args.structural_elf_admission and device is not None else None),
        "connected_vpu_multiply_pairs": connected_multiplies,
        "connected_vpu_sigmoid_multiply_chains": sigmoid_chains,
        "precision_overrides": overrides or {},
        "catalog_manifest": str(catalog_path) if device is not None else None,
        "execution_route": route,
        "diagnostic_host_vpu": args.diagnostic_host_vpu,
        "enabled_vpu_classes": sorted(enabled_vpu_classes),
        "enabled_vpu_elementwise_operations": (
            sorted(enabled_elementwise_operations)
            if enabled_elementwise_operations is not None else None),
        "enabled_vpu_elementwise_regions": (
            sorted(enabled_elementwise_regions)
            if enabled_elementwise_regions is not None else None),
        "selected_operations": catalog["covered_contractions"],
        "expected_selected_operations": len(expected_regions),
        "host_singleton_matrix_regions": sorted(host_singleton_regions),
        "host_singleton_preparation": singleton_receipt,
        "host_singleton_receipt_sha256": _sha(singleton_path) if singleton_receipt else None,
        "raw_matrix_operations_accounted": len(expected_regions) + len(host_singleton_regions),
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
        "prepack_checkpoint_weights": args.prepack_checkpoint_weights,
        "packed_data_storage_reuse": catalog.get("packed_data_storage_reuse"),
        "packed_object_storage_reuse": catalog.get("packed_object_storage_reuse"),
        "share_activation_packing": args.share_activation_packing,
        **({"reuse_batch_rhs": True} if args.reuse_batch_rhs else {}),
        **({"compact_batch_rhs": True} if args.compact_batch_rhs else {}),
        "retain_shared_activation_rhs": args.retain_shared_activation_rhs,
        "matrix_vpu_transport": args.matrix_vpu_transport,
        "vpu_matrix_transport": args.vpu_matrix_transport,
        "private_vpu_matrix_transport_rewrite": (
            json.loads((work / "device_host_abi/mx_vpu_matrix_transport_rewrite.json").read_text())
            if args.vpu_matrix_transport and device is not None else None),
        "private_matrix_transport_rewrite": (
            json.loads((work / "device_host_abi/mx_matrix_transport_rewrite.json").read_text())
            if args.matrix_vpu_transport and device is not None else None),
        "shared_activation_rewrite": (
            json.loads((work / "device_host_abi/mx_shared_activation_rewrite.json").read_text())
            if args.share_activation_packing and device is not None else None),
        "prune_unused_checkpoint_weights": args.prune_unused_checkpoint_weights,
        "entry_weight_projection_receipt_sha256": (
            _sha(work / "entry_weight_projection/entry_weight_projection.json")
            if args.prune_unused_checkpoint_weights else None),
        "prepacked_checkpoint_kernels": catalog.get("prepacked_checkpoint_kernels", 0),
        "op_profile": args.op_profile,
        "op_profile_table": str(profile_table_path) if args.op_profile else None,
        "op_profile_table_sha256": _sha(profile_table_path) if args.op_profile else None,
        "op_profile_operations": profile_op_count,
        "op_profile_capacity": (max(8192, 1 << (profile_op_count - 1).bit_length())
                                if profile_op_count is not None else None),
        "checkpoint_weights_sha256": weights_sha,
        "elf": str(elf), "elf_bytes": elf.stat().st_size,
        "build_hash": result.get("build_hash"),
        "mem_bytes": result["mem_bytes"],
        "arena_mb": result["arena_bytes"] // (1024 * 1024),
        "spike_arena_base": result["arena_base"],
        "arena_estimate": result["arena_estimate"],
        "arena_selection": "llvm_site_estimate" if args.arena_mb is None else "explicit",
        "compile_timeout_s": args.compile_timeout_s,
        "first_output_sha256_enabled": output_sha256,
        "all_output_sha256_enabled": True,
        "virtual_zero_padding": args.virtual_zero_padding,
        "integer_quant_groups": args.integer_quant_groups,
        "vpu_softmax_implementation": args.vpu_softmax_implementation,
        "host_softmax_policy": args.host_softmax_policy,
        "host_softmax_rewrite": host_softmax_rewrite,
        "vpu_sigmoid_implementation": args.vpu_sigmoid_implementation,
        "fuse_gated_projections": args.fuse_gated_projections,
        "gated_activation_arithmetic": args.gated_activation_arithmetic,
        "gated_projection_rewrite": gated_projection_rewrite,
        "fuse_softmax_packing": args.fuse_softmax_packing,
        "softmax_packing_rewrite": softmax_packing_rewrite,
        "group_projections": args.group_projections,
        "projection_group_rewrite": projection_rewrite,
        "fuse_rms_packing": args.fuse_rms_packing,
        "rms_packing_rewrite": rms_rewrite,
        "fuse_f32_rms_packing": args.fuse_f32_rms_packing,
        "f32_rms_reduction": args.f32_rms_reduction,
        "f32_rms_packing_rewrite": f32_rms_rewrite,
        "native_tile_policy": args.native_tile_policy,
        **({"host_math_guard": guarded_math.to_dict()} if guarded_math is not None else {}),
        "spike_isa": args.spike_isa,
    }
    if args.spike_extlib is not None:
        extlib = args.spike_extlib.resolve(strict=True)
        if not extlib.is_file():
            raise ValueError("Spike extension must be a regular shared library")
        command = [str(riscv_root / "bin/spike"), f"--isa={args.spike_isa}",
                   f"--extlib={extlib}", "--extension=gemmini",
                   f"-m{hex(spike_model.DRAM_BASE)}:{hex(result['mem_bytes'])}",
                   str(elf)]
        log = work / "spike.log"
        spike_environment = os.environ.copy()
        with log.open("wb") as stream:
            try:
                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                           timeout=args.spike_timeout, check=False, env=spike_environment)
                exit_code = completed.returncode
            except subprocess.TimeoutExpired:
                exit_code = 124
        console = log.read_text(encoding="utf-8", errors="replace")
        receipt["spike"] = {
            "command": command, "extension_sha256": hashlib.sha256(extlib.read_bytes()).hexdigest(),
            "exit_code": exit_code, "done": exit_code == 0 and "DONE\n" in console,
            "log": str(log),
            "environment": {name: spike_environment.get(name) for name in (
                "GEMMINI_MODE", "GEMMINI_PERF_CONFIG", "GEMMINI_PERF_SET", "GEMMINI_PERF_OUT")},
        }
        if receipt["spike"]["done"]:
            try:
                receipt["spike"]["output_digests"] = _observe_spike_evidence(
                    console, source, result["build_hash"], first_typed=output_sha256,
                    require_complete=args.output_dump_cap > 1_048_576)
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
