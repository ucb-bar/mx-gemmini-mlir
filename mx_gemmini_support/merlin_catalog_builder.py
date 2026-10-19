"""Build a source-bound MX dense catalog for Merlin's external device hook.

The builder owns MX packing, schedule selection, shared serial workspace, and
RV64 object emission. Merlin owns graph parsing, exact source matching,
memref shims, host lowering, and the final link. Numerical acceptance for a
whole model still requires independent quantized reference comparisons.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Callable

from .dense_contraction import (emit_dense_batch_contraction, emit_dense_contraction,
                                plan_dense_shape)
from .dense_kernel import emit_shared_workspace_definition
from .merlin_catalog_plan import plan_merlin_catalog
from .merlin_schedule_policy import build_policy_plan, selected_policy
from .offline_weight_pack import pack_weight
from .merlin_vpu_add import emit_memref_add_shim, selected_add_sites
from .merlin_vpu_elementwise import selected_elementwise_sites
from .merlin_vpu_softmax import emit_memref_softmax_shim, selected_softmax_sites
from .merlin_vpu_unary import emit_memref_unary_shim, selected_unary_sites
from .merlin_vpu_reduce import emit_memref_reduce_shim, selected_row_reduce_sites
from .target_profile import load_profile, profile_sha256
from .vpu_add_kernel import emit_vpu_add, emit_vpu_elementwise
from .vpu_row_softmax import emit_vpu_row_softmax
from .vpu_unary_kernel import emit_vpu_unary
from .vpu_row_reduce import emit_vpu_row_reduce


_WORKSPACE_SYMBOL = "mx_catalog_workspace"
_SUPPORT = Path(__file__).resolve().parent
_QUANTIZER = _SUPPORT / "runtime_quant.c"
_QUANTIZER_LUT = _SUPPORT / "runtime_quant_lut.h"


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _run(command: list[str], *, label: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"MX {label} failed: {result.stderr[-1200:]}")


def select_routed_vpu_sites(name: str, sites: list[dict],
                            enabled: dict[str, frozenset[str]] | None) -> list[dict]:
    """Compile only VPU kernels that the captured execution policy can call."""
    if enabled is None:
        return sites
    selected = enabled[name]
    if not selected <= {site["region"] for site in sites}:
        raise ValueError(f"MX VPU {name} policy differs from prepared source")
    return [site for site in sites if site["region"] in selected]


def build_merlin_catalog(prepared: Path, out_dir: Path, *, profile: dict,
                         riscv_root: Path,
                         checkpoint_manifest: dict | None = None,
                         checkpoint_weights_path: Path | None = None,
                         cache_immutable_weights: bool = False,
                         prepack_immutable_weights: bool = False,
                         precision_overrides: dict[str, str] | None = None,
                         enabled_vpu_regions: dict[str, frozenset[str]] | None = None,
                         connect_multiplies: bool = False
                         ) -> tuple[Path, Path]:
    """Compile selected source sites through checked MX native schedules."""
    plan = plan_merlin_catalog(
        prepared, profile, checkpoint_manifest=checkpoint_manifest,
        cache_immutable_weights=cache_immutable_weights,
        prepack_immutable_weights=prepack_immutable_weights,
        precision_overrides=precision_overrides)
    if prepack_immutable_weights and (checkpoint_weights_path is None or
                                      not Path(checkpoint_weights_path).is_file() or
                                      Path(checkpoint_weights_path).is_symlink()):
        raise ValueError("MX build-time packing needs regular captured checkpoint weights")
    if not plan["kernels"]:
        raise ValueError("MX catalog has no bindable direct f32 contractions")
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError("MX catalog output directory already exists")
    riscv_root = Path(riscv_root)
    bin_dir = riscv_root / "bin"
    cc = bin_dir / "riscv64-unknown-elf-gcc"
    ld = bin_dir / "riscv64-unknown-elf-ld"
    nm = bin_dir / "riscv64-unknown-elf-nm"
    if not all(tool.is_file() for tool in (cc, ld, nm)):
        raise ValueError("MX catalog needs an explicit RV64 compiler, linker, and nm")
    out_dir.mkdir(parents=True)
    flags = ["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2",
             "-ffreestanding", "-fno-fast-math", "-fno-common",
             "-fno-tree-loop-distribute-patterns", "-std=gnu11",
             "-I", str(_SUPPORT)]
    objects: list[Path] = []
    source_hashes = {}

    def compile_source(name: str, source: str | Path) -> None:
        path = out_dir / f"{name}.c"
        if isinstance(source, Path):
            path = source
        else:
            path.write_text(source, encoding="utf-8")
        obj = out_dir / f"{name}.o"
        _run([str(cc), *flags, "-c", str(path), "-o", str(obj)],
             label=f"{name} compilation")
        objects.append(obj)
        source_hashes[name] = _sha(path)

    kernel_records = []
    max_workspace = 0
    issuer_modes = {}
    issuer_sources = {}
    fixed_data: list[tuple[str, Path]] = []
    fixed_records: dict[str, dict] = {}
    checkpoint = None
    if prepack_immutable_weights:
        import torch
        from safetensors import safe_open

        checkpoint = safe_open(str(checkpoint_weights_path), framework="pt", device="cpu")
    for entry in plan["kernels"]:
        symbol = entry["symbol"]
        issuer_symbol = entry["issuer_symbol"]
        shape = tuple(entry["shape"])
        precision = "FP4" if entry["precision"] == "mxfp4" else "FP8"
        fixed_symbols = None
        if entry["prepacked_checkpoint_weight"]:
            assert checkpoint is not None
            parameter_name = entry["checkpoint_parameter_name"]
            if parameter_name not in checkpoint.keys():
                raise ValueError(f"MX checkpoint is missing {parameter_name}")
            tensor = checkpoint.get_tensor(parameter_name)
            if list(tensor.shape) != entry["checkpoint_parameter_shape"]:
                raise ValueError(f"MX checkpoint changed shape for {parameter_name}")
            values = tensor.to(dtype=torch.float32).numpy()
            for transform in reversed(entry["checkpoint_parameter_transforms"]):
                if transform == "linalg.transpose":
                    values = values.T
                elif transform != "linalg.generic":
                    raise ValueError("MX checkpoint has an unqualified RHS transform")
            if entry["rank"] != 2 or values.shape != (shape[2], shape[1]):
                raise ValueError("MX checkpoint RHS differs from its selected contraction")
            padded = plan_dense_shape(profile, shape, precision=precision).padded_shape_mnk
            codes, scales = pack_weight(
                values, precision=precision, padded_shape=(padded[2], padded[1]),
                workdir=out_dir)
            fixed_symbols = (f"{symbol}_fixed_codes", f"{symbol}_fixed_scales")
            fixed_records[symbol] = {
                "checkpoint_parameter_name": parameter_name,
                "codes_sha256": hashlib.sha256(codes).hexdigest(),
                "scales_sha256": hashlib.sha256(scales).hexdigest(),
                "codes_bytes": len(codes), "scales_bytes": len(scales),
            }
            for suffix, payload, fixed_symbol in zip(
                    ("codes", "scales"), (codes, scales), fixed_symbols, strict=True):
                path = out_dir / f"{symbol}_{suffix}.bin"
                path.write_bytes(payload)
                fixed_data.append((fixed_symbol, path))
        policy_text = build_policy_plan(symbol, schedule="native")
        selected = selected_policy(policy_text, symbol=symbol)
        policy_path = out_dir / f"{symbol}.schedule.mlir"
        policy_path.write_text(policy_text, encoding="utf-8")
        if entry["rank"] == 2:
            generated = emit_dense_contraction(
                profile, shape, precision=precision, symbol=symbol,
                issuer_symbol=issuer_symbol,
                shared_workspace_symbol=_WORKSPACE_SYMBOL,
                cache_immutable_weight=entry["cache_immutable_weight"],
                prepacked_weight_symbols=fixed_symbols,
                schedule=selected)
            batch, m, n, k = 1, *shape
        elif entry["rank"] == 3:
            generated = emit_dense_batch_contraction(
                profile, shape, precision=precision, symbol=symbol,
                issuer_symbol=issuer_symbol,
                shared_workspace_symbol=_WORKSPACE_SYMBOL,
                schedule=selected)
            batch, m, n, k = shape
        else:
            raise ValueError("MX catalog has an unsupported source rank")
        previous_issuer = issuer_sources.get(issuer_symbol)
        if previous_issuer is None:
            compile_source(issuer_symbol, generated.issuer.source)
            issuer_sources[issuer_symbol] = generated.issuer.source
        elif previous_issuer != generated.issuer.source:
            raise ValueError("MX shared issuer differs across cached source sites")
        compile_source(symbol, generated.kernel.source)
        if generated.schedule_text is None or generated.schedule_digest is None:
            raise ValueError("MX catalog kernel has no checked schedule IR")
        schedule_path = out_dir / f"{issuer_symbol}.mk"
        schedule_path.write_text(generated.schedule_text, encoding="utf-8")
        if _sha(schedule_path) != generated.schedule_digest:
            raise ValueError("MX schedule artifact differs from its canonical digest")
        max_workspace = max(max_workspace, generated.kernel.workspace_bytes)
        issuer_modes[symbol] = generated.issuer_mode
        kernel_records.append({
            "symbol": symbol, "batched": entry["rank"] == 3,
            "dimensions": {"batch": batch, "m": m, "n": n, "k": k},
            "precision": entry["precision"],
            "issuer_mode": generated.issuer_mode,
            "policy_ir": "merlin.schedule",
            "policy_file": policy_path.name,
            "policy_sha256": _sha(policy_path),
            "schedule_ir": "merlin.mk",
            "schedule_file": schedule_path.name,
            "schedule_sha256": generated.schedule_digest,
            "instruction_set_sha256": generated.instruction_set_digest,
            "workspace_bytes": generated.kernel.workspace_bytes,
            "persistent_weight_bytes": getattr(generated.kernel, "persistent_bytes", 0),
            "cache_immutable_weight": entry["cache_immutable_weight"],
            "prepacked_checkpoint_weight": entry["prepacked_checkpoint_weight"],
            **fixed_records.get(symbol, {}),
            **({"checkpoint_parameter_arg": entry["checkpoint_parameter_arg"],
                "checkpoint_parameter_name": entry["checkpoint_parameter_name"],
                "checkpoint_parameter_transforms": entry["checkpoint_parameter_transforms"],
                "checkpoint_parameter_shape": entry["checkpoint_parameter_shape"]}
               if (entry["cache_immutable_weight"] or
                   entry["prepacked_checkpoint_weight"]) else {}),
            "padded_shape_mnk": list(generated.shape_plan.padded_shape_mnk),
            "tile_mnk": list(generated.shape_plan.tile_mnk),
        })
    prepared_text = Path(prepared).read_text()
    route_aliases = {}
    if enabled_vpu_regions is not None:
        from .execution_policy import reconcile_vpu_regions

        enabled_vpu_regions, route_aliases = reconcile_vpu_regions(
            prepared_text, profile, enabled_vpu_regions)

    def routed(name: str, sites: list[dict]) -> list[dict]:
        return select_routed_vpu_sites(name, sites, enabled_vpu_regions)

    softmax_sites = routed("softmax", selected_softmax_sites(prepared_text, profile))
    softmax_shapes = {tuple(site["shape"]): site for site in softmax_sites}
    softmax_kernels = []
    for shape, site in softmax_shapes.items():
        generated = emit_vpu_row_softmax(
            profile, logical_rows=site["logical_rows"],
            row_width=site["row_width"], symbol=site["kernel_symbol"])
        compile_source(site["kernel_symbol"], generated.source)
        compile_source(site["symbol"], emit_memref_softmax_shim(shape))
        softmax_kernels.append({
            "shape": list(shape), "logical_rows": site["logical_rows"],
            "row_width": site["row_width"],
            "padded_width": generated.padded_width,
            "rows_per_chunk": generated.rows_per_chunk,
            "symbol": site["symbol"],
            "kernel_symbol": site["kernel_symbol"],
            "vpu_steps": list(generated.vpu_steps),
        })
    add_sites = routed("add", selected_add_sites(prepared_text, profile))
    add_shapes = {site["symbol"]: site for site in add_sites}
    add_kernels = []
    for site in add_shapes.values():
        generated = emit_vpu_add(
            profile, elements=site["elements"], row_width=site["row_width"],
            dtype=site["dtype"], broadcast=site["broadcast"],
            symbol=site["kernel_symbol"])
        compile_source(site["kernel_symbol"], generated.source)
        compile_source(site["symbol"], emit_memref_add_shim(site))
        add_kernels.append({
            "symbol": site["symbol"], "kernel_symbol": site["kernel_symbol"],
            "shape": list(site["shape"]), "rhs_shape": list(site["rhs_shape"]),
            "dtype": site["dtype"], "broadcast": site["broadcast"],
            "elements_per_chunk": generated.elements_per_chunk,
        })
    elementwise_sites = routed("elementwise", selected_elementwise_sites(
        prepared_text, profile))
    connected_multiplies = []
    if connect_multiplies:
        from merlin.frontends.linalg_mlir import parse_mlir_file
        from .merlin_connected_multiply import emit_chain_kernel, emit_chain_shim, multiply_pairs

        pairs = multiply_pairs(parse_mlir_file(prepared), elementwise_sites)
        compiled = {}
        for pair in pairs:
            if pair["symbol"] not in compiled:
                source, removed = emit_chain_kernel(pair, profile)
                compile_source(pair["kernel_symbol"], source)
                compile_source(pair["symbol"], emit_chain_shim(pair))
                compiled[pair["symbol"]] = removed
            connected_multiplies.append({key: pair[key] for key in
                                         ("regions", "shape", "symbol", "kernel_symbol")} |
                                        {"removed_transfer_bytes": compiled[pair["symbol"]]})
    elementwise_shapes = {site["symbol"]: site for site in elementwise_sites}
    elementwise_kernels = []
    for site in elementwise_shapes.values():
        generated = emit_vpu_elementwise(
            profile, result_shape=site["shape"],
            lhs_shape=site["lhs_shape"], lhs_axes=site["lhs_axes"],
            rhs_shape=site["rhs_shape"],
            dtype=site["dtype"], lhs_dtype=site["lhs_dtype"],
            rhs_dtype=site["rhs_dtype"], operation=site["operation"],
            symbol=site["kernel_symbol"], rhs_axes=site["rhs_axes"])
        compile_source(site["kernel_symbol"], generated.source)
        compile_source(site["symbol"], emit_memref_add_shim(site))
        elementwise_kernels.append({
            "symbol": site["symbol"], "kernel_symbol": site["kernel_symbol"],
            "operation": site["operation"], "dtype": site["dtype"],
            "lhs_dtype": site["lhs_dtype"], "rhs_dtype": site["rhs_dtype"],
            "shape": list(site["shape"]),
            "lhs_shape": list(site["lhs_shape"]),
            "lhs_axes": list(site["lhs_axes"]),
            "rhs_shape": list(site["rhs_shape"]),
            "rhs_axes": list(site["rhs_axes"]),
            "elements_per_chunk": generated.elements_per_chunk,
        })
    unary_sites = routed("unary", selected_unary_sites(prepared_text, profile))
    unary_shapes = {site["symbol"]: site for site in unary_sites}
    unary_kernels = []
    for site in unary_shapes.values():
        generated = emit_vpu_unary(
            profile, shape=site["shape"], dtype=site["dtype"],
            operation=site["operation"], symbol=site["kernel_symbol"])
        compile_source(site["kernel_symbol"], generated.source)
        compile_source(site["symbol"], emit_memref_unary_shim(site))
        unary_kernels.append({
            "symbol": site["symbol"], "kernel_symbol": site["kernel_symbol"],
            "operation": site["operation"], "dtype": site["dtype"],
            "shape": list(site["shape"]),
            "elements_per_chunk": generated.elements_per_chunk,
        })
    reduce_sites = routed("reduce", selected_row_reduce_sites(prepared_text, profile))
    reduce_shapes = {site["symbol"]: site for site in reduce_sites}
    reduce_kernels = []
    for site in reduce_shapes.values():
        generated = emit_vpu_row_reduce(
            profile, logical_rows=site["logical_rows"],
            row_width=site["row_width"], operation=site["operation"],
            symbol=site["kernel_symbol"])
        compile_source(site["kernel_symbol"], generated.source)
        compile_source(site["symbol"], emit_memref_reduce_shim(site))
        reduce_kernels.append({
            "symbol": site["symbol"], "kernel_symbol": site["kernel_symbol"],
            "operation": site["operation"], "shape": list(site["shape"]),
            "output_shape": list(site["output_shape"]),
            "padded_width": generated.padded_width,
            "rows_per_chunk": generated.rows_per_chunk,
        })
    compile_source("runtime_quant", _QUANTIZER)
    source_hashes["runtime_quant_lut"] = _sha(_QUANTIZER_LUT)
    compile_source("shared_workspace", emit_shared_workspace_definition(
        max_workspace, symbol=_WORKSPACE_SYMBOL))
    if fixed_data:
        lines = [".section .rodata", ".balign 64"]
        for fixed_symbol, path in fixed_data:
            escaped = str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"')
            lines.extend((f".global {fixed_symbol}", f"{fixed_symbol}:",
                          f'  .incbin "{escaped}"', ".balign 64"))
        assembly = out_dir / "checkpoint_weights.S"
        assembly.write_text("\n".join(lines) + "\n", encoding="utf-8")
        obj = out_dir / "checkpoint_weights.o"
        _run([str(cc), "-march=rv64gc", "-mabi=lp64d", "-c", str(assembly),
              "-o", str(obj)], label="prepacked checkpoint assembly")
        objects.append(obj)
        source_hashes["checkpoint_weights"] = _sha(assembly)
    object_path = out_dir / "mx_catalog.o"
    _run([str(ld), "-r", "-o", str(object_path), *(str(obj) for obj in objects)],
         label="relocatable catalog link")
    symbols = subprocess.run([str(nm), "--format=posix", "--extern-only",
                              str(object_path)], capture_output=True, text=True, check=True)
    rows = [line.split() for line in symbols.stdout.splitlines()]
    defined = [row[0] for row in rows if len(row) >= 2 and row[1].upper() == "T"]
    if (any(len(row) < 2 or row[1].upper() == "U" for row in rows) or
            any(defined.count(entry["symbol"]) != 1 for entry in plan["kernels"]) or
            any(defined.count(site[name]) != 1 for site in softmax_shapes.values()
                for name in ("symbol", "kernel_symbol")) or
            any(defined.count("_mlir_ciface_" + site["symbol"]) != 1
                for site in softmax_shapes.values()) or
            any(defined.count(site[name]) != 1 for site in add_shapes.values()
                for name in ("symbol", "kernel_symbol")) or
            any(defined.count("_mlir_ciface_" + site["symbol"]) != 1
                for site in add_shapes.values()) or
            any(defined.count(site[name]) != 1 for site in elementwise_shapes.values()
                for name in ("symbol", "kernel_symbol")) or
            any(defined.count("_mlir_ciface_" + site["symbol"]) != 1
                for site in elementwise_shapes.values()) or
            any(defined.count(site[name]) != 1 for site in connected_multiplies
                for name in ("symbol", "kernel_symbol")) or
            any(defined.count("_mlir_ciface_" + site["symbol"]) != 1
                for site in connected_multiplies) or
            any(defined.count(site[name]) != 1 for site in unary_shapes.values()
                for name in ("symbol", "kernel_symbol")) or
            any(defined.count("_mlir_ciface_" + site["symbol"]) != 1
                for site in unary_shapes.values()) or
            any(defined.count(site[name]) != 1 for site in reduce_shapes.values()
                for name in ("symbol", "kernel_symbol")) or
            any(defined.count("_mlir_ciface_" + site["symbol"]) != 1
                for site in reduce_shapes.values())):
        raise ValueError("MX catalog object has unresolved or duplicate kernel symbols")
    target_dtypes = sorted({tuple(entry["target_dtypes"]) for entry in plan["kernels"]})
    policy = {
        "schema": "mx_gemmini.dense_f32_conversion.v1",
        "input_rounding": "bf16_rne",
        "scale": "e8m0_block32",
        "fp4": "e2m1_via_e3m1_intermediate",
        "fp8": "e4m3_rne",
        "edge": "zero_pad_to_mesh_and_wave",
        "accumulation_and_readout": "bf16_then_exact_widen_to_f32",
        "batch": "serial_disjoint_slices",
    }
    policy_sha = hashlib.sha256(json.dumps(
        policy, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    manifest = {
        "schema": "mx_gemmini.merlin_device_catalog.v1",
        "status": "compiled_experimental_numeric_qualification_pending",
        "source_sha256": plan["source_sha256"],
        "profile_sha256": profile_sha256(profile),
        "abi": {"argument_order": ["lhs", "rhs", "out"],
                "pointee_layout": "dense_row_major", "batch_call": "whole_batch",
                "dtypes": ["f32", "f32", "f32"]},
        "source_conversion": {
            "schema": "target_owned_dtype_conversion_v1",
            "source_dtypes": ["f32", "f32", "f32"],
            "target_dtypes": [list(row) for row in target_dtypes],
            "policy": policy, "policy_sha256": policy_sha,
        },
        "coverage_complete": True,
        "matched_contractions": len(plan["bindings"]),
        "covered_contractions": len(plan["bindings"]),
        "precision_proposals": plan["precision_proposals"],
        "unbound_precision_proposals": plan["unbound_precision_proposals"],
        "unique_kernels": len(kernel_records),
        "unique_issuers": len(issuer_sources),
        "cached_checkpoint_kernels": plan["cached_checkpoint_kernels"],
        "prepacked_checkpoint_kernels": plan["prepacked_checkpoint_kernels"],
        "checkpoint_weights_sha256": (_sha(Path(checkpoint_weights_path))
                                      if prepack_immutable_weights else None),
        "kernels": kernel_records,
        "bindings": plan["bindings"],
        "precision_overrides": plan.get("precision_overrides", {}),
        "resolved_vpu_regions": ({name: sorted(sites) for name, sites in
                                  enabled_vpu_regions.items()}
                                 if enabled_vpu_regions is not None else None),
        "vpu_source_aliases": route_aliases,
        "vpu_softmax": {
            "status": "source_bound_f32_to_bf16_vpu_to_f32",
            "numeric_policy": "f32_scores_to_bf16_rne_vpu_round_each_step_to_f32_probabilities",
            "source_regions": [site["region"] for site in softmax_sites],
            "kernels": softmax_kernels,
        },
        "vpu_add": {
            "status": "source_bound_bf16_or_explicit_f32_conversion",
            "numeric_policy": "bf16_vpu_add_or_f32_to_bf16_rne_vpu_add_to_f32",
            "source_regions": [site["region"] for site in add_sites],
            "kernels": add_kernels,
        },
        "vpu_elementwise": {
            "status": "source_bound_bf16_or_explicit_f32_conversion",
            "numeric_policy": "f32_to_bf16_rne_vpu_to_f32_or_direct_bf16",
            "source_regions": [site["region"] for site in elementwise_sites],
            "kernels": elementwise_kernels,
            "connected_multiplies": connected_multiplies,
            "connected_selection": "experimental_legality_only" if connect_multiplies else "disabled",
        },
        "vpu_unary": {
            "status": "source_bound_bf16_or_explicit_f32_conversion",
            "numeric_policy": "f32_to_bf16_rne_vpu_unary_to_f32_or_direct_bf16",
            "source_regions": [site["region"] for site in unary_sites],
            "kernels": unary_kernels,
        },
        "vpu_reduce": {
            "status": "source_bound_f32_to_bf16_reduce_to_f32",
            "numeric_policy": "f32_to_bf16_rne_vpu_row_reduce_to_f32",
            "source_regions": [site["region"] for site in reduce_sites],
            "kernels": reduce_kernels,
        },
        "compilation": {
            "object_sha256": _sha(object_path),
            "source_sha256": source_hashes,
            "shared_workspace_bytes": max_workspace,
            "issuer_modes": issuer_modes,
            "compiler": str(cc), "flags": flags,
        },
    }
    manifest_path = out_dir / "mx_catalog.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    return manifest_path, object_path


def make_merlin_catalog_builder(*, profile_path: Path,
                                riscv_root: Path,
                                cached_weight_stage: Path | None = None,
                                prepacked_weight_stage: Path | None = None,
                                precision_overrides: dict[str, str] | None = None,
                                enabled_vpu_regions: dict[str, frozenset[str]] | None = None,
                                connect_multiplies: bool = False
                                ) -> Callable[[Path, Path], tuple[Path, Path]]:
    """Return the target-owned callback accepted by DeviceRouting."""
    profile_path = Path(profile_path).resolve(strict=True)
    riscv_root = Path(riscv_root).resolve(strict=True)
    if cached_weight_stage is not None and prepacked_weight_stage is not None:
        raise ValueError("MX checkpoint weights need one preparation mode")
    selected_stage = (cached_weight_stage if cached_weight_stage is not None
                      else prepacked_weight_stage)
    manifest_path = (Path(selected_stage).resolve(strict=True) /
                     "weights.safetensors.manifest.json"
                     if selected_stage is not None else None)
    if manifest_path is not None and (manifest_path.is_symlink()
                                      or not manifest_path.is_file()):
        raise ValueError("MX checkpoint cache needs a captured weights manifest")

    def build(prepared: Path, work: Path) -> tuple[Path, Path]:
        return build_merlin_catalog(prepared, work,
                                    profile=load_profile(profile_path),
                                    riscv_root=riscv_root,
                                    checkpoint_manifest=(json.loads(manifest_path.read_text())
                                                         if manifest_path else None),
                                    checkpoint_weights_path=(manifest_path.with_name(
                                        "weights.safetensors") if prepacked_weight_stage is not None
                                                             else None),
                                    cache_immutable_weights=cached_weight_stage is not None,
                                    prepack_immutable_weights=prepacked_weight_stage is not None,
                                    precision_overrides=precision_overrides,
                                    enabled_vpu_regions=enabled_vpu_regions,
                                    connect_multiplies=connect_multiplies)

    return build
