"""Build a source-bound MX dense catalog for Merlin's external device hook.

The builder owns MX packing, schedule selection, shared serial workspace, and
RV64 object emission. Merlin owns graph parsing, exact source matching,
memref shims, host lowering, and the final link. Numerical acceptance for a
whole model still requires independent quantized reference comparisons.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from .dense_contraction import (
    emit_dense_batch_contraction,
    emit_dense_contraction,
    plan_dense_shape,
)
from .dense_kernel import emit_shared_workspace_definition
from .merlin_catalog_plan import plan_merlin_catalog
from .merlin_schedule_policy import build_policy_plan, selected_policy
from .merlin_vpu_add import emit_memref_add_shim, selected_add_sites
from .merlin_vpu_elementwise import selected_elementwise_sites
from .merlin_vpu_reduce import emit_memref_reduce_shim, selected_row_reduce_sites
from .merlin_vpu_softmax import emit_memref_softmax_shim, selected_softmax_sites
from .merlin_vpu_unary import emit_memref_unary_shim, selected_unary_sites
from .offline_weight_pack import pack_weight
from .target_profile import load_profile, profile_sha256
from .vpu_add_kernel import emit_vpu_add, emit_vpu_elementwise
from .vpu_row_reduce import emit_vpu_row_reduce
from .vpu_row_softmax import emit_vpu_row_softmax
from .vpu_unary_kernel import emit_vpu_unary

_WORKSPACE_SYMBOL = "mx_catalog_workspace"
_SUPPORT = Path(__file__).resolve().parent
_QUANTIZER = _SUPPORT / "runtime_quant.c"
_QUANTIZER_LUT = _SUPPORT / "runtime_quant_lut.h"


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _run(command: list[str], *, label: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"MX {label} failed: {result.stderr[-1200:]}")


def _write_packed_data(path: Path, payload: bytes, reuse_dir: Path | None) -> None:
    """Reuse storage only after freshly derived packed bytes prove its identity."""
    if reuse_dir is None:
        path.write_bytes(payload)
        return
    source = reuse_dir / path.name
    expected = hashlib.sha256(payload).hexdigest()
    if (source.is_symlink() or not source.is_file() or
            source.stat().st_size != len(payload) or _sha(source) != expected):
        raise ValueError("packed weight reuse differs from freshly computed bytes: " + str(source))
    os.link(source, path)
    if not os.path.samefile(source, path) or _sha(path) != expected:
        raise ValueError("packed weight reuse changed during binding")


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
                         connect_multiplies: bool = False,
                         connect_broadcast_multiplies: bool = False,
                         prepare_broadcast_operands: bool = False,
                         broadcast_preparation_budget_bytes: int = 1048576,
                         connect_sigmoid_multiplies: bool = False,
                         share_activation_packing: bool = False,
                         retain_shared_activation_rhs: bool = False,
                         matrix_vpu_transport: bool = False,
                         vpu_matrix_transport: bool = False,
                         virtual_zero_padding: bool = False,
                         integer_group_packing: bool = False,
                         reuse_batch_rhs: bool = False,
                         compact_batch_rhs: bool = False,
                         native_tile_policy: str = "legacy",
                         reuse_prepacked_data: Path | None = None,
                         reuse_prepacked_object: Path | None = None
                         ) -> tuple[Path, Path]:
    """Compile selected source sites through checked MX native schedules."""
    if connect_broadcast_multiplies and not connect_multiplies:
        raise ValueError("broadcast connections require connected multiplies")
    if prepare_broadcast_operands and not connect_broadcast_multiplies:
        raise ValueError("broadcast preparation requires broadcast connections")
    if type(broadcast_preparation_budget_bytes) is not int or broadcast_preparation_budget_bytes < 0:
        raise ValueError("broadcast preparation requires a nonnegative integer byte budget")
    if type(integer_group_packing) is not bool:
        raise ValueError("integer group packing requires an explicit boolean selection")
    if type(reuse_batch_rhs) is not bool or reuse_batch_rhs and not integer_group_packing:
        raise ValueError("batch RHS reuse requires explicit integer group packing")
    if type(compact_batch_rhs) is not bool or compact_batch_rhs and not reuse_batch_rhs:
        raise ValueError("compact batch RHS requires explicit batch preparation reuse")
    if type(vpu_matrix_transport) is not bool:
        raise ValueError("VPU matrix transport requires an explicit boolean selection")
    if vpu_matrix_transport and not (share_activation_packing and connect_multiplies and integer_group_packing):
        raise ValueError("VPU matrix transport requires shared packing, connected multiplies and integer groups")
    if native_tile_policy not in {"legacy", "movement"}:
        raise ValueError("native tile policy must be legacy or movement")
    if share_activation_packing and not prepack_immutable_weights:
        raise ValueError("shared activation packing requires immutable prepacked weights")
    if connect_sigmoid_multiplies and (not connect_multiplies or enabled_vpu_regions is None
                                      or not prepack_immutable_weights):
        raise ValueError("sigmoid chains require connected multiplies, prepacking and explicit site policy")
    if matrix_vpu_transport and not prepack_immutable_weights:
        raise ValueError("matrix transport requires immutable prepacked weights")
    if reuse_prepacked_object is not None and not prepack_immutable_weights:
        raise ValueError("constant object reuse requires immutable prepacked weights")
    if retain_shared_activation_rhs and not share_activation_packing:
        raise ValueError("retained shared RHS requires shared activation packing")
    if reuse_prepacked_data is not None:
        reuse_prepacked_data = Path(reuse_prepacked_data)
        if (not prepack_immutable_weights or reuse_prepacked_data.is_symlink()
                or not reuse_prepacked_data.is_dir()):
            raise ValueError("packed data reuse needs explicit prepacking and a regular source directory")
    plan = plan_merlin_catalog(
        prepared, profile, checkpoint_manifest=checkpoint_manifest,
        cache_immutable_weights=cache_immutable_weights,
        prepack_immutable_weights=prepack_immutable_weights,
        precision_overrides=precision_overrides)
    repeated_inputs = {}
    if reuse_batch_rhs:
        from .merlin_batch_rhs import analyze_bound_repetitions

        repeated_inputs = analyze_bound_repetitions(prepared, plan)
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
    if reuse_batch_rhs:
        snapshot = out_dir / "batch_rhs_source.mlir"
        snapshot.write_bytes(Path(prepared).read_bytes())
        if _sha(snapshot) != plan["source_sha256"]:
            raise ValueError("batch RHS source changed before snapshot binding")
    flags = ["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2",
             "-ffreestanding", "-fno-fast-math", "-fno-common",
             "-fno-tree-loop-distribute-patterns", "-std=gnu11",
             "-I", str(_SUPPORT)]
    objects: list[Path] = []
    source_hashes = {}
    source_flags = {}
    # The same immutable source bytes encode checkpoint operands and build the
    # runtime object. Retain their complete local header inputs with the catalog.
    runtime_source = out_dir / "runtime_quant.c"
    for original in (_QUANTIZER, _QUANTIZER_LUT, _SUPPORT / "runtime_quant.h"):
        (out_dir / original.name).write_bytes(original.read_bytes())
    source_hashes["runtime_quant_lut"] = _sha(out_dir / "runtime_quant_lut.h")
    source_hashes["runtime_quant_header"] = _sha(out_dir / "runtime_quant.h")

    def compile_source(name: str, source: str | Path, *, extra_flags: tuple[str, ...] = ()) -> None:
        path = out_dir / f"{name}.c"
        if isinstance(source, Path):
            path.write_bytes(source.read_bytes())
        else:
            path.write_text(source, encoding="utf-8")
        obj = out_dir / f"{name}.o"
        _run([str(cc), *flags, *extra_flags, "-c", str(path), "-o", str(obj)],
             label=f"{name} compilation")
        objects.append(obj)
        source_hashes[name] = _sha(path)
        if extra_flags:
            source_flags[name] = list(extra_flags)

    kernel_records = []
    max_workspace = 0
    issuer_modes = {}
    issuer_sources = {}
    fixed_data: list[tuple[str, Path]] = []
    fixed_records: dict[str, dict] = {}
    checkpoint = None
    preparation_sources = {}
    prepared_entries = {}
    compact_preparation_selected = False
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
            if parameter_name not in checkpoint.keys():  # noqa: SIM118 -- safe_open exposes keys, not membership
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
                workdir=out_dir, quantizer_source=runtime_source)
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
                _write_packed_data(path, payload, reuse_prepacked_data)
                fixed_data.append((fixed_symbol, path))
        policy_text = build_policy_plan(symbol, schedule="native" if native_tile_policy == "legacy" else "movement")
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
                virtual_zero_padding=virtual_zero_padding,
                schedule=selected)
            batch, m, n, k = 1, *shape
        elif entry["rank"] == 3:
            generated = emit_dense_batch_contraction(
                profile, shape, precision=precision, symbol=symbol,
                issuer_symbol=issuer_symbol,
                shared_workspace_symbol=_WORKSPACE_SYMBOL,
                virtual_zero_padding=virtual_zero_padding,
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
        if reuse_batch_rhs and entry["rank"] == 3:
            from .merlin_batch_rhs import register_repetitions

            register_repetitions(entry, generated, plan["bindings"], repeated_inputs,
                                 _WORKSPACE_SYMBOL, compile_source, prepared_entries,
                                 virtual_zero_padding=virtual_zero_padding, compact_rhs=compact_batch_rhs)
        activation_preparation = None
        private_bf16_transport = None
        prepared_private_bf16_transport = None
        if matrix_vpu_transport and fixed_symbols is not None:
            from .merlin_matrix_transport import register_matrix_transport

            private_bf16_transport = register_matrix_transport(
                generated, shape, symbol, issuer_symbol, fixed_symbols,
                _WORKSPACE_SYMBOL, compile_source, prepared_entries,
                virtual_zero_padding=virtual_zero_padding)
        if share_activation_packing and fixed_symbols is not None:
            from merlin.llvmlower.device_shim import LogicalBuffer

            from .dense_kernel import emit_activation_preparation, emit_dense_f32_kernel

            pm, _, pk = generated.issuer.shape_mnk
            prepare_entry = f"mx_prepare_a_{precision.lower()}_m{m}_k{k}_pm{pm}_pk{pk}"
            prepare_kernel = prepare_entry + "_kernel"
            source, packed_bytes, padding_bytes = emit_activation_preparation(
                generated.issuer, shape, symbol=prepare_kernel, virtual_zero_padding=virtual_zero_padding)
            previous = preparation_sources.get(prepare_entry)
            if previous is None:
                compile_source(prepare_kernel, source)
                preparation_sources[prepare_entry] = source
            elif previous != source:
                raise ValueError("shared activation preparation changed across consumers")
            packed_entry, packed_kernel = symbol + "_prepared", symbol + "_prepared_kernel"
            packed = emit_dense_f32_kernel(
                generated.issuer, shape, symbol=packed_kernel, issuer_symbol=issuer_symbol,
                shared_workspace_symbol=_WORKSPACE_SYMBOL,
                prepacked_weight_symbols=fixed_symbols, prepared_activation=True,
                retain_prepared_rhs=retain_shared_activation_rhs)
            compile_source(packed_kernel, packed.source)
            prepared_entries[prepare_entry] = (prepare_kernel, (
                LogicalBuffer("activation", "input", "input_0", (m, k), "f32"),
                LogicalBuffer("packed", "output", "out_0", (packed_bytes,), "i8")))
            prepared_entries[packed_entry] = (packed_kernel, (
                LogicalBuffer("packed", "input", "input_0", (packed_bytes,), "i8"),
                *((LogicalBuffer("unused_rhs", "input", "input_1", (k, n), "f32"),)
                  if retain_shared_activation_rhs else ()),
                LogicalBuffer("output", "output", "out_0", (m, n), "f32")))
            activation_preparation = {
                "prepare_symbol": prepare_entry, "prepare_kernel": prepare_kernel,
                "consumer_symbol": packed_entry, "consumer_kernel": packed_kernel,
                "packed_bytes": packed_bytes, "padding_workspace_bytes": padding_bytes,
                "consumer_workspace_bytes": packed.workspace_bytes,
                "consumer_arguments": [0, 1, 2] if retain_shared_activation_rhs else [0, 2],
                "ownership": "explicit_readonly_ssa_tensor",
                "completion": "synchronous_native_issuer",
            }
            if vpu_matrix_transport and not (virtual_zero_padding and m < pm):
                from .merlin_vpu_matrix_transport import register_preparation

                compact = register_preparation(generated, shape, prepare_entry, compile_source,
                                                prepared_entries, preparation_sources)
                if compact["padding_workspace_bytes"] > padding_bytes:
                    raise ValueError("compact preparation increased padding scratch")
                activation_preparation["private_bf16_input"] = compact
                compact_preparation_selected = True
            if matrix_vpu_transport:
                prepared_private_bf16_transport = register_matrix_transport(
                    generated, shape, symbol, issuer_symbol, fixed_symbols,
                    _WORKSPACE_SYMBOL, compile_source, prepared_entries,
                    prepared_activation=True, retain_prepared_rhs=retain_shared_activation_rhs)
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
            **({"activation_preparation": activation_preparation}
               if activation_preparation is not None else {}),
            **({"private_bf16_transport": private_bf16_transport}
               if private_bf16_transport is not None else {}),
            **({"prepared_private_bf16_transport": prepared_private_bf16_transport}
               if prepared_private_bf16_transport is not None else {}),
            **fixed_records.get(symbol, {}),
            **({"checkpoint_parameter_arg": entry["checkpoint_parameter_arg"],
                "checkpoint_parameter_name": entry["checkpoint_parameter_name"],
                "checkpoint_parameter_transforms": entry["checkpoint_parameter_transforms"],
                "checkpoint_parameter_shape": entry["checkpoint_parameter_shape"]}
               if (entry["cache_immutable_weight"] or
                   entry["prepacked_checkpoint_weight"]) else {}),
            "padded_shape_mnk": list(generated.shape_plan.padded_shape_mnk),
            "tile_mnk": list(generated.shape_plan.tile_mnk),
            **({"native_tile_selection": generated.issuer.plan["tile_selection"]}
               if "tile_selection" in generated.issuer.plan else {}),
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
        private_bf16_inputs = []
        if matrix_vpu_transport:
            from .merlin_matrix_transport import register_add_transports

            private_bf16_inputs = register_add_transports(site, profile, compile_source, prepared_entries)
        add_kernels.append({
            "symbol": site["symbol"], "kernel_symbol": site["kernel_symbol"],
            "shape": list(site["shape"]), "rhs_shape": list(site["rhs_shape"]),
            "dtype": site["dtype"], "broadcast": site["broadcast"],
            "elements_per_chunk": generated.elements_per_chunk,
            **({"private_bf16_inputs": private_bf16_inputs} if private_bf16_inputs else {}),
        })
    elementwise_sites = routed("elementwise", selected_elementwise_sites(
        prepared_text, profile))
    unary_sites = routed("unary", selected_unary_sites(prepared_text, profile))
    connected_sigmoid_multiplies = []
    fused_regions = set()
    if connect_sigmoid_multiplies:
        from merlin.frontends.linalg_mlir import parse_mlir_file

        from .merlin_connected_sigmoid import (
            chain_record,
            register_chain,
            sigmoid_chains,
        )

        chains = sigmoid_chains(parse_mlir_file(prepared), elementwise_sites, unary_sites)
        compiled_chains = {}
        for chain in chains:
            if chain["symbol"] not in compiled_chains:
                compiled_chains[chain["symbol"]] = register_chain(
                    chain, profile, compile_source, prepared_entries)
            connected_sigmoid_multiplies.append(chain_record(chain) | compiled_chains[chain["symbol"]])
            fused_regions.update(chain["regions"])
    connected_multiplies = []
    if connect_multiplies:
        from merlin.frontends.linalg_mlir import parse_mlir_file

        from .merlin_connected_multiply import (
            chain_adapter,
            emit_chain_kernel,
            emit_chain_shim,
            multiply_pairs,
        )

        pairs = multiply_pairs(parse_mlir_file(prepared), elementwise_sites,
                               include_broadcasts=connect_broadcast_multiplies)
        compiled = {}
        for pair in pairs:
            if fused_regions.intersection(pair["regions"]):
                continue
            if pair["symbol"] not in compiled:
                selection = {"prepare_broadcasts": prepare_broadcast_operands,
                             "broadcast_preparation_budget_bytes": broadcast_preparation_budget_bytes}
                source, removed = emit_chain_kernel(pair, profile, **selection)
                compile_source(pair["kernel_symbol"], source)
                compile_source(pair["symbol"], emit_chain_shim(pair))
                _, adapter = chain_adapter(pair, profile, **selection)
                compiled[pair["symbol"]] = {
                    "removed_transfer_bytes": removed,
                    "prepared_input_bytes": sum(plan.bytes for plan in adapter.broadcast_preparations),
                    "broadcast_preparations": [asdict(plan) for plan in adapter.broadcast_preparations]}
                if vpu_matrix_transport and compact_preparation_selected:
                    from .merlin_vpu_matrix_transport import register_producer

                    compiled[pair["symbol"]]["private_bf16_output"] = register_producer(
                        pair, profile, selection, compile_source, prepared_entries)
            connected_multiplies.append({key: pair[key] for key in
                                         ("regions", "shape", "input_shapes", "input_axes", "middle_index",
                                          "symbol", "kernel_symbol")} |
                                        compiled[pair["symbol"]])
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
    if prepared_entries:
        from merlin.llvmlower.device_shim import emit_logical_translation_unit

        unit = emit_logical_translation_unit("mx_gemmini", {
            entry: (buffers, [{"role": b.role, "shape": b.shape, "dtype": b.dtype} for b in buffers])
            for entry, (_, buffers) in prepared_entries.items()
        }, kernel_symbol_for=lambda entry: prepared_entries[entry][0])
        if unit.skipped or set(unit.symbols) != set(prepared_entries):
            raise ValueError("shared activation ABI could not be emitted")
        compile_source("shared_activation_shims", unit.text)
    quant_flags = ()
    if integer_group_packing:
        from .runtime_group_preparation import emit_runtime_group_preparation

        group_header = out_dir / "runtime_group_prepare.h"
        group_header.write_text(emit_runtime_group_preparation(
            profile["resources"]["scale_block_elements"], bf16_input=compact_preparation_selected))
        source_hashes["runtime_group_prepare"] = _sha(group_header)
        quant_flags = ("-DMX_QUANT_GROUP_WORDS", "-DMX_QUANT_INLINE_GROUP", "-I", str(out_dir))
        if compact_preparation_selected:
            quant_flags += ("-DMX_QUANT_BF16_INPUT",)
    compile_source("runtime_quant", runtime_source, extra_flags=quant_flags)
    compile_source("shared_workspace", emit_shared_workspace_definition(
        max_workspace, symbol=_WORKSPACE_SYMBOL))
    packed_object_storage_reuse = None
    if fixed_data:
        lines = [".section .rodata", ".balign 64"]
        for fixed_symbol, path in fixed_data:
            escaped = str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"')
            lines.extend((f".global {fixed_symbol}", f"{fixed_symbol}:",
                          f'  .incbin "{escaped}"', ".balign 64"))
        assembly = out_dir / "checkpoint_weights.S"
        assembly.write_text("\n".join(lines) + "\n", encoding="utf-8")
        obj = out_dir / "checkpoint_weights.o"
        if reuse_prepacked_object is None:
            _run([str(cc), "-march=rv64gc", "-mabi=lp64d", "-c", str(assembly),
                  "-o", str(obj)], label="prepacked checkpoint assembly")
        else:
            from merlin.llvmlower.constant_object import (
                ConstantBlob,
                reuse_constant_object,
            )

            # The fresh selected assembler supplies target metadata. Merlin
            # checks every data byte, symbol, alignment and padding; a stale
            # source manifest cannot authorize sharing a different object.
            reference = out_dir / "constant_object_reference.S"
            reference.write_text(".section .rodata\n.balign 64\n"
                                 ".global merlin_constant_probe\nmerlin_constant_probe:\n"
                                 "  .byte 0\n.balign 64\n", encoding="utf-8")
            reference_object = reference.with_suffix(".o")
            _run([str(cc), "-march=rv64gc", "-mabi=lp64d", "-c", str(reference),
                  "-o", str(reference_object)], label="constant object target reference")
            packed_object_storage_reuse = reuse_constant_object(
                reuse_prepacked_object, obj,
                tuple(ConstantBlob(symbol, path, 64) for symbol, path in fixed_data),
                reference_object=reference_object, section=".rodata")
            (out_dir / "constant_object_reuse.json").write_text(
                json.dumps(packed_object_storage_reuse, indent=2) + "\n")
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
            any(defined.count(name) != 1 for entry, (kernel, _) in prepared_entries.items()
                for name in (entry, kernel, "_mlir_ciface_" + entry)) or
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
        "packed_object_storage_reuse": packed_object_storage_reuse,
        "packed_data_storage_reuse": ({
            "source_directory": str(reuse_prepacked_data.resolve()),
            "verification": "fresh_quantization_full_byte_hash_then_hardlink",
            "files": len(fixed_data),
            "shared_bytes": sum(path.stat().st_size for _, path in fixed_data),
        } if reuse_prepacked_data is not None else None),
        "kernels": kernel_records,
        "shared_activation_packing": "experimental_explicit_tensor" if share_activation_packing else "disabled",
        "matrix_vpu_transport": "experimental_private_bf16" if matrix_vpu_transport else "disabled",
        "vpu_matrix_transport": "experimental_private_bf16" if vpu_matrix_transport else "disabled",
        "virtual_zero_padding": virtual_zero_padding,
        "integer_group_packing": integer_group_packing,
        **({"batch_rhs_reuse": "experimental_exact_contiguous_repetition"} if reuse_batch_rhs else {}),
        **({"compact_batch_rhs": True} if compact_batch_rhs else {}),
        **({"batch_rhs_source": {"file": "batch_rhs_source.mlir", "sha256": plan["source_sha256"]}}
           if reuse_batch_rhs else {}),
        "native_tile_policy": native_tile_policy,
        "connect_sigmoid_multiplies": connect_sigmoid_multiplies,
        "connect_broadcast_multiplies": connect_broadcast_multiplies,
        "prepare_broadcast_operands": prepare_broadcast_operands,
        "broadcast_preparation_budget_bytes": broadcast_preparation_budget_bytes,
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
            "connected_sigmoid_multiplies": connected_sigmoid_multiplies,
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
            "source_flags": source_flags,
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
                                connect_multiplies: bool = False,
                                connect_broadcast_multiplies: bool = False,
                                prepare_broadcast_operands: bool = False,
                                broadcast_preparation_budget_bytes: int = 1048576,
                                connect_sigmoid_multiplies: bool = False,
                                share_activation_packing: bool = False,
                                retain_shared_activation_rhs: bool = False,
                                matrix_vpu_transport: bool = False,
                                vpu_matrix_transport: bool = False,
                                virtual_zero_padding: bool = False,
                                integer_group_packing: bool = False,
                                reuse_batch_rhs: bool = False,
                                compact_batch_rhs: bool = False,
                                native_tile_policy: str = "legacy",
                                reuse_prepacked_data: Path | None = None,
                                reuse_prepacked_object: Path | None = None
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
                                    connect_multiplies=connect_multiplies,
                                    connect_broadcast_multiplies=connect_broadcast_multiplies,
                                    prepare_broadcast_operands=prepare_broadcast_operands,
                                    broadcast_preparation_budget_bytes=broadcast_preparation_budget_bytes,
                                    connect_sigmoid_multiplies=connect_sigmoid_multiplies,
                                    share_activation_packing=share_activation_packing,
                                    retain_shared_activation_rhs=retain_shared_activation_rhs,
                                    matrix_vpu_transport=matrix_vpu_transport,
                                    vpu_matrix_transport=vpu_matrix_transport,
                                    virtual_zero_padding=virtual_zero_padding,
                                    integer_group_packing=integer_group_packing,
                                    reuse_batch_rhs=reuse_batch_rhs,
                                    compact_batch_rhs=compact_batch_rhs,
                                    native_tile_policy=native_tile_policy,
                                    reuse_prepacked_data=reuse_prepacked_data,
                                    reuse_prepacked_object=reuse_prepacked_object)

    return build
