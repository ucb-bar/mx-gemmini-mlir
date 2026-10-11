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

from .dense_contraction import emit_dense_batch_contraction, emit_dense_contraction
from .dense_kernel import emit_shared_workspace_definition
from .merlin_catalog_plan import plan_merlin_catalog
from .merlin_schedule_policy import build_policy_plan, selected_policy
from .target_profile import load_profile, profile_sha256


_WORKSPACE_SYMBOL = "mx_catalog_workspace"
_SUPPORT = Path(__file__).resolve().parent
_QUANTIZER = _SUPPORT / "runtime_quant.c"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], *, label: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"MX {label} failed: {result.stderr[-1200:]}")


def build_merlin_catalog(prepared: Path, out_dir: Path, *, profile: dict,
                         riscv_root: Path) -> tuple[Path, Path]:
    """Compile selected source sites through checked MX native schedules."""
    plan = plan_merlin_catalog(prepared, profile)
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
    for entry in plan["kernels"]:
        symbol = entry["symbol"]
        issuer_symbol = entry["issuer_symbol"]
        shape = tuple(entry["shape"])
        precision = "FP4" if entry["precision"] == "mxfp4" else "FP8"
        policy_text = build_policy_plan(symbol, schedule="native")
        selected = selected_policy(policy_text, symbol=symbol)
        policy_path = out_dir / f"{symbol}.schedule.mlir"
        policy_path.write_text(policy_text, encoding="utf-8")
        if entry["rank"] == 2:
            generated = emit_dense_contraction(
                profile, shape, precision=precision, symbol=symbol,
                issuer_symbol=issuer_symbol,
                shared_workspace_symbol=_WORKSPACE_SYMBOL,
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
        compile_source(issuer_symbol, generated.issuer.source)
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
            "padded_shape_mnk": list(generated.shape_plan.padded_shape_mnk),
            "tile_mnk": list(generated.shape_plan.tile_mnk),
        })
    compile_source("runtime_quant", _QUANTIZER)
    compile_source("shared_workspace", emit_shared_workspace_definition(
        max_workspace, symbol=_WORKSPACE_SYMBOL))
    object_path = out_dir / "mx_catalog.o"
    _run([str(ld), "-r", "-o", str(object_path), *(str(obj) for obj in objects)],
         label="relocatable catalog link")
    symbols = subprocess.run([str(nm), "--format=posix", "--extern-only",
                              str(object_path)], capture_output=True, text=True, check=True)
    rows = [line.split() for line in symbols.stdout.splitlines()]
    defined = [row[0] for row in rows if len(row) >= 2 and row[1].upper() == "T"]
    if (any(len(row) < 2 or row[1].upper() == "U" for row in rows) or
            any(defined.count(entry["symbol"]) != 1 for entry in plan["kernels"])):
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
        "kernels": kernel_records,
        "bindings": plan["bindings"],
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
                                riscv_root: Path) -> Callable[[Path, Path], tuple[Path, Path]]:
    """Return the target-owned callback accepted by DeviceRouting."""
    profile_path = Path(profile_path).resolve(strict=True)
    riscv_root = Path(riscv_root).resolve(strict=True)

    def build(prepared: Path, work: Path) -> tuple[Path, Path]:
        return build_merlin_catalog(prepared, work,
                                    profile=load_profile(profile_path),
                                    riscv_root=riscv_root)

    return build
