"""Prove owned native constant readers before sharing their immutable storage."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from .catalog_toolchain import native_catalog_compile_flags
from .dense_contraction import emit_dense_contraction
from .dense_kernel import emit_dense_f32_kernel
from .target_profile import profile_sha256


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _run(command):
    return subprocess.run(list(map(str, command)), check=True, capture_output=True, text=True).stdout


def _readers(catalog, profile):
    """Regenerate the complete owned uses; unknown emitters grant no sharing."""
    result = {}
    for row in catalog["kernels"]:
        if not row["prepacked_checkpoint_weight"]:
            continue
        if row["batched"] or row["cache_immutable_weight"] or row["issuer_mode"] != "merlin_mk_native_dram_loop":
            raise ValueError("constant pooling requires owned direct readonly native readers")
        symbol = row["symbol"]
        issuer_symbol = Path(row["schedule_file"]).stem
        shape = tuple(row["dimensions"][key] for key in ("m", "n", "k"))
        precision = {"mxfp8": "FP8", "mxfp4": "FP4"}[row["precision"]]
        fixed = symbol + "_fixed_codes", symbol + "_fixed_scales"
        generated = emit_dense_contraction(
            profile, shape, precision=precision, symbol=symbol, issuer_symbol=issuer_symbol,
            shared_workspace_symbol="mx_catalog_workspace", prepacked_weight_symbols=fixed,
            virtual_zero_padding=catalog["virtual_zero_padding"],
            schedule="movement" if catalog["native_tile_policy"] == "movement" else "native")
        candidates = {symbol: generated.kernel.source, issuer_symbol: generated.issuer.source}
        preparation = row.get("activation_preparation")
        retain_rhs = preparation is not None and len(preparation["consumer_arguments"]) == 3
        if preparation is not None:
            name = preparation["consumer_kernel"]
            candidates[name] = emit_dense_f32_kernel(
                generated.issuer, shape, symbol=name, issuer_symbol=issuer_symbol,
                shared_workspace_symbol="mx_catalog_workspace", prepacked_weight_symbols=fixed,
                prepared_activation=True, retain_prepared_rhs=retain_rhs).source
        for field, prepared in [("private_bf16_transport", False), ("prepared_private_bf16_transport", True)]:
            if field not in row:
                continue
            name = row[field]["kernel_symbol"]
            candidates[name] = emit_dense_f32_kernel(
                generated.issuer, shape, symbol=name, issuer_symbol=issuer_symbol,
                shared_workspace_symbol="mx_catalog_workspace", prepacked_weight_symbols=fixed,
                output_dtype="bf16", prepared_activation=prepared, retain_prepared_rhs=retain_rhs if prepared else False,
                virtual_zero_padding=catalog["virtual_zero_padding"]).source
        for name, source in candidates.items():
            if name in result and result[name] != source:
                raise ValueError("constant pool reader has conflicting owned definitions")
            result[name] = source
    return result


def prepare_shared_catalog_constants(builds, *, profile, output, riscv_root, readelf):
    """Return checked code-only catalogs, a shared pool and explicit symbol maps.

    Full original catalogs are first reconstructed byte-for-byte. Every object
    referencing selected constants is freshly regenerated and recompiled from
    the owned readonly/address-independent templates, with exact original object
    equality. Constants have no CPU address observations: native issuers use
    their nonnull aligned addresses solely for synchronous readonly DMA. Unknown
    references, changed source, objects, bytes or format contracts refuse.
    """
    from merlin.llvmlower import immutable_constant_pool
    from merlin.llvmlower.artifact_storage import share_identical_artifact
    from merlin.llvmlower.constant_object import ConstantBlob, reuse_constant_object
    from merlin.llvmlower.immutable_constant_pool import (
        ACCESS,
        PoolOperand,
        plan_immutable_constant_pool,
    )

    from . import (
        catalog_toolchain,
        dense_contraction,
        dense_kernel,
        scheduled_native_issuer,
    )

    tools = [Path(riscv_root) / "bin/riscv64-unknown-elf-gcc",
             Path(riscv_root) / "bin/riscv64-unknown-elf-ld",
             Path(riscv_root) / "bin/riscv64-unknown-elf-nm", Path(readelf)]
    producers = [Path(__file__), *(Path(module.__file__) for module in
                 (immutable_constant_pool, catalog_toolchain, dense_contraction, dense_kernel, scheduled_native_issuer)),
                 Path(__file__).resolve().parent / "runtime_quant.h"]
    pins = {str(path): _sha(path) for path in tools + producers}
    commands = []

    def run(command):
        observed = _run(command)
        commands.append({"argv": list(map(str, command)), "exit_code": 0,
                         "stdout_sha256": hashlib.sha256(observed.encode()).hexdigest()})
        return observed

    output = Path(output)
    output.mkdir(exist_ok=False)
    bins = Path(riscv_root) / "bin"
    cc, ld, nm = (bins / name for name in ("riscv64-unknown-elf-gcc", "riscv64-unknown-elf-ld", "riscv64-unknown-elf-nm"))
    operands, code_objects, observations = [], {}, {}
    for group, build in builds.items():
        if not group.isascii() or not group.isidentifier():
            raise ValueError("constant pool group needs a safe program identifier")
        directory = output / group
        directory.mkdir()
        source = Path(build) / "device_catalog"
        catalog = json.loads((source / "mx_catalog.json").read_text())
        original = source / "mx_catalog.o"
        if (catalog["profile_sha256"] != profile_sha256(profile)
                or catalog["compilation"]["compiler"] != str(cc)
                or _sha(original) != catalog["compilation"]["object_sha256"]):
            raise ValueError("constant pooling catalog, profile or compiler differs")
        listing = run([readelf, "-sW", original])
        names = [row[7] for line in listing.splitlines()
                 if len(row := line.split(maxsplit=7)) == 8 and row[3] == "FILE" and row[6] == "ABS"]
        if (not names or len(names) != len(set(names)) or
                any(not name.endswith(".c") or not name[:-2].isascii() or not name[:-2].isidentifier() for name in names)):
            raise ValueError("constant pooling cannot close original component ownership")
        components = [source / (name[:-2] + ".o") for name in names]
        constant_object = source / "checkpoint_weights.o"
        before = {str(path): _sha(path) for path in [*components, constant_object]}
        rebuilt = directory / "original.o"
        run([ld, "-r", "-o", rebuilt, *components, constant_object])
        identity = share_identical_artifact(original, rebuilt)
        if not identity["shared_inode"]:
            raise ValueError("constant pool components do not reconstruct the complete original catalog")
        expected = _readers(catalog, profile)
        # Fresh source/object equality binds the semantic template to actual machine inputs.
        for header in ("runtime_quant.h",):
            if _sha(source / header) != _sha(Path(__file__).resolve().parent / header):
                raise ValueError("constant pool reader header differs from the owned ABI")
            (directory / header).write_bytes((source / header).read_bytes())
        proofs = {}
        for name, text in expected.items():
            path, obj = source / (name + ".c"), source / (name + ".o")
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest != catalog["compilation"]["source_sha256"][name] or _sha(path) != digest or obj not in components:
                raise ValueError("constant pool owned reader source differs from compiled catalog")
            fresh = directory / path.name
            fresh.write_text(text)
            fresh_object = fresh.with_suffix(".o")
            flags = native_catalog_compile_flags()
            run([cc, *flags, "-c", fresh, "-o", fresh_object])
            if _sha(fresh_object) != before[str(obj)]:
                raise ValueError("constant pool regenerated reader object differs")
            proofs[name] = {"source_sha256": digest, "object_sha256": before[str(obj)]}
        blobs, fixed = [], set()
        for row in catalog["kernels"]:
            if not row["prepacked_checkpoint_weight"]:
                continue
            for role in ("codes", "scales"):
                name = row["symbol"] + "_fixed_" + role
                path = source / (row["symbol"] + "_" + role + ".bin")
                if _sha(path) != row[role + "_sha256"] or path.stat().st_size != row[role + "_bytes"]:
                    raise ValueError("constant pool original packed input differs")
                representation = json.dumps({"precision": row["precision"], "role": role,
                    "padded_kn": row["padded_shape_mnk"][1:][::-1],
                    "layout": "native_row_major_kn_codes_group_major_e8_scales_v1"}, sort_keys=True)
                operands.append(PoolOperand(group, name, path, 64, representation, ACCESS))
                blobs.append(ConstantBlob(name, path, 64))
                fixed.add(name)
        # Close every actual undefined reference, not just the known reader list.
        external = [Path(build) / name for name in ("model.o", "device/device_catalog_shim.o")]
        undefined = run([nm, "-A", "--undefined-only", "--format=posix", *components,
                         *(path for path in external if path.is_file())])
        observed = set()
        for line in undefined.splitlines():
            path, separator, fields = line.rpartition(": ")
            parts = fields.split()
            if not separator or len(parts) < 2:
                raise ValueError("constant pool received incomplete relocation ownership metadata")
            if parts[0] in fixed:
                owner = Path(path).stem
                if Path(path) not in components or owner not in expected:
                    raise ValueError("constant pool has an unqualified address observer")
                observed.add(parts[0])
        if observed != fixed:
            raise ValueError("constant pool lacks complete selected constant use coverage")
        reference = directory / "reference.S"
        reference.write_text(".section .rodata\n.balign 64\n.global pool_reference\npool_reference:\n.byte 0\n.balign 64\n")
        run([cc, "-march=rv64gc", "-mabi=lp64d", "-c", reference, "-o", reference.with_suffix(".o")])
        constant_proof = reuse_constant_object(constant_object, directory / "constants.o", blobs,
                                               reference_object=reference.with_suffix(".o"), section=".rodata")
        code = directory / "code.o"
        run([ld, "-r", "-o", code, *components])
        if any(_sha(path) != before[str(path)] for path in [*components, constant_object]):
            raise ValueError("constant pool components changed during proof")
        code_objects[group] = code
        observations[group] = {"catalog_sha256": _sha(source / "mx_catalog.json"),
            "original_object_sha256": _sha(original), "component_sha256": before,
            "reader_proofs": proofs, "constant_object": constant_proof, "code_sha256": _sha(code)}
    plan = plan_immutable_constant_pool(operands, symbol_prefix="merlin_shared_constant_", reserved_symbols=())
    if plan.pooled_bytes >= plan.original_bytes:
        raise ValueError("constant pool offers no storage reduction; retain the original route")
    assembly = output / "shared_constants.S"
    lines = [".section .rodata"]
    for blob in plan.blobs:
        path = str(blob.path.resolve())
        if any(char in path for char in ('"', "\\", "\n", "\r")):
            raise ValueError("constant pool assembly input path needs unambiguous quoting")
        lines.extend([f".balign {blob.alignment}", f".global {blob.symbol}", f"{blob.symbol}:", f'.incbin "{path}"'])
    lines.append(f".balign {plan.blobs[-1].alignment}")
    assembly.write_text("\n".join(lines) + "\n")
    pool_object = output / "shared_constants.o"
    run([cc, "-march=rv64gc", "-mabi=lp64d", "-c", assembly, "-o", pool_object])
    plan.recheck()
    pool_proof = reuse_constant_object(
        pool_object, output / "verified_shared_constants.o",
        [ConstantBlob(blob.symbol, blob.path, blob.alignment) for blob in plan.blobs],
        reference_object=reference.with_suffix(".o"), section=".rodata")
    if any(_sha(path) != digest for path, digest in pins.items()):
        raise ValueError("constant pool producer or selected tools changed during proof")
    record = {"schema": "mx_gemmini.shared_catalog_constants.v1", "groups": observations, "producer_and_tool_pins": pins, "commands": commands,
              "pool": plan.receipt(), "pool_object_sha256": _sha(pool_object),
              "pool_object": str(pool_object), "pool_object_proof": pool_proof,
              "scope": "owned_readonly_reader_and_storage_proof_not_final_image_admission_or_performance"}
    (output / "receipt.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return code_objects, pool_object, plan.symbol_maps(), record
