"""Relink an MX session with replacement catalogs or bounded output dumps.

The original host, shim, VPU, state adapters, and weight blob remain the same.
The selected catalog objects or session entry change without materializing
another multi-gigabyte weight blob.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .link_merlin_session import (
    DEFAULT_DUMP_LIMIT_BYTES, _selected_output_dumps, _session_main,
    _symbol_address,
)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def named_paths(values: list[str], *, label: str) -> dict[str, Path]:
    result = {}
    for value in values:
        name, sep, raw_path = value.partition("=")
        if not sep or not name.isidentifier() or name in result or not raw_path:
            raise ValueError(f"invalid or duplicate {label} {value!r}; use name=path")
        result[name] = Path(raw_path).resolve(strict=True)
    return result


def relink(base_link: Path, builds: dict[str, Path],
           replacements: dict[str, Path], output: Path, *,
           merlin_src: Path, riscv_root: Path,
           dump_outputs: tuple[str, ...] = (),
           dump_limit_bytes: int = DEFAULT_DUMP_LIMIT_BYTES) -> dict:
    base = json.loads(base_link.read_text())
    programs = [row["name"] for row in base["programs"]]
    if (base.get("schema") != "mx_gemmini.linked_merlin_session.v1"
            or set(builds) != set(programs)
            or not (replacements or dump_outputs)
            or not set(replacements) <= set(programs)):
        raise ValueError("base session, stage builds, and replacements do not agree")
    base_dir = base_link.parent
    base_elf = Path(base["elf"])
    if sha(base_elf) != base["elf_sha256"]:
        raise ValueError("base ELF differs from its link receipt")
    for name, build in builds.items():
        stage = json.loads((build / "mx_stage_probe.json").read_text())
        if (stage.get("profile_sha256") != base["profile_sha256"]
                or stage.get("capture_model_sha256") != next(
                    row["capture_model_sha256"] for row in base["programs"]
                    if row["name"] == name)):
            raise ValueError(f"stage {name} differs from the base session")
        for relative, expected in base["stage_object_sha256"][name].items():
            if sha(build / relative) != expected:
                raise ValueError(f"stage {name} object {relative} differs from base")

    selected_dumps = None
    if dump_outputs:
        sys.path.insert(0, str(merlin_src))
        from merlin.llvmlower import session_bundle
        session = session_bundle.load(Path(base["session_root"]))
        if session.program_names != tuple(programs):
            raise ValueError("base session programs differ from capture")
        selected_dumps = _selected_output_dumps(
            session, dump_outputs, dump_limit_bytes)

    if output.exists():
        raise ValueError("output directory must not already exist")
    output.mkdir(parents=True)
    bin_dir = riscv_root / "bin"
    cc = bin_dir / "riscv64-unknown-elf-gcc"
    objcopy = bin_dir / "riscv64-unknown-elf-objcopy"
    nm = bin_dir / "riscv64-unknown-elf-nm"
    if not cc.is_file() or not objcopy.is_file() or not nm.is_file():
        raise ValueError("relink needs RV64 GCC, objcopy, and nm")

    def defined(path: Path) -> set[str]:
        result = subprocess.run(
            [str(nm), "--extern-only", "--defined-only", "--format=posix", str(path)],
            check=True, capture_output=True, text=True)
        return {line.split()[0] for line in result.stdout.splitlines()
                if len(line.split()) >= 2}

    objects = []
    changed = {}
    previous_symbols: set[str] = set()
    record = json.loads(json.dumps(base))
    for index, name in enumerate(programs):
        build = builds[name]
        model = build / "model.o"
        shim = build / "device/device_catalog_shim.o"
        catalog = build / "device_catalog/mx_catalog.o"
        if index:
            stage_objects = [base_dir / f"stage_{index}_{part}.o"
                             for part in range(3 if shim.is_file() else 1)]
        else:
            stage_objects = [model, shim, catalog] if shim.is_file() else [model]
        if name in replacements:
            source_dir = replacements[name]
            source = source_dir / "mx_catalog_nicolas.o"
            receipt_path = source_dir / "receipt.json"
            receipt = json.loads(receipt_path.read_text())
            original_digest = base["stage_object_sha256"][name][
                "device_catalog/mx_catalog.o"]
            if (not shim.is_file() or not source.is_file()
                    or sha(catalog) != original_digest
                    or receipt.get("original_catalog_object_sha256") != original_digest
                    or receipt.get("replacement_catalog_object_sha256") != sha(source)
                    or receipt.get("stage_receipt_sha256") != sha(build / "mx_stage_probe.json")):
                raise ValueError(f"replacement {name} is not bound to this stage")
            if index:
                target = output / f"stage_{index}_catalog.o"
                base_mapping = base_dir / f"stage_{index}_symbols.txt"
                mapping = output / f"stage_{index}_symbols.txt"
                original_map = dict(line.split() for line in base_mapping.read_text().splitlines())
                extra = defined(source) & previous_symbols - set(original_map)
                for symbol in sorted(extra):
                    original_map[symbol] = f"merlin_stage_{index}_{symbol}"
                mapping.write_text("".join(
                    f"{old} {new}\n" for old, new in sorted(original_map.items())))
                subprocess.run([str(objcopy), "--redefine-syms",
                                str(mapping),
                                str(source), str(target)], check=True)
            else:
                target = source
            stage_objects[-1] = target
            record["stage_object_sha256"][name][
                "device_catalog/mx_catalog.o"] = sha(source)
            changed[name] = {"receipt_sha256": sha(receipt_path),
                             "source_revision": receipt["source_revision"],
                             "catalog_sha256": sha(source)}
        if any(not path.is_file() for path in stage_objects):
            raise ValueError(f"missing stage objects for {name}")
        symbols = set().union(*(defined(path) for path in stage_objects))
        if symbols & previous_symbols:
            raise ValueError(f"stage {name} has unrenamed global collisions")
        previous_symbols.update(symbols)
        objects.extend(stage_objects)
    first = builds[programs[0]]
    objects.extend(first / f"{name}.o" for name in (
        "model_call", "merlin_model", "mlir_rt", "crt", "console", "libc_min"))
    objects.extend([base_dir / "session_malloc.o", base_dir / "weights_blob.o"])
    for index in range(len(programs)):
        objects.append(base_dir / f"adapter_{index}.o")
        if index:
            objects.append(base_dir / f"call_{index}.o")
    objects.append(base_dir / "session.o")
    if selected_dumps is None:
        objects.append(base_dir / "session_main.o")
    else:
        generated = (base_dir / "session.o").resolve().parent / "generated"
        if not (generated / "merlin_session.h").is_file():
            raise ValueError("base session lacks its generated ABI header")
        source = output / "session_main.c"
        source.write_text(_session_main(
            [len(program.outputs) for program in session.programs],
            weights_base=base["arena_layout"]["weights_base"],
            dump_outputs=tuple((row["program"], row["output"])
                               for row in selected_dumps)))
        main_object = output / "session_main.o"
        flags = ["-O2", "-ffreestanding", "-fno-builtin", "-march=rv64gc",
                 "-mabi=lp64d", "-mcmodel=medany"]
        bare = merlin_src.parent / "merlin/runtime/baremetal/spike"
        subprocess.run([str(cc), *flags, "-I", str(generated), "-I", str(bare),
                        "-c", str(source), "-o", str(main_object)], check=True)
        objects.append(main_object)
    if any(not path.is_file() for path in objects):
        raise ValueError("base session objects are incomplete")
    bare = merlin_src.parent / "merlin/runtime/baremetal/spike"
    libm = riscv_root / "riscv64-unknown-elf/lib/libm.a"
    elf = output / "session.elf"
    layout = base["arena_layout"]
    command = [str(cc), "-O2", "-ffreestanding", "-fno-builtin", "-march=rv64gc",
               "-mabi=lp64d", "-mcmodel=medany", "-nostdlib", "-nostartfiles",
               f"-Wl,--defsym,MERLIN_WEIGHTS_BASE={hex(layout['weights_base'])}",
               "-Wl,--defsym,MERLIN_STACK_BYTES=0x40000", "-T",
               str(bare / "model_link.ld"), *(str(path) for path in objects),
               str(libm), "-o", str(elf)]
    with (output / "link.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    static_end = _symbol_address(nm, elf, "_stack_top")
    if static_end > layout["arena_base"]:
        raise ValueError("replacement catalog extends into the session arena")
    record["arena_layout"]["static_end"] = static_end
    record["elf"] = str(elf)
    record["elf_bytes"] = elf.stat().st_size
    record["elf_sha256"] = sha(elf)
    record["catalog_replacements"] = changed
    if selected_dumps is not None:
        record["dump_outputs"] = selected_dumps
        record["dump_limit_bytes"] = dump_limit_bytes
    record["relinked_from_sha256"] = sha(base_link)
    (output / "mx_session_link.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-link", type=Path, required=True)
    parser.add_argument("--stage-build", action="append", required=True)
    parser.add_argument("--replacement", action="append", default=[])
    parser.add_argument("--dump-output", action="append", default=[])
    parser.add_argument("--dump-limit-bytes", type=int,
                        default=DEFAULT_DUMP_LIMIT_BYTES)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--merlin-src", type=Path, required=True)
    parser.add_argument("--riscv-root", type=Path, required=True)
    args = parser.parse_args()
    record = relink(args.base_link.resolve(strict=True),
                    named_paths(args.stage_build, label="stage build"),
                    named_paths(args.replacement, label="replacement"),
                    args.out.absolute(), merlin_src=args.merlin_src.resolve(strict=True),
                    riscv_root=args.riscv_root.resolve(strict=True),
                    dump_outputs=tuple(args.dump_output),
                    dump_limit_bytes=args.dump_limit_bytes)
    print(json.dumps({"elf_sha256": record["elf_sha256"],
                      "replaced": sorted(record["catalog_replacements"]),
                      "dump_outputs": record["dump_outputs"]}))


if __name__ == "__main__":
    main()
