"""Link receipt-bound MX stage builds into one Merlin multi-program Spike ELF.

Merlin generates the session ABI and state handoff. This target tool supplies
MX stage objects, a shared weight blob, and the pinned RV64 runtime. It refuses
stages with different nonempty weight layouts; silently sharing their pointers
would corrupt inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _run(argv: list[str | Path], *, cwd: Path | None = None) -> None:
    subprocess.run([str(part) for part in argv], check=True, cwd=cwd)


def _stage_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name.isidentifier() or name in result or not path:
            raise ValueError(f"invalid or duplicate --stage-build {value!r}; use name=directory")
        result[name] = Path(path).resolve(strict=True)
    return result


def _defined(nm: Path, object_path: Path) -> set[str]:
    result = subprocess.run(
        [str(nm), "--extern-only", "--defined-only", "--format=posix", str(object_path)],
        check=True, capture_output=True, text=True,
    )
    return {row.split()[0] for row in result.stdout.splitlines() if len(row.split()) >= 2}


def _symbol_address(nm: Path, elf: Path, symbol: str) -> int:
    result = subprocess.run(
        [str(nm), "--defined-only", "--format=posix", str(elf)],
        check=True, capture_output=True, text=True,
    )
    addresses = [int(parts[2], 16) for row in result.stdout.splitlines()
                 if (parts := row.split()) and parts[0] == symbol and len(parts) >= 3]
    if len(addresses) != 1:
        raise ValueError(f"linked session lacks one {symbol} symbol")
    return addresses[0]


def _stage_objects(build: Path) -> list[Path]:
    model = build / "model.o"
    if not model.is_file():
        raise ValueError(f"stage build lacks {model}")
    objects = [model]
    for relative in ("device/device_catalog_shim.o", "device_catalog/mx_catalog.o"):
        candidate = build / relative
        if candidate.is_file():
            objects.append(candidate)
    if len(objects) == 2:
        raise ValueError(f"stage build {build} has an incomplete MX shim/catalog pair")
    return objects


def _namespace_stage(
    *, index: int, name: str, objects: list[Path], previous: set[str],
    nm: Path, objcopy: Path, output: Path,
) -> tuple[list[Path], set[str]]:
    symbols = set().union(*(_defined(nm, path) for path in objects))
    if "forward" not in symbols or "_mlir_ciface_forward" not in symbols:
        raise ValueError(f"stage {name} has no compiled forward ABI")
    collisions = symbols & previous
    collisions.update(("forward", "_mlir_ciface_forward"))
    renamed = {
        symbol: (f"merlin_stage_{index}_{name}" if symbol == "forward" else
                 f"_mlir_ciface_merlin_stage_{index}_{name}" if symbol == "_mlir_ciface_forward" else
                 f"merlin_stage_{index}_{symbol}")
        for symbol in sorted(collisions)
    }
    mapping = output / f"stage_{index}_symbols.txt"
    mapping.write_text("".join(f"{old} {new}\n" for old, new in renamed.items()),
                       encoding="utf-8")
    result = []
    for object_index, path in enumerate(objects):
        destination = output / f"stage_{index}_{object_index}.o"
        _run([objcopy, "--redefine-syms", mapping, path, destination])
        result.append(destination)
    renamed_symbols = {renamed.get(symbol, symbol) for symbol in symbols}
    if renamed_symbols & previous:
        raise ValueError(f"stage {name} retains duplicate global symbols")
    return result, renamed_symbols


def _session_main(output_counts: list[int], *, weights_base: int,
                  dump_outputs: tuple[tuple[int, int], ...] = ()) -> str:
    declarations = "\n".join(
        f"void *merlin_stage_{index}_output(int);\n"
        f"size_t merlin_stage_{index}_output_bytes(int);"
        for index in range(len(output_counts))
    )
    pointers = "\n".join(
        f"    case {index}: return merlin_stage_{index}_output(output);"
        for index in range(len(output_counts))
    )
    sizes = "\n".join(
        f"    case {index}: return merlin_stage_{index}_output_bytes(output);"
        for index in range(len(output_counts))
    )
    counts = ", ".join(str(value) for value in output_counts)
    dump_condition = " || ".join(
        f"(program == {program} && output == {output})"
        for program, output in dump_outputs) or "0"
    return f'''/* MX target's Spike entry; state scheduling is Merlin-generated. */
#include <stddef.h>
#include <stdint.h>
#include "merlin_session.h"
#include "output_sha256.h"
void console_init(void);
void htif_puts(const char *);
void htif_putd(long);
void htif_puthex(unsigned long long);
void htif_putc(char);
void htif_exit(int);
void merlin_arena_reset(void);
unsigned long long merlin_memref_rank_mismatches(void);
{declarations}
static uint64_t read_cycle(void) {{
  uint64_t value;
  __asm__ volatile ("rdcycle %0" : "=r" (value));
  return value;
}}
static uint64_t read_instret(void) {{
  uint64_t value;
  __asm__ volatile ("rdinstret %0" : "=r" (value));
  return value;
}}
static const int OUTPUT_COUNTS[MERLIN_SESSION_N_PROGRAMS] = {{{counts}}};
static void *output_pointer(int program, int output) {{
  switch (program) {{
{pointers}
    default: return 0;
  }}
}}
static size_t output_bytes(int program, int output) {{
  switch (program) {{
{sizes}
    default: return 0;
  }}
}}
static void report(int program, long step, int output, const void *value, size_t bytes) {{
  static const char hex[] = "0123456789abcdef";
  uint8_t hash[32];
  merlin_output_sha256_raw(value, bytes, hash);
  htif_puts("SESSION_SHA "); htif_putd(program); htif_putc(' ');
  htif_putd(step); htif_putc(' '); htif_putd(output); htif_putc(' ');
  htif_putd((long)bytes); htif_putc(' ');
  for (int i = 0; i < 32; ++i) {{
    htif_putc(hex[hash[i] >> 4]); htif_putc(hex[hash[i] & 15]);
  }}
  htif_putc('\\n');
}}
static void dump_output(int program, long step, int output,
                        const void *value, size_t bytes) {{
  static const char hex[] = "0123456789abcdef";
  const uint8_t *data = (const uint8_t *)value;
  htif_puts("SESSION_HEX "); htif_putd(program); htif_putc(' ');
  htif_putd(step); htif_putc(' '); htif_putd(output); htif_putc(' ');
  htif_putd((long)bytes); htif_putc(' ');
  for (size_t i = 0; i < bytes; ++i) {{
    htif_putc(hex[data[i] >> 4]); htif_putc(hex[data[i] & 15]);
  }}
  htif_putc('\\n');
}}
int main(int hart) {{
  if (hart != 0) for (;;) ;
  console_init();
  const void *weights = (const void *){hex(weights_base)}ULL;
  merlin_session_reset();
  uint64_t total_cycles = 0, total_instructions = 0;
  for (int program = 0; program < MERLIN_SESSION_N_PROGRAMS; ++program) {{
    if (merlin_session_prepare_program(program) != 0) {{
      htif_puts("FAIL session binding\\n"); htif_exit(1); return 1;
    }}
    long steps = merlin_session_program_steps(program);
    for (long step = 0; step < steps; ++step) {{
      htif_puts("SESSION_BEGIN "); htif_putd(program); htif_putc(' ');
      htif_putd(step); htif_putc('\\n');
      uint64_t start_cycles = read_cycle();
      uint64_t start_instructions = read_instret();
      merlin_arena_reset();
      if (merlin_session_run_step(program, weights, step, 0) != 0) {{
        htif_puts("FAIL session state\\n"); htif_exit(1); return 1;
      }}
      uint64_t instructions = read_instret() - start_instructions;
      uint64_t cycles = read_cycle() - start_cycles;
      total_cycles += cycles;
      total_instructions += instructions;
      htif_puts("SESSION_PERF "); htif_putd(program); htif_putc(' ');
      htif_putd(step); htif_putc(' '); htif_putd((long)cycles);
      htif_putc(' '); htif_putd((long)instructions); htif_putc('\\n');
      for (int output = 0; output < OUTPUT_COUNTS[program]; ++output) {{
        void *value = output_pointer(program, output);
        size_t bytes = output_bytes(program, output);
        if (!value || !bytes) {{
          htif_puts("FAIL session output ABI program="); htif_putd(program);
          htif_puts(" step="); htif_putd(step);
          htif_puts(" output="); htif_putd(output);
          htif_puts(" pointer="); htif_puthex((unsigned long long)(uintptr_t)value);
          htif_puts(" bytes="); htif_putd((long)bytes);
          htif_putc('\\n'); htif_exit(1); return 1;
        }}
        report(program, step, output, value, bytes);
        if ({dump_condition}) dump_output(program, step, output, value, bytes);
      }}
    }}
  }}
  htif_puts("SESSION_TOTAL "); htif_putd((long)total_cycles);
  htif_putc(' '); htif_putd((long)total_instructions); htif_putc('\\n');
  htif_puts("METRIC memref_rank_mismatch ");
  htif_putd((long)merlin_memref_rank_mismatches()); htif_putc('\\n');
  htif_puts("DONE\\n"); htif_exit(0); return 0;
}}
'''


def link_session(
    session_root: Path, builds: dict[str, Path], output: Path, *,
    profile: Path, merlin_src: Path, riscv_root: Path,
    dump_outputs: tuple[str, ...] = (),
) -> dict:
    sys.path.insert(0, str(merlin_src))
    from merlin.llvmlower import session_bundle
    from merlin.runtime.backends.spike_arena import (
        estimate_llvm_malloc, recommended_arena_bytes, spike_session_layout,
    )
    from mx_gemmini_support.target_profile import load_profile, profile_sha256

    session = session_bundle.load(session_root)
    names = set(session.program_names)
    if set(builds) != names:
        raise ValueError(f"stage builds must be exactly {sorted(names)}")
    selected_dumps = []
    seen_dumps = set()
    output_sizes = {"f32": 4, "i64": 8, "i1": 1, "bf16": 2, "f16": 2, "i32": 4}
    for request in dump_outputs:
        name, separator, raw_index = request.partition(":")
        if not separator or name not in names or not raw_index.isdecimal():
            raise ValueError(f"invalid --dump-output {request!r}; use program:output-index")
        program_index = session.program_names.index(name)
        program = session.programs[program_index]
        output_index = int(raw_index)
        if output_index >= len(program.outputs) or (program_index, output_index) in seen_dumps:
            raise ValueError(f"unknown or duplicate --dump-output {request!r}")
        shape, dtype = program.outputs[output_index]
        byte_count = output_sizes[dtype]
        for dimension in shape:
            byte_count *= dimension
        if byte_count > 16_384:
            raise ValueError(f"dump output {request!r} exceeds the 16 KiB console limit")
        seen_dumps.add((program_index, output_index))
        selected_dumps.append({"program": program_index, "name": name,
                               "output": output_index, "bytes": byte_count})
    selected_dumps.sort(key=lambda row: (row["program"], row["output"]))
    expected_profile = profile_sha256(load_profile(profile))
    receipts = {}
    source_objects = {}
    for program in session.programs:
        build = builds[program.name]
        receipt = json.loads((build / "mx_stage_probe.json").read_text(encoding="utf-8"))
        if (receipt.get("schema") != "mx_gemmini.merlin_stage_probe.v1"
                or receipt.get("profile_sha256") != expected_profile
                or receipt.get("capture_model_sha256") != _sha(program.bundle / "model.mlir")
                or not str(receipt.get("status", "")).startswith("linked_")
                or receipt.get("diagnostic_host_vpu") is True):
            raise ValueError(f"stage {program.name} build receipt differs from the session capture/profile")
        objects = _stage_objects(build)
        if len(objects) != (3 if receipt.get("selected_operations", 0) else 1):
            raise ValueError(f"stage {program.name} device objects differ from its selection count")
        catalog = build / "device_catalog/mx_catalog.json"
        if len(objects) == 3:
            details = json.loads(catalog.read_text(encoding="utf-8"))
            if (details.get("coverage_complete") is not True
                    or details.get("covered_contractions") != receipt["selected_operations"]
                    or details.get("unbound_precision_proposals")):
                raise ValueError(f"stage {program.name} catalog has incomplete MX coverage")
        source_objects[program.name] = {
            str(path.relative_to(build)): _sha(path) for path in objects}
        receipts[program.name] = receipt
    if output.exists():
        raise ValueError("session output directory must not already exist")
    output.mkdir(parents=True)
    generated = session_bundle.generate(
        session_root, output / "generated", materialize_weights=False)
    shared_weights = session_bundle.materialize_shared_weights(
        session, generated, output / "shared_weights")
    weight_name = shared_weights["program"]
    estimates = {program.name: estimate_llvm_malloc(
        builds[program.name] / "lower/model.ll") for program in session.programs}
    arena_bytes = max(recommended_arena_bytes(value) for value in estimates.values())
    weight_bytes = shared_weights["bytes"]
    layout = spike_session_layout(weights_bytes=weight_bytes, arena_bytes=arena_bytes)
    bin_dir = riscv_root / "bin"
    cc, nm, objcopy, ld = (bin_dir / name for name in (
        "riscv64-unknown-elf-gcc", "riscv64-unknown-elf-nm",
        "riscv64-unknown-elf-objcopy", "riscv64-unknown-elf-ld"))
    if not all(path.is_file() for path in (cc, nm, objcopy, ld)):
        raise ValueError("session link needs explicit RV64 gcc, nm, objcopy, and ld")
    runtime_c = merlin_src.parent / "merlin/runtime/c"
    bare = merlin_src.parent / "merlin/runtime/baremetal/spike"
    flags = ["-O2", "-ffreestanding", "-fno-builtin", "-march=rv64gc",
             "-mabi=lp64d", "-mcmodel=medany"]

    def compile_c(source: Path, object_path: Path, *includes: Path) -> Path:
        include_flags = [part for directory in includes for part in ("-I", str(directory))]
        _run([cc, *flags, *include_flags, "-c", source, "-o", object_path])
        return object_path

    all_objects: list[Path] = []
    previous: set[str] = set()
    for index, program in enumerate(session.programs):
        stage_objects = _stage_objects(builds[program.name])
        if index:
            stage_objects, symbols = _namespace_stage(
                index=index, name=program.name, objects=stage_objects,
                previous=previous, nm=nm, objcopy=objcopy, output=output)
        else:
            symbols = set().union(*(_defined(nm, path) for path in stage_objects))
        previous.update(symbols)
        all_objects.extend(stage_objects)
    first = builds[session.programs[0].name]
    all_objects.extend(first / f"{name}.o" for name in (
        "model_call", "merlin_model", "mlir_rt", "crt", "console", "libc_min"))
    malloc_source = bare / "merlin_malloc.c"
    malloc_object = output / "session_malloc.o"
    all_objects.append(malloc_object)
    weight_object = output / "weights_blob.o"
    _run([ld, "-r", "-b", "binary", "-o", weight_object, "weights.bin"],
         cwd=Path(shared_weights["path"]).parent)
    all_objects.append(weight_object)
    generated_root = output / "generated"
    for index, program in enumerate(session.programs):
        stage_dir = generated_root / f"stage_{index}_{program.name}"
        all_objects.append(compile_c(
            stage_dir / "stage_adapter.c", output / f"adapter_{index}.o", runtime_c, stage_dir))
        if index:
            all_objects.append(compile_c(
                stage_dir / "model_call.c", output / f"call_{index}.o", runtime_c, stage_dir))
    all_objects.append(compile_c(
        generated_root / "merlin_session.c", output / "session.o", generated_root))
    main_source = output / "session_main.c"
    main_source.write_text(
        _session_main([len(program.outputs) for program in session.programs],
                      weights_base=layout["weights_base"],
                      dump_outputs=tuple((row["program"], row["output"])
                                         for row in selected_dumps)),
        encoding="utf-8")
    all_objects.append(compile_c(main_source, output / "session_main.o", generated_root, bare))
    elf = output / "session.elf"
    libm = riscv_root / "riscv64-unknown-elf/lib/libm.a"
    for attempt in range(3):
        _run([cc, *flags,
              f"-DMERLIN_ARENA_BASE_ADDR={hex(layout['arena_base'])}ULL",
              f"-DMERLIN_ARENA_SIZE_BYTES={hex(layout['arena_bytes'])}ULL",
              f"-DMERLIN_WEIGHTS_BASE_ADDR={hex(layout['weights_base'])}ULL",
              "-c", malloc_source, "-o", malloc_object])
        _run([cc, *flags, "-nostdlib", "-nostartfiles",
              f"-Wl,--defsym,MERLIN_WEIGHTS_BASE={hex(layout['weights_base'])}",
              "-Wl,--defsym,MERLIN_STACK_BYTES=0x40000",
              "-T", bare / "model_link.ld", *all_objects, libm, "-o", elf])
        static_end = _symbol_address(nm, elf, "_stack_top")
        checked = spike_session_layout(weights_bytes=weight_bytes,
                                       arena_bytes=arena_bytes,
                                       static_end=static_end)
        if checked["arena_base"] == layout["arena_base"]:
            layout = checked
            break
        layout = checked
    else:
        raise ValueError("linked static extent did not stabilize below the arena")
    record = {
        "schema": "mx_gemmini.linked_merlin_session.v1",
        "performance_schema": "spike_rdcycle_rdinstret_per_step.v1",
        "dump_outputs": selected_dumps,
        "status": "linked_experimental_numeric_qualification_pending",
        "session_root": str(session_root),
        "programs": [{"name": program.name, "steps": program.steps,
                      "outputs": len(program.outputs),
                      "capture_model_sha256": receipts[program.name]["capture_model_sha256"],
                      "selected_operations": receipts[program.name]["selected_operations"]}
                     for program in session.programs],
        "bindings": [binding.__dict__ for binding in session.bindings],
        "profile_sha256": expected_profile,
        "shared_weight_stage": weight_name,
        "shared_weight_bytes": weight_bytes,
        "shared_weight_sha256": shared_weights["sha256"],
        "stage_object_sha256": source_objects,
        "arena_layout": layout,
        "static_malloc_estimates": {
            name: {"callsites": value.callsites, "literal_bytes": value.literal_bytes,
                   "dynamic_sites": value.dynamic_sites}
            for name, value in estimates.items()},
        "elf": str(elf), "elf_bytes": elf.stat().st_size,
        "elf_sha256": _sha(elf),
        "mem_bytes": max(layout["mem_bytes"],
                         *(receipt["mem_bytes"] for receipt in receipts.values())),
    }
    (output / "mx_session_link.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", required=True, type=Path)
    parser.add_argument("--stage-build", action="append", required=True, default=[])
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--merlin-src", required=True, type=Path)
    parser.add_argument("--riscv-root", required=True, type=Path)
    parser.add_argument("--dump-output", action="append", default=[],
                        help="program:output-index; record at most 16 KiB per selected result")
    args = parser.parse_args()
    record = link_session(
        args.session_root.resolve(strict=True), _stage_paths(args.stage_build),
        args.workdir.absolute(), profile=args.profile.resolve(strict=True),
        merlin_src=args.merlin_src.resolve(strict=True),
        riscv_root=args.riscv_root.resolve(strict=True),
        dump_outputs=tuple(args.dump_output))
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
