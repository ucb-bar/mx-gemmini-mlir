"""Link receipt-bound MX stage builds into one Merlin multi-program Spike ELF.

Merlin generates the session ABI and state handoff. This target tool supplies
MX stage objects, explicit immutable weight storage, and the pinned RV64 runtime.
Different stage-local layouts require an explicitly selected generic program
weight pool; otherwise the original shared-layout check remains in force.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_DUMP_LIMIT_BYTES = 16_384
MAX_DUMP_LIMIT_BYTES = 262_144
MAX_SESSION_DUMP_BYTES = 8 << 20


def _selected_output_dumps(session, requests: tuple[str, ...],
                           byte_limit: int) -> list[dict]:
    if not DEFAULT_DUMP_LIMIT_BYTES <= byte_limit <= MAX_DUMP_LIMIT_BYTES:
        raise ValueError("output dump byte limit must be 16 KiB to 256 KiB")
    names = set(session.program_names)
    selected = []
    seen = set()
    total_bytes = 0
    output_sizes = {"f32": 4, "i64": 8, "i1": 1, "bf16": 2,
                    "f16": 2, "i32": 4}
    for request in requests:
        name, separator, raw_index = request.partition(":")
        if not separator or name not in names or not raw_index.isdecimal():
            raise ValueError(f"invalid --dump-output {request!r}; use program:output-index")
        program_index = session.program_names.index(name)
        program = session.programs[program_index]
        output_index = int(raw_index)
        if output_index >= len(program.outputs) or (program_index, output_index) in seen:
            raise ValueError(f"unknown or duplicate --dump-output {request!r}")
        shape, dtype = program.outputs[output_index]
        byte_count = output_sizes[dtype]
        for dimension in shape:
            byte_count *= dimension
        if byte_count > byte_limit:
            raise ValueError(f"dump output {request!r} exceeds the console byte limit")
        total_bytes += byte_count * program.steps
        if total_bytes > MAX_SESSION_DUMP_BYTES:
            raise ValueError("selected output dumps exceed 8 MiB per session")
        seen.add((program_index, output_index))
        selected.append({"program": program_index, "name": name,
                         "output": output_index, "bytes": byte_count})
    return sorted(selected, key=lambda row: (row["program"], row["output"]))


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


def _symbol_names(nm: Path, object_path: Path) -> tuple[str, ...]:
    """Observe local/undefined names too: objcopy may rewrite either kind."""
    result = subprocess.run([str(nm), "--format=posix", str(object_path)],
                            check=True, capture_output=True, text=True)
    rows = [line.split() for line in result.stdout.splitlines() if line.strip()]
    if any(len(row) < 2 for row in rows):
        raise ValueError("namespace inspector returned an incomplete symbol record")
    return tuple(row[0] for row in rows)


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


def _catalog_linker_script(output: Path, catalogs: list[Path]) -> Path:
    """Keep each large prepacked catalog's code and constants in PC-relative reach.

    Two TinyLlama stage catalogs hold more than 2 GiB of packed constants.
    Placing all text before all rodata makes the second stage's references and
    the common stack exceed RISC-V medany's signed 32-bit PC-relative range.
    Place the small common image and stack first, then each catalog's text,
    constants, and workspace together. The shared checkpoint blob still uses
    Merlin's separate absolute-address weights section.
    """
    names = [path.name for path in catalogs]
    if (len(names) != len(set(names)) or
            any(not re.fullmatch(r"[A-Za-z0-9_.-]+", name) for name in names)):
        raise ValueError("catalog link objects need distinct simple basenames")
    excluded = " ".join(f"*{name}" for name in names)
    groups = []
    for index, name in enumerate(names):
        pattern = f"*{name}"
        groups.extend((
            f"  .mx_catalog_{index}_text : {{ {pattern}(.text .text.*) }}",
            (f"  .mx_catalog_{index}_rodata : {{ {pattern}(.rodata .rodata.* "
             ".srodata .srodata.*) }"),
            (f"  .mx_catalog_{index}_data : {{ {pattern}(.data .data.* "
             ".sdata .sdata.*) }"),
            (f"  .mx_catalog_{index}_bss (NOLOAD) : {{ {pattern}(.bss .bss.* "
             ".sbss .sbss.*) }"),
        ))
    group_text = "\n".join(groups)
    source = f'''/* MX multi-stage layout; common code, then nearby catalog islands. */
OUTPUT_ARCH("riscv")
ENTRY(_start)
MERLIN_WEIGHTS_BASE = DEFINED(MERLIN_WEIGHTS_BASE) ? MERLIN_WEIGHTS_BASE : 0x200000000;
MERLIN_STACK_BYTES = DEFINED(MERLIN_STACK_BYTES) ? MERLIN_STACK_BYTES : 0x40000;
SECTIONS {{
  . = 0x80000000;
  .text : {{ *(.text.init) *(EXCLUDE_FILE({excluded}) .text*) }}
  . = ALIGN(0x1000);
  .htif : {{ *(.htif) }}
  . = ALIGN(8);
  .rodata : {{ *(EXCLUDE_FILE(*weights_blob.o {excluded}) .rodata*)
               *(EXCLUDE_FILE(*weights_blob.o {excluded}) .srodata*) }}
  . = ALIGN(8);
  .data : {{ *(EXCLUDE_FILE(*weights_blob.o {excluded}) .data*)
             *(EXCLUDE_FILE({excluded}) .sdata*) }}
  . = ALIGN(8);
  _bss_start = .;
  .bss : {{ *(EXCLUDE_FILE({excluded}) .bss*)
            *(EXCLUDE_FILE({excluded}) .sbss*) *(COMMON) }}
  _bss_end = .;
  . = ALIGN(16);
  . += MERLIN_STACK_BYTES;
  _stack_top = .;
{group_text}
  _merlin_static_end = .;
  . = MERLIN_WEIGHTS_BASE;
  .weights : {{ *weights_blob.o(.data .data.* .rodata .rodata.*) }}
}}
'''
    script = output / "mx_session_link.ld"
    script.write_text(source, encoding="utf-8")
    return script


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


def _projected_stage_entry(program, build: Path, receipt: dict):
    """Join the actual compiled stage to a fully rederived generic ABI proof."""
    from merlin.llvmlower.session_projection import read_program_entry_projection

    path = build / "entry_weight_projection/entry_weight_projection.json"
    if not receipt.get("prune_unused_checkpoint_weights"):
        if path.exists():
            raise ValueError("unselected stage entry projection cannot alter a session ABI")
        return None
    if (not path.is_file() or path.is_symlink() or
            _sha(path) != receipt.get("entry_weight_projection_receipt_sha256")):
        raise ValueError("stage entry projection differs from its build receipt")
    recipe = json.loads((build / "compilation_recipe.json").read_text())
    preparation = recipe.get("preparation", {}).get("entry_weight_projection", {})
    outputs = [row.get("output", {}) for row in recipe.get("commands", [])
               if Path(row.get("output", {}).get("path", "")).resolve() == (build / "model.o").resolve()]
    if (recipe.get("status") != "completed" or preparation.get("sha256") != _sha(path) or
            Path(preparation.get("path", "")).resolve() != path.resolve() or
            len(outputs) != 1 or outputs[0].get("sha256") != _sha(build / "model.o")):
        raise ValueError("projected session stage is not bound to its compiled model object")
    projection = json.loads(path.read_text())
    if "source_path" in projection:
        source = Path(projection["source_path"])
    else:
        # Older receipts lack an input filename. Reopen the checked original
        # post-offload selection, then reprove its canonical module identity.
        original = json.loads((build / "device_host_abi/post_offload_transform.json").read_text())
        source = Path(original["selected_path"])
        if _sha(source) != original["selected_sha256"]:
            raise ValueError("stage post-offload source differs from its original selection")
    return read_program_entry_projection(model_dir=program.bundle, prepared_dir=build,
                                          source=source, receipt=path)


def _namespace_stage(
    *, index: int, name: str, objects: list[Path], previous: set[str],
    nm: Path, objcopy: Path, output: Path,
) -> tuple[list[Path], set[str]]:
    from merlin.llvmlower.object_namespace import namespace_object

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
        receipt = namespace_object(
            path, destination, renamed, inspect_symbols=lambda obj: _symbol_names(nm, obj),
            rewrite=lambda source, target, selected: _run(
                [objcopy, "--redefine-syms", mapping, source, target]))
        destination.with_suffix(".namespace.json").write_text(json.dumps(receipt, indent=2) + "\n")
        result.append(destination)
    renamed_symbols = {renamed.get(symbol, symbol) for symbol in symbols}
    if renamed_symbols & previous:
        raise ValueError(f"stage {name} retains duplicate global symbols")
    return result, renamed_symbols


def _session_main(output_counts: list[int], *, weights_base: int,
                  dump_outputs: tuple[tuple[int, int], ...] = (),
                  measure_preparation: bool = False) -> str:
    if type(measure_preparation) is not bool:
        raise ValueError("session preparation timing requires an explicit boolean selection")
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
    preparation_helper, preparation_total = "", ""
    reset = "merlin_session_reset();"
    prepare = "merlin_session_prepare_program(program)"
    preparation_call = ""
    counter_barrier = ""
    if measure_preparation:
        counter_barrier = ' : : "memory"'
        preparation_helper = r'''
static void report_preparation(int program, uint64_t start_cycles, uint64_t start_instructions,
                               uint64_t *total_cycles, uint64_t *total_instructions) {
  uint64_t instructions = read_instret() - start_instructions;
  uint64_t cycles = read_cycle() - start_cycles;
  *total_cycles += cycles; *total_instructions += instructions;
  htif_puts("SESSION_PREP_PERF "); htif_putd(program); htif_putc(' ');
  htif_putd((long)cycles); htif_putc(' '); htif_putd((long)instructions); htif_putc('\n');
}
'''
        reset = '''uint64_t preparation_total_cycles = 0, preparation_total_instructions = 0;
  uint64_t preparation_start_cycles = read_cycle(), preparation_start_instructions = read_instret();
  merlin_session_reset();
  report_preparation(-1, preparation_start_cycles, preparation_start_instructions,
                      &preparation_total_cycles, &preparation_total_instructions);'''
        preparation_call = '''preparation_start_cycles = read_cycle();
    preparation_start_instructions = read_instret();
    int preparation_status = merlin_session_prepare_program(program);
    report_preparation(program, preparation_start_cycles, preparation_start_instructions,
                        &preparation_total_cycles, &preparation_total_instructions);
    '''
        prepare = "preparation_status"
        preparation_total = r'''htif_puts("SESSION_EXECUTION_TOTAL ");
  htif_putd((long)(total_cycles + preparation_total_cycles)); htif_putc(' ');
  htif_putd((long)(total_instructions + preparation_total_instructions)); htif_putc('\n');
  '''
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
  __asm__ volatile ("rdcycle %0" : "=r" (value){counter_barrier});
  return value;
}}
static uint64_t read_instret(void) {{
  uint64_t value;
  __asm__ volatile ("rdinstret %0" : "=r" (value){counter_barrier});
  return value;
}}
{preparation_helper}
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
  {reset}
  uint64_t total_cycles = 0, total_instructions = 0;
  for (int program = 0; program < MERLIN_SESSION_N_PROGRAMS; ++program) {{
    {preparation_call}if ({prepare} != 0) {{
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
  {preparation_total}htif_puts("METRIC memref_rank_mismatch ");
  htif_putd((long)merlin_memref_rank_mismatches()); htif_putc('\\n');
  htif_puts("DONE\\n"); htif_exit(0); return 0;
}}
'''


def link_session(
    session_root: Path, builds: dict[str, Path], output: Path, *,
    profile: Path, merlin_src: Path, riscv_root: Path,
    dump_outputs: tuple[str, ...] = (),
    dump_limit_bytes: int = DEFAULT_DUMP_LIMIT_BYTES,
    linked_elf_admission=None,
    measure_preparation: bool = False,
    pack_program_weights: bool = False,
) -> dict:
    sys.path.insert(0, str(merlin_src))
    from merlin.llvmlower import session_bundle
    from merlin.llvmlower.device_build import (
        DeviceRouting,
        admit_linked_device_elf,
        freeze_device_elf_admission,
        require_device_elf_admission,
    )
    from merlin.runtime.backends.spike_arena import (
        estimate_llvm_malloc,
        recommended_arena_bytes,
        spike_session_layout,
    )

    from mx_gemmini_support.target_profile import load_profile, profile_sha256

    admission_route = DeviceRouting(
        device="mx_gemmini", package_dir=Path(__file__).resolve().parents[1],
        operand_dtype="catalog_mixed_direct_fp4_fp8", accum_dtype="bf16",
        linked_elf_admission=linked_elf_admission)
    admission_selection = freeze_device_elf_admission(admission_route)

    session = session_bundle.load(session_root)
    names = set(session.program_names)
    if set(builds) != names:
        raise ValueError(f"stage builds must be exactly {sorted(names)}")
    selected_dumps = _selected_output_dumps(session, dump_outputs,
                                             dump_limit_bytes)
    expected_profile = profile_sha256(load_profile(profile))
    receipts = {}
    source_objects = {}
    projections = {}
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
            require_device_elf_admission(admission_route, admission_selection)
            details = json.loads(catalog.read_text(encoding="utf-8"))
            if (details.get("coverage_complete") is not True
                    or details.get("covered_contractions") != receipt["selected_operations"]
                    or details.get("unbound_precision_proposals") or
                    details.get("profile_sha256") != expected_profile or
                    details.get("compilation", {}).get("object_sha256") != _sha(objects[-1])):
                raise ValueError(f"stage {program.name} catalog has incomplete MX coverage")
        projection = _projected_stage_entry(program, build, receipt)
        if projection is not None:
            projections[program.name] = projection
        source_objects[program.name] = {
            str(path.relative_to(build)): _sha(path) for path in objects}
        receipts[program.name] = receipt
    if output.exists():
        raise ValueError("session output directory must not already exist")
    output.mkdir(parents=True)
    generated = session_bundle.generate(
        session_root, output / "generated", materialize_weights=False,
        entry_projections=projections or None, pack_program_weights=pack_program_weights)
    weight_sources = [builds[row["name"]] / "cgen/weights.bin" for row in generated["programs"]
                      if row["weights_bytes"]]
    if pack_program_weights:
        from merlin.llvmlower.session_weight_pool import materialize_program_weight_pool

        shared_weights = materialize_program_weight_pool(
            session, generated, output / "shared_weights", entry_projections=projections or None,
            weights_sources={row["name"]: builds[row["name"]] / "cgen/weights.bin"
                             for row in generated["programs"]})
    else:
        shared_weights = session_bundle.materialize_shared_weights(
            session, generated, output / "shared_weights", entry_projections=projections or None,
            weights_source=next((path for path in weight_sources if path.is_file()), None))
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
    catalog_objects: list[Path] = []
    previous: set[str] = set()
    for index, program in enumerate(session.programs):
        stage_objects = _stage_objects(builds[program.name])
        if index or projections:
            stage_objects, symbols = _namespace_stage(
                index=index, name=program.name, objects=stage_objects,
                previous=previous, nm=nm, objcopy=objcopy, output=output)
        else:
            symbols = set().union(*(_defined(nm, path) for path in stage_objects))
        previous.update(symbols)
        all_objects.extend(stage_objects)
        if len(stage_objects) == 3:
            catalog_objects.append(stage_objects[-1])
    first = builds[session.programs[0].name]
    runtime_names = ("merlin_model", "mlir_rt", "crt", "console", "libc_min")
    if not projections:
        runtime_names = ("model_call", *runtime_names)
    all_objects.extend(first / f"{name}.o" for name in runtime_names)
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
        if index or projections:
            all_objects.append(compile_c(
                stage_dir / "model_call.c", output / f"call_{index}.o", runtime_c, stage_dir))
    all_objects.append(compile_c(
        generated_root / "merlin_session.c", output / "session.o", generated_root))
    main_source = output / "session_main.c"
    main_source.write_text(
        _session_main([len(program.outputs) for program in session.programs],
                      weights_base=layout["weights_base"],
                      dump_outputs=tuple((row["program"], row["output"])
                                         for row in selected_dumps),
                      measure_preparation=measure_preparation),
        encoding="utf-8")
    all_objects.append(compile_c(main_source, output / "session_main.o", generated_root, bare))
    elf = output / "session.elf"
    linker_script = (_catalog_linker_script(output, catalog_objects)
                     if catalog_objects else bare / "model_link.ld")
    static_end_symbol = "_merlin_static_end" if catalog_objects else "_stack_top"
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
              "-T", linker_script, *all_objects, libm, "-o", elf])
        static_end = _symbol_address(nm, elf, static_end_symbol)
        checked = spike_session_layout(weights_bytes=weight_bytes,
                                       arena_bytes=arena_bytes,
                                       static_end=static_end)
        if checked["arena_base"] == layout["arena_base"]:
            layout = checked
            break
        layout = checked
    else:
        raise ValueError("linked static extent did not stabilize below the arena")
    admission = (admit_linked_device_elf(
        admission_route, admission_selection, elf=elf, linked_sha256=_sha(elf), directory=output)
        if catalog_objects else None)
    record = {
        "schema": "mx_gemmini.linked_merlin_session.v1",
        "performance_schema": "spike_rdcycle_rdinstret_per_step.v1",
        "preparation_performance_schema": (
            "spike_rdcycle_rdinstret_reset_prepare.v1" if measure_preparation else None),
        "execution_timing_scope": ("reset_plus_program_state_handoff_plus_existing_step_intervals; "
                                   "console_and_phase_bookkeeping_excluded" if measure_preparation else
                                   "existing_step_intervals_only; reset_and_program_state_handoff_excluded"),
        "dump_outputs": selected_dumps,
        "dump_limit_bytes": dump_limit_bytes,
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
        "shared_weight_storage_source": shared_weights["storage_source"],
        "program_weight_pool": shared_weights.get("program_weight_pool"),
        "object_namespace_storage": {path.name: json.loads(path.read_text())
                                     for path in sorted(output.glob("*.namespace.json"))},
        "stage_object_sha256": source_objects,
        "entry_projections": {name: generated["programs"][session.program_names.index(name)]["entry_projection"]
                              for name in projections},
        "linked_elf_admission": admission,
        "linked_elf_admission_required": bool(catalog_objects),
        "linked_elf_admission_selection": admission_selection[2] if admission is not None else None,
        "arena_layout": layout,
        "static_end_symbol": static_end_symbol,
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
                        help="program:output-index; record selected result bytes")
    parser.add_argument("--structural-elf-admission", action="store_true",
                        help="explicitly select digest/declared-ISA final session ELF admission")
    parser.add_argument("--measure-preparation", action="store_true",
                        help="also measure session reset and cross-program state handoffs")
    parser.add_argument("--pack-program-weights", action="store_true",
                        help="explicitly bind distinct stage-local constant blobs through a generic aligned pool")
    parser.add_argument("--dump-limit-bytes", type=int,
                        default=DEFAULT_DUMP_LIMIT_BYTES,
                        help="per-output console limit (16–256 KiB; 8 MiB total)")
    args = parser.parse_args()
    sys.path.insert(0, str(args.merlin_src.resolve(strict=True)))
    from merlin.model_check.admission import service as structural_admission
    record = link_session(
        args.session_root.resolve(strict=True), _stage_paths(args.stage_build),
        args.workdir.absolute(), profile=args.profile.resolve(strict=True),
        merlin_src=args.merlin_src.resolve(strict=True),
        riscv_root=args.riscv_root.resolve(strict=True),
        dump_outputs=tuple(args.dump_output),
        dump_limit_bytes=args.dump_limit_bytes,
        linked_elf_admission=(structural_admission("mx_gemmini")
                              if args.structural_elf_admission else None),
        measure_preparation=args.measure_preparation,
        pack_program_weights=args.pack_program_weights)
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
