"""Attribute declared session-state copies using the actual compiled runtime/libc.

This isolates copy work on synthetic raw bytes at the captured state extents.
It changes no session code and establishes neither a buffer-rotation permission
nor a whole-model speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from math import prod
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage-build", "session-root", "profile", "riscv-root", "extlib", "out-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--program", required=True)
    parser.add_argument("--structural-elf-admission", action="store_true")
    args = parser.parse_args()
    if args.out_dir.exists() or not args.structural_elf_admission:
        parser.error("fresh output and explicit structural admission required")
    from merlin.common.yaml import load_yaml
    from merlin.llvmlower.device_build import (
        DeviceRouting,
        admit_linked_device_elf,
        freeze_device_elf_admission,
    )
    from merlin.llvmlower.session_bundle import load
    from merlin.llvmlower.session_projection import read_program_entry_projection
    from merlin.model_check.admission import service

    from mx_gemmini_support.target_profile import load_profile, profile_sha256

    session = load(args.session_root)
    program, = [item for item in session.programs if item.name == args.program]
    build = args.stage_build.resolve(strict=True)
    proof_path = build / "entry_weight_projection/entry_weight_projection.json"
    projection_data = json.loads(proof_path.read_text())
    projection = read_program_entry_projection(model_dir=program.bundle, prepared_dir=build,
                                               source=Path(projection_data["source_path"]), receipt=proof_path)
    remap = projection.projection.original_to_projected
    child = load_yaml(program.bundle / "session_contract.yaml")
    sizes = {"f32": 4, "bf16": 2, "f16": 2, "i64": 8, "i32": 4, "i1": 1}
    pairs = []
    for state in child.get("states", []):
        original, output = state["input_arg"], state["output_index"]
        if program.inputs[original] != program.outputs[output] or remap[original] is None:
            raise ValueError("state copy requires the original exact typed input/output mapping")
        shape, dtype = program.inputs[original]
        pairs.append(dict(input=remap[original], output=output, bytes=prod(shape) * sizes[dtype]))
    if not pairs or any(pair["bytes"] <= 0 for pair in pairs) or sum(pair["bytes"] for pair in pairs) > 64 << 20:
        raise ValueError("state attribution requires positive bounded complete state extents")
    recipe = json.loads((build / "compilation_recipe.json").read_text())
    out = args.out_dir.resolve()
    out.mkdir(parents=True)
    objects = []
    for name in ("merlin_model", "crt", "console", "libc_min"):
        path = build / (name + ".o")
        observed, = [row for row in recipe["commands"] if row.get("output", {}).get("path") == str(path)]
        if observed["output"]["sha256"] != sha(path):
            raise ValueError("compiled runtime object changed from its original stage recipe")
        for item in observed["inputs"]:
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("runtime source changed from its compiled attribution")
        target = out / path.name
        target.write_bytes(path.read_bytes())
        objects.append(target)
    # Resolve retained runtime sources through Merlin's repository location API.
    from merlin.common.paths import merlin_dir

    runtime = merlin_dir() / "runtime"
    for path in (runtime / "c/merlin_model.h", build / "cgen/model_gen.h"):
        (out / path.name).write_bytes(path.read_bytes())
    declarations, initialize, checks, inputs, outputs = [], [], [], [], []
    for index, pair in enumerate(pairs):
        count = pair["bytes"]
        declarations.append(f"static unsigned char a{index}[{count + 128}] __attribute__((aligned(64)));static unsigned char b{index}[{count + 128}] __attribute__((aligned(64)));")
        initialize.append(f"for(size_t i=0;i<{count + 128};++i){{a{index}[i]=0xa5;b{index}[i]=0xa5;}} for(size_t i=0;i<{count};++i){{a{index}[i+64]=0;b{index}[i+64]=(unsigned char)(i*17+{index}*31+run);}}")
        checks.append(f"for(size_t i=0;i<{count};++i)bad+=a{index}[i+64]!=b{index}[i+64]||b{index}[i+64]!=(unsigned char)(i*17+{index}*31+run);for(size_t i=0;i<64;++i)bad+=a{index}[i]!=0xa5||b{index}[i]!=0xa5||a{index}[{count}+64+i]!=0xa5||b{index}[{count}+64+i]!=0xa5;")
        inputs.append(f"in[{pair['input']}]=a{index}+64;")
        outputs.append(f"out[{pair['output']}]=b{index}+64;")
    source = f'''#include <stdint.h>
#include "model_gen.h"
void console_init(void);void htif_puts(const char*);void htif_putd(long);void htif_exit(int);
void merlin_invoke(void **d){{(void)d;}}
{''.join(declarations)}
static const int input_args[]={{ {','.join(str(pair['input']) for pair in pairs)} }};
static const int output_indices[]={{ {','.join(str(pair['output']) for pair in pairs)} }};
static uint64_t cycle(void){{uint64_t r;asm volatile("rdcycle %0":"=r"(r)::"memory");return r;}}
int main(void){{console_init();void *in[MERLIN_N_ARGS]={{0}},*out[MERLIN_N_OUTPUTS]={{0}};
{''.join(inputs + outputs)}
for(unsigned run=0;run<4;++run){{unsigned calls=run<3?1:{program.steps};
{''.join(initialize)}
uint64_t begin=cycle();int status=0;for(unsigned i=0;i<calls;++i)status|=merlin_commit_state(MERLIN_ARGS,MERLIN_N_ARGS,in,out,{len(pairs)},input_args,output_indices);
uint64_t elapsed=cycle()-begin;unsigned bad=0;{''.join(checks)}
htif_puts("STATE_COPY run=");htif_putd(run);htif_puts(" calls=");htif_putd(calls);htif_puts(" cycles=");htif_putd(elapsed);htif_puts(" bad=");htif_putd(bad);htif_puts("\\n");if(status||bad)htif_exit(3);
}}htif_puts("DONE\\n");htif_exit(0);return 0;}}
'''
    (out / "driver.c").write_text(source)
    (out / "original_compilation_recipe.json").write_bytes((build / "compilation_recipe.json").read_bytes())
    (out / "projection.json").write_bytes(proof_path.read_bytes())
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    cc = args.riscv_root.resolve() / "bin/riscv64-unknown-elf-gcc"
    flags = ["-O2", "-ffreestanding", "-fno-builtin", "-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany"]
    commands = [[str(cc), *flags, "-I", str(out), "-c", str(out / "driver.c"), "-o", str(out / "driver.o")],
                [str(cc), *flags, "-nostdlib", "-nostartfiles", "-T", str(runtime / "baremetal/spike/model_link.ld"),
                 str(out / "driver.o"), *map(str, objects), "-o", str(out / "section.elf")]]
    for index, command in enumerate(commands):
        result = subprocess.run(command, capture_output=True, check=False)
        (out / f"compile{index}.log").write_bytes(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError("state copy attribution compilation/link failed")
    elf = out / "section.elf"
    route = DeviceRouting(device="mx_gemmini", package_dir=Path(__file__).resolve().parents[1],
                          operand_dtype="opaque_state_bytes", accum_dtype="opaque_state_bytes", linked_elf_admission=service("mx_gemmini"))
    admission = admit_linked_device_elf(route, freeze_device_elf_admission(route), elf=elf,
                                       linked_sha256=sha(elf), directory=out)
    env = os.environ | {"GEMMINI_MODE": "both", "GEMMINI_PERF_CONFIG": "e4m3_vpu", "GEMMINI_PERF_SET": "mem.host_tracking=0", "GEMMINI_PERF_OUT": str(out / "perf.txt")}
    command = [str(args.riscv_root.resolve() / "bin/spike"), "--isa=rv64gc_zicntr", "--extlib=" + str(args.extlib.resolve()), "--extension=gemmini", str(elf)]
    with (out / "spike.log").open("wb") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=False)
    rows = [dict(field.split("=", 1) for field in line.split()[1:]) for line in (out / "spike.log").read_text().splitlines() if line.startswith("STATE_COPY ")]
    if (result.returncode or len(rows) != 4 or any(int(row["bad"]) for row in rows)
            or [int(row["calls"]) for row in rows] != [1, 1, 1, program.steps]
            or (out / "spike.log").read_text().splitlines().count("DONE") != 1):
        raise ValueError("state copy attribution did not validate every byte and guard")
    record = {"schema": "mx_gemmini.state_copy_attribution.v1", "scope": "synthetic_state_bytes_actual_runtime_objects_not_whole_model_or_rotation_permission",
              "pairs": pairs, "per_call_bytes": sum(pair["bytes"] for pair in pairs), "performance": rows,
              "command": command, "commands": commands, "admission": admission, "profile_sha256": profile_sha256(load_profile(args.profile)),
              "extension_sha256": sha(args.extlib), "exit_code": result.returncode,
              "compiler_sha256": sha(cc),
              "environment": {key: env[key] for key in ("GEMMINI_MODE", "GEMMINI_PERF_CONFIG", "GEMMINI_PERF_SET", "GEMMINI_PERF_OUT")},
              "files_sha256": {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()}}
    (out / "receipt.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"bytes": record["per_call_bytes"], "performance": rows}), flush=True)


if __name__ == "__main__":
    main()
