"""Link a generated MX MMIO issuer with the Radiance Muon runtime.

The guarded callback proves the C++ calling convention and runtime link. Its
operand pointers are deliberately null, and the probe never enables the call;
this is a link qualification, not an executable kernel or numerical test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], cwd: Path, log: Path) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Muon link probe failed ({result.returncode}); see {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("object-dir", "radiance-root", "muon-clangxx", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    for name in ("object_dir", "radiance_root", "muon_clangxx", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    manifest_path = args.object_dir / "object_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    obj = args.object_dir / "mx_issue.o"
    header = args.object_dir / "mx_issue.h"
    if (manifest.get("schema") != "mx_gemmini.muon_mmio_object.v1" or
            manifest.get("transport") != "muon_mmio" or
            manifest.get("object_sha256") != _sha(obj) or
            manifest.get("issuer_h_sha256") != _sha(header)):
        parser.error("selected Muon object or header differs from its manifest")
    names = [entry["name"] for entry in manifest["buffer_abi"]]
    if not names or len(names) != len(set(names)) or any(
            re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name) is None for name in names):
        parser.error("invalid Muon object buffer ABI")
    lib = args.radiance_root / "lib"
    runtime = lib / "libmuonrt.a"
    linker = lib / "linker/mu_link.ld"
    tohost = lib / "tohost.S"
    if not all(path.is_file() for path in (runtime, linker, tohost, args.muon_clangxx)):
        parser.error("selected Muon runtime, linker script, or compiler is missing")
    args.out_dir.mkdir(parents=True)
    source = args.out_dir / "link_probe.cpp"
    source.write_text(
        '#include <stdint.h>\n#include <mu_schedule.h>\n#include "mx_issue.h"\n\n'
        'volatile uint32_t mx_probe_enable = 0;\n'
        'static void worker(void *, uint32_t lane, uint32_t, uint32_t block) {\n'
        '  if (lane == 0 && block == 0 && mx_probe_enable)\n'
        f'    mx_issue({", ".join("nullptr" for _ in names)}, '
        f'UINT32_C(0x{manifest["radiance_gateway_control_base"]:08x}));\n'
        '}\n'
        'extern "C" int main() {\n'
        '  mu_schedule(worker, nullptr, 1);\n'
        '  return 0;\n}\n')
    flags = ["--target=riscv32-unknown-elf", "-Xclang", "-target-feature",
             "-Xclang", "+vortex", "-march=rv32im_zfinx_zhinx", "-mabi=ilp32",
             "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-exceptions",
             "-fno-rtti", "-fdata-sections", "-ffunction-sections",
             f"-ffile-prefix-map={args.out_dir}=."]
    caller = args.out_dir / "link_probe.o"
    _run([str(args.muon_clangxx), *flags, f"-I{lib / 'include'}",
          f"-I{args.object_dir}", "-c", str(source), "-o", str(caller)],
         args.out_dir, args.out_dir / "compile.log")
    elf = args.out_dir / "link_probe.elf"
    _run([str(args.muon_clangxx), *flags, "-nodefaultlibs", "-nostartfiles",
          f"-Wl,-Bstatic,-T,{linker},-z,norelro", "-fuse-ld=lld",
          str(caller), str(obj), str(runtime), str(tohost), "-o", str(elf)],
         args.out_dir, args.out_dir / "link.log")
    bin_dir = args.muon_clangxx.parent
    readelf, nm = bin_dir / "llvm-readelf", bin_dir / "llvm-nm"
    if not readelf.is_file() or not nm.is_file():
        parser.error("selected Muon toolchain lacks llvm-readelf or llvm-nm")
    elf_header = subprocess.check_output([str(readelf), "-h", str(elf)], text=True)
    symbols = subprocess.check_output([str(nm), "-g", str(elf)], text=True)
    undefined = subprocess.check_output([str(nm), "-u", str(elf)], text=True)
    if ("Class:                             ELF32" not in elf_header or
            "Machine:                           RISC-V" not in elf_header or
            not all(re.search(rf"\bT {name}$", symbols, re.MULTILINE)
                    for name in ("_start", "main", "mx_issue")) or undefined.strip()):
        raise ValueError("Muon probe is not a complete RV32 runtime-linked ELF")
    receipt = {
        "schema": "mx_gemmini.muon_mmio_link_probe.v1",
        "status": "runtime_linked_unexecuted",
        "qualification": "structural_link_only",
        "link_probe_emitter_sha256": _sha(Path(__file__)),
        "object_manifest_sha256": _sha(manifest_path),
        "object_sha256": _sha(obj), "issuer_header_sha256": _sha(header),
        "probe_source_sha256": _sha(source), "probe_object_sha256": _sha(caller),
        "linked_elf_sha256": _sha(elf),
        "muon_clangxx_sha256": _sha(args.muon_clangxx),
        "muon_runtime_sha256": _sha(runtime),
        "muon_linker_script_sha256": _sha(linker),
        "muon_tohost_sha256": _sha(tohost),
        "gateway_control_base": manifest["radiance_gateway_control_base"],
        "mx_buffer_arguments": len(names),
        "defined_symbols_checked": ["_start", "main", "mx_issue"],
        "undefined_symbols": [],
    }
    (args.out_dir / "link_manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"Muon runtime-linked MX issuer (unexecuted): {elf}")


if __name__ == "__main__":
    main()
