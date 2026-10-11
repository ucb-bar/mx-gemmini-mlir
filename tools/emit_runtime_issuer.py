"""Build a callable MX issuer object with live encoded operands for Spike.

This is the MX half of a future Merlin device catalog entry.  The pointer ABI
uses packed codes and E8M0 scales; it is not yet Merlin's dense tensor ABI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.runtime_issuer import emit_runtime_issuer
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mlir", "bundle", "profile", "rtl-root", "riscv-root", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, name).resolve() for name in
             ("mlir", "bundle", "profile", "rtl_root", "riscv_root", "out_dir")}
    if paths["out_dir"].exists():
        parser.error("output directory already exists")
    cc = paths["riscv_root"] / "bin/riscv64-unknown-elf-gcc"
    nm = paths["riscv_root"] / "bin/riscv64-unknown-elf-nm"
    objdump = paths["riscv_root"] / "bin/riscv64-unknown-elf-objdump"
    if not all(tool.is_file() for tool in (cc, nm, objdump)):
        parser.error("selected RV64 compiler or object inspectors are unavailable")
    profile = load_profile(paths["profile"], rtl_root=paths["rtl_root"])
    manifest, resources = load_bundle(paths["bundle"])
    program = lower_bound_source(paths["mlir"].read_text(), profile, manifest, resources)
    issuer = emit_runtime_issuer(program, manifest)
    out = paths["out_dir"]
    out.mkdir(parents=True)
    source = out / "mx_issue_encoded.c"
    obj = out / "mx_issue_encoded.o"
    source.write_text(issuer.source)
    build = subprocess.run([
        str(cc), "-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O1",
        "-ffreestanding", "-fno-common", "-c", str(source), "-o", str(obj),
    ], capture_output=True, text=True)
    if build.returncode:
        raise RuntimeError(f"RV64 issuer compilation failed: {build.stderr[-1200:]}")
    symbols = subprocess.run([str(nm), "-g", "-P", str(obj)],
                             capture_output=True, text=True, check=True)
    rows = [line.split() for line in symbols.stdout.splitlines()]
    if (sum(len(row) >= 2 and row[0] == "mx_issue_encoded" and row[1].upper() == "T"
            for row in rows) != 1 or
            any(len(row) < 2 or row[1].upper() == "U" or
                row[1].upper() == "T" and row[0] != "mx_issue_encoded" for row in rows)):
        raise ValueError("runtime issuer object has missing or unexpected external symbols")
    sections = subprocess.run([str(objdump), "-h", str(obj)],
                              capture_output=True, text=True, check=True)
    for line in sections.stdout.splitlines():
        fields = line.split()
        if (len(fields) >= 3 and fields[0].isdigit() and
                fields[1] in {".data", ".sdata", ".rodata", ".bss", ".sbss"} and
                int(fields[2], 16) != 0):
            raise ValueError("runtime issuer object embeds data or allocates fixed buffers")
    receipt = {
        "schema": "mx_gemmini.encoded_runtime_issuer.v1",
        "status": "encoded_pointer_object_built_not_whole_model",
        "abi": {"symbol": "mx_issue_encoded", "argument_order": [
            "activation", "activation_scales", "weight", "weight_scales",
            "output_bf16", "scratch_output_scales"],
            "pointee_layout": "packed_codes_and_e8m0_scales",
            "buffer_bytes": dict(issuer.buffer_bytes),
            "output_dtype": "bf16"},
        "precision": issuer.precision,
        "shape_mnk": list(issuer.shape_mnk),
        "profile_sha256": issuer.profile_sha256,
        "payload_manifest_sha256": issuer.payload_manifest_sha256,
        "bound_mlir_sha256": _sha(paths["mlir"]),
        "issuer_source_sha256": _sha(source),
        "object_sha256": _sha(obj),
    }
    (out / "runtime_issuer.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"object": str(obj), "manifest": str(out / "runtime_issuer.json"),
                      "object_sha256": receipt["object_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
