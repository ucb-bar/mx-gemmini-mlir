"""Generate five DIM16 BF16 test headers with Nicolas's pinned model.

Run this only in an isolated checkout of gemmini-mx-cleanup. The script first
regenerates a checked-in E2M3×E4M3 header and requires byte-identical output.
It then registers the missing legal pairs with the existing generator; no
quantization or matrix arithmetic is reimplemented here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


MODES = {
    "e2m3_e4m3s": ("fp6:e2m3", "fp8:e4m3s"),
    "e3m2_e3m2": ("fp6:e3m2", "fp6:e3m2"),
    "e4m3s_e2m3": ("fp8:e4m3s", "fp6:e2m3"),
    "e4m3s_e4m3": ("fp8:e4m3s", "fp8:e4m3"),
    "e4m3_e4m3s": ("fp8:e4m3", "fp8:e4m3s"),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                   text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtl-root", required=True, type=Path)
    parser.add_argument("--microxcaling-root", required=True, type=Path)
    args = parser.parse_args()
    rtl = args.rtl_root.resolve()
    microxcaling = args.microxcaling_root.resolve()
    software = rtl / "software/gemmini-rocc-tests"
    generator = software / "gen_asym.py"
    if not generator.is_file() or not (microxcaling / "mx/elemwise_ops.py").is_file():
        parser.error("pinned Nicolas generator or microxcaling package is absent")
    if _revision(rtl) != "266c593f2cb51d7e3fe83fc0317072b585ac3c52":
        parser.error("the selected Gemmini MX checkout is not Nicolas's pinned revision")
    if _revision(microxcaling) != "7bc41952de394f5cc5e782baf132e7c7542eb4e4":
        parser.error("the selected microxcaling checkout differs from the validated revision")
    sys.path[:0] = [str(software), str(microxcaling)]
    import mx
    if Path(mx.__file__).resolve().parents[1] != microxcaling:
        raise RuntimeError("MX emulation package resolved to a different checkout")
    spec = importlib.util.spec_from_file_location("nicolas_gen_asym", generator)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def generate(act: str, wei: str, *, header: Path | None = None) -> None:
        sys.argv = [str(generator), "--act", act, "--wei", wei, "--dim", "16"]
        if header is not None:
            sys.argv += ["--header-path", str(header)]
        module.main()

    # The reference library and generator must reproduce an existing Nicolas
    # golden before their newly registered pairs are trusted.
    baseline = software / "include/matmul_data_asym_e2m3_e4m3.h"
    original = baseline.read_bytes()
    try:
        generate("fp6:e2m3", "fp8:e4m3")
        if baseline.read_bytes() != original:
            raise RuntimeError("Nicolas's checked-in E2M3×E4M3 header did not reproduce")
    finally:
        if baseline.read_bytes() != original:
            baseline.write_bytes(original)

    hashes = {}
    for name, (act, wei) in MODES.items():
        module.COMBOS[(act, wei)] = (name, None, {})
        header = software / f"include/matmul_data_asym_{name}.h"
        if header.exists():
            parser.error(f"refusing to overwrite {header}")
        generate(act, wei, header=header)
        hashes[name] = _sha(header)
    manifest = {
        "schema": "mx_gemmini.nicolas_generated_asymmetric_headers.v1",
        "rtl_revision": _revision(rtl),
        "software_revision": _revision(software),
        "generator_sha256": _sha(generator),
        "wrapper_sha256": _sha(Path(__file__)),
        "microxcaling_revision": _revision(microxcaling),
        "microxcaling_elemwise_sha256": _sha(microxcaling / "mx/elemwise_ops.py"),
        "baseline_sha256": _sha(baseline),
        "headers_sha256": hashes,
    }
    destination = software / "include/mx_gemmini_generated_modes_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
