"""Regenerate or create the 15 missing-source MX modes for DIM8 or DIM32.

The pinned Nicolas generator supplies all operand codes, scales, LUTs, and
BF16 goldens. Existing headers must reproduce byte for byte. New pair entries
only select the generator's existing math; they add no replacement kernel.
Run in an isolated gemmini-mx-cleanup checkout.
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
    "fp4_fp4": ("fp4:e2m1", "fp4:e2m1"),
    "fp4_e4m3": ("fp4:e2m1", "fp8:e4m3"),
    "fp4_e4m3s": ("fp4:e2m1", "fp8:e4m3s"),
    "e2m3_e4m3s": ("fp6:e2m3", "fp8:e4m3s"),
    "e2m3_e2m3": ("fp6:e2m3", "fp6:e2m3"),
    "e2m3_e4m3": ("fp6:e2m3", "fp8:e4m3"),
    "e3m2_e4m3": ("fp6:e3m2", "fp8:e4m3"),
    "e3m2_e3m2": ("fp6:e3m2", "fp6:e3m2"),
    "e3m2_e4m3s": ("fp6:e3m2", "fp8:e4m3s"),
    "e4m3s_e2m3": ("fp8:e4m3s", "fp6:e2m3"),
    "e4m3s_e4m3s": ("fp8:e4m3s", "fp8:e4m3s"),
    "e4m3s_e4m3": ("fp8:e4m3s", "fp8:e4m3"),
    "e4m3_e4m3s": ("fp8:e4m3", "fp8:e4m3s"),
    "e4m3_e4m3": ("fp8:e4m3", "fp8:e4m3"),
    "e5m2_e5m2": ("fp8:e5m2", "fp8:e5m2"),
}
BASELINE = {
    8: ("e2m3_e3m2", "fp6:e2m3", "fp6:e3m2",
        "0ae643c6be0e288d9d9b3cb9c163ac8c669b0fb8941cee85cd1a7daa35c94b83"),
    32: ("e2m3_e4m3", "fp6:e2m3", "fp8:e4m3",
         "13730bd97ec1306bb93d11f7cb7464dd5ce5b1b0a111f524c5679524c9a507b5"),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                   text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtl-root", required=True, type=Path)
    parser.add_argument("--microxcaling-root", required=True, type=Path)
    parser.add_argument("--mesh-dim", required=True, type=int, choices=(8, 32))
    args = parser.parse_args()
    dim = args.mesh_dim
    rtl, microxcaling = args.rtl_root.resolve(), args.microxcaling_root.resolve()
    software = rtl / "software/gemmini-rocc-tests"
    generator = software / "gen_asym.py"
    if (_git(rtl) != "266c593f2cb51d7e3fe83fc0317072b585ac3c52" or
            _git(software) != "350547f9843f46f485d4d1dd4c20b2f52f4844bf" or
            _git(microxcaling) != "7bc41952de394f5cc5e782baf132e7c7542eb4e4"):
        parser.error("selected generator, software, or microxcaling revision differs")
    sys.path[:0] = [str(software), str(microxcaling)]
    import mx
    if Path(mx.__file__).resolve().parents[1] != microxcaling:
        raise RuntimeError("MX emulation package resolved to a different checkout")
    spec = importlib.util.spec_from_file_location("nicolas_gen_asym_mesh", generator)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def generate(act: str, wei: str) -> None:
        sys.argv = [str(generator), "--act", act, "--wei", wei,
                    "--dim", str(dim)]
        module.main()

    base_name, base_act, base_wei, base_sha = BASELINE[dim]
    baseline = software / f"include/matmul_data_asym_{base_name}_dim{dim}.h"
    original = baseline.read_bytes()
    if _sha(baseline) != base_sha:
        parser.error("checked-in mesh baseline differs from pinned source")
    try:
        generate(base_act, base_wei)
        if baseline.read_bytes() != original:
            raise RuntimeError("Nicolas mesh baseline did not regenerate byte for byte")
    finally:
        if baseline.read_bytes() != original:
            baseline.write_bytes(original)

    hashes, origins = {}, {}
    for name, (act, wei) in MODES.items():
        key = (module.combo_key(act), module.combo_key(wei))
        if key not in module.COMBOS:
            module.COMBOS[key] = (name, None, {})
        elif module.COMBOS[key][0] != name:
            raise ValueError(f"generator names {key} differently")
        header = software / f"include/matmul_data_asym_{name}_dim{dim}.h"
        before = header.read_bytes() if header.exists() else None
        generate(act, wei)
        if before is not None and header.read_bytes() != before:
            header.write_bytes(before)
            raise RuntimeError(f"checked-in {name} DIM{dim} header did not reproduce")
        hashes[name] = _sha(header)
        relative = str(header.relative_to(software))
        tracked = subprocess.run(["git", "-C", str(software), "ls-files",
                                  "--error-unmatch", relative],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL).returncode == 0
        origins[name] = "checked_in" if tracked else "generated"
    manifest = {
        "schema": "mx_gemmini.nicolas_generated_mesh_headers.v1",
        "mesh_dim": dim,
        "rtl_revision": _git(rtl), "software_revision": _git(software),
        "generator_sha256": _sha(generator),
        "wrapper_sha256": _sha(Path(__file__)),
        "microxcaling_revision": _git(microxcaling),
        "microxcaling_elemwise_sha256": _sha(microxcaling / "mx/elemwise_ops.py"),
        "baseline_sha256": _sha(baseline),
        "headers_sha256": hashes,
        "header_origin": origins,
    }
    destination = software / f"include/mx_gemmini_generated_modes_dim{dim}_manifest.json"
    if destination.exists():
        parser.error(f"refusing to overwrite {destination}")
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
