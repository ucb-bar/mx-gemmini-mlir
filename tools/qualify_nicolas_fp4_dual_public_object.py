"""Replay Nicolas's flat+tiled FP4 SPAD_REQUANT with the public object CLI."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
import shutil
import sys

from mx_gemmini_support.fp4_dual_requant import render_fp4_dual_requant
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _require_gitlink, _run, _sha, _source_closure
from tools.qualify_nicolas_spad_requant_fp4 import (
    DEFAULT_PROFILE, ROOT, _compile_program, _compiler_driver, _run_spike,
)


RTL_REVISION = "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
SOURCE_SHA256 = "812842fd751d7fa0d8fb98a150df29424bdaba65613531335001f9f2260eb884"
REFERENCE_SHA256 = "72ed54217ac242f78b24399fb1620c59b5842110290d0ee768e1fbdc84011c3e"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rtl-root", "riscv-root", "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--baseline-receipt", type=Path,
                        help="require a fresh replay to reproduce every recorded digest")
    args = parser.parse_args()
    rtl, riscv, out = args.rtl_root.resolve(), args.riscv_root.resolve(), args.out_dir.resolve()
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    software, extension = (rtl / "software/gemmini-rocc-tests",
                           rtl / "software/libgemmini")
    if _git_revision(rtl) != RTL_REVISION:
        raise ValueError("public FP4 source replay needs pinned Nicolas RTL")
    _require_gitlink(rtl, "software/gemmini-rocc-tests")
    _require_gitlink(rtl, "software/libgemmini")
    profile = load_profile(args.profile, rtl_root=rtl)
    if (profile["transport"] != "rocket_rocc" or
            profile["geometry"]["mesh_columns"] != 16 or
            not profile["resources"].get("spad_requant") or
            "fp4_e2m1" not in profile["candidate_output_modes"]):
        raise ValueError("selected profile lacks Rocket DIM16 FP4 SPAD_REQUANT")
    cc, spike = riscv / "bin/riscv64-unknown-elf-gcc", riscv / "bin/spike"
    if not cc.is_file() or not spike.is_file() or shutil.which("g++") is None:
        parser.error("RISC-V GCC, Spike, and host g++ are required")
    source = software / "bareMetalC/spad_requant_fp4.c"
    reference = software / "include/mx_e4m3_ref.h"
    if not source.is_file() or not reference.is_file():
        parser.error("Nicolas FP4 source or reference header is absent")
    if _sha(source) != SOURCE_SHA256 or _sha(reference) != REFERENCE_SHA256:
        raise ValueError("Nicolas FP4 source or reference differs from pinned code")

    out.mkdir(parents=True)
    original = source.read_text()
    modified = _compiler_driver(original)
    driver = out / "compiler_driver.c"
    driver.write_text(modified)
    patch = out / "compiler_driver.patch"
    patch.write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), modified.splitlines(keepends=True),
        fromfile="spad_requant_fp4.c", tofile="compiler_driver.c")))
    mlir = out / "connected.mlir"
    mlir.write_text(render_fp4_dual_requant(
        profile, source_sha256=SOURCE_SHA256, header_sha256=REFERENCE_SHA256))
    object_dir = out / "object"
    _run([sys.executable, "-m", "tools.compile_object", "--mlir", str(mlir),
          "--profile", str(args.profile.resolve()), "--rtl-root", str(rtl),
          "--riscv-root", str(riscv), "--mx-opt", str(args.mx_opt.resolve()),
          "--out-dir", str(object_dir)], cwd=ROOT, log=out / "object_compile.log")
    dispatch = json.loads((object_dir / "compile_manifest.json").read_text())
    object_manifest = json.loads((object_dir / "object_manifest.json").read_text())
    expected_abi = ("X", "scales_hw", "scales_hw2", "codes_flat_hw", "codes_tiled_hw")
    if (dispatch["lowering_family"] != "fp4_dual_requant" or
            [slot["name"] for slot in object_manifest["buffer_abi"]] != list(expected_abi) or
            object_manifest["object_sha256"] != _sha(object_dir / "mx_issue.o")):
        raise ValueError("public FP4 object ABI or dispatch differs from source driver")

    baseline_elf = _compile_program(out / "source", source, software, cc, None)
    compiled_elf = _compile_program(out / "compiled", driver, software, cc,
                                    object_dir / "mx_issue.o")
    extension_sources = [extension / "gemmini.cc", extension / "gemmini_perf.cc"]
    extension_sources += sorted((extension / "perf").rglob("*.cc"))
    so = out / "libgemmini.so"
    _run(["g++", "-L", str(riscv / "lib"),
          f"-Wl,-rpath,{riscv / 'lib'}", "-shared", "-o", str(so),
          "-std=c++17", "-I", str(riscv / "include"), "-fPIC", "-O3",
          f"-ffile-prefix-map={extension}=software/libgemmini",
          *(str(path) for path in extension_sources)],
         cwd=out, log=out / "extension_build.log")
    source_result = _run_spike(spike, so, baseline_elf, out / "source/spike.log")
    compiled_result = _run_spike(spike, so, compiled_elf, out / "compiled/spike.log")
    if (source_result["compared_fp4_codes"] != 16384 or
            compiled_result["compared_fp4_codes"] != 16384 or
            source_result["compared_e8m0_scales"] != 512 or
            compiled_result["compared_e8m0_scales"] != 512):
        raise ValueError("public FP4 replay compared an unexpected source extent")
    receipt = {
        "schema": "mx_gemmini.nicolas_fp4_dual_public_object_spike.v1",
        "status": "source_and_public_object_matched_on_pinned_spike",
        "scope": "Nicolas source input and per-element oracle; public compiled flat+tiled FP4 SPAD_REQUANT",
        "compiler_revision": _git_revision(ROOT),
        "compiler_source_closure_sha256": _source_closure(
            ROOT, sorted((ROOT / "mx_gemmini_support").glob("*.py")) +
            sorted((ROOT / "tools").glob("*.py"))),
        "rtl_revision": _git_revision(rtl),
        "software_revision": _git_revision(software),
        "extension_revision": _git_revision(extension),
        "source_sha256": SOURCE_SHA256,
        "reference_header_sha256": REFERENCE_SHA256,
        "profile_name": profile["name"],
        "profile_sha256": profile_sha256(profile),
        "connected_mlir_sha256": _sha(mlir),
        "driver_sha256": _sha(driver),
        "driver_patch_sha256": _sha(patch),
        "object_dispatch_manifest_sha256": _sha(object_dir / "compile_manifest.json"),
        "object_manifest_sha256": _sha(object_dir / "object_manifest.json"),
        "object_sha256": _sha(object_dir / "mx_issue.o"),
        "command_count": object_manifest["command_count"],
        "allocated_data_section_bytes": object_manifest["allocated_data_section_bytes"],
        "physical_program_sha256": _sha(object_dir / "physical_program.json"),
        "issuer_c_sha256": _sha(object_dir / "mx_issue.c"),
        "spike_sha256": _sha(spike),
        "extension_sha256": _sha(so),
        "source_spike": source_result,
        "compiler_spike": compiled_result,
    }
    if (args.baseline_receipt and
            receipt != json.loads(args.baseline_receipt.read_text())):
        raise ValueError("public FP4 source/compiler Spike run differs from baseline receipt")
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(receipt["status"])


if __name__ == "__main__":
    main()
