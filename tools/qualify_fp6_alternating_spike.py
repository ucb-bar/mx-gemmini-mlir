"""Qualify an FP6 alternating-scale program against stock and corrected Spike.

This copies Nicolas's pinned extension into an isolated directory. The only
model change is applying the CONFIG_SCALE_MEM half selector in the FP6 LUT
compute branch, as the RTL ScaleFactorMem does. Both models run the same ELF.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "docs/evidence/radiance_fp6_fullout_266c593/fp6_128x128x1024"
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE3M2OnlyGemminiRocketConfig.json"
OLD = ("const size_t a_off = group * (size_t)M_DIM + (size_t)(i*TM + r);\n"
       "              const size_t b_off = group * (size_t)N_DIM + (size_t)(j*TN + c);")
NEW = ("const size_t a_off = gemmini_state.mx_scale_act_sel * 4096 + "
       "group * (size_t)M_DIM + (size_t)(i*TM + r);\n"
       "              const size_t b_off = gemmini_state.mx_scale_wgt_sel * 4096 + "
       "group * (size_t)N_DIM + (size_t)(j*TN + c);")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv: list[str], *, cwd: Path, log: Path) -> subprocess.CompletedProcess:
    result = subprocess.run(argv, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log.write_text(result.stdout)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtl-root", required=True, type=Path)
    parser.add_argument("--riscv-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    rtl = args.rtl_root.resolve()
    riscv = args.riscv_root.resolve()
    out = args.out_dir.resolve()
    source = rtl / "software/libgemmini"
    pinned = subprocess.check_output(["git", "-C", str(rtl), "ls-tree", "HEAD",
                                      "software/libgemmini"], text=True).split()
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                       text=True).strip()
    if len(pinned) < 3 or pinned[2] != revision:
        parser.error("Spike extension does not match Nicolas's RTL gitlink")
    out.mkdir(parents=True)
    isolated = out / "software/libgemmini"
    isolated.parent.mkdir()
    shutil.copytree(source, isolated, ignore=shutil.ignore_patterns(".git"))
    (isolated.parent / "gemmini-rocc-tests").symlink_to(
        rtl / "software/gemmini-rocc-tests", target_is_directory=True)
    original = (source / "gemmini.cc").read_text()
    if original.count(OLD) != 1:
        raise ValueError("pinned FP6 LUT scale-selector omission changed")
    corrected = original.replace(OLD, NEW)
    (isolated / "gemmini.cc").write_text(corrected)
    patch = out / "spike_fp6_scale_selector.patch"
    patch.write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), corrected.splitlines(keepends=True),
        fromfile="a/gemmini.cc", tofile="b/gemmini.cc")))

    build = out / "build"
    compile_cmd = [sys.executable, "-m", "tools.compile_mx",
                   "--mlir", str(CASE / "bound.mlir"),
                   "--bundle", str(CASE / "bundle"),
                   "--profile", str(PROFILE), "--rtl-root", str(rtl),
                   "--riscv-root", str(riscv), "--out-dir", str(build),
                   "--physical-mode", "rtl_alternating",
                   "--experimental-spike-extension-root", str(isolated),
                   "--run-spike"]
    compiled = run(compile_cmd, cwd=ROOT, log=out / "compile.log")
    if compiled.returncode:
        raise RuntimeError("alternating FP6 compile or corrected Spike run failed")
    receipt = json.loads((build / "artifact_manifest.json").read_text())
    if (receipt["status"] != "source_golden_matched_on_experimental_spike" or
            receipt["mode"] != "rtl_alternating" or
            receipt["compared_bf16_outputs"] != 16384 or
            receipt["fp6_spike_scale_selector_workaround"]):
        raise ValueError("corrected Spike result is not the expected full-output FP6 run")

    sources = [source / "gemmini.cc", source / "gemmini_perf.cc"]
    sources += sorted((source / "perf").rglob("*.cc"))
    stock_so = out / "libgemmini_stock.so"
    stock_compile = run(["g++", "-L", str(riscv / "lib"),
                         f"-Wl,-rpath,{riscv / 'lib'}", "-shared", "-o", str(stock_so),
                         "-std=c++17", "-I", str(riscv / "include"), "-fPIC", "-O3",
                         f"-ffile-prefix-map={source}=software/libgemmini",
                         *(str(path) for path in sources)],
                        cwd=out, log=out / "stock_extension_build.log")
    if stock_compile.returncode:
        raise RuntimeError("pinned Spike extension build failed")
    stock = run([str(riscv / "bin/spike"), f"--extlib={stock_so}",
                 "--extension=gemmini", str(build / "mx_program.elf")],
                cwd=out, log=out / "stock_spike.log")
    found = re.search(r"lowered MX 128x128x1024: (\d+) BF16 mismatches",
                      stock.stdout)
    if stock.returncode == 0 or not found or int(found.group(1)) == 0:
        raise ValueError("stock Spike unexpectedly passed the alternating FP6 ELF")
    index = {
        "schema": "mx_gemmini.fp6_alternating_spike_experiment.v1",
        "status": "same_elf_failed_stock_and_matched_corrected_spike",
        "rtl_revision": subprocess.check_output(["git", "-C", str(rtl), "rev-parse", "HEAD"],
                                                text=True).strip(),
        "gemmini_extension_revision": revision,
        "source_bound_mlir_sha256": sha(CASE / "bound.mlir"),
        "source_bundle_manifest_sha256": sha(CASE / "bundle/manifest.json"),
        "profile_file_sha256": sha(PROFILE),
        "model_patch_sha256": sha(patch),
        "original_gemmini_cc_sha256": sha(source / "gemmini.cc"),
        "corrected_gemmini_cc_sha256": sha(isolated / "gemmini.cc"),
        "physical_program_sha256": sha(build / "physical_program.json"),
        "elf_sha256": sha(build / "mx_program.elf"),
        "stock_extension_sha256": sha(stock_so),
        "corrected_extension_sha256": sha(build / "libgemmini.so"),
        "stock_spike_log_sha256": sha(out / "stock_spike.log"),
        "corrected_spike_log_sha256": sha(build / "spike.log"),
        "stock_bf16_mismatches": int(found.group(1)),
        "corrected_bf16_mismatches": 0,
        "compared_bf16_outputs": 16384,
    }
    (out / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"FP6 alternating: {index['stock_bf16_mismatches']} stock mismatches; "
          "0 corrected mismatches over 16,384 BF16 outputs")


if __name__ == "__main__":
    main()
