"""Run Nicolas's pinned E4M3 LUT-enable PE test and archive its exact scope."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


GEMMINI_REVISION = "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
MXGEN_REVISION = "dba3e7e706efa156d96efe595cd9935c825ea14d"
TEST = "mxgen.MxFpMul_MxGemminiE4M3Lut_Mode9_Spec"
TEST_SOURCE = "mxgen/test/src/mxgen/MxGemminiE4M3LutTest.scala"
CONTROLLER_SOURCE = "src/main/scala/gemmini/ExecuteController.scala"
TEST_SHA256 = "90a191936865e390fa1b04c368d83f342fd18b77fad842100195da2f674591fc"
CONTROLLER_SHA256 = "b99add3039d5c64425670cfee15fa5c9eb3aa7fd49e505b93d37562cd9abd7df"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemmini-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    gemmini, out = args.gemmini_root.resolve(), args.out_dir.resolve()
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    mxgen = gemmini / "mxgen"
    if (_revision(gemmini) != GEMMINI_REVISION or
            _revision(mxgen) != MXGEN_REVISION or
            _sha(gemmini / TEST_SOURCE) != TEST_SHA256 or
            _sha(gemmini / CONTROLLER_SOURCE) != CONTROLLER_SHA256):
        parser.error("Nicolas RTL, MxGen, PE test, or controller differs from pinned source")
    controller = (gemmini / CONTROLLER_SOURCE).read_text()
    if "(e4m3QuadThroughput.B && io.weight_lut_en)" not in controller:
        parser.error("controller no longer selects quad weight throughput with E4M3 LUT")
    out.mkdir(parents=True)
    command = [str(mxgen / "mill"), "test.testOnly", TEST]
    run = subprocess.run(command, cwd=mxgen, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, check=False)
    log = out / "test.log"
    log.write_text(run.stdout)
    if (run.returncode != 0 or
            "Total number of tests run: 1" not in run.stdout or
            "Tests: succeeded 1, failed 0" not in run.stdout or
            "All tests passed." not in run.stdout):
        raise RuntimeError(f"pinned MxGen PE test failed; see {log}")
    receipt = {
        "schema": "mx_gemmini.nicolas_e4m3_lut_pe_rtl_test.v1",
        "status": "pe_mode9_lut_enable_arithmetic_passed",
        "scope": "MxGen PE arithmetic; no LUT DMA, ExecuteController simulation, "
                 "full RoCC loop, or FPGA claim",
        "gemmini_revision": GEMMINI_REVISION,
        "mxgen_revision": MXGEN_REVISION,
        "test_class": TEST,
        "test_source_sha256": TEST_SHA256,
        "controller_source_sha256": CONTROLLER_SHA256,
        "command": ["./mill", "test.testOnly", TEST],
        "test_exit_code": run.returncode,
        "test_log_sha256": _sha(log),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("Nicolas MxGen E4M3 LUT-enable PE test passed; full loop remains unqualified")


if __name__ == "__main__":
    main()
