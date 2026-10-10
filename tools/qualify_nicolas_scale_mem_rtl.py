"""Run a scoped Chisel test on Nicolas's unmodified MX scale-memory RTL.

The isolated Mill project supplies only the two interface bundles needed to
compile ScalingFactorMem.scala. The tested module is copied byte-for-byte from
the selected RTL checkout. This probes scale-half selection and E8M0 addition;
it does not elaborate the ExecuteController, RoCC DMA, or a complete MX tile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "tools/rtl_scale_half_probe"
RTL_REVISION = "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
MXGEN_REVISION = "dba3e7e706efa156d96efe595cd9935c825ea14d"
SCALE_MEM_SHA256 = "ed44c21427c8fd9e0bd74ea072be7dd971a593827ff59dcd97d0ea6acff2b549"
CONFIG_SHA256 = "4fa439ea860da11e61d2e389ee628dc769600c3a7b44f22be465b7dd21d498ce"
MILL_SHA256 = "3893a7aab91d35fb9d1196b0d9e1f59371e58fa553885d6a14fd7bd103de2cab"
SELECTORS = ((0, 0, 11), (0, 1, 14), (1, 0, 41), (1, 1, 44), (0, 0, 11))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                   text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtl-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    rtl, out = args.rtl_root.resolve(), args.out_dir.resolve()
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    module = rtl / "src/main/scala/gemmini/ScaleFactorMem.scala"
    config = rtl / "src/main/scala/gemmini/MxConfigFragments.scala"
    mxgen = rtl / "mxgen"
    launcher = mxgen / "mill"
    gitlink = subprocess.check_output(
        ["git", "-C", str(rtl), "ls-tree", "HEAD", "mxgen"], text=True).split()
    if (_revision(rtl) != RTL_REVISION or
            len(gitlink) < 3 or gitlink[2] != MXGEN_REVISION or
            _revision(mxgen) != MXGEN_REVISION or
            _sha(module) != SCALE_MEM_SHA256 or
            _sha(config) != CONFIG_SHA256 or
            _sha(launcher) != MILL_SHA256):
        parser.error("selected Nicolas RTL module, interface, or Mill launcher differs")

    project = out / "probe"
    project.mkdir(parents=True)
    for source in PROBE.rglob("*"):
        if source.is_file():
            dest = project / source.relative_to(PROBE)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
    dest_module = project / "src/main/scala/gemmini/ScaleFactorMem.scala"
    shutil.copyfile(module, dest_module)
    shutil.copyfile(launcher, project / "mill")
    (project / "mill").chmod(0o755)
    if _sha(dest_module) != SCALE_MEM_SHA256:
        raise ValueError("isolated RTL copy differs from Nicolas's source")
    expected = [f"act={act} weight={weight} combined_e8m0={value}"
                for act, weight, value in SELECTORS]
    probes = []
    for spec, name, mesh_rows, reset_between_cases in (
            ("ScaleHalfSelectorSpec", "dim16", 16, True),
            ("ScaleHalfWaveSpec", "wave4", 4, False)):
        command = ["./mill", "test.testOnly", f"gemmini.{spec}"]
        result = subprocess.run(command, cwd=project, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                check=False)
        log = out / f"{name}.log"
        log.write_text(result.stdout)
        found = [line.strip().split("] ", 1)[-1] for line in result.stdout.splitlines()
                 if "combined_e8m0=" in line]
        if (result.returncode or found != expected or
                "All tests passed." not in result.stdout or
                "Total number of tests run: 1" not in result.stdout):
            raise RuntimeError(f"scale-memory RTL {name} test failed; see {log}")
        probes.append({"name": name, "mesh_rows": mesh_rows,
                       "reset_between_selector_cases": reset_between_cases,
                       "command": command, "test_log_sha256": _sha(log)})

    inputs = ["build.mill.yaml", "test/package.mill.yaml",
              "src/main/scala/gemmini/ScaleFactorStubs.scala",
              "test/src/gemmini/ScaleHalfSelectorSpec.scala",
              "test/src/gemmini/ScaleHalfWaveSpec.scala"]
    receipt = {
        "schema": "mx_gemmini.nicolas_scale_half_rtl_probe.v1",
        "status": "scale_half_selectors_matched_on_actual_scaling_factor_mem_rtl",
        "scope": ("FP6 quad-scale DIM16 selector routing with reset between cases "
                  "and four-lane natural row-cycle alternation; depth-16 128-bit "
                  "scale SRAM; module alone, no controller/RoCC/FPGA"),
        "rtl_revision": RTL_REVISION,
        "mxgen_revision": MXGEN_REVISION,
        "rtl_scale_mem_sha256": _sha(module),
        "rtl_interface_sha256": _sha(config),
        "mill_launcher_sha256": _sha(launcher),
        "probe_inputs_sha256": {name: _sha(PROBE / name) for name in inputs},
        "scale_sram_depth": 16,
        "scale_sram_width_bits": 128,
        "selectors": [{"activation_half": a, "weight_half": w,
                       "combined_e8m0": value} for a, w, value in SELECTORS],
        "probes": probes,
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("Nicolas ScalingFactorMem: DIM16 routing and four-lane alternation passed")


if __name__ == "__main__":
    main()
