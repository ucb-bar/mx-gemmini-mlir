"""Resolve a packaged bare-metal workload using the pinned FireSim parser."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
from pathlib import Path

from tools.model_release import registry, sha
from tools.package_firesim_model import load_segments


def check(package, firesim):
    package, firesim = package.resolve(), firesim.resolve()
    pin = registry()["repositories"]["firesim"]["revision"]
    revision = subprocess.check_output(["git", "-C", str(firesim), "rev-parse", "HEAD"], text=True).strip()
    if revision != pin:
        raise ValueError("FireSim checkout differs from the published parser revision")
    path = firesim / "deploy/runtools/workload.py"
    original = subprocess.check_output(["git", "-C", str(firesim), "show", pin + ":deploy/runtools/workload.py"])
    if path.read_bytes() != original:
        raise ValueError("FireSim workload parser has local changes")
    spec = importlib.util.spec_from_file_location("firesim_workload_parser", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    requirements = json.loads((package / "requirements.json").read_text())
    previous = Path.cwd()
    try:
        os.chdir(package)
        workload = module.WorkloadConfig(requirements["workload"], "validation", "")
        job = workload.get_job(0)
        elf = Path(job.bootbinary_path()).resolve()
        if not elf.is_relative_to(package) or job.rootfs_path() is not None or job.simoutputs != ["uartlog"]:
            raise ValueError("workload resolves outside package or changes bare-metal console selection")
        if sha(elf) != requirements["elf_sha256"] or elf.stat().st_size != requirements["elf_bytes"]:
            raise ValueError("packaged executable changed")
        entry, segments = load_segments(elf)
        if entry != requirements["entry"] or segments != requirements["load_segments"]:
            raise ValueError("packaged executable layout changed")
    finally:
        os.chdir(previous)
    return {"schema": "mx_gemmini.firesim_package_check.v1", "status": "parser_and_elf_checked_not_fpga_executed",
            "firesim_revision": revision, "parser_sha256": sha(path), "elf_sha256": requirements["elf_sha256"],
            "recommended_dram_bytes": requirements["recommended_dram_bytes"], "rootfs": None,
            "simulation_outputs": ["uartlog"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--firesim-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("result path must be fresh")
    result = check(args.package, args.firesim_root)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
