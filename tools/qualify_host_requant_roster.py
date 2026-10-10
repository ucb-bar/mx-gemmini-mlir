"""Compile every source-bound Radiance requant graph as an object and run Spike."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from mx_gemmini_support.target_profile import load_profile, profile_sha256


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile_for(mlir: Path, profiles: Path, rtl_root: Path) -> Path:
    match = re.search(r'mx.profile_sha256 = "([0-9a-f]{64})"', mlir.read_text())
    if not match:
        raise ValueError(f"{mlir} has no selected profile digest")
    matching = [path for path in profiles.glob("Mx*RocketConfig.json")
                if profile_sha256(load_profile(path, rtl_root=rtl_root)) == match[1]]
    if len(matching) != 1:
        raise ValueError(f"{mlir} has {len(matching)} matching Rocket profiles")
    return matching[0]


def _run_case(case: Path, profiles: Path, rtl_root: Path, riscv_root: Path,
              mx_opt: Path, output: Path) -> dict:
    mlir, bundle = case / "payload_bound.mlir", case / "bundle"
    profile = _profile_for(mlir, profiles, rtl_root)
    obj, spike = output / case.name / "object", output / case.name / "spike"
    common = ["--mlir", str(mlir), "--bundle", str(bundle),
              "--profile", str(profile), "--rtl-root", str(rtl_root),
              "--riscv-root", str(riscv_root)]
    subprocess.run([sys.executable, "-m", "tools.compile_object", *common,
                    "--mx-opt", str(mx_opt), "--out-dir", str(obj)],
                   cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "tools.qualify_host_requant_object",
                    "--object-dir", str(obj), *common,
                    "--out-dir", str(spike)], cwd=ROOT, check=True)
    result = json.loads((spike / "index.json").read_text())
    if result["status"] != "source_quantized_output_matched_on_pinned_spike":
        raise ValueError(f"{case.name} did not match its source golden")
    return {
        "case": case.name,
        "profile": profile.name,
        "bound_mlir_sha256": _sha(mlir),
        "bundle_manifest_sha256": _sha(bundle / "manifest.json"),
        "object_sha256": _sha(obj / "mx_issue.o"),
        "object_manifest_sha256": _sha(obj / "object_manifest.json"),
        "elf_sha256": _sha(spike / "mx_host_object.elf"),
        "spike_log_sha256": _sha(spike / "spike.log"),
        "qualifier_index_sha256": _sha(spike / "index.json"),
        "compared_codes": result["compared_codes"],
        "compared_scales": result["compared_scales"],
        "status": result["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "profiles", "rtl-root", "riscv-root", "mx-opt",
                 "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    for name in ("source_root", "profiles", "rtl_root", "riscv_root", "mx_opt",
                 "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    if not 1 <= args.workers <= 4:
        parser.error("workers must be between 1 and 4")
    if not args.mx_opt.is_file():
        parser.error("selected native MX dialect verifier is missing")
    cases = sorted(case for case in args.source_root.glob("*.requant")
                   if (case / "payload_bound.mlir").is_file() and
                   (case / "bundle/manifest.json").is_file())
    if len(cases) != 8:
        raise ValueError(f"current Radiance MX roster needs 8 requant cases; found {len(cases)}")
    args.out_dir.mkdir(parents=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_case, case, args.profiles, args.rtl_root,
                               args.riscv_root, args.mx_opt, args.out_dir): case
                   for case in cases}
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["case"])
    index = {
        "schema": "mx_gemmini.radiance_host_object_roster.v1",
        "status": "all_8_requant_objects_matched_source_goldens_on_pinned_spike",
        "source_roster_index_sha256": _sha(args.source_root / "index.json")
        if (args.source_root / "index.json").is_file() else None,
        "native_verifier_sha256": _sha(args.mx_opt),
        "cases": rows,
        "compared_codes": sum(row["compared_codes"] for row in rows),
        "compared_scales": sum(row["compared_scales"] for row in rows),
    }
    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"{len(rows)} host objects: {index['compared_codes']} codes, "
          f"{index['compared_scales']} scales matched on pinned Spike")


if __name__ == "__main__":
    main()
