"""Replay padded FP4 MLP projections from the pinned small LLaMA capture.

This checks isolated, complete rank-two projections. It does not execute the
other MLP operators or a full inference session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = (ROOT / "docs/evidence/"
           "model2mlir_small_llama_mlp_fp4_padding_e9ded36_266c593")
REMAINING_ARCHIVE = (ROOT / "docs/evidence/"
                     "model2mlir_small_llama_remaining_mlp_fp4_e9ded36_266c593")
GRAPH = (ROOT / "docs/evidence/"
         "model2mlir_small_llama_precision_e9ded36_266c593/"
         "small_llama.model2mlir.mlir.gz")
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
SITES = {"g": ("matmul_6", 344, 128), "dn": ("matmul_8", 128, 32)}
REMAINING_SITES = {"b0.u": ("matmul_7", 344, 128),
                   "b1.g": ("matmul_15", 344, 128),
                   "b1.u": ("matmul_16", 344, 128),
                   "b1.dn": ("matmul_17", 128, 32)}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model2mlir-root", "mxq-root", "reference-root", "rtl-root",
                 "riscv-root", "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--remaining-mlp", action="store_true",
                        help="replay the other four padded MLP sites")
    args = parser.parse_args()
    for name in ("model2mlir_root", "mxq_root", "reference_root", "rtl_root",
                 "riscv_root", "mx_opt", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    archive = REMAINING_ARCHIVE if args.remaining_mlp else ARCHIVE
    sites = REMAINING_SITES if args.remaining_mlp else SITES
    args.out_dir.mkdir(parents=True)
    inputs = args.out_dir / "inputs"
    capture = [sys.executable, "-m", "tools.capture_small_llama_mlp_inputs",
               "--model2mlir-root", str(args.model2mlir_root),
               "--out-dir", str(inputs)]
    if args.remaining_mlp:
        capture.append("--all-mlp")
    _run(capture)
    input_names = ["input_ids.npy", "manifest.json"]
    for name in (("b0.g", "b0.u", "b0.dn", "b1.g", "b1.u", "b1.dn")
                 if args.remaining_mlp else ("g", "dn")):
        input_names.extend((f"{name}.activation.npy", f"{name}.weight.npy"))
    for name in input_names:
        if _sha(inputs / name) != _sha(archive / "inputs" / name):
            raise ValueError(f"model input replay differs: {name}")
    results = {}
    for name, (region, columns, tile_k) in sites.items():
        site_out = args.out_dir / name
        _run([sys.executable, "-m", "tools.qualify_model_projection",
              "--full-model-mlir", str(GRAPH),
              "--activation-npy", str(inputs / f"{name}.activation.npy"),
              "--weight-npy", str(inputs / f"{name}.weight.npy"),
              "--model2mlir-root", str(args.model2mlir_root),
              "--mxq-root", str(args.mxq_root),
              "--reference-root", str(args.reference_root),
              "--profile", str(PROFILE), "--rtl-root", str(args.rtl_root),
              "--riscv-root", str(args.riscv_root),
              "--mx-opt", str(args.mx_opt), "--out-dir", str(site_out),
              "--site-region", region, "--row-count", "8",
              "--column-count", str(columns), "--tile-k", str(tile_k),
              "--precision", "fp4", "--run-spike"])
        archived = archive / name
        for relative in ("projection.model2mlir.mlir", "projection.payload_bound.mlir",
                         "bundle/manifest.json", "bundle/golden_bf16.bin",
                         "object/mx_issue.o", "run/mx_program.elf"):
            if _sha(site_out / relative) != _sha(archived / relative):
                raise ValueError(f"replay differs from archive: {name}/{relative}")
        expected = json.loads((archived / "summary.json").read_text())
        actual = json.loads((site_out / "summary.json").read_text())
        # The receipt also records this checkout's compiler source closure and
        # linker log. Those change when the replay tool or output path changes.
        expected.pop("run_receipt_sha256")
        actual.pop("run_receipt_sha256")
        if actual != expected:
            raise ValueError(f"projection summary differs from archive: {name}")
        log = (site_out / "run/spike.log").read_text()
        m, n, k = actual["padded_shape_mnk"]
        if f"lowered MX {m}x{n}x{k}: 0 BF16 mismatches" not in log:
            raise ValueError(f"Spike did not match every BF16 output: {name}")
        results[name] = {
            "region_id": region, "padded_shape_mnk": [m, n, k],
            "compared_bf16_values": actual["compared_bf16_values"],
            "model_output_values": actual["model_output_values"],
            "object_sha256": actual["object_sha256"],
            "elf_sha256": actual["elf_sha256"],
        }
    if args.remaining_mlp:
        status = "four_remaining_projections_matched_on_pinned_spike"
        scope = ("layer-0 up and layer-1 gate/up/down rank-two matmuls; "
                 "other model operators uncompiled")
    else:
        status = "two_complete_projections_matched_on_pinned_spike"
        scope = ("layer-0 MLP gate and down rank-two matmuls; "
                 "other model operators uncompiled")
    receipt = {"schema": "mx_gemmini.small_llama_mlp_fp4_padding_replay.v1",
               "status": status, "scope": scope,
               "input_manifest_sha256": _sha(inputs / "manifest.json"),
               "source_model_mlir_sha256": _sha(GRAPH),
               "sites": results}
    (args.out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2,
                                                         sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
