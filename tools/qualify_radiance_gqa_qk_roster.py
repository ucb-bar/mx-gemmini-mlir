"""Run every used GQA QK head/block tile from a generated fixture on Nicolas's Spike.

First run qualify_radiance_gqa_qk --generated-hardware-model. This command
reuses its pinned model2MLIR capture and source-generated header, then compiles
the 8 query heads x 2 causally used key blocks as separate MX programs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from mx_gemmini_support.bind_payload import bind_payload
from mx_gemmini_support.source_attention_qk import (
    HARDWARE_SHIFT, SUBMODULE_REVISION, derive_generated_hardware_qk,
    read_gqa_qk_tile)
from mx_gemmini_support.source_fp6 import _bytes
from mx_gemmini_support.source_gemm import SourceGemm
from mx_gemmini_support.source_payload import (
    ATTENTION_QK_CANDIDATE_ORIGIN, make_manifest)
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.verify_profile_ir import verify_ir
from tools.qualify_radiance_gqa_qk import HARDWARE_PATCH, PROFILE
from tools.qualify_radiance_ws_roster import _revision


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _run(command: list[str], *, log: Path) -> None:
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, check=False)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-dir", "source-root", "rtl-root", "riscv-root",
                 "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--baseline-index", type=Path)
    args = parser.parse_args()
    for name in ("prepared_dir", "source_root", "rtl_root", "riscv_root",
                 "mx_opt", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    if not 1 <= args.jobs <= 8:
        parser.error("--jobs must be in 1..8")
    prepared = json.loads((args.prepared_dir / "index.json").read_text())
    if (prepared.get("status") != "generated_source_qk_candidate_matched_on_pinned_spike" or
            prepared.get("source_revision") != _revision(args.source_root) or
            prepared.get("submodule_revision") != _revision(args.source_root / "lib/mxgemmini") or
            prepared["submodule_revision"] != SUBMODULE_REVISION or
            prepared.get("source_header_sha256") !=
            _sha(args.prepared_dir / "patched_source/kernels/flash_attention_mx_gqa/include/fa_data.h") or
            prepared.get("generator_patch_sha256") != _sha(HARDWARE_PATCH) or
            prepared.get("compared_bf16_outputs") != 4096):
        raise ValueError("prepared GQA fixture differs from pinned generator qualification")
    patched_root = args.prepared_dir / "patched_source"
    directory = patched_root / "kernels/flash_attention_mx_gqa"
    profile = load_profile(PROFILE, rtl_root=args.rtl_root)
    profile_hash = profile_sha256(profile)
    profile_bound = (args.prepared_dir / "frontend/profile_bound.mlir").read_text()
    if _sha(args.prepared_dir / "frontend/model2mlir.mlir") != prepared["frontend_mlir_sha256"]:
        raise ValueError("prepared model2MLIR capture changed")

    sys.path[:0] = [str(directory), str(args.source_root / "lib/mxgemmini")]
    import torch
    import fp8_matmul_model as low_level_model
    import flash_attention_model as fa_model
    if (Path(low_level_model.__file__).resolve() !=
            (args.source_root / "lib/mxgemmini/fp8_matmul_model.py").resolve() or
            Path(fa_model.__file__).resolve() !=
            (directory / "flash_attention_model.py").resolve()):
        raise ValueError("GQA numerical model resolved to another checkout")
    fa_model.MX_HARDWARE_SHIFT = HARDWARE_SHIFT
    q, k, _ = fa_model.make_inputs_gqa(64, 256, 64, 8, 2, 0)
    args.out_dir.mkdir(parents=True)
    cases = []
    for head in range(8):
        for block in range(2):
            tile = read_gqa_qk_tile(patched_root, head=head, block=block,
                                    hardware_generated=True)
            resources, policy = derive_generated_hardware_qk(
                tile, torch=torch, low_level_model=low_level_model,
                patch_sha256=_sha(HARDWARE_PATCH))
            qa, qs = fa_model.mx_quantize_cols(q[head])
            kb, ks = fa_model.mx_quantize_cols(k[head // 4, block * 64:(block + 1) * 64])
            direct = fa_model.mx_gemm(qa, qs, kb.t().contiguous(), ks.t().contiguous())
            codes, bits = low_level_model.tensor_to_custom_fp_codes(direct, "bf16")
            if (bits != 16 or _bytes(tuple(c for row in codes for c in row), 2) !=
                    resources["golden_bf16"].data):
                raise ValueError(f"head {head} block {block} disagrees with source model")
            manifest = make_manifest(
                SourceGemm(tile.driver, tile.header, (64, 64, 64), (64, 64, 64),
                           "FP8", False, False, True), resources,
                site_id="functional:matmul", profile_sha256=profile_hash)
            manifest["origin"] = ATTENTION_QK_CANDIDATE_ORIGIN
            manifest["source_derivation"] = policy
            bound = bind_payload(profile_bound, profile, manifest)
            verify_ir(bound, profile)
            case = args.out_dir / f"head{head}_block{block}"
            bundle = case / "bundle"
            bundle.mkdir(parents=True)
            for name, resource in resources.items():
                (bundle / f"{name}.bin").write_bytes(resource.data)
            _write(bundle / "manifest.json", manifest)
            mlir = case / "payload_bound.mlir"
            mlir.write_text(bound)
            _run([str(args.mx_opt), str(mlir), "-o", "/dev/null"],
                 log=case / "mx_parse.log")
            cases.append((head, block, case, tile.hashes(),
                          _sha(bundle / "golden_bf16.bin")))

    def compile_case(item):
        head, block, case, hashes, golden_hash = item
        build = case / "build"
        _run([sys.executable, "-m", "tools.compile_mx", "--mlir",
              str(case / "payload_bound.mlir"), "--bundle", str(case / "bundle"),
              "--profile", str(PROFILE), "--rtl-root", str(args.rtl_root),
              "--riscv-root", str(args.riscv_root), "--out-dir", str(build),
              "--run-spike"], log=case / "compile.log")
        receipt = json.loads((build / "artifact_manifest.json").read_text())
        if (receipt.get("status") != "source_golden_matched_on_pinned_spike" or
                receipt.get("compared_bf16_outputs") != 4096 or
                receipt.get("spike_exit_code") != 0 or
                receipt.get("source_header_sha256") != prepared["source_header_sha256"]):
            raise ValueError(f"GQA QK head {head} block {block} failed Spike qualification")
        program = json.loads((build / "physical_program.json").read_text())
        if program.get("shape_mnk") != [64, 64, 64] or len(program.get("steps", [])) != 83:
            raise ValueError(f"GQA QK head {head} block {block} changed physical schedule")
        return {
            "head": head, "block": block, "kv_head": head // 4,
            "source_arrays_sha256": hashes, "golden_bf16_sha256": golden_hash,
            "payload_bound_mlir_sha256": _sha(case / "payload_bound.mlir"),
            "physical_program_sha256": _sha(build / "physical_program.json"),
            "physical_steps": len(program["steps"]),
            "elf_sha256": receipt["elf_sha256"],
            "spike_log_sha256": receipt["spike_log_sha256"],
            "compared_bf16_outputs": receipt["compared_bf16_outputs"],
            "receipt": str((build / "artifact_manifest.json").relative_to(args.out_dir)),
        }

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(compile_case, cases))
    if (len(rows) != 16 or
            sum(row["compared_bf16_outputs"] for row in rows) != 65536 or
            len({row["source_arrays_sha256"]["activation"] for row in rows}) != 8 or
            len({row["source_arrays_sha256"]["weight"] for row in rows}) != 4):
        raise ValueError("GQA QK roster lacks expected query/KV sharing")
    index = {
        "schema": "mx_gemmini.gqa_qk_roster.v1",
        "status": "all_16_generated_source_qk_tiles_matched_on_pinned_spike",
        "scope": "QK stages only; no causal softmax, P requantization, PV, Muon scheduling, or final O parity",
        "source_revision": prepared["source_revision"],
        "source_header_sha256": prepared["source_header_sha256"],
        "generator_patch_sha256": prepared["generator_patch_sha256"],
        "prepared_index_sha256": _sha(args.prepared_dir / "index.json"),
        "frontend_mlir_sha256": prepared["frontend_mlir_sha256"],
        "profile_sha256": profile_hash,
        "rtl_revision": _revision(args.rtl_root),
        "compiler_revision": _revision(ROOT),
        "tile_count": 16, "compared_bf16_outputs": 65536,
        "rows": rows,
    }
    if args.baseline_index and json.loads(args.baseline_index.read_text()) != index:
        raise ValueError("GQA QK roster differs from pinned baseline")
    _write(args.out_dir / "index.json", index)
    print("qualified 16 generated GQA QK tiles: 65,536 BF16 outputs on Nicolas's Spike")


if __name__ == "__main__":
    main()
