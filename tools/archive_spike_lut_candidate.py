"""Archive full legal-mode results from an isolated Spike weight-LUT patch.

Each candidate receipt must match a pinned stock receipt's ELF. All stock
passing output logs must stay identical; only the three known failures may
change. This is simulator diagnostic evidence, not RTL qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence"
REPAIRED = "e4m3s_e4m3"
STOCK_EXTENSION_CLOSURE = (
    "dbbc62171a9393691c575ec278dfdd75ca20fd6a2c9b52825380dad4e5d0af3f")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def stock_receipts() -> list[tuple[Path, dict]]:
    found = []
    for path in EVIDENCE.rglob("*.json"):
        try:
            data = read(path)
        except (ValueError, UnicodeError):
            continue
        if isinstance(data, dict) and all(data.get(key) for key in (
                "source_header_sha256", "profile_sha256", "elf_sha256",
                "spike_log_sha256", "extension_sha256")) and (
                    data.get("source_driver_sha256") or
                    data.get("source_generation_manifest_sha256")) and (
                    data.get("extension_source_closure_sha256",
                             data.get("gemmini_extension_source_closure_sha256")) ==
                    STOCK_EXTENSION_CLOSURE):
            found.append((path, data))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dim8-root", "dim16-root", "dim32-root", "dim16-direct-root",
                 "extension-root", "patch", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    extension = args.extension_root.resolve()
    patch = args.patch.resolve()
    subprocess.run(["git", "-C", str(extension), "apply", "--reverse", "--check",
                    str(patch)], check=True, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE)
    changed = subprocess.check_output(
        ["git", "-C", str(extension), "diff", "--name-only"], text=True).splitlines()
    if changed != ["gemmini.cc"]:
        raise ValueError("candidate extension must change only gemmini.cc")
    actual_patch = subprocess.check_output(
        ["git", "-C", str(extension), "diff", "--", "gemmini.cc"])
    if actual_patch != patch.read_bytes():
        raise ValueError("candidate extension differs from archived patch")
    stock = stock_receipts()
    rows = []
    extensions, compilers, frontends, rtls = set(), set(), set(), set()

    def add(dim: int, suffix: str, receipt_path: Path, log_path: Path,
            *, direct: bool = False) -> None:
        receipt = read(receipt_path)
        if (receipt.get("status") != "source_golden_matched_on_pinned_spike" or
                receipt.get("spike_exit_code") != 0 or
                receipt.get("compared_bf16_outputs") != 4096 or
                sha(log_path) != receipt.get("spike_log_sha256")):
            raise ValueError(f"candidate source parity failed: DIM{dim} {suffix}")
        extensions.add(receipt["gemmini_extension_source_closure_sha256"]
                       if direct else receipt["extension_source_closure_sha256"])
        compilers.add(receipt["compiler_revision"])
        if not direct:
            frontends.add(receipt["model2mlir_revision"])
        rtls.add(receipt["rtl_revision"])
        matches = [(path, data) for path, data in stock
                   if data.get("source_driver_sha256") ==
                   receipt.get("source_driver_sha256")
                   and data.get("source_generation_manifest_sha256") ==
                   receipt.get("source_generation_manifest_sha256")
                   and data.get("source_header_sha256") == receipt["source_header_sha256"]
                   and data.get("profile_sha256") == receipt["profile_sha256"]
                   and (direct or data.get("mesh_dim") == dim)]
        repaired = suffix == REPAIRED
        eligible = [(path, data) for path, data in matches
                    if data.get("status") == ("source_golden_failed_on_pinned_spike"
                                              if repaired else
                                              "source_golden_matched_on_pinned_spike")
                    and data["elf_sha256"] == receipt["elf_sha256"]
                    and ((data["spike_log_sha256"] != receipt["spike_log_sha256"])
                         if repaired else
                         (data["spike_log_sha256"] == receipt["spike_log_sha256"]))]
        if not eligible:
            raise ValueError(f"stock comparison is missing: DIM{dim} {suffix}")
        stock_path, original = sorted(eligible, key=lambda pair: str(pair[0]))[0]
        if original["extension_sha256"] == receipt["extension_sha256"]:
            raise ValueError(f"candidate extension equals stock: DIM{dim} {suffix}")
        prefix = Path(f"dim{dim}") / suffix
        target_receipt, target_log = prefix / "receipt.json", prefix / "spike.log"
        copy(receipt_path, args.out_dir / target_receipt)
        copy(log_path, args.out_dir / target_log)
        rows.append({
            "mesh_dim": dim, "source_suffix": suffix,
            "comparison": "stock_failure_repaired" if repaired else "stock_pass_unchanged",
            "candidate_receipt": str(target_receipt),
            "candidate_receipt_sha256": sha(receipt_path),
            "candidate_log": str(target_log), "candidate_log_sha256": sha(log_path),
            "stock_receipt": str(stock_path.relative_to(ROOT)),
            "stock_receipt_sha256": sha(stock_path),
            "elf_sha256": receipt["elf_sha256"],
        })

    for dim, directory in ((8, args.dim8_root), (16, args.dim16_root),
                           (32, args.dim32_root)):
        matrix_path = directory / "matrix_receipt.json"
        matrix = read(matrix_path)
        expected = 35 if dim == 16 else 36
        if (matrix.get("schema") != "mx_gemmini.nicolas_asymmetric_mode_matrix.v1" or
                matrix.get("mesh_dim") != dim or
                matrix.get("selected_modes") != expected or
                matrix.get("passed_modes") != expected or
                matrix.get("legal_mode_count") != 36 or
                matrix.get("selected_but_failed_compute") or
                bool(matrix.get("profile_complete")) != (dim != 16)):
            raise ValueError(f"candidate DIM{dim} mode matrix is incomplete")
        if dim == 16:
            missing = matrix["uncovered_legal_compute"]
            direct_mode = {
                "activation_format": "fp8_e4m3", "activation_projection": "direct",
                "weight_format": "fp8_e4m3", "weight_projection": "direct",
                "pe_mode": 8,
            }
            if len(missing) != 1 or missing[0]["compute"] != direct_mode:
                raise ValueError("DIM16 matrix must omit only direct E4M3")
        elif matrix["uncovered_legal_compute"]:
            raise ValueError(f"DIM{dim} matrix leaves legal modes uncovered")
        copy(matrix_path, args.out_dir / f"dim{dim}/matrix_receipt.json")
        for row in matrix["rows"]:
            suffix = row["source_suffix"]
            receipt = directory / suffix / "receipt.json"
            if row["receipt_sha256"] != sha(receipt):
                raise ValueError(f"matrix receipt hash differs: DIM{dim} {suffix}")
            add(dim, suffix, receipt, directory / suffix / "physical/spike.log")

    direct = args.dim16_direct_root
    capture_path = direct / "capture/receipt.json"
    bound = direct / "capture/mx_gemm.profile_bound.mlir"
    capture = read(capture_path)
    if (capture.get("model2mlir_revision") not in frontends or
            capture.get("target_binding", {}).get("bound_mlir_sha256") != sha(bound) or
            any(fragment not in bound.read_text() for fragment in (
                'activation_format = "fp8_e4m3"', 'weight_format = "fp8_e4m3"',
                'activation_projection = "direct"', 'weight_projection = "direct"',
                'pe_mode = 8 : i32'))):
        raise ValueError("DIM16 direct E4M3 frontend does not cover the missing mode")
    copy(capture_path, args.out_dir / "dim16/direct_e4m3/capture_receipt.json")
    copy(bound, args.out_dir / "dim16/direct_e4m3/profile_bound.mlir")
    add(16, "direct_e4m3", direct / "spike/build/artifact_manifest.json",
        direct / "spike/build/spike.log", direct=True)
    if (len(rows) != 108 or
            sum(row["comparison"] == "stock_pass_unchanged" for row in rows) != 105 or
            sum(row["comparison"] == "stock_failure_repaired" for row in rows) != 3 or
            any(len(values) != 1 for values in (extensions, compilers, frontends, rtls))):
        raise ValueError("candidate matrix does not cover exactly 108 stable modes")
    index = {
        "schema": "mx_gemmini.spike_weight_lut_candidate_all_modes.v1",
        "scope": "isolated patched Spike extension; stock Spike and RTL qualification remain separate",
        "rtl_revision": next(iter(rtls)),
        "compiler_revision": next(iter(compilers)),
        "model2mlir_revision": next(iter(frontends)),
        "spike_extension_revision": subprocess.check_output(
            ["git", "-C", str(extension), "rev-parse", "HEAD"], text=True).strip(),
        "patched_gemmini_cc_sha256": sha(extension / "gemmini.cc"),
        "extension_source_closure_sha256": next(iter(extensions)),
        "patch": str(patch.relative_to(ROOT)), "patch_sha256": sha(patch),
        "legal_mode_count_per_mesh": 36,
        "candidate_passes": 108, "stock_passes_unchanged": 105,
        "stock_failures_repaired_in_candidate": 3, "rows": rows,
    }
    (args.out_dir / "qualification.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"archived 108 candidate Spike mode receipts -> {args.out_dir}")


if __name__ == "__main__":
    main()
