"""Qualify the typed fused EXPSUM flash object against Nicolas's source on Spike."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

from mx_gemmini_support.nicolas_flash_mid import render_flash_mid
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision
from tools.flash_spike_parity import run_flash_spike_parity
from tools.qualify_nicolas_flash_qk0 import DATA_SOURCE, FLASH_SOURCE, _run, _sha


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_flash_mid_expsum_typed_a042643_0344bcb"
WRAPPER = ROOT / "tools/fixtures/nicolas_flash_mid_expsum.c"
DRIVER = ROOT / "tools/fixtures/nicolas_flash_mid_serial_driver.c"
EXPECTED_HASH = "cd7aef4d68d1e0d9"
WRAPPER_SHA256 = "549b0440b05b149c84059985f3b20f55338da7af879d8272557f662cb263824d"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("npu-root", "profile", "rtl-root", "riscv-root",
                 "spike-extension", "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    for name in ("npu_root", "profile", "rtl_root", "riscv_root",
                 "spike_extension", "mx_opt", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    baseline = json.loads((FLASH_SOURCE / "receipt.json").read_text())
    data_baseline = json.loads((DATA_SOURCE / "receipt.json").read_text())
    core = args.npu_root / "baremetal/mxgemmini/src/attn_flash.c"
    frontend = DATA_SOURCE / "attention.model2mlir.mlir"
    header_archive = DATA_SOURCE / "attn_vpu_mid.h.gz"
    expect = FLASH_SOURCE / "attn_flash_mid_expect.h"
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    header = gzip.decompress(header_archive.read_bytes())
    if (_git_revision(args.npu_root) != baseline["npu_revision"] or
            _git_revision(args.rtl_root) != baseline["rtl_revision"] or
            profile_sha256(profile) != data_baseline["profile_sha256"] or
            _sha(core) != baseline["source_core_sha256"] or
            _sha(frontend) != data_baseline["frontend_mlir_sha256"] or
            _sha(header_archive) != data_baseline["source_header_gzip_sha256"] or
            hashlib.sha256(header).hexdigest() != data_baseline["source_header_sha256"] or
            _sha(expect) != baseline["expect_header_sha256"] or
            _sha(WRAPPER) != WRAPPER_SHA256):
        raise ValueError("pinned EXPSUM flash source, frontend, RTL, or profile differs")
    cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    spike = args.riscv_root / "bin/spike"
    if (_sha(cc) != data_baseline["riscv_gcc_sha256"] or
            _sha(spike) != data_baseline["spike_binary_sha256"] or
            _sha(args.spike_extension) != data_baseline["spike_extension_sha256"]):
        raise ValueError("pinned RV64 toolchain or Spike extension differs")
    mlir, binding = render_flash_mid(
        frontend.read_text(), core.read_text(), WRAPPER.read_text(),
        header.decode(), profile, softmax_mode="expsum")
    if (mlir != (ARCHIVE / "flash_mid_expsum.mlir").read_text() or
            binding != json.loads((ARCHIVE / "binding.json").read_text())):
        raise ValueError("archived EXPSUM flash graph or binding differs")

    out = args.out_dir
    out.mkdir(parents=True)
    (out / "attn_vpu_mid.h").write_bytes(header)
    shutil.copyfile(expect, out / expect.name)
    shutil.copyfile(WRAPPER, out / "attn_flash_mid.c")
    shutil.copyfile(DRIVER, out / "driver.c")
    object_dir = out / "object"
    _run([sys.executable, "-m", "tools.compile_object",
          "--mlir", str(ARCHIVE / "flash_mid_expsum.mlir"),
          "--precursor-mlir", str(frontend),
          "--binding-json", str(ARCHIVE / "binding.json"),
          "--source-core", str(core), "--source-wrapper", str(WRAPPER),
          "--source-header", str(header_archive),
          "--profile", str(args.profile), "--rtl-root", str(args.rtl_root),
          "--riscv-root", str(args.riscv_root), "--mx-opt", str(args.mx_opt),
          "--out-dir", str(object_dir)], cwd=ROOT, log=out / "compile_object.log")
    object_manifest = json.loads((object_dir / "object_manifest.json").read_text())
    if (object_manifest["softmax_mode"] != "expsum" or
            (object_manifest["command_count"], object_manifest["fence_count"]) !=
            (148, 21) or object_manifest["allocated_data_section_bytes"] != 0):
        raise ValueError("EXPSUM flash object inventory or data section differs")
    run_flash_spike_parity(
        out=out, npu_root=args.npu_root, rtl_root=args.rtl_root,
        riscv_root=args.riscv_root, spike_extension=args.spike_extension,
        object_path=object_dir / "mx_issue.o", slug="flash_mid_expsum",
        expected_hash=EXPECTED_HASH, output_words=4096)
    receipt = {
        "schema": "mx_gemmini.flash_mid_expsum_serial_spike_parity.v1",
        "status": "typed_flash_mid_expsum_full_bf16_source_parity",
        "source_baseline_sha256": _sha(FLASH_SOURCE / "receipt.json"),
        "frontend_baseline_sha256": _sha(DATA_SOURCE / "receipt.json"),
        "npu_revision": _git_revision(args.npu_root),
        "rtl_revision": _git_revision(args.rtl_root),
        "model2mlir_revision": data_baseline["model2mlir_revision"],
        "source_core_sha256": _sha(core),
        "source_wrapper_sha256": _sha(WRAPPER),
        "source_header_sha256": data_baseline["source_header_sha256"],
        "source_header_gzip_sha256": _sha(header_archive),
        "expected_output_hash": EXPECTED_HASH,
        "compared_bf16_words": 4096, "source_runs": 2, "compiler_runs": 2,
        "bound_mlir_sha256": _sha(ARCHIVE / "flash_mid_expsum.mlir"),
        "binding_json_sha256": _sha(ARCHIVE / "binding.json"),
        "object_sha256": _sha(object_dir / "mx_issue.o"),
        "object_manifest_sha256": _sha(object_dir / "object_manifest.json"),
        "compile_manifest_sha256": _sha(object_dir / "compile_manifest.json"),
        "physical_program_sha256": _sha(object_dir / "physical_program.json"),
        "source_elf_sha256": _sha(out / "flash_mid_expsum_source.elf"),
        "generated_elf_sha256": _sha(out / "flash_mid_expsum_generated.elf"),
        "source_log_sha256": _sha(out / "source_spike_1.log"),
        "generated_log_sha256": _sha(out / "generated_spike_1.log"),
        "native_verifier_sha256": _sha(args.mx_opt),
        "spike_extension_sha256": _sha(args.spike_extension),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("typed EXPSUM flash: all 4,096 BF16 outputs match source twice on Spike")


if __name__ == "__main__":
    main()
