"""Compare split or fused typed four-block FP4 flash with source on Spike."""

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
ARCHIVE = ROOT / "docs/evidence/nicolas_flash_mid_fp4_typed_a042643_0344bcb"
WRAPPER = ROOT / "tools/fixtures/nicolas_flash_mid_fp4.c"
EXPSUM_ARCHIVE = ROOT / "docs/evidence/nicolas_flash_mid_fp4_expsum_typed_a042643_0344bcb"
EXPSUM_WRAPPER = ROOT / "tools/fixtures/nicolas_flash_mid_fp4_expsum.c"
DRIVER = ROOT / "tools/fixtures/nicolas_flash_mid_fp4_serial_driver.c"
FRONTEND = DATA_SOURCE / "attention.model2mlir.mlir"
EXPECTED_HASH = "f0caacc02eab4345"
HEADER_SHA256 = "0d7b01496bd4c0ad6f7fe49bccbb0cb8f97e5f2acdf210bf207b0de808344678"
EXPECT_SHA256 = "81e64cbff6b6d32286b5fcd2a45c5319a3203bcf288010f356429f393c9ce447"
WRAPPER_SHA256 = "650ceb3c3b77cfd5b91521b879921debe9e35832fe1d1c5435ad09ee1fb6d6c0"
EXPSUM_WRAPPER_SHA256 = "fca0f83b3a81ecb5210bc69a08f2329bcb70447441b2e02b88e5388e7aa9218c"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--softmax-mode", choices=("split", "expsum"), default="split")
    for name in ("npu-root", "profile", "rtl-root", "riscv-root",
                 "spike-extension", "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    for name in ("npu_root", "profile", "rtl_root", "riscv_root",
                 "spike_extension", "mx_opt", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    fused = args.softmax_mode == "expsum"
    archive = EXPSUM_ARCHIVE if fused else ARCHIVE
    wrapper = EXPSUM_WRAPPER if fused else WRAPPER
    slug = "flash_mid_fp4_expsum" if fused else "flash_mid_fp4"
    command_count = 146 if fused else 154
    baseline = json.loads((FLASH_SOURCE / "receipt.json").read_text())
    data_baseline = json.loads((DATA_SOURCE / "receipt.json").read_text())
    core = args.npu_root / "baremetal/mxgemmini/src/attn_flash.c"
    header_archive = archive / "attn_vpu_mid_fp4.h.gz"
    expect = archive / "attn_flash_mid_fp4_expect.h"
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    if (_git_revision(args.npu_root) != baseline["npu_revision"] or
            _git_revision(args.rtl_root) != baseline["rtl_revision"] or
            profile_sha256(profile) != data_baseline["profile_sha256"] or
            _sha(core) != baseline["source_core_sha256"] or
            _sha(FRONTEND) != data_baseline["frontend_mlir_sha256"] or
            _sha(wrapper) != (EXPSUM_WRAPPER_SHA256 if fused else WRAPPER_SHA256) or
            _sha(expect) != EXPECT_SHA256 or
            hashlib.sha256(gzip.decompress(header_archive.read_bytes())).hexdigest()
            != HEADER_SHA256):
        raise ValueError("pinned FP4 flash source, data, frontend, RTL, or profile differs")
    cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    spike = args.riscv_root / "bin/spike"
    if (_sha(cc) != baseline["riscv_gcc_sha256"] or
            _sha(spike) != baseline["spike_binary_sha256"] or
            _sha(args.spike_extension) != baseline["spike_extension_sha256"]):
        raise ValueError("pinned RV64 toolchain or Spike extension differs")
    mlir, binding = render_flash_mid(
        FRONTEND.read_text(), core.read_text(), wrapper.read_text(),
        gzip.decompress(header_archive.read_bytes()).decode(), profile,
        input_format="fp4_e2m1", softmax_mode=args.softmax_mode)
    if (mlir != (archive / f"{slug}.mlir").read_text() or
            binding != json.loads((archive / "binding.json").read_text())):
        raise ValueError("archived FP4 flash graph or binding differs")

    out = args.out_dir
    out.mkdir(parents=True)
    (out / "attn_vpu_mid_fp4.h").write_bytes(gzip.decompress(header_archive.read_bytes()))
    shutil.copyfile(expect, out / expect.name)
    shutil.copyfile(wrapper, out / "nicolas_flash_mid_fp4.c")
    shutil.copyfile(DRIVER, out / "driver.c")
    object_dir = out / "object"
    _run([sys.executable, "-m", "tools.compile_object",
          "--mlir", str(archive / f"{slug}.mlir"),
          "--precursor-mlir", str(FRONTEND),
          "--binding-json", str(archive / "binding.json"),
          "--source-core", str(core), "--source-wrapper", str(wrapper),
          "--source-header", str(header_archive),
          "--profile", str(args.profile), "--rtl-root", str(args.rtl_root),
          "--riscv-root", str(args.riscv_root), "--mx-opt", str(args.mx_opt),
          "--out-dir", str(object_dir)], cwd=ROOT, log=out / "compile_object.log")
    object_manifest = json.loads((object_dir / "object_manifest.json").read_text())
    if (object_manifest["input_format"] != "fp4_e2m1" or
            object_manifest["softmax_mode"] != args.softmax_mode or
            (object_manifest["command_count"], object_manifest["fence_count"]) !=
            (command_count, 21) or object_manifest["allocated_data_section_bytes"] != 0):
        raise ValueError("FP4 flash object inventory or data section differs")

    run_flash_spike_parity(
        out=out, npu_root=args.npu_root, rtl_root=args.rtl_root,
        riscv_root=args.riscv_root, spike_extension=args.spike_extension,
        object_path=object_dir / "mx_issue.o", slug=slug,
        expected_hash=EXPECTED_HASH, output_words=4096)
    receipt = {
        "schema": f"mx_gemmini.{slug}_serial_spike_parity.v1",
        "status": f"typed_{slug}_full_bf16_source_parity",
        "source_baseline_sha256": _sha(FLASH_SOURCE / "receipt.json"),
        "frontend_baseline_sha256": _sha(DATA_SOURCE / "receipt.json"),
        "npu_revision": _git_revision(args.npu_root),
        "rtl_revision": _git_revision(args.rtl_root),
        "model2mlir_revision": data_baseline["model2mlir_revision"],
        "source_core_sha256": _sha(core),
        "source_wrapper_sha256": _sha(wrapper),
        "source_header_sha256": HEADER_SHA256,
        "source_header_gzip_sha256": _sha(header_archive),
        "source_generator_sha256": _sha(args.npu_root / "baremetal/mxgemmini/gen/gen_attn_fp4.py"),
        "golden_model_sha256": _sha(args.npu_root / "baremetal/mxgemmini/gen/golden_attn_flash.py"),
        "expected_output_hash": EXPECTED_HASH,
        "compared_bf16_words": 4096, "source_runs": 2, "compiler_runs": 2,
        "bound_mlir_sha256": _sha(archive / f"{slug}.mlir"),
        "binding_json_sha256": _sha(archive / "binding.json"),
        "object_sha256": _sha(object_dir / "mx_issue.o"),
        "object_manifest_sha256": _sha(object_dir / "object_manifest.json"),
        "compile_manifest_sha256": _sha(object_dir / "compile_manifest.json"),
        "physical_program_sha256": _sha(object_dir / "physical_program.json"),
        "source_elf_sha256": _sha(out / f"{slug}_source.elf"),
        "generated_elf_sha256": _sha(out / f"{slug}_generated.elf"),
        "source_log_sha256": _sha(out / "source_spike_1.log"),
        "generated_log_sha256": _sha(out / "generated_spike_1.log"),
        "native_verifier_sha256": _sha(args.mx_opt),
        "spike_extension_sha256": _sha(args.spike_extension),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"typed FP4 {args.softmax_mode} flash: all 4,096 BF16 outputs "
          "match source twice on Spike")


if __name__ == "__main__":
    main()
