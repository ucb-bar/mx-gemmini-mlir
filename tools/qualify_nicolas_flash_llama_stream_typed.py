"""Qualify the typed 2,048-key streamed flash object against source on Spike."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

from mx_gemmini_support.nicolas_flash_mid import STREAM_SHAPE, render_flash_mid
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision
from tools.flash_spike_parity import run_flash_spike_parity
from tools.qualify_nicolas_flash_qk0 import _run, _sha


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/evidence/nicolas_flash_llama_stream_source_baseline_e9ded36_0344bcb"
ARCHIVE = ROOT / "docs/evidence/nicolas_flash_llama_stream_typed_e9ded36_0344bcb"
DRIVER = ROOT / "tools/fixtures/nicolas_flash_llama_parity_driver.c"
EXPECTED_HASH = "dc33d078bc537b19"


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
    baseline = json.loads((SOURCE / "receipt.json").read_text())
    npu = args.npu_root / "baremetal/mxgemmini"
    core, wrapper = npu / "src/attn_flash.c", npu / "src/attn_flash_llama.c"
    frontend = SOURCE / "attention.model2mlir.mlir"
    header_archive = SOURCE / "attn_vpu_llama.h.gz"
    expect = SOURCE / "attn_flash_llama_expect.h"
    header = gzip.decompress(header_archive.read_bytes())
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    if (_git_revision(args.npu_root) != baseline["npu_revision"] or
            _git_revision(args.rtl_root) != baseline["rtl_revision"] or
            profile_sha256(profile) != baseline["profile_sha256"] or
            _sha(core) != baseline["source_core_sha256"] or
            _sha(wrapper) != baseline["source_wrapper_sha256"] or
            _sha(frontend) != baseline["frontend_mlir_sha256"] or
            _sha(header_archive) != baseline["source_header_gzip_sha256"] or
            hashlib.sha256(header).hexdigest() != baseline["source_header_sha256"] or
            _sha(expect) != baseline["source_expect_sha256"] or
            baseline["source_output_hash"] != EXPECTED_HASH):
        raise ValueError("pinned streamed flash source, frontend, RTL, or profile differs")
    cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    spike = args.riscv_root / "bin/spike"
    if (_sha(cc) != baseline["riscv_gcc_sha256"] or
            _sha(spike) != baseline["spike_binary_sha256"] or
            _sha(args.spike_extension) != baseline["spike_extension_sha256"]):
        raise ValueError("pinned RV64 toolchain or Spike extension differs")
    mlir, binding = render_flash_mid(
        frontend.read_text(), core.read_text(), wrapper.read_text(),
        header.decode(), profile, shape=STREAM_SHAPE, block_k=256)
    if (mlir != (ARCHIVE / "flash_llama_stream.mlir").read_text() or
            binding != json.loads((ARCHIVE / "binding.json").read_text())):
        raise ValueError("archived streamed flash graph or binding differs")

    out = args.out_dir
    out.mkdir(parents=True)
    (out / "attn_vpu_llama.h").write_bytes(header)
    shutil.copyfile(expect, out / "attn_flash_llama_expect.h")
    shutil.copyfile(DRIVER, out / "driver.c")
    object_dir = out / "object"
    _run([sys.executable, "-m", "tools.compile_object",
          "--mlir", str(ARCHIVE / "flash_llama_stream.mlir"),
          "--precursor-mlir", str(frontend),
          "--binding-json", str(ARCHIVE / "binding.json"),
          "--source-core", str(core), "--source-wrapper", str(wrapper),
          "--source-header", str(header_archive),
          "--profile", str(args.profile), "--rtl-root", str(args.rtl_root),
          "--riscv-root", str(args.riscv_root), "--mx-opt", str(args.mx_opt),
          "--out-dir", str(object_dir)], cwd=ROOT, log=out / "compile_object.log")
    manifest = json.loads((object_dir / "object_manifest.json").read_text())
    if (manifest["shape_sq_sk_d_bk"] != [64, 2048, 64, 256] or
            manifest["input_format"] != "fp8_e4m3" or
            (manifest["command_count"], manifest["fence_count"]) != (484, 50) or
            manifest["allocated_data_section_bytes"] != 0):
        raise ValueError("streamed flash object inventory or data section differs")
    run_flash_spike_parity(
        out=out, npu_root=args.npu_root, rtl_root=args.rtl_root,
        riscv_root=args.riscv_root, spike_extension=args.spike_extension,
        object_path=object_dir / "mx_issue.o", slug="flash_llama_stream",
        expected_hash=EXPECTED_HASH, output_words=4096)
    receipt = {
        "schema": "mx_gemmini.flash_llama_stream_serial_spike_parity.v1",
        "status": "typed_streamed_flash_full_bf16_source_parity",
        "source_baseline_sha256": _sha(SOURCE / "receipt.json"),
        "model2mlir_revision": baseline["model2mlir_revision"],
        "frontend_mlir_sha256": _sha(frontend),
        "npu_revision": _git_revision(args.npu_root),
        "rtl_revision": _git_revision(args.rtl_root),
        "profile_sha256": profile_sha256(profile),
        "source_core_sha256": _sha(core),
        "source_wrapper_sha256": _sha(wrapper),
        "source_header_sha256": baseline["source_header_sha256"],
        "source_header_gzip_sha256": _sha(header_archive),
        "expected_output_hash": EXPECTED_HASH,
        "compared_bf16_words": 4096, "source_runs": 2, "compiler_runs": 2,
        "bound_mlir_sha256": _sha(ARCHIVE / "flash_llama_stream.mlir"),
        "binding_json_sha256": _sha(ARCHIVE / "binding.json"),
        "object_sha256": _sha(object_dir / "mx_issue.o"),
        "object_manifest_sha256": _sha(object_dir / "object_manifest.json"),
        "compile_manifest_sha256": _sha(object_dir / "compile_manifest.json"),
        "physical_program_sha256": _sha(object_dir / "physical_program.json"),
        "source_elf_sha256": _sha(out / "flash_llama_stream_source.elf"),
        "generated_elf_sha256": _sha(out / "flash_llama_stream_generated.elf"),
        "source_log_sha256": _sha(out / "source_spike_1.log"),
        "generated_log_sha256": _sha(out / "generated_spike_1.log"),
        "driver_sha256": _sha(DRIVER),
        "native_verifier_sha256": _sha(args.mx_opt),
        "spike_extension_sha256": _sha(args.spike_extension),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("typed streamed flash: all 4,096 BF16 outputs match source twice on Spike")


if __name__ == "__main__":
    main()
