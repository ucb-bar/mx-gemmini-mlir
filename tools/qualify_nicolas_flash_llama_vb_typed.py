"""Replay the reference implementation’s 64+7×256+192 streamed flash against typed MX on Spike."""

from __future__ import annotations

import argparse
import gzip
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
ARCHIVE = ROOT / "docs/evidence/nicolas_flash_llama_vb_stream_typed_e9ded36_0344bcb"
EXPECTED_HASH = "747ca99b19c32066"
WRAPPER_HASH = "945c3e655e017d62a595c59b4ae0c0ced52bda550df570d30ac01801c48b7bec"


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
    core, wrapper = npu / "src/attn_flash.c", npu / "src/attn_flash_llama_vb.c"
    frontend = SOURCE / "attention.model2mlir.mlir"
    header_archive = SOURCE / "attn_vpu_llama.h.gz"
    header = gzip.decompress(header_archive.read_bytes())
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    if (_git_revision(args.npu_root) != baseline["npu_revision"] or
            _git_revision(args.rtl_root) != baseline["rtl_revision"] or
            profile_sha256(profile) != baseline["profile_sha256"] or
            _sha(core) != baseline["source_core_sha256"] or
            _sha(wrapper) != WRAPPER_HASH or
            _sha(frontend) != baseline["frontend_mlir_sha256"] or
            _sha(header_archive) != baseline["source_header_gzip_sha256"] or
            (ARCHIVE / "attn_flash_llama_vb_expect.h").read_text() !=
            f"#define EXP_FLASH_HASH_O 0x{EXPECTED_HASH}ULL\n" or
            _sha(args.riscv_root / "bin/riscv64-unknown-elf-gcc") !=
            baseline["riscv_gcc_sha256"] or
            _sha(args.riscv_root / "bin/spike") !=
            baseline["spike_binary_sha256"] or
            _sha(args.spike_extension) != baseline["spike_extension_sha256"]):
        raise ValueError("pinned variable-block flash source, RTL, or toolchain differs")
    mlir, binding = render_flash_mid(
        frontend.read_text(), core.read_text(), wrapper.read_text(),
        header.decode(), profile, shape=STREAM_SHAPE, block_k=256,
        block_first=64, block_last=192)
    graph = ARCHIVE / "flash_llama_vb_stream.mlir"
    if (mlir != graph.read_text() or
            binding != json.loads((ARCHIVE / "binding.json").read_text())):
        raise ValueError("archived variable-block graph or source binding differs")
    out = args.out_dir
    out.mkdir(parents=True)
    (out / "attn_vpu_llama.h").write_bytes(header)
    for name in ("attn_flash_llama_vb_expect.h", "driver.c"):
        shutil.copyfile(ARCHIVE / name, out / name)
    obj = out / "object"
    _run([sys.executable, "-m", "tools.compile_object",
          "--mlir", str(graph), "--precursor-mlir", str(frontend),
          "--binding-json", str(ARCHIVE / "binding.json"),
          "--source-core", str(core), "--source-wrapper", str(wrapper),
          "--source-header", str(header_archive),
          "--profile", str(args.profile), "--rtl-root", str(args.rtl_root),
          "--riscv-root", str(args.riscv_root), "--mx-opt", str(args.mx_opt),
          "--out-dir", str(obj)], cwd=ROOT, log=out / "compile_object.log")
    manifest = json.loads((obj / "object_manifest.json").read_text())
    if (manifest["shape_sq_sk_d_bk"] != [64, 2048, 64, 256] or
            [manifest["block_first"], manifest["block_last"]] != [64, 192] or
            (manifest["command_count"], manifest["fence_count"]) != (508, 56) or
            manifest["allocated_data_section_bytes"] != 0):
        raise ValueError("variable-block flash object schedule differs")
    run_flash_spike_parity(
        out=out, npu_root=args.npu_root, rtl_root=args.rtl_root,
        riscv_root=args.riscv_root, spike_extension=args.spike_extension,
        object_path=obj / "mx_issue.o", slug="flash_llama_vb_stream",
        expected_hash=EXPECTED_HASH, output_words=4096)
    for name in ("mx_issue.c", "mx_issue.h", "mx_issue.o",
                 "physical_program.json"):
        if _sha(obj / name) != _sha(ARCHIVE / "object" / name):
            raise ValueError(f"variable-block object replay differs: {name}")
    for kind in ("source", "generated"):
        elf = f"flash_llama_vb_stream_{kind}.elf"
        if _sha(out / elf) != _sha(ARCHIVE / elf):
            raise ValueError(f"variable-block ELF replay differs: {elf}")
        for run in (1, 2):
            log = f"{kind}_spike_{run}.log"
            if _sha(out / log) != _sha(ARCHIVE / log):
                raise ValueError(f"variable-block Spike replay differs: {log}")
    receipt = {
        "schema": "mx_gemmini.flash_llama_vb_stream_spike_parity.v1",
        "status": "typed_variable_block_flash_full_bf16_source_parity",
        "source_baseline_sha256": _sha(SOURCE / "receipt.json"),
        "model2mlir_revision": baseline["model2mlir_revision"],
        "npu_revision": baseline["npu_revision"],
        "rtl_revision": baseline["rtl_revision"],
        "profile_sha256": profile_sha256(profile),
        "source_core_sha256": _sha(core),
        "source_wrapper_sha256": _sha(wrapper),
        "frontend_mlir_sha256": _sha(frontend),
        "source_header_sha256": baseline["source_header_sha256"],
        "expected_output_hash": EXPECTED_HASH,
        "block_lengths": [64] + [256] * 7 + [192],
        "compared_bf16_words": 4096, "source_runs": 2, "compiler_runs": 2,
        "bound_mlir_sha256": _sha(graph),
        "binding_json_sha256": _sha(ARCHIVE / "binding.json"),
        "object_sha256": _sha(obj / "mx_issue.o"),
        "physical_program_sha256": _sha(obj / "physical_program.json"),
        "source_elf_sha256": _sha(out / "flash_llama_vb_stream_source.elf"),
        "generated_elf_sha256": _sha(out / "flash_llama_vb_stream_generated.elf"),
        "source_log_sha256": _sha(out / "source_spike_1.log"),
        "generated_log_sha256": _sha(out / "generated_spike_1.log"),
        "driver_sha256": _sha(ARCHIVE / "driver.c"),
        "riscv_gcc_sha256": _sha(args.riscv_root / "bin/riscv64-unknown-elf-gcc"),
        "spike_binary_sha256": _sha(args.riscv_root / "bin/spike"),
        "spike_extension_sha256": _sha(args.spike_extension),
        "native_verifier_sha256": _sha(args.mx_opt),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2,
                                                  sort_keys=True) + "\n")
    if (ARCHIVE / "receipt.json").is_file() and receipt != json.loads(
            (ARCHIVE / "receipt.json").read_text()):
        raise ValueError("variable-block replay receipt differs")
    print("typed variable-block flash: all 4,096 BF16 outputs match source twice on Spike")


if __name__ == "__main__":
    main()
