"""Replay source versus typed causal streamed flash on the pinned MX Spike."""

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
ARCHIVE = ROOT / "docs/evidence/nicolas_flash_llama_causal_stream_typed_e9ded36_0344bcb"
EXPECTED_HASH = "23d84ffca3505276"
NPU_REVISION = "0344bcba70b899039ccd0ac0801ca5605d46d3eb"
RTL_REVISION = "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
FRONTEND_SHA256 = "56099369c7eb23486170f0d092651fcd7c7b1e608fe8f14ed2f4bee13a5b4fb9"
HEADER_SHA256 = "efe35106618683a67c6cd58ef9189e266d8771ad47a06018b72aad4f5b355b7d"
SOURCE_HASH = "004459aa0e95def61427d5a6e9d8f71be59fcbc281a93a35745c1f33591bf7c0"
WRAPPER_HASH = "687f2d4be5cf00dabbe4daccc0b07f71a2cc24e6521e17cbbff9db48fac3319f"
GENERATOR_HASH = "0a30effcffb85da5db7e100068cfb3f0df683a15a3f6adb415cb622d55ebced6"


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
    npu = args.npu_root / "baremetal/mxgemmini"
    core = npu / "src/attn_flash.c"
    wrapper = npu / "src/attn_flash_llama_causal.c"
    generator = npu / "gen/gen_attn_vpu.py"
    frontend = ARCHIVE / "attention.model2mlir.mlir"
    header_archive = ARCHIVE / "attn_vpu_llama_causal.h.gz"
    header = gzip.decompress(header_archive.read_bytes())
    if (_git_revision(args.npu_root) != NPU_REVISION or
            _git_revision(args.rtl_root) != RTL_REVISION or
            _sha(core) != SOURCE_HASH or _sha(wrapper) != WRAPPER_HASH or
            _sha(generator) != GENERATOR_HASH or
            _sha(frontend) != FRONTEND_SHA256 or
            hashlib.sha256(header).hexdigest() != HEADER_SHA256 or
            (ARCHIVE / "attn_flash_llama_causal_expect.h").read_text() !=
            f"#define EXP_FLASH_HASH_O 0x{EXPECTED_HASH}ULL\n"):
        raise ValueError("pinned causal source, frontend, or RTL differs")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    mlir, binding = render_flash_mid(
        frontend.read_text(), core.read_text(), wrapper.read_text(),
        header.decode(), profile, shape=STREAM_SHAPE, block_k=256, causal=True)
    graph = ARCHIVE / "flash_llama_causal_stream.mlir"
    if (mlir != graph.read_text() or
            binding != json.loads((ARCHIVE / "binding.json").read_text())):
        raise ValueError("archived causal graph or source binding differs")
    out = args.out_dir
    out.mkdir(parents=True)
    (out / "attn_vpu_llama_causal.h").write_bytes(header)
    for name in ("attn_flash_llama_causal_expect.h", "driver.c"):
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
            manifest["input_format"] != "fp8_e4m3" or
            manifest["causal"] is not True or
            (manifest["command_count"], manifest["fence_count"]) != (581, 50) or
            manifest["allocated_data_section_bytes"] != 0 or
            manifest["buffer_abi"][-1]["name"] != "MASK_BF16" or
            manifest["buffer_abi"][-1]["minimum_bytes"] != 8192):
        raise ValueError("causal object inventory or mask ABI differs")
    run_flash_spike_parity(
        out=out, npu_root=args.npu_root, rtl_root=args.rtl_root,
        riscv_root=args.riscv_root, spike_extension=args.spike_extension,
        object_path=obj / "mx_issue.o", slug="flash_llama_causal_stream",
        expected_hash=EXPECTED_HASH, output_words=4096)
    for name in ("mx_issue.c", "mx_issue.h", "mx_issue.o",
                 "physical_program.json"):
        if _sha(obj / name) != _sha(ARCHIVE / "object" / name):
            raise ValueError(f"causal object replay differs: {name}")
    for kind in ("source", "generated"):
        elf = f"flash_llama_causal_stream_{kind}.elf"
        if _sha(out / elf) != _sha(ARCHIVE / elf):
            raise ValueError(f"causal ELF replay differs: {elf}")
        for run in (1, 2):
            log = f"{kind}_spike_{run}.log"
            if _sha(out / log) != _sha(ARCHIVE / log):
                raise ValueError(f"causal Spike replay differs: {log}")
    receipt = {
        "schema": "mx_gemmini.flash_llama_causal_stream_spike_parity.v1",
        "status": "typed_causal_streamed_flash_full_bf16_source_parity",
        "model2mlir_revision": "e9ded36eb85abf2d9097ac4dc11457c825853388",
        "model2mlir_capture_backend": "fx_importer",
        "npu_revision": NPU_REVISION,
        "rtl_revision": RTL_REVISION,
        "profile_sha256": profile_sha256(profile),
        "source_core_sha256": _sha(core),
        "source_wrapper_sha256": _sha(wrapper),
        "frontend_mlir_sha256": _sha(frontend),
        "source_header_sha256": HEADER_SHA256,
        "source_header_gzip_sha256": _sha(header_archive),
        "source_generator_sha256": _sha(generator),
        "generator_args": ["--sq", "64", "--sk", "2048", "--d", "64",
                           "--seed", "0", "--tag", "_llama_causal_probe",
                           "--no-s-golden", "--causal"],
        "expected_output_hash": EXPECTED_HASH,
        "compared_bf16_words": 4096,
        "source_runs": 2,
        "compiler_runs": 2,
        "bound_mlir_sha256": _sha(graph),
        "binding_json_sha256": _sha(ARCHIVE / "binding.json"),
        "object_sha256": _sha(obj / "mx_issue.o"),
        "physical_program_sha256": _sha(obj / "physical_program.json"),
        "source_elf_sha256": _sha(out / "flash_llama_causal_stream_source.elf"),
        "generated_elf_sha256": _sha(out / "flash_llama_causal_stream_generated.elf"),
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
        raise ValueError("causal replay receipt differs")
    print("typed causal streamed flash: all 4,096 BF16 outputs match source twice on Spike")


if __name__ == "__main__":
    main()
