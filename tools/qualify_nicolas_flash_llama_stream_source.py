"""Replay the canonical streamed flash source and full BF16 output on Spike."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from mx_gemmini_support.model2mlir_worklist import build_model2mlir_worklist
from mx_gemmini_support.nicolas_attention_shape import DenseAttentionShape
from mx_gemmini_support.nicolas_attention_vector import _audit_frontend_edges
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision
from tools.qualify_nicolas_flash_mid_typed import _output_words
from tools.qualify_nicolas_flash_qk0 import DATA_SOURCE, _run, _sha


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_flash_llama_stream_source_baseline_e9ded36_0344bcb"
DRIVER = ROOT / "tools/fixtures/nicolas_flash_llama_serial_driver.c"
HEADER_SHA256 = "6f978631f1c50dbf04b1b743aafc31715ce73b434e1b790fa5362b4bbda9dc52"
FRONTEND_SHA256 = "abdb301d802b2af42fcc91ad5aff60955fbca82d10926590519a0979063ffd5e"
CORE_SHA256 = "004459aa0e95def61427d5a6e9d8f71be59fcbc281a93a35745c1f33591bf7c0"
WRAPPER_SHA256 = "1245d6c5d994687eda049ff4ddfa7577c0cf0e246b26767c4bb6ba6e641224f2"
GENERATOR_SHA256 = "0a30effcffb85da5db7e100068cfb3f0df683a15a3f6adb415cb622d55ebced6"
EXPECTED_HASH = "dc33d078bc537b19"
MODEL2MLIR_REVISION = "e9ded36eb85abf2d9097ac4dc11457c825853388"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("npu-root", "profile", "rtl-root", "riscv-root",
                 "spike-extension", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    for name in ("npu_root", "profile", "rtl_root", "riscv_root",
                 "spike_extension", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    old = json.loads((DATA_SOURCE / "receipt.json").read_text())
    header_archive = ARCHIVE / "attn_vpu_llama.h.gz"
    header = gzip.decompress(header_archive.read_bytes())
    frontend = ARCHIVE / "attention.model2mlir.mlir"
    expect = ARCHIVE / "attn_flash_llama_expect.h"
    npu = args.npu_root / "baremetal/mxgemmini"
    core, wrapper = npu / "src/attn_flash.c", npu / "src/attn_flash_llama.c"
    generator = npu / "gen/gen_attn_vpu.py"
    cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    spike = args.riscv_root / "bin/spike"
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    if (_git_revision(args.npu_root) != old["npu_revision"] or
            _git_revision(args.rtl_root) != old["rtl_revision"] or
            profile_sha256(profile) != old["profile_sha256"] or
            hashlib.sha256(header).hexdigest() != HEADER_SHA256 or
            _sha(frontend) != FRONTEND_SHA256 or
            _sha(core) != CORE_SHA256 or _sha(wrapper) != WRAPPER_SHA256 or
            _sha(generator) != GENERATOR_SHA256 or
            expect.read_text() != f"#define EXP_FLASH_HASH_O 0x{EXPECTED_HASH}ULL\n" or
            _sha(cc) != old["riscv_gcc_sha256"] or
            _sha(spike) != old["spike_binary_sha256"] or
            _sha(args.spike_extension) != old["spike_extension_sha256"]):
        raise ValueError("pinned streamed flash source, frontend, RTL, or toolchain differs")
    shape = DenseAttentionShape.from_header(header.decode())
    if shape != DenseAttentionShape(64, 2048, 64):
        raise ValueError("streamed flash source shape differs")
    _audit_frontend_edges(frontend.read_text(), shape)
    worklist = build_model2mlir_worklist(frontend.read_text(), profile)
    if [site["shape_bmnk"] for site in worklist["rank3_batch_matmuls"]] != [
            [1, 64, 2048, 64], [1, 64, 64, 2048]]:
        raise ValueError("streamed flash frontend contraction shapes differ")

    out = args.out_dir
    out.mkdir(parents=True)
    (out / "attn_vpu_llama.h").write_bytes(header)
    shutil.copyfile(expect, out / "attn_flash_llama_expect.h")
    shutil.copyfile(DRIVER, out / "driver.c")
    software = args.rtl_root / "software/gemmini-rocc-tests"
    bench = software / "riscv-tests/benchmarks/common"
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET", "-DBAREMETAL=1",
             "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math", "-fno-common",
             "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
             "-march=rv64gc", "-Wa,-march=rv64gc"]
    for path in (out, npu, npu / "src", software / "riscv-tests",
                 software / "riscv-tests/env", software, bench):
        flags.extend(("-I", str(path)))
    objects = []
    for index, source in enumerate([*sorted(bench.glob("*.c")),
                                    *sorted(bench.glob("*.S")), out / "driver.c"]):
        obj = out / f"source_{index}.o"
        _run([str(cc), *flags, "-c", str(source), "-o", str(obj)],
             cwd=out, log=out / f"compile_{index}.log")
        objects.append(obj)
    elf = out / "source.elf"
    _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"), *(str(path) for path in objects),
          "-lm", "-lgcc", "-o", str(elf)], cwd=out, log=out / "link.log")
    outputs = []
    for run in (1, 2):
        log = out / f"source_spike_{run}.log"
        _run([str(spike), f"--extlib={args.spike_extension}",
              "--extension=gemmini", str(elf)], cwd=out, log=log)
        if "check O hash vs Spike: match" not in log.read_text() or \
                "K/V streamed (K x2, V x3 block buffers)" not in log.read_text():
            raise ValueError("canonical streamed flash did not match its source expectation")
        outputs.append(_output_words(log, EXPECTED_HASH, 4096))
    if outputs[0] != outputs[1]:
        raise ValueError("streamed source BF16 words vary between Spike runs")
    receipt = {
        "schema": "mx_gemmini.nicolas_flash_llama_stream_source_baseline.v1",
        "status": "canonical_streamed_source_full_bf16_spike_parity",
        "scope": "unmodified attn_flash_llama.c wrapper and attn_flash.c core; source oracle only",
        "shape_sq_sk_d_bk": [64, 2048, 64, 256],
        "model2mlir_revision": MODEL2MLIR_REVISION,
        "model2mlir_capture_backend": "fx_importer",
        "frontend_mlir_sha256": _sha(frontend),
        "npu_revision": _git_revision(args.npu_root),
        "rtl_revision": _git_revision(args.rtl_root),
        "profile_sha256": profile_sha256(profile),
        "source_core_sha256": _sha(core),
        "source_wrapper_sha256": _sha(wrapper),
        "source_generator_sha256": _sha(generator),
        "generator_dependencies_sha256": {
            "gen_llama_layer": _sha(npu / "gen/gen_llama_layer.py"),
            "mesh_par": _sha(npu / "gen/mesh_par.py"),
            "gen_matmul_llama": _sha(software / "gen_matmul_llama.py"),
            "compiler_operands": _sha(args.npu_root / "compiler/operands.py"),
            "compiler_wire": _sha(args.npu_root / "compiler/wire.py"),
            "mxquant_block": _sha(args.npu_root / "models/mxquant/block.py"),
            "quantization_scheme": _sha(args.npu_root / "config/scheme.py"),
        },
        "generator_args": ["--sq", "64", "--sk", "2048", "--d", "64",
                           "--seed", "0", "--tag", "_llama_probe", "--no-s-golden"],
        "source_header_sha256": HEADER_SHA256,
        "source_header_gzip_sha256": _sha(header_archive),
        "source_expect_sha256": _sha(expect),
        "source_driver_sha256": _sha(DRIVER),
        "source_output_hash": EXPECTED_HASH,
        "compared_bf16_words": 4096, "source_runs": 2,
        "riscv_gcc_sha256": _sha(cc),
        "spike_binary_sha256": _sha(spike),
        "spike_extension_sha256": _sha(args.spike_extension),
        "source_elf_sha256": _sha(elf),
        "source_spike_log_sha256": _sha(out / "source_spike_1.log"),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if (ARCHIVE / "receipt.json").is_file() and receipt != json.loads(
            (ARCHIVE / "receipt.json").read_text()):
        raise ValueError("streamed flash source replay receipt differs")
    print("canonical streamed flash: all 4,096 BF16 outputs match twice on Spike")


if __name__ == "__main__":
    main()
