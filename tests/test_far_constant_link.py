"""Read immutable data beyond medany reach without moving the runtime stack."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from mx_gemmini_support.catalog_toolchain import native_catalog_compile_flags


def test_far_readonly_data_keeps_runtime_state_near_code(tmp_path):
    from merlin.common.paths import runtime_dir

    cc = os.environ.get("MERLIN_RISCV_GCC") or shutil.which("riscv64-unknown-elf-gcc")
    spike = os.environ.get("MERLIN_SPIKE") or shutil.which("spike")
    if not cc or not spike:
        pytest.skip("requires the selected RV64 toolchain and Spike")
    start = tmp_path / "start.S"
    start.write_text(
        ".section .text.init\n.global _start\n_start:\n"
        "la sp, _stack_top\ncall check\n1: j 1b\n"
    )
    source = tmp_path / "check.c"
    source.write_text("""extern const unsigned char packed[];
volatile unsigned long tohost __attribute__((section(".htif"), aligned(64)));
volatile unsigned long fromhost __attribute__((section(".htif"), aligned(64)));
static volatile unsigned long scratch;
void check(void) {
  scratch = 7;
  tohost = (packed[0] == 0xa5 && scratch == 7) ? 1 : 3;
}
""")
    data = tmp_path / "packed.S"
    data.write_text(
        '.section .merlin.large_rodata,"a",@progbits\n'
        ".global packed\npacked:\n.byte 0xa5\n"
    )
    objects = []
    assert "-fPIC" not in native_catalog_compile_flags()
    for path in [start, source, data]:
        obj = path.with_suffix(".o")
        flags = (
            native_catalog_compile_flags(far_constant_data=True)
            if path == source
            else ["-march=rv64gc", "-mabi=lp64d"]
        )
        subprocess.run(
            [cc, *flags, "-c", str(path), "-o", str(obj)],
            check=True,
            capture_output=True,
            text=True,
        )
        objects.append(obj)
    elf = tmp_path / "probe.elf"
    script = runtime_dir() / "baremetal/spike/model_link.ld"
    subprocess.run(
        [
            cc,
            "-nostdlib",
            "-nostartfiles",
            "-T",
            str(script),
            "-Wl,--section-start=.merlin.large_rodata=0x110000000",
            *(str(p) for p in objects),
            "-o",
            str(elf),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    nm = Path(cc).with_name("riscv64-unknown-elf-nm")
    output = subprocess.check_output([str(nm), "--format=posix", str(elf)], text=True)
    symbols = {
        parts[0]: int(parts[2], 16)
        for line in output.splitlines()
        if len(parts := line.split()) >= 3
    }
    assert symbols["packed"] - symbols["check"] > (1 << 31)
    assert abs(symbols["scratch"] - symbols["check"]) < (1 << 31)
    assert symbols["_stack_top"] < symbols["packed"]
    result = subprocess.run(
        [spike, "--isa=rv64gc", "-m0x80000000:0x100000000", str(elf)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_large_readonly_extent_does_not_push_runtime_state_out_of_reach(tmp_path):
    from merlin.common.paths import runtime_dir

    cc = os.environ.get("MERLIN_RISCV_GCC") or shutil.which("riscv64-unknown-elf-gcc")
    if not cc:
        pytest.skip("requires the selected RV64 toolchain")
    source = tmp_path / "layout.S"
    source.write_text(""".section .text.init
.global _start
_start:
  la sp, _stack_top
  la t0, near_mutable
  ret
.section .bss
.global near_mutable
near_mutable:
  .zero 8
.section .merlin.large_rodata,"a",@nobits
.global far_start
far_start:
  .zero 0x90000000
.global far_end
far_end:
""")
    obj, elf = tmp_path / "layout.o", tmp_path / "layout.elf"
    subprocess.run(
        [cc, "-march=rv64gc", "-mabi=lp64d", "-c", str(source), "-o", str(obj)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            cc,
            "-nostdlib",
            "-nostartfiles",
            "-T",
            str(runtime_dir() / "baremetal/spike/model_link.ld"),
            str(obj),
            "-o",
            str(elf),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = subprocess.check_output(
        [str(Path(cc).with_name("riscv64-unknown-elf-nm")), "--format=posix", str(elf)],
        text=True,
    )
    symbols = {
        parts[0]: int(parts[2], 16)
        for line in output.splitlines()
        if len(parts := line.split()) >= 3
    }
    assert symbols["_stack_top"] <= symbols["far_start"]
    assert symbols["far_end"] - symbols["far_start"] > (1 << 31)
    assert elf.stat().st_size < 65536
    overlap = subprocess.run(
        [
            cc,
            "-nostdlib",
            "-nostartfiles",
            "-T",
            str(runtime_dir() / "baremetal/spike/model_link.ld"),
            "-Wl,--defsym,MERLIN_ARENA_BASE=0x90000000",
            str(obj),
            "-o",
            str(tmp_path / "overlap.elf"),
        ],
        capture_output=True,
        text=True,
    )
    assert overlap.returncode != 0
    assert "static data overlaps the activation arena" in overlap.stderr


def test_far_data_mode_requires_prepacked_constants(tmp_path):
    from mx_gemmini_support.merlin_catalog_builder import build_merlin_catalog

    with pytest.raises(ValueError, match="prepacked"):
        build_merlin_catalog(
            tmp_path / "missing.mlir",
            tmp_path / "output",
            profile={},
            riscv_root=tmp_path,
            far_constant_data=True,
        )


def test_far_catalog_emits_allocated_data_and_got_references(tmp_path, monkeypatch):
    import numpy as np
    from safetensors.numpy import save_file

    from mx_gemmini_support.merlin_catalog_builder import build_merlin_catalog
    from mx_gemmini_support.target_profile import load_profile

    cc = os.environ.get("MERLIN_RISCV_GCC") or shutil.which("riscv64-unknown-elf-gcc")
    if not cc:
        pytest.skip("requires the selected RV64 toolchain")
    root = Path(cc).parent.parent
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(Path(__file__).resolve().parents[1]))
    source = tmp_path / "model.mlir"
    source.write_text("""builtin.module attributes {prov.level = "linalg-on-tensors"} {
      func.func @forward(%x: tensor<16x32xf32>, %w: tensor<32x32xf32>) -> tensor<16x32xf32> {
        %zero = arith.constant 0.0 : f32
        %empty = tensor.splat %zero : tensor<16x32xf32>
        %result = linalg.matmul {prov.region_id="matmul_0",
          prov.op="matmul", prov.family="contraction", prov.fqn="projection",
          prov.orig_dtype="float32", prov.aten="aten.mm.default"}
          ins(%x, %w : tensor<16x32xf32>, tensor<32x32xf32>)
          outs(%empty : tensor<16x32xf32>)
          -> tensor<16x32xf32>
        func.return %result : tensor<16x32xf32>
      }
    }""")
    checkpoint = tmp_path / "weights.safetensors"
    save_file(
        {"w": np.arange(1024, dtype=np.float32).reshape(32, 32) / 1024}, checkpoint
    )
    manifest = {
        "0": {"kind": "input", "name": "x"},
        "1": {"kind": "param", "weight": "w", "dtype": "float32", "shape": [32, 32]},
    }
    profile = load_profile(
        Path(__file__).resolve().parents[1]
        / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
    )
    options = {
        "profile": profile,
        "riscv_root": root,
        "checkpoint_manifest": manifest,
        "checkpoint_weights_path": checkpoint,
        "prepack_immutable_weights": True,
        "precision_overrides": {"matmul_0": "mxfp8"},
        "far_constant_data": True,
    }
    first = tmp_path / "first"
    path, obj = build_merlin_catalog(source, first, **options)
    catalog = json.loads(path.read_text())
    assert catalog["compilation"]["far_constant_data"] is True
    assert "-fPIC" in catalog["compilation"]["flags"]
    sections = subprocess.check_output(
        [
            str(Path(cc).with_name("riscv64-unknown-elf-readelf")),
            "--section-headers",
            "--wide",
            str(obj),
        ],
        text=True,
    )
    assert re.search(r"\.merlin\.large_rodata\s+PROGBITS[^\n]+\s+A\s+", sections)
    relocs = subprocess.check_output(
        [str(Path(cc).with_name("riscv64-unknown-elf-objdump")), "-r", str(obj)],
        text=True,
    )
    assert re.search(r"R_RISCV_GOT_HI20\s+mx_dense_kernel_0_fixed_codes", relocs)
    second = tmp_path / "second"
    path, _ = build_merlin_catalog(
        source,
        second,
        **options,
        reuse_prepacked_data=first,
        reuse_prepacked_object=first / "checkpoint_weights.o",
    )
    assert json.loads(path.read_text())["packed_object_storage_reuse"] is not None
    assert os.path.samefile(
        first / "checkpoint_weights.o", second / "checkpoint_weights.o"
    )
