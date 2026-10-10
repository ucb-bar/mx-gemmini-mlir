"""Cross-check the source Muon GQA kernel against compiled MX PV on Cyclotron.

The source kernel is copied into an isolated build tree. Two diagnostic stores
snapshot its first P/scale and PV tiles; they do not replace the source math.
The unmodified and probed kernels must produce identical final O buffers.
"""

from __future__ import annotations

import argparse
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from mx_gemmini_support.source_attention_qk import LATEST_SOURCE_REVISION, SUBMODULE_REVISION


ROOT = Path(__file__).resolve().parents[1]
PV_EVIDENCE = ROOT / "docs/evidence/radiance_gqa_pv_proxy_80f84ca"
QK_EVIDENCE = ROOT / "docs/evidence/radiance_gqa_qk_roster_80f84ca"
SOURCE_KERNEL_SHA256 = "fd6b7d62322a97b1ded81390400b76eb714a62149b665f2d7ad034401cbc1a0f"
SOURCE_HEADER_SHA256 = "01e041222fec2508b585707d768620a5126e16647d406a2e85bfb99737fd78b8"
CYCLOTRON_REVISION = "2d6adad4ad94d1621fdff9c9a1e5eac871048f24"
CYCLOTRON_CONFIG_SHA256 = "70ef96c088c01caf3093bdff649fdb1926ea60ee68c4fcbd98af5edbf6dbf508"

P_ANCHOR = """            mu_barrier(2, wpb);

            // pack P scales"""
P_PROBE = """            mu_barrier(2, wpb);
            // Qualification only: snapshot the first MX input after Muon requantization.
            if (h == 0 && j == 0 && tid == 0) {
                const volatile __shared uint32_t *p =
                    reinterpret_cast<const volatile __shared uint32_t *>(P_BYTE[cur]);
                volatile uint32_t *p_out = reinterpret_cast<volatile uint32_t *>(P_GMEM);
                for (uint32_t i = 0; i < FA_SQ * FA_BK / 4; ++i) p_out[i] = p[i];
                const volatile __shared uint32_t *scales =
                    reinterpret_cast<const volatile __shared uint32_t *>(SCALE_SMEM);
                volatile uint32_t *s_out = reinterpret_cast<volatile uint32_t *>(PS_GMEM);
                for (uint32_t i = 0; i < FA_GKB * FA_SQ; ++i) s_out[i] = scales[i];
            }

            // pack P scales"""
PV_ANCHOR = """            mu_barrier(4, wpb);

            // O_acc ="""
PV_PROBE = """            mu_barrier(4, wpb);
            // Qualification only: snapshot the first BF16 PV result before accumulation.
            if (h == 0 && j == 0) {
                const volatile __shared uint32_t *pv =
                    reinterpret_cast<const volatile __shared uint32_t *>(PVOUT_SMEM);
                volatile uint32_t *pv_out = reinterpret_cast<volatile uint32_t *>(0x40060000);
                for (uint32_t i = tid; i < FA_SQ * FA_D / 2; i += thr) pv_out[i] = pv[i];
                mu_barrier(7, wpb);
            }

            // O_acc ="""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _revision(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def _run(command: list[str], *, cwd: Path, log: Path, env: dict[str, str] | None = None) -> None:
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=cwd, stdout=stream,
                                stderr=subprocess.STDOUT, env=env, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def _instrument(source: str) -> str:
    for anchor, replacement in ((P_ANCHOR, P_PROBE), (PV_ANCHOR, PV_PROBE)):
        if source.count(anchor) != 1:
            raise ValueError("source kernel no longer has the expected probe point")
        source = source.replace(anchor, replacement)
    return source


def _stage(source_root: Path, out_dir: Path, kernel: str, header: bytes) -> Path:
    kernel_dir = out_dir / "kernels/flash_attention_mx_gqa"
    kernel_dir.mkdir(parents=True)
    shutil.copyfile(source_root / "kernels/common.mk", out_dir / "kernels/common.mk")
    for name in ("Makefile", "mxgemm_core.hpp", "flash_mx_impl.hpp"):
        shutil.copyfile(source_root / "kernels/flash_attention_mx_gqa" / name,
                        kernel_dir / name)
    (kernel_dir / "kernel.cpp").write_text(kernel)
    (kernel_dir / "include").mkdir()
    (kernel_dir / "include/fa_data.h").write_bytes(header)
    return kernel_dir


def _compile(kernel_dir: Path, args: argparse.Namespace, log: Path) -> Path:
    source_lib = args.radiance_lib_root
    riscv_sysroot = args.riscv_root / "riscv64-unknown-elf"
    _run(["make", "-j2", "kernel.radiance.elf",
          f"MU_CXX={args.llvm_muon / 'bin/clang++'} -stdlib=libc++",
          f"LLVM_MUON={args.llvm_muon}",
          f"RADIANCE_LIB_PATH={source_lib}",
          f"GEMMINI_SW_PATH={args.source_root / 'lib/mxgemmini'}",
          f"RISCV_TOOLCHAIN_PATH={args.riscv_root}",
          f"RISCV_SYSROOT={riscv_sysroot}",
          f"MU_LIBC_INCLUDE={riscv_sysroot / 'include'}"], cwd=kernel_dir, log=log)
    return kernel_dir / "kernel.radiance.elf"


def _simulate(args: argparse.Namespace, elf: Path, address: int, length: int,
              target: Path, log: Path) -> bytes:
    import os
    env = os.environ.copy()
    env["CYCLOTRON_MXGEMMINI"] = "1"
    env["CYCLOTRON_DUMP_GMEM"] = f"{address:#x}:{length}:{target}"
    _run([str(args.cyclotron_root / "target/release/cyclotron"), "config.toml",
          "--binary-path", str(elf)], cwd=args.cyclotron_root, log=log, env=env)
    data = target.read_bytes()
    if len(data) != length or b"simulation finished" not in log.read_bytes():
        raise ValueError(f"Cyclotron did not finish and dump {length} bytes: {log}")
    return data


def _untile_p(raw: bytes) -> bytes:
    # The MX A scratchpad stores four 16x16 tiles, not one row-major 64x64 tile.
    assert len(raw) == 4096
    return bytes(raw[((r // 16) * 4 + c // 16) * 256 + (r % 16) * 16 + c % 16]
                 for r in range(64) for c in range(64))


def _scales(raw: bytes) -> bytes:
    assert len(raw) == 512
    words = [int.from_bytes(raw[i:i + 4], "little") for i in range(0, len(raw), 4)]
    if any(word > 255 for word in words):
        raise ValueError("Muon P scale scratch has nonzero upper bytes")
    return bytes(words)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "cyclotron-root", "llvm-muon", "riscv-root", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--radiance-lib-root", type=Path,
                        help="built Muon runtime library; defaults to source-root/lib")
    parser.add_argument("--baseline-index", type=Path)
    args = parser.parse_args()
    for name in ("source_root", "cyclotron_root", "llvm_muon", "riscv_root", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    args.radiance_lib_root = (args.radiance_lib_root or
                              args.source_root / "lib").resolve()
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    if (_revision(args.source_root) != LATEST_SOURCE_REVISION or
            _revision(args.source_root / "lib/mxgemmini") != SUBMODULE_REVISION or
            _revision(args.cyclotron_root) != CYCLOTRON_REVISION):
        raise ValueError("source, MX software, or Cyclotron revision differs from pinned model")
    if _sha((args.cyclotron_root / "config.toml").read_bytes()) != CYCLOTRON_CONFIG_SHA256:
        raise ValueError("Cyclotron configuration differs from pinned model")
    original = (args.source_root / "kernels/flash_attention_mx_gqa/kernel.cpp").read_bytes()
    header = gzip.decompress((QK_EVIDENCE / "fa_data.h.gz").read_bytes())
    if _sha(original) != SOURCE_KERNEL_SHA256 or _sha(header) != SOURCE_HEADER_SHA256:
        raise ValueError("source kernel or generated fixture differs from pinned source")
    pv_index = json.loads((PV_EVIDENCE / "index.json").read_text())
    activation = (PV_EVIDENCE / "bundle/activation.bin").read_bytes()
    scales = (PV_EVIDENCE / "bundle/activation_scales.bin").read_bytes()
    golden = (PV_EVIDENCE / "bundle/golden_bf16.bin").read_bytes()
    if (_sha(activation) != pv_index["activation_sha256"] or
            _sha(scales) != pv_index["activation_scales_sha256"] or
            _sha(golden) != pv_index["golden_bf16_sha256"]):
        raise ValueError("archived compiler PV inputs or Spike golden changed")
    args.out_dir.mkdir(parents=True)
    probe_source = _instrument(original.decode())
    (args.out_dir / "probe.patch").write_text("".join(difflib.unified_diff(
        original.decode().splitlines(keepends=True),
        probe_source.splitlines(keepends=True),
        fromfile="kernel.cpp", tofile="kernel.cpp.probe", n=0)))
    baseline_dir = _stage(args.source_root, args.out_dir / "baseline", original.decode(), header)
    probe_dir = _stage(args.source_root, args.out_dir / "probe", probe_source, header)
    baseline_elf = _compile(baseline_dir, args, args.out_dir / "build_baseline.log")
    probe_elf = _compile(probe_dir, args, args.out_dir / "build_probe.log")
    baseline_o = _simulate(args, baseline_elf, 0x40040000, 65536,
                           args.out_dir / "baseline_o.bin", args.out_dir / "baseline_o.log")
    p_raw = _simulate(args, probe_elf, 0x40010000, 66048,
                      args.out_dir / "p_scratch.bin", args.out_dir / "p_scratch.log")
    p = _untile_p(p_raw[:4096])
    p_scales = _scales(p_raw[65536:66048])
    pv = _simulate(args, probe_elf, 0x40060000, 8192,
                   args.out_dir / "pv_bf16.bin", args.out_dir / "pv_bf16.log")
    probe_o = _simulate(args, probe_elf, 0x40040000, 65536,
                        args.out_dir / "probe_o.bin", args.out_dir / "probe_o.log")
    if p != activation or p_scales != scales or pv != golden or probe_o != baseline_o:
        raise ValueError("Cyclotron Muon P, MX PV, or final O differs from compiled baseline")
    index = {
        "schema": "mx_gemmini.gqa_cyclotron_pv_crosscheck.v1",
        "status": "source_muon_p_and_mx_pv_match_compiler_spike",
        "scope": "head0 block0 of 8-head GQA; Cyclotron functional MX co-model, not RTL SFU",
        "source_revision": _revision(args.source_root),
        "mx_software_revision": _revision(args.source_root / "lib/mxgemmini"),
        "cyclotron_revision": _revision(args.cyclotron_root),
        "cyclotron_binary_sha256": _sha((args.cyclotron_root / "target/release/cyclotron").read_bytes()),
        "cyclotron_config_sha256": CYCLOTRON_CONFIG_SHA256,
        "source_kernel_sha256": _sha(original),
        "source_header_sha256": _sha(header),
        "muon_runtime_archive_sha256": _sha((args.radiance_lib_root / "libmuonrt.a").read_bytes()),
        "probe_kernel_sha256": _sha(probe_source.encode()),
        "probe_patch_sha256": _sha((args.out_dir / "probe.patch").read_bytes()),
        "baseline_elf_sha256": _sha(baseline_elf.read_bytes()),
        "probe_elf_sha256": _sha(probe_elf.read_bytes()),
        "p_scratch_sha256": _sha(p_raw),
        "muon_p_codes_sha256": _sha(p),
        "compiler_p_codes_sha256": _sha(activation),
        "muon_p_scales_sha256": _sha(p_scales),
        "compiler_p_scales_sha256": _sha(scales),
        "cyclotron_pv_bf16_sha256": _sha(pv),
        "spike_pv_bf16_sha256": _sha(golden),
        "baseline_o_bf16_sha256": _sha(baseline_o),
        "probe_o_bf16_sha256": _sha(probe_o),
        "compared_p_codes": len(p),
        "compared_p_scales": len(p_scales),
        "compared_pv_bf16": len(pv) // 2,
        "compared_final_o_bf16": len(baseline_o) // 2,
    }
    if args.baseline_index and index != json.loads(args.baseline_index.read_text()):
        raise ValueError("Cyclotron cross-check differs from archived baseline")
    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print("Cyclotron P 0/4096, scales 0/128, PV 0/4096 differences; final O probe unchanged")


if __name__ == "__main__":
    main()
