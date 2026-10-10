"""Reconstruct the full GQA output from executed MX PV and Muon row state.

The source kernel runs on the isolated corrected Cyclotron MX model. A narrow
diagnostic store captures each row's BF16 correction and running denominator.
The two source PV tiles per head come from the independently qualified MX
roster. Torch BF16 arithmetic applies the exact source recurrence and checks
all final output values against the unmodified source kernel's GMEM buffer.
"""

from __future__ import annotations

import argparse
import difflib
import gzip
import json
from pathlib import Path

from mx_gemmini_support.source_attention_pv import MUON_REQUANT_SOURCE_SHA256
from mx_gemmini_support.source_attention_qk import LATEST_SOURCE_REVISION, SUBMODULE_REVISION
from mx_gemmini_support.source_fp6 import _array, _bytes
from tools.qualify_radiance_gqa_cyclotron_pv import (
    QK_EVIDENCE, SOURCE_HEADER_SHA256, SOURCE_KERNEL_SHA256,
    _compile, _revision, _sha, _simulate, _stage)
from tools.qualify_radiance_gqa_pv_roster import EVIDENCE as PV_ROSTER


ROOT = Path(__file__).resolve().parents[1]
ANCHOR = """            mu_barrier(5, wpb);
        }

        // ---- finalize:"""
PROBE = """            mu_barrier(5, wpb);
            // Qualification only: snapshot the row correction and denominator.
            {
                const uint32_t tile = h * FA_NBLK_USED + j;
                const volatile __shared uint32_t *corr =
                    reinterpret_cast<const volatile __shared uint32_t *>(CORR_SMEM);
                const volatile __shared uint32_t *denom =
                    reinterpret_cast<const volatile __shared uint32_t *>(LS_SMEM);
                volatile uint32_t *corr_out = reinterpret_cast<volatile uint32_t *>(0x40080000)
                    + tile * (FA_SQ / 2);
                volatile uint32_t *denom_out = reinterpret_cast<volatile uint32_t *>(0x40081000)
                    + tile * (FA_SQ / 2);
                for (uint32_t i = tid; i < FA_SQ / 2; i += thr) {
                    corr_out[i] = corr[i];
                    denom_out[i] = denom[i];
                }
                mu_barrier(7, wpb);
            }
        }

        // ---- finalize:"""


def _instrument(source: str) -> str:
    if source.count(ANCHOR) != 1:
        raise ValueError("source kernel no longer has the pinned finalization probe point")
    return source.replace(ANCHOR, PROBE)


def _bf16_view(data: bytes, *, torch, np):
    return torch.from_numpy(np.frombuffer(data, dtype="<u2").copy()).view(torch.bfloat16)


def _reconstruct(states: bytes, pv: bytes, *, torch, np) -> bytes:
    if len(states) != 6144 or len(pv) != 16 * 8192:
        raise ValueError("GQA recurrence requires 16 correction, denominator, and PV tiles")
    if any(states[2048:4096]):
        raise ValueError("unexpected writes between correction and denominator snapshots")
    corr = _bf16_view(states[:2048], torch=torch, np=np).reshape(16, 64)
    denom = _bf16_view(states[4096:], torch=torch, np=np).reshape(16, 64)
    pv_tiles = _bf16_view(pv, torch=torch, np=np).reshape(16, 64, 64)
    outputs = []
    for head in range(8):
        # Source rescale_accumulate: O_acc = bf16(PV0*corr1 + PV1).
        acc = (pv_tiles[2 * head].float() * corr[2 * head + 1].float()[:, None] +
               pv_tiles[2 * head + 1].float()).to(torch.bfloat16)
        # Source finalize_O rounds the reciprocal to BF16 before multiplication.
        reciprocal = (1.0 / denom[2 * head + 1].float()).to(torch.bfloat16)
        outputs.append((acc.float() * reciprocal.float()[:, None]).to(torch.bfloat16))
    return torch.stack(outputs).view(torch.int16).numpy().tobytes()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "radiance-lib-root", "cyclotron-root", "llvm-muon",
                 "riscv-root", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--baseline-index", type=Path)
    args = parser.parse_args()
    for name in ("source_root", "radiance_lib_root", "cyclotron_root", "llvm_muon",
                 "riscv_root", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    roster = json.loads((PV_ROSTER / "index.json").read_text())
    cyclotron_source = args.cyclotron_root / "src/muon/mxgemmini/mod.rs"
    muon_runtime = args.radiance_lib_root / "libmuonrt.a"
    muon_impl = args.source_root / "kernels/flash_attention_mx_gqa/flash_mx_impl.hpp"
    if (_revision(args.source_root) != LATEST_SOURCE_REVISION or
            _revision(args.source_root / "lib/mxgemmini") != SUBMODULE_REVISION or
            _revision(args.cyclotron_root) != roster["cyclotron_revision"] or
            _sha(cyclotron_source.read_bytes()) != roster["cyclotron_model_source_sha256"] or
            _sha((args.cyclotron_root / "target/release/cyclotron").read_bytes()) !=
            roster["cyclotron_binary_sha256"] or
            _sha(muon_runtime.read_bytes()) != roster["muon_runtime_archive_sha256"] or
            _sha(muon_impl.read_bytes()) != MUON_REQUANT_SOURCE_SHA256):
        raise ValueError("source, Muon runtime, or corrected Cyclotron differs from PV roster")
    original = (args.source_root / "kernels/flash_attention_mx_gqa/kernel.cpp").read_bytes()
    header = gzip.decompress((QK_EVIDENCE / "fa_data.h.gz").read_bytes())
    pv = gzip.decompress((PV_ROSTER / "pv_tiles.bin.gz").read_bytes())
    baseline_o = gzip.decompress((PV_ROSTER / "corrected_baseline_o.bin.gz").read_bytes())
    if (_sha(original) != SOURCE_KERNEL_SHA256 or
            _sha(header) != SOURCE_HEADER_SHA256 or
            _sha(pv) != roster["pv_tiles_sha256"] or
            _sha(baseline_o) != roster["unmodified_corrected_o_sha256"]):
        raise ValueError("source fixture or executed PV evidence changed")
    args.out_dir.mkdir(parents=True)
    probe = _instrument(original.decode())
    (args.out_dir / "probe.patch").write_text("".join(difflib.unified_diff(
        original.decode().splitlines(keepends=True), probe.splitlines(keepends=True),
        fromfile="kernel.cpp", tofile="kernel.cpp.final_probe", n=0)))
    kernel_dir = _stage(args.source_root, args.out_dir / "probe", probe, header)
    elf = _compile(kernel_dir, args, args.out_dir / "build_probe.log")
    states = _simulate(args, elf, 0x40080000, 6144,
                       args.out_dir / "corr_l.bin", args.out_dir / "corr_l.log")
    actual_o = _simulate(args, elf, 0x40040000, 65536,
                         args.out_dir / "final_o.bin", args.out_dir / "final_o.log")
    if actual_o != baseline_o:
        raise ValueError("row-state diagnostic changed the source kernel's final output")
    import numpy as np
    import torch
    predicted_o = _reconstruct(states, pv, torch=torch, np=np)
    if predicted_o != actual_o:
        mismatches = sum(predicted_o[i:i + 2] != actual_o[i:i + 2]
                         for i in range(0, len(actual_o), 2))
        raise ValueError(f"Muon final recurrence has {mismatches}/32768 BF16 mismatches")
    source_o = _bytes(_array(
        header.decode("ascii"), name="O_gold", ctype="uint16_t",
        dimensions="[FA_NQ*FA_SQ][FA_D]", count=8 * 64 * 64, maximum=65535), 2)
    source_o_mismatches = sum(source_o[i:i + 2] != actual_o[i:i + 2]
                              for i in range(0, len(actual_o), 2))
    if source_o_mismatches != roster["source_o_gold_bf16_mismatches"]:
        raise ValueError("source O_gold diagnostic changed")
    (args.out_dir / "reconstructed_o.bin").write_bytes(predicted_o)
    index = {
        "schema": "mx_gemmini.gqa_final_recurrence_cyclotron.v1",
        "status": "executed_pv_and_muon_row_state_reconstruct_full_o",
        "scope": "all 8 heads and 2 blocks; corrected Cyclotron MX co-model; no compiled mixed Muon/MX kernel or RTL SFU claim",
        "source_revision": LATEST_SOURCE_REVISION,
        "cyclotron_revision": roster["cyclotron_revision"],
        "cyclotron_binary_sha256": roster["cyclotron_binary_sha256"],
        "muon_runtime_archive_sha256": roster["muon_runtime_archive_sha256"],
        "source_kernel_sha256": _sha(original),
        "source_header_sha256": _sha(header),
        "muon_finalization_source_sha256": _sha(muon_impl.read_bytes()),
        "pv_roster_index_sha256": _sha((PV_ROSTER / "index.json").read_bytes()),
        "pv_tiles_sha256": _sha(pv),
        "row_state_sha256": _sha(states),
        "source_o_gold_sha256": _sha(source_o),
        "source_o_gold_bf16_mismatches": source_o_mismatches,
        "probe_patch_sha256": _sha((args.out_dir / "probe.patch").read_bytes()),
        "probe_elf_sha256": _sha(elf.read_bytes()),
        "executed_final_o_sha256": _sha(actual_o),
        "reconstructed_final_o_sha256": _sha(predicted_o),
        "compared_bf16_outputs": 32768,
    }
    if args.baseline_index and index != json.loads(args.baseline_index.read_text()):
        raise ValueError("final GQA recurrence differs from archived baseline")
    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print("executed MX PV plus Muon row state reconstruct all 32768 BF16 GQA output values")


if __name__ == "__main__":
    main()
