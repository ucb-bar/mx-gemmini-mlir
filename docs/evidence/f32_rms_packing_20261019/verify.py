"""Check complete target packed outputs against separate FP32/scalar-quant references."""

import argparse
import ctypes
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), "output must be fresh"
    sha = lambda b: hashlib.sha256(b).hexdigest()
    verifier_sha = sha(Path(__file__).read_bytes())
    manifest_bytes = (HERE / "artifacts.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    helper = HERE.parent / "mx_compiler_optimization_probes_20261019/verify.py"
    assert sha(helper.read_bytes()) == manifest["archive_reader_sha256"]
    reader = load("f32_rms_archive_reader", helper)
    reader.HERE = HERE
    reader.unpack(args.workdir)
    common, folder = args.workdir / "common", args.workdir / "production"
    elf_reader = load("f32_rms_elf_reader", common / "elf_payload.py")
    with elf_reader.RV64PayloadImage(folder / "pair.elf") as elf:
        for name in ("input", "gamma", "expected_ordered", "expected_striped16"):
            blob = (folder / (name + ".bin")).read_bytes()
            assert elf.read(name, len(blob)) == blob
    x = np.fromfile(folder / "input.bin", "<f4").reshape(160, 2048)
    gamma = np.fromfile(folder / "gamma.bin", "<f4")
    assert gamma.shape == (2048,) and np.isfinite(x).all() and np.isfinite(gamma).all()
    oracle = args.workdir / "oracle.c"
    oracle.write_text("""#include <math.h>
void normalized(const float *x,const float *g,unsigned mode,float *out){
 float square[2048];
 for(unsigned row=0;row<160;row++){
  for(unsigned col=0;col<2048;col++)square[col]=x[row*2048+col]*x[row*2048+col];
  float sum=0.0f;
  if(mode){float partial[16]={0};
   for(unsigned lane=0;lane<16;lane++)for(unsigned start=lane;start<1024;start+=128)
    for(unsigned j=0;j<8;j++){unsigned col=2*(start+j*16);partial[lane]+=square[col]+square[col+1];}
   for(unsigned lane=0;lane<16;lane++)sum+=partial[lane];
  }else for(unsigned col=0;col<2048;col++)sum+=square[col];
  float inv=1.0f/sqrtf(sum/2048.0f+1e-5f);
  for(unsigned col=0;col<2048;col++){float first=x[row*2048+col]*inv;out[row*2048+col]=g[col]*first;}
 }
}
""")
    library = args.workdir / "oracle.so"
    subprocess.run(
        [
            "cc",
            "-O2",
            "-shared",
            "-fPIC",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-DMX_QUANT_REFERENCE",
            "-I",
            str(common),
            str(oracle),
            str(common / "reference_quant.c"),
            "-lm",
            "-o",
            str(library),
        ],
        check=True,
    )
    dll = ctypes.CDLL(str(library))
    dll.normalized.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    dll.mx_pack_activation_f32.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    text = (folder / "spike.log").read_text()
    assert text.splitlines().count("F32_RMS_DONE") == 1 and "MODEL DEADLOCK" not in text
    records = re.findall(
        r"^F32_RMS mode=(\d+) trial=(\d+) role=(\d+) cycles=(\d+) bad=0 checked=337920$",
        text,
        re.MULTILINE,
    )
    assert [(int(m), int(t), int(r)) for m, t, r, _ in records] == [
        (m, t, r) for m in (0, 1) for t, r in ((0, 0), (0, 1), (1, 1), (1, 0))
    ]
    observations = {(m, t, r): bytearray() for m, t, r, _ in records}
    for line in text.splitlines():
        if not line.startswith("F32_RMS_BYTES "):
            continue
        _, mode, trial, role, offset, count, *words = line.split()
        data = observations[mode, trial, role]
        assert int(offset) == len(data) and int(count) == len(words) == 32
        assert all(re.fullmatch("[0-9a-f]{2}", word) for word in words)
        data.extend(int(word, 16) for word in words)
    report = {}
    for mode, name in enumerate(("ordered", "striped16")):
        values = np.empty_like(x)
        dll.normalized(x.ctypes.data, gamma.ctypes.data, mode, values.ctypes.data)
        gold = np.empty(337920, np.uint8)
        assert (
            dll.mx_pack_activation_f32(
                values.ctypes.data,
                160,
                2048,
                8,
                gold.ctypes.data,
                gold[327680:].ctypes.data,
            )
            == 0
        )
        assert gold.tobytes() == (folder / ("expected_" + name + ".bin")).read_bytes()
        assert all(
            bytes(observations[m, t, r]) == gold.tobytes()
            for m, t, r, _ in records
            if int(m) == mode
        )
        cycles = {
            role: [
                int(c) for m, _, r, c in records if int(m) == mode and int(r) == role
            ]
            for role in (0, 1)
        }
        report[name] = {
            "control_cycles": cycles[0],
            "fused_cycles": cycles[1],
            "cycle_ratio": float(np.mean(cycles[0]) / np.mean(cycles[1])),
            "checked_bytes_per_call": 337920,
            "calls": 4,
        }
    perf = (folder / "perf.txt").read_text()
    assert re.findall(
        r"mesh: (\d+) tiles, (\d+) busy cycles; VPU: (\d+) commands", perf
    ) == [("0", "0", "0")]
    assert "host tracking off" in perf
    assert sha(Path(__file__).read_bytes()) == verifier_sha
    result = {
        "passed": True,
        "manifest_sha256": sha(manifest_bytes),
        "verifier_sha256": verifier_sha,
        "all_observed_packed_bytes_checked": True,
        "actual_ELF_payloads_reopened": True,
        "scope": "normalization plus packing probe; not full-model timing",
        "modes": report,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
