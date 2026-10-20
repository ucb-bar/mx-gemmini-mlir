"""Check every retained gate/up packed byte against an independent target oracle."""

import argparse
import ctypes
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), "output must be fresh"
    helper = HERE.parent / "mx_compiler_optimization_probes_20261019/verify.py"
    spec = importlib.util.spec_from_file_location("probe_archive_reader", helper)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    manifest = json.loads((HERE / "artifacts.json").read_text())
    assert reader.sha(helper.read_bytes()) == manifest["archive_reader_sha256"]
    reader.HERE = HERE
    reader.unpack(args.workdir)
    common = args.workdir / "common"
    assert (
        reader.sha((common / "vpu_ref.h").read_bytes())
        == "459726e52e7d19c51bada75b2e3ded16f64371799d5bc443750eb22e8616b17b"
    )
    spec = importlib.util.spec_from_file_location(
        "probe_elf_payload", common / "elf_payload.py"
    )
    elf_reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(elf_reader)
    source = args.workdir / "reference.c"
    source.write_text("""#include "vpu_ref.h"
void reference(const float *g,const float *u,float *out){
 for(unsigned i=0;i<160u*5632u;++i){
  float s=1.0f/(1.0f+vpu_bf16_to_f(vpu_exp(vpu_f_to_bf16(-g[i]))));
  out[i]=(g[i]*s)*u[i];
 }
}
""")
    library = args.workdir / "reference.so"
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
            str(source),
            str(common / "reference_quant.c"),
            "-lm",
            "-o",
            str(library),
        ],
        check=True,
    )
    dll = ctypes.CDLL(str(library))
    pointer = ctypes.c_void_p
    dll.reference.argtypes = [pointer] * 3
    dll.mx_pack_activation_f32.argtypes = [
        pointer,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        pointer,
        pointer,
    ]

    # Decode the actual FP8 payloads; require the separable power-of-two matrix case.
    def fp8(codes):
        exponent, mantissa = (codes.astype(np.int32) >> 3) & 15, codes & 7
        result = np.where(
            exponent == 0,
            mantissa / 8 * 2.0**-6,
            (1 + mantissa / 8) * np.exp2(exponent - 7),
        )
        return (result * np.where(codes & 128, -1, 1)).astype(np.float32)

    report = {}
    for name in manifest["probes"]:
        folder = args.workdir / name
        payloads = [("input", "input.bin"), ("expected", "expected.bin")]
        payloads += [
            (f"{label}{i}", f"{label}{i}.bin")
            for i in range(2)
            for label in ("weight", "scales", "expected")
        ]
        with elf_reader.RV64PayloadImage(folder / "pair.elf") as image:
            for symbol, filename in payloads:
                blob = (folder / filename).read_bytes()
                assert image.read(symbol, len(blob)) == blob
        raw = np.frombuffer((folder / "input.bin").read_bytes(), np.uint8)
        a = raw[: 160 * 2048].reshape(160, 2048)
        scales = raw[160 * 2048 :].reshape(64, 160)
        assert np.all(a == a[:, :1]) and np.all(scales == scales[:1])
        av = fp8(a[:, 0]) * np.exp2(scales[0].astype(np.int32) - 127).astype(np.float32)
        projections = []
        for part in range(2):
            weights = np.frombuffer(
                (folder / f"weight{part}.bin").read_bytes(), np.uint8
            ).reshape(2048, 5632)
            ws = np.frombuffer(
                (folder / f"scales{part}.bin").read_bytes(), np.uint8
            ).reshape(64, 5632)
            assert np.all(weights == weights[:1]) and np.all(ws == ws[:1])
            values = (
                av[:, None]
                * fp8(weights[0])[None, :]
                * np.exp2(ws[0].astype(np.int32) - 127).astype(np.float32)[None, :]
                * np.float32(2048)
            )
            values = np.ascontiguousarray(values, dtype="<f4")
            assert values.tobytes() == (folder / f"expected{part}.bin").read_bytes()
            assert np.all((values.view(np.uint32) & 65535) == 0)
            projections.append(values)
        values = np.empty((160, 5632), np.float32)
        dll.reference(*(x.ctypes.data for x in projections), values.ctypes.data)
        gold = np.empty(929280, np.uint8)
        assert (
            dll.mx_pack_activation_f32(
                values.ctypes.data,
                160,
                5632,
                8,
                gold.ctypes.data,
                gold[160 * 5632 :].ctypes.data,
            )
            == 0
        )
        assert gold.tobytes() == (folder / "expected.bin").read_bytes()
        text = (folder / "spike.log").read_text()
        assert (
            text.splitlines().count("GATED_DONE") == 1 and "MODEL DEADLOCK" not in text
        )
        records = re.findall(
            r"^GATED trial=(\d+) role=(\d+) cycles=(\d+) bad=0 checked=929280$",
            text,
            re.M,
        )
        assert [(int(t), int(r)) for t, r, _ in records] == [
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 0),
        ]
        observed = {(t, r): bytearray() for t, r, _ in records}
        for line in text.splitlines():
            if not line.startswith("GATED_BYTES "):
                continue
            _, t, r, offset, count, *words = line.split()
            data = observed[t, r]
            assert int(offset) == len(data) and int(count) == len(words) == 32
            assert all(re.fullmatch("[0-9a-f]{2}", word) for word in words)
            data.extend(int(word, 16) for word in words)
        assert all(bytes(data) == gold.tobytes() for data in observed.values())
        perf = (folder / "perf.txt").read_text()
        assert re.findall(
            r"mesh: (\d+) tiles, (\d+) busy cycles; VPU: (\d+) commands", perf
        ) == [("3604480", "57671680", "540")]
        cycles = {
            role: [int(c) for _, r, c in records if int(r) == role] for role in (0, 1)
        }
        report[name] = {
            "control_cycles": cycles[0],
            "candidate_cycles": cycles[1],
            "cycle_ratio": float(np.mean(cycles[0]) / np.mean(cycles[1])),
            "checked_bytes_per_call": len(gold),
            "calls": 4,
        }
    args.out.write_text(
        json.dumps(
            {"passed": True, "probes": report, "full_model_speedup_claim": False},
            indent=2,
        )
        + "\n"
    )
    print(args.out.read_text())


if __name__ == "__main__":
    main()
