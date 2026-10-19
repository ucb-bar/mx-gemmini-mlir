"""Check complete angle outputs; optionally recheck the full sensitivity arrays."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output(path):
    console = gzip.decompress(path.read_bytes()).decode()
    assert console.splitlines().count("DONE") == 1
    assert "METRIC memref_rank_mismatch 0\n" in console
    begins = [line.split() for line in console.splitlines() if line.startswith("OUT_BEGIN ")]
    assert begins == [["OUT_BEGIN", "f32", "5120"]]
    words = []
    lines = console.splitlines()
    begin = next(i for i, line in enumerate(lines) if line.startswith("OUT_BEGIN "))
    done = lines.index("DONE")
    assert begin < done
    for index, line in enumerate(lines):
        if line.startswith("OUT_CHUNK "):
            assert begin < index < done
            fields = line.split()
            assert fields[1] == "f32" and int(fields[2]) == len(words)
            assert 1 <= int(fields[3]) <= 1024 and len(fields[4:]) == int(fields[3])
            words.extend(map(int, fields[4:]))
    assert len(words) == 5120 and all(0 <= word <= 0xffffffff for word in words)
    values = np.asarray(words, dtype="<u4").view("<f4").reshape(32, 160)
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    assert f"OUT_SHA256 f32le 5120 20480 {digest}\n" in console
    cycles, = [int(line.split()[2]) for line in console.splitlines() if line.startswith("METRIC cycles ")]
    return values, cycles


def kl(reference, candidate):
    def log_softmax(values):
        shifted = values.astype(np.float64) - values.max(axis=-1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    ref, got = log_softmax(reference), log_softmax(candidate)
    return np.maximum(0, (np.exp(ref) * (ref - got)).sum(-1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity-output", type=Path)
    parser.add_argument("--full-reference", type=Path)
    args = parser.parse_args()
    if bool(args.sensitivity_output) != bool(args.full_reference):
        parser.error("supply both sensitivity output and full reference")
    here = Path(__file__).resolve().parent
    for name, digest in json.loads((here / "artifacts.json").read_text()).items():
        assert sha(here / name) == digest, name
    with np.load(here / "operands.npz", allow_pickle=False) as data:
        exact = data["a"][:, :1] * data["b"][:1, :]
        assert np.array_equal(exact.view(np.uint32), data["exact"].view(np.uint32))
    matrix, matrix_cycles = output(here / "matrix_spike.log.gz")
    host, host_cycles = output(here / "host_spike.log.gz")
    mixed, _ = output(here / "mixed_spike.log.gz")
    assert np.array_equal(host.view(np.uint32), exact.view(np.uint32))
    assert np.array_equal(mixed.view(np.uint32), exact.view(np.uint32))
    report = json.loads((here / "result.json").read_text())
    assert matrix_cycles == report["spike"]["cycles"] == 427146
    assert host_cycles == report["transformed_host"]["cycles"] == 54418
    assert float(abs(matrix - exact).max()) == report["spike"]["maximum_angle_error_radians"]
    assert abs(float(abs(np.cos(matrix) - np.cos(exact)).max()) -
               report["spike"]["maximum_cosine_error"]) < 2e-6
    for label, expected in (("matrix", 1), ("host", 0), ("mixed", 1)):
        receipt = json.loads((here / (label + "_mx_stage_probe.json")).read_text())
        assert receipt["selected_operations"] == expected
        assert receipt["spike"]["done"] and receipt["spike"]["exit_code"] == 0
        assert receipt["raw_selected_source_coverage_complete"]
        assert receipt["spike"]["extension_sha256"] == report["spike"]["extension_sha256"]
        if label != "matrix":
            assert receipt["host_singleton_matrix_regions"] == ["matmul_0"]
            record, = receipt["host_singleton_preparation"]["regions"]
            assert record["scalar_operations"] == ["arith.mulf", "arith.addf", "linalg.yield"]
            assert receipt["raw_matrix_operations_accounted"] == expected + 1
    if args.sensitivity_output:
        sensitivity = json.loads((here / "sensitivity_result.json").read_text())
        capture = json.loads((here / "full_capture_receipt.json").read_text())
        assert sha(args.sensitivity_output) == sensitivity["output_sha256"]
        assert sha(args.full_reference) == capture["artifacts"]["golden.npy"]["sha256"]
        got = np.load(args.sensitivity_output, allow_pickle=False).reshape(160, 32000)
        ref = np.load(args.full_reference, allow_pickle=False).reshape(got.shape)
        for name, indices in (("all_rows", slice(None)), ("late_33_rows", slice(-33, None))):
            measured = sensitivity["metrics"][name]
            delta = kl(ref[indices], got[indices])
            assert abs(float(delta.mean()) - measured["mean_kl_nats"]) < 1e-12
            assert abs(float(delta.max()) - measured["worst_kl_nats"]) < 1e-12
            assert np.count_nonzero(ref[indices].argmax(-1) == got[indices].argmax(-1)) == measured["top1_agreement"]
        print("full sensitivity arrays and metrics verified")
    print("all 5120 angle words, mixed placement, and stage cycles verified")


if __name__ == "__main__":
    main()
