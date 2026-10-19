"""Independently reconstruct packed bytes and reopen all 48 executed ELF payloads."""

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import numpy as np


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def positive_grid(codes, fraction_bits, bias):
    codes = np.asarray(codes, dtype=np.int32)
    exponent, fraction = codes >> fraction_bits, codes & ((1 << fraction_bits) - 1)
    return np.where(exponent == 0, np.ldexp(fraction.astype(np.float32), 1 - bias - fraction_bits),
                    np.ldexp(1 + fraction.astype(np.float32) / (1 << fraction_bits), exponent - bias)).astype(np.float32)


def nearest(values, grid):
    upper = np.clip(np.searchsorted(grid, values), 0, len(grid) - 1)
    lower = np.maximum(upper - 1, 0)
    upper_gap, lower_gap = grid[upper] - values, values - grid[lower]
    return np.where((upper_gap < lower_gap) | ((upper_gap == lower_gap) & ((lower & 1) != 0)), upper, lower)


def reconstruct(report, values):
    rows, cols, precision = (report[name] for name in ("rows", "cols", "precision"))
    physical = np.zeros((rows, cols), dtype="<f4")
    assert values.shape == (report["live_rows"] if report["kind"] == "zero_rows" else rows, cols)
    physical[:len(values)] = values
    bits = physical.view("<u4")
    rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view("<f4")
    weight = report["kind"] == "weight"
    groups = (rounded.reshape(rows // 32, 32, cols).transpose(0, 2, 1) if weight else
              rounded.reshape(rows, cols // 32, 32).transpose(1, 0, 2))
    finite = np.isfinite(groups).all(axis=2).reshape(-1)
    refused = np.flatnonzero(~finite)
    valid = (np.arange(finite.size) < (refused[0] if refused.size else finite.size)).reshape(groups.shape[:2])
    safe = np.where(valid[..., None], groups, np.float32(0))
    maximum = np.maximum(np.abs(safe).max(axis=2), np.float32(2 ** -23))
    exponent = np.clip(np.frexp(maximum)[1] - 1, -23, 127)
    magnitude = np.abs(safe) / np.ldexp(np.ones_like(maximum), exponent)[..., None]
    sign = np.signbit(safe).astype(np.uint8)
    if precision == 8:
        codes = nearest(magnitude, positive_grid(np.arange(127), 3, 7)).astype(np.uint8) | (sign << 7)
    else:
        intermediate = positive_grid(np.arange(16), 1, 3)
        rounded_intermediate = intermediate[nearest(magnitude, intermediate)]
        codes = nearest(rounded_intermediate, positive_grid(np.arange(8), 1, 1)).astype(np.uint8)
        codes = np.where(codes == 0, 0, codes | (sign << 3)).astype(np.uint8)
    scales = np.full(valid.shape, 0xa5, dtype=np.uint8)
    scales[valid] = (exponent[valid] + 127).astype(np.uint8)
    if weight:
        logical = codes.transpose(0, 2, 1).reshape(rows, cols)
        active = np.repeat(valid, 32, axis=0)
    else:
        logical = codes.transpose(1, 0, 2).reshape(rows, cols)
        active = np.repeat(valid.T, 32, axis=1)
    if precision == 8:
        packed = np.full((rows, cols), 0xa5, dtype=np.uint8)
        packed[active] = logical[active]
    else:
        low, high = (logical[:, ::2], logical[:, 1::2]) if weight else (logical[::2], logical[1::2])
        low_active, high_active = (active[:, ::2], active[:, 1::2]) if weight else (active[::2], active[1::2])
        packed = np.full(low.shape, 0xa5, dtype=np.uint8)
        # The original ABI clears the whole byte when writing its low nibble.
        assert not np.any(high_active & ~low_active)
        packed[low_active] = low[low_active]
        packed[high_active] |= high[high_active] << 4
    return packed.tobytes(), scales.tobytes(), -1 if refused.size else 0


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "artifacts.json").read_text())
    archive = root / manifest["archive"]["name"]
    assert archive.stat().st_size == manifest["archive"]["bytes"]
    assert sha(archive.read_bytes()) == manifest["archive"]["sha256"]
    reader_path = root.parent / "broadcast_multiply_20261019/verify.py"
    assert sha(reader_path.read_bytes()) == manifest["elf_reader_sha256"]
    spec = importlib.util.spec_from_file_location("retained_elf_reader", reader_path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert all(member.isfile() for member in members)
        assert len({member.name for member in members}) == len(members)
        assert {member.name for member in members} == set(manifest["members"])
        data = {member.name: tar.extractfile(member).read() for member in members}
    for name, raw in data.items():
        assert {"bytes": len(raw), "sha256": sha(raw)} == manifest["members"][name]

    def read(name):
        return json.loads(data[name])

    profile_sha = sha((json.dumps(read("profile.json"), sort_keys=True, separators=(",", ":")) + "\n").encode())
    defines = {"original": [], "words": ["-DMX_QUANT_GROUP_WORDS"],
               "original_inline": ["-DMX_QUANT_INLINE_GROUP"],
               "words_inline": ["-DMX_QUANT_GROUP_WORDS", "-DMX_QUANT_INLINE_GROUP"]}
    images = 0
    for label, case in sorted(manifest["cases"].items()):
        prefix = "cases/" + label + "/"
        report = read(prefix + "receipt.json")
        assert report["profile_sha256"] == profile_sha
        assert report["extension_sha256"] == "5058ebfd1b1457d5897a87ad2208f67cb98e72c27ba111a771c981e5f6b5cdc5"
        for name, digest in report["files_sha256"].items():
            assert sha(data[prefix + name]) == digest
        for name, digest in report["source_sha256"].items():
            assert sha(data[prefix + "sources/" + name]) == digest
        if case["phase"] == "merlin_preparation":
            for name in ("runtime_quant.c", "runtime_quant.h", "runtime_quant_lut.h", "runtime_group_prepare.h"):
                assert data[prefix + "sources/" + name] == data["resources/" + name]
        values = np.load(io.BytesIO(data[prefix + "input.npy"]), allow_pickle=False)
        codes, scales, status = reconstruct(report, values)
        assert codes == data[prefix + "expected_codes.bin"]
        assert scales == data[prefix + "expected_scales.bin"]
        assert status == report["reference_status"]
        payload = values.tobytes() + codes + scales
        assert payload == data[prefix + "payload.bin"]
        digest = 2166136261
        for byte in codes + scales:
            digest = ((digest ^ byte) * 16777619) & 0xffffffff
        assert set(report["results"]) == set(defines)
        for variant, result in report["results"].items():
            base = prefix + variant + "/"
            assert result["exit_code"] == 0 and result["defines"] == defines[variant]
            for name, file_sha in result["files_sha256"].items():
                assert sha(data[base + name]) == file_sha
            image = data[base + "section.elf"]
            reader.embedded_payload(image, payload)
            admission = result["elf_admission"]
            assert admission["status"] == "accepted" and admission["elf_sha256"] == sha(image)
            candidates = [name for name in data if name.startswith(base + "invocations/") and name.endswith("/policy/admission.json")]
            assert len(candidates) == 1 and sha(data[candidates[0]]) == admission["report_sha256"]
            assert read(candidates[0])["status"] == "accepted"
            lines = data[base + "spike.log"].decode().splitlines()
            checks = [dict(part.split("=", 1) for part in line.split()[1:]) for line in lines if line.startswith("PACK ")]
            assert checks == result["checks"] and lines.count("DONE") == 1
            assert [int(row["run"]) for row in checks] == [0, 1, 2]
            assert all(int(row["status"]) == status and int(row["hash"]) == digest
                       and int(row["mismatch"]) == 0 and int(row["guard"]) == 0 for row in checks)
            assert result["environment"]["GEMMINI_MODE"] == "both"
            assert result["environment"]["GEMMINI_PERF_CONFIG"] == "e4m3_vpu"
            assert result["environment"]["GEMMINI_PERF_SET"] == "mem.host_tracking=0"
            images += 1
        before, after = (int(report["results"][name]["checks"][1]["cycles"]) for name in ("original", "words_inline"))
        print(f"{label}: {before} -> {after} cycles; all {len(codes) + len(scales)} bytes reconstructed")
    resources = read("resources/resources.json")
    for variant in defines:
        raw = data["resources/" + variant + ".o"]
        assert len(raw) == resources[variant]["object_bytes"] and sha(raw) == resources[variant]["object_sha256"]
        assert data["resources/" + variant + ".su"].decode() == resources[variant]["stack_usage"]
    identity = resources["default_object_identity"]
    assert identity["all_bytes_identical"] and identity["reference_sha256"] == resources["original"]["object_sha256"]
    assert images == 48
    print("PASS: 48 native ELF payloads, 144 complete byte checks; section results only")


if __name__ == "__main__":
    main()
