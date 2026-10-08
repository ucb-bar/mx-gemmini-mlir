"""Compare one source MX FP8 header with the OOT payload and transfer planner.

This is byte/layout parity. It does not run MX arithmetic or qualify the SoC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mx_gemmini_support.contraction import plan_mx_contraction_payload
from mx_gemmini_support.transfer_ir import plan_uploads


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dimension(source: str, name: str) -> int:
    match = re.search(rf"^#define {re.escape(name)} (\d+)$", source, re.MULTILINE)
    if match is None:
        raise ValueError(f"source header lacks {name}")
    return int(match.group(1))


def matrix(source: str, name: str, rows: int, columns: int) -> list[list[int]]:
    match = re.search(
        rf"static const uint8_t {re.escape(name)}\[[^=]+\] = \{{(.*?)\n\}};",
        source, re.DOTALL,
    )
    if match is None:
        raise ValueError(f"source header lacks {name}")
    result = [
        [int(item, 0) for item in re.findall(r"0x[0-9a-fA-F]+|\b\d+\b", row)]
        for row in re.findall(r"\{([^{}]+)\}", match.group(1), re.DOTALL)
    ]
    if len(result) != rows or any(len(row) != columns or any(not 0 <= value < 256 for value in row)
                                   for row in result):
        raise ValueError(f"source {name} dimensions or byte values disagree with its declaration")
    return result


def check(header: Path, source_root: Path, profile_path: Path) -> dict:
    header = header.resolve()
    source_root = source_root.resolve()
    if not header.is_relative_to(source_root) or not header.name.startswith("mxgemm.data.fp8."):
        raise ValueError("expected a source-tree MX FP8 data header")
    source_bytes = header.read_bytes()
    source = source_bytes.decode()
    m = dimension(source, "MATMUL_M")
    k = dimension(source, "MATMUL_K")
    n = dimension(source, "MATMUL_N")
    if dimension(source, "MATMUL_GK") != k // 32 or k % 32:
        raise ValueError("source E8M0 K groups disagree with the contraction")
    a = matrix(source, "A_in", m, k)
    b = matrix(source, "B_in", k, n)
    sa = matrix(source, "A_scales_row", k // 32, m)
    sb = matrix(source, "B_scales_col", k // 32, n)
    activation_scales = [[sa[group][row] for group in range(k // 32)] for row in range(m)]
    weight_scales = [[sb[group][column] for group in range(k // 32)] for column in range(n)]
    payload = plan_mx_contraction_payload("mxfp8", a, b, activation_scales, weight_scales)
    profile_bytes = profile_path.read_bytes()
    profile = yaml.safe_load(profile_bytes)
    uploads = plan_uploads(payload, profile)
    waves = []
    for index, (wave, transfer) in enumerate(zip(payload.waves, uploads, strict=True)):
        start, stop = wave.wave.block_start, wave.wave.block_stop
        a_expected = bytes(value for row in a for value in row[start * 32:stop * 32])
        b_expected = bytes(value for row in b[start * 32:stop * 32] for value in row)
        sa_expected = bytes(value for row in sa[start:stop] for value in row)
        sb_expected = bytes(value for row in sb[start:stop] for value in row)
        actual = (wave.activation_bytes, wave.weight_bytes,
                  wave.activation_scale_bytes, wave.weight_scale_bytes)
        expected = (a_expected, b_expected, sa_expected, sb_expected)
        if actual != expected:
            raise ValueError(f"source operand or scale bytes differ in K wave {index}")
        waves.append({
            "k_block_start": start, "k_block_stop": stop,
            "activation_sha256": digest(actual[0]), "weight_sha256": digest(actual[1]),
            "activation_scales_sha256": digest(actual[2]),
            "weight_scales_sha256": digest(actual[3]),
            "upload_functs": [command.funct for command in transfer.commands],
            "upload_rs2": [command.rs2.immediate for command in transfer.commands],
        })
    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True,
    ).strip()
    return {
        "schema": "mx_gemmini.source_payload_check.v1",
        "status": "analytical_byte_parity_not_executed",
        "source_revision": revision,
        "source_header": str(header.relative_to(source_root)),
        "source_header_sha256": digest(source_bytes),
        "profile_name": profile["name"],
        "profile_sha256": digest(profile_bytes),
        "format": "mxfp8", "shape": [m, k, n],
        "waves": waves,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = check(args.header, args.source_root, args.profile)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "shape": receipt["shape"],
                      "waves": len(receipt["waves"])}))


if __name__ == "__main__":
    main()
