"""Run the pinned masked flash source separately for captured SmolVLA heads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
HASH_RE = re.compile(r"check O hash: serial ([0-9a-f]+), pipelined ([0-9a-f]+) \(equal\)")
CYCLES_RE = re.compile(r"PERF pipelined\s+(\d+) cycles")
ERROR_RE = re.compile(r"accuracy O head 0 \(50 rows\): rel_fro (\d+) ppm")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--heads", default="0-14", help="Head range a-b or one head")
    args = parser.parse_args()
    if "-" in args.heads:
        lo, hi = map(int, args.heads.split("-", 1))
        heads = range(lo, hi + 1)
    else:
        heads = [int(args.heads)]
    if not heads or min(heads) < 0 or max(heads) >= 15:
        parser.error("heads must be in 0..14")
    rocc_tools = Path(os.environ.get("ROCC_TOOLS", os.environ["ROCC_ROOT"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for head in heads:
        out = args.out_dir / f"head_{head:02d}"
        out.mkdir(exist_ok=True)
        header = out / "header.h"
        with (out / "generate.log").open("w") as stream:
            subprocess.run(
                [sys.executable, "-B", str(HERE / "generate_header.py"),
                 "--rocc-tools", str(rocc_tools), "--head-index", str(head),
                 "--out", str(header)], check=True, stdout=stream,
                stderr=subprocess.STDOUT)
        expected = out / "expect.h"
        expected.write_text("#define EXP_FLASH_HASH_O 0ULL\n")
        env = os.environ.copy()
        env.update({"OUT_DIR": str(out.resolve()),
                    "SOURCE_OPERAND_HEADER": str(header.resolve()),
                    "SOURCE_EXPECT_HEADER": str(expected.resolve())})
        with (out / "first_build.log").open("w") as stream:
            first = subprocess.run(["bash", str(HERE / "run_source.sh")], env=env,
                                   stdout=stream, stderr=subprocess.STDOUT)
        assert first.returncode == 1, (head, first.returncode)
        first_log = (out / "source_spike.log").read_text()
        hash_match = HASH_RE.search(first_log)
        assert hash_match and hash_match.group(1) == hash_match.group(2), head
        output_hash = hash_match.group(1)
        expected.write_text(f"#define EXP_FLASH_HASH_O 0x{output_hash}ULL\n")
        with (out / "passing_build.log").open("w") as stream:
            subprocess.run(["bash", str(HERE / "run_source.sh")], env=env,
                           check=True, stdout=stream, stderr=subprocess.STDOUT)
        log_path = out / "source_spike.log"
        log = log_path.read_text()
        assert "attn_flash PASSED" in log, head
        assert HASH_RE.search(log).group(1) == output_hash, head
        cycles = int(CYCLES_RE.search(log).group(1))
        error_ppm = int(ERROR_RE.search(log).group(1))
        row = {
            "head": head,
            "pipelined_cycles": cycles,
            "relative_frobenius_error_ppm_vs_captured_fp32": error_ppm,
            "output_hash": output_hash,
            "generated_header_sha256": sha256(header),
            "elf_sha256": sha256(out / "smol_head0.elf"),
            "source_log_sha256": sha256(log_path),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    summary = {
        "scope": "independent standalone attention-head source runs; one denoise step",
        "rows": rows,
        "sum_of_separately_timed_head_cycles": sum(row["pipelined_cycles"] for row in rows),
        "is_full_model_cycle_result": False,
    }
    (args.out_dir / "all_head_results.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
