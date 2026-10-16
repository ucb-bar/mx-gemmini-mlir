"""Assemble verified MX session output dumps into a NumPy quality array."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


_NUMPY_DTYPE = {"f32": "<f4", "f16": "<f2", "bf16": "<u2",
                "i64": "<i8", "i32": "<i4", "i1": "u1"}


def assemble(session, run: dict, root: Path, *, programs: tuple[str, ...],
             output: int, squeeze_leading_one: bool,
             omit_single_step_axis: bool) -> np.ndarray:
    from merlin.llvmlower.session_outputs import assembled_output_bytes

    if (run.get("schema") != "mx_gemmini.merlin_session_spike_run.v1" or
            run.get("complete") is not True):
        raise ValueError("MX session run is incomplete")
    indices = [session.program_names.index(name) for name in programs]
    if not indices or indices != sorted(set(indices)):
        raise ValueError("programs must be unique and in session order")
    signatures = []
    for index in indices:
        if output < 0 or output >= len(session.programs[index].outputs):
            raise ValueError("unknown session output")
        shape, dtype = session.programs[index].outputs[output]
        signatures.append((tuple(shape), dtype))
    if len(set(signatures)) != 1:
        raise ValueError("selected session outputs have different ABIs")
    shape, dtype = signatures[0]
    if dtype not in _NUMPY_DTYPE:
        raise ValueError("unsupported session output dtype")
    if squeeze_leading_one and (not shape or shape[0] != 1):
        raise ValueError("selected output has no leading singleton dimension")

    digest_rows = run.get("output_digests", [])
    digests = {(row["program"], row["step"], row["output"]): row
               for row in digest_rows}
    dumps = run.get("output_dumps", [])
    dump_keys = [(row["program"], row["step"], row["output"]) for row in dumps]
    if len(digests) != len(digest_rows) or len(set(dump_keys)) != len(dump_keys):
        raise ValueError("MX session run has duplicate output records")
    for row in dumps:
        key = (row["program"], row["step"], row["output"])
        expected = digests.get(key)
        if expected is None or any(row[field] != expected[field]
                                   for field in ("bytes", "sha256")):
            raise ValueError("MX output dump differs from the guest digest")

    values = []
    for name in programs:
        for data in assembled_output_bytes(session, root, dumps,
                                           program=name, output=output):
            array = np.frombuffer(data, dtype=_NUMPY_DTYPE[dtype]).reshape(shape)
            values.append(array[0] if squeeze_leading_one else array)
    if omit_single_step_axis and len(values) != 1:
        raise ValueError("single-step axis can be omitted only for one step")
    return values[0].copy() if omit_single_step_axis else np.stack(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--run-receipt", type=Path, required=True)
    parser.add_argument("--merlin-src", type=Path, required=True)
    parser.add_argument("--program", action="append", required=True)
    parser.add_argument("--output", type=int, default=0)
    parser.add_argument("--squeeze-leading-one", action="store_true")
    parser.add_argument("--omit-single-step-axis", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.out.with_suffix(".json")
    if (args.out.suffix != ".npy" or args.out.exists() or
            args.out.is_symlink() or receipt_path.exists() or
            receipt_path.is_symlink()):
        parser.error("--out must name a new .npy file and companion receipt")
    sys.path.insert(0, str(args.merlin_src.resolve(strict=True)))
    from merlin.llvmlower.session_bundle import load

    session = load(args.session_root)
    path = args.run_receipt.resolve(strict=True)
    run = json.loads(path.read_text())
    array = assemble(
        session, run, path.parent, programs=tuple(args.program),
        output=args.output, squeeze_leading_one=args.squeeze_leading_one,
        omit_single_step_axis=args.omit_single_step_axis)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, array, allow_pickle=False)
    receipt = {"schema": "mx_gemmini.assembled_session_output.v1",
               "run_receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "programs": args.program, "output": args.output,
               "squeeze_leading_one": args.squeeze_leading_one,
               "omit_single_step_axis": args.omit_single_step_axis,
               "shape": list(array.shape), "dtype": str(array.dtype),
               "array_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "shape": receipt["shape"],
                      "dtype": receipt["dtype"]}))


if __name__ == "__main__":
    main()
