"""Project the captured flow catalog to its attention matrix roster."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    args = parser.parse_args()
    raw = json.loads(args.catalog.read_text())
    probe = json.loads(args.probe.read_text())
    assert raw["coverage_complete"] and raw["covered_contractions"] == 148
    assert probe["selected_operations"] == 148
    assert raw["profile_sha256"] == probe["profile_sha256"]
    assert args.catalog.resolve() == Path(probe["catalog_manifest"]).resolve()
    rows = [
        {"region": row["region"], "tensor_types": row["tensor_types"]}
        for row in raw["bindings"]
        if row["tensor_types"][0] in ("tensor<15x50x64xf32>",
                                      "tensor<15x50x113xf32>",
                                      "tensor<15x50x163xf32>")
    ]
    assert len(rows) == 32
    summary = {
        "schema": "mx_gemmini.smolvla_flow_attention_catalog.v1",
        "raw_catalog_sha256": sha256(args.catalog),
        "flow_capture_sha256": probe["capture_model_sha256"],
        "transformed_source_sha256": raw["source_sha256"],
        "profile_sha256": raw["profile_sha256"],
        "covered_contractions": 148,
        "attention_bindings": rows,
    }
    (HERE / "flow_attention_catalog.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"projected {len(rows)} attention matrix bindings")


if __name__ == "__main__":
    main()
