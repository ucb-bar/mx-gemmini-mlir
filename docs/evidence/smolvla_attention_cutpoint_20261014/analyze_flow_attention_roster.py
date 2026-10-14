"""Map the captured flow catalog to separately timed source attention heads."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
LINK = HERE.parent / "smolvla_scalar_full_action_20261014/link.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    catalog = json.loads((HERE / "flow_attention_catalog.json").read_text())
    link = json.loads(LINK.read_text())
    source_by_keys = {
        163: json.loads((HERE / "source_all_heads.json").read_text()),
        113: json.loads((HERE / "source_113_all_heads.json").read_text()),
    }
    assert catalog["covered_contractions"] == 148
    assert catalog["profile_sha256"] == link["profile_sha256"]
    flow = next(row for row in link["programs"] if row["name"] == "flow_denoise")
    assert flow["steps"] == 10 and flow["selected_operations"] == 148
    assert catalog["flow_capture_sha256"] == flow["capture_model_sha256"]

    pairs = []
    for keys in (163, 113):
        qk = [f"tensor<15x50x64xf32>", f"tensor<15x64x{keys}xf32>",
              f"tensor<15x50x{keys}xf32>"]
        pv = [f"tensor<15x50x{keys}xf32>", f"tensor<15x{keys}x64xf32>",
              "tensor<15x50x64xf32>"]
        qk_regions = [row["region"] for row in catalog["attention_bindings"]
                      if row["tensor_types"] == qk]
        pv_regions = [row["region"] for row in catalog["attention_bindings"]
                      if row["tensor_types"] == pv]
        assert len(qk_regions) == len(pv_regions) == 8
        assert all(int(pv_region.split("_")[1]) == int(qk_region.split("_")[1]) + 1
                   for qk_region, pv_region in zip(qk_regions, pv_regions))
        source = source_by_keys[keys]
        rows = source["rows"]
        assert [row["head"] for row in rows] == list(range(15))
        per_head = {row["pipelined_cycles"] for row in rows}
        assert len(per_head) == 1
        cycles = per_head.pop()
        pairs.append({
            "logical_keys": keys,
            "qk_regions": qk_regions,
            "pv_regions": pv_regions,
            "attention_calls_per_flow_step": len(qk_regions),
            "heads_per_call": len(rows),
            "first_call_standalone_source_cycles_per_head": cycles,
            "illustrative_warm_source_head_sum_per_flow_step": len(rows) * len(qk_regions) * cycles,
            "source_receipt_sha256": sha256(HERE / ("source_all_heads.json"
                                                  if keys == 163 else
                                                  "source_113_all_heads.json")),
        })
    assert Counter(row["logical_keys"] for row in pairs) == Counter({163: 1, 113: 1})
    per_step = sum(row["illustrative_warm_source_head_sum_per_flow_step"] for row in pairs)
    report = {
        "schema": "mx_gemmini.smolvla_attention_roster_analysis.v1",
        "flow_attention_catalog_sha256": sha256(HERE / "flow_attention_catalog.json"),
        "raw_flow_catalog_sha256": catalog["raw_catalog_sha256"],
        "flow_capture_sha256": catalog["flow_capture_sha256"],
        "transformed_source_sha256": catalog["transformed_source_sha256"],
        "profile_sha256": catalog["profile_sha256"],
        "full_session_link_sha256": sha256(LINK),
        "flow_steps": flow["steps"],
        "pairs": pairs,
        "illustrative_warm_source_head_sum_per_flow_step": per_step,
        "illustrative_warm_source_head_sum_for_ten_steps": per_step * flow["steps"],
        "is_measured_stitched_stage_or_full_model": False,
        "limitations": [
            "Only the first call of each key shape in the first flow step was measured.",
            "The arithmetic repeats separately warmed source-head timings across calls and steps.",
            "The source head passes omit the rest of the flow graph, host work, and cross-call scheduling.",
            "No compiler-versus-source cycle ratio follows from this arithmetic.",
        ],
    }
    (HERE / "flow_attention_roster.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"calls_per_step": sum(row["attention_calls_per_flow_step"]
                                           for row in pairs),
                      "flow_steps": flow["steps"],
                      "illustrative_per_step": per_step,
                      "illustrative_ten_steps": per_step * flow["steps"]}, indent=2))


if __name__ == "__main__":
    main()
