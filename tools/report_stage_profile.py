"""Summarize one completed MX stage's Merlin operation profile from Spike."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def _source_regions(build: Path, table: list[dict]) -> dict[int, str]:
    binding_path = build / "device/catalog_binding.json"
    catalog_path = build / "device_catalog/mx_catalog.json"
    if not binding_path.exists() and not catalog_path.exists():
        return {}
    if not binding_path.exists() or not catalog_path.exists():
        raise ValueError("incomplete MX catalog binding for operation profile")
    entry_to_kernel = json.loads(binding_path.read_text())["entry_to_kernel"]
    bindings = json.loads(catalog_path.read_text())["bindings"]
    calls: dict[str, list[dict]] = defaultdict(list)
    sources: dict[str, list[dict]] = defaultdict(list)
    for record in table:
        callee = (record.get("callee") or "").removeprefix("@")
        if callee in entry_to_kernel:
            calls[entry_to_kernel[callee]].append(record)
    for binding in bindings:
        sources[binding["symbol"]].append(binding)
    if calls.keys() != sources.keys():
        raise ValueError("profiled MX kernels differ from source bindings")
    matched = []
    for kernel, kernel_calls in calls.items():
        kernel_sources = sources[kernel]
        if len(kernel_calls) != len(kernel_sources):
            raise ValueError(f"profiled MX call count differs from {kernel} bindings")
        ordered_calls = sorted(kernel_calls, key=lambda item: item["id"])
        ordered_sources = sorted(kernel_sources,
                                 key=lambda item: item["source_operation_ordinal"])
        matched.extend((call["id"], source["source_operation_ordinal"],
                        source["region"]) for call, source in
                       zip(ordered_calls, ordered_sources))
    matched.sort()
    if any(left[1] >= right[1] for left, right in zip(matched, matched[1:])):
        raise ValueError("profiled MX call order differs from source operation order")
    return {ident: region for ident, _, region in matched}


def report_stage_profile(build: Path, spike_log: Path) -> dict:
    receipt = json.loads((build / "mx_stage_probe.json").read_text())
    if not receipt.get("op_profile"):
        raise ValueError("stage was not built with --op-profile")
    table_path = build / "op_profile_table.json"
    if (receipt.get("op_profile_table_sha256") and
            hashlib.sha256(table_path.read_bytes()).hexdigest()
            != receipt["op_profile_table_sha256"]):
        raise ValueError("operation profile table differs from the stage receipt")
    table = json.loads(table_path.read_text())
    if not isinstance(table, list) or len(table) > receipt.get("op_profile_capacity", 8192):
        raise ValueError("operation table exceeds the linked profiler capacity")
    if (receipt.get("op_profile_operations") is not None and
            len(table) != receipt["op_profile_operations"]):
        raise ValueError("operation profile table count differs from the stage receipt")
    console = spike_log.read_text(encoding="utf-8")
    if "DONE\n" not in console or "METRIC memref_rank_mismatch 0\n" not in console:
        raise ValueError("Spike did not complete the profiled stage")
    metrics = {}
    samples = {}
    for line in console.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "METRIC":
            metrics[parts[1]] = parts[2]
        elif len(parts) == 4 and parts[0] == "PROF":
            ident, ticks, hits = map(int, parts[1:])
            if ident in samples:
                raise ValueError(f"duplicate profiler id {ident}")
            samples[ident] = (ticks, hits)
    if metrics.get("build_hash") != receipt["build_hash"] or "cycles" not in metrics:
        raise ValueError("Spike log is not for this stage build")
    if -1 in samples or any(ident < 0 or ident >= len(table) for ident in samples):
        raise ValueError("profiler overflowed or emitted an unknown operation id")
    if not samples or metrics.get("prof_total_ticks") is None:
        raise ValueError("Spike did not emit an operation profile")
    if (metrics.get("prof_marks") is not None and
            sum(hits for _, hits in samples.values()) != int(metrics["prof_marks"]) - 1):
        raise ValueError("profiler mark count differs from operation hits")
    total = sum(ticks for ticks, _ in samples.values())
    if total != int(metrics["prof_total_ticks"]):
        raise ValueError("profile ticks differ from the guest's total")
    cycles = int(metrics["cycles"])
    if total > cycles:
        raise ValueError("profile attributed more cycles than the stage executed")
    source_regions = _source_regions(build, table)
    buckets: dict[str, int] = defaultdict(int)
    families: dict[str, int] = defaultdict(int)
    names: dict[str, int] = defaultdict(int)
    regions: dict[str, int] = defaultdict(int)
    rows = []
    for record in table:
        ident = record["id"]
        ticks, hits = samples.get(ident, (0, 0))
        if not hits:
            continue
        callee = record.get("callee") or ""
        owner = ("mx" if callee.startswith("@merlin_dev_mx_gemmini_") else
                 "vpu" if callee.startswith("@mx_vpu_") else "host")
        family = record.get("family") or record.get("mlir_op") or "unknown"
        buckets[owner] += ticks
        families[family] += ticks
        names[callee or record.get("mlir_op") or "unknown"] += ticks
        region = source_regions.get(ident) or record.get("region_id")
        if region:
            regions[region] += ticks
        rows.append({"id": ident, "cycles": ticks, "hits": hits,
                     "owner": owner, "mlir_op": record.get("mlir_op"),
                     "region": region, "fqn": record.get("fqn"),
                     "callee": callee or None, "family": family})
    rows.sort(key=lambda row: row["cycles"], reverse=True)
    return {
        "schema": "mx_gemmini.stage_op_profile.v1",
        "build_hash": receipt["build_hash"],
        "cycles": cycles,
        "attributed_cycles": total,
        "attribution_coverage": total / cycles,
        "operations": len(table), "observed_operations": len(samples),
        "by_owner": dict(sorted(buckets.items())),
        "by_family": dict(sorted(families.items(), key=lambda item: -item[1])),
        "by_operation": dict(sorted(names.items(), key=lambda item: -item[1])),
        "by_source_region": dict(sorted(regions.items(), key=lambda item: -item[1])),
        "top_ops": rows[:30],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--spike-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = report_stage_profile(args.build, args.spike_log)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "cycles", "attributed_cycles", "attribution_coverage", "by_owner")}))


if __name__ == "__main__":
    main()
