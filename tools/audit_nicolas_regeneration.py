"""Classify Nicolas MX programs using validated generated-object Spike receipts.

The selected matrix result is narrower than rebuilding every instruction and
check in the source C program. Other receipt statuses stay separate so a
source hash or a source-only Spike run never becomes a regeneration claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs/evidence/nicolas_mx_source_inventory_266c593/index.json"
OUTPUT = ROOT / "docs/evidence/nicolas_mx_regeneration_audit_266c593/index.json"
NUMERIC_FIELDS = ("outputs_checked", "compared_bf16_outputs",
                  "compared_packed_lut_bytes", "compared_e8m0_scales",
                  "codes_checked", "packed_bytes_checked", "scales_checked")
MATRIX_FAMILIES = {"asymmetric_matrix", "other_tiled_matrix"}
HEX256 = re.compile(r"[0-9a-f]{64}\Z")
HEX160 = re.compile(r"[0-9a-f]{40}\Z")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_digest(value: object, label: str) -> None:
    if isinstance(value, str) and HEX256.fullmatch(value):
        return
    if isinstance(value, dict) and value and all(
            isinstance(key, str) and isinstance(digest, str) and
            HEX256.fullmatch(digest) for key, digest in value.items()):
        return
    raise ValueError(f"{label} is not a SHA-256 digest or digest map")


def _full_matrix_result(entry: dict) -> dict | None:
    if entry["family"] not in MATRIX_FAMILIES:
        return None
    candidates = [ref for ref in entry["evidence_references"]
                  if ref["status"] == "source_golden_matched_on_pinned_spike"]
    for ref in candidates:
        path = ROOT / ref["path"]
        if _sha(path) != ref["sha256"]:
            raise ValueError(f"changed Nicolas receipt: {ref['path']}")
        receipt = json.loads(path.read_text())
        if (receipt.get("schema") != ref["schema"] or
                receipt.get("status") != ref["status"] or
                receipt.get(ref["field"]) != entry["source_sha256"]):
            raise ValueError(f"stale source binding in {ref['path']}")
        for field in ("profile_sha256", "object_sha256", "elf_sha256",
                      "spike_log_sha256"):
            _require_digest(receipt.get(field), f"{ref['path']} {field}")
        for field in ("compiler_revision", "model2mlir_revision"):
            if not isinstance(receipt.get(field), str) or not HEX160.fullmatch(receipt[field]):
                raise ValueError(f"{ref['path']} {field} is not a pinned git revision")
        if receipt.get("mismatches") != 0 and receipt.get("spike_exit_code") != 0:
            raise ValueError(f"no zero-error Spike outcome in {ref['path']}")
        metrics = {field: receipt[field] for field in NUMERIC_FIELDS
                   if isinstance(receipt.get(field), int) and receipt[field] > 0}
        if not metrics:
            raise ValueError(f"no full-output comparison count in {ref['path']}")
        return {"receipt": ref["path"], "receipt_sha256": ref["sha256"],
                "receipt_schema": ref["schema"],
                "profile_sha256": receipt["profile_sha256"],
                "compiler_revision": receipt["compiler_revision"],
                "model2mlir_revision": receipt["model2mlir_revision"],
                "object_sha256": receipt["object_sha256"],
                "elf_sha256": receipt["elf_sha256"],
                "spike_log_sha256": receipt["spike_log_sha256"],
                "checked_output_metrics": metrics}
    return None


def build_report(inventory: dict) -> dict:
    if (inventory.get("schema") != "mx_gemmini.nicolas_mx_source_inventory.v1" or
            inventory.get("programs") != len(inventory.get("entries", [])) or
            inventory.get("rtl_revision") !=
            "266c593f2cb51d7e3fe83fc0317072b585ac3c52"):
        raise ValueError("Nicolas regeneration needs the pinned source inventory")
    rows = []
    for entry in inventory["entries"]:
        proof = _full_matrix_result(entry)
        if proof is not None:
            status = "generated_object_selected_spike_result_matched"
        elif entry["evidence_references"]:
            status = "separate_evidence_requires_scope_review"
        else:
            status = "no_direct_source_receipt"
        row = {"name": entry["name"], "family": entry["family"],
               "source_sha256": entry["source_sha256"], "status": status,
               "source_reference_count": len(entry["evidence_references"])}
        if proof is not None:
            row["selected_spike_result"] = proof
        else:
            row["other_receipt_statuses"] = sorted({
                ref["status"] for ref in entry["evidence_references"]
                if isinstance(ref["status"], str)})
        rows.append(row)
    counts = Counter(row["status"] for row in rows)
    return {
        "schema": "mx_gemmini.nicolas_mx_regeneration_audit.v1",
        "scope": ("a matched generated-object result is a selected Spike execution "
                  "path, not complete C control flow, RTL timing, or FPGA parity"),
        "rtl_revision": inventory["rtl_revision"],
        "source_inventory_sha256": _sha(INVENTORY),
        "programs": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "entries": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.inventory.resolve() != INVENTORY.resolve():
        parser.error("regeneration audit currently requires the archived pinned inventory")
    report = build_report(json.loads(args.inventory.read_text()))
    content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.out.is_file() or args.out.read_text() != content:
            raise SystemExit("Nicolas regeneration audit differs from current receipts")
    else:
        if args.out.exists():
            parser.error(f"refusing to overwrite {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(content)
    print(args.out)


if __name__ == "__main__":
    main()
