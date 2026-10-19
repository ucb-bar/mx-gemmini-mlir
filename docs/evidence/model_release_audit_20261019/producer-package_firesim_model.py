"""Package a completed session ELF as a bare-metal FireSim workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path

from tools.model_release import ROOT, registry, sha


def load_segments(elf):
    size = elf.stat().st_size
    with elf.open("rb") as stream:
        header = stream.read(64)
        if header[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<HH", header, 16) != (2, 243):
            raise ValueError("workload needs a linked little-endian RV64 executable")
        entry, table = struct.unpack_from("<QQ", header, 24)
        width, count = struct.unpack_from("<HH", header, 54)
        if width != 56 or not count or table + width * count > size:
            raise ValueError("invalid ELF program headers")
        segments = []
        for index in range(count):
            stream.seek(table + index * width)
            kind, flags, offset, virtual, physical, stored, memory, align = struct.unpack("<IIQQQQQQ", stream.read(width))
            if kind != 1 or not memory:
                continue
            if stored > memory or offset + stored > size or virtual != physical:
                raise ValueError("invalid or nonidentity bare-metal load segment")
            segments.append(dict(address=physical, file_bytes=stored, memory_bytes=memory, flags=flags, alignment=align))
        if not segments or not any(row["flags"] & 1 and row["address"] <= entry < row["address"] + row["memory_bytes"] for row in segments):
            raise ValueError("entry is outside executable load segments")
        return entry, segments


def package(elf, link_path, out, name):
    if not name.isascii() or not name.replace("-", "_").isidentifier():
        raise ValueError("workload name must be a plain identifier")
    if out.exists():
        raise ValueError("package directory must be fresh")
    link = json.loads(link_path.read_text())
    release = registry()
    if (link.get("schema") != "mx_gemmini.linked_merlin_session.v1"
            or link.get("profile_sha256") != release["profile"]["canonical_sha256"]
            or sha(elf) != link.get("elf_sha256") or elf.stat().st_size != link.get("elf_bytes")):
        raise ValueError("ELF or selected target differs from its session receipt")
    profile = json.loads((ROOT / release["profile"]["path"]).read_text())
    canonical = (json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if hashlib.sha256(canonical).hexdigest() != link["profile_sha256"]:
        raise ValueError("target profile changed")
    entry, segments = load_segments(elf)
    base = min(row["address"] for row in segments)
    end = max(row["address"] + row["memory_bytes"] for row in segments)
    # The runtime arena may be NOBITS or represented by a separate linker symbol.
    required = max(end - base, int(link["mem_bytes"]))
    dram = 1 << (required - 1).bit_length()
    directory = out / "workloads" / name
    directory.mkdir(parents=True)
    copied = directory / "session.elf"
    shutil.copyfile(elf, copied)
    if sha(copied) != link["elf_sha256"]:
        raise ValueError("ELF changed while packaging")
    workload = {"benchmark_name": name, "common_bootbinary": "session.elf", "common_rootfs": None,
                "common_outputs": [], "common_simulation_outputs": ["uartlog"]}
    (out / "workloads" / (name + ".json")).write_text(json.dumps(workload, indent=2) + "\n")
    requirements = {"schema": "mx_gemmini.firesim_model_package.v1", "status": "packaged_not_fpga_executed",
                    "workload": name + ".json", "elf_sha256": link["elf_sha256"], "elf_bytes": link["elf_bytes"],
                    "link_receipt_sha256": sha(link_path), "profile_sha256": link["profile_sha256"],
                    "entry": entry, "dram_base": base, "required_memory_span_bytes": required,
                    "recommended_dram_bytes": dram, "load_segments": segments,
                    "hardware": {"gemmini_revision": profile["source"]["gemmini_revision"],
                                 "configuration": profile["chipyard_config"], "geometry": profile["geometry"],
                                 "resources": profile["resources"], "chipyard_overrides": profile["chipyard_overrides"]},
                    "console_protocol": "SESSION_SHA plus SESSION_PERF and DONE; clocks on hardware are physical cycle CSR values",
                    "validation_required": ["matching FireSim hardware configuration and bitstream", "DRAM covers load segments and runtime arena",
                                            "host-target console and exit transport", "complete output/state comparison after execution"]}
    (out / "requirements.json").write_text(json.dumps(requirements, indent=2, sort_keys=True) + "\n")
    (out / "mx_session_link.json").write_bytes(link_path.read_bytes())
    (out / "target_profile.json").write_bytes((ROOT / release["profile"]["path"]).read_bytes())
    return requirements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--link-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default="mx-model")
    args = parser.parse_args()
    result = package(args.elf, args.link_receipt, args.out, args.name)
    print(json.dumps({key: result[key] for key in ["status", "elf_sha256", "recommended_dram_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
