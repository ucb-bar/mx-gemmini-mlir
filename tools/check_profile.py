"""Check the standalone MX target profile against its selected Scala sources."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--chipyard", type=Path, required=True)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    if profile.get("schema") != "mx_gemmini.target_profile.v1":
        parser.error("unsupported target profile schema")
    if profile.get("config") != "MxGemminiRocketConfig":
        parser.error("unexpected standalone config")
    sources = profile.get("source_files")
    if not isinstance(sources, dict) or len(sources) != 2:
        parser.error("standalone profile needs both Scala source hashes")
    contents = {}
    for relative, expected in sources.items():
        path = args.chipyard / relative
        if not path.is_file():
            parser.error(f"missing source: {path}")
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != expected:
            parser.error(f"profile source hash differs: {relative}")
        contents[Path(relative).name] = source.decode()
    config = contents.get("GemminiConfigs.scala", "")
    implementation = contents.get("ConfigsFP.scala", "")
    if ("class MxGemminiRocketConfig extends Config(" not in config or
            "new gemmini.GemminiMxFPStandaloneConfig" not in config or
            "class GemminiMxFPStandaloneConfig extends Config(" not in implementation or
            "LazyModule(new Gemmini(GemminiMxFPConfigs.standaloneMxFPConfig))"
            not in implementation):
        parser.error("standalone MX config chain differs")
    mx = profile.get("mx", {})
    if (mx.get("mesh_rows"), mx.get("mesh_columns"), mx.get("scratchpad"),
            mx.get("requant_interface")) != (16, 16, "internal", "mmio"):
        parser.error("unsupported standalone MX geometry or interface")
    if set(mx.get("formats", [])) != {"mxfp8", "mxfp6", "mxfp4"}:
        parser.error("unsupported standalone MX formats")
    print(json.dumps({"profile": profile["name"], "config": profile["config"],
                      "source_files_verified": len(sources), "status": "source_bound"}))


if __name__ == "__main__":
    main()
