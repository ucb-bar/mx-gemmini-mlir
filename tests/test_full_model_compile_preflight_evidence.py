"""Keep reduced-model frontend probes distinct from executable MX results."""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re

import pytest

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_object import classify


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/full_model_compile_preflight_a042643_748b984"


def test_four_model_frontend_capture_is_not_a_full_model_mx_compile() -> None:
    index = json.loads((ARCHIVE / "index.json").read_text())
    profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                           "MxGemminiRocketConfig.json")
    assert index["schema"] == "mx_gemmini.full_model_compile_preflight.v1"
    assert index["status"] == "frontend_smoke_captured_full_model_compile_declined"
    assert index["model2mlir_revision"] == "a042643e31366724ca0482389c4ee85a9f88e343"
    assert index["compiler_revision"] == "748b9842463b883d88e60196b270d40aa5c20d96"
    assert index["profile_sha256"] == profile_sha256(profile)
    assert [row["model"] for row in index["rows"]] == [
        "tinyllama", "deepseek_r1_distill_qwen_1_5b", "gemma2_2b", "smolvla"]
    for row in index["rows"]:
        text = gzip.decompress((ARCHIVE / row["artifact"]).read_bytes())
        assert hashlib.sha256(text).hexdigest() == row["sha256"]
        assert len(text) == row["bytes"]
        graph = text.decode()
        ops = Counter(re.findall(
            r"(?m)^\s*(?:%[^=]+ = )?((?:linalg|tensor|arith|math|func|mx_gemmini)\.[A-Za-z_0-9]+)",
            graph))
        assert ops["linalg.matmul"] == row["linalg_matmul"]
        assert ops["linalg.generic"] == row["linalg_generic"]
        assert sum(count for op, count in ops.items() if op.startswith("mx_gemmini.")) == 0
        assert row["mx_ops"] == row["opaque"] == 0
        assert "func.call" not in graph
        with pytest.raises(Exception) as error:
            classify(graph, profile)
        assert row["compile_object_result"] == (
            f"{type(error.value).__name__}: {str(error.value).splitlines()[0]}")
